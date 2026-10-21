"""The public text-to-query datasets, converted to `Example` records for the SFT mix.

One adapter per dataset (sqale, gretel, text2cypher, cypherbench). Each reads its `raw/` folder
(see `download.py`), keeps read-only single-statement queries, renders the database's schema
as text, and writes `public/<name>/examples_train.jsonl`, `examples_val.jsonl` and `stats.json`.

The goal of the public data is the behaviour (read the schema in front of you, write one
read-only query, reply with one JSON object) across thousands of schemas, so:

- **Schema text.** SQL: the CREATE TABLE / CREATE VIEW statements, one line each (INSERTs,
  indexes and comments dropped). Cypher: one compact text of node labels with properties and
  relationship types with their endpoints (`public_text.render_graph_schema`). An example whose
  schema text is over `SCHEMA_LIMIT` characters is dropped. SQaLe's databases are huge (median
  113 tables), so its schema is trimmed to the tables the gold query uses (and the tables of the
  prompt's examples) plus other tables of the question's own sub-schema while they fit.
- **Queries.** One read-only statement, no comments; whitespace collapsed; a (question, query)
  pair is kept once; a Cypher gold that returns a whole node or relationship (`RETURN m`) is
  dropped (the prompt says to return named properties; `--keep-whole-nodes` keeps them). SQL
  is also compiled against its schema in SQLite (EXPLAIN) so the `sqlite` label is true;
  Cypher is not executed (no Neo4j here), it is only checked.
- **Prompt examples.** For about half the examples (seeded), 1-2 other pairs of the same
  database appear in the prompt; the rest show "(none)". Never the example's own pair.
- **Validation.** The dataset's own test split (sampled to `VAL_MAX`); `--val-by-db` instead
  holds out 1% of whole databases from the train rows.
- **Deterministic.** Every random choice is seeded from (seed, dataset, row id) or a seeded
  shuffle of sorted rows: the same raw files and seed give the same files.

    python -m asktrainer.data.public --dataset all [--download] [--convert] [--seed 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from asktrainer.data import download as dl
from asktrainer.data.public_prompt import system_prompt, target, user_message
from asktrainer.data.public_text import (
    GraphSchema,
    Lang,
    ddl_statements,
    norm_key,
    norm_type,
    normalize_query,
    parse_graph_schema,
    question_aligned,
    render_graph_schema,
    returns_whole_entity,
    sqlite_problem,
    statement_creates,
    used_tables,
)
from asktrainer.data.schema import Example, JepaViews, Message, write_jsonl
from asktrainer.settings import get_paths

SCHEMA_LIMIT = 6000  # characters of schema text; longer is dropped (or trimmed, SQaLe)
VAL_MAX = 300  # validation examples per dataset
EXAMPLES_SHARE = 0.5  # share of prompts that carry other pairs of the same database
EXAMPLE_PAIR_MAX = 900  # characters of question + query for a pair to be shown in a prompt
QUESTION_MIN, QUESTION_MAX = 6, 1500
HOLDOUT_FRACTION = 0.01  # --val-by-db: share of databases held out from train

NAMES = ("sqale", "gretel", "text2cypher", "cypherbench")


# ---- shared pieces ----------------------------------------------------------------------------


@dataclass
class Cand:
    """One cleaned (question, query) pair of a database, before the prompt is built."""

    row_id: str
    db_id: str  # the database / graph (reporting, held-out splits)
    pool: str  # pairs that share this key can be each other's prompt examples
    question: str
    query: str
    schema: str = ""  # prompt schema text; "" when the adapter builds it per example
    ddl: list[str] | None = None  # CREATE statements to compile an SQL query against
    meta: dict[str, Any] = field(default_factory=dict)  # goes into Example.meta
    extra: dict[str, Any] = field(default_factory=dict)  # adapter-private, never written


def stable_rng(*parts: object) -> random.Random:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def clean_pair(
    question: object, query: object, lang: Lang, drops: Counter, *, keep_whole: bool = False
) -> tuple[str, str] | None:
    """The question (whitespace collapsed) and the normalised read-only query, or None with the
    reason counted in `drops`. A Cypher query that returns a whole node, relationship or path
    (`RETURN m`) is dropped unless `keep_whole`: the prompt tells the model to return named
    properties."""
    q = " ".join(question.split()) if isinstance(question, str) else ""
    if not (QUESTION_MIN <= len(q) <= QUESTION_MAX):
        drops["bad_question"] += 1
        return None
    sql, reason = normalize_query(query, lang)
    if sql is None:
        drops[reason or "bad_query"] += 1
        return None
    if lang == "cypher" and not keep_whole and returns_whole_entity(sql):
        drops["returns_whole_node"] += 1
        return None
    return q, sql


def dedupe(cands: list[Cand], drops: Counter) -> list[Cand]:
    """Keep a (question, query) pair once (the first by row id)."""
    seen: set[tuple[str, str]] = set()
    out = []
    for c in sorted(cands, key=lambda c: c.row_id):
        key = (norm_key(c.question), norm_key(c.query))
        if key in seen:
            drops["duplicate"] += 1
            continue
        seen.add(key)
        out.append(c)
    return out


def round_robin(cands: list[Cand], rng: random.Random) -> Iterator[Cand]:
    """One pair per database in turn (databases and pairs shuffled): a prefix of this order is
    spread over as many databases as possible, with the fewest pairs per database."""
    groups: dict[str, list[Cand]] = {}
    for c in sorted(cands, key=lambda c: c.row_id):
        groups.setdefault(c.db_id, []).append(c)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    depth = 0
    while keys:
        keys = [k for k in keys if len(groups[k]) > depth]
        for k in keys:
            yield groups[k][depth]
        depth += 1


def split_by_db(
    cands: list[Cand], fraction: float, rng: random.Random
) -> tuple[list[Cand], list[Cand]]:
    """Hold out whole databases: about `fraction` of the database ids (at least one) go to val."""
    ids = sorted({c.db_id for c in cands})
    rng.shuffle(ids)
    held = set(ids[: max(1, round(len(ids) * fraction))])
    return [c for c in cands if c.db_id not in held], [c for c in cands if c.db_id in held]


def pick_examples(rng: random.Random, pool: list[Cand], me: Cand) -> list[Cand]:
    """For about half the examples, 1-2 other pairs of the same pool; never `me`."""
    use = rng.random() < EXAMPLES_SHARE
    want = rng.choice((1, 2))
    if not use or len(pool) < 2:
        return []
    chosen: list[Cand] = []
    my_q, my_sql = norm_key(me.question), norm_key(me.query)
    for _ in range(40):
        p = pool[rng.randrange(len(pool))]
        if (
            p.row_id == me.row_id
            or norm_key(p.question) == my_q
            or norm_key(p.query) == my_sql
            or len(p.question) + len(p.query) > EXAMPLE_PAIR_MAX
            or any(p.row_id == x.row_id for x in chosen)
        ):
            continue
        chosen.append(p)
        if len(chosen) == want:
            break
    return chosen


def percentiles(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    a = np.asarray(values)
    return {
        "min": int(a.min()),
        "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
    }


# ---- the adapters -----------------------------------------------------------------------------


class Source:
    """One dataset: how to read its raw files into `Cand`s, and how a prompt's schema is made."""

    name = ""
    dialect = ""  # sqlite | cypher
    lang: Lang = "sql"
    license = ""
    target: int | None = None  # train examples to aim for (None: every usable row)

    def __init__(self, raw: Path, seed: int, keep_whole: bool = False):
        self.raw = raw
        self.seed = seed
        self.keep_whole = keep_whole  # keep Cypher golds that return whole nodes

    def candidates(self, split: str, drops: Counter) -> tuple[list[Cand], int]:
        """The cleaned pairs of `train` or `val` (the dataset's own test split) and the number of
        raw rows read. Reasons for dropped rows are counted in `drops`."""
        raise NotImplementedError

    def prompt_schema(
        self, c: Cand, examples: list[Cand]
    ) -> tuple[str, list[str] | None, list[Cand]] | str:
        """(schema text, CREATE statements to compile against or None, examples that fit) or a
        drop reason. The default is the schema stored on the pair."""
        return c.schema, c.ddl, examples

    def rng(self, *parts: object) -> random.Random:
        return stable_rng(self.seed, self.name, *parts)

    def clean(self, question: object, query: object, drops: Counter) -> tuple[str, str] | None:
        return clean_pair(question, query, self.lang, drops, keep_whole=self.keep_whole)


class Sqale(Source):
    """trl-lab/SQaLe-2-text-to-SQL-Queries (MIT): questions in eight phrasings over 9,259 SQLite
    databases; the DDL comes from the companion schemas dataset (`raw/schemas_ddl.parquet`).
    One phrasing is drawn per record (short, verbose, casual, plus the occasional evidence /
    typo style); the bulleted and fragment styles are skipped."""

    name = "sqale"
    dialect = "sqlite"
    lang = "sql"
    license = "MIT"
    target = 30_000
    PHRASINGS: tuple[tuple[str, int], ...] = (
        ("short_high_level", 3),
        ("verbose", 3),
        ("casual", 2),
        ("evidence_supported", 1),
        ("spelling_grammar_mistakes", 1),
    )
    COLUMNS = ["question_id", "schema_id", "sql", "difficulty", "questions", "relevant_tables"]
    _PLACEHOLDER = re.compile(r"^(?:[.\W_]*|<[^>]*>|simple|moderate|hard|n/?a|none|null)$", re.I)

    def __init__(self, raw: Path, seed: int, keep_whole: bool = False):
        super().__init__(raw, seed, keep_whole)
        self._ddl_raw: dict[tuple[str, str], str] | None = None
        self._parsed: dict[tuple[str, str], tuple[list[Any], dict[str, Any]]] = {}

    def usable(self, text: object) -> str | None:
        if not isinstance(text, str):
            return None
        t = " ".join(text.split())
        if len(t) < 8 or not re.search(r"[A-Za-z]", t) or self._PLACEHOLDER.match(t):
            return None
        return t

    def question(
        self, rng: random.Random, questions: dict[str, Any], sql: str
    ) -> tuple[str, str] | None:
        """(phrasing, text): phrasings are tried in a weighted random order until one is a real
        question that matches the query (see `question_aligned`)."""
        keys = [k for k, _ in self.PHRASINGS]
        weights = [w for _, w in self.PHRASINGS]
        while keys:
            k = rng.choices(keys, weights)[0]
            i = keys.index(k)
            keys.pop(i)
            weights.pop(i)
            text = questions.get(k)
            if k == "evidence_supported" and isinstance(text, str):
                text = re.split(r",\s*evidence:", text, maxsplit=1)[0]
            text = self.usable(text)
            if text and question_aligned(text, sql):
                return k, text
        return None

    def candidates(self, split: str, drops: Counter) -> tuple[list[Cand], int]:
        import pyarrow.parquet as pq

        stem = "test" if split == "val" else "train"
        files = sorted((self.raw / "data").glob(f"{stem}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {stem} parquet in {self.raw / 'data'}: run --download")
        have = self._ddl_ids(stem)
        out: list[Cand] = []
        n_raw = 0
        for f in files:
            for r in pq.read_table(f, columns=self.COLUMNS).to_pylist():
                n_raw += 1
                if r["schema_id"] not in have:
                    drops["no_schema"] += 1
                    continue
                picked = self.question(
                    self.rng("phrasing", r["question_id"]), r["questions"] or {}, r["sql"] or ""
                )
                if picked is None:
                    drops["bad_question"] += 1
                    continue
                pair = self.clean(picked[1], r["sql"], drops)
                if pair is None:
                    continue
                try:
                    relevant = [str(t).lower() for t in json.loads(r["relevant_tables"] or "[]")]
                except (ValueError, TypeError):
                    relevant = []
                out.append(
                    Cand(
                        row_id=r["question_id"],
                        db_id=r["schema_id"],
                        pool=r["schema_id"],
                        question=pair[0],
                        query=pair[1],
                        meta={"phrasing": picked[0], "difficulty": r["difficulty"]},
                        extra={"split": stem, "relevant": relevant},
                    )
                )
        return out, n_raw

    # -- schemas --

    def _ddl_ids(self, stem: str) -> set[str]:
        return {sid for (sp, sid) in self._load_ddl() if sp == stem}

    def _load_ddl(self) -> dict[tuple[str, str], str]:
        if self._ddl_raw is None:
            import pyarrow.parquet as pq

            t = pq.read_table(
                self.raw / dl.SQALE_DDL_FILE, columns=["split", "schema_id", "full_schema"]
            )
            self._ddl_raw = {
                (s, i): d
                for s, i, d in zip(
                    t.column("split").to_pylist(),
                    t.column("schema_id").to_pylist(),
                    t.column("full_schema").to_pylist(),
                    strict=True,
                )
            }
        return self._ddl_raw

    def _tables(self, stem: str, schema_id: str) -> tuple[list[Any], dict[str, Any]]:
        key = (stem, schema_id)
        if key not in self._parsed:
            stmts = ddl_statements(self._load_ddl().get(key))
            self._parsed[key] = (stmts, {s.name: s for s in stmts})
        return self._parsed[key]

    def prompt_schema(
        self, c: Cand, examples: list[Cand]
    ) -> tuple[str, list[str] | None, list[Cand]] | str:
        stmts, by_name = self._tables(c.extra["split"], c.db_id)
        if not stmts:
            return "no_schema"
        used = used_tables(c.query, by_name)
        if not used:
            return "no_known_table"
        if not all(statement_creates(by_name[n].text) for n in used):
            return "ddl_not_sqlite"
        # an example joins the prompt only when its own tables exist and can be created
        usable: list[tuple[Cand, set[str]]] = []
        for e in examples:
            u = used_tables(e.query, by_name)
            if u and all(statement_creates(by_name[n].text) for n in u):
                usable.append((e, u))
        order = [s.name for s in stmts]

        def render(keep: set[str]) -> str:
            return "\n".join(by_name[n].text for n in order if n in keep)

        keep = set(used)
        for k in range(len(usable), -1, -1):  # as many examples as the budget allows
            keep = set(used).union(*(u for _, u in usable[:k]))
            if len(render(keep)) <= SCHEMA_LIMIT:
                usable = usable[:k]
                break
        else:
            return "schema_too_long"
        # then the other tables of the question's own sub-schema, while they fit
        size = len(render(keep))
        for n in order:
            if n in c.extra["relevant"] and n not in keep and statement_creates(by_name[n].text):
                add = len(by_name[n].text) + 1
                if size + add <= SCHEMA_LIMIT:
                    keep.add(n)
                    size += add
        return render(keep), [by_name[n].text for n in order if n in keep], [e for e, _ in usable]


class Gretel(Source):
    """gretelai/synthetic_text_to_sql (Apache-2.0): every row carries its own tiny schema
    (`sql_context`, with INSERTs); the SELECT tasks are the rows that pass the read-only filter."""

    name = "gretel"
    dialect = "sqlite"
    lang = "sql"
    license = "Apache-2.0"
    target = 30_000
    FILES = {
        "train": "synthetic_text_to_sql_train.snappy.parquet",
        "val": "synthetic_text_to_sql_test.snappy.parquet",
    }
    COLUMNS = [
        "id",
        "domain",
        "sql_complexity",
        "sql_task_type",
        "sql_prompt",
        "sql_context",
        "sql",
    ]

    def candidates(self, split: str, drops: Counter) -> tuple[list[Cand], int]:
        import pyarrow.parquet as pq

        rows = pq.read_table(self.raw / self.FILES[split], columns=self.COLUMNS).to_pylist()
        out = []
        for r in rows:
            stmts = ddl_statements(r["sql_context"])
            if not stmts:
                drops["schema_empty"] += 1
                continue
            pair = self.clean(r["sql_prompt"], r["sql"], drops)
            if pair is None:
                continue
            schema = "\n".join(s.text for s in stmts)
            key = short_hash(schema)
            out.append(
                Cand(
                    row_id=str(r["id"]),
                    db_id=f"ctx-{key}",
                    pool=key,
                    question=pair[0],
                    query=pair[1],
                    schema=schema,
                    ddl=[s.text for s in stmts],
                    meta={
                        "domain": r["domain"],
                        "sql_complexity": r["sql_complexity"],
                        "sql_task_type": r["sql_task_type"],
                    },
                )
            )
        return out, len(rows)


class Text2Cypher(Source):
    """neo4j/text2cypher-2025v1 (Apache-2.0): schemas come as text in a handful of formats; the
    two big ones are parsed and re-rendered compactly, the rest are dropped."""

    name = "text2cypher"
    dialect = "cypher"
    lang = "cypher"
    license = "Apache-2.0"
    COLUMNS = [
        "question",
        "schema",
        "cypher",
        "data_source",
        "instance_id",
        "database_reference_alias",
    ]

    def __init__(self, raw: Path, seed: int, keep_whole: bool = False):
        super().__init__(raw, seed, keep_whole)
        self._schemas: dict[str, str | None] = {}

    def render(self, raw_schema: object) -> str | None:
        key = raw_schema if isinstance(raw_schema, str) else ""
        if key not in self._schemas:
            g = parse_graph_schema(raw_schema)
            self._schemas[key] = render_graph_schema(g) if g else None
        return self._schemas[key]

    def candidates(self, split: str, drops: Counter) -> tuple[list[Cand], int]:
        import pyarrow.parquet as pq

        f = self.raw / "data" / ("test.parquet" if split == "val" else "train.parquet")
        rows = pq.read_table(f, columns=self.COLUMNS).to_pylist()
        out = []
        for i, r in enumerate(rows):
            schema = self.render(r["schema"])
            if schema is None:
                drops["schema_unparseable"] += 1
                continue
            pair = self.clean(r["question"], r["cypher"], drops)
            if pair is None:
                continue
            key = short_hash(schema)
            out.append(
                Cand(
                    row_id=str(r["instance_id"] or i),
                    db_id=r["database_reference_alias"] or f"schema-{key}",
                    pool=key,
                    question=pair[0],
                    query=pair[1],
                    schema=schema,
                    meta={
                        "data_source": r["data_source"],
                        "database_reference_alias": r["database_reference_alias"],
                    },
                )
            )
        return out, len(rows)


class CypherBench(Source):
    """megagonlabs/cypherbench (Apache-2.0): ~10k questions over 11 Wikidata graphs (train and
    test use different graphs). The schemas are the small JSON files of `graphs/schemas`; every
    node also has an implicit `name` (the gold queries use `n.name` everywhere), listed first."""

    name = "cypherbench"
    dialect = "cypher"
    lang = "cypher"
    license = "Apache-2.0"

    def schemas(self) -> dict[str, str]:
        out = {}
        for f in sorted((self.raw / "graphs" / "schemas").glob("*_schema.json")):
            js = json.loads(f.read_text(encoding="utf-8"))
            g = GraphSchema()
            for e in js["entities"]:
                g.nodes[e["label"]] = {
                    "name": "",
                    **{k: norm_type(v) for k, v in e["properties"].items()},
                }
            for r in js["relations"]:
                g.rels.append((r["subj_label"], r["label"], r["obj_label"]))
                if r["properties"]:
                    g.rel_props[r["label"]] = {k: norm_type(v) for k, v in r["properties"].items()}
            rendered = render_graph_schema(g)
            if rendered:
                out[js["name"]] = rendered
        return out

    def candidates(self, split: str, drops: Counter) -> tuple[list[Cand], int]:
        rows = json.loads(
            (self.raw / ("test.json" if split == "val" else "train.json")).read_text("utf-8")
        )
        schemas = self.schemas()
        out = []
        for r in rows:
            schema = schemas.get(r["graph"])
            if schema is None:
                drops["schema_unparseable"] += 1
                continue
            pair = self.clean(r["nl_question"], r["gold_cypher"], drops)
            if pair is None:
                continue
            tpl = r.get("from_template") or {}
            out.append(
                Cand(
                    row_id=str(r["qid"]),
                    db_id=r["graph"],
                    pool=r["graph"],
                    question=pair[0],
                    query=pair[1],
                    schema=schema,
                    meta={
                        "graph": r["graph"],
                        "match_category": tpl.get("match_category"),
                        "return_pattern_id": tpl.get("return_pattern_id"),
                    },
                )
            )
        return out, len(rows)


SOURCES: dict[str, type[Source]] = {
    "sqale": Sqale,
    "gretel": Gretel,
    "text2cypher": Text2Cypher,
    "cypherbench": CypherBench,
}


# ---- building the examples --------------------------------------------------------------------


def make_example(
    src: Source, c: Cand, schema: str, examples: list[Cand], split: str, used_ids: set[str]
) -> Example:
    eid = f"{src.name}-{c.row_id}"
    if eid in used_ids:  # the same row id in both splits (gretel): the val one gets a suffix
        eid = f"{eid}-{split}"
    n = 1
    while eid in used_ids:  # never emit two records with one id
        n += 1
        eid = f"{src.name}-{c.row_id}-{split}{n}"
    used_ids.add(eid)
    return Example(
        id=eid[:200],
        origin=src.name,  # type: ignore[arg-type]
        task="generate",
        split=split,  # type: ignore[arg-type]
        group="public",
        messages=[
            Message(
                role="system",
                content=system_prompt(schema, [(e.question, e.query) for e in examples]),
            ),
            Message(role="user", content=user_message(c.question, src.dialect)),
            Message(role="assistant", content=target(c.query)),
        ],
        jepa=JepaViews(text=c.question, code=c.query),
        dialect=src.dialect,  # type: ignore[arg-type]
        meta={
            "row_id": c.row_id,
            "db_id": c.db_id,
            "license": src.license,
            "source_split": "test" if split == "val" else "train",
            "n_prompt_examples": len(examples),
            "schema_chars": len(schema),
            **c.meta,
        },
    )


def build_split(
    src: Source,
    cands: list[Cand],
    split: str,
    limit: int | None,
    drops: Counter,
    used_ids: set[str],
) -> tuple[list[Example], dict[str, Any]]:
    """Walk `cands` in a spread-over-databases order until `limit` examples are built. The
    drops of this stage (schema too long, compile errors) count the pairs reached. `used_ids`
    is shared by the splits of one dataset so ids stay unique across train and val."""
    pools: dict[str, list[Cand]] = {}
    for c in sorted(cands, key=lambda c: c.row_id):
        pools.setdefault(c.pool, []).append(c)
    out: list[Example] = []
    reached = 0
    for c in round_robin(cands, stable_rng(src.seed, src.name, split, "order")):
        if limit is not None and len(out) >= limit:
            break
        reached += 1
        rng = src.rng("prompt", split, c.row_id)
        res = src.prompt_schema(c, pick_examples(rng, pools[c.pool], c))
        if isinstance(res, str):
            drops[res] += 1
            continue
        schema, ddl, examples = res
        if len(schema) > SCHEMA_LIMIT:
            drops["schema_too_long"] += 1
            continue
        if src.dialect == "sqlite":
            problem = sqlite_problem(ddl or [], c.query)
            if problem:
                drops[problem] += 1
                continue
            examples = [e for e in examples if sqlite_problem(ddl or [], e.query) is None]
        out.append(make_example(src, c, schema, examples, split, used_ids))
    return out, {"reached": reached, "not_reached": len(cands) - reached}


def example_stats(examples: list[Example]) -> dict[str, Any]:
    """Length percentiles (characters) and the share of prompts that carry other pairs."""
    schema, question, query, prompt = [], [], [], []
    with_examples = 0
    for ex in examples:
        schema.append(ex.meta["schema_chars"])
        question.append(len(ex.jepa.text))
        query.append(len(ex.jepa.code))
        prompt.append(sum(len(m.content) for m in ex.messages[:-1]))
        with_examples += ex.meta["n_prompt_examples"] > 0
    return {
        "examples": len(examples),
        "databases": len({ex.meta["db_id"] for ex in examples}),
        "examples_in_prompt_share": round(with_examples / len(examples), 4) if examples else 0.0,
        "schema_chars": percentiles(schema),
        "question_chars": percentiles(question),
        "query_chars": percentiles(query),
        "prompt_chars": percentiles(prompt),  # system + user
    }


def convert(
    name: str,
    *,
    seed: int = 0,
    val_by_db: bool = False,
    keep_whole: bool = False,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Convert one dataset's raw files; writes the two jsonl files and stats.json."""
    raw = dl.raw_dir(name) if out_dir is None else out_dir / "raw"
    src = SOURCES[name](raw, seed, keep_whole)
    base = (get_paths().public / name) if out_dir is None else out_dir
    drops = {"train": Counter(), "val": Counter()}
    train, n_train = src.candidates("train", drops["train"])
    val, n_val = src.candidates("val", drops["val"])
    train = dedupe(train, drops["train"])
    val = dedupe(val, drops["val"])
    if val_by_db:
        train, val = split_by_db(train + val, HOLDOUT_FRACTION, stable_rng(seed, name, "holdout"))
        n_val = len(val)
    train_keys = {(norm_key(c.question), norm_key(c.query)) for c in train}
    before = len(val)
    val = [c for c in val if (norm_key(c.question), norm_key(c.query)) not in train_keys]
    drops["val"]["duplicate_of_train"] += before - len(val)

    used_ids: set[str] = set()
    train_ex, train_info = build_split(src, train, "train", src.target, drops["train"], used_ids)
    val_ex, val_info = build_split(src, val, "val", VAL_MAX, drops["val"], used_ids)
    for lst in (train_ex, val_ex):  # a seeded shuffle: the files are not grouped by database
        stable_rng(seed, name, "shuffle", len(lst)).shuffle(lst)

    write_jsonl(base / "examples_train.jsonl", train_ex)
    write_jsonl(base / "examples_val.jsonl", val_ex)
    train_dbs = {c.db_id for c in train}
    manifest = base / "raw" / "manifest.json"
    stats = {
        "dataset": name,
        "seed": seed,
        "license": src.license,
        "revision": json.loads(manifest.read_text("utf-8"))["revision"]
        if manifest.exists()
        else None,
        "val_source": "held-out databases (--val-by-db)"
        if val_by_db
        else "the dataset's own test split",
        "schema_limit_chars": SCHEMA_LIMIT,
        "train": {
            "raw_rows": n_train,
            "usable_pairs": len(train),
            **train_info,
            "dropped": dict(sorted(drops["train"].items())),
            **example_stats(train_ex),
        },
        "val": {
            "raw_rows": n_val,
            "usable_pairs": len(val),
            **val_info,
            "dropped": dict(sorted(drops["val"].items())),
            **example_stats(val_ex),
            "databases_also_in_train": len({e.meta["db_id"] for e in val_ex} & train_dbs),
        },
    }
    (base / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="all", choices=[*NAMES, "all"])
    ap.add_argument("--download", action="store_true", help="fetch the raw files first")
    ap.add_argument("--convert", action="store_true", help="convert the raw files to examples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-by-db", action="store_true", help="hold out whole databases as val")
    ap.add_argument(
        "--keep-whole-nodes",
        action="store_true",
        help="keep Cypher golds that return whole nodes / relationships (dropped by default)",
    )
    args = ap.parse_args(argv)
    do_download = args.download or not args.convert
    do_convert = args.convert or not args.download
    for name in NAMES if args.dataset == "all" else [args.dataset]:
        if do_download:
            dl.download(name)
        if do_convert:
            s = convert(
                name, seed=args.seed, val_by_db=args.val_by_db, keep_whole=args.keep_whole_nodes
            )
            print(
                f"[{name}] train {s['train']['examples']} / val {s['val']['examples']} "
                f"(drops: {sum(s['train']['dropped'].values())} train)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
