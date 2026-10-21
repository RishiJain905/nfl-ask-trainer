"""Scoring generated queries on the public val sets, here in the trainer (no NFL repo needed).

- **gretel** (SQLite): executed. Each example's database is rebuilt in an in-memory SQLite
  from the dataset's own `sql_context` (its CREATE TABLEs and INSERTs, kept by the public
  adapter in `public/gretel/raw/`); the gold and the predicted query both run, and the
  result sets are compared order-insensitively (as multisets of rows), numbers rounded to the
  gold's precision per column (an int and the same float are equal). A gold that fails to run
  is reported (`gold_failed`) and left out of `exec_acc`.
- **sqale** (SQLite) and the other SQL sets: a parse check, `EXPLAIN <query>` against the
  schema (sqale's DDL from `public/sqale/raw/schemas_ddl.parquet`), which also catches unknown
  tables and columns; then executed on randomly filled copies of that schema
  (`querycheck.check_sql`, the test-suite idea): `exec_ok` is True / False when some copy
  returned rows, None when none did; and an exact match after whitespace normalisation.
- **text2cypher, cypherbench** (Cypher): exact match after whitespace normalisation, plus a
  light syntax check (balanced brackets and quotes outside strings, a read clause first, a
  RETURN, no write clause, no $parameters).

Every reply must be one JSON object `{"query": ...}` (`json_ok`); a reply that's a refusal
counts as wrong (every public gold is a query). Queries run with a time limit (a progress
handler aborts runaway joins).
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from asktrainer.data.schema import Example
from asktrainer.querycheck import (  # noqa: F401  (re-exported for callers and tests)
    CYPHER_START,
    CYPHER_WRITE,
    MAX_ROWS,
    SQL_READ,
    STRINGS,
    TIME_LIMIT_S,
    _connect,
    _decimals,
    _norm_value,
    _run,
    cypher_ok,
    normalise,
    same_results,
)


def parse_reply(output: str) -> tuple[bool, str | None]:
    """(json_ok, the query or None)."""
    try:
        obj = json.loads(output.strip())
    except json.JSONDecodeError:
        return False, None
    if not isinstance(obj, dict):
        return False, None
    q = obj.get("query")
    return True, q if isinstance(q, str) and q.strip() else None


class PublicData:
    """The raw context each scorer needs, read lazily (only the ids asked for)."""

    def __init__(self, public_root: Path):
        self.root = Path(public_root)
        self._gretel: dict[str, dict[str, str]] | None = None
        self._sqale: dict[str, str] | None = None

    def gretel_context(self, row_id: str, split: str = "test") -> str | None:
        """The row's `sql_context` from the raw parquet of its source split (ids repeat
        across the dataset's train and test files)."""
        if self._gretel is None:
            self._gretel = {}
        if split not in self._gretel:
            import pyarrow.parquet as pq

            ctx: dict[str, str] = {}
            for f in sorted((self.root / "gretel" / "raw").glob(f"*_{split}*.parquet")):
                t = pq.read_table(f, columns=["id", "sql_context"])
                ids, texts = t.column("id").to_pylist(), t.column("sql_context").to_pylist()
                ctx.update((str(i), c) for i, c in zip(ids, texts, strict=True))
            self._gretel[split] = ctx
        return self._gretel[split].get(str(row_id))

    def sqale_schema(self, db_id: str) -> str | None:
        if self._sqale is None:
            import pyarrow.parquet as pq

            self._sqale = {}
            f = self.root / "sqale" / "raw" / "schemas_ddl.parquet"
            if f.exists():
                t = pq.read_table(f, columns=["schema_id", "full_schema"])
                ids, ddl = t.column("schema_id").to_pylist(), t.column("full_schema").to_pylist()
                self._sqale = dict(zip(ids, ddl, strict=True))
        return self._sqale.get(db_id)

    def schema_script(self, ex: Example) -> str | None:
        if ex.origin == "gretel":
            split = ex.meta.get("source_split") or "test"
            return self.gretel_context(ex.meta.get("row_id", ""), split)
        if ex.origin == "sqale":
            return self.sqale_schema(ex.meta.get("db_id", ""))
        return None


def parse_call(ex: Example, output: str) -> tuple[bool, str | None, str]:
    """(a parsable tool call, its query or None, why not) for a `call` example's generation:
    the thought up to `</think>`, then Spark's `<tool_call>` text."""
    from asktrainer.querycheck import calls_from_message

    text = output.split("</think>", 1)[1] if "</think>" in output else output
    calls = calls_from_message({"content": text})
    if not calls:
        return False, None, "no_call"
    fn = ex.messages[-1].tool_calls[0]["function"]
    name, arg = fn["name"], next(iter(fn["arguments"]))
    if len(calls) > 1:
        return True, None, "several_calls"
    if calls[0][0] != name:
        return True, None, "wrong_tool"
    q = calls[0][1].get(arg)
    return True, (q if isinstance(q, str) and q.strip() else None), "bad_argument"


def score(ex: Example, output: str, data: PublicData) -> dict[str, Any]:
    item: dict[str, Any] = {"id": ex.id, "origin": ex.origin, "dialect": ex.dialect,
                            "json_ok": False, "parse_ok": False, "exact": False,
                            "exec_ok": None, "gold_failed": False, "reason": ""}  # fmt: skip
    if ex.task == "call":  # `json_ok` here means: a parsable call to the right tool
        ok, q, why = parse_call(ex, output)
        item["json_ok"] = ok and q is not None
        if q is None:
            item["reason"] = why
            return item
        gold = ex.meta.get("gold") or ex.target  # a trace's target is the model's own query
    else:
        ok, q = parse_reply(output)
        item["json_ok"] = ok
        if not ok:
            item["reason"] = "not_json"
            return item
        if q is None:
            item["reason"] = "no_query"
            return item
        gold = json.loads(ex.target).get("query", "")
    item["exact"] = normalise(q) == normalise(gold)
    if ex.dialect == "cypher":
        item["parse_ok"], item["reason"] = cypher_ok(q)
        if item["parse_ok"] and ex.task == "call":
            _execute_cypher(ex, q, gold, item)
        return item
    if not SQL_READ.match(q):
        item["reason"] = "not_read_only"
        return item
    script = data.schema_script(ex)
    try:
        db = _connect(script)
    except sqlite3.Error:
        db = None  # the schema itself doesn't build in SQLite: no parse check possible
    if db is None:
        item["parse_ok"] = None
        item["reason"] = "schema_failed"
        return item
    try:
        try:
            db.execute("EXPLAIN " + q.strip().rstrip(";"))
            item["parse_ok"] = True
        except sqlite3.Error as e:
            item["reason"] = f"parse:{type(e).__name__}"
            item["exec_ok"] = False  # every SQL set is executed (real rows or filled copies)
            return item
        if ex.origin != "gretel":  # no rows: compare on randomly filled copies of the schema
            from asktrainer.querycheck import check_sql

            chk = check_sql(q, gold, script, real_rows=False)
            item["exec_ok"] = {"pass": True, "fail": False}.get(chk["verdict"])
            if item["exec_ok"] is None:
                item["reason"] = "ok"
            else:
                item["reason"] = "match" if item["exec_ok"] else chk["reason"]
            item["checked_by"] = chk.get("checked_by")
            return item
        try:
            g = _run(db, gold)
        except Exception:
            item["gold_failed"] = True
            item["reason"] = "gold_failed"
            return item
        try:
            p = _run(db, q)
        except Exception as e:
            item["exec_ok"] = False
            item["reason"] = f"pred:{type(e).__name__}"
            return item
        item["exec_ok"] = same_results(g, p)
        item["reason"] = "match" if item["exec_ok"] else "rows_differ"
        return item
    finally:
        db.close()


def _rate(items, key) -> float | None:
    vals = [bool(i[key]) for i in items if i.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def summarise(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def block(sub):
        return {"n": len(sub), "json_ok": _rate(sub, "json_ok"), "parse_ok": _rate(sub, "parse_ok"),
                "exact": _rate(sub, "exact"), "exec_acc": _rate(sub, "exec_ok"),
                "gold_failed": sum(1 for i in sub if i.get("gold_failed"))}  # fmt: skip

    by: dict[str, list] = defaultdict(list)
    for i in items:
        by[i["origin"]].append(i)
    return {"all": block(list(items)), "set": {k: block(v) for k, v in sorted(by.items())},
            "reasons": dict(Counter(i["reason"] for i in items).most_common(12))}  # fmt: skip


_GRAPH: dict[str, Any] = {"db": None, "loaded": None, "tried": False}


def _graph():
    """The local Neo4j that checks Cypher (`cyphercheck`), or None when it doesn't answer."""
    if not _GRAPH["tried"]:
        from asktrainer.cyphercheck import Neo4j

        _GRAPH["tried"] = True
        db = Neo4j()
        _GRAPH["db"] = db if db.ok() else None
    return _GRAPH["db"]


def _execute_cypher(ex: Example, q: str, gold: str, item: dict[str, Any]) -> None:
    """exec_ok for a Cypher call: on the question's CypherBench graph, or on graphs made from
    the prompt's schema (text2cypher); None when no Neo4j answers or nothing is proven."""
    from asktrainer import cyphercheck as cc

    db = _graph()
    if db is None:
        return
    try:
        if ex.origin == "cypherbench":
            graph = ex.meta.get("db_id") or ""
            if _GRAPH["loaded"] != graph:
                cc.load(db, *cc.cypherbench_graph(graph), key="eid", keep_key=True, index=True)
                _GRAPH["loaded"] = graph
            v = cc.check_on_graph(db, gold, [q])[0]
        else:
            _GRAPH["loaded"] = None  # the random graphs replace whatever was loaded
            v = cc.check_fuzz(db, ex.messages[0].content, gold, [q])[0]
    except Exception as e:  # the checker or the server: unproven, never a crash in training
        item["reason"] = f"check_error:{type(e).__name__}"
        return
    item["exec_ok"] = {"pass": True, "fail": False}.get(v["verdict"])
    item["checked_by"] = v.get("checked_by")
    item["reason"] = v.get("reason") or item["reason"]


def verify(examples: Sequence[Example], preds: Sequence[dict[str, str]], data: PublicData):
    by_id = {e.id: e for e in examples}
    todo = [p for p in preds if p["id"] in by_id]
    # one graph load per CypherBench graph: score grouped by graph, report in the given order
    order = sorted(
        range(len(todo)),
        key=lambda i: (by_id[todo[i]["id"]].origin, by_id[todo[i]["id"]].meta.get("db_id") or ""),
    )
    scored: dict[int, dict[str, Any]] = {}
    for i in order:
        scored[i] = score(by_id[todo[i]["id"]], todo[i]["output"], data)
    items = [scored[i] for i in range(len(todo))]
    return {"summary": summarise(items), "items": items}


def metrics(result: dict[str, Any]) -> dict[str, float]:
    """val/public/<metric> overall and val/<set>/<metric> per dataset."""
    out: dict[str, float] = {}
    s = result["summary"]
    for name, block in [("public", s["all"]), *s["set"].items()]:
        for m in ("json_ok", "parse_ok", "exact", "exec_acc"):
            if block.get(m) is not None:
                out[f"val/{name}/{m}"] = float(block[m])
        out[f"val/{name}/n"] = float(block["n"])
    return out
