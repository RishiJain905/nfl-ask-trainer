"""Checking Cypher answers on a real graph: CypherBench's own (sampled) graphs, and for text2cypher,
whose databases aren't public, random graphs made from the prompt's schema (the Cypher twin of
`querycheck`'s fuzzed SQLite databases).

    python -m asktrainer.cyphercheck recheck --run D:\\nfl-ml-data\\ask\\traces\\pilot
    python -m asktrainer.cyphercheck probe-cypherbench --graph art [--n 200]

The graph is a local Neo4j (`asktrainer-neo4j`: auth off, HTTP on 127.0.0.1:7476, nothing else
in it). Only the standard library is used, so this runs on Windows next to it and `tracegen`
(WSL can't reach Windows' 127.0.0.1). Answers run read-only (the `access-mode: READ` header and a
write-clause guard) with the server's transaction timeout.

`recheck` reads a trace run's Cypher answers that aren't passed yet and appends a verdict per
answer to `<run>/rechecks.jsonl` (the export and `tracegen`'s next round read it):
- cypherbench: gold and answer both run on the question's sampled graph (`checked_by` graph);
- text2cypher: gold and answer run on `FUZZ_SEEDS` random graphs from the schema, the query's
  literals planted in the properties they're compared with (`checked_by` fuzz_graph); a sorted
  gold is also checked for order on a graph with (almost) no ties.
A run where the gold returns nothing, or only zeros and nulls, proves nothing: unverified.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from asktrainer.querycheck import (
    NUM_LITERAL,
    STRINGS,
    _norm_value,
    align,
    check_sql,
    contained_in,
    cypher_ok,
    normalise,
    same_results,
)

NEO4J_URL = "http://127.0.0.1:7476"
CYPHERBENCH = Path(r"D:\nfl-ml-data\ask\benchmarks\cypherbench")
FUZZ_SEEDS = (11, 23, 37)
ORDER_SEED = 53
NODES_PER_LABEL = 14
RELS_PER_PATTERN = 24
MAX_ROWS = 10_000
# the checking rules' version: a re-check redoes the answers an older version didn't pass
# (2: the same columns in another order count; names, titles and ids unique per node)
CHECKER_VERSION = 2


class CypherError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message[:200]}")
        self.code = code


class Neo4j:
    def __init__(self, url: str = NEO4J_URL, db: str = "neo4j", timeout: float = 60):
        self.url, self.db, self.timeout = url.rstrip("/"), db, timeout

    def _post(self, statements: list[dict[str, Any]], read: bool) -> list[dict[str, Any]]:
        body = json.dumps({"statements": statements}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if read:
            headers["access-mode"] = "READ"
        req = urllib.request.Request(
            f"{self.url}/db/{self.db}/tx/commit", data=body, headers=headers
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        if out.get("errors"):
            e = out["errors"][0]
            raise CypherError(e.get("code", ""), e.get("message", ""))
        return out["results"]

    def run(self, q: str, params: dict[str, Any] | None = None, read: bool = True):
        """(column names, rows as tuples of hashable values)."""
        st = {"statement": q, "parameters": params or {}, "resultDataContents": ["row"]}
        res = self._post([st], read)[0]
        rows = [tuple(_hashable(v) for v in d["row"]) for d in res["data"]]
        if len(rows) > MAX_ROWS:
            raise RuntimeError("too many rows")
        return res["columns"], rows

    def write(self, statements: list[tuple[str, dict[str, Any]]]) -> None:
        self._post([{"statement": s, "parameters": p} for s, p in statements], read=False)

    def wipe(self) -> None:
        while True:
            _, rows = self.run(
                "MATCH (n) WITH n LIMIT 20000 DETACH DELETE n RETURN count(*)", read=False
            )
            if not rows or not rows[0][0]:
                return

    def ok(self) -> bool:
        try:
            return self.run("RETURN 1")[1] == [(1,)]
        except Exception:
            return False


def _hashable(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


def _q(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def load(
    db: Neo4j,
    nodes: dict[str, list[dict[str, Any]]],
    rels: list[tuple[str, str, str, str, str, dict[str, Any]]],
    types: dict[str, dict[str, str]],
    batch: int = 2000,
    key: str = "_k",
    keep_key: bool = False,
    index: bool = False,
) -> None:
    """Wipe the graph and load nodes ({label: [props with `key`]}) and relationships
    ([(type, start label, start key, end label, end key, props)]). `types`: label or
    relationship type -> {property: date | datetime}, converted after loading."""
    db.wipe()
    if index:
        for label in nodes:
            db.write([(f"CREATE INDEX IF NOT EXISTS FOR (n:{_q(label)}) ON (n.{_q(key)})", {})])
        db.write([("CALL db.awaitIndexes(300)", {})])
    stmts: list[tuple[str, dict[str, Any]]] = []
    for label, rows in nodes.items():
        conv = "".join(
            f" SET n.{_q(p)} = {fn}(n.{_q(p)})" for p, fn in (types.get(label) or {}).items()
        )
        for i in range(0, len(rows), batch):
            stmts.append(
                (
                    f"UNWIND $rows AS r CREATE (n:{_q(label)}) SET n = r{conv}",
                    {"rows": rows[i : i + batch]},
                )
            )
    by_pattern: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for t, a_label, a, b_label, b, props in rels:
        by_pattern.setdefault((t, a_label, b_label), []).append({"a": a, "b": b, "p": props})
    for (t, a_label, b_label), rows in by_pattern.items():
        conv = "".join(
            f" SET x.{_q(p)} = {fn}(x.{_q(p)})" for p, fn in (types.get(t) or {}).items()
        )
        for i in range(0, len(rows), batch):
            stmts.append(
                (
                    f"UNWIND $rows AS r MATCH (a:{_q(a_label)} {{{_q(key)}: r.a}}) "
                    f"MATCH (b:{_q(b_label)} {{{_q(key)}: r.b}}) "
                    f"CREATE (a)-[x:{_q(t)}]->(b) SET x = r.p{conv}",
                    {"rows": rows[i : i + batch]},
                )
            )
    if not keep_key:
        stmts.append((f"MATCH (n) REMOVE n.{_q(key)}", {}))
    for i in range(0, len(stmts), 20):
        db.write(stmts[i : i + 20])


# ---- the schema in the prompt and random graphs from it -------------------------------------

NODE_LINE = re.compile(r"^-\s*([A-Za-z_][\w]*)\s*\{(.*)\}\s*$")
REL_LINE = re.compile(
    r"^-\s*\(([A-Za-z_]\w*)\)-\[:([A-Za-z_]\w*)(?:\s*\{(.*)\})?\]->\(([A-Za-z_]\w*)\)\s*$"
)


def _props(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (text or "").split(","):
        name, _, kind = part.strip().partition(":")
        name = name.strip().strip("`")
        if name:
            out[name] = (kind.strip() or "STRING").upper()
    return out


def parse_schema(system: str) -> dict[str, Any]:
    """{"nodes": {label: {prop: TYPE}}, "rels": [(start, type, end, {prop: TYPE})]} from the
    tool prompt's "Nodes (...)" / "Relationships:" lists."""
    nodes: dict[str, dict[str, str]] = {}
    rels: list[tuple[str, str, str, dict[str, str]]] = []
    section = None
    for raw in system.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("Nodes"):
            section = "nodes"
            continue
        if line.startswith("Relationships"):
            section = "rels"
            continue
        if not line.startswith("-"):
            section = None  # "Examples ...", "Rules:": the lists are over
            continue
        if section == "nodes" and (m := NODE_LINE.match(line)):
            nodes[m.group(1)] = _props(m.group(2))
        elif section == "rels" and (m := REL_LINE.match(line)):
            rels.append((m.group(1), m.group(2), m.group(4), _props(m.group(3) or "")))
    for a, _, b, _ in rels:  # a relationship may name a label the node list left out
        nodes.setdefault(a, {})
        nodes.setdefault(b, {})
    return {"nodes": nodes, "rels": rels}


COMPARE = r"(?:=|<>|<=|>=|<|>|CONTAINS|STARTS\s+WITH|ENDS\s+WITH|IN)"
STR = r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\""
NUM = r"(-?\d+(?:\.\d+)?)"
MAP_LIT = re.compile(rf"([A-Za-z_]\w*)\s*:\s*(?:{STR}|{NUM})")
CMP_LIT = re.compile(
    rf"\.([A-Za-z_]\w*)\)?\s*{COMPARE}\s*(?:\w+\()?\s*(?:{STR}|{NUM})", re.IGNORECASE
)
CMP_LIT_REV = re.compile(
    rf"(?:{STR}|{NUM})\s*{COMPARE}\s*(?:\w+\()?\s*\w+\.([A-Za-z_]\w*)", re.IGNORECASE
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
DATE_NAME = re.compile(r"(date|Date|_at$|At$|[Tt]ime|born|died|[Rr]eleased)")
# names, titles and ids: unique per node, as in real data (a pool would make, e.g., grouping by
# title differ from grouping by the node where real titles don't repeat)
ID_NAME = re.compile(r"(^name$|^title$|Name$|_name$|Title$|^id$|Id$|ID$|_id$|^url$|Url$|^code$)")


def literals(queries: list[str]) -> dict[str, Any]:
    """The literals the queries compare properties with: {"by_prop": {prop: [values]},
    "strs": [...], "nums": [...]} (values from map patterns and comparisons)."""
    by_prop: dict[str, list[Any]] = {}
    strs: set[str] = set()
    nums: set[float] = set()

    def add(prop: str | None, s1: str | None, s2: str | None, n: str | None) -> None:
        if s1 is not None or s2 is not None:
            v: Any = (s1 if s1 is not None else s2).replace("\\'", "'").replace('\\"', '"')
            strs.add(v)
        elif n is not None:
            v = float(n)
            v = int(v) if v == int(v) and "." not in n else v
            nums.add(float(n))
        else:
            return
        if prop:
            by_prop.setdefault(prop, [])
            if v not in by_prop[prop]:
                by_prop[prop].append(v)

    for q in queries:
        q = q or ""
        for m in MAP_LIT.finditer(q):
            add(m.group(1), m.group(2), m.group(3), m.group(4))
        for m in CMP_LIT.finditer(q):
            add(m.group(1), m.group(2), m.group(3), m.group(4))
        for m in CMP_LIT_REV.finditer(q):
            add(m.group(4), m.group(1), m.group(2), m.group(3))
        for s1, s2 in re.findall(STR, q):
            v = s1 or s2
            if 0 < len(v) <= 80:
                strs.add(v)
        for n in NUM_LITERAL.findall(STRINGS.sub("''", q)):
            try:
                nums.add(float(n))
            except ValueError:
                pass
    return {"by_prop": by_prop, "strs": sorted(strs), "nums": sorted(nums)}


WORDS = ("alpha", "beta", "gamma", "delta", "omega")


def fuzz_graph(
    schema: dict[str, Any], seed: int, lits: dict[str, Any], distinct: bool = False
) -> tuple[dict[str, list[dict[str, Any]]], list[tuple], dict[str, dict[str, str]]]:
    """Random nodes and relationships for the schema (inputs for `load`). Properties a query
    compares with a literal get that literal often; the other text and numbers come from small
    pools so joins and groups repeat. `distinct`: wide ranges and unique text (no ties)."""
    rnd = random.Random(seed)
    by_prop, strs, nums = lits["by_prop"], lits["strs"], lits["nums"]
    text_pool = list(strs) + list(WORDS)
    int_pool = [int(n) for n in nums if n == int(n) and abs(n) < 10**9]
    real_pool = [n for n in nums if abs(n) < 10**9]

    def date(hm: bool) -> str:
        if distinct:
            d = f"{rnd.randint(1950, 2030)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"
        else:
            d = f"{rnd.randint(2015, 2024)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"
        if hm:
            d += f"T{rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d}:{rnd.randint(0, 59):02d}"
        return d

    def number(kind: str, prop: str) -> Any:
        planted = [v for v in by_prop.get(prop, []) if isinstance(v, (int, float))]
        if planted and rnd.random() < 0.5:
            base = rnd.choice(planted)
            v = base + rnd.choice((-1, 0, 0, 1)) * (1 if kind == "INTEGER" else rnd.random())
        elif kind == "INTEGER":
            pool = int_pool
            v = (
                rnd.choice(pool)
                if pool and rnd.random() < 0.25
                else (rnd.randint(1, 10**6) if distinct else rnd.randint(1, 8))
            )
        else:
            pool = real_pool
            v = (
                rnd.choice(pool)
                if pool and rnd.random() < 0.25
                else round(rnd.uniform(0, 10**6 if distinct else 100), 2)
            )
        return int(round(v)) if kind == "INTEGER" else float(v)

    def value(kind: str, prop: str, i: int | None = None) -> Any:
        if kind.startswith("INT") or kind == "FLOAT":
            return number("INTEGER" if kind.startswith("INT") else "FLOAT", prop)
        if kind.startswith("BOOL"):
            planted = [v for v in by_prop.get(prop, []) if isinstance(v, bool)]
            return rnd.random() < 0.5 if not planted else rnd.choice([True, False])
        if kind in ("DATE", "DATE_TIME", "LOCAL_DATE_TIME"):
            planted = [v for v in by_prop.get(prop, []) if isinstance(v, str) and DATE_RE.match(v)]
            if planted and rnd.random() < 0.5:
                v = rnd.choice(planted)[:10]
                return v + ("T12:00:00" if kind != "DATE" else "")
            return date(kind != "DATE")
        if kind == "LIST":
            pool = [v for v in by_prop.get(prop, []) if isinstance(v, str)] or text_pool
            return [rnd.choice(pool) for _ in range(rnd.randint(1, 3))]
        if kind == "POINT":
            return None
        planted = by_prop.get(prop, [])
        if planted and rnd.random() < 0.5:
            return str(rnd.choice(planted))
        if distinct and rnd.random() < 0.8:
            return f"v{rnd.randrange(10**9):09d}"
        if any(isinstance(v, (int, float)) for v in planted):  # a number kept as text
            return str(number("FLOAT", prop))
        if i is not None and ID_NAME.search(prop) and rnd.random() < 0.85:
            return f"{prop}_{i}"
        if DATE_NAME.search(prop) and rnd.random() < 0.8:  # a date kept as text
            return date(False)
        return rnd.choice(text_pool)

    nodes: dict[str, list[dict[str, Any]]] = {}
    types: dict[str, dict[str, str]] = {}
    for label, props in schema["nodes"].items():
        rows = []
        for i in range(NODES_PER_LABEL):
            r: dict[str, Any] = {"_k": f"{label}:{i}"}
            for p, kind in props.items():
                if rnd.random() < 0.05:
                    continue  # a missing property, as real graphs have
                v = value(kind, p, i)
                if v is not None:
                    r[p] = v
            rows.append(r)
        nodes[label] = rows
        conv = {p: _converter(k) for p, k in props.items() if _converter(k)}
        if conv:
            types[label] = conv
    rels: list[tuple] = []
    for a, t, b, props in schema["rels"]:
        seen = set()
        for _ in range(RELS_PER_PATTERN):
            i, j = rnd.randrange(NODES_PER_LABEL), rnd.randrange(NODES_PER_LABEL)
            if (i, j) in seen:
                continue
            seen.add((i, j))
            p = {k: v for k, kind in props.items() if (v := value(kind, k)) is not None}
            rels.append((t, a, f"{a}:{i}", b, f"{b}:{j}", p))
        conv = {p: _converter(k) for p, k in props.items() if _converter(k)}
        if conv:
            types.setdefault(t, {}).update(conv)
    return nodes, rels, types


def _converter(kind: str) -> str | None:
    return {"DATE": "date", "DATE_TIME": "datetime", "LOCAL_DATE_TIME": "localdatetime"}.get(
        kind.upper()
    )


# ---- the final RETURN's ORDER BY / LIMIT --------------------------------------------------------


def _bare(q: str) -> str:
    return STRINGS.sub(lambda m: "'" + "_" * (len(m.group()) - 2) + "'", q)


def _final(q: str, keyword: str) -> int | None:
    """Where `keyword` starts after the last top-level RETURN (outside strings and brackets)."""
    bare = _bare(q)
    depth, last_return, hit = 0, None, None
    kw = re.compile(keyword, re.IGNORECASE)
    ret = re.compile(r"RETURN\b", re.IGNORECASE)
    for i, ch in enumerate(bare):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and (i == 0 or not (bare[i - 1].isalnum() or bare[i - 1] == "_")):
            if ret.match(bare, i):
                last_return, hit = i, None
            elif last_return is not None and hit is None and kw.match(bare, i):
                hit = i
    return hit


def sorts_result(q: str) -> bool:
    return _final(q, r"ORDER\s+BY\b") is not None and not re.search(
        r"\bUNION\b", _bare(q), re.IGNORECASE
    )


def strip_limit(q: str) -> str | None:
    i = _final(q, r"LIMIT\b")
    if i is None or _final(q, r"SKIP\b") is not None:
        return None
    return q[:i].rstrip()


def _sort_key_column(q: str, names: list[str]) -> int | None:
    i = _final(q, r"ORDER\s+BY\b")
    if i is None:
        return None
    ends = [j for j in (_final(q, r"SKIP\b"), _final(q, r"LIMIT\b")) if j is not None and j > i]
    clause = q[i : min(ends) if ends else len(q)]
    first = re.sub(r"^ORDER\s+BY\s+", "", clause.strip(), flags=re.IGNORECASE)
    depth, cut = 0, len(first)
    for n, ch in enumerate(first):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            cut = n
            break
    expr = re.sub(r"\s+(ASC|DESC|ASCENDING|DESCENDING)\s*$", "", first[:cut].strip(), flags=re.I)
    expr = expr.strip().strip("`")
    low = [c.lower().strip("`") for c in names]
    if expr.lower() in low:
        return low.index(expr.lower())
    base = expr.split(".")[-1].lower()
    tails = [c.split(".")[-1] for c in low]
    return tails.index(base) if tails.count(base) == 1 else None


def _informative(rows: list[tuple]) -> bool:
    """Rows that say something: not empty, not only zeros, nulls and empty lists."""
    for r in rows:
        for v in r:
            if v not in (None, 0, 0.0, (), "", False):
                return True
    return False


def _tie(db: Neo4j, gold: str, g: list[tuple], p: list[tuple]) -> bool:
    unl = strip_limit(gold)
    if unl is None or len(p) != len(g) or not g:
        return False
    try:
        names, full = db.run(unl)
    except Exception:
        return False
    if len(full) <= len(g):
        return False
    if not sorts_result(unl):
        return contained_in(p, full)
    k, key = len(g), _sort_key_column(unl, names)
    if key is None:
        return contained_in(p, full[: k + 1])
    cut = _norm_value(full[k - 1][key], 6)
    if _norm_value(full[k][key], 6) != cut:
        return False
    above = [r for r in full[:k] if _norm_value(r[key], 6) != cut]
    tied = [r for r in full if _norm_value(r[key], 6) == cut]
    return contained_in(above, p) and contained_in(p, above + tied)


def _ordered_equal(gold: list[tuple], pred: list[tuple]) -> bool:
    return len(gold) == len(pred) and all(
        same_results([a], [b]) for a, b in zip(gold, pred, strict=True)
    )


def compare(db: Neo4j, gold: str, preds: list[str]) -> list[dict[str, Any]]:
    """Run gold and each prediction on the loaded graph: per prediction one of
    match | tie | differ | pred_error:<code> | gold_error | uninformative."""
    try:
        _, g = db.run(gold)
    except Exception as e:
        return [{"outcome": "gold_error", "error": str(e)[:120]} for _ in preds]
    out = []
    for q in preds:
        try:
            _, p = db.run(q)
        except CypherError as e:
            out.append({"outcome": f"pred_error:{e.code.split('.')[-1]}", "error": str(e)[:160]})
            continue
        except Exception as e:  # a timeout of the HTTP call, too many rows
            out.append({"outcome": f"pred_error:{type(e).__name__}", "error": str(e)[:160]})
            continue
        moved = align(g, p)
        if moved is not None:
            o = {"outcome": "match" if _informative(g) else "uninformative"}
            if moved is not p:
                o["columns_moved"] = True
            out.append(o)
        elif _tie(db, gold, g, p):
            out.append({"outcome": "tie"})
        else:
            out.append({"outcome": "differ", "gold_rows": len(g), "pred_rows": len(p)})
    return out


def order_check(db: Neo4j, gold: str, preds: list[str]) -> list[bool | None]:
    """On a graph without ties: does each prediction keep the gold's order? None = can't tell."""
    try:
        _, g = db.run(gold)
    except Exception:
        return [None for _ in preds]
    if len(g) < 2:
        return [None for _ in preds]
    res: list[bool | None] = []
    for q in preds:
        try:
            _, p = db.run(q)
        except Exception:
            res.append(False)
            continue
        moved = align(g, p)
        res.append(moved is not None and _ordered_equal(g, moved))
    return res


# ---- the checks per dataset -----------------------------------------------------------------


def _guard(q: str) -> str | None:
    ok, why = cypher_ok(q)
    return None if ok else f"syntax:{why}"


def check_fuzz(db: Neo4j, system: str, gold: str, preds: list[str]) -> list[dict[str, Any]]:
    """text2cypher: each prediction against the gold on random graphs from the schema."""
    schema = parse_schema(system)
    res = [
        {"verdict": None, "checked_by": "", "reason": "", "fuzz_agree": 0, "fuzz_informative": 0}
        for _ in preds
    ]
    for r, q in zip(res, preds, strict=True):
        if normalise(q) == normalise(gold):
            r.update(verdict="pass", checked_by="exact", reason="exact")
        elif why := _guard(q):
            r.update(verdict="fail", reason=why)
    todo = [i for i, r in enumerate(res) if r["verdict"] is None]
    if not todo:
        return res
    if not schema["nodes"]:
        for i in todo:
            res[i].update(verdict="unverified", reason="no_schema")
        return res
    lits = literals([gold] + [preds[i] for i in todo])
    ties = dict.fromkeys(todo, 0)
    gold_errors = 0
    for seed in FUZZ_SEEDS:
        if not todo:
            break
        nodes, rels, types = fuzz_graph(schema, seed, lits)
        try:
            load(db, nodes, rels, types)
        except Exception as e:
            for i in todo:
                res[i].update(verdict="unverified", reason=f"load_failed:{str(e)[:60]}")
            return res
        outs = compare(db, gold, [preds[i] for i in todo])
        if outs and outs[0]["outcome"] == "gold_error":
            gold_errors += 1
            continue
        still = []
        for i, o in zip(todo, outs, strict=True):
            k = o["outcome"]
            if k == "match":
                res[i]["fuzz_agree"] += 1
                res[i]["fuzz_informative"] += 1
            elif k == "uninformative":
                res[i]["fuzz_agree"] += 1
            elif k == "tie":
                ties[i] += 1
            else:
                res[i].update(verdict="fail", reason="fuzz_rows_differ" if k == "differ" else k)
                continue
            still.append(i)
        todo = still
    sorted_gold = sorts_result(gold)
    proven = [i for i in todo if res[i]["fuzz_informative"]]
    if proven and sorted_gold:
        nodes, rels, types = fuzz_graph(schema, ORDER_SEED, lits, distinct=True)
        try:
            load(db, nodes, rels, types)
            for i, ok in zip(
                proven, order_check(db, gold, [preds[i] for i in proven]), strict=True
            ):
                if ok is False:
                    res[i].update(verdict="fail", reason="order_differs")
                elif ok:
                    res[i]["order_checked"] = True
        except Exception:
            pass
    for i in todo:
        if res[i]["verdict"] is not None:
            continue
        if res[i]["fuzz_informative"]:
            res[i].update(verdict="pass", checked_by="fuzz_graph", reason="match")
        else:
            why = "ties_only" if ties[i] else "no_rows_returned"
            if gold_errors == len(FUZZ_SEEDS):
                why = "gold_error"
            res[i].update(verdict="unverified", reason=why)
    return res


def cypherbench_graph(name: str, root: Path = CYPHERBENCH) -> tuple[dict, list, dict]:
    """A CypherBench sampled graph as `load` inputs (key `eid`, dates converted)."""
    d = json.loads(
        (root / "graphs" / "simplekg_sampled" / f"{name}_sampled_simplekg.json").read_text(
            encoding="utf-8"
        )
    )
    types: dict[str, dict[str, str]] = {}
    for ent in d["schema"]["entities"]:
        conv = {p: "date" for p, k in (ent.get("properties") or {}).items() if k == "date"}
        if conv:
            types[ent["label"]] = conv
    for rel in d["schema"]["relations"]:
        conv = {p: "date" for p, k in (rel.get("properties") or {}).items() if k == "date"}
        if conv:
            types.setdefault(rel["label"], {}).update(conv)
    nodes: dict[str, list[dict[str, Any]]] = {}
    label_of: dict[str, str] = {}
    for e in d["entities"]:
        props = {k: v for k, v in (e.get("properties") or {}).items() if v is not None}
        for p in types.get(e["label"], {}):
            if p in props and not (isinstance(props[p], str) and DATE_RE.match(props[p])):
                props.pop(p)  # a date the date() function can't read
            elif p in props:
                props[p] = props[p][:10]
        row = {"eid": e["eid"], "name": e.get("name"), **props}
        nodes.setdefault(e["label"], []).append({k: v for k, v in row.items() if v is not None})
        label_of[e["eid"]] = e["label"]
    rels = []
    for r in d["relations"]:
        a, b = label_of.get(r["subj_id"]), label_of.get(r["obj_id"])
        if a and b:
            props = {k: v for k, v in (r.get("properties") or {}).items() if v is not None}
            for p in types.get(r["label"], {}):
                if p in props and isinstance(props[p], str) and DATE_RE.match(props[p]):
                    props[p] = props[p][:10]
                else:
                    props.pop(p, None)
            rels.append((r["label"], a, r["subj_id"], b, r["obj_id"], props))
    return nodes, rels, types


def check_on_graph(db: Neo4j, gold: str, preds: list[str]) -> list[dict[str, Any]]:
    """cypherbench: each prediction against the gold on the loaded (real) graph."""
    res = []
    todo, qs = [], []
    for q in preds:
        r = {"verdict": None, "checked_by": "", "reason": ""}
        if normalise(q) == normalise(gold):
            r.update(verdict="pass", checked_by="exact", reason="exact")
        elif why := _guard(q):
            r.update(verdict="fail", reason=why)
        else:
            todo.append(len(res))
            qs.append(q)
        res.append(r)
    for i, o in zip(todo, compare(db, gold, qs) if qs else [], strict=True):
        k = o["outcome"]
        if k == "match":
            res[i].update(verdict="pass", checked_by="graph", reason="match")
        elif k in ("uninformative", "tie"):
            res[i].update(
                verdict="unverified", reason="no_rows_returned" if k != "tie" else "ties_only"
            )
        elif k == "gold_error":
            res[i].update(verdict="unverified", reason="gold_error")
        else:
            res[i].update(verdict="fail", reason="rows_differ" if k == "differ" else k)
    return res


# ---- re-checking a trace run ----------------------------------------------------------------


def _lines(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # blank, or a line still being written
    return out


def recheck(run_dir: Path, db: Neo4j, root: Path = CYPHERBENCH, log=print) -> dict[str, int]:
    questions = {q["qid"]: q for q in _lines(run_dir / "questions.jsonl")}
    done = {
        r["rid"]
        for r in _lines(run_dir / "rechecks.jsonl")
        if r.get("checker", 1) >= CHECKER_VERSION or r.get("verdict") == "pass"
    }
    todo: dict[str, list[dict[str, Any]]] = {}
    sql: list[dict[str, Any]] = []
    for r in _lines(run_dir / "results.jsonl"):
        if not r.get("query") or r["rid"] in done or r.get("verdict") == "pass":
            continue
        if r.get("dialect") != "cypher":
            if r.get("reason") in ("rows_differ", "fuzz_rows_differ", "order_differs"):
                sql.append(r)  # the column-order rule (version 2) may pass it now
            continue
        if str(r.get("reason", "")).startswith("syntax"):
            continue
        todo.setdefault(r["qid"], []).append(r)
    counts: dict[str, int] = {}
    out_path = run_dir / "rechecks.jsonl"

    def write(rows: list[dict[str, Any]], verdicts: list[dict[str, Any]]) -> None:
        with out_path.open("a", encoding="utf-8", newline="\n") as f:
            for r, v in zip(rows, verdicts, strict=True):
                rec = {"rid": r["rid"], "qid": r["qid"], "level": r["level"], **v}
                rec["checker"] = CHECKER_VERSION
                rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[f"{r['origin']}/{v['verdict']}"] = (
                    counts.get(f"{r['origin']}/{v['verdict']}", 0) + 1
                )

    for r in sql:
        q = questions[r["qid"]]
        try:
            v = check_sql(r["query"], q["gold"], q.get("schema_script"), bool(q.get("real_rows")))
        except Exception as e:
            v = {
                "verdict": "unverified",
                "checked_by": "",
                "reason": f"check_error:{type(e).__name__}",
            }
        write([r], [{k: v[k] for k in ("verdict", "checked_by", "reason")}])
    if sql:
        log(f"sql: re-ran {len(sql)} failed answers with the column-order rule")
    cb: dict[str, list[str]] = {}
    for qid, rows in todo.items():
        if rows[0]["origin"] == "cypherbench":
            cb.setdefault(questions[qid]["meta"].get("db_id") or "", []).append(qid)
    for graph, qids in sorted(cb.items()):
        t0 = time.time()
        load(db, *cypherbench_graph(graph, root), key="eid", keep_key=True, index=True)
        log(f"loaded {graph} in {time.time() - t0:.0f}s; checking {len(qids)} questions")
        for qid in qids:
            rows = todo[qid]
            write(rows, check_on_graph(db, questions[qid]["gold"], [r["query"] for r in rows]))
    t2 = [qid for qid, rows in todo.items() if rows[0]["origin"] != "cypherbench"]
    t0 = time.time()
    for n, qid in enumerate(t2, 1):
        rows, q = todo[qid], questions[qid]
        write(rows, check_fuzz(db, q["system"], q["gold"], [r["query"] for r in rows]))
        if n % 200 == 0:
            log(f"text2cypher {n} of {len(t2)} ({(time.time() - t0) / n:.2f}s each)")
    db.wipe()
    return counts


def probe_cypherbench(db: Neo4j, graph: str, n: int, root: Path = CYPHERBENCH) -> dict[str, Any]:
    """How often the gold, on the sampled graph, returns the answer the dataset records (from
    the full graph), and how often it returns something informative at all."""
    d = [
        x
        for x in json.loads((root / "train.json").read_text(encoding="utf-8"))
        if x["graph"] == graph
    ]
    random.Random(0).shuffle(d)
    load(db, *cypherbench_graph(graph, root), key="eid", keep_key=True, index=True)
    same = informative = errors = 0
    for x in d[:n]:
        try:
            _, rows = db.run(x["gold_cypher"])
        except Exception:
            errors += 1
            continue
        want = [tuple(_hashable(v) for v in r) for r in json.loads(x["answer_json"])]
        same += same_results(want, rows)
        informative += _informative(rows)
    return {
        "graph": graph,
        "n": min(n, len(d)),
        "same_as_recorded": same,
        "informative": informative,
        "errors": errors,
    }


def _ended(run: Path) -> bool:
    kinds = {e.get("kind") for e in _lines(run / "events.jsonl")}
    return bool(kinds & {"DONE", "stopped", "aborted"})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recheck")
    r.add_argument("--run", required=True)
    r.add_argument(
        "--every",
        type=int,
        default=0,
        help="seconds: re-check again until the run ends, then once more",
    )
    p = sub.add_parser("probe-cypherbench")
    p.add_argument("--graph", required=True)
    p.add_argument("--n", type=int, default=200)
    ap.add_argument("--url", default=NEO4J_URL)
    a = ap.parse_args(argv)
    db = Neo4j(a.url)
    if not db.ok():
        print(f"no Neo4j answering at {a.url}", file=sys.stderr)
        return 2
    if a.cmd == "recheck":
        run = Path(a.run)
        while True:
            ended = _ended(run)  # read first: the pass after the end sees every answer
            counts = recheck(run, db, log=lambda m: print(m, flush=True))
            print(json.dumps({"ts": time.strftime("%H:%M:%S"), **counts}), flush=True)
            if not a.every or ended:
                break
            time.sleep(a.every)
    else:
        print(json.dumps(probe_cypherbench(db, a.graph, a.n)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
