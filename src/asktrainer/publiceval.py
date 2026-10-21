"""Scoring generated queries on the public val sets, here in the trainer (no NFL repo needed).

- **gretel** (SQLite): executed. Each example's database is rebuilt in an in-memory SQLite
  from the dataset's own `sql_context` (its CREATE TABLEs and INSERTs, kept by the public
  adapter in `public/gretel/raw/`); the gold and the predicted query both run, and the
  result sets are compared order-insensitively (as multisets of rows), numbers rounded to the
  gold's precision per column (an int and the same float are equal). A gold that fails to run
  is reported (`gold_failed`) and left out of `exec_acc`.
- **sqale** (SQLite) and the other SQL sets: exact match after whitespace normalisation, plus
  a parse check: `EXPLAIN <query>` against the schema (sqale's DDL from
  `public/sqale/raw/schemas_ddl.parquet`), which also catches unknown tables and columns.
- **text2cypher, cypherbench** (Cypher): exact match after whitespace normalisation, plus a
  light syntax check (balanced brackets and quotes outside strings, a read clause first, a
  RETURN, no write clause, no $parameters).

Every reply must be one JSON object `{"query": ...}` (`json_ok`); a reply that's a refusal
counts as wrong (every public gold is a query). Queries run with a time limit (a progress
handler aborts runaway joins).
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from asktrainer.data.schema import Example

TIME_LIMIT_S = 2.0
MAX_ROWS = 10_000

CYPHER_START = re.compile(r"^(OPTIONAL\s+MATCH|MATCH|WITH|UNWIND|CALL|RETURN)\b", re.IGNORECASE)
CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE
)
SQL_READ = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")


def normalise(q: str) -> str:
    return " ".join(q.strip().rstrip(";").split())


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


def cypher_ok(q: str) -> tuple[bool, str]:
    s = q.strip()
    if not CYPHER_START.match(s):
        return False, "start"
    bare = STRINGS.sub("''", s)
    if s.count("'") % 2 and bare.count("'") % 2:  # an unterminated string literal
        return False, "quotes"
    depth = {"(": 0, "[": 0, "{": 0}
    close = {")": "(", "]": "[", "}": "{"}
    for ch in bare:
        if ch in depth:
            depth[ch] += 1
        elif ch in close:
            depth[close[ch]] -= 1
            if depth[close[ch]] < 0:
                return False, "brackets"
    if any(depth.values()):
        return False, "brackets"
    if not re.search(r"\bRETURN\b", bare, re.IGNORECASE):
        return False, "no_return"
    if CYPHER_WRITE.search(bare):
        return False, "write"
    if re.search(r"\$\w+", bare):
        return False, "parameter"
    return True, "ok"


def _connect(script: str | None) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    if script:
        db.executescript(script)
    return db


def _run(db: sqlite3.Connection, q: str) -> list[tuple]:
    deadline = time.monotonic() + TIME_LIMIT_S
    db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = db.execute(q)
        rows = cur.fetchmany(MAX_ROWS + 1)
    finally:
        db.set_progress_handler(None, 0)
    if len(rows) > MAX_ROWS:
        raise RuntimeError("too many rows")
    return rows


def _decimals(v: float) -> int:
    d = Decimal(repr(v)).as_tuple().exponent
    return min(6, max(0, -d)) if isinstance(d, int) else 6


def _norm_value(v: Any, places: int | None) -> Any:
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, (int, float)):
        return round(float(v), places if places is not None else 6)
    return v


def same_results(gold: list[tuple], pred: list[tuple]) -> bool:
    """Order-insensitive multiset comparison; numbers at the gold's precision per column."""
    if not gold and not pred:
        return True
    if not gold or not pred or len(gold[0]) != len(pred[0]):
        return False
    ncol = len(gold[0])
    places: list[int | None] = []
    for c in range(ncol):
        floats = [r[c] for r in gold if isinstance(r[c], float)]
        places.append(max((_decimals(f) for f in floats), default=None))

    def norm(rows):
        return Counter(tuple(_norm_value(r[c], places[c]) for c in range(ncol)) for r in rows)

    return norm(gold) == norm(pred)


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


def score(ex: Example, output: str, data: PublicData) -> dict[str, Any]:
    item: dict[str, Any] = {"id": ex.id, "origin": ex.origin, "dialect": ex.dialect,
                            "json_ok": False, "parse_ok": False, "exact": False,
                            "exec_ok": None, "gold_failed": False, "reason": ""}  # fmt: skip
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
            if ex.origin == "gretel":
                item["exec_ok"] = False
            return item
        if ex.origin != "gretel":
            item["reason"] = "ok"
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


def verify(examples: Sequence[Example], preds: Sequence[dict[str, str]], data: PublicData):
    by_id = {e.id: e for e in examples}
    items = [score(by_id[p["id"]], p["output"], data) for p in preds if p["id"] in by_id]
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
