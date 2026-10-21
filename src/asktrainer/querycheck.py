"""Checking a generated query against the reference one, with the standard library only.

Used by the public validation scorer (`publiceval`) and by the trace generator (`tracegen`), which
runs on Windows next to llama-server with nothing but the Python standard library.

SQL (SQLite):
- read only and one statement, then `EXPLAIN` against the schema (unknown tables or columns fail);
- **real rows** where the dataset has them (gretel's `sql_context` INSERTs): gold and prediction
  run on that database and their results are compared;
- **fuzzed rows** everywhere: the schema is filled with random rows three times (small integer
  ranges so joins and groups line up, the queries' own literals mixed into the value pools, a
  few NULLs) and gold and prediction are compared on each copy. This is the idea of Spider's
  test-suite accuracy (Zhong, Yu and Klein 2020): two queries that agree on several random
  databases very likely mean the same thing. A database where the gold returns no rows tells
  nothing and doesn't count as evidence.
- results are compared as multisets of rows, numbers rounded to the gold's precision per column.
  When the gold sorts its final result (a top-level ORDER BY), one more copy is filled with
  distinct values (wide ranges, unique text), where ties are rare, and there the rows must also
  come back in the same order.
- ties at a LIMIT: with `ORDER BY wins DESC LIMIT 1` and two teams level, the gold and a correct
  answer can pick different rows. When the rows differ and the gold has a top-level LIMIT, the
  gold is run again without it (still sorted). When the first sort key is a returned column, a
  tie means the last kept row and the next one share its value, and the answer must hold the
  rows above the tie plus some of the tied ones; when it isn't, only the row right after the cut
  may stand in. A tie counts neither way; an answer sorted the wrong way fails; databases
  without a tie decide.

Cypher: no graph is available locally yet, so only a syntax check and an exact match after
normalising whitespace; anything else is `unverified`.
"""

from __future__ import annotations

import json
import random
import re
import sqlite3
import time
from collections import Counter
from decimal import Decimal
from typing import Any

TIME_LIMIT_S = 2.0
MAX_ROWS = 10_000
FUZZ_SEEDS = (11, 23, 37, 41)
FUZZ_ROWS = 30
ORDER_SEED = 53

CYPHER_START = re.compile(r"^(OPTIONAL\s+MATCH|MATCH|WITH|UNWIND|CALL|RETURN)\b", re.IGNORECASE)
CYPHER_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b", re.IGNORECASE
)
SQL_READ = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
STRINGS = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`")
SQ_LITERAL = re.compile(r"'((?:[^']|'')*)'")
NUM_LITERAL = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])")
DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def normalise(q: str) -> str:
    return " ".join(q.strip().rstrip(";").split())


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
    return _run_named(db, q)[1]


def _run_named(db: sqlite3.Connection, q: str) -> tuple[list[str], list[tuple]]:
    """(column names, rows)."""
    deadline = time.monotonic() + TIME_LIMIT_S
    db.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = db.execute(q)
        rows = cur.fetchmany(MAX_ROWS + 1)
        names = [d[0] for d in cur.description or []]
    finally:
        db.set_progress_handler(None, 0)
    if len(rows) > MAX_ROWS:
        raise RuntimeError("too many rows")
    return names, rows


def _decimals(v: float) -> int:
    d = Decimal(repr(v)).as_tuple().exponent
    return min(6, max(0, -d)) if isinstance(d, int) else 6


def _norm_value(v: Any, places: int | None) -> Any:
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, (int, float)):
        return round(float(v), places if places is not None else 6)
    return v


def _bags(gold: list[tuple], pred: list[tuple]) -> tuple[Counter, Counter] | None:
    """Both results as multisets of rows, numbers at the gold's precision per column (None when
    the column counts differ)."""
    if gold and pred and len(gold[0]) != len(pred[0]):
        return None
    ncol = len(gold[0]) if gold else len(pred[0]) if pred else 0
    places: list[int | None] = []
    for c in range(ncol):
        floats = [r[c] for r in gold if isinstance(r[c], float)]
        places.append(max((_decimals(f) for f in floats), default=None))

    def norm(rows):
        return Counter(tuple(_norm_value(r[c], places[c]) for c in range(ncol)) for r in rows)

    return norm(gold), norm(pred)


def same_results(gold: list[tuple], pred: list[tuple]) -> bool:
    """Order-insensitive multiset comparison; numbers at the gold's precision per column."""
    if not gold and not pred:
        return True
    if not gold or not pred:
        return False
    b = _bags(gold, pred)
    return b is not None and b[0] == b[1]


def contained_in(pred: list[tuple], pool: list[tuple]) -> bool:
    """Is every row of `pred` (counted with repeats) in `pool`?"""
    if not pred:
        return True
    b = _bags(pool, pred)
    return b is not None and not (b[1] - b[0])


# ---- tool calls -------------------------------------------------------------------------------

RAW_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
RAW_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


def calls_from_message(msg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The tool calls of an OpenAI-style assistant message as (name, arguments) pairs; falls
    back to Spark's raw `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
    text when the server left them in the content."""
    out: list[tuple[str, dict[str, Any]]] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        out.append((str(fn.get("name") or ""), args if isinstance(args, dict) else {}))
    if not out:
        for body in RAW_CALL.findall(msg.get("content") or ""):
            name = body.split("<arg_key>", 1)[0].strip()
            out.append((name, {k.strip(): v for k, v in RAW_ARG.findall(body)}))
    return out


# ---- fuzzed databases -------------------------------------------------------------------------


def _literals(queries: list[str]) -> tuple[list[str], list[float]]:
    strs: list[str] = []
    nums: list[float] = []
    for q in queries:
        for s in SQ_LITERAL.findall(q or ""):
            s = s.replace("''", "'")
            if 0 < len(s) <= 80 and "%" not in s:
                strs.append(s)
            elif "%" in s:  # a LIKE pattern: put its fixed part into the pool
                core = s.strip("%")
                if core and "%" not in core and "_" not in core:
                    strs.append(core)
        for n in NUM_LITERAL.findall(STRINGS.sub("''", q or "")):
            try:
                nums.append(float(n))
            except ValueError:
                pass
    return sorted(set(strs)), sorted(set(nums))


def _kind(decl: str, name: str) -> str:
    d, n = (decl or "").upper(), name.lower()
    if (
        "DATE" in d
        or "TIME" in d
        or n.endswith(("_date", "_at", "_time"))
        or n in ("date", "year_month")
    ):
        return "date"
    if "BOOL" in d:
        return "bool"
    if "INT" in d:
        return "int"
    if any(k in d for k in ("CHAR", "CLOB", "TEXT", "STRING")):
        return "text"
    if any(k in d for k in ("REAL", "FLOA", "DOUB", "DEC", "NUM", "MONEY")):
        return "real"
    return "text"


def _schema_tables(db: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY rowid"
        )
    ]


def _creates(script: str) -> list[str]:
    """The CREATE TABLE / VIEW statements of a schema script (rows and indexes left out)."""
    scratch = _connect(script)
    try:
        rows = scratch.execute(
            "SELECT type, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, rowid"
        ).fetchall()
    finally:
        scratch.close()
    return [sql for _, sql in rows]


def _pk_of(db: sqlite3.Connection, table: str) -> str | None:
    try:
        cols = list(db.execute(f'PRAGMA table_info("{table}")'))
    except sqlite3.Error:
        return None
    pks = [c[1] for c in cols if c[5]]
    return pks[0] if pks else None


def fuzz_db(
    creates: list[str],
    seed: int,
    literals: tuple[list[str], list[float]],
    rows: int = FUZZ_ROWS,
    distinct: bool = False,
) -> sqlite3.Connection:
    """A copy of the schema filled with random rows. `distinct`: wide value ranges and unique
    text, so sorting has (almost) no ties."""
    rnd = random.Random(seed)
    strs, nums = literals
    db = sqlite3.connect(":memory:")
    for sql in creates:
        try:
            db.execute(sql)
        except sqlite3.Error:
            pass
    text_pool = strs + ["alpha", "beta", "gamma", "delta"]
    date_pool = [s for s in strs if DATE_LIKE.match(s)]
    int_pool = [int(n) for n in nums if n == int(n) and abs(n) < 10**9]
    real_pool = [n for n in nums if abs(n) < 10**9]
    made: dict[tuple[str, str], list[Any]] = {}
    tables = _schema_tables(db)
    # referenced tables first so foreign keys can pick existing values
    deps = {t: {r[2] for r in db.execute(f'PRAGMA foreign_key_list("{t}")')} for t in tables}
    order: list[str] = []
    for _ in range(len(tables)):
        for t in tables:
            if t not in order and all(d in order or d == t or d not in tables for d in deps[t]):
                order.append(t)
    order += [t for t in tables if t not in order]

    def value(kind: str) -> Any:
        if kind == "int":
            if int_pool and rnd.random() < 0.3:
                return rnd.choice(int_pool)
            return rnd.randint(1, 10**6) if distinct else rnd.randint(1, 6)
        if kind == "real":
            if real_pool and rnd.random() < 0.3:
                return rnd.choice(real_pool)
            return round(rnd.uniform(0, 10**6 if distinct else 100), 2)
        if kind == "bool":
            return rnd.randint(0, 1)
        if kind == "date":
            if date_pool and rnd.random() < 0.35:
                return rnd.choice(date_pool)
            if distinct:
                day = f"{rnd.randint(1990, 2030)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"
                clock = (rnd.randint(0, 23), rnd.randint(0, 59), rnd.randint(0, 59))
                return f"{day} {clock[0]:02d}:{clock[1]:02d}:{clock[2]:02d}"
            return f"{rnd.randint(2015, 2025)}-{rnd.randint(1, 12):02d}-{rnd.randint(1, 28):02d}"
        if distinct and not (strs and rnd.random() < 0.3):
            return f"v{rnd.randrange(10**9):09d}"
        return rnd.choice(text_pool)

    for t in order:
        cols = list(db.execute(f'PRAGMA table_info("{t}")'))  # cid, name, type, notnull, dflt, pk
        fks = {
            r[3]: (r[2], r[4] or _pk_of(db, r[2]))
            for r in db.execute(f'PRAGMA foreign_key_list("{t}")')
        }
        names = [c[1] for c in cols]
        n_pk = sum(1 for c in cols if c[5])
        for i in range(rows):
            vals = []
            for c in cols:
                name, decl, notnull, pk = c[1], c[2], c[3], c[5]
                kind = _kind(decl, name)
                if pk and n_pk == 1 and kind in ("int", "text", "real"):
                    v: Any = i + 1 if kind != "text" else f"{name}_{i + 1}"
                elif name in fks and made.get(fks[name]):
                    v = rnd.choice(made[fks[name]])
                elif not notnull and not pk and rnd.random() < 0.06:
                    v = None
                else:
                    v = value(kind)
                vals.append(v)
            ph = ",".join("?" * len(vals))
            cl = ",".join(f'"{n}"' for n in names)
            try:
                db.execute(f'INSERT OR IGNORE INTO "{t}" ({cl}) VALUES ({ph})', vals)
            except sqlite3.Error:
                pass
        for name in names:
            try:
                made[(t, name)] = [r[0] for r in db.execute(f'SELECT "{name}" FROM "{t}"')]
            except sqlite3.Error:
                pass
    return db


# ---- the checks -------------------------------------------------------------------------------


def sorts_result(q: str) -> bool:
    """Does the outermost SELECT have an ORDER BY (not one in a subquery or a window)?"""
    return _top_level(q, r"ORDER\s+BY\b") is not None


def _top_level(q: str, keyword: str) -> int | None:
    """Where `keyword` (e.g. "LIMIT") starts at parenthesis depth 0, outside strings, or None."""
    bare = STRINGS.sub(lambda m: "'" + "_" * (len(m.group()) - 2) + "'", q)  # same length as q
    depth = 0
    kw = re.compile(keyword, re.IGNORECASE)
    for i, ch in enumerate(bare):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif (
            depth == 0
            and kw.match(bare, i)
            and (i == 0 or not (bare[i - 1].isalnum() or bare[i - 1] == "_"))
        ):
            return i
    return None


def strip_limit(q: str) -> str | None:
    """The query without its top-level LIMIT (and OFFSET), or None when it has none."""
    i = _top_level(q, r"LIMIT\b")
    return q[:i].rstrip() if i is not None else None


def _tie(db: sqlite3.Connection, gold: str, g: list[tuple], p: list[tuple]) -> bool:
    """Do gold and prediction differ only by which tied rows a LIMIT kept? The gold without its
    LIMIT is still sorted, so tied alternatives sit right after the cut; without an ORDER BY any
    rows of the full result are as good."""
    unl = strip_limit(gold)
    if unl is None or len(p) != len(g) or not g:
        return False
    try:
        names, full = _run_named(db, unl)
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
        return False  # no tie at the cut: the rows are simply different
    above = [r for r in full[:k] if _norm_value(r[key], 6) != cut]
    tied = [r for r in full if _norm_value(r[key], 6) == cut]
    return contained_in(above, p) and contained_in(p, above + tied)


def _sort_key_column(q: str, names: list[str]) -> int | None:
    """The result column the top-level ORDER BY sorts on first (by alias, name, or position)."""
    i = _top_level(q, r"ORDER\s+BY\b")
    if i is None:
        return None
    j = _top_level(q[i:], r"LIMIT\b")
    clause = q[i : i + j] if j is not None else q[i:]
    first = re.sub(r"^ORDER\s+BY\s+", "", clause.strip(), flags=re.IGNORECASE)
    depth, cut = 0, len(first)
    for n, ch in enumerate(first):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            cut = n
            break
    expr = re.sub(
        r"\s+(ASC|DESC)\b.*$|\s+NULLS\s+(FIRST|LAST)\b.*$",
        "",
        first[:cut].strip(),
        flags=re.IGNORECASE,
    )
    expr = expr.strip().strip('"`[]')
    if expr.isdigit():
        n = int(expr) - 1
        return n if 0 <= n < len(names) else None
    base = expr.split(".")[-1].strip('"`[]').lower()
    low = [c.lower() for c in names]
    return low.index(base) if base in low else None


def _ordered_equal(gold: list[tuple], pred: list[tuple]) -> bool:
    if len(gold) != len(pred):
        return False
    return all(same_results([g], [p]) for g, p in zip(gold, pred, strict=True))


def check_sql(pred: str, gold: str, schema_script: str | None, real_rows: bool) -> dict[str, Any]:
    """verdict: pass | fail | unverified (plus `checked_by` and a reason)."""
    out: dict[str, Any] = {
        "verdict": "fail",
        "checked_by": "",
        "reason": "",
        "parse_ok": False,
        "real": None,
        "fuzz_agree": 0,
        "fuzz_informative": 0,
        "exact": False,
    }
    q = (pred or "").strip().rstrip(";").strip()
    out["exact"] = normalise(q) == normalise(gold)
    if not SQL_READ.match(q):
        out["reason"] = "not_read_only"
        return out
    if ";" in STRINGS.sub("''", q):
        out["reason"] = "multiple_statements"
        return out
    if not schema_script:
        out.update(
            verdict="pass" if out["exact"] else "unverified",
            checked_by="exact" if out["exact"] else "",
            reason="no_schema",
        )
        return out
    try:
        creates = _creates(schema_script)
    except sqlite3.Error:
        out.update(verdict="pass" if out["exact"] else "unverified", reason="schema_failed")
        return out
    probe = sqlite3.connect(":memory:")
    for sql in creates:
        try:
            probe.execute(sql)
        except sqlite3.Error:
            pass
    try:
        probe.execute("EXPLAIN " + q)
        out["parse_ok"] = True
    except sqlite3.Error as e:
        out["reason"] = f"parse:{str(e)[:60]}"
        return out
    finally:
        probe.close()
    evidence = 0
    if real_rows:
        db = _connect(schema_script)
        try:
            g = _run(db, gold)
            try:
                p = _run(db, q)
            except Exception as e:
                out["real"], out["reason"] = False, f"pred:{type(e).__name__}"
                return out
            out["real"] = same_results(g, p)
            if not out["real"]:
                if _tie(db, gold, g, p):
                    out["real"], out["ties"] = None, out.get("ties", 0) + 1
                else:
                    out["reason"] = "rows_differ"
                    return out
            else:
                evidence += 1 if g else 0
        except Exception:
            out["real"] = None  # the gold itself doesn't run on its data
        finally:
            db.close()
    lits = _literals([gold, q])
    for seed in FUZZ_SEEDS:
        db = fuzz_db(creates, seed, lits)
        try:
            try:
                g = _run(db, gold)
            except Exception:
                continue
            try:
                p = _run(db, q)
            except Exception as e:
                out["reason"] = f"pred:{type(e).__name__}"
                return out
            if not same_results(g, p):
                if _tie(db, gold, g, p):
                    out["ties"] = out.get("ties", 0) + 1
                    continue
                out["reason"] = "fuzz_rows_differ"
                return out
            out["fuzz_agree"] += 1
            if g:
                out["fuzz_informative"] += 1
                evidence += 1
        finally:
            db.close()
    if evidence and sorts_result(gold):
        db = fuzz_db(creates, ORDER_SEED, lits, distinct=True)
        try:
            g = _run(db, gold)
            if len(g) > 1:
                try:
                    p = _run(db, q)
                except Exception as e:
                    out["reason"] = f"pred:{type(e).__name__}"
                    return out
                if not _ordered_equal(g, p):
                    out["reason"] = "order_differs" if same_results(g, p) else "fuzz_rows_differ"
                    return out
                out["order_checked"] = True
        except Exception:
            pass
        finally:
            db.close()
    if evidence:
        out.update(
            verdict="pass", checked_by="rows+fuzz" if out["real"] else "fuzz", reason="match"
        )
    elif out["exact"]:
        out.update(verdict="pass", checked_by="exact", reason="exact")
    else:
        out.update(
            verdict="unverified", reason="ties_only" if out.get("ties") else "no_rows_returned"
        )
    return out


def check_cypher(pred: str, gold: str) -> dict[str, Any]:
    ok, why = cypher_ok(pred or "")
    exact = normalise(pred or "") == normalise(gold)
    if not ok:
        return {
            "verdict": "fail",
            "checked_by": "",
            "reason": f"syntax:{why}",
            "parse_ok": False,
            "exact": exact,
        }
    if exact:
        return {
            "verdict": "pass",
            "checked_by": "exact",
            "reason": "exact",
            "parse_ok": True,
            "exact": True,
        }
    return {
        "verdict": "unverified",
        "checked_by": "",
        "reason": "no_graph",
        "parse_ok": True,
        "exact": False,
    }
