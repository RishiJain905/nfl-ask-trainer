"""Text helpers for the public-data adapters (`public.py`): no files, no network.

- A small regex lexer for SQL and Cypher that splits a query into code, string literals, quoted
  identifiers and comments, so the checks below never look inside a literal.
- The read-only filter (`normalize_query`): exactly one statement, no comments, no writes, no
  procedure calls; returns the query with its whitespace collapsed.
- SQL schema text: the CREATE TABLE / CREATE VIEW statements of a DDL blob (INSERTs, indexes and
  comments dropped), table names, which tables a query uses, and a SQLite compile check.
- Graph schema text: parsers for the schema formats the Cypher datasets ship, and one compact
  renderer (`render_graph_schema`).
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, NamedTuple

Lang = Literal["sql", "cypher"]

# ---- lexing -----------------------------------------------------------------------------------

_SQL_LEX = re.compile(
    r"""(?P<comment>--[^\n]*|/\*.*?\*/)
      | (?P<str>'(?:[^']|'')*')
      | (?P<ident>"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]\n]*\])
    """,
    re.S | re.X,
)
_CYPHER_LEX = re.compile(
    r"""(?P<comment>//[^\n]*|/\*.*?\*/)
      | (?P<str>'(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")
      | (?P<ident>`(?:[^`]|``)*`)
    """,
    re.S | re.X,
)


def pieces(text: str, lang: Lang) -> list[tuple[str, str]]:
    """Split `text` into ("code" | "comment" | "str" | "ident", piece) in order."""
    rx = _SQL_LEX if lang == "sql" else _CYPHER_LEX
    out: list[tuple[str, str]] = []
    pos = 0
    for m in rx.finditer(text):
        if m.start() > pos:
            out.append(("code", text[pos : m.start()]))
        out.append((m.lastgroup or "code", m.group()))
        pos = m.end()
    if pos < len(text):
        out.append(("code", text[pos:]))
    return out


def skeleton(text: str, lang: Lang) -> str:
    """The code only: literals, quoted identifiers and comments are blanked to one space."""
    return "".join(s if k == "code" else " " for k, s in pieces(text, lang))


def has_comment(text: str, lang: Lang) -> bool:
    """A comment outside literals (an unterminated `/*` counts)."""
    for kind, s in pieces(text, lang):
        if kind == "comment" or (kind == "code" and "/*" in s):
            return True
    return False


def has_stray_quote(text: str, lang: Lang) -> bool:
    """An unterminated string or quoted identifier leaves a quote character in the code."""
    return any(k == "code" and ("'" in s or '"' in s or "`" in s) for k, s in pieces(text, lang))


def strip_comments(text: str, lang: Lang) -> str:
    return "".join(" " if k == "comment" else s for k, s in pieces(text, lang))


def collapse(text: str, lang: Lang, *, tidy: bool = False) -> str:
    """Collapse whitespace runs in the code to one space (literals untouched). Cypher also loses
    the literal two-character `\\n` sequences some datasets leave between clauses. `tidy` also
    removes the space inside `( x )` and before `,` (for DDL)."""
    out = []
    for kind, s in pieces(text, lang):
        if kind == "code":
            if lang == "cypher":
                s = re.sub(r"\\[nrt]", " ", s)
            s = re.sub(r"\s+", " ", s)
            if tidy:
                s = re.sub(r"\(\s+", "(", s)
                s = re.sub(r"\s+([,)])", r"\1", s)
        out.append(s)
    return "".join(out).strip()


def split_statements(text: str, lang: Lang) -> list[str]:
    """Split on `;` outside literals and comments; empty statements are dropped."""
    stmts: list[str] = []
    cur: list[str] = []
    for kind, s in pieces(text, lang):
        if kind == "code" and ";" in s:
            parts = s.split(";")
            for j, part in enumerate(parts):
                cur.append(part)
                if j < len(parts) - 1:
                    stmts.append("".join(cur))
                    cur = []
        else:
            cur.append(s)
    stmts.append("".join(cur))
    return [x.strip() for x in stmts if x.strip()]


def _words(skel: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9$]*", skel)


# ---- the read-only filter ---------------------------------------------------------------------

_SQL_WRITE = frozenset(
    "INSERT UPDATE DELETE DROP ALTER CREATE TRUNCATE PRAGMA ATTACH DETACH VACUUM REINDEX GRANT "
    "REVOKE EXEC EXECUTE MERGE UPSERT".split()
)
_CYPHER_WRITE = re.compile(
    r"(?<![\w.:$])(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD|FOREACH)\b(?!\s*:)", re.I
)
_CYPHER_START = re.compile(r"\s*(MATCH|OPTIONAL|WITH|UNWIND|CALL|RETURN)\b", re.I)
_CALL = re.compile(r"(?<![\w.:$])CALL\b", re.I)
_CALL_SUBQUERY = re.compile(r"\s*(\([^)]*\)\s*)?\{")
_CYPHER_NAMESPACE = re.compile(r"(?<![\w.:$])(apoc|gds|dbms)\s*\.", re.I)
_BRACKETS = {")": "(", "]": "[", "}": "{"}


def _sql_problem(skel: str) -> str | None:
    words = [w.upper() for w in _words(skel)]
    if _SQL_WRITE.intersection(words) or re.search(r"\bREPLACE\b(?!\s*\()", skel, re.I):
        return "write_statement"
    if not words or words[0] not in ("SELECT", "WITH") or "SELECT" not in words:
        return "not_select"
    return None


def _balanced(skel: str) -> bool:
    stack: list[str] = []
    for ch in skel:
        if ch in "([{":
            stack.append(ch)
        elif ch in _BRACKETS:
            if not stack or stack.pop() != _BRACKETS[ch]:
                return False
    return not stack


def _cypher_problem(skel: str) -> str | None:
    if _CYPHER_WRITE.search(skel) or re.search(r"\bIN\s+TRANSACTIONS\b", skel, re.I):
        return "write_statement"
    if not _CYPHER_START.match(skel):
        return "not_read_query"
    for m in _CALL.finditer(skel):
        if not _CALL_SUBQUERY.match(skel, m.end()):
            return "procedure_call"  # CALL db.x() / apoc.x(): only CALL { } subqueries pass
    if _CYPHER_NAMESPACE.search(skel):
        return "procedure_call"
    if not re.search(r"(?<![\w.:$])RETURN\b", skel, re.I):
        return "no_return"
    if not _balanced(skel):
        return "unbalanced"
    return None


def normalize_query(text: object, lang: Lang) -> tuple[str | None, str | None]:
    """The query as it will be trained on, or the reason it is dropped: `(query, None)` or
    `(None, reason)`. Reasons: empty_query, comment, multi_statement, unterminated_quote,
    not_select / not_read_query, write_statement, procedure_call, no_return, unbalanced."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty_query"
    if has_comment(text, lang):
        return None, "comment"
    stmts = split_statements(text, lang)
    if not stmts:
        return None, "empty_query"
    if len(stmts) > 1:
        return None, "multi_statement"
    q = collapse(stmts[0], lang)
    if not q:
        return None, "empty_query"
    if has_stray_quote(q, lang):
        return None, "unterminated_quote"
    skel = skeleton(q, lang)
    problem = _sql_problem(skel) if lang == "sql" else _cypher_problem(skel)
    return (None, problem) if problem else (q, None)


_PATTERN_NODE = re.compile(r"\(\s*(\w+)\s*(?::|\)|\{|\|)")
_PATTERN_REL = re.compile(r"\[\s*(\w+)\s*(?::|\]|\{|\*)")
_PATTERN_PATH = re.compile(
    r"(?<![\w.])(\w+)\s*=\s*(?:(?:shortestPath|allShortestPaths)\s*\(\s*)?\("
)
_LAST_RETURN = re.compile(
    r"(?<![\w.:$])RETURN\b(.*?)(?=(?<![\w.:$])(?:ORDER\s+BY|SKIP|LIMIT|UNION)\b|\}|$)",
    re.I | re.S,
)


def _top_level_items(text: str) -> list[str]:
    items, depth, cur = [], 0, []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    items.append("".join(cur).strip())
    return items


def returns_whole_entity(query: str) -> bool:
    """Whether the final RETURN gives back a bare node, relationship or path variable (or `*`),
    e.g. `RETURN m` / `RETURN n, r`: the Ask prompts want named properties instead."""
    skel = skeleton(query, "cypher")
    bound = (
        set(_PATTERN_NODE.findall(skel))
        | set(_PATTERN_REL.findall(skel))
        | set(_PATTERN_PATH.findall(skel))
    )
    found = list(_LAST_RETURN.finditer(skel))
    if not found:
        return False
    body = re.sub(r"^\s*DISTINCT\s+", "", found[-1].group(1), flags=re.I)
    items = _top_level_items(body)
    return any(it == "*" or (re.fullmatch(r"\w+", it) and it in bound) for it in items)


_JUNK_QUESTION = re.compile(
    r"domain grounding|generated question|question set|questions? only|rewrite|strategy|"
    r"\b(?:easy|simple|moderate|hard|basic|straightforward|difficult)\s+questions?\b|"
    r"^\W*(?:simple|moderate|hard|easy)\W*$|^keep it\b|^provide (?:a |the )?(?:mid|moderate|hard)",
    re.I,
)
_STOP = frozenset(
    "the and for with that this from all any each every per are was were has have had not but "
    "what which who whom how many much does did can could would should please give show list "
    "find get tell let its their there then than them they you your our out one two into over "
    "who when where why also only just more most less least some such these those been being "
    "select from where group order join left right inner outer null count sum avg max min "
    "distinct having like between case when then else end limit desc asc union cast".split()
)


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {_stem(w) for w in words if len(w) >= 3 and w not in _STOP}


def question_aligned(question: str, sql: str) -> bool:
    """A cheap guard against junk questions (SQaLe has generator notes such as "Easy questions",
    "**Moderate**" or "Define domain grounding" in place of a question): the text is not a known
    junk pattern, starts with a letter or digit, and shares at least one content word with the
    query (identifier parts split at `_` and camelCase, plus the words of its string literals)."""
    q = question.strip()
    if not q or not q[0].isalnum() or _JUNK_QUESTION.search(q):
        return False
    sql_tokens: set[str] = set()
    for kind, s in pieces(sql, "sql"):
        if kind == "str":
            sql_tokens |= _content_tokens(s)
        else:
            for w in re.findall(r"[A-Za-z0-9_]+", s):
                words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", w).replace("_", " ")
                sql_tokens |= _content_tokens(words)
    return bool(_content_tokens(q) & sql_tokens)


def norm_key(text: str) -> str:
    """Dedupe key: case-folded, whitespace-collapsed."""
    return " ".join(text.split()).casefold()


# ---- SQL schema text --------------------------------------------------------------------------

_NAME = r'(?:"(?:[^"]|"")+"|`[^`]+`|\[[^\]]+\]|[\w$]+)'
_CREATE = re.compile(
    r"^\s*CREATE\s+(?:(?:LOCAL|GLOBAL)\s+)?(?:TEMP(?:ORARY)?\s+)?(?P<kind>TABLE|VIEW)\s+"
    rf"(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>{_NAME}(?:\s*\.\s*{_NAME})*)",
    re.I,
)


class DdlStatement(NamedTuple):
    name: str  # the table / view name, lower-cased, quotes removed
    kind: str  # "TABLE" | "VIEW"
    text: str  # one line, ending with ";"


def _bare(name: str) -> str:
    last = re.findall(_NAME, name)[-1]
    if last[0] in '"`[':
        last = last[1:-1]
    return last.lower()


def ddl_statements(ddl: object) -> list[DdlStatement]:
    """The CREATE TABLE / CREATE VIEW statements of `ddl`, one line each, comments removed.
    INSERTs, indexes, pragmas and the rest are dropped; a repeated name keeps the first."""
    if not isinstance(ddl, str):
        return []
    out: list[DdlStatement] = []
    seen: set[str] = set()
    for st in split_statements(strip_comments(ddl, "sql"), "sql"):
        m = _CREATE.match(st)
        if not m:
            continue
        name = _bare(m.group("name"))
        if name in seen:
            continue
        seen.add(name)
        out.append(
            DdlStatement(name, m.group("kind").upper(), collapse(st, "sql", tidy=True) + ";")
        )
    return out


def tokens_of(query: str) -> set[str]:
    """Lower-cased words of the code plus the contents of quoted identifiers."""
    toks: set[str] = set()
    for kind, s in pieces(query, "sql"):
        if kind == "code":
            toks.update(w.lower() for w in _words(s))
        elif kind == "ident":
            toks.add(s[1:-1].lower())
    return toks


def used_tables(query: str, names: Iterable[str]) -> set[str]:
    """Which of `names` (lower-case table names) the query mentions."""
    return set(names) & tokens_of(query)


_OK_CACHE: dict[str, bool] = {}


def statement_creates(text: str) -> bool:
    """Whether SQLite accepts this one CREATE statement (cached)."""
    ok = _OK_CACHE.get(text)
    if ok is None:
        con = sqlite3.connect(":memory:")
        try:
            con.execute(text)
            ok = True
        except Exception:
            ok = False
        finally:
            con.close()
        if len(_OK_CACHE) > 200_000:
            _OK_CACHE.clear()
        _OK_CACHE[text] = ok
    return ok


def sqlite_problem(ddl: Iterable[str], query: str) -> str | None:
    """Create `ddl` in an empty in-memory SQLite database and compile `query` (EXPLAIN: nothing
    runs). Returns ddl_not_sqlite, sqlite_compile_error or None."""
    con = sqlite3.connect(":memory:")
    try:
        for st in ddl:
            try:
                con.execute(st)
            except Exception:
                return "ddl_not_sqlite"
        try:
            con.execute("EXPLAIN " + query)
        except Exception:
            return "sqlite_compile_error"
        return None
    finally:
        con.close()


# ---- graph schema text ------------------------------------------------------------------------


@dataclass
class GraphSchema:
    nodes: dict[str, dict[str, str]] = field(default_factory=dict)  # label -> {property: type}
    rels: list[tuple[str, str, str]] = field(default_factory=list)  # (start, type, end)
    rel_props: dict[str, dict[str, str]] = field(default_factory=dict)  # type -> {property: type}


def norm_type(t: str | None) -> str:
    """A property type in one vocabulary; '' means STRING (the renderer leaves it out)."""
    t = (t or "").strip()
    u = t.upper()
    if u in ("", "STRING", "STR", "TEXT"):
        return ""
    if u in ("INT", "INTEGER", "LONG"):
        return "INTEGER"
    if u in ("FLOAT", "DOUBLE", "NUMBER"):
        return "FLOAT"
    if u.startswith("LIST") or u.startswith("ARRAY"):
        return "LIST"
    if u in ("BOOL", "BOOLEAN"):
        return "BOOLEAN"
    return re.sub(r"[^A-Z0-9_]", "", u) or ""


def _unquote(label: str) -> str:
    label = label.strip()
    return label[1:-1] if len(label) > 1 and label[0] == label[-1] == "`" else label


def render_graph_schema(g: GraphSchema) -> str | None:
    """`Nodes (...)` with `- Label {prop, prop: TYPE}` lines and `Relationships:` with
    `- (A)-[:TYPE {prop: TYPE}]->(B)` lines; None when there is no node label."""
    if not g.nodes:
        return None

    def props(d: dict[str, str]) -> str:
        return ", ".join(f"{k}: {t}" if t else k for k, t in d.items())

    lines = ["Nodes (properties are STRING unless typed):"]
    for label, d in g.nodes.items():
        lines.append(f"- {label} {{{props(d)}}}" if d else f"- {label}")
    lines.append("Relationships:")
    seen: set[tuple[str, str, str]] = set()
    n_rel = 0
    for s, t, e in g.rels:
        if (s, t, e) in seen:
            continue
        seen.add((s, t, e))
        rp = g.rel_props.get(t)
        body = f"[:{t} {{{props(rp)}}}]" if rp else f"[:{t}]"
        lines.append(f"- ({s})-{body}->({e})")
        n_rel += 1
    if not n_rel:
        lines.append("- (none)")
    return "\n".join(lines)


_A_LABEL = re.compile(r"^- \*\*(?P<label>.+)\*\*\s*$")
_A_PROP = re.compile(r"^- `(?P<name>[^`]+)`(?::\s*(?P<type>[A-Za-z_]+))?")
_A_REL = re.compile(r"^\(:(?P<s>.+?)\)-\[:(?P<t>.+?)\]->\(:(?P<e>.+?)\)\s*$")
_B_NODE = re.compile(r"^(?P<label>[^\s{]+)\s*\{(?P<props>.*)\}\s*$")
_B_REL = re.compile(
    r"\{\s*'start':\s*(?P<s>.+?),\s*'type':\s*(?P<t>.+?),\s*'end':\s*(?P<e>.+?)\s*\}\s*$"
)


def _braced_props(inner: str) -> dict[str, str]:
    """`a: STRING, b: INTEGER` -> {a: '', b: 'INTEGER'}."""
    out: dict[str, str] = {}
    for part in inner.split(","):
        if part.strip():
            k, _, typ = part.partition(":")
            out[k.strip()] = norm_type(typ)
    return out


def _rel(m: re.Match[str]) -> tuple[str, str, str]:
    return _unquote(m.group("s")), _unquote(m.group("t")), _unquote(m.group("e"))


def parse_neo4j_schema(text: str) -> GraphSchema | None:
    """The `Node properties:` / `Relationship properties:` / `The relationships:` text that
    Neo4j's schema helpers print (the most common format in text2cypher-2025v1), in its
    bulleted form (`- **Label**` / `- `prop`: TYPE`) and its inline form (`Label {prop: TYPE}`)."""
    g = GraphSchema()
    section = ""
    cur = ""
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("Node properties:"):
            section, cur = "node", ""
            continue
        if t.startswith("Relationship properties:"):
            section, cur = "relprops", ""
            continue
        if t.startswith("The relationships:"):
            section, cur = "rels", ""
            continue
        if section in ("node", "relprops"):
            target = g.nodes if section == "node" else g.rel_props
            if not line.startswith((" ", "\t")):
                m = _A_LABEL.match(t)
                if m:
                    cur = _unquote(m.group("label"))
                    target.setdefault(cur, {})
                elif m := _B_NODE.match(t):  # inline: Label {prop: TYPE, ...}
                    target[_unquote(m.group("label"))] = _braced_props(m.group("props"))
                continue
            m = _A_PROP.match(t)
            if m and cur:
                name, typ = m.group("name"), m.group("type")
                if ":" in name:  # `roles: LIST` Min Size: ...
                    name, _, typ = (x.strip() for x in name.partition(":"))
                target[cur][name] = norm_type(typ)
        elif section == "rels":
            m = _A_REL.match(t)
            if m:
                g.rels.append(_rel(m))
    return g if g.nodes else None


def parse_relevant_schema(text: str) -> GraphSchema | None:
    """`Graph schema: Relevant node labels and their properties (with datatypes) are:` followed by
    `Label {prop: TYPE, ...}` lines (or bare labels) and `Relevant relationships are:` with
    `{'start': A, 'type': T, 'end': B}` lines (the other big format in text2cypher-2025v1)."""
    g = GraphSchema()
    mode = ""
    for line in text.splitlines():
        t = line.strip()
        if not t:
            continue
        if "node labels" in t and t.endswith("are:"):
            mode = "nodes"
            continue
        if "relationship properties" in t and t.endswith("are:"):
            mode = "relprops"
            continue
        if "relationships are" in t and t.endswith(":"):
            mode = "rels"
            continue
        if mode in ("nodes", "relprops"):
            target = g.nodes if mode == "nodes" else g.rel_props
            if m := _B_NODE.match(t):
                target[_unquote(m.group("label"))] = _braced_props(m.group("props"))
            elif re.fullmatch(r"[\w`]+", t):  # a label with no property list
                target[_unquote(t)] = {}
        elif mode == "rels":
            m = _B_REL.match(t)
            if m:
                g.rels.append(_rel(m))
    return g if g.nodes else None


def parse_apoc_schema(text: str) -> GraphSchema | None:
    """The JSON that `apoc.meta.schema()` returns: `{"Label": {"type": "node", "properties":
    {...}, "relationships": {"REL": {"direction": "out", "labels": [...]}}}, "REL":
    {"type": "relationship", "properties": {...}}}`."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    g = GraphSchema()

    def props(d: object) -> dict[str, str]:
        if not isinstance(d, dict):
            return {}
        return {k: norm_type(v.get("type") if isinstance(v, dict) else None) for k, v in d.items()}

    for name, d in data.items():
        if isinstance(d, dict) and d.get("type") == "node":
            g.nodes[name] = props(d.get("properties"))
        elif isinstance(d, dict) and d.get("type") == "relationship" and d.get("properties"):
            g.rel_props[name] = props(d["properties"])
    for name, d in data.items():
        if not (isinstance(d, dict) and d.get("type") == "node"):
            continue
        for rel, rd in (d.get("relationships") or {}).items():
            for other in rd.get("labels") or []:
                g.rels.append(
                    (name, rel, other) if rd.get("direction") == "out" else (other, rel, name)
                )
    return g if g.nodes else None


def parse_graph_schema(text: object) -> GraphSchema | None:
    """Parse whichever of the text2cypher schema formats `text` is (the neo4j bulleted and
    inline text, the `Relevant node labels` text, apoc's JSON); None for the rest."""
    if not isinstance(text, str):
        return None
    t = text.lstrip()
    if t.startswith("Node properties:"):
        return parse_neo4j_schema(t)
    if t.startswith(("Graph schema: Relevant node labels", "Relevant node labels")):
        return parse_relevant_schema(t)
    if t.startswith("{") and '"type"' in t[:4000]:
        return parse_apoc_schema(t)
    return None
