"""The public-data adapters (no network): tiny hand-made rows in each dataset's raw format."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from asktrainer.data import public
from asktrainer.data.public import (
    Cand,
    clean_pair,
    convert,
    dedupe,
    pick_examples,
    round_robin,
    split_by_db,
    stable_rng,
)
from asktrainer.data.public_prompt import system_prompt
from asktrainer.data.public_text import (
    ddl_statements,
    normalize_query,
    parse_graph_schema,
    question_aligned,
    render_graph_schema,
    returns_whole_entity,
    sqlite_problem,
    used_tables,
)
from asktrainer.data.schema import Example, read_jsonl

# ---- the read-only filter ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT name FROM emp WHERE salary > 10;", "SELECT name FROM emp WHERE salary > 10"),
        ("WITH t AS (SELECT 1 AS x) SELECT x FROM t", "WITH t AS (SELECT 1 AS x) SELECT x FROM t"),
        ("SELECT\n  a,\n  b   FROM  t", "SELECT a, b FROM t"),
        ("SELECT a FROM t WHERE n = 'two  spaces'", "SELECT a FROM t WHERE n = 'two  spaces'"),
        (
            "SELECT a FROM t WHERE n = '--not a comment'",
            "SELECT a FROM t WHERE n = '--not a comment'",
        ),
        (
            "SELECT a FROM t WHERE n = 'delete; drop table x'",
            "SELECT a FROM t WHERE n = 'delete; drop table x'",
        ),
        ("SELECT REPLACE(name, 'a', 'b') FROM t", "SELECT REPLACE(name, 'a', 'b') FROM t"),
        ('SELECT "delete" FROM t', 'SELECT "delete" FROM t'),
    ],
)
def test_sql_kept(sql, expected):
    assert normalize_query(sql, "sql") == (expected, None)


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        ("", "empty_query"),
        (None, "empty_query"),
        ("INSERT INTO t VALUES (1)", "write_statement"),
        ("UPDATE t SET a = 1", "write_statement"),
        ("DELETE FROM t", "write_statement"),
        ("DROP TABLE t", "write_statement"),
        ("CREATE TABLE t (a INT)", "write_statement"),
        ("WITH x AS (SELECT 1) DELETE FROM t", "write_statement"),
        ("SELECT 1; SELECT 2", "multi_statement"),
        ("SELECT 1; DROP TABLE t", "multi_statement"),
        ("SELECT a FROM t -- why", "comment"),
        ("SELECT a /* why */ FROM t", "comment"),
        ("SELECT a FROM t /* unterminated", "comment"),
        ("SELECT a FROM t WHERE n = 'oops", "unterminated_quote"),
        ("REPLACE INTO t VALUES (1)", "write_statement"),
        ("PRAGMA table_info(t)", "write_statement"),
    ],
)
def test_sql_dropped(sql, reason):
    assert normalize_query(sql, "sql") == (None, reason)


@pytest.mark.parametrize(
    ("cy", "expected"),
    [
        (
            "MATCH (n:P)\nWHERE n.age > 3\nRETURN n.name AS name;",
            "MATCH (n:P) WHERE n.age > 3 RETURN n.name AS name",
        ),
        # text2cypher keeps a literal backslash-n between clauses
        ("MATCH (n:P)\\nRETURN n.name", "MATCH (n:P) RETURN n.name"),
        ("OPTIONAL MATCH (n:P) RETURN n.name", "OPTIONAL MATCH (n:P) RETURN n.name"),
        (
            "UNWIND [1, 2] AS x MATCH (n:P {id: x}) RETURN n.name",
            "UNWIND [1, 2] AS x MATCH (n:P {id: x}) RETURN n.name",
        ),
        (
            "MATCH (n:P) WHERE n.url = 'http://a.b' RETURN n.name",
            "MATCH (n:P) WHERE n.url = 'http://a.b' RETURN n.name",
        ),
        ("MATCH (n:P {set: 1}) RETURN n.create AS c", "MATCH (n:P {set: 1}) RETURN n.create AS c"),
        ("MATCH (n:Delete) RETURN n.name", "MATCH (n:Delete) RETURN n.name"),
        (
            "CALL { MATCH (a:A) RETURN a.x AS x UNION MATCH (b:B) RETURN b.x AS x } RETURN x",
            "CALL { MATCH (a:A) RETURN a.x AS x UNION MATCH (b:B) RETURN b.x AS x } RETURN x",
        ),
        (
            "MATCH (m:M) CALL (m) { MATCH (m)-->(d) RETURN count(d) AS c } RETURN m.t, c",
            "MATCH (m:M) CALL (m) { MATCH (m)-->(d) RETURN count(d) AS c } RETURN m.t, c",
        ),
    ],
)
def test_cypher_kept(cy, expected):
    assert normalize_query(cy, "cypher") == (expected, None)


@pytest.mark.parametrize(
    ("cy", "reason"),
    [
        ("CREATE (n:P {name: 'x'}) RETURN n.name", "write_statement"),
        ("MATCH (n:P) SET n.x = 1 RETURN n.name", "write_statement"),
        ("MATCH (n:P) DETACH DELETE n RETURN 1", "write_statement"),
        ("MERGE (n:P {id: 1}) RETURN n.id", "write_statement"),
        ("MATCH (n:P) REMOVE n.x RETURN n.id", "write_statement"),
        ("LOAD CSV FROM 'file:///a.csv' AS r RETURN r", "write_statement"),
        ("MATCH (n) FOREACH (x IN [1] | SET n.v = x) RETURN n.id", "write_statement"),
        ("CALL db.labels() YIELD label RETURN label", "procedure_call"),
        (
            "MATCH (n:P) CALL apoc.create.node(['A'], {}) YIELD node RETURN node.id",
            "procedure_call",
        ),
        ("MATCH (n:P) RETURN apoc.text.join([n.a], ',') AS s", "procedure_call"),
        ("MATCH (n:P) WHERE n.x = 1", "no_return"),
        ("MATCH (n:P RETURN n.name", "unbalanced"),
        ("MATCH (n:P) RETURN n.name // trailing", "comment"),
        ("MATCH (n:P) RETURN n.name; MATCH (m) RETURN m.id", "multi_statement"),
        ("EXPLAIN MATCH (n) RETURN n.id", "not_read_query"),
    ],
)
def test_cypher_dropped(cy, reason):
    assert normalize_query(cy, "cypher") == (None, reason)


def test_returns_whole_entity():
    assert returns_whole_entity("MATCH (m:Movie) RETURN m")
    assert returns_whole_entity("MATCH (a)-[r:R]->(b) RETURN a.name, r")
    assert returns_whole_entity("MATCH p = (a)-[:R]->(b) RETURN p")
    assert returns_whole_entity("MATCH (m) RETURN DISTINCT m ORDER BY m.x LIMIT 3")
    assert returns_whole_entity("MATCH (m) RETURN *")
    assert not returns_whole_entity("MATCH (m:Movie) RETURN m.title AS title")
    assert not returns_whole_entity("MATCH (a:A) WITH collect(a.x) AS xs RETURN xs")
    assert not returns_whole_entity("MATCH (m) RETURN count(m) AS c")


def test_question_aligned_rejects_generator_notes():
    sql = "SELECT tag_type, description FROM tags_type WHERE tag_type = 'ldap_bind'"
    assert question_aligned("Which tag types describe the ldap bind setting?", sql)
    assert question_aligned("Show the description per tagType", sql)  # camelCase / plural
    for junk in (
        "Easy questions",
        "**Moderate**",
        "Keep it basic.",
        "Define domain grounding.",
        "`tags_type`",
    ):
        assert not question_aligned(junk, sql)
    assert not question_aligned("Total refunded amount?", sql)  # shares nothing with the query


def test_clean_pair_counts_reasons():
    drops: Counter = Counter()
    assert clean_pair("How many?", "SELECT COUNT(*) FROM t", "sql", drops) == (
        "How many?",
        "SELECT COUNT(*) FROM t",
    )
    assert clean_pair("hi", "SELECT 1", "sql", drops) is None  # question too short
    assert clean_pair("How many rows?", "DELETE FROM t", "sql", drops) is None
    assert clean_pair("List the movies", "MATCH (m:M) RETURN m", "cypher", drops) is None
    assert clean_pair("List the movies", "MATCH (m:M) RETURN m", "cypher", drops, keep_whole=True)
    assert drops == Counter(bad_question=1, write_statement=1, returns_whole_node=1)


# ---- schema text ------------------------------------------------------------------------------


def test_ddl_statements_keeps_tables_and_views_only():
    ddl = """
    -- a comment
    CREATE TABLE IF NOT EXISTS "Emp Loyee" (
        id INTEGER PRIMARY KEY, -- inline comment
        name TEXT NOT NULL DEFAULT 'a;b'
    );
    CREATE INDEX idx ON emp(name);
    INSERT INTO emp VALUES (1, 'x');
    CREATE VIEW v AS SELECT id FROM emp;
    CREATE TABLE dept (id INT, label VARCHAR(20));
    CREATE TABLE dept (id INT);
    PRAGMA foreign_keys = ON;
    """
    stmts = ddl_statements(ddl)
    assert [s.name for s in stmts] == ["emp loyee", "v", "dept"]
    assert [s.kind for s in stmts] == ["TABLE", "VIEW", "TABLE"]
    assert stmts[0].text == (
        'CREATE TABLE IF NOT EXISTS "Emp Loyee" (id INTEGER PRIMARY KEY, '
        "name TEXT NOT NULL DEFAULT 'a;b');"
    )
    for s in stmts:
        assert "\n" not in s.text and s.text.endswith(";") and "--" not in s.text
    assert ddl_statements(None) == []


def test_used_tables_and_sqlite_problem():
    names = {"emp", "dept", "other"}
    assert used_tables('SELECT e.name FROM emp e JOIN "dept" d ON e.d = d.id', names) == {
        "emp",
        "dept",
    }
    assert used_tables("SELECT 'other' AS label FROM emp", names) == {"emp"}
    ddl = ["CREATE TABLE t (a INT, b TEXT);"]
    assert sqlite_problem(ddl, "SELECT a FROM t WHERE b = 'x'") is None
    assert sqlite_problem(ddl, "SELECT a FROM nope") == "sqlite_compile_error"
    assert sqlite_problem(ddl, "SELECT c FROM t") == "sqlite_compile_error"
    assert sqlite_problem(ddl, "SELECT a FROM t WHERE b ILIKE 'x'") == "sqlite_compile_error"
    mysql = ["CREATE TABLE t (a INT) ENGINE=InnoDB;"]
    assert sqlite_problem(mysql, "SELECT 1") == "ddl_not_sqlite"


NEO4J_A = """Node properties:
- **Movie**
  - `title`: STRING Example: "The Matrix"
  - `released`: INTEGER Min: 1975, Max: 2012
- **Person**
  - `name`: STRING Example: "Keanu Reeves"
Relationship properties:
- **ACTED_IN**
  - `roles: LIST` Min Size: 1, Max Size: 6
- **REVIEWED**
  - `rating`: INTEGER Min: 45, Max: 100
The relationships:
(:Person)-[:ACTED_IN]->(:Movie)
(:Person)-[:REVIEWED]->(:Movie)
(:Person)-[:FOLLOWS]->(:Person)"""

NEO4J_B = """Graph schema: Relevant node labels and their properties (with datatypes) are:
Article {abstract: STRING, year: INTEGER}
Keyword {}

Relevant relationships are:
{'start': Article, 'type': HAS_KEY, 'end': Keyword }

Relevant relationship properties (with datatypes) are:
HAS_KEY {weight: FLOAT}"""


def test_graph_schema_formats_render_compactly():
    a = render_graph_schema(parse_graph_schema(NEO4J_A))
    assert a == "\n".join(
        [
            "Nodes (properties are STRING unless typed):",
            "- Movie {title, released: INTEGER}",
            "- Person {name}",
            "Relationships:",
            "- (Person)-[:ACTED_IN {roles: LIST}]->(Movie)",
            "- (Person)-[:REVIEWED {rating: INTEGER}]->(Movie)",
            "- (Person)-[:FOLLOWS]->(Person)",
        ]
    )
    b = render_graph_schema(parse_graph_schema(NEO4J_B))
    assert b == "\n".join(
        [
            "Nodes (properties are STRING unless typed):",
            "- Article {abstract, year: INTEGER}",
            "- Keyword",
            "Relationships:",
            "- (Article)-[:HAS_KEY {weight: FLOAT}]->(Keyword)",
        ]
    )
    assert parse_graph_schema('{"Student": {"count": 3}}') is None
    assert parse_graph_schema(None) is None


NEO4J_INLINE = """Node properties:
Department {departmentNo: INTEGER, departmentName: STRING}
Customer {customerId: STRING, age: FLOAT}
Relationship properties:
PURCHASED {price: FLOAT}
The relationships:
(:Customer)-[:PURCHASED]->(:Department)"""

APOC_JSON = json.dumps(
    {
        "Student": {
            "count": 8,
            "labels": [],
            "type": "node",
            "properties": {
                "STU_NUM": {"type": "INTEGER", "unique": False},
                "STU_NAME": {"type": "STRING"},
            },
            "relationships": {
                "WORKS_IN": {
                    "count": 0,
                    "direction": "out",
                    "labels": ["Department"],
                    "properties": {},
                }
            },
        },
        "Department": {
            "count": 3,
            "labels": [],
            "type": "node",
            "properties": {"DEPT_CODE": {"type": "STRING"}},
            "relationships": {
                "WORKS_IN": {"count": 6, "direction": "in", "labels": ["Student"], "properties": {}}
            },
        },
        "WORKS_IN": {
            "count": 28,
            "type": "relationship",
            "properties": {"since": {"type": "DATE"}},
        },
    }
)

NEO4J_BARE_LABELS = """Graph schema: Relevant node labels and their properties  are:
Author
DOI

Relevant relationships are:
{'start': Author, 'type': HAS_DOI, 'end': DOI }"""


def test_graph_schema_more_formats():
    inline = render_graph_schema(parse_graph_schema(NEO4J_INLINE))
    assert "- Department {departmentNo: INTEGER, departmentName}" in inline
    assert "- (Customer)-[:PURCHASED {price: FLOAT}]->(Department)" in inline
    apoc = render_graph_schema(parse_graph_schema(APOC_JSON))
    assert apoc == "\n".join(
        [
            "Nodes (properties are STRING unless typed):",
            "- Student {STU_NUM: INTEGER, STU_NAME}",
            "- Department {DEPT_CODE}",
            "Relationships:",
            "- (Student)-[:WORKS_IN {since: DATE}]->(Department)",
        ]
    )
    bare = render_graph_schema(parse_graph_schema(NEO4J_BARE_LABELS))
    assert bare == "\n".join(
        [
            "Nodes (properties are STRING unless typed):",
            "- Author",
            "- DOI",
            "Relationships:",
            "- (Author)-[:HAS_DOI]->(DOI)",
        ]
    )


# ---- sampling helpers -------------------------------------------------------------------------


def _cand(i: int, db: str, q: str | None = None, query: str | None = None) -> Cand:
    return Cand(
        row_id=f"r{i}",
        db_id=db,
        pool=db,
        question=q or f"question number {i}",
        query=query or f"SELECT {i} FROM t",
    )


def test_dedupe_keeps_pair_once():
    drops: Counter = Counter()
    cands = [
        _cand(2, "a", "How many?", "SELECT COUNT(*) FROM t"),
        _cand(1, "a", "how  many?", "select count(*) from t"),
        _cand(3, "a", "How many?", "SELECT COUNT(1) FROM t"),
    ]
    kept = dedupe(cands, drops)
    assert [c.row_id for c in kept] == ["r1", "r3"] and drops["duplicate"] == 1


def test_round_robin_spreads_databases_and_is_deterministic():
    cands = [_cand(i, f"db{i % 4}") for i in range(40)]
    first = [c.row_id for c in round_robin(cands, stable_rng(0, "x"))]
    assert sorted(first) == sorted(c.row_id for c in cands)
    assert len({c.db_id for c in round_robin(cands, stable_rng(0, "x")) if True}) == 4
    head = [c.db_id for c in round_robin(cands, stable_rng(0, "x"))][:4]
    assert sorted(head) == ["db0", "db1", "db2", "db3"]  # one pair per database first
    assert first == [c.row_id for c in round_robin(cands, stable_rng(0, "x"))]
    assert first != [c.row_id for c in round_robin(cands, stable_rng(1, "x"))]


def test_split_by_db_holds_out_whole_databases():
    cands = [_cand(i, f"db{i % 50}") for i in range(500)]
    train, val = split_by_db(cands, 0.1, stable_rng(0, "h"))
    assert {c.db_id for c in train}.isdisjoint({c.db_id for c in val})
    assert len(train) + len(val) == 500 and len({c.db_id for c in val}) == 5
    one, held = split_by_db(cands[:3], 0.01, stable_rng(0, "h"))  # at least one database
    assert len({c.db_id for c in held}) == 1


def test_pick_examples_never_own_pair_and_about_half():
    pool = [_cand(i, "db") for i in range(30)]
    shown = 0
    for me in pool:
        rng = stable_rng(0, "pick", me.row_id)
        ex = pick_examples(rng, pool, me)
        assert me.row_id not in {e.row_id for e in ex} and len(ex) <= 2
        assert len({e.row_id for e in ex}) == len(ex)
        shown += bool(ex)
    assert 8 <= shown <= 22  # about half of 30
    assert pick_examples(stable_rng(0, "k"), [pool[0]], pool[0]) == []


# ---- raw-format fixtures ----------------------------------------------------------------------

SQALE_QUESTIONS = (
    "verbose",
    "evidence_supported",
    "structured",
    "requirements_list",
    "short_ambiguous",
    "short_high_level",
    "casual",
    "spelling_grammar_mistakes",
)


def _sqale_q(text: str, placeholder: bool = False) -> dict[str, str]:
    if placeholder:
        return dict.fromkeys(SQALE_QUESTIONS, "...")
    if text == "JUNK":  # generator notes where the questions should be
        return {
            "verbose": "**Moderate**",
            "evidence_supported": "Easy questions, evidence: none",
            "structured": "### Simple Questions",
            "requirements_list": "Define domain grounding.",
            "short_ambiguous": "?",
            "short_high_level": "Keep it basic.",
            "casual": "Create generated questions.",
            "spelling_grammar_mistakes": "`emp`",
        }
    return {
        "verbose": f"Please tell me {text} in full detail for the report.",
        "evidence_supported": f"Tell me {text}., evidence: the emp table lists people.",
        "structured": f"1. {text}",
        "requirements_list": f"{text}",
        "short_ambiguous": "stuff?",
        "short_high_level": f"Give me {text}.",
        "casual": f"Hey can you get {text} for me?",
        "spelling_grammar_mistakes": f"Plz giv me {text} thx.",
    }


COMMON_DDL = (
    "CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER, salary REAL);  "
    "CREATE TABLE dept (id INTEGER PRIMARY KEY, label TEXT);  "
    "CREATE TABLE audit_log (id INTEGER, note TEXT);  "
    "CREATE INDEX idx_emp ON emp (name);  "
    "INSERT INTO emp VALUES (1, 'a', 1, 2.0);"
)
HUGE_COLS = ", ".join(f"column_number_{i} INTEGER" for i in range(450))
HUGE_DDL = f"CREATE TABLE huge ({HUGE_COLS});  CREATE TABLE small (id INTEGER);"


def _sqale_rows(db: str, specs: list[tuple[str, str, list[str], bool]]) -> list[dict]:
    rows = []
    for i, (text, sql, relevant, placeholder) in enumerate(specs):
        rows.append(
            {
                "question_id": f"{db}_q{i}",
                "schema_id": db,
                "sql": sql,
                "difficulty": "simple",
                "questions": _sqale_q(text, placeholder),
                "relevant_tables": json.dumps(relevant),
                "number_of_relevant_tables": len(relevant),
                "execution_result": "[]",
            }
        )
    return rows


def make_sqale_raw(raw: Path) -> None:
    (raw / "data").mkdir(parents=True)
    ok = [
        (
            "the names of high earners",
            "SELECT name FROM emp WHERE salary > 100",
            ["emp", "dept"],
            False,
        ),
        (
            "the headcount per department label",
            "SELECT d.label, COUNT(*) AS n\n  FROM emp e JOIN dept d ON e.dept_id = d.id\n"
            "  GROUP BY d.label",
            ["emp", "dept", "audit_log"],
            False,
        ),
        ("the audit notes", "SELECT note FROM audit_log WHERE id = 3;", ["audit_log"], False),
        ("the average salary", "SELECT AVG(salary) AS avg_salary FROM emp", ["emp"], False),
        ("the department labels", "SELECT label FROM dept ORDER BY label", ["dept"], False),
        ("the names starting with A", "SELECT name FROM emp WHERE name LIKE 'A%'", ["emp"], False),
        ("the highest salary", "SELECT MAX(salary) FROM emp", ["emp"], False),
        ("the salary total", "SELECT SUM(salary) FROM emp", ["emp"], False),
    ]
    bad = [
        ("remove all emp rows", "DELETE FROM emp", ["emp"], False),
        ("the emp names and ids", "SELECT name FROM emp; SELECT id FROM emp", ["emp"], False),
        ("the emp names with a comment", "SELECT name FROM emp -- why", ["emp"], False),
        ("the nope of emp", "SELECT nope FROM emp", ["emp"], False),
        ("the missing table rows", "SELECT 1 AS x FROM missing_table", ["emp"], False),
        ("placeholder", "SELECT name FROM emp", ["emp"], True),
        ("JUNK", "SELECT name FROM emp", ["emp"], False),
    ]
    train = _sqale_rows("schema_000001", ok + bad) + _sqale_rows("schema_000002", ok[:5])
    train += _sqale_rows(
        "schema_000003",
        [("the huge table ids", "SELECT column_number_1 FROM huge", ["huge", "small"], False)],
    )
    # the first table alone is over the limit, so this is dropped; the small one fits
    train += _sqale_rows(
        "schema_000004",
        [
            ("the huge table ids", "SELECT column_number_1 FROM huge", ["huge", "small"], False),
            ("the small ids", "SELECT id FROM small", ["small"], False),
        ],
    )
    test = _sqale_rows("schema_900001", ok[:4]) + _sqale_rows("schema_900002", ok[4:7])
    for name, rows in (("train", train), ("test", test)):
        pq.write_table(pa.Table.from_pylist(rows), raw / "data" / f"{name}-00000-of-00001.parquet")
    ddl = [
        ("train", "schema_000001", COMMON_DDL),
        ("train", "schema_000002", COMMON_DDL),
        ("train", "schema_000003", HUGE_DDL),
        ("train", "schema_000004", HUGE_DDL),
        ("test", "schema_900001", COMMON_DDL),
        ("test", "schema_900002", COMMON_DDL),
    ]
    pq.write_table(
        pa.table(
            {
                "split": [d[0] for d in ddl],
                "schema_id": [d[1] for d in ddl],
                "full_schema": [d[2] for d in ddl],
                "number_of_tables": [3] * len(ddl),
            }
        ),
        raw / "schemas_ddl.parquet",
    )


def make_gretel_raw(raw: Path) -> None:
    raw.mkdir(parents=True)

    def row(i, prompt, ctx, sql, task="analytics and reporting"):
        return {
            "id": i,
            "domain": "forestry",
            "sql_complexity": "basic SQL",
            "sql_task_type": task,
            "sql_prompt": prompt,
            "sql_context": ctx,
            "sql": sql,
        }

    shared = (
        "CREATE TABLE trees (id INT, species VARCHAR(50), height FLOAT); "
        "INSERT INTO trees VALUES (1, 'oak', 2.5);"
    )
    train = [
        row(1, "What is the tallest tree?", shared, "SELECT MAX(height) FROM trees;"),
        row(2, "Count the oak trees", shared, "SELECT COUNT(*) FROM trees WHERE species = 'oak';"),
        row(3, "List every species", shared, "SELECT DISTINCT species FROM trees"),
        row(
            4,
            "Average height per species",
            shared,
            "SELECT species, AVG(height) AS h FROM trees GROUP BY species",
        ),
        row(
            5,
            "Add a tree",
            shared,
            "INSERT INTO trees VALUES (2, 'elm', 3.0);",
            "data manipulation",
        ),
        row(6, "Make a table", "", "CREATE TABLE x (a INT);", "data definition"),
        row(7, "Two statements", shared, "SELECT 1; SELECT 2;"),
        row(8, "Postgres only", shared, "SELECT species FROM trees WHERE species ILIKE 'o%'"),
        row(9, "A comment", shared, "SELECT species FROM trees -- all"),
        row(10, "Tall trees", shared, "SELECT species FROM trees WHERE height > 3"),
        row(
            11,
            "Another context",
            "CREATE TABLE cars (make TEXT, year INT);",
            "SELECT make FROM cars WHERE year > 2000",
        ),
        row(
            12,
            "Another context",
            "CREATE TABLE cars (make TEXT, year INT);",
            "SELECT make FROM cars WHERE year > 2000",
        ),
        row(
            13,
            "Too long context",
            "CREATE TABLE big (" + ", ".join(f"c{i} INT" for i in range(1500)) + ");",
            "SELECT c1 FROM big",
        ),
    ]
    test = [
        row(1, "Shortest tree?", shared, "SELECT MIN(height) FROM trees"),
        row(
            2,
            "Cars from 2010?",
            "CREATE TABLE autos (make TEXT, year INT);",
            "SELECT make FROM autos WHERE year = 2010",
        ),
    ]
    pq.write_table(pa.Table.from_pylist(train), raw / "synthetic_text_to_sql_train.snappy.parquet")
    pq.write_table(pa.Table.from_pylist(test), raw / "synthetic_text_to_sql_test.snappy.parquet")


def make_text2cypher_raw(raw: Path) -> None:
    (raw / "data").mkdir(parents=True)
    big = "Node properties:\n" + "".join(
        f"- **L{i}**\n"
        + "".join(f"  - `property_with_a_long_name_{j}`: STRING\n" for j in range(12))
        for i in range(60)
    )

    def row(i, q, schema, cy, alias="neo4jlabs_demo_db_movies"):
        return {
            "question": q,
            "schema": schema,
            "cypher": cy,
            "data_source": "synthetic",
            "instance_id": f"instance_id_{i}",
            "database_reference_alias": alias,
        }

    train = [
        row(
            1,
            "Who acted in The Matrix?",
            NEO4J_A,
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'})\\n"
            "RETURN p.name AS actor",
        ),
        row(
            2,
            "Which movies came out after 2000?",
            NEO4J_A,
            "MATCH (m:Movie) WHERE m.released > 2000 RETURN m.title AS title",
        ),
        row(3, "Show me all the movies", NEO4J_A, "MATCH (m:Movie) RETURN m"),
        row(4, "Delete the movies", NEO4J_A, "MATCH (m:Movie) DETACH DELETE m"),
        row(5, "Show the labels", NEO4J_A, "CALL db.labels() YIELD label RETURN label"),
        row(
            6,
            "What keywords does an article have?",
            NEO4J_B,
            "MATCH (a:Article)-[:HAS_KEY]->(k:Keyword) "
            "RETURN a.abstract AS abstract, k.name AS keyword",
            None,
        ),
        row(
            7, "Count the articles", NEO4J_B, "MATCH (a:Article) RETURN count(a) AS articles", None
        ),
        row(
            8, "Unparseable schema", '{"Student": {"count": 3}}', "MATCH (s:Student) RETURN s.name"
        ),
        row(
            9, "Huge schema question", big, "MATCH (n:L1) RETURN n.property_with_a_long_name_1 AS p"
        ),
        row(
            10,
            "Who directed it?",
            NEO4J_A,
            "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name AS actor, m.title AS movie",
        ),
    ]
    test = [
        row(
            101,
            "Which people follow someone?",
            NEO4J_A,
            "MATCH (a:Person)-[:FOLLOWS]->(b:Person) RETURN a.name AS follower, b.name AS followed",
        ),
        row(102, "Which movies are there?", NEO4J_A, "MATCH (m:Movie) RETURN m.title AS title"),
    ]
    pq.write_table(pa.Table.from_pylist(train), raw / "data" / "train.parquet")
    pq.write_table(pa.Table.from_pylist(test), raw / "data" / "test.parquet")


def make_cypherbench_raw(raw: Path) -> None:
    (raw / "graphs" / "schemas").mkdir(parents=True)

    def schema(name, label, rel):
        return {
            "name": name,
            "entities": [
                {
                    "label": label,
                    "description": None,
                    "properties": {"year": "int", "tags": "list[str]"},
                },
                {"label": "Place", "description": None, "properties": {}},
            ],
            "relations": [
                {
                    "label": rel,
                    "subj_label": label,
                    "obj_label": "Place",
                    "properties": {"since": "date"},
                }
            ],
        }

    for name, label, rel in (("g1", "Club", "basedIn"), ("g2", "Band", "formedIn")):
        (raw / "graphs" / "schemas" / f"{name}_schema.json").write_text(
            json.dumps(schema(name, label, rel)), encoding="utf-8"
        )

    def row(i, graph, q, cy):
        return {
            "qid": f"qid-{i}",
            "graph": graph,
            "nl_question": q,
            "gold_cypher": cy,
            "answer_json": "[]",
            "from_template": {"match_category": "basic", "return_pattern_id": "n_name"},
        }

    train = [
        row(
            1,
            "g1",
            "Which clubs are based in Paris?",
            "MATCH (n:Club)-[r0:basedIn]->(m0:Place {name: 'Paris'}) RETURN n.name",
        ),
        row(2, "g1", "How many clubs are there?", "MATCH (n:Club) RETURN count(n)"),
        row(
            3,
            "g1",
            "Which clubs started after 1990?",
            "MATCH (n:Club) WHERE n.year > 1990 RETURN n.name",
        ),
        row(
            4,
            "g1",
            "Union of two places",
            "CALL { MATCH (n:Club)-[:basedIn]->(m0:Place {name: 'A'}) RETURN n "
            "UNION MATCH (n:Club)-[:basedIn]->(m1:Place {name: 'B'}) RETURN n } "
            "WITH DISTINCT n RETURN n.name",
        ),
        row(5, "g1", "A broken one", "MATCH (n:Club RETURN n.name"),
    ]
    test = [
        row(
            101,
            "g2",
            "Which bands formed in Berlin?",
            "MATCH (n:Band)-[r0:formedIn]->(m0:Place {name: 'Berlin'}) RETURN n.name",
        ),
        row(102, "g2", "How many bands are there?", "MATCH (n:Band) RETURN count(n)"),
    ]
    (raw / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (raw / "test.json").write_text(json.dumps(test), encoding="utf-8")


MAKERS = {
    "sqale": make_sqale_raw,
    "gretel": make_gretel_raw,
    "text2cypher": make_text2cypher_raw,
    "cypherbench": make_cypherbench_raw,
}


@pytest.fixture
def converted(tmp_path_factory):
    """name -> (dir, stats): every adapter run once on its fixture."""
    out = {}
    for name, maker in MAKERS.items():
        base = tmp_path_factory.mktemp(name)
        maker(base / "raw")
        out[name] = (base, convert(name, out_dir=base))
    return out


def _load(base: Path, split: str) -> list[Example]:
    return list(read_jsonl(base / f"examples_{split}.jsonl"))


def _prompt_examples(ex: Example) -> str:
    sys_msg = ex.messages[0].content
    return sys_msg.split("Examples (tested queries on this database):\n", 1)[1].split(
        "\n\nRules:", 1
    )[0]


# ---- the adapters -----------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MAKERS))
def test_examples_are_valid_and_shaped(converted, name):
    base, stats = converted[name]
    dialect = "cypher" if name in ("text2cypher", "cypherbench") else "sqlite"
    for split in ("train", "val"):
        exs = _load(base, split)  # read_jsonl validates every record against the contract
        assert exs, f"{name} {split} is empty"
        assert stats[split]["examples"] == len(exs)
        for ex in exs:
            assert ex.origin == name and ex.task == "generate" and ex.group == "public"
            assert ex.split == split and ex.dialect == dialect
            assert [m.role for m in ex.messages] == ["system", "user", "assistant"]
            q = json.loads(ex.target)
            assert list(q) == ["query"] and q["query"] == ex.jepa.code
            assert ex.jepa.text in ex.messages[1].content
            assert ex.messages[1].content.startswith(
                "Source: graph (Neo4j Cypher)"
                if dialect == "cypher"
                else "Source: sql (SQLite SQL)"
            )
            assert ex.meta["license"] in ("MIT", "Apache-2.0") and ex.meta["row_id"]
            assert ex.meta["db_id"] and ex.meta["schema_chars"] <= public.SCHEMA_LIMIT
            assert "--" not in ex.jepa.code and ";" not in ex.jepa.code.rstrip(";")
    ids = [e.id for s in ("train", "val") for e in _load(base, s)]
    assert len(ids) == len(set(ids))
    assert (base / "stats.json").exists()


@pytest.mark.parametrize("name", list(MAKERS))
def test_prompt_examples_are_other_pairs_of_the_same_database(converted, name):
    base, _ = converted[name]
    for split in ("train", "val"):
        for ex in _load(base, split):
            block = _prompt_examples(ex)
            n = ex.meta["n_prompt_examples"]
            assert block.count("\nQ: ") + block.startswith("Q: ") == n if n else block == "(none)"
            assert ex.jepa.code not in block  # never its own pair
            assert f'Q: "{ex.jepa.text}"' not in block
            sys_msg = ex.messages[0].content
            assert sys_msg == system_prompt(
                sys_msg.split("\n\n", 1)[1].split("\n\nExamples (tested", 1)[0],
                _parse_examples(block),
            )


def _parse_examples(block: str) -> list[tuple[str, str]]:
    if block == "(none)":
        return []
    return [
        (m.group(1), m.group(2).strip())
        for m in re.finditer(r'Q: "(.*?)"\n(.*?)(?=\n\nQ: "|\Z)', block, re.S)
    ]


def test_sqale(converted):
    base, stats = converted["sqale"]
    train, val = _load(base, "train"), _load(base, "val")
    d = stats["train"]["dropped"]
    # clean stage
    assert d["write_statement"] == 1  # DELETE FROM emp
    assert (
        d["multi_statement"] == 1 and d["comment"] == 1 and d["bad_question"] == 2
    )  # the placeholders; the generator notes
    # build stage
    assert d["sqlite_compile_error"] == 1  # SELECT nope FROM emp
    assert d["no_known_table"] == 1  # FROM missing_table
    assert d["schema_too_long"] == 2  # the huge table alone is over the limit (two databases)
    assert {e.meta["db_id"] for e in train} == {"schema_000001", "schema_000002", "schema_000004"}
    # the schema is the CREATE TABLEs only: no INSERT, no index
    for ex in train + val:
        schema = ex.messages[0].content.split("\n\n", 1)[1].split("\n\nExamples (tested", 1)[0]
        assert "INSERT" not in schema and "CREATE INDEX" not in schema
        assert all(line.startswith("CREATE TABLE") for line in schema.splitlines())
    # trimmed to the tables the query uses plus the question's own sub-schema
    by_row = {e.meta["row_id"]: e for e in train}
    audit = by_row["schema_000001_q2"]  # SELECT note FROM audit_log, relevant [audit_log]
    schema = audit.messages[0].content
    assert "CREATE TABLE audit_log" in schema and "CREATE TABLE emp" not in schema
    join = by_row["schema_000001_q1"]  # emp + dept + audit_log
    assert "CREATE TABLE dept" in join.messages[0].content
    # the SQL was collapsed to one line, and one phrasing was drawn from the allowed set
    assert join.jepa.code == (
        "SELECT d.label, COUNT(*) AS n FROM emp e JOIN dept d ON e.dept_id = d.id GROUP BY d.label"
    )
    for ex in train:
        assert ex.meta["phrasing"] in {p for p, _ in public.Sqale.PHRASINGS}
        assert "evidence" not in ex.jepa.text and ex.jepa.text != "..."
        assert ex.jepa.text not in ("stuff?",)  # short_ambiguous is never drawn
    # whole databases: val schemas are not train schemas
    assert {e.meta["db_id"] for e in val}.isdisjoint({e.meta["db_id"] for e in train})
    assert stats["val"]["databases_also_in_train"] == 0


def test_gretel(converted):
    base, stats = converted["gretel"]
    train = _load(base, "train")
    d = stats["train"]["dropped"]
    assert d["write_statement"] == 1  # the INSERT task
    assert d["schema_empty"] == 1  # the data-definition row has no CREATE TABLE context
    assert d["multi_statement"] == 1 and d["comment"] == 1
    assert d["sqlite_compile_error"] == 1  # ILIKE
    assert d["duplicate"] == 1
    assert d["schema_too_long"] == 1
    kept = {e.meta["row_id"] for e in train}
    assert kept == {"1", "2", "3", "4", "10", "11"}
    ex = next(e for e in train if e.meta["row_id"] == "1")
    assert (
        ex.messages[0].content.count(
            "CREATE TABLE trees (id INT, species VARCHAR(50), height FLOAT);"
        )
        == 1
    )
    assert "INSERT" not in ex.messages[0].content.split("Rules:")[0]
    assert ex.jepa.code == "SELECT MAX(height) FROM trees"  # trailing ; removed
    # rows that share a schema can show each other; the one-off context has no partner
    assert next(e for e in train if e.meta["row_id"] == "11").meta["n_prompt_examples"] == 0


def test_text2cypher(converted):
    base, stats = converted["text2cypher"]
    train = _load(base, "train")
    d = stats["train"]["dropped"]
    assert d["schema_unparseable"] == 1
    assert d["returns_whole_node"] == 1 and d["procedure_call"] == 1
    assert d["write_statement"] == 1
    assert d["schema_too_long"] == 1
    kept = {e.meta["row_id"] for e in train}
    assert kept == {f"instance_id_{i}" for i in (1, 2, 6, 7, 10)}
    ex = next(e for e in train if e.meta["row_id"] == "instance_id_1")
    assert (
        ex.jepa.code
        == "MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'The Matrix'}) RETURN p.name AS actor"
    )
    assert "- (Person)-[:ACTED_IN {roles: LIST}]->(Movie)" in ex.messages[0].content
    # an alias-less graph gets a schema-hash id
    assert (
        next(e for e in train if e.meta["row_id"] == "instance_id_6")
        .meta["db_id"]
        .startswith("schema-")
    )
    # the dataset's test split is the val split (it shares the training graphs, so say so)
    assert stats["val"]["databases_also_in_train"] == 1


def test_text2cypher_keep_whole_nodes(tmp_path):
    make_text2cypher_raw(tmp_path / "raw")
    stats = convert("text2cypher", out_dir=tmp_path, keep_whole=True)
    assert "returns_whole_node" not in stats["train"]["dropped"]
    assert any(e.jepa.code == "MATCH (m:Movie) RETURN m" for e in _load(tmp_path, "train"))


def test_cypherbench(converted):
    base, stats = converted["cypherbench"]
    train, val = _load(base, "train"), _load(base, "val")
    assert stats["train"]["dropped"] == {"unbalanced": 1}
    assert len(train) == 4 and len(val) == 2
    ex = train[0]
    schema = ex.messages[0].content
    assert "- Club {name, year: INTEGER, tags: LIST}" in schema  # the implicit name is listed first
    assert "- Place {name}" in schema
    assert "- (Club)-[:basedIn {since: DATE}]->(Place)" in schema
    # CALL { ... UNION ... } subqueries are read-only and kept
    assert any(e.jepa.code.startswith("CALL {") for e in train)
    # the dataset's test graphs are not its train graphs
    assert {e.meta["db_id"] for e in val}.isdisjoint({e.meta["db_id"] for e in train})
    assert stats["val"]["databases_also_in_train"] == 0


@pytest.mark.parametrize("name", list(MAKERS))
def test_deterministic(tmp_path, name):
    MAKERS[name](tmp_path / "raw")
    convert(name, seed=3, out_dir=tmp_path)
    first = {f: (tmp_path / f).read_bytes() for f in ("examples_train.jsonl", "examples_val.jsonl")}
    convert(name, seed=3, out_dir=tmp_path)
    assert first == {f: (tmp_path / f).read_bytes() for f in first}


def test_seed_changes_choices(tmp_path):
    make_sqale_raw(tmp_path / "raw")
    convert("sqale", seed=0, out_dir=tmp_path)
    a = (tmp_path / "examples_train.jsonl").read_bytes()
    convert("sqale", seed=1, out_dir=tmp_path)
    assert a != (tmp_path / "examples_train.jsonl").read_bytes()


def test_val_by_db_holds_out_whole_databases(tmp_path):
    make_sqale_raw(tmp_path / "raw")
    stats = convert("sqale", seed=0, val_by_db=True, out_dir=tmp_path)
    train, val = _load(tmp_path, "train"), _load(tmp_path, "val")
    assert val and train
    assert {e.meta["db_id"] for e in val}.isdisjoint({e.meta["db_id"] for e in train})
    assert stats["val"]["databases_also_in_train"] == 0
    assert "held-out" in stats["val_source"]


def test_stats_json(converted):
    base, stats = converted["gretel"]
    on_disk = json.loads((base / "stats.json").read_text(encoding="utf-8"))
    assert on_disk == stats
    t = on_disk["train"]
    assert {"raw_rows", "dropped", "examples", "examples_in_prompt_share"} <= set(t)
    for key in ("schema_chars", "question_chars", "query_chars", "prompt_chars"):
        assert set(t[key]) == {"min", "p50", "p90", "p99", "max"}
    assert 0.0 <= t["examples_in_prompt_share"] <= 1.0
    assert t["schema_chars"]["max"] <= public.SCHEMA_LIMIT
    assert t["schema_chars"]["max"] == max(e.meta["schema_chars"] for e in _load(base, "train"))
