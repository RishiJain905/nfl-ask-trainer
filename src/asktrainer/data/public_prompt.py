"""The prompt public text-to-query examples are rendered into.

It mirrors the NFL repo's generator prompt (`src/nflengine/ask/prompts/generate.md` and
`route.AskEngine._generate_prompt`) line for line where the rules are general, so the behaviour
learnt on thousands of public schemas (read the schema in front of you, write one read-only
query, use literal values, return named columns in the question's shape, reply with one JSON
object) is the behaviour Ask asks for. NFL-only rules (playoffs, the Names list's ids) are left
out; the reply shape and the user message layout are the same.
"""

from __future__ import annotations

import json

LANGUAGE = {"sqlite": "SQLite SQL", "duckdb": "DuckDB SQL", "cypher": "Neo4j Cypher"}

SYSTEM = """You write ONE read-only query that answers a question about the database below. The query runs as written, so use literal values, not parameters.

{schema}

Examples (tested queries on this database):
{examples}

Rules:
- Exactly one statement; read only (SELECT for SQL; MATCH ... RETURN for Cypher). No comments.
- Return named columns: the names and the numbers the answer needs. Never return whole nodes or rows of every column unless the question asks for them.
- Match names and values as the data spells them.
- Return exactly the rows the question asks for, in its shape: one row for a single answer ("which is the most", "the highest": ORDER BY the measure, LIMIT 1); one row per entity for "which ...", "every", "list"; the top N for "top N".
- If the data can't answer it, reply {{"refuse": "no_data", "explain": "<one short sentence>"}} instead.

Reply with exactly ONE JSON object: {{"query": "<the query>"}}"""


def system_prompt(schema: str, examples: list[tuple[str, str]] | None = None) -> str:
    """The system prompt for one public example: its database's schema text and up to a few
    other (question, query) pairs from the same database as examples."""
    ex = "\n\n".join(f'Q: "{q}"\n{query.strip()}' for q, query in (examples or [])) or "(none)"
    return SYSTEM.format(schema=schema.strip(), examples=ex)


def user_message(question: str, dialect: str) -> str:
    source = "graph" if dialect == "cypher" else "sql"
    return (
        f"Source: {source} ({LANGUAGE[dialect]})\n"
        f"Question: {' '.join(question.split())}\nNames:\n(none found)"
    )


def target(query: str) -> str:
    """The assistant turn: `{"query": "<query>"}` (one line, the query's whitespace kept)."""
    return json.dumps({"query": query.strip()}, ensure_ascii=False)
