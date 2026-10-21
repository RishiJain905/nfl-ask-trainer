"""The tool-call prompt: a public example rendered for Spark's native tool calling.

The model answers by calling a query tool, thinking first, instead of replying with a JSON
object (`public_prompt`). One example becomes:

- `tools`: one query tool, its name and argument varied across examples (`run_query(query)`,
  `execute_sql(sql)`, `cypher_query(cypher)`, ...) so the model reads the tool definition
  instead of memorising one name, plus, in some prompts, an unrelated tool it must ignore;
- `system`: what the tool is for, the schema, up to two tested examples, the rules;
- `user`: the question, as a person typed it;
- the reply (built later, from a checked trace): `reasoning_content` and one tool call.

Standard library only: the trace generator imports it on Windows next to llama-server.
`from_example_dict` takes an `Example` already dumped to a dict (the converted public records).
"""

from __future__ import annotations

import random
import re
from typing import Any

LANGUAGE = {"sqlite": "SQLite SQL", "duckdb": "DuckDB SQL", "cypher": "Neo4j Cypher"}

# (name, argument, description); {lang} is filled with LANGUAGE[dialect]
SQL_TOOLS = [
    (
        "run_query",
        "query",
        "Run one read-only {lang} query against the database and return the result rows.",
    ),
    (
        "execute_sql",
        "sql",
        "Execute a single read-only {lang} statement on the database and return the rows.",
    ),
    (
        "sql_query",
        "query",
        "Query the database with one read-only {lang} SELECT and get the result table back.",
    ),
    (
        "query_database",
        "query",
        "Send one read-only {lang} query to the database; returns the matching rows.",
    ),
]
CYPHER_TOOLS = [
    (
        "run_query",
        "query",
        "Run one read-only {lang} query against the graph database and return the result rows.",
    ),
    (
        "run_cypher",
        "query",
        "Execute a single read-only {lang} query on the graph and return the records.",
    ),
    (
        "cypher_query",
        "cypher",
        "Query the graph with one read-only {lang} statement; returns the result records.",
    ),
    (
        "query_graph",
        "cypher",
        "Send one read-only {lang} query to the Neo4j database and get the rows back.",
    ),
]
DISTRACTORS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time in a timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA timezone, e.g. Europe/Paris."}
            },
            "required": ["timezone"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email to a recipient.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
]
DISTRACTOR_SHARE = 0.25

SYSTEM = """You answer questions about the {what} below by calling the `{tool}` tool with ONE read-only {lang} query.

{schema}

Examples (tested queries on this database):
{examples}

Rules:
- One statement, read only ({read}). No comments. The query runs as written, so use literal values, not parameters.
- Return named columns: the names and the numbers the answer needs. Never return whole {whole} unless the question asks for them.
- Match names and values as the data spells them.
- Return exactly the rows the question asks for, in its shape: one row for a single answer ("which is the most", "the highest": ORDER BY the measure, LIMIT 1); one row per entity for "which ...", "every", "list"; the top N for "top N".
- If the database can't answer the question, don't call the tool: say why in one short sentence."""

# the old JSON-reply system prompt (public_prompt.SYSTEM) these parts are read back from
OLD_SCHEMA = re.compile(
    r"^.*?\n\n(?P<schema>.*?)\n\nExamples \(tested queries on this database\):\n"
    r"(?P<examples>.*?)\n\nRules:\n",
    re.DOTALL,
)
OLD_EXAMPLE = re.compile(r'Q: "(?P<q>.*?)"\n(?P<query>.*?)(?=\n\nQ: "|\Z)', re.DOTALL)


def tool_def(name: str, arg: str, description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {arg: {"type": "string", "description": "The query text."}},
                "required": [arg],
            },
        },
    }


def split_old_system(system: str) -> tuple[str, list[tuple[str, str]]]:
    """(schema text, [(question, query), ...]) from a converted example's JSON-reply prompt."""
    m = OLD_SCHEMA.match(system)
    if not m:
        raise ValueError("not a public_prompt system message")
    ex = m.group("examples").strip()
    pairs = (
        []
        if ex == "(none)"
        else [(e.group("q"), e.group("query").strip()) for e in OLD_EXAMPLE.finditer(ex)]
    )
    return m.group("schema").strip(), pairs


def build(
    schema: str, examples: list[tuple[str, str]], question: str, dialect: str, rnd: random.Random
) -> dict[str, Any]:
    """The prompt for one question: {system, user, tools, tool: {name, arg}, distractor}."""
    lang = LANGUAGE[dialect]
    name, arg, desc = rnd.choice(CYPHER_TOOLS if dialect == "cypher" else SQL_TOOLS)
    tools = [tool_def(name, arg, desc.format(lang=lang))]
    distractor = None
    if rnd.random() < DISTRACTOR_SHARE:
        d = rnd.choice(DISTRACTORS)
        distractor = d["name"]
        tools.insert(rnd.randint(0, 1), {"type": "function", "function": d})
    ex = "\n\n".join(f'Q: "{q}"\n{query.strip()}' for q, query in examples) or "(none)"
    cypher = dialect == "cypher"
    system = SYSTEM.format(
        what="graph database" if cypher else "database",
        tool=name,
        lang=lang,
        schema=schema.strip(),
        examples=ex,
        read="MATCH ... RETURN" if cypher else "SELECT or WITH ... SELECT",
        whole="nodes" if cypher else "rows of every column",
    )
    return {
        "system": system,
        "user": " ".join(question.split()),
        "tools": tools,
        "tool": {"name": name, "arg": arg},
        "distractor": distractor,
    }


def from_example_dict(ex: dict[str, Any], rnd: random.Random) -> dict[str, Any]:
    """The tool-call prompt for a converted public example (an `Example` as a dict)."""
    system = next(m["content"] for m in ex["messages"] if m["role"] == "system")
    schema, pairs = split_old_system(system)
    return build(schema, pairs, ex["jepa"]["text"], ex["dialect"], rnd)


def assistant_turn(reasoning: str | None, tool: dict[str, str], query: str) -> dict[str, Any]:
    """The training target in the chat-template shape Spark's template renders:
    `<think>reasoning</think><tool_call>name<arg_key>arg</arg_key><arg_value>query</arg_value></tool_call>`."""
    turn: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": tool["name"], "arguments": {tool["arg"]: query}},
            }
        ],
    }
    if reasoning:
        turn["reasoning_content"] = reasoning
    return turn
