"""Smoke-test llama-server with Spark-X2.6-4B: thinking levels and Spark's tool-call format."""

import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8")
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run one read-only SQL query on the database and return the rows.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "One SELECT statement."}},
                "required": ["query"],
            },
        },
    }
]
SYSTEM = (
    "You answer questions about the user's database by calling run_query.\n"
    "Database schema (SQLite):\n"
    "CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT, conference TEXT);\n"
    "CREATE TABLE games (game_id INTEGER PRIMARY KEY, season INTEGER, "
    "home_id INTEGER REFERENCES teams(team_id), "
    "away_id INTEGER REFERENCES teams(team_id), home_score INTEGER, away_score INTEGER);"
)
QUESTION = "Which team won the most home games in the 2023 season, and how many?"
MESSAGES = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": QUESTION}]


def post(path, body):
    req = urllib.request.Request(
        URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def case(name, extra):
    body = {
        "messages": MESSAGES,
        "tools": TOOLS,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 4096,
        **extra,
    }
    out, secs = post("/v1/chat/completions", body)
    msg = out["choices"][0]["message"]
    usage = out.get("usage", {})
    print(
        f"\n===== {name}: {secs:.1f} s, {usage.get('completion_tokens')} completion tokens, "
        f"finish={out['choices'][0].get('finish_reason')}"
    )
    rc = msg.get("reasoning_content") or ""
    print(f"reasoning_content: {len(rc)} chars | {rc[:300]!r}{' ...' if len(rc) > 300 else ''}")
    print(f"content: {(msg.get('content') or '')[:400]!r}")
    print(f"tool_calls: {json.dumps(msg.get('tool_calls'), ensure_ascii=False)}")
    return msg


case("thinking on, effort high (template default)", {})
case("thinking on, effort low", {"chat_template_kwargs": {"reasoning_effort": "low"}})
case("thinking off", {"chat_template_kwargs": {"enable_thinking": False}})

# The raw text the model produced, to compare with what the parser extracted.
prompt = post("/apply-template", {"messages": MESSAGES, "tools": TOOLS})[0]["prompt"]
raw, secs = post(
    "/completion", {"prompt": prompt, "n_predict": 4096, "temperature": 0.6, "top_p": 0.95}
)
print(f"\n===== raw /completion ({secs:.1f} s): prompt ends {prompt[-40:]!r}")
print(f"tail of raw output: {raw['content'][-400:]!r}")
