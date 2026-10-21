"""The scored evaluation of a trained adapter through llama.cpp: the held-out `evalset`
questions (traceexport.evalset) answered by llama-server running the Q8_0 base with the run's
LoRA converted to GGUF, then scored like the trainer's own evaluation (`publiceval`: SQL
executed on real or fuzzed rows, Cypher on CypherBench's graphs or schema-fuzzed ones).

    python -m asktrainer.llamaeval generate --examples <evalset jsonl> --out <preds jsonl>
        [--url http://127.0.0.1:8092] [--workers 16] [--max-tokens 2048]
    ask-trainer evalrun --run <run> --preds <preds jsonl> --tag <tag> [--step N]   (WSL)

Why llama.cpp and not the trainer's own generation: transformers generates ~24 tokens/s on
this card even in batches of 8, so 200 thinking answers take ~50 minutes; llama-server answers
16 at a time at ~37 answers a minute (~6 minutes), and it is the setup the model is served and
RL-trained with. Greedy decoding (temperature 0), the example's thinking level, and the
server's thinking budget (scripts/llamacpp/serve.ps1 -Extra "--reasoning-budget 1024").

`generate` uses only the standard library (it runs on Windows next to llama-server). A
prediction line: {"id", "output", "finish", "secs", "completion_tokens"}; `output` is the
thought, "</think>", then the call in Spark's text form (what `publiceval.parse_call` reads).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _post(url: str, body: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def as_text(msg: dict[str, Any]) -> str:
    """The reply as Spark's own text: thought, `</think>`, then each call as
    `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`."""
    out = msg.get("reasoning_content") or ""
    out += "</think>" + (msg.get("content") or "")
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        out += "<tool_call>" + (fn.get("name") or "")
        for k, v in (args or {}).items():
            out += f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>"
        out += "</tool_call>"
    return out


def request_body(ex: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    level = ex.get("thinking") or "high"
    kwargs: dict[str, Any] = {"enable_thinking": level != "off"}
    if level != "off":
        kwargs["reasoning_effort"] = level
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in ex["messages"][:2]]
    return {
        "messages": msgs,
        "tools": ex["tools"],
        "temperature": 0.0,
        "top_k": 1,
        "max_tokens": max_tokens,
        "chat_template_kwargs": kwargs,
    }


def generate(
    examples: Path, out: Path, url: str, workers: int = 16, max_tokens: int = 2048
) -> dict[str, Any]:
    with examples.open(encoding="utf-8") as f:
        exs = [json.loads(line) for line in f if line.strip()]
    done: set[str] = set()
    if out.exists():  # a restart keeps what was answered
        with out.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [e for e in exs if e["id"] not in done]
    base = url.rstrip("/")
    t0 = time.time()

    def one(ex: dict[str, Any]) -> dict[str, Any]:
        t = time.time()
        try:
            r = _post(base + "/v1/chat/completions", request_body(ex, max_tokens))
        except Exception as e:  # the server: an empty answer scores as no call
            return {"id": ex["id"], "output": "", "error": f"{type(e).__name__}: {str(e)[:160]}"}
        ch = r["choices"][0]
        return {
            "id": ex["id"],
            "output": as_text(ch.get("message") or {}),
            "finish": ch.get("finish_reason"),
            "secs": round(time.time() - t, 2),
            "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
        }

    errors = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        out.open("a", encoding="utf-8", newline="\n") as f,
    ):
        for res in pool.map(one, todo):
            errors += "error" in res
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
    return {
        "examples": len(exs),
        "answered": len(todo),
        "resumed": len(done),
        "errors": errors,
        "seconds": round(time.time() - t0, 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--examples", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--url", default="http://127.0.0.1:8092")
    g.add_argument("--workers", type=int, default=16)
    g.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args(argv)
    res = generate(Path(a.examples), Path(a.out), a.url, a.workers, a.max_tokens)
    print(json.dumps(res))
    return 0 if not res["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
