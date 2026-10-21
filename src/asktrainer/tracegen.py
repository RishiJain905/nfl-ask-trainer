"""Generating thinking traces against llama-server, checking every answer as it arrives.

    python -m asktrainer.tracegen --run D:\\nfl-ml-data\\ask\\traces\\tsmoke-1
        [--url http://127.0.0.1:8090] [--workers 16]

Runs on Windows next to llama-server with the standard library only (set PYTHONPATH to the
repo's `src`). It reads the run folder `traceprep` wrote and sends every planned request (each
question x each thinking level x its tries) to `/v1/chat/completions` with the question's tools,
`workers` at a time. Each answer is checked at once (`querycheck`) and appended to
`results.jsonl`; `progress.jsonl` gets a line every 10 s and `events.jsonl` the start, errors
and the end. A restarted run skips the requests already in `results.jsonl`.

A result line: the request (qid, origin, level, try), the thought (`reasoning`, its token count
from the server's own tokenizer), the tool call (name, argument, query) or why there is none,
the check (verdict pass | fail | unverified, checked_by, reason), tokens and seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asktrainer.querycheck import calls_from_message, check_cypher, check_sql


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Run:
    def __init__(self, run_dir: Path, url: str):
        self.dir, self.url = run_dir, url.rstrip("/")
        self.cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        with (run_dir / "questions.jsonl").open(encoding="utf-8") as f:
            self.questions = [json.loads(line) for line in f if line.strip()]
        self.lock = threading.Lock()
        self.done: set[str] = set()
        self.totals = {"done": 0, "completion_tokens": 0, "prompt_tokens": 0, "errors": 0}
        rp = run_dir / "results.jsonl"
        if rp.exists():  # a restart carries the totals on, so the progress lines stay monotonic
            with rp.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        self.done.add(r["rid"])
                        self.totals["completion_tokens"] += r.get("completion_tokens") or 0
                        self.totals["prompt_tokens"] += r.get("prompt_tokens") or 0
        self.totals["done"] = len(self.done)
        self.stop = threading.Event()

    def requests(self) -> list[tuple[str, dict[str, Any], str, int]]:
        out = []
        plan = self.cfg["plan"]
        for q in self.questions:
            for level, n in plan.items():
                for i in range(n):
                    rid = f"{q['qid']}#{level}#{i}"
                    if rid not in self.done:
                        out.append((rid, q, level, i))
        return out

    def _post(self, path: str, body: dict[str, Any], timeout: float = 900) -> dict[str, Any]:
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def event(self, kind: str, **kw: Any) -> None:
        with self.lock, (self.dir / "events.jsonl").open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"ts": _now(), "kind": kind, **kw}) + "\n")

    def one(self, rid: str, q: dict[str, Any], level: str, i: int) -> dict[str, Any]:
        s = self.cfg["sampling"]
        kwargs: dict[str, Any] = {"enable_thinking": level != "off"}
        if level != "off":
            kwargs["reasoning_effort"] = level
        body = {
            "messages": [
                {"role": "system", "content": q["system"]},
                {"role": "user", "content": q["user"]},
            ],
            "tools": q["tools"],
            "temperature": s["temperature"],
            "top_p": s["top_p"],
            "max_tokens": s["max_tokens"],
            "chat_template_kwargs": kwargs,
        }
        t0 = time.time()
        resp = self._post("/v1/chat/completions", body)
        secs = time.time() - t0
        ch = resp["choices"][0]
        msg, usage = ch.get("message") or {}, resp.get("usage") or {}
        reasoning = msg.get("reasoning_content") or ""
        think_tokens = 0
        if reasoning:
            think_tokens = len(
                self._post("/tokenize", {"content": reasoning}, timeout=60).get("tokens") or []
            )
        calls = calls_from_message(msg)
        res: dict[str, Any] = {
            "rid": rid,
            "qid": q["qid"],
            "origin": q["origin"],
            "dialect": q["dialect"],
            "level": level,
            "try": i,
            "ts": _now(),
            "secs": round(secs, 2),
            "finish": ch.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "think_tokens": think_tokens,
            "reasoning": reasoning,
            "content": msg.get("content") or "",
            "tool": q["tool"]["name"],
            "distractor": q.get("distractor"),
            "calls": len(calls),
            "query": None,
        }
        tool, arg = q["tool"]["name"], q["tool"]["arg"]
        mine = [c for c in calls if c[0] == tool]
        if not calls:
            res.update(
                verdict="fail",
                checked_by="",
                reason="truncated" if ch.get("finish_reason") == "length" else "no_call",
            )
        elif not mine:
            res.update(verdict="fail", checked_by="", reason="wrong_tool")
        elif len(calls) > 1:
            res.update(verdict="fail", checked_by="", reason="several_calls")
        elif not isinstance(mine[0][1].get(arg), str) or not mine[0][1][arg].strip():
            res.update(verdict="fail", checked_by="", reason="bad_argument")
        else:
            query = mine[0][1][arg]
            res["query"] = query
            try:
                if q["dialect"] == "cypher":
                    res.update(check_cypher(query, q["gold"]))
                else:
                    res.update(
                        check_sql(
                            query, q["gold"], q.get("schema_script"), bool(q.get("real_rows"))
                        )
                    )
            except Exception as e:  # a checker bug must not lose the generated answer
                res.update(
                    verdict="unverified", checked_by="", reason=f"check_error:{type(e).__name__}"
                )
        return res

    def progress_loop(self) -> None:
        while not self.stop.wait(10):
            self.write_progress()

    def write_progress(self) -> None:
        with (
            self.lock,
            (self.dir / "progress.jsonl").open("a", encoding="utf-8", newline="\n") as f,
        ):
            f.write(json.dumps({"ts": _now(), **self.totals}) + "\n")

    def go(self, workers: int) -> int:
        todo = self.requests()
        total = len(self.questions) * sum(self.cfg["plan"].values())
        self.event(
            "start",
            url=self.url,
            workers=workers,
            todo=len(todo),
            total=total,
            resumed=len(self.done),
        )
        self.write_progress()
        threading.Thread(target=self.progress_loop, daemon=True).start()
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(self.one, *r): r[0] for r in todo}
            for fut in as_completed(futs):
                rid = futs[fut]
                try:
                    res = fut.result()
                except Exception as e:  # the server: HTTP errors, timeouts, a malformed reply
                    failures += 1
                    with self.lock:
                        self.totals["errors"] += 1
                    self.event(
                        "request_failed", rid=rid, error=f"{type(e).__name__}: {str(e)[:160]}"
                    )
                    if failures >= 20 and failures > self.totals["done"] // 4:
                        self.event("aborted", reason="too many failed requests")
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    continue
                with self.lock:
                    with (self.dir / "results.jsonl").open(
                        "a", encoding="utf-8", newline="\n"
                    ) as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    self.totals["done"] += 1
                    self.totals["completion_tokens"] += res.get("completion_tokens") or 0
                    self.totals["prompt_tokens"] += res.get("prompt_tokens") or 0
        self.stop.set()
        self.write_progress()
        state = "aborted" if failures >= 20 and failures > self.totals["done"] // 4 else "done"
        self.event(
            "DONE" if state == "done" else "stopped",
            done=self.totals["done"],
            errors=self.totals["errors"],
        )
        return 0 if state == "done" else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="the run folder traceprep wrote")
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args(argv)
    return Run(Path(a.run), a.url).go(a.workers)


if __name__ == "__main__":
    sys.exit(main())
