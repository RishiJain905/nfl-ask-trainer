"""The trace pipeline on the CPU: the query checker, the tool-call prompt, the generator against a
fake llama-server, and the dashboard feed."""

from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from asktrainer import querycheck as qc
from asktrainer import tracegen
from asktrainer.data import public_prompt, tool_prompt

SCHEMA = """CREATE TABLE teams (team_id INTEGER PRIMARY KEY, name TEXT, conference TEXT);
CREATE TABLE games (game_id INTEGER PRIMARY KEY, season INTEGER,
  home_id INTEGER REFERENCES teams(team_id),
  away_id INTEGER REFERENCES teams(team_id), home_score INTEGER, away_score INTEGER);"""
GOLD = (
    "SELECT t.name, COUNT(*) AS wins FROM games g JOIN teams t ON t.team_id = g.home_id "
    "WHERE g.home_score > g.away_score GROUP BY t.name ORDER BY wins DESC LIMIT 1"
)


# ---- querycheck -------------------------------------------------------------------------------


def test_gold_passes_itself_by_fuzzing():
    r = qc.check_sql(GOLD, GOLD, SCHEMA, real_rows=False)
    assert r["verdict"] == "pass" and r["checked_by"] == "fuzz" and r["fuzz_informative"] >= 1


def test_equivalent_rewrite_passes():
    alt = (
        "SELECT name, wins FROM (SELECT t.name AS name, COUNT(*) AS wins FROM teams t JOIN games g "
        "ON g.home_id = t.team_id WHERE g.away_score < g.home_score GROUP BY t.name) "
        "ORDER BY wins DESC LIMIT 1"
    )
    assert qc.check_sql(alt, GOLD, SCHEMA, real_rows=False)["verdict"] == "pass"


@pytest.mark.parametrize(
    "bad, reason",
    [
        (
            GOLD.replace("g.home_score > g.away_score", "g.home_score < g.away_score"),
            "fuzz_rows_differ",
        ),
        (GOLD.replace("t.name,", "t.nam,"), "parse"),
        ("DELETE FROM games", "not_read_only"),
        (GOLD + "; SELECT 1", "multiple_statements"),
    ],
)
def test_wrong_queries_fail(bad, reason):
    r = qc.check_sql(bad, GOLD, SCHEMA, real_rows=False)
    assert r["verdict"] == "fail" and r["reason"].startswith(reason)


def test_sort_order_matters_when_the_gold_sorts():
    gold = "SELECT name FROM teams ORDER BY team_id DESC"
    r = qc.check_sql("SELECT name FROM teams ORDER BY team_id ASC", gold, SCHEMA, real_rows=False)
    assert r["verdict"] == "fail" and r["reason"] == "order_differs"
    assert qc.sorts_result(gold) and not qc.sorts_result(
        "SELECT a FROM (SELECT a FROM t ORDER BY a)"
    )
    assert not qc.sorts_result("SELECT ROW_NUMBER() OVER (ORDER BY a) FROM t")


def test_real_rows_are_used_when_present():
    script = (
        SCHEMA + "\nINSERT INTO teams VALUES (1,'Bears','N'),(2,'Lions','N');"
        "\nINSERT INTO games VALUES (1,2023,1,2,20,10),(2,2023,2,1,30,3),(3,2023,2,1,7,3);"
    )
    r = qc.check_sql(GOLD, GOLD, script, real_rows=True)
    assert r["verdict"] == "pass" and r["checked_by"] == "rows+fuzz" and r["real"] is True


def test_empty_results_everywhere_are_unverified():
    gold = "SELECT name FROM teams WHERE conference = 'X' AND conference = 'Y'"
    other = "SELECT name FROM teams WHERE 1 = 0"
    assert qc.check_sql(other, gold, SCHEMA, real_rows=False)["verdict"] == "unverified"
    assert qc.check_sql(gold, gold, SCHEMA, real_rows=False)["checked_by"] == "exact"


def test_near_miss_copies_catch_starts_with_for_contains():
    # a real trace answer that version 3 passed: "starts with" for the reference's "contains"
    schema = "CREATE TABLE tweets (id INT, user_id INT, content TEXT, retweets INT, hashtags TEXT);"
    gold = "SELECT MAX(retweets) FROM tweets WHERE hashtags LIKE '%#zerowaste%'"
    pred = "SELECT MAX(retweets) AS max_retweets FROM tweets WHERE hashtags LIKE '#zerowaste%'"
    assert qc.check_sql(pred, gold, schema, real_rows=False, version=3)["verdict"] == "pass"
    r = qc.check_sql(pred, gold, schema, real_rows=False)
    assert r["verdict"] == "fail" and r["reason"] == "near_miss_rows_differ"
    alias = "SELECT MAX(t.retweets) FROM tweets AS t WHERE t.hashtags LIKE '%#zerowaste%'"
    assert qc.check_sql(alias, gold, schema, real_rows=False)["verdict"] == "pass"
    assert "x #zerowaste" in qc.near_misses(["#zerowaste"])


def test_a_tie_at_the_answers_limit_counts_neither_way():
    # SQLite's bare column with MAX picks one of the tied rows; LIMIT 1 may pick another
    schema = "CREATE TABLE worker_scores (worker_id INTEGER, score INTEGER);"
    gold = "SELECT worker_id, MAX(score) FROM worker_scores"
    pred = "SELECT worker_id, score FROM worker_scores ORDER BY score DESC, worker_id LIMIT 1"
    assert qc.check_sql(pred, gold, schema, real_rows=False)["verdict"] == "pass"
    wrong = "SELECT worker_id, score FROM worker_scores ORDER BY score ASC, worker_id LIMIT 1"
    assert qc.check_sql(wrong, gold, schema, real_rows=False)["verdict"] == "fail"


def test_only_nulls_and_zeros_are_no_evidence():
    gold = "SELECT MAX(home_score) FROM games WHERE season = 1999 AND season = 2000"
    other = "SELECT MAX(home_score) FROM games WHERE 1 = 0"  # (NULL,) as well, by accident
    assert qc.check_sql(other, gold, SCHEMA, real_rows=False, version=3)["verdict"] == "pass"
    assert qc.check_sql(other, gold, SCHEMA, real_rows=False)["verdict"] == "unverified"


def test_rescue_copies_verify_a_narrow_filter():
    schema = "CREATE TABLE orders (id INTEGER PRIMARY KEY, country TEXT, year INT, status TEXT);"
    gold = "SELECT id FROM orders WHERE country = 'Peru' AND year > 2020 AND status = 'active'"
    same = "SELECT id FROM orders WHERE status = 'active' AND country = 'Peru' AND year >= 2021"
    looser = "SELECT id FROM orders WHERE status = 'active' AND country = 'Peru' AND year >= 2020"
    r = qc.check_sql(same, gold, schema, real_rows=False)
    assert r["verdict"] == "pass" and r["fuzz_informative"] >= 1
    assert qc.check_sql(looser, gold, schema, real_rows=False)["verdict"] == "fail"


def test_cypher_is_exact_or_unverified():
    gold = (
        "MATCH (p:Person)-[:ACTED_IN]->(m:Movie) RETURN p.name, count(m) AS n "
        "ORDER BY n DESC LIMIT 1"
    )
    assert qc.check_cypher("  " + gold + " ", gold)["verdict"] == "pass"
    assert qc.check_cypher("MATCH (p:Person) RETURN p.name", gold)["verdict"] == "unverified"
    assert qc.check_cypher("MATCH (p) DETACH DELETE p RETURN 1", gold)["verdict"] == "fail"


def test_calls_from_message_parsed_and_raw():
    msg = {
        "tool_calls": [
            {"function": {"name": "run_query", "arguments": json.dumps({"query": "SELECT 1"})}}
        ]
    }
    assert qc.calls_from_message(msg) == [("run_query", {"query": "SELECT 1"})]
    raw = {
        "content": "<tool_call>execute_sql<arg_key>sql</arg_key>"
        "<arg_value>SELECT 2</arg_value></tool_call>"
    }
    assert qc.calls_from_message(raw) == [("execute_sql", {"sql": "SELECT 2"})]
    assert qc.calls_from_message({"content": "I can't answer that."}) == []


# ---- tool_prompt ------------------------------------------------------------------------------


def _example(dialect="sqlite", examples=None):
    system = public_prompt.system_prompt(SCHEMA, examples)
    return {
        "id": "x-1",
        "origin": "sqale",
        "dialect": dialect,
        "jepa": {"text": "Who won  the most?", "code": GOLD},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": public_prompt.target(GOLD)},
        ],
    }


def test_tool_prompt_reads_back_schema_and_examples():
    ex = _example(
        examples=[
            ("How many teams?", "SELECT COUNT(*) FROM teams"),
            ('List "names"', "SELECT name FROM teams"),
        ]
    )
    schema, pairs = tool_prompt.split_old_system(ex["messages"][0]["content"])
    assert schema == SCHEMA.strip() and pairs == [
        ("How many teams?", "SELECT COUNT(*) FROM teams"),
        ('List "names"', "SELECT name FROM teams"),
    ]
    assert tool_prompt.split_old_system(_example()["messages"][0]["content"])[1] == []


def test_tool_prompt_varies_tools_and_keeps_the_question():
    seen, distractors = set(), 0
    for seed in range(60):
        p = tool_prompt.from_example_dict(_example(), random.Random(seed))
        seen.add((p["tool"]["name"], p["tool"]["arg"]))
        distractors += p["distractor"] is not None
        assert p["user"] == "Who won the most?"
        assert f"`{p['tool']['name']}`" in p["system"] and SCHEMA.strip() in p["system"]
        names = [t["function"]["name"] for t in p["tools"]]
        assert p["tool"]["name"] in names and len(names) == 1 + (p["distractor"] is not None)
        assert "JSON" not in p["system"]
    assert len(seen) == len(tool_prompt.SQL_TOOLS) and 5 <= distractors <= 30


def test_assistant_turn_renders_in_spark_shape():
    turn = tool_prompt.assistant_turn("think", {"name": "execute_sql", "arg": "sql"}, "SELECT 1")
    assert turn["reasoning_content"] == "think"
    assert turn["tool_calls"][0]["function"] == {
        "name": "execute_sql",
        "arguments": {"sql": "SELECT 1"},
    }
    assert "reasoning_content" not in tool_prompt.assistant_turn(
        None, {"name": "a", "arg": "b"}, "q"
    )


# ---- tracegen against a fake llama-server, then the dashboard feed ----------------------------


class FakeServer(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            out = {"tokens": list(range(len(body["content"].split())))}
        else:
            tool = body["tools"][-1]["function"]
            arg = next(iter(tool["parameters"]["properties"]))
            thinking = body["chat_template_kwargs"].get("enable_thinking")
            wrong = (
                "home_score < g.away_score"
                if "low" == body["chat_template_kwargs"].get("reasoning_effort")
                else None
            )
            q = GOLD.replace("home_score > g.away_score", wrong) if wrong else GOLD
            out = {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "reasoning_content": "count the home wins per team" if thinking else "",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": tool["name"],
                                        "arguments": json.dumps({arg: q}),
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 40},
            }
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "tr"
    run.mkdir()
    qs = []
    for i in range(3):
        p = tool_prompt.build(SCHEMA, [], f"q{i}", "sqlite", random.Random(i))
        p["tools"] = [t for t in p["tools"] if t["function"]["name"] == p["tool"]["name"]]
        p["distractor"] = None
        qs.append(
            {
                "qid": f"sqale-{i}",
                "origin": "sqale",
                "dialect": "sqlite",
                "question": f"q{i}",
                **p,
                "gold": GOLD,
                "schema_script": SCHEMA,
                "real_rows": False,
                "meta": {},
            }
        )
    (run / "questions.jsonl").write_text(
        "".join(json.dumps(q) + "\n" for q in qs), encoding="utf-8"
    )
    cfg = {
        "name": "tr",
        "plan": {"high": 2, "low": 1, "off": 1},
        "questions": 3,
        "requests": 12,
        "sampling": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 64},
        "sets": {"sqale": 3},
    }
    (run / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return run


def test_tracegen_against_a_fake_server_then_resume(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        run = _run_dir(tmp_path)
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        assert tracegen.Run(run, url).go(workers=4) == 0
        res = [
            json.loads(line)
            for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(res) == 12 and len({r["rid"] for r in res}) == 12
        by = {r["level"]: r for r in res}
        assert by["high"]["verdict"] == "pass" and by["high"]["think_tokens"] == 6
        assert by["low"]["verdict"] == "fail" and by["low"]["reason"] == "fuzz_rows_differ"
        assert by["off"]["think_tokens"] == 0 and by["off"]["verdict"] == "pass"
        assert tracegen.Run(run, url).requests() == []  # a restart has nothing left to do
        events = [
            json.loads(line)["kind"]
            for line in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert events[0] == "start" and events[-1] == "DONE"
    finally:
        srv.shutdown()
    feed = Path(__file__).resolve().parents[1] / "scripts" / "trace_feed.py"
    out = subprocess.run(
        [sys.executable, str(feed), "--run", str(run), "--gpu-dir", str(tmp_path / "none")],
        capture_output=True,
        text=True,
        check=True,
    )
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert line["status"] == "done" and line["done"] == 12 and line["run_doc"]
    tick = json.loads(Path(line["tick_file"]).read_text(encoding="utf-8"))
    s = tick["summary"]
    assert s["by_level"]["high"]["pass_rate"] == 1.0 and s["by_level"]["low"]["pass_rate"] == 0.0
    assert s["pass_at"]["high"] == {"1": 1.0, "2": 1.0} and s["questions"]["sqale"] == {
        "seen": 3,
        "with_pass": 3,
    }
    assert json.loads(Path(line["run_doc"]).read_text(encoding="utf-8"))["kind"] == "trace"
    again = json.loads(
        subprocess.run(
            [sys.executable, str(feed), "--run", str(run), "--gpu-dir", str(tmp_path / "none")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    assert again["tick_file"] is None  # nothing new


def test_ties_at_a_limit_neither_pass_nor_fail_but_a_flipped_sort_fails():
    script = (
        "CREATE TABLE t (name TEXT, n INTEGER);"
        "INSERT INTO t VALUES ('a', 5), ('b', 5), ('c', 1), ('d', 0), ('e', 0);"
    )
    gold = "SELECT name, n FROM t ORDER BY n DESC LIMIT 1"
    db = qc._connect(script)
    g = qc._run(db, gold)
    other = [("b", 5)] if g == [("a", 5)] else [("a", 5)]
    assert qc._tie(db, gold, g, other)  # the other team on 5 is a fair answer
    assert not qc._tie(db, gold, g, [("d", 0)])  # sorted the wrong way
    assert qc._sort_key_column(gold, ["name", "n"]) == 1
    assert qc._sort_key_column("SELECT a, b FROM t ORDER BY 2 DESC LIMIT 3", ["a", "b"]) == 1
    assert qc._sort_key_column("SELECT x.a AS a FROM t x ORDER BY x.a LIMIT 3", ["a"]) == 0
    flipped = qc.check_sql(
        "SELECT name, n FROM t ORDER BY n ASC LIMIT 1", gold, script, real_rows=True
    )
    assert flipped["verdict"] == "fail"
    assert (
        qc.strip_limit("SELECT a FROM t WHERE b = 'LIMIT 3' LIMIT 5 OFFSET 2")
        == "SELECT a FROM t WHERE b = 'LIMIT 3'"
    )


# ---- `call` examples: the schema, the export, and the rendering with Spark's own tokenizer ----


def _call_example(level="high", reasoning="The home wins per team, the most first."):
    from asktrainer.traceexport import example

    p = tool_prompt.build(SCHEMA, [], "Who won the most home games?", "sqlite", random.Random(3))
    q = {
        "qid": "sqale-9",
        "origin": "sqale",
        "dialect": "sqlite",
        "question": "Who won the most?",
        **p,
        "gold": GOLD,
        "meta": {"db_id": "x"},
    }
    r = {
        "qid": "sqale-9",
        "level": level,
        "try": 0,
        "query": GOLD,
        "reasoning": reasoning,
        "checked_by": "fuzz",
        "think_tokens": 9,
    }
    return example(q, r, "train", "tr")


def test_call_example_validates_and_exposes_the_query():
    from asktrainer.data.schema import Example

    ex = _call_example()
    assert ex.task == "call" and ex.target == GOLD and ex.thinking == "high"
    assert ex.chat_messages[-1]["reasoning_content"].startswith("The home wins")
    assert "reasoning_content" not in _call_example("off", "").chat_messages[-1]
    bad = ex.model_dump()
    bad["thinking"] = "off"  # thinking off with a thought is inconsistent
    with pytest.raises(ValueError):
        Example.model_validate(bad)


def test_export_keeps_the_first_passing_answer_per_question_and_level(tmp_path):
    from asktrainer.traceexport import export

    run = _run_dir(tmp_path)
    rows = []
    for i in range(3):
        for level, tries in (("high", [("fail", None), ("pass", GOLD)]), ("off", [("pass", GOLD)])):
            for t, (v, qy) in enumerate(tries):
                rows.append(
                    {
                        "qid": f"sqale-{i}",
                        "level": level,
                        "try": t,
                        "verdict": v,
                        "query": qy,
                        "reasoning": "" if level == "off" else f"think {t}",
                        "checked_by": "fuzz",
                    }
                )
    (run / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    stats = export(run, tmp_path / "out", val_share=0.34, seed=0)
    assert stats["train"] + stats["val"] == 6 and stats["val_questions"] == 1
    lines = (tmp_path / "out" / "examples_train.jsonl").read_text(encoding="utf-8").splitlines()
    ex = [json.loads(line) for line in lines]
    assert all(e["task"] == "call" for e in ex)
    assert {e["messages"][-1].get("reasoning_content") for e in ex if e["thinking"] == "high"} == {
        "think 1"
    }


def test_prep_with_a_mix_gives_each_question_one_level_and_interleaves(tmp_path):
    from asktrainer import traceprep

    pub = tmp_path / "public"
    for name in ("sqale", "gretel"):
        (pub / name).mkdir(parents=True)
        rows = []
        for i in range(40):
            ex = _example()
            ex.update(id=f"{name}-{i}", origin=name, meta={})
            rows.append(json.dumps(ex) + "\n")
        (pub / name / "examples_train.jsonl").write_text("".join(rows), encoding="utf-8")
    assert traceprep.parse_sizes("25") == 25
    sizes = traceprep.parse_sizes("sqale:30,gretel:10")
    mix = traceprep.parse_mix("high:0.6,low:0.3,off:0.1")
    with pytest.raises(ValueError):
        traceprep.parse_sizes("spider:5")
    with pytest.raises(ValueError):
        traceprep.parse_mix("medium:1")
    cfg = traceprep.prep(tmp_path / "r", pub, sizes, {}, seed=0, mix=mix, tries=2, budget=1024)
    qs = [json.loads(x) for x in (tmp_path / "r" / "questions.jsonl").read_text().splitlines()]
    assert cfg["sets"] == {"sqale": 30, "gretel": 10} and cfg["questions"] == 40
    assert all(len(q["plan"]) == 1 and next(iter(q["plan"].values())) == 2 for q in qs)
    assert cfg["requests"] == 80 and sum(cfg["levels"].values()) == 40
    assert cfg["levels"]["high"] > cfg["levels"]["off"] and cfg["adaptive"]
    assert cfg["reasoning_budget"] == 1024
    first = [q["origin"] for q in qs[:20]]
    assert "gretel" in first  # interleaved, not 30 sqale then 10 gretel


def test_adaptive_tracegen_asks_again_only_the_unsolved(tmp_path):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        run = _run_dir(tmp_path)
        qs = [json.loads(x) for x in (run / "questions.jsonl").read_text().splitlines()]
        for q, level in zip(qs, ("high", "low", "off"), strict=True):
            q["plan"] = {level: 2}  # the fake server answers low wrongly, high and off rightly
        (run / "questions.jsonl").write_text("".join(json.dumps(q) + "\n" for q in qs))
        cfg = json.loads((run / "config.json").read_text())
        cfg.update(plan={}, mix={"high": 1, "low": 1, "off": 1}, tries=2, adaptive=True)
        cfg["requests"] = 6
        (run / "config.json").write_text(json.dumps(cfg))
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        assert tracegen.Run(run, url).go(workers=2) == 0
        res = [json.loads(x) for x in (run / "results.jsonl").read_text().splitlines()]
        assert sorted(r["rid"] for r in res) == [
            "sqale-0#high#0",
            "sqale-1#low#0",
            "sqale-1#low#1",
            "sqale-2#off#0",
        ]
        ev = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
        assert [e["todo"] for e in ev if e["kind"] == "round"] == [3, 1]
        # a later re-check that passes an answer counts as solved for the next round
        (run / "rechecks.jsonl").write_text(
            json.dumps(
                {"rid": "sqale-1#low#0", "qid": "sqale-1", "level": "low", "verdict": "pass"}
            )
            + "\n"
        )
        assert ("sqale-1", "low") in tracegen.Run(run, url).passed
    finally:
        srv.shutdown()
    feed = Path(__file__).resolve().parents[1] / "scripts" / "trace_feed.py"
    out = subprocess.run(
        [sys.executable, str(feed), "--run", str(run), "--gpu-dir", str(tmp_path / "none")],
        capture_output=True,
        text=True,
        check=True,
    )
    line = json.loads(out.stdout.strip().splitlines()[-1])
    assert line["done"] == 4
    tick = json.loads(Path(line["tick_file"]).read_text(encoding="utf-8"))
    assert tick["status"]["total"] == 4  # 3 first tries + the 1 the second round asked
    assert tick["summary"]["pass_at"]["high"] == {"1": 1.0, "2": 1.0}  # solved at try 0
    # the feed applies the re-check: low's first try now passes
    assert tick["summary"]["pass_at"]["low"] == {"1": 1.0, "2": 1.0}


def test_export_leaves_out_cut_thoughts_and_applies_rechecks(tmp_path):
    from asktrainer.traceexport import export

    run = _run_dir(tmp_path)
    cfg = json.loads((run / "config.json").read_text())
    cfg["reasoning_budget"] = 1024
    (run / "config.json").write_text(json.dumps(cfg))
    rows = [  # q0: passed but cut by the budget; q1: failed, passed on a re-check; q2: plain
        {"rid": "a", "qid": "sqale-0", "level": "high", "try": 0, "verdict": "pass",
         "query": GOLD, "reasoning": "long", "think_tokens": 1024, "checked_by": "fuzz"},
        {"rid": "b", "qid": "sqale-1", "level": "high", "try": 0, "verdict": "unverified",
         "query": GOLD, "reasoning": "ok", "think_tokens": 50, "checked_by": ""},
        {"rid": "c", "qid": "sqale-2", "level": "low", "try": 0, "verdict": "pass",
         "query": GOLD, "reasoning": "fine", "think_tokens": 80, "checked_by": "fuzz"},
    ]  # fmt: skip
    (run / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "rechecks.jsonl").write_text(
        json.dumps({"rid": "b", "verdict": "pass", "checked_by": "graph"}) + "\n"
    )
    stats = export(run, tmp_path / "out", val_share=0.0, seed=0)
    assert stats["train"] == 2 and stats["left_out_cut_by_budget"] == 1
    assert stats["checked_by"] == {"graph": 1, "fuzz": 1}


SPARK = Path("/mnt/f/LOCAL-Models/XHToken/Spark-X2.6-4B")


@pytest.mark.skipif(not (SPARK / "tokenizer.json").exists(), reason="Spark's tokenizer isn't here")
def test_call_example_renders_and_masks_in_spark_format():
    from transformers import AutoTokenizer

    from asktrainer.chat import ChatTokenizer

    ct = ChatTokenizer(
        AutoTokenizer.from_pretrained(str(SPARK), trust_remote_code=True), max_len=8192
    )
    for level in ("high", "low", "off"):
        ex = _call_example(
            level, "" if level == "off" else "The home wins per team, the most first."
        )
        ids, plen = ct.full_ids(ex)
        prompt = ct.tok.decode(ids[:plen], skip_special_tokens=False)
        target = ct.tok.decode(ids[plen:], skip_special_tokens=False)
        assert prompt.endswith("<|Bot|><think>" if level != "off" else "<|Bot|></think>")
        assert (
            ("thinking_effort=" + level) in prompt
            if level != "off"
            else "thinking_effort" not in prompt
        )
        assert "<tools>" in prompt and ex.tools[0]["function"]["name"] in prompt
        call = f"<tool_call>{ex.messages[-1].tool_calls[0]['function']['name']}<arg_key>"
        assert call in target and target.endswith("</tool_call><｜end▁of▁sentence｜>")
        if level != "off":
            assert target.startswith("The home wins") and "</think>" in target
