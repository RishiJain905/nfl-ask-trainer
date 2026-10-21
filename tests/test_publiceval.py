from __future__ import annotations

import json

from asktrainer import publiceval as pe
from asktrainer.data.schema import Example

CTX = (
    "CREATE TABLE sales (id INT, region TEXT, amount FLOAT); "
    "INSERT INTO sales (id, region, amount) VALUES (1, 'EU', 10.5), (2, 'US', 20.25), "
    "(3, 'EU', 4.0);"
)


def ex(origin: str, query: str, dialect: str = "sqlite", **meta) -> Example:
    return Example(
        id=f"{origin}-1", origin=origin, task="generate", split="val", group="public",
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "q"},
                  {"role": "assistant", "content": json.dumps({"query": query})}],
        jepa={"text": "q", "code": query}, dialect=dialect, meta=meta,
    )  # fmt: skip


class FakeData(pe.PublicData):
    def __init__(self, scripts):
        self.scripts = scripts

    def schema_script(self, e):
        return self.scripts.get(e.origin)


def reply(q: str) -> str:
    return json.dumps({"query": q})


GOLD = "SELECT region, SUM(amount) AS total FROM sales GROUP BY region"


def test_gretel_executes_and_ignores_order_and_names():
    data = FakeData({"gretel": CTX})
    e = ex("gretel", GOLD, row_id="1")
    same = "SELECT region, sum(amount) FROM sales GROUP BY region ORDER BY region DESC"
    item = pe.score(e, reply(same), data)
    assert item["exec_ok"] is True and item["parse_ok"] and not item["exact"]
    wrong = "SELECT region, COUNT(*) FROM sales GROUP BY region"
    assert pe.score(e, reply(wrong), data)["exec_ok"] is False
    assert pe.score(e, reply(GOLD + " ;"), data)["exact"] is True


def test_gretel_parse_errors_and_bad_replies():
    data = FakeData({"gretel": CTX})
    e = ex("gretel", GOLD, row_id="1")
    item = pe.score(e, reply("SELECT nope FROM sales"), data)
    assert item["parse_ok"] is False and item["exec_ok"] is False
    assert pe.score(e, "not json", data)["json_ok"] is False
    assert pe.score(e, json.dumps({"refuse": "no_data"}), data)["reason"] == "no_query"
    assert pe.score(e, reply("DELETE FROM sales"), data)["reason"] == "not_read_only"


def test_gold_failure_is_reported_not_scored():
    data = FakeData({"gretel": CTX})
    e = ex("gretel", "SELECT x FROM missing", row_id="1")
    item = pe.score(e, reply("SELECT region FROM sales"), data)
    assert item["gold_failed"] is True and item["exec_ok"] is None


def test_precision_follows_the_gold():
    assert pe.same_results([(1, 0.33)], [(1.0, 0.3333333)])  # the gold has 2 decimals
    assert not pe.same_results([(1, 0.33)], [(1.0, 0.34)])
    assert pe.same_results([("a", 1), ("a", 1), ("b", 2)], [("b", 2), ("a", 1), ("a", 1)])
    assert not pe.same_results([("a", 1), ("a", 1)], [("a", 1)])  # a multiset, not a set
    assert not pe.same_results([("a", 1)], [("a", 1, 2)])
    assert pe.same_results([], [])


def test_sqale_parse_check_against_its_schema():
    data = FakeData({"sqale": "CREATE TABLE t (id INTEGER, name TEXT);"})
    e = ex("sqale", "SELECT name FROM t", db_id="schema_1")
    assert pe.score(e, reply("SELECT name FROM t WHERE id = 3"), data)["parse_ok"] is True
    item = pe.score(e, reply("SELECT nam FROM t"), data)
    assert item["parse_ok"] is False and item["exec_ok"] is None  # sqale isn't executed


def test_cypher_light_check():
    ok, _ = pe.cypher_ok("MATCH (a:Player {name: 'Tom (TB)'})-[:PLAYED]->(t) RETURN t.name")
    assert ok
    assert pe.cypher_ok("MATCH (a RETURN a")[1] == "brackets"
    assert pe.cypher_ok("MATCH (a) RETURN a.name WHERE a.x = 'oops")[1] == "quotes"
    assert pe.cypher_ok("MATCH (a) SET a.x = 1 RETURN a")[1] == "write"
    assert pe.cypher_ok("MATCH (a) WHERE a.id = $id RETURN a")[1] == "parameter"
    assert pe.cypher_ok("MATCH (a)")[1] == "no_return"
    assert pe.cypher_ok("SELECT 1")[1] == "start"
    assert pe.cypher_ok("MATCH (a) WHERE a.note = 'set $x' RETURN a")[0]  # inside a string


def test_summary_and_metrics():
    data = FakeData({"gretel": CTX})
    exs = [ex("gretel", GOLD, row_id="1"),
           ex("text2cypher", "MATCH (n) RETURN n.name", dialect="cypher")]  # fmt: skip
    exs[1] = exs[1].model_copy(update={"id": "text2cypher-2"})
    preds = [{"id": exs[0].id, "output": reply(GOLD)},
             {"id": exs[1].id, "output": reply("MATCH (n) RETURN n.name")}]  # fmt: skip
    res = pe.verify(exs, preds, data)
    m = pe.metrics(res)
    assert m["val/gretel/exec_acc"] == 1.0 and m["val/text2cypher/exact"] == 1.0
    assert m["val/public/json_ok"] == 1.0 and m["val/public/n"] == 2.0
    assert "val/text2cypher/exec_acc" not in m
