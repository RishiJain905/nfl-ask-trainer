"""The Cypher checker: the schema parser, literal planting, random graphs and the final-clause
helpers on the CPU; the checks themselves against the local Neo4j when it answers (Windows side:
`asktrainer-neo4j` on 127.0.0.1:7476; skipped elsewhere, e.g. in WSL)."""

from __future__ import annotations

import pytest

from asktrainer import cyphercheck as cc

SYSTEM = """You answer questions about the graph database below by calling the `run_query` tool.

Nodes (properties are STRING unless typed):
- Product {productName, unitPrice: FLOAT, unitsInStock: INTEGER, discontinued: BOOLEAN}
- Category {categoryName, description}
- Order {orderID, orderDate, freight}
Relationships:
- (Product)-[:PART_OF]->(Category)
- (Order)-[:ORDERS {quantity: INTEGER, discount}]->(Product)

Examples (tested queries on this database):
Q: "Which products are in Beverages?"
MATCH (p:Product)-[:PART_OF]->(c:Category {categoryName: 'Beverages'}) RETURN p.productName

Rules:
- One statement, read only (MATCH ... RETURN).
"""


def test_parse_schema_reads_nodes_types_and_relationships():
    s = cc.parse_schema(SYSTEM)
    assert s["nodes"]["Product"] == {
        "productName": "STRING",
        "unitPrice": "FLOAT",
        "unitsInStock": "INTEGER",
        "discontinued": "BOOLEAN",
    }
    assert set(s["nodes"]) == {"Product", "Category", "Order"}
    assert ("Order", "ORDERS", "Product", {"quantity": "INTEGER", "discount": "STRING"}) in s[
        "rels"
    ]
    assert len(s["rels"]) == 2  # the rules' "- One statement" line is not a relationship


def test_literals_are_tied_to_the_properties_they_are_compared_with():
    lits = cc.literals(
        [
            "MATCH (c:Category {categoryName: 'Beverages'})<-[:PART_OF]-(p:Product) "
            "WHERE p.unitPrice > 20 AND toFloat(o.freight) < 1.5 AND 'x' = p.productName RETURN p"
        ]
    )
    bp = lits["by_prop"]
    assert bp["categoryName"] == ["Beverages"] and bp["unitPrice"] == [20]
    assert bp["freight"] == [1.5] and bp["productName"] == ["x"]
    assert "Beverages" in lits["strs"] and 20.0 in lits["nums"]


def test_fuzz_graph_is_seeded_typed_and_plants_literals():
    schema = cc.parse_schema(SYSTEM)
    lits = cc.literals(["MATCH (c:Category {categoryName: 'Beverages'}) RETURN c"])
    a = cc.fuzz_graph(schema, 11, lits)
    assert a == cc.fuzz_graph(schema, 11, lits) and a != cc.fuzz_graph(schema, 23, lits)
    nodes, rels, types = a
    assert len(nodes["Product"]) == cc.NODES_PER_LABEL
    names = [n.get("categoryName") for n in nodes["Category"]]
    assert names.count("Beverages") >= 3  # planted often enough to match
    assert all(isinstance(n.get("unitsInStock", 0), int) for n in nodes["Product"])
    assert all(isinstance(n.get("discontinued", True), bool) for n in nodes["Product"])
    dates = [n["orderDate"] for n in nodes["Order"] if "orderDate" in n]
    assert sum(1 for d in dates if cc.DATE_RE.match(d)) >= len(dates) // 2  # a date kept as text
    assert {r[0] for r in rels} == {"PART_OF", "ORDERS"} and types == {}
    distinct = cc.fuzz_graph(schema, 53, lits, distinct=True)[0]
    prices = [n["unitPrice"] for n in distinct["Product"] if "unitPrice" in n]
    assert len(set(prices)) == len(prices)


def test_final_clause_helpers_look_after_the_last_return():
    q = (
        "MATCH (o:Officer)-[:officer_of]->(e) WITH o, count(e) AS n ORDER BY n DESC LIMIT 1 "
        "RETURN o.name AS name, n"
    )
    assert not cc.sorts_result(q) and cc.strip_limit(q) is None
    q2 = "MATCH (p:Product) RETURN p.productName, p.unitPrice ORDER BY p.unitPrice DESC LIMIT 5"
    assert cc.sorts_result(q2)
    assert cc.strip_limit(q2) == q2[: q2.index(" LIMIT")]
    assert cc._sort_key_column(q2, ["p.productName", "p.unitPrice"]) == 1
    q3 = "MATCH (p:Product) RETURN p.productName AS name ORDER BY name SKIP 2 LIMIT 3"
    assert cc._sort_key_column(q3, ["name"]) == 0 and cc.strip_limit(q3) is None
    sub = "CALL { MATCH (a) RETURN a ORDER BY a.x LIMIT 2 } RETURN a.name"
    assert not cc.sorts_result(sub) and cc.strip_limit(sub) is None


def test_informative_rows():
    assert not cc._informative([]) and not cc._informative([(0,), (None,)])
    assert cc._informative([(0, "x")]) and cc._informative([((1, 2),)])


DB = cc.Neo4j()
neo4j = pytest.mark.skipif(not DB.ok(), reason="no local Neo4j on 127.0.0.1:7476")

GOLD = (
    "MATCH (p:Product)-[:PART_OF]->(c:Category {categoryName: 'Beverages'}) "
    "RETURN p.productName AS name, p.unitPrice AS price ORDER BY price DESC LIMIT 3"
)


@neo4j
def test_fuzz_check_passes_rewrites_and_fails_wrong_answers():
    preds = [
        GOLD,
        "MATCH (c:Category)<-[:PART_OF]-(p:Product) WHERE c.categoryName = 'Beverages' "
        "RETURN p.productName, p.unitPrice ORDER BY p.unitPrice DESC LIMIT 3",
        "MATCH (p:Product)-[:PART_OF]->(c:Category {categoryName: 'Beverages'}) "
        "RETURN p.productName, p.unitPrice ORDER BY p.unitPrice ASC LIMIT 3",
        "MATCH (p:Product) RETURN p.productName, p.unitPrice ORDER BY p.unitPrice DESC LIMIT 3",
        "MATCH (p:Product) GROUP BY p RETURN p.productName",
        "MATCH (p:Product) SET p.x = 1 RETURN p.productName",
    ]
    res = cc.check_fuzz(DB, SYSTEM, GOLD, preds)
    assert res[0]["verdict"] == "pass" and res[0]["checked_by"] == "exact"
    assert res[1]["verdict"] == "pass" and res[1]["checked_by"] == "fuzz_graph"
    assert res[2]["verdict"] == "fail"
    assert res[3]["verdict"] == "fail" and res[3]["reason"] == "fuzz_rows_differ"
    assert res[4]["verdict"] == "fail" and "SyntaxError" in res[4]["reason"]
    assert res[5]["verdict"] == "fail" and res[5]["reason"] == "syntax:write"


@neo4j
def test_a_wrong_order_fails_on_the_tie_free_graph():
    gold = "MATCH (p:Product) RETURN p.productName AS n, p.unitPrice AS u ORDER BY u DESC"
    flipped = "MATCH (p:Product) RETURN p.productName AS n, p.unitPrice AS u ORDER BY u ASC"
    assert cc.check_fuzz(DB, SYSTEM, gold, [flipped])[0]["reason"] == "order_differs"


@neo4j
def test_check_on_a_loaded_graph_and_read_only_runs():
    nodes = {"Person": [{"eid": "p1", "name": "Ada"}, {"eid": "p2", "name": "Bo"}]}
    nodes["Painting"] = [{"eid": "x1", "name": "Sun", "creation_year": 1900}]
    rels = [("createdBy", "Painting", "x1", "Person", "p1", {})]
    cc.load(DB, nodes, rels, {}, key="eid", keep_key=True, index=True)
    gold = "MATCH (p:Painting)-[:createdBy]->(a:Person) RETURN a.name"
    res = cc.check_on_graph(
        DB,
        gold,
        [
            "MATCH (a:Person)<-[:createdBy]-(:Painting) RETURN a.name",
            "MATCH (a:Person) RETURN a.name",
        ],
    )
    assert [r["verdict"] for r in res] == ["pass", "fail"]
    with pytest.raises(cc.CypherError):
        DB.run("CREATE (n:Hack) RETURN n")  # the READ access mode refuses writes
    DB.wipe()
