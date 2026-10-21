# Public datasets in the SFT mix

The Ask model's job is a *behaviour*: read the schema in front of it, write one read-only query,
reply with one JSON object. Our own NFL examples teach it our schemas; these four public
text-to-query datasets teach the behaviour across thousands of other schemas (SQL and Cypher).
Everything below is converted into the `Example` record of `src/asktrainer/data/schema.py` with
the prompt of `src/asktrainer/data/public_prompt.py`.

| `origin` | dialect | source | licence | train examples | val examples |
|---|---|---|---|---:|---:|
| `sqale` | `sqlite` | `trl-lab/SQaLe-2-text-to-SQL-Queries` (+ DDL from `trl-lab/SQaLe-2-text-to-SQL-Schemas`) | MIT | 30,000 | 300 |
| `gretel` | `sqlite` | `gretelai/synthetic_text_to_sql` | Apache-2.0 | 30,000 | 300 |
| `text2cypher` | `cypher` | `neo4j/text2cypher-2025v1` | Apache-2.0 | 27,084 | 300 |
| `cypherbench` | `cypher` | `megagonlabs/cypherbench` | Apache-2.0 | 8,534 | 300 |
| | | | | **95,618** | **1,200** |

Files, per dataset, under `/mnt/d/nfl-ml-data/ask/training/public/<name>/` (`D:\nfl-ml-data\...`):

- `raw/` : what was downloaded, plus `raw/manifest.json` (repo id, revision, files, sizes, sha256,
  licence, download date). 438 MB for all four.
- `examples_train.jsonl`, `examples_val.jsonl`, `stats.json` : the converted examples (299 MB
  for all four) and the counts behind this file.

Rebuild (CPU only, about 4 minutes for all four after the download):

```
python -m asktrainer.data.public --dataset all --download --convert --seed 0
python -m asktrainer.data.public --dataset text2cypher --convert --val-by-db   # unseen-graph val
python -m asktrainer.data.public --dataset text2cypher --convert --keep-whole-nodes
```

Code: `src/asktrainer/data/download.py` (fetch + manifest), `public.py` (adapters, CLI),
`public_text.py` (lexer, read-only filter, schema parsers), tests in `tests/test_public.py`.

## What every dataset goes through

- **Schema text.** SQL: the `CREATE TABLE` / `CREATE VIEW` statements only, one line each,
  comments removed (INSERTs, indexes, pragmas dropped). Cypher: one compact text, node labels
  with properties and relationship types with their endpoints:

  ```
  Nodes (properties are STRING unless typed):
  - Movie {title, released: INTEGER}
  Relationships:
  - (Person)-[:ACTED_IN {roles: LIST}]->(Movie)
  ```

  Example values and min/max hints of the source schemas are dropped. An example whose schema
  text is over 6,000 characters is dropped, except SQaLe, whose databases are far too big for the
  prompt and are **trimmed** instead (below).
- **Queries.** One statement (a trailing `;` is removed), read-only, no comments (a comment drops
  the row), whitespace collapsed outside string literals. SQL: starts with `SELECT` or `WITH`,
  no `INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/PRAGMA/...`. Cypher: `MATCH / OPTIONAL MATCH / WITH
  / UNWIND / CALL { }` ... `RETURN`, no `CREATE/MERGE/SET/DELETE/REMOVE/DROP/LOAD/FOREACH`, no
  procedure calls (`CALL db.x()`, `apoc.*`, `gds.*`); subquery `CALL { ... }` and `CALL (x) { ... }`
  are read-only and are kept. Brackets must balance. Strings and quoted identifiers are never
  scanned for keywords (a lexer separates code from literals).
- **Compiled, for SQL.** Every SQL gold is compiled (`EXPLAIN`) in an in-memory SQLite database
  holding the prompt's `CREATE` statements, so the `sqlite` label is true. Cypher cannot be run here
  (no Neo4j), it is only checked as above.
- **Cypher golds that return a whole node, relationship or path** (`RETURN m`, `RETURN n, r`)
  are dropped by default because the prompt tells the model to return named properties
  (`--keep-whole-nodes` keeps them). SQL `SELECT *` is kept (it is legitimate in subqueries).
- **Deduplication.** A (question, query) pair is kept once per dataset; a val pair that also
  appears in train is removed from val.
- **Prompt examples.** For half of the prompts (seeded per example id) 1-2 *other* (question,
  query) pairs of the same database / graph (same schema text) are shown under "Examples (tested
  queries on this database)"; the others say "(none)". Never the example's own pair or question.
  The share really achieved is in `stats.json` (`examples_in_prompt_share`): about 50% except where
  databases are not shared between rows (gretel, 8%).
- **Validation.** Each dataset's own test split, sampled to 300 with one pair per database in
  turn. `--val-by-db` holds out 1% of whole databases instead. `stats.json` records how many val
  databases also occur in train.
- **Deterministic.** The same raw files and `--seed` give the same bytes (tested).
- **Fields.** `group="public"`, `task="generate"`, `dialect` as above, `jepa = {text: question,
  code: query}`, `meta = {row_id, db_id, license, source_split, n_prompt_examples, schema_chars,
  ...dataset fields}`, id `<origin>-<row id>` (a val row that reuses a train row id gets `-val`).

## sqale

- **What.** `trl-lab/SQaLe-2-text-to-SQL-Queries`, revision `85fc905c3484f9d197ee8736d672a7d0e4903fc6`,
  MIT. 177,377 question records (169,277 train / 8,100 test) with gold SQLite SQL over 9,259
  populated databases (schemas from SchemaPile, extended by an LLM; median 113 tables, 538 columns).
  Every gold ran against the populated database and passed an LLM judge. Each record has the question
  in eight phrasings. Authors: Wolff, Gomm, Hulsebos (University of Amsterdam / CWI), "SQaLe: A Large
  Realistic Dataset to Empower Small Specialised Text-to-SQL Models", 2026. Please cite it.
- **DDL.** From the companion `trl-lab/SQaLe-2-text-to-SQL-Schemas` (MIT, revision
  `091b1a9e946dec386470c7dc381af166d2b41c38`). That repo is 4.0 GB, 98% of it generated table rows;
  we read only the `schema_id`, `full_schema` and `number_of_tables` columns of its 17 parquet shards
  with HTTP range requests (about 76 MB transferred) into `raw/schemas_ddl.parquet` (52 MB). No
  populated database was downloaded. Raw total 343 MB.
- **Schema trimmed, not dropped.** The prompt's schema is the `CREATE TABLE`s of the tables the gold
  query uses (found by matching identifiers outside literals), plus the tables of the other
  examples shown in that prompt, plus the remaining tables of the question's own sub-schema
  (`relevant_tables`) while the text stays under 6,000 characters, in the database's own table order
  so the used tables are not marked. Median 6 tables, at most 26. FOREIGN KEYs may name tables that
  are not shown (as in any trimmed schema).
- **One phrasing per record**, drawn by weight: `short_high_level` 3, `verbose` 3, `casual` 2,
  `evidence_supported` 1 (the `, evidence: ...` tail is cut), `spelling_grammar_mistakes` 1. The
  bulleted `structured`, fragment `requirements_list` and underspecified `short_ambiguous` styles
  are not used. A phrasing must look like a question and share a content word with the query
  (identifier parts split at `_` / camelCase, plus the words of its string literals); the next
  phrasing is tried otherwise.
- **Sampling.** 30,000 train examples, one query per database in turn (at most 4 per database,
  8,835 databases). Val: SQaLe's own test split (423 databases, none in train), 300 databases.
- **Counts (train).** 169,277 rows; dropped before sampling: no usable question 5,721, comment 77,
  write statement 9, duplicate 2; 163,468 usable; 30,034 reached
  (34 of them name no table of their schema, 0 compile errors, 0 too-long schemas). Val: 8,100 rows, 260 no usable question, 2 comments, 1 no table.
- **Lengths (chars, p50 / p90 / p99 / max).** schema 1,421 / 3,204 / 5,131 / 5,998; question 149 /
  303 / 552 / 1,010; query 181 / 528 / 1,193 / 6,036; whole prompt 2,857 / 4,781 / 6,784 / 8,656.
  Examples in prompt: 49.6%.
- **Quirks.** The data card says about one record in ten has a placeholder phrasing; there are
  also generator notes where the question should be ("Easy questions", "**Moderate**", "Define
  domain grounding", "Keep it basic."); questions of five words or fewer were 1.3% of a first,
  unfiltered 30k sample and many of those were such notes, hence the alignment check. Gold SQL carries stray indentation (collapsed). `full_schema` also holds
  `CREATE INDEX`, `IF NOT EXISTS`, a few `INSERT`s inside default values (all handled). Gold
  queries compile in SQLite in every one of the 30,034 cases reached (they were executed upstream).

## gretel

- **What.** `gretelai/synthetic_text_to_sql`, revision `740ab236e64503fba51be1101df7a1be83bf455d`,
  Apache-2.0. 105,851 synthetic (question, schema, SQL) records over 100 domains (100,000 train /
  5,851 test); `sql_context` holds `CREATE TABLE` (sometimes `CREATE VIEW`) and `INSERT` statements.
  Gretel AI, April 2024. Raw total 34 MB.
- **Kept.** The `sql_prompt`, the `CREATE` statements of `sql_context`, the `sql`. SELECT tasks
  only: the data-manipulation and data-definition tasks fall out of the read-only filter.
- **SQLite compile check matters here.** The golds are written for MySQL / PostgreSQL and the
  contexts are not SQLite either. Of the rows reaching the compile check, 22% fail in SQLite:
  `DATE_SUB(..., INTERVAL ...)` and other syntax (7%), `no such column` (5%), unknown functions such
  as `YEAR()` / `STR_TO_DATE` (5%), schema-qualified names such as `defense.table` (4%), ambiguous
  columns (1%), window-function misuse; 549 contexts do not create in SQLite at all (mostly
  schema-qualified `CREATE TABLE public.t`, plus MySQL / PostgreSQL-only syntax).
- **Sampling.** 30,000 train examples, a seeded random pick of the usable rows. Val: the test split, 300.
- **Counts (train).** 100,000 rows; before sampling: write statements 10,170 (the
  data-manipulation and data-definition tasks), multi-statement 473, no CREATE in the context 230,
  duplicate 11, comment 1, other non-SELECT 9; 89,106 usable; 39,514 reached while sampling, of
  which SQLite compile error 8,965 and DDL not SQLite 549. Val: 5,851 rows, DML 609, compile errors 84, multi-statement 20, no schema 19.
- **Lengths (chars, p50 / p90 / p99 / max).** schema 99 / 171 / 288 / 825 (tiny: one to three
  tables); question 78 / 114 / 160 / 302; query 102 / 197 / 317 / 758; whole prompt 1,147 / 1,275 /
  1,542 / 2,402.
- **Quirks.** Every row has its own schema (30,000 different ones), so only 8.3% of prompts can
  show other pairs of the same database (rows that share an identical context); 41 of the 300 val
  schemas also occur in train. Some golds do not match their question ("names and dates" answered
  with `date, location`); the synthetic generator is not perfect.

## text2cypher

- **What.** `neo4j/text2cypher-2025v1`, revision `d88b8a7d477c8c1809218f6c566bd31f1ad1e504`,
  Apache-2.0. 35,946 train / 4,438 test (question, schema text, Cypher) merged from Neo4j's earlier
  releases and community sets (`data_source`: synthetic GPT-4o / GPT-4-turbo / Gemini / Claude Opus,
  functional queries, the 2023 train set, crowd-sourced, Cyspider). Neo4j; the Text2Cypher paper is
  arXiv:2412.10064 (from memory, not checked against the download). The dataset card itself has no
  citation. Raw total 5 MB.
- **Schemas.** Stored as text in several formats. Four are parsed (neo4j bulleted `Node properties:`
  text, its inline `Label {prop: TYPE}` form, `Graph schema: Relevant node labels ...`, and
  `apoc.meta.schema()` JSON) and re-rendered compactly; the rest (Cyspider's SQL-style lists, a
  `(:A{...})-[..]` pattern list, a quoted `Node properties are the following` form) are dropped as
  unparseable (845 rows, 2.3%).
- **Graph id.** `database_reference_alias` (16 demo databases, e.g. `neo4jlabs_demo_db_movies`); 43%
  of the kept rows have none, they get `schema-<hash of the schema text>`. Prompt examples are drawn
  among rows with the same schema text.
- **Counts (train).** 35,946 rows; dropped: returns a whole node / relationship 7,828 (21.8%),
  unparseable schema 845, procedure call 48 (`CALL db.schema.visualization()` etc.), duplicate 117,
  bad question 24; **27,084 kept**. Val: 4,438 rows; 863 whole-node, 118 unparseable, 23 also in
  train; 300 sampled. 13 rows have a write gold (`CREATE`, `DETACH DELETE ... IN TRANSACTIONS`,
  `SET`); all 13 sit on schemas the parsers do not read, so the schema check drops them first (the
  write filter itself is covered by the tests).
- **Lengths (chars, p50 / p90 / p99 / max).** schema 379 / 2,447 / 2,879 / 5,404; question 79 /
  149 / 627 / 1,499; query 120 / 192 / 620 / 1,512; whole prompt 1,718 / 3,471 / 4,120 / 7,332.
  Examples in prompt: 49.8%.
- **Quirks.** The Cypher of 99.9% of the rows has a literal backslash-n between clauses instead of
  a newline (removed outside strings). Questions repeat with different golds (6,412 duplicate
  question texts, no duplicate pairs). Many golds `RETURN m` (whole nodes), which contradicts the
  prompt rule, hence the default drop. **The val split shares its graphs with train**: every val
  alias and 268 of the 300 val graph ids occur in train (the dataset's test is a random split of
  the same databases). Use `--val-by-db` for unseen graphs; with it 1% of the graph ids (about 8) are
  held out.

## cypherbench

- **What.** `megagonlabs/cypherbench`, revision `fd80c26a9f5447e4c9ba952cc70575544117306c`,
  Apache-2.0 (the data are derived from Wikidata, which is CC0). 10,882 (question, Cypher) pairs over
  11 Wikidata property graphs: 8,534 train over 4 graphs (art, biology, soccer, terrorist_attack) and
  2,348 test over 7 other graphs (movie, politics, fictional_character, geography, company, nba,
  flight_accident). Feng, Papicchio, Rahman, "CypherBench: Towards Precise Retrieval over
  Full-scale Modern Knowledge Graphs in the LLM Era", arXiv:2412.18702 (2024). Only the question
  files and the 11 small schema JSONs are downloaded (56 MB); the graph dumps (5 GB) are not.
- **Schema.** From `graphs/schemas/<graph>_schema.json`. Every node also has an implicit `name`
  property (the only property the gold queries use that is not in the schema JSON, 8,688 references),
  listed first; types `str / int / float / date / list[str]` map to
  unset / `INTEGER` / `FLOAT` / `DATE` / `LIST`.
- **Counts.** Nothing is dropped in train (8,534 of 8,534 pass); val: 11 of 2,348 are exact
  (question, query) duplicates (2 inside test, 9 also in train).
  Val sampled to 300 over the 7 test graphs, none seen in train.
- **Lengths (chars, p50 / p90 / p99 / max).** schema 667 / 693 / 693 / 693; question 102 / 137 /
  168 / 277; query 161 / 220 / 261 / 343; whole prompt 1,770 / 2,233 / 2,419 / 2,670.
  Examples in prompt: 49.8%.
- **Quirks.** The golds are template-generated, very regular, and 11% use
  `CALL { MATCH ... RETURN n UNION MATCH ... RETURN n } WITH DISTINCT n RETURN ...` (an
  uncorrelated subquery, read-only and valid in Neo4j 5, kept). Only 4 graphs in train means
  little schema variety (8.5k examples over 4 schemas); it is the most template-like of the four
  sets, so consider down-weighting it in the mix.

## Notes for the mix

- The two SQL sets and the two Cypher sets use the same prompt as the NFL generate prompt; the
  NFL-only rules (playoffs, the `Names` list) are not in the public prompt, `Names:` always says
  `(none found)`.
- SQaLe is the only set with real, large, messy schemas and varied question phrasing; weigh it
  highest for the "read the schema" behaviour. Gretel adds breadth but tiny schemas; text2cypher adds
  the Cypher schema formats; CypherBench adds regular multi-hop Cypher.
- A val loss over these 1,200 examples measures the format and the behaviour, not NFL accuracy:
  NFL val is scored by executing queries.
