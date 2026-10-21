"""The training example: the one record every data source writes and the trainer reads.

This is the contract between the NFL repo's exporter (`nfl ask train-data`, which writes the
NFL examples and can't import this package) and this repo (the public-data adapters, the mix,
the trainer, the per-epoch verification). The NFL side mirrors these fields exactly; a test on
each side pins them. One JSON object per line (`*.jsonl`, UTF-8).

Fields:
- `id`: unique across a mix (`nfl-route-<n>`, `sqale-<n>`, ...).
- `origin`: where it came from: `nfl` (our schema, our templates) or a public dataset
  (`sqale`, `gretel`, `cypherbench`, `text2cypher`).
- `task`: `route` (pick a template / generate / refuse: the router's JSON), `generate` (write
  one query: `{"query": ...}`) or `call` (think, then call a query tool: Spark's native format;
  the assistant turn has `reasoning_content` and exactly one `tool_calls` entry, `tools` holds
  the tool definitions and `thinking` the level the reasoning was made at).
- `split`: `train` or `val`. NFL val examples are scored by executing them (the verifier);
  public val examples only feed the validation loss.
- `group`: the reporting slice inside a split (`seen`: new phrasings of training templates;
  `unseen_template`: templates never trained on; `refuse`; `public`; ...).
- `messages`: the chat, exactly as the model sees it at inference: `system`, `user`, then the
  target `assistant` turn (always last). The loss is on the assistant turn only.
- `jepa`: the two LLM-JEPA views: `text` (the question, as a person wrote it) and `code` (the
  target the model must produce). Both short; the predictor works from `text` alone.
- `dialect`: the target language (`duckdb`, `sqlite`, `cypher`), or None for a route decision.
- `verify`: what the NFL verifier needs to score a prediction by running it (NFL only).
- `meta`: anything else (template id, filled values, skeleton id, db id, ...); never trained on.
- `tools`, `thinking`: `call` examples only (above); None otherwise.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ORIGINS = ("nfl", "sqale", "gretel", "cypherbench", "text2cypher")
TASKS = ("route", "generate", "call")
THINKING = ("high", "low", "off")
SPLITS = ("train", "val")
DIALECTS = ("duckdb", "sqlite", "cypher")
ROLES = ("system", "user", "assistant")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = ""
    reasoning_content: str | None = None  # `call` targets: the thought
    tool_calls: list[dict[str, Any]] | None = None  # `call` targets: one query-tool call


class JepaViews(BaseModel):
    text: str = Field(min_length=1)
    code: str = Field(min_length=1)


class Example(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    origin: Literal["nfl", "sqale", "gretel", "cypherbench", "text2cypher"]
    task: Literal["route", "generate", "call"]
    split: Literal["train", "val"]
    group: str = "train"
    messages: list[Message]
    jepa: JepaViews
    dialect: Literal["duckdb", "sqlite", "cypher"] | None = None
    verify: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    tools: list[dict[str, Any]] | None = None
    thinking: Literal["high", "low", "off"] | None = None

    @field_validator("messages")
    @classmethod
    def _shape(cls, v: list[Message]) -> list[Message]:
        if len(v) < 2 or v[-1].role != "assistant":
            raise ValueError("messages must end with the target assistant turn")
        if any(m.role == "assistant" for m in v[:-1]):
            raise ValueError("only the last message may be the assistant's (single-turn)")
        if not v[-1].content.strip() and not v[-1].tool_calls:
            raise ValueError("the target is empty")
        return v

    @model_validator(mode="after")
    def _target_is_json(self) -> Example:
        if self.task == "call":
            return self._check_call()
        try:
            obj = json.loads(self.messages[-1].content)
        except json.JSONDecodeError as e:
            raise ValueError(f"the target is not JSON ({e.msg})") from None
        if not isinstance(obj, dict):
            raise ValueError("the target must be one JSON object")
        if self.task == "generate" and not (isinstance(obj.get("query"), str) or "refuse" in obj):
            raise ValueError('a generate target is {"query": ...} (or a refusal)')
        if self.task == "route" and obj.get("action") not in ("template", "generate", "refuse"):
            raise ValueError("a route target needs action template | generate | refuse")
        return self

    def _check_call(self) -> Example:
        last = self.messages[-1]
        if not self.tools or not last.tool_calls or len(last.tool_calls) != 1:
            raise ValueError("a call target has the tools and exactly one tool call")
        fn = last.tool_calls[0].get("function") or {}
        args = fn.get("arguments")
        names = {t.get("function", {}).get("name") for t in self.tools}
        if fn.get("name") not in names or not isinstance(args, dict) or len(args) != 1:
            raise ValueError("the call names one of the tools with one argument (a dict)")
        if not isinstance(next(iter(args.values())), str) or not next(iter(args.values())).strip():
            raise ValueError("the call's argument is the query text")
        if self.thinking is None:
            raise ValueError("a call example says its thinking level")
        if (self.thinking == "off") == bool((last.reasoning_content or "").strip()):
            raise ValueError("reasoning_content is required with thinking on, absent with it off")
        return self

    @property
    def target(self) -> str:
        """The answer text: the query of a `call`, else the assistant turn's content."""
        if self.task == "call":
            return next(iter(self.messages[-1].tool_calls[0]["function"]["arguments"].values()))
        return self.messages[-1].content

    @property
    def prompt_messages(self) -> list[dict[str, Any]]:
        return [m.model_dump(exclude_none=True) for m in self.messages[:-1]]

    @property
    def chat_messages(self) -> list[dict[str, Any]]:
        """The whole chat as the template takes it (reasoning and tool calls only when set)."""
        return [m.model_dump(exclude_none=True) for m in self.messages]


def read_jsonl(path: Path) -> Iterator[Example]:
    with Path(path).open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if line.strip():
                try:
                    yield Example.model_validate_json(line)
                except Exception as e:  # name the line: a bad record is the writer's bug
                    raise ValueError(f"{path}:{i}: {e}") from None


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(ex.model_dump_json(exclude_none=False) + "\n")
            n += 1
    tmp.replace(path)
    return n
