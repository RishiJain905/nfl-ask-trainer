"""load_base (sequential shard reads onto the meta-built model), save_adapter and
merge_and_save, on a tiny Llama written to disk (CPU)."""

from __future__ import annotations

import json

import pytest
import torch
from conftest import TEMPLATE, example
from test_train_smoke import LLAMA_TARGETS, tiny_llama

from asktrainer.chat import ChatTokenizer
from asktrainer.config import LoraSpec
from asktrainer.modelio import (
    apply_lora,
    load_base,
    merge_and_save,
    no_thinking_template,
    save_adapter,
)


@pytest.fixture
def base_dir(tmp_path, tiny_tok):
    d = tmp_path / "base"
    m = tiny_llama(tiny_tok)
    m.save_pretrained(str(d), safe_serialization=True, max_shard_size="100KB")  # several shards
    tiny_tok.save_pretrained(str(d))
    (d / "modeling_dummy.py").write_text("# remote code stand-in\n")
    return d, m


def test_load_base_matches_the_saved_model(base_dir):
    d, ref = base_dir
    assert len(list(d.glob("model-*.safetensors"))) > 1
    model, tok, info = load_base(d, device="cpu", dtype=torch.float32, log=lambda *_: None)
    assert info["tied"]
    sd, rsd = model.state_dict(), ref.state_dict()
    assert sd.keys() == rsd.keys()
    for k in sd:
        assert torch.equal(sd[k], rsd[k]), k
    ids = torch.tensor([[5, 6, 7, 8]])
    with torch.no_grad():
        assert torch.allclose(model(input_ids=ids).logits, ref(input_ids=ids).logits)


def test_no_thinking_template():
    out = no_thinking_template(TEMPLATE)
    assert "default(false)" in out and "default(true)" not in out
    with pytest.raises(ValueError):
        no_thinking_template("{{ messages }}")


def test_merge_and_save(base_dir, tmp_path, tiny_tok):
    d, _ = base_dir
    model, tok, _ = load_base(d, device="cpu", dtype=torch.float32, log=lambda *_: None)
    pm = apply_lora(model, LoraSpec(r=4, alpha=8, dropout=0.0, targets=LLAMA_TARGETS),
                    gradient_checkpointing=False)  # fmt: skip
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    pm.eval()
    adapter = save_adapter(pm, tmp_path / "adapter")
    ids = torch.tensor([ChatTokenizer(tiny_tok).prompt_ids(example(1))])
    with torch.no_grad():
        want = pm(input_ids=ids).logits
    out = merge_and_save(adapter, tmp_path / "merged", d, card={"run": "r1", "objective": "plain"},
                         device="cpu", log=lambda *_: None, dtype=torch.float32)  # fmt: skip
    assert (out / "modeling_dummy.py").exists()
    card = json.loads((out / "ask_model_card.json").read_text())
    assert card["run"] == "r1" and card["objective"] == "plain"
    tpl = (out / "chat_template.jinja").read_text()
    assert "default(false)" in tpl
    tc = json.loads((out / "tokenizer_config.json").read_text())
    assert "default(false)" in tc["chat_template"]
    assert json.loads((out / "generation_config.json").read_text())["do_sample"] is False
    merged, mtok, _ = load_base(out, device="cpu", dtype=torch.float32, log=lambda *_: None)
    msgs = example(1).prompt_messages
    assert mtok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True).endswith(
        "<|Bot|></think>"
    )  # served without the flag = no thinking
    with torch.no_grad():
        got = merged(input_ids=ids).logits
    assert torch.allclose(got, want, atol=1e-4)
