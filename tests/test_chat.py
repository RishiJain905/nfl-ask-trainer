from __future__ import annotations

import pytest
import torch
from conftest import EOS, example, spark_dir

from asktrainer.chat import (
    ChatTokenizer,
    collate,
    left_pad,
    length_grouped_batches,
    predictor_tokens,
)


def _check_masking(ct: ChatTokenizer, ex) -> None:
    f = ct.features(ex)
    prompt = ct.tok.decode(f.input_ids[: f.prompt_len])
    target = ct.tok.decode(f.input_ids[f.prompt_len :])
    assert prompt.endswith("<|Bot|></think>")
    assert target == ex.target + EOS  # the loss covers the content and its end token only
    assert f.input_ids[-1] == ct.eos_id
    b = collate([f], ct.pad_id)
    # hidden state t predicts token t+1: positions prompt_len-1 .. len-2
    assert b.target_index[:, 1].tolist() == list(range(f.prompt_len - 1, len(f.input_ids) - 1))
    assert b.target_ids.tolist() == f.input_ids[f.prompt_len :]
    assert ct.prompt_ids(ex) == f.input_ids[: f.prompt_len]  # the prompt is a token prefix


def test_masking_tiny(tiny_tok):
    ct = ChatTokenizer(tiny_tok, max_len=4096)
    for task in ("route", "generate"):
        _check_masking(ct, example(3, task))


def test_no_thinking_rendering(tiny_tok):
    ct = ChatTokenizer(tiny_tok)
    ex = example(1)
    prompt = ct.render(ex.prompt_messages, add_generation_prompt=True)
    assert prompt.endswith("<|Bot|></think>") and "<think>" not in prompt


def test_jepa_views_tiny(tiny_tok):
    ct = ChatTokenizer(tiny_tok, jepa=True, k=2, last_token=-2)
    ex = example(2)
    f = ct.features(ex)
    text = ct.tok.decode(f.text_ids)
    assert ex.jepa.text in text and "System" in text and "<|Bot|>" not in text
    assert text.endswith("<||unused3||><||unused2||>" + EOS)  # predictor_2, predictor_1
    assert ct.tok.convert_ids_to_tokens(f.text_ids[f.text_pos]) == "<||unused2||>"
    code = ct.tok.decode(f.code_ids)
    assert code.endswith("<|Bot|></think>" + ex.jepa.code + EOS)
    assert ct.tok.decode([f.code_ids[f.code_pos]]) == ex.jepa.code[-1]  # the closing "}"
    b = collate([f], ct.pad_id, with_views=True)
    assert b.segments == [[len(f.input_ids), len(f.text_ids), len(f.code_ids)]]
    row = b.input_ids[0].tolist()
    assert row[b.text_index[0, 1]] == f.text_ids[f.text_pos]
    assert row[b.code_index[0, 1]] == f.code_ids[f.code_pos]
    # positions restart per segment; targets are only in the main sequence
    n = len(f.input_ids)
    assert b.position_ids[0, n].item() == 0 and b.position_ids[0, n - 1].item() == n - 1
    assert int(b.target_index[:, 1].max()) < n


def test_k0_uses_last_text_token(tiny_tok):
    ct = ChatTokenizer(tiny_tok, jepa=True, k=0)
    f = ct.features(example(4))
    assert ct.tok.decode([f.text_ids[f.text_pos]]) == "?"


def test_predictor_order():
    assert predictor_tokens(3) == ["<||unused4||>", "<||unused3||>", "<||unused2||>"]
    assert predictor_tokens(0) == []
    with pytest.raises(ValueError):
        predictor_tokens(11)


def test_max_len_drops_and_counts(tiny_tok):
    ct = ChatTokenizer(tiny_tok, max_len=200)
    exs = [example(i) for i in range(4)]
    lens = {e.id: len(ChatTokenizer(tiny_tok).features(e).input_ids) for e in exs}
    kept, dropped = ct.tokenize_all(exs)
    assert sorted(dropped) == sorted(i for i, n in lens.items() if n > 200)
    assert len(kept) + len(dropped) == 4


def test_collate_padding(tiny_tok):
    ct = ChatTokenizer(tiny_tok)
    fs = [ct.features(example(i)) for i in (0, 3)]
    b = collate(fs, ct.pad_id)
    L = max(len(f.input_ids) for f in fs)
    assert b.input_ids.shape == (2, L)
    assert b.attention_mask.sum(1).tolist() == [len(f.input_ids) for f in fs]
    assert b.n_target == sum(f.n_target for f in fs)


def test_left_pad():
    ids, att = left_pad([[5, 6, 7], [8]], pad_id=0)
    assert ids.tolist() == [[5, 6, 7], [0, 0, 8]]
    assert att.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_length_grouped_batches():
    lengths = [((i * 37) % 101) + 1 for i in range(50)]
    a = length_grouped_batches(lengths, 4, seed=1, mega=2)
    assert a == length_grouped_batches(lengths, 4, seed=1, mega=2)
    assert a != length_grouped_batches(lengths, 4, seed=2, mega=2)
    flat = sorted(i for b in a for i in b)
    assert flat == list(range(50))
    assert max(lengths[i] for i in a[0]) == max(lengths)
    # within a mega-batch, batches are length-sorted: spreads are small
    assert (
        torch.tensor([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in a])
        .float()
        .mean()
        < 40
    )


# -- the real tokenizer (skipped when the model folder isn't there) -----------------------
@pytest.fixture(scope="module")
def spark_tok():
    d = spark_dir()
    if d is None:
        pytest.skip("the Spark-X2.6 model folder is not available")
    from asktrainer.modelio import load_tokenizer

    return load_tokenizer(d)


def test_masking_real(spark_tok):
    ct = ChatTokenizer(spark_tok, jepa=True, k=3)
    for task in ("route", "generate"):
        ex = example(5, task)
        _check_masking(ct, ex)
        f = ct.features(ex)
        full = ct.render([m.model_dump() for m in ex.messages])
        assert full.startswith("<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.")
        assert ct.tok.decode(f.text_ids[f.text_pos]) == "<||unused2||>"
        assert f.code_ids[-1] == ct.eos_id
        assert ct.tok.decode([f.code_ids[f.code_pos]]).endswith("}")


def test_real_predictor_tokens_are_single(spark_tok):
    ChatTokenizer(spark_tok, jepa=True, k=10)  # raises if any is not one token


def test_no_thinking_template_real(spark_tok):
    from asktrainer.modelio import no_thinking_template

    msgs = example(1).prompt_messages
    patched = no_thinking_template(spark_tok.chat_template)
    default = spark_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            chat_template=patched)  # fmt: skip
    explicit = spark_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)  # fmt: skip
    assert default == explicit and default.endswith("<|Bot|></think>")
