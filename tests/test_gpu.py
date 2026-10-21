"""GPU checks on the real Spark-X2.6-4B (deselected by default: `pytest -m gpu`).

Batched, left-padded greedy generation must give what one-at-a-time generation gives (Spark's
RoPE comes from cache positions, so padding keeps every real token's relative offsets), and
the chunked attention must agree with the original on a long sequence.
"""

from __future__ import annotations

import pytest
import torch

from asktrainer.settings import get_paths

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def loaded():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    from asktrainer.modelio import load_base

    model, tok, info = load_base(get_paths().base_model)
    return model, tok, info


def _val_examples(n: int):
    from asktrainer.config import DataConfig
    from asktrainer.data.schema import read_jsonl

    d = DataConfig().nfl_dir(get_paths())
    exs = list(read_jsonl(d / "val.jsonl"))
    route = [e for e in exs if e.task == "route"][: n // 2]
    gen = [e for e in exs if e.task == "generate"][: n - len(route)]
    return route + gen


def test_load(loaded):
    model, _, info = loaded
    assert info["tied"]
    assert model.get_output_embeddings().weight.data_ptr() == (
        model.get_input_embeddings().weight.data_ptr()
    )


def test_batched_generation_equals_single(loaded):
    from asktrainer.chat import ChatTokenizer
    from asktrainer.evaluation import generate

    model, tok, _ = loaded
    model.eval()
    ct = ChatTokenizer(tok)
    exs = [e for e in _val_examples(8) if e.task == "route"][:4]
    prompts = [ct.prompt_ids(e) for e in exs]
    batched = generate(model, prompts, 48, ct.pad_id, ct.eos_id, "cuda", batch_size=4,
                       prefill_tokens=10**6)  # fmt: skip
    single = [generate(model, [p], 48, ct.pad_id, ct.eos_id, "cuda", batch_size=1)[0]
              for p in prompts]  # fmt: skip
    same = sum(b == s for b, s in zip(batched, single, strict=True))
    print("batched == single:", same, "/", len(prompts))
    for b, s in zip(batched, single, strict=True):
        print(repr(tok.decode(b)), "|", repr(tok.decode(s)))
    assert same == len(prompts)


def test_chunked_attention_is_within_bf16_noise(loaded):
    """Not bit-identical on the GPU (other GEMM row counts round differently), so the check
    is against a higher-precision reference (the same attention with fp32 matmuls): the
    chunked path must be no further from it than the original bf16 path (measured on a
    3,000-token NFL example: original 2.4%, chunked 2.1% relative distance; CE 0.4332 /
    0.4347 vs the reference's 0.4363)."""
    from asktrainer.attn_chunked import _module_of, use_chunked
    from asktrainer.chat import ChatTokenizer
    from asktrainer.modelio import backbone_of, head_weight

    model, tok, _ = loaded
    model.eval()
    ct = ChatTokenizer(tok)
    ex = min((e for e in _val_examples(400) if e.task == "generate"),
             key=lambda e: abs(len(ct.full_ids(e)[0]) - 3000))  # fmt: skip
    full, plen = ct.full_ids(ex)
    ids = torch.tensor([full]).cuda()
    bb, w = backbone_of(model), head_weight(model)

    def run():
        with torch.no_grad():
            h = bb(input_ids=ids, use_cache=False).last_hidden_state[0, plen - 1 : -1]
            ce = torch.nn.functional.cross_entropy(torch.nn.functional.linear(h, w).float(),
                                                   ids[0, plen:])  # fmt: skip
        return h.float(), float(ce)

    mod = _module_of(model)
    use_chunked(model, on=False)
    orig = mod.eager_attention_forward
    h_orig, ce_orig = run()

    def fp32_attention(module, q, k, v, attention_mask=None, scaling=None, dropout=0.0, **kw):
        m = None if attention_mask is None else attention_mask.float()
        out, _ = orig(module, q.float(), k.float(), v.float(), attention_mask=m,
                      scaling=scaling, dropout=dropout)  # fmt: skip
        return out.to(q.dtype), None

    mod.eager_attention_forward = fp32_attention
    h_ref, ce_ref = run()
    use_chunked(model, chunk=1024, threshold=1024)  # force chunking at 3k tokens
    h_chunk, ce_chunk = run()
    use_chunked(model, on=False)

    def rel(a, b):
        return float((a - b).norm() / a.norm())

    print("rel to fp32 ref: original", rel(h_ref, h_orig), "chunked", rel(h_ref, h_chunk))
    print("ce: ref", ce_ref, "original", ce_orig, "chunked", ce_chunk)
    assert rel(h_ref, h_chunk) <= 1.5 * rel(h_ref, h_orig)
    assert abs(ce_chunk - ce_ref) <= 1.5 * abs(ce_orig - ce_ref) + 1e-3
