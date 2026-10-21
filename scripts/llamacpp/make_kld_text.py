"""Write a domain text for the BF16-vs-quant KL-divergence check.

Takes the first N validation examples of each public set (held out from training), renders each
conversation (schema + question + gold query) with Spark's own chat template and joins them.
Run inside WSL with the trainer venv (2026-10-20: 60 per set, 147,968 tokens):
    python scripts/llamacpp/make_kld_text.py <model_dir> <public_dir> <out_txt> [per_set]
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

model_dir, public_dir, out_txt = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
per_set = int(sys.argv[4]) if len(sys.argv) > 4 else 60
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

parts, counts, n_tokens = [], {}, 0
for name in ["sqale", "gretel", "text2cypher", "cypherbench"]:
    text = (public_dir / name / "examples_val.jsonl").read_text(encoding="utf-8")
    rows = text.splitlines()[:per_set]
    for line in rows:
        ex = json.loads(line)
        text = tok.apply_chat_template(ex["messages"], tokenize=False, enable_thinking=False)
        parts.append(text)
        n_tokens += len(tok(text, add_special_tokens=False)["input_ids"])
    counts[name] = len(rows)

out_txt.parent.mkdir(parents=True, exist_ok=True)
out_txt.write_text("\n\n".join(parts), encoding="utf-8")
print(json.dumps({"out": str(out_txt), "examples": counts, "tokens": n_tokens}))
