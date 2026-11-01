"""Write a calibration text for llama.cpp's importance matrix (the 4- and 6-bit release quants).

Takes the first N *training* examples of each origin from the sets the model was trained on
(traces-pilot-full: the model's own checked thoughts + calls; gold-full: thinking-off calls) and
renders each exactly as training did (`ChatTokenizer.render`). Training examples only, so the
fidelity check on held-out validation text (`make_kld_text.py`) stays fair.
Run inside WSL with the trainer venv:
    python scripts/llamacpp/make_calib_text.py <model_dir> <public_dir> <out_txt> [per_origin]
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

from asktrainer.chat import ChatTokenizer
from asktrainer.data.schema import Example

model_dir, public_dir, out_txt = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
per_origin = int(sys.argv[4]) if len(sys.argv) > 4 else 40
ct = ChatTokenizer(AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True))

parts, counts, n_tokens = [], {}, 0
for set_name in ["traces-pilot-full", "gold-full"]:
    taken: dict[str, int] = {}
    for line in (public_dir / set_name / "examples_train.jsonl").open(encoding="utf-8"):
        ex = Example.model_validate_json(line)
        if taken.get(ex.origin, 0) >= per_origin:
            continue
        taken[ex.origin] = taken.get(ex.origin, 0) + 1
        text = ct.render(ex.chat_messages, ex=ex)
        parts.append(text)
        n_tokens += len(ct.encode(text))
    counts[set_name] = taken

out_txt.parent.mkdir(parents=True, exist_ok=True)
out_txt.write_text("\n\n".join(parts), encoding="utf-8")
print(json.dumps({"out": str(out_txt), "examples": counts, "tokens": n_tokens}))
