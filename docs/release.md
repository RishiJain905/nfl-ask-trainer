# The release (`Spark-X2.6-4B-SQL-Cypher-Agent`)

What was published from the real run (`docs/real-run.md`), how it was made and measured, and how
to redo it. Rishi chose the adapter (step 1,500 of `sft-main`) on 2026-10-31.

## On Hugging Face

| Repo | Visibility | What |
|---|---|---|
| [trjxter/Spark-X2.6-4B-SQL-Cypher-Agent](https://huggingface.co/trjxter/Spark-X2.6-4B-SQL-Cypher-Agent) | public | the merged bf16 model (5 shards, 8.2 GB), remote code, thinking on by default |
| [trjxter/Spark-X2.6-4B-SQL-Cypher-Agent-GGUF](https://huggingface.co/trjxter/Spark-X2.6-4B-SQL-Cypher-Agent-GGUF) | public | Q8_0, Q6_K_XL, Q6_K, Q4_K_XL, Q4_K_M and the importance matrix |
| [trjxter/Spark-X2.6-4B-SQL-Cypher-Agent-LoRA](https://huggingface.co/trjxter/Spark-X2.6-4B-SQL-Cypher-Agent-LoRA) | private | the adapter (PEFT, r16 / alpha 32), the start for RL |

Each repo has a model card (`README.md` in its release folder). The bf16 card is the main one:
results, prompt format, training, limitations, licences.

**Licence:** Apache-2.0, like the base (`XHToken/Spark-X2.6-4B`). Training data: SQaLe (MIT),
gretel, text2cypher and CypherBench (Apache-2.0). Attributions are in the bf16 card.

## Steps

All the files are under `D:\nfl-ml-data\ask\release\`, one folder per repo. Nothing in those
folders points back to this PC; the last section explains how that was checked.

1. **Merge.** Run in WSL:
   ```
   ask-trainer merge --run sft-main --adapter adapter_step001500 --thinking --out <release>/Spark-X2.6-4B-SQL-Cypher-Agent
   ```
   `--thinking` keeps the base's chat template and generation config. Without it, the merge
   writes the thinking-off template and a greedy config, which suits a model served for the NFL
   repo. `ask_model_card.json` names the base by its Hub id, and it stays local.
2. **LoRA folder.** `adapter_model.safetensors` and `adapter_config.json` from the run's
   `adapter_step001500`, with `base_model_name_or_path` set to `XHToken/Spark-X2.6-4B`.
3. **GGUFs.** `powershell -File scripts\llamacpp\release_quants.ps1 -Step <step>`, in this
   order:
   - `convert`: the BF16 GGUF, with upstream's b11779 converter;
   - `calib`: `make_calib_text.py`, 40 training examples per origin from both training sets,
     rendered as training saw them, 241,966 tokens;
   - `imatrix`;
   - `quantize`;
   - `kldbase` and `kld`: against BF16, on the held-out validation text `kld-domain.txt`, never
     the calibration text;
   - `bench`.

   The work files (BF16 GGUF, `imatrix.gguf`, `calib.txt`, logs) are in
   `D:\nfl-ml-data\ask\quant\spark-x2.6-4b-sql-cypher-agent\`.
4. **Scores.** `C:\Python314\python.exe scripts\release_eval.py`:
   - every GGUF on the evalset;
   - `--testset --only Q8_0` for the test set;
   - `--testset --gguf <base Q8_0> --label base-Q8_0` for the base model on the same test.

   It serves each file with `serve.ps1 -Model` (thinking budget 1,024, greedy) and scores it
   with checker 4, plus checker 3 beside it. Results go to `release\eval\{eval,test}.json` and
   into `sft-main`'s W&B run as `release-eval-<quant>` and `release-test-<quant>`. Needs
   `asktrainer-neo4j` running.
5. **Metadata clean-up.** See the last section.
6. **Upload.** `C:\Python314\python.exe scripts\release_upload.py` (Windows Python is logged in
   to the Hub):
   - creates the three repos, refusing one that exists with the other visibility;
   - uploads the LoRA folder with `upload_folder`;
   - uploads the bf16 and GGUF folders with `upload_large_folder`, which resumes after an
     interruption when run again.

   It's slow enough (~8 MB/s, about an hour) to start detached via WMI.

## Results

**Test set** (1,000 questions; checker 4; the scored count is in brackets):

| Model | Overall | CypherBench | gretel | SQaLe | text2cypher | Well-formed call | Parses |
|---|---|---|---|---|---|---|---|
| base Spark-X2.6-4B, Q8_0 | 32.5% (293/902) | 22.4 | 55.3 | 36.6 | 12.4 | 95.3% | 92.2% |
| adapter step 1500 on bf16 | 43.9% (401/913) | 37.6 | 62.9 | 42.5 | 29.2 | 98.6% | 96.9% |
| **merged, Q8_0** | **43.7%** (393/899) | 35.5 | 64.6 | 43.2 | 27.7 | 97.8% | 95.8% |

- **Merged Q8_0 against the base,** on the 856 questions both runs could score: 129 right only
  for the merged model, 37 only for the base. Sign test p ≈ 4e-13; +10.7 points, 95% range ±2.9.
- **Merging and Q8_0 lose nothing** that this test can see.

**The quants:**

| Quant | Size | KLD vs BF16 | Same top token | Evalset (200) | tok/s |
|---|---|---|---|---|---|
| Q8_0 | 4.38 GB | 0.0005 | 99.45% | 43.2% | 90 |
| Q6_K_XL | 3.69 GB | 0.0016 | 99.08% | 44.5% | 105 |
| Q6_K | 3.38 GB | 0.0026 | 98.81% | 41.5% | 119 |
| Q4_K_XL | 2.85 GB | 0.0119 | 97.36% | 42.7% | 126 |
| Q4_K_M | 2.60 GB | 0.0224 | 96.51% | 42.5% | 138 |

- The adapter itself scored 45.4% on the evalset, and BF16 runs at 58 tok/s.
- On 200 questions, ±3.5 points is noise, so the KLD column ranks the quants, not the evalset.
- **XL** is our own recipe, not Unsloth's UD: the tied embedding / output head at Q8_0, and every
  attention matrix one step up (Q6_K to Q8_0, Q4_K_M to Q6_K). For Q4 this halves the KLD for
  250 MB.

## Nothing that points back to this PC

Every file in the three folders was swept before the upload for:

- drive paths (`X:\`, `X:/`), `/mnt/`, `/home/`;
- the Windows user name, the data folder and repo names, the email.

What turned up and what was done:

- **llama-quantize's K-quants** recorded where their imatrix and calibration text were, in
  `quantize.imatrix.file` and `quantize.imatrix.dataset`. Both fields were removed with gguf-py's
  `gguf_new_metadata.py --remove-metadata`; this changes no weights. The calibration itself is in
  the weights; `entries_count` and `chunks_count` stay.
- **`imatrix.gguf`** recorded the calibration text's full path in `imatrix.datasets`. The
  published copy says `calib.txt` instead. Its tensors are byte for byte the work file's.
  - Block 0's `stats` tensor holds a few NaNs: summary statistics such as skewness that
    `llama-quantize` doesn't read. `in_sum2` and `counts` are clean.
- **Safetensors headers** hold only `{"format": "pt"}`. The bf16 configs, the tokenizer, the
  template and the remote code are the base's, with no paths in them.
- **`ask_model_card.json`** (the trainer's record of the merge) is not uploaded.

Before any future upload, sweep the same way. A new GGUF made with `--imatrix` carries these
two fields again.
