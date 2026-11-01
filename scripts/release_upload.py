"""Upload the release to Hugging Face (docs/release.md), from Windows Python, already logged in:

    C:\\Python314\\python.exe scripts\\release_upload.py [--only bf16,gguf,lora]

Three repos under the logged-in account: the merged bf16 model (public), the GGUFs as one folder
(public), the LoRA adapter (private). Each is created if missing; an existing repo whose
visibility differs stops the upload instead of being changed. The big folders go through
`upload_large_folder`, which resumes after an interruption (run it again). The trainer's own
record (`ask_model_card.json`) stays local.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

NAME = "Spark-X2.6-4B-SQL-Cypher-Agent"
RELEASE = Path(r"D:\nfl-ml-data\ask\release")
REPOS = {  # key: (folder, repo suffix, private)
    "lora": (f"{NAME}-LoRA", "-LoRA", True),
    "bf16": (NAME, "", False),
    "gguf": (f"{NAME}-GGUF", "-GGUF", False),
}
IGNORE = ["ask_model_card.json", ".cache/**", "*.tmp"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated: bf16, gguf, lora (default: all)")
    args = ap.parse_args()
    only = {k for k in args.only.split(",") if k}
    api = HfApi()
    user = api.whoami()["name"]
    for key, (folder, suffix, private) in REPOS.items():
        if only and key not in only:
            continue
        repo = f"{user}/{NAME}{suffix}"
        api.create_repo(repo, private=private, exist_ok=True)
        if api.repo_info(repo).private != private:
            print(f"{repo}: exists with the other visibility; not uploading", flush=True)
            return 1
        print(f"{repo}: uploading {folder} ({'private' if private else 'public'})", flush=True)
        path = RELEASE / folder
        if key == "lora":
            api.upload_folder(repo_id=repo, folder_path=path, ignore_patterns=IGNORE,
                              commit_message="LoRA adapter (sft-main, step 1500)")  # fmt: skip
        else:
            api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=path,
                                    ignore_patterns=IGNORE)  # fmt: skip
        files = api.list_repo_files(repo)
        print(f"{repo}: done, {len(files)} files: {sorted(files)}", flush=True)
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
