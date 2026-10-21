#!/usr/bin/env bash
# Creates the training venv inside WSL Ubuntu (ROCm 7.2, RX 7800 XT / gfx1101).
#
#   bash scripts/setup_wsl.sh            # from the repo folder, inside WSL
#
# Why WSL: AMD's ROCm PyTorch runs on the RX 7800 XT there (the /dev/dxg bridge + /opt/rocm 7.2),
# proven by the ew3 venv. The venv lives on WSL's own disk (fast for many small files); the repo
# stays on F: and is installed editable. The torch / triton wheels are AMD's "lw" builds: they use
# the system's /opt/rocm HSA runtime (no bundled libhsa), which is what WSL needs.
set -euo pipefail

VENV="${ASK_TRAINER_VENV:-$HOME/venvs/nfl-ask-trainer}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WHEELS="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2"
TORCH="torch-2.9.1%2Brocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl"
TRITON="triton-3.5.1%2Brocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl"

echo "venv: $VENV"
echo "repo: $REPO_DIR"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip wheel setuptools
"$VENV/bin/pip" install "$WHEELS/$TORCH" "$WHEELS/$TRITON"
"$VENV/bin/pip" install -e "$REPO_DIR[dev]"
"$VENV/bin/pip" freeze --exclude-editable > "$REPO_DIR/requirements-lock.txt"

"$VENV/bin/python" - <<'EOF'
import torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("gpu", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
EOF
