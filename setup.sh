#!/usr/bin/env bash
# ─────────────────────────────────────────────────────
# setup.sh  —  Environment setup for LLM-from-Scratch
# Usage:  bash setup.sh [--gpu]
# ─────────────────────────────────────────────────────
set -euo pipefail

GPU=false
for arg in "$@"; do [[ "$arg" == "--gpu" ]] && GPU=true; done

echo "════════════════════════════════════════"
echo "  LLM-from-Scratch · Environment Setup  "
echo "════════════════════════════════════════"

# ── 1. Conda environment ──────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "[error] conda not found. Install Miniconda first: https://docs.conda.io"
  exit 1
fi

ENV_NAME="llm_from_scratch"
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[skip] conda env '${ENV_NAME}' already exists"
else
  echo "[step] Creating conda env '${ENV_NAME}' (Python 3.11)..."
  conda create -y -n "$ENV_NAME" python=3.11
fi

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 2. Install dependencies ───────────────────────────
echo "[step] Installing Python packages..."
pip install --upgrade pip

if [[ "$GPU" == "true" ]]; then
  echo "  → GPU mode (CUDA 12.1)"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "  → CPU mode"
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt
pip install fastapi "uvicorn[standard]" pydantic

# ── 3. Create directory structure ────────────────────
echo "[step] Creating output directories..."
mkdir -p runs/{part2,part4,sft-demo,rm-demo,ppo-demo,grpo-demo}
mkdir -p logs

# ── 4. Verify imports ─────────────────────────────────
echo "[step] Verifying PyTorch..."
python -c "import torch; print(f'  PyTorch {torch.__version__}  CUDA={torch.cuda.is_available()}')"

echo ""
echo "✅  Setup complete!"
echo ""
echo "Next steps:"
echo "  conda activate ${ENV_NAME}"
echo "  # Train a tiny model:"
echo "  cd part_2 && python train.py --data tiny.txt --steps 500"
echo "  # Or Part 4 with BPE:"
echo "  cd part_4 && python train.py --data ../part_2/tiny.txt --bpe --steps 1000"
echo "  # Start API server:"
echo "  uvicorn api.server:app --reload"
