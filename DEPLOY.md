# LLM-from-Scratch — Deployment Guide

## Quick Start

### 1. Local Setup
```bash
# Install conda environment + dependencies
bash setup.sh          # CPU
bash setup.sh --gpu    # GPU (CUDA 12.1)

conda activate llm_from_scratch
```

### 2. Train (full pipeline)
```bash
# Run all stages: Part2 → Part4 → SFT → RM → PPO
python run_pipeline.py --data part_2/tiny.txt --steps 500

# Or individual stages:
cd part_2 && python train.py --data tiny.txt --steps 500
cd part_4 && python train.py --data ../part_2/tiny.txt --bpe --steps 1000
```

### 3. API Server (local)
```bash
uvicorn api.server:app --reload --port 8000
```

API Endpoints:
| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/models` | List checkpoints |
| POST | `/generate` | Text generation |
| POST | `/generate/stream` | Streaming (SSE) |

**Example request:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time", "max_new_tokens": 100, "temperature": 0.8}'
```

---

## Docker

```bash
# Build & run locally
docker compose up --build

# Train only
docker compose --profile train up trainer

# Services:
#   API         → http://localhost:8000
#   TensorBoard → http://localhost:6006
```

---

## Cloud Deployment

### GCP Cloud Run (recommended — serverless)
```bash
bash deploy_cloud.sh gcp YOUR_GCP_PROJECT_ID
```

### AWS EC2 (GPU)
```bash
bash deploy_cloud.sh aws g4dn.xlarge my-keypair-name
```

### fly.io (simplest)
```bash
bash deploy_cloud.sh fly my-llm-api
```

---

## Project Structure

```
.
├── api/
│   └── server.py          ← FastAPI inference server
├── part_1/                ← Transformer fundamentals
├── part_2/                ← Tiny GPT training
├── part_3/                ← Modern architecture (RoPE, RMSNorm, SwiGLU)
├── part_4/                ← BPE, AMP, checkpointing
├── part_5/                ← Mixture-of-Experts
├── part_6/                ← Supervised Fine-Tuning
├── part_7/                ← Reward Modeling
├── part_8/                ← PPO RLHF
├── part_9/                ← GRPO RLHF
├── runs/                  ← Checkpoints (created at runtime)
├── Dockerfile
├── docker-compose.yml
├── run_pipeline.py        ← Full pipeline runner
├── setup.sh               ← Environment setup
└── deploy_cloud.sh        ← Cloud deployment
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT_DIR` | `./runs` | Where to look for `.pt` files |
| `PYTHONPATH` | auto | Set by setup.sh / Docker |
