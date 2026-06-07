"""
api/server.py  —  FastAPI inference endpoint for LLM-from-Scratch
─────────────────────────────────────────────────────────────────
Endpoints
  GET  /health              → liveness check
  GET  /models              → list available checkpoints
  POST /generate            → text generation
  POST /generate/stream     → streaming generation (SSE)
"""

import os, glob, time, asyncio
from pathlib import Path
from typing import Optional, List, Iterator

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Lazy-import the model (so server starts even without checkpoints) ──
import sys
_HERE = Path(__file__).resolve().parent.parent
for sub in ("part_3", "part_4", "part_6"):
    sys.path.insert(0, str(_HERE / sub))

try:
    from model_modern import GPTModern
    from tokenizer_bpe import BPETokenizer
    from tokenizer import ByteTokenizer          # fallback
    _MODEL_CLS_OK = True
except ImportError as e:
    print(f"[warn] Could not import model classes: {e}")
    _MODEL_CLS_OK = False

# ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LLM-from-Scratch API",
    description="Inference API for GPTModern checkpoints",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CHECKPOINT_DIR = Path(os.getenv("CHECKPOINT_DIR", _HERE / "runs"))
_loaded: dict = {}   # cache: ckpt_path → {model, tok, cfg}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ───────────────────────────────────────────────────────────

def _find_checkpoints() -> List[str]:
    """Recursively find all model_*.pt files under CHECKPOINT_DIR."""
    return sorted(glob.glob(str(CHECKPOINT_DIR / "**" / "model_*.pt"), recursive=True))


def _load_model(ckpt_path: str):
    """Load and cache a model from a checkpoint path."""
    if ckpt_path in _loaded:
        return _loaded[ckpt_path]

    if not _MODEL_CLS_OK:
        raise RuntimeError("Model classes not importable. Check PYTHONPATH.")

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    vocab_size = cfg.get("vocab_size", 256)
    block_size = cfg.get("block_size", 256)

    # Build model
    model = GPTModern(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=cfg.get("n_layer", 4),
        n_head=cfg.get("n_head", 4),
        n_embd=cfg.get("n_embd", 256),
        dropout=0.0,
        use_rmsnorm=cfg.get("use_rmsnorm", True),
        use_swiglu=cfg.get("use_swiglu", True),
        rope=cfg.get("rope", True),
    ).to(DEVICE).eval()
    model.load_state_dict(ckpt["model"])

    # Tokenizer: prefer BPE saved alongside checkpoint
    tok_dir_file = p.with_name("tokenizer_dir.txt")
    tok = None
    if tok_dir_file.exists():
        try:
            tok = BPETokenizer(vocab_size=vocab_size)
            tok.load(tok_dir_file.read_text().strip())
        except Exception as e:
            print(f"[warn] BPE load failed: {e}")
            tok = None
    if tok is None:
        tok = ByteTokenizer()

    _loaded[ckpt_path] = {"model": model, "tok": tok, "cfg": cfg, "block_size": block_size}
    print(f"[api] Loaded model from {ckpt_path}  device={DEVICE}")
    return _loaded[ckpt_path]


def _encode(tok, text: str) -> List[int]:
    ids = tok.encode(text)
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return ids


def _decode(tok, ids: List[int]) -> str:
    if hasattr(tok, "decode"):
        return tok.decode(ids)
    return bytes(ids).decode("utf-8", errors="ignore")


# ── Pydantic schemas ──────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text / instruction")
    max_new_tokens: int = Field(200, ge=1, le=2048)
    temperature: float = Field(1.0, ge=0.01, le=4.0)
    top_k: Optional[int] = Field(50, ge=1)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    checkpoint: Optional[str] = Field(None, description="Relative or absolute path to .pt file")


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    generated_tokens: int
    elapsed_s: float
    checkpoint: str
    device: str


# ── Routes ────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "cuda": torch.cuda.is_available()}


@app.get("/models")
def list_models():
    ckpts = _find_checkpoints()
    return {"checkpoints": ckpts, "count": len(ckpts)}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Pick checkpoint
    ckpts = _find_checkpoints()
    if req.checkpoint:
        ckpt_path = str(Path(req.checkpoint).resolve() if Path(req.checkpoint).is_absolute()
                        else CHECKPOINT_DIR / req.checkpoint)
    elif ckpts:
        ckpt_path = ckpts[-1]   # latest
    else:
        raise HTTPException(404, "No checkpoints found. Train a model first.")

    entry = _load_model(ckpt_path)
    model, tok, block_size = entry["model"], entry["tok"], entry["block_size"]

    ids = _encode(tok, req.prompt)
    idx = torch.tensor([ids[-block_size:]], dtype=torch.long, device=DEVICE)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
    elapsed = time.time() - t0

    out_ids = out[0].tolist()
    full_text = _decode(tok, out_ids)

    return GenerateResponse(
        text=full_text,
        prompt_tokens=len(ids),
        generated_tokens=len(out_ids) - len(ids),
        elapsed_s=round(elapsed, 3),
        checkpoint=ckpt_path,
        device=str(DEVICE),
    )


@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Server-Sent Events streaming generation."""
    ckpts = _find_checkpoints()
    if req.checkpoint:
        ckpt_path = str(CHECKPOINT_DIR / req.checkpoint)
    elif ckpts:
        ckpt_path = ckpts[-1]
    else:
        raise HTTPException(404, "No checkpoints found.")

    entry = _load_model(ckpt_path)
    model, tok, block_size = entry["model"], entry["tok"], entry["block_size"]

    ids = _encode(tok, req.prompt)
    idx = torch.tensor([ids[-block_size:]], dtype=torch.long, device=DEVICE)

    async def token_stream() -> Iterator[str]:
        current = idx.clone()
        for _ in range(req.max_new_tokens):
            with torch.no_grad():
                logits, _, _ = model(current[:, -block_size:])
            next_logits = logits[:, -1, :] / max(req.temperature, 1e-6)
            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            current = torch.cat([current, next_id], dim=1)
            token_text = _decode(tok, [next_id.item()])
            yield f"data: {token_text}\n\n"
            await asyncio.sleep(0)   # yield control
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
