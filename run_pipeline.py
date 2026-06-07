#!/usr/bin/env python3
"""
run_pipeline.py  —  End-to-end training pipeline
Runs: Part2 → Part4 → Part6 (SFT) → Part7 (RM) → Part8 (PPO)

Usage:
  python run_pipeline.py --data part_2/tiny.txt [--steps 500] [--gpu]
"""

import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: str, cwd: Path):
    print(f"\n{'─'*60}")
    print(f"▶  {cmd}")
    print(f"{'─'*60}")
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        print(f"[error] Command failed (exit {r.returncode})")
        sys.exit(r.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="part_2/tiny.txt", help="Training text file")
    p.add_argument("--steps", type=int, default=300, help="Optimizer steps per stage")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=2)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--skip_to", type=str, default="part2",
                   choices=["part2", "part4", "sft", "rm", "ppo"],
                   help="Skip to a specific stage")
    args = p.parse_args()

    data_path = (ROOT / args.data).resolve()
    if not data_path.exists():
        print(f"[error] Data file not found: {data_path}")
        sys.exit(1)

    sizes = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                 block_size=args.block, batch_size=args.batch, steps=args.steps)
    sz = " ".join(f"--{k} {v}" for k, v in sizes.items())

    stages_done = {"part2": False, "part4": False, "sft": False, "rm": False, "ppo": False}
    skip = args.skip_to
    for k in stages_done:
        if k == skip:
            break
        stages_done[k] = True

    # ── Stage 1: Part 2 — Byte-level GPT ──────────────────────────
    if not stages_done["part2"]:
        print("\n🏗️  STAGE 1 — Part 2: Train byte-level GPT")
        run(
            f"python train.py --data {data_path} {sz} "
            f"--out ../runs/part2 --sample_every 100 --eval_interval 100",
            cwd=ROOT / "part_2",
        )

    # ── Stage 2: Part 4 — BPE + modern architecture ───────────────
    if not stages_done["part4"]:
        print("\n🏗️  STAGE 2 — Part 4: Train with BPE tokenizer")
        run(
            f"python train.py --data {data_path} --bpe --vocab_size 2000 {sz} "
            f"--out ../runs/part4 --log tensorboard --grad_accum_steps 2",
            cwd=ROOT / "part_4",
        )

    bpe_dir  = ROOT / "runs/part4/tokenizer"
    sft_ckpt = ROOT / "runs/part4/model_last.pt"

    # ── Stage 3: SFT ──────────────────────────────────────────────
    if not stages_done["sft"]:
        print("\n🏗️  STAGE 3 — Part 6: Supervised Fine-Tuning (SFT)")
        bpe_arg = f"--bpe_dir {bpe_dir}" if bpe_dir.exists() else ""
        ckpt_arg = f"--ckpt {sft_ckpt}" if sft_ckpt.exists() else ""
        run(
            f"python train_sft.py {ckpt_arg} {bpe_arg} --out ../runs/sft-demo "
            f"--steps {args.steps} {sz}",
            cwd=ROOT / "part_6",
        )

    sft_ckpt_out = ROOT / "runs/sft-demo/model_last.pt"

    # ── Stage 4: Reward Model ─────────────────────────────────────
    if not stages_done["rm"]:
        print("\n🏗️  STAGE 4 — Part 7: Train Reward Model")
        bpe_arg = f"--bpe_dir {bpe_dir}" if bpe_dir.exists() else ""
        run(
            f"python train_rm.py {bpe_arg} --out ../runs/rm-demo "
            f"--steps {args.steps} {sz}",
            cwd=ROOT / "part_7",
        )

    rm_ckpt = ROOT / "runs/rm-demo/model_last.pt"

    # ── Stage 5: PPO ──────────────────────────────────────────────
    if not stages_done["ppo"] and sft_ckpt_out.exists() and rm_ckpt.exists():
        print("\n🏗️  STAGE 5 — Part 8: RLHF with PPO")
        bpe_arg = f"--bpe_dir {bpe_dir}" if bpe_dir.exists() else ""
        run(
            f"python train_ppo.py "
            f"--policy_ckpt {sft_ckpt_out} --reward_ckpt {rm_ckpt} {bpe_arg} "
            f"--out ../runs/ppo-demo --steps {args.steps} --batch_size {args.batch} "
            f"--block_size {args.block} --resp_len 64",
            cwd=ROOT / "part_8",
        )
    elif not stages_done["ppo"]:
        print("[skip] PPO skipped — SFT or RM checkpoint not found")

    print("\n" + "═"*60)
    print("✅  Pipeline complete!")
    print("═"*60)
    print(f"\nCheckpoints saved under:  {ROOT}/runs/")
    print(f"Start API server:          uvicorn api.server:app --reload")
    print(f"TensorBoard:               tensorboard --logdir runs/")


if __name__ == "__main__":
    main()
