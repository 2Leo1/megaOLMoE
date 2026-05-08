#!/usr/bin/env python3
"""Router health check for an OLMo MoE checkpoint.

Loads a checkpoint, runs N held-out batches, and reports:
- Cross-entropy loss and perplexity
- Per-MoE-layer expert utilization (% of tokens per expert)
- Routing entropy from router logits vs. uniform max
- Load-balance coefficient of variation (CV) over experts
- Average top-k weight magnitude

A healthy router has near-uniform expert usage (CV < 0.3), entropy
close to log(num_experts), and no expert with 0% usage.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src/OLMo", REPO_ROOT / "src/megablocks", REPO_ROOT / "_stk"):
    sys.path.insert(0, str(_p))

DEFAULT_RUN_DIR = REPO_ROOT / "runs/tiny-router"
DEFAULT_DATA = REPO_ROOT / "data/olmoe-router-500m/wikipedia_en/part-00000.npy"


def find_latest_checkpoint(run_dir: Path) -> Path:
    cands = sorted(
        [p for p in run_dir.glob("step*-unsharded") if p.is_dir() and (p / "config.yaml").is_file()],
        key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)),
    )
    if not cands:
        raise SystemExit(f"No checkpoints found under {run_dir}")
    return cands[-1]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="Path to a uint16 token memmap (.npy treated as raw)")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    from megablocks.layers.moe import MoE
    from olmo.model import OLMo

    ckpt = args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint(args.run_dir.resolve())
    print(f"Loading {ckpt}")
    model = OLMo.from_checkpoint(str(ckpt), device=args.device)
    model.eval()

    n_experts = model.config.moe_num_experts
    top_k = model.config.moe_top_k
    print(f"Experts: {n_experts}, top_k: {top_k}")
    print(
        "MoE aux: "
        f"loss_weight={getattr(model.config, 'moe_loss_weight', 'n/a')}, "
        f"zloss_weight={getattr(model.config, 'moe_zloss_weight', 'n/a')}"
    )

    # Find all MoE layers and hook their routers.
    moe_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, MoE)]
    print(f"Found {len(moe_layers)} MoE layer(s)")

    layer_buf: list[dict] = [{"logits": [], "weights": [], "indices": []} for _ in moe_layers]

    def make_hook(idx: int):
        def hook(_module, _inp, out):
            _scores, logits, expert_weights, expert_indices = out[:4]
            layer_buf[idx]["logits"].append(logits.detach().float().cpu())
            layer_buf[idx]["weights"].append(expert_weights.detach().float().cpu())
            layer_buf[idx]["indices"].append(expert_indices.detach().cpu())
        return hook

    for i, (_name, m) in enumerate(moe_layers):
        m.router.register_forward_hook(make_hook(i))

    # Load held-out tokens.
    arr = np.memmap(args.data, dtype=np.uint16, mode="r")
    bs, sl = args.batch_size, args.seq_len
    chunk = bs * sl + 1
    rng = np.random.default_rng(args.seed)

    total_loss = 0.0
    total_tokens = 0

    print(f"Running {args.batches} batches of {bs}x{sl} from {args.data.name}...")

    with torch.inference_mode():
        for b in range(args.batches):
            start = int(rng.integers(0, len(arr) - chunk - 1))
            tokens = np.asarray(arr[start:start + chunk], dtype=np.int64)
            input_ids = torch.from_numpy(tokens[:-1].reshape(bs, sl)).to(args.device)
            labels = torch.from_numpy(tokens[1:].reshape(bs, sl)).to(args.device)

            out = model(input_ids=input_ids)
            logits = out.logits
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += labels.numel()

            if (b + 1) % 16 == 0:
                print(f"  batch {b + 1}/{args.batches}")

    avg_loss = total_loss / total_tokens
    print("\n=== Held-out CE ===")
    print(f"Tokens evaluated: {total_tokens}")
    print(f"CE loss:          {avg_loss:.4f}")
    print(f"Perplexity:       {math.exp(avg_loss):.2f}")
    print(f"BPB (log2):       {avg_loss / math.log(2):.4f}")

    print("\n=== Per-layer router stats ===")
    max_entropy = math.log(n_experts)
    print(f"(uniform entropy = log({n_experts}) = {max_entropy:.3f})")

    for i, buf in enumerate(layer_buf):
        if not buf["indices"]:
            print(f"\nLayer {i}: no data captured")
            continue

        indices = torch.cat(buf["indices"]).flatten()           # [N * top_k]
        weights = torch.cat(buf["weights"]).flatten()           # [N * top_k]
        logits = torch.cat(buf["logits"])                       # [N, num_experts]

        counts = torch.bincount(indices.long(), minlength=n_experts).float()
        usage_pct = counts / counts.sum() * 100

        routing_logits = logits

        probs = torch.softmax(routing_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean().item()

        top_values = torch.topk(routing_logits, k=min(3, n_experts), dim=-1).values
        top12_gap = (top_values[:, 0] - top_values[:, 1]).mean().item() if n_experts > 1 else float("nan")
        top23_gap = (top_values[:, 1] - top_values[:, 2]).mean().item() if n_experts > 2 else float("nan")

        mean_count = counts.mean().item()
        cv = (counts.std().item() / mean_count) if mean_count > 0 else float("inf")

        dead = int((counts == 0).sum().item())
        max_share = usage_pct.max().item()
        min_share = usage_pct.min().item()

        usage_str = " ".join(f"{u:5.2f}" for u in usage_pct.tolist())
        print(f"\nLayer {i}:")
        print(f"  expert usage %  : [{usage_str}]")
        print(f"  dead experts    : {dead}/{n_experts}")
        print(f"  max / min share : {max_share:.2f}% / {min_share:.2f}%")
        print(f"  routing entropy : {entropy:.3f}  ({entropy / max_entropy * 100:.1f}% of max)")
        print(f"  logit top gaps  : top1-top2={top12_gap:.4f}, top2-top3={top23_gap:.4f}")
        print(f"  load-balance CV : {cv:.3f}  (0=perfect, >0.5=imbalanced)")
        print(f"  weight mean/std : {weights.mean().item():.3f} / {weights.std().item():.3f}")

    print("\nHints:")
    print("  - If any layer shows a dead expert or CV > 0.5, the router is collapsing.")
    print("  - Entropy near 0 means router is over-confident (potentially picking same experts).")
    print("  - Entropy near log(N) with balanced usage means router is healthy.")


if __name__ == "__main__":
    main()
