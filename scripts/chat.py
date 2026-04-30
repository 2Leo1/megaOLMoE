#!/usr/bin/env python3
"""Tiny interactive generation script for local OLMo/OLMoE checkpoints."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src/OLMo", REPO_ROOT / "src/megablocks", REPO_ROOT / "_stk"):
    sys.path.insert(0, str(path))


DEFAULT_RUN_DIR = REPO_ROOT / "runs/tiny-router-smoke"


def checkpoint_step(path: Path) -> int:
    match = re.search(r"step(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        [p for p in run_dir.glob("step*") if p.is_dir() and (p / "config.yaml").is_file()],
        key=checkpoint_step,
    )
    if not candidates:
        raise SystemExit(f"No checkpoints found under {run_dir}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a local OLMo/OLMoE checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--prompt", default=None)
    return parser.parse_args()


def load_model(checkpoint: Path, device: str) -> tuple[Any, Any]:
    from olmo.model import OLMo
    from olmo.tokenizer import Tokenizer

    print(f"Loading checkpoint: {checkpoint}")
    model = OLMo.from_checkpoint(str(checkpoint), device=device)
    model.eval()
    tokenizer = Tokenizer.from_checkpoint(str(checkpoint))
    params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {params:,} params on {device}")
    print(f"Tokenizer vocab={tokenizer.vocab_size}, eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}\n")
    return model, tokenizer


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
    device: str,
) -> str:
    from olmo.beam_search import TopPSampler

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_prompt_len = max(model.config.max_sequence_length - max_new_tokens - 1, 1)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    sampler = None if greedy or temperature <= 0 else TopPSampler(p=top_p, temperature=temperature)

    with torch.inference_mode():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_steps=max_new_tokens,
            beam_size=1,
            sampler=sampler,
            min_steps=1,
        )

    completion_ids = out.token_ids[0, 0].tolist()
    if tokenizer.eos_token_id in completion_ids:
        completion_ids = completion_ids[: completion_ids.index(tokenizer.eos_token_id)]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint(args.run_dir.resolve())
    model, tokenizer = load_model(checkpoint, args.device)

    def run_once(prompt: str) -> None:
        completion = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
            device=args.device,
        )
        print(f"\n{prompt}{completion}\n")

    if args.prompt is not None:
        run_once(args.prompt)
        return

    print("Type a prompt. Use Ctrl+C or /exit to quit.\n")
    while True:
        try:
            prompt = input("Prompt> ")
        except KeyboardInterrupt:
            print("\nBye!")
            return
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt in {"/exit", "/quit"}:
            print("Bye!")
            return
        run_once(prompt)


if __name__ == "__main__":
    main()
