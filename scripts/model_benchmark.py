#!/usr/bin/env python3
"""Run OLMo model benchmarks for a saved checkpoint."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from torchmetrics import Metric


REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src/OLMo", REPO_ROOT / "src/megablocks", REPO_ROOT / "_stk"):
    sys.path.insert(0, str(_p))


log = logging.getLogger("model_benchmark")


def find_latest_checkpoint(run_dir: Path) -> Path:
    candidates = sorted(
        [p for p in run_dir.glob("step*-unsharded") if p.is_dir() and (p / "config.yaml").is_file()],
        key=lambda p: int(re.search(r"step(\d+)", p.name).group(1)),
    )
    if not candidates:
        raise SystemExit(f"No unsharded checkpoints found under {run_dir}")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--benchmark-config", type=Path, default=REPO_ROOT / "configs/model_benchmark.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def load_benchmark_config(path: Path) -> DictConfig:
    if not path.is_file():
        raise SystemExit(f"Benchmark config not found: {path}")
    config = om.load(path)
    if not isinstance(config, DictConfig):
        raise SystemExit(f"Benchmark config must be a mapping: {path}")
    return config


def apply_benchmark_config(train_config, benchmark_config: DictConfig, max_batches: int | None):
    from olmo.config import EvaluatorConfig, TrainConfig

    if "eval_subset_num_batches" in benchmark_config:
        train_config.eval_subset_num_batches = benchmark_config.eval_subset_num_batches
    if "device_eval_batch_size" in benchmark_config:
        train_config.device_eval_batch_size = benchmark_config.device_eval_batch_size
    if max_batches is not None:
        train_config.eval_subset_num_batches = max_batches

    evaluators = []
    for raw_eval in benchmark_config.get("evaluators", []):
        eval_config = om.merge(om.structured(EvaluatorConfig), raw_eval)
        if max_batches is not None:
            eval_config.subset_num_batches = max_batches
        evaluators.append(om.to_object(eval_config))

    if not evaluators:
        raise SystemExit("Benchmark config does not define any evaluators")

    train_config.evaluators = evaluators
    train_config.save_folder = str(REPO_ROOT / "runs" / "_benchmark_tmp")
    train_config.save_overwrite = True
    train_config.save_data_indices = False
    train_config.no_pre_train_checkpoint = True
    train_config.load_path = None
    train_config.wandb = None
    assert isinstance(train_config, TrainConfig)
    return train_config


def get_labels(batch: Dict[str, Any]) -> torch.Tensor:
    labels, label_mask, attention_mask, instance_mask = (
        batch["input_ids"].clone(),
        batch.get("label_mask"),
        batch.get("attention_mask"),
        batch.get("instance_mask"),
    )
    if label_mask is not None:
        labels.masked_fill_(~label_mask, -100)
    if attention_mask is not None:
        labels.masked_fill_(attention_mask == 0.0, -100)
    if instance_mask is not None:
        labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
    return labels[..., 1:].contiguous()


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def eval_batch(model, batch: Dict[str, Any], device: torch.device, precision: torch.dtype | None):
    autocast_enabled = device.type == "cuda" and precision is not None
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=precision, enabled=autocast_enabled):
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
            doc_lens=batch.get("doc_lens"),
            max_doc_lens=batch.get("max_doc_lens"),
        ).logits

        logits_for_loss = logits[..., :-1, :].contiguous()
        labels = get_labels(batch)
        ce_loss = F.cross_entropy(
            logits_for_loss.view(-1, logits_for_loss.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )
        ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1).mean(dim=-1)
    return ce_loss, logits


def run_benchmark(model, evaluators, device: torch.device, precision: torch.dtype | None):
    model.eval()
    all_metrics: Dict[str, float] = {}

    for evaluator in evaluators:
        log.info("Running %s", evaluator.label)
        evaluator.reset_metrics()

        num_batches = evaluator.subset_num_batches
        if num_batches is not None and num_batches > 0:
            num_batches = min(num_batches, len(evaluator.eval_loader))
        else:
            num_batches = len(evaluator.eval_loader)

        for step, batch in enumerate(evaluator.eval_loader):
            if step >= num_batches:
                break
            batch = move_to_device(batch, device)
            ce_loss, logits = eval_batch(model, batch, device, precision)
            evaluator.update_metrics(batch, ce_loss, logits)
            if step == 0 or (step + 1) == num_batches or (step + 1) % 10 == 0:
                log.info("[%s] batch %d/%d", evaluator.label, step + 1, num_batches)

        metrics = evaluator.compute_metrics()
        all_metrics.update(metrics)
        for metric in evaluator.eval_metric.values() if isinstance(evaluator.eval_metric, dict) else [evaluator.eval_metric]:
            assert isinstance(metric, Metric)
            metric.reset()

    return all_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if args.checkpoint is None and args.run_dir is None:
        raise SystemExit("Pass --checkpoint or --run-dir")

    from olmo.config import TrainConfig
    from olmo.eval import build_evaluators
    from olmo.model import OLMo

    checkpoint = args.checkpoint.resolve() if args.checkpoint else find_latest_checkpoint(args.run_dir.resolve())
    device = torch.device(args.device)

    log.info("Loading checkpoint: %s", checkpoint)
    model = OLMo.from_checkpoint(str(checkpoint), device=str(device))
    train_config = TrainConfig.load(checkpoint / "config.yaml", validate_paths=False)
    train_config = apply_benchmark_config(
        train_config,
        load_benchmark_config(args.benchmark_config),
        args.max_batches,
    )

    precision = train_config.autocast_precision if device.type == "cuda" else None
    evaluators = build_evaluators(train_config, device)
    metrics = run_benchmark(model, evaluators, device, precision)

    print("\n=== Benchmark Results ===")
    for key in sorted(metrics):
        print(f"{key}: {metrics[key]:.6g}")


if __name__ == "__main__":
    main()
