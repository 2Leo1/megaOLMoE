#!/usr/bin/env python3
"""Build an OLMo/OLMoE-compatible pretraining token dataset.

The OLMo MemMapDataset expects raw numpy memmap bytes, not files written with
np.save(). This script streams public Hugging Face datasets, tokenizes them
with the same tokenizer as the training config, and writes raw memmap shards
that can be used directly in train.py via data.paths or data.datasets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER = REPO_ROOT / "src/OLMo/olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json"


@dataclass(frozen=True)
class SourceSpec:
    label: str
    path: str
    name: str | None
    split: str
    text_field: str
    weight: float


DEFAULT_MIX: tuple[SourceSpec, ...] = (
    SourceSpec(
        label="fineweb_edu_dedup",
        path="HuggingFaceTB/smollm-corpus",
        name="fineweb-edu-dedup",
        split="train",
        text_field="text",
        weight=0.60,
    ),
    SourceSpec(
        label="cosmopedia_v2",
        path="HuggingFaceTB/smollm-corpus",
        name="cosmopedia-v2",
        split="train",
        text_field="text",
        weight=0.20,
    ),
    SourceSpec(
        label="finemath_4plus",
        path="HuggingFaceTB/finemath",
        name="finemath-4plus",
        split="train",
        text_field="text",
        weight=0.15,
    ),
    SourceSpec(
        label="wikipedia_en",
        path="wikimedia/wikipedia",
        name="20231101.en",
        split="train",
        text_field="text",
        weight=0.05,
    ),
)


def parse_count(raw: str | int) -> int:
    if isinstance(raw, int):
        return raw
    value = str(raw).strip().replace("_", "").lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgtb]?)", value)
    if not match:
        raise argparse.ArgumentTypeError(f"Invalid count: {raw!r}")
    number = float(match.group(1))
    suffix = match.group(2)
    multiplier = {
        "": 1,
        "k": 1_000,
        "m": 1_000_000,
        "g": 1_000_000_000,
        "t": 1_000_000_000_000,
        "b": 1_000_000_000,
    }[suffix]
    return int(number * multiplier)


def human_count(value: int) -> str:
    for suffix, divider in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if abs(value) >= divider:
            return f"{value / divider:.3g}{suffix}"
    return str(value)


def load_train_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required when --train-config is used.") from exc

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path} does not look like an OLMo YAML config")
    return data


def nested_get(mapping: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_tokenizer_identifier(identifier: str | None) -> str:
    if identifier is None:
        return str(DEFAULT_TOKENIZER)

    path = Path(identifier)
    if path.is_file():
        return str(path)

    repo_relative = REPO_ROOT / identifier
    if repo_relative.is_file():
        return str(repo_relative)

    olmo_data_relative = REPO_ROOT / "src/OLMo/olmo_data" / identifier
    if olmo_data_relative.is_file():
        return str(olmo_data_relative)

    if identifier.startswith("tokenizers/"):
        tokenizers_relative = REPO_ROOT / "src/OLMo/olmo_data" / identifier
        if tokenizers_relative.is_file():
            return str(tokenizers_relative)

    return identifier


def load_tokenizer(identifier: str):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise SystemExit("The 'tokenizers' package is required.") from exc

    if Path(identifier).is_file():
        tokenizer = Tokenizer.from_file(identifier)
    else:
        tokenizer = Tokenizer.from_pretrained(identifier)
    tokenizer.no_truncation()
    return tokenizer


def dtype_from_name(name: str) -> np.dtype:
    dtype = np.dtype(name)
    if dtype.kind != "u":
        raise SystemExit(f"Use an unsigned integer dtype for token IDs, got {dtype}")
    return dtype


def check_dtype_capacity(dtype: np.dtype, vocab_size: int | None) -> None:
    if vocab_size is None:
        return
    max_value = np.iinfo(dtype).max
    if vocab_size - 1 > max_value:
        raise SystemExit(
            f"vocab_size={vocab_size} does not fit into {dtype}. "
            "Use --dtype uint32 and set data.memmap_dtype: uint32 in the train config."
        )


def load_sources(path: Path | None) -> list[SourceSpec]:
    if path is None:
        return list(DEFAULT_MIX)

    with path.open("r", encoding="utf-8") as f:
        raw_sources = json.load(f)
    if not isinstance(raw_sources, list):
        raise SystemExit("--sources-json must contain a JSON list")

    out: list[SourceSpec] = []
    for item in raw_sources:
        if not isinstance(item, dict):
            raise SystemExit("Each source in --sources-json must be an object")
        out.append(
            SourceSpec(
                label=str(item["label"]),
                path=str(item["path"]),
                name=None if item.get("name") in (None, "") else str(item["name"]),
                split=str(item.get("split", "train")),
                text_field=str(item.get("text_field", "text")),
                weight=float(item["weight"]),
            )
        )
    return out


def normalize_weights(sources: list[SourceSpec]) -> list[SourceSpec]:
    total = sum(source.weight for source in sources)
    if total <= 0:
        raise SystemExit("Source weights must sum to a positive number")
    return [
        SourceSpec(
            label=source.label,
            path=source.path,
            name=source.name,
            split=source.split,
            text_field=source.text_field,
            weight=source.weight / total,
        )
        for source in sources
    ]


def allocate_budgets(total_tokens: int, sources: list[SourceSpec], align_to: int) -> dict[str, int]:
    if align_to < 1:
        raise SystemExit("--align-to must be >= 1")
    aligned_total = total_tokens - (total_tokens % align_to)
    if aligned_total <= 0:
        raise SystemExit(f"--tokens must be at least --align-to ({align_to})")

    budgets: dict[str, int] = {}
    remaining = aligned_total
    for source in sources[:-1]:
        budget = int(math.floor(aligned_total * source.weight / align_to) * align_to)
        budgets[source.label] = budget
        remaining -= budget
    budgets[sources[-1].label] = remaining
    return budgets


def batched(values: Iterable[str], batch_size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for value in values:
        batch.append(value)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_hf_texts(
    source: SourceSpec,
    *,
    cache_dir: str | None,
    seed: int,
    shuffle_buffer: int,
    min_chars: int,
    max_chars: int,
    trust_remote_code: bool,
) -> Iterator[str]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The 'datasets' package is required.") from exc

    kwargs: dict[str, Any] = {
        "path": source.path,
        "name": source.name,
        "split": source.split,
        "streaming": True,
        "cache_dir": cache_dir,
    }
    if trust_remote_code:
        kwargs["trust_remote_code"] = True

    print(f"[{source.label}] opening HF stream...", flush=True)
    dataset = load_dataset(**kwargs)
    if shuffle_buffer > 0:
        print(
            f"[{source.label}] shuffle_buffer={shuffle_buffer:,}; "
            "first rows may take a bit while the streaming buffer fills.",
            flush=True,
        )
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
    else:
        print(f"[{source.label}] shuffle disabled; rows should start as soon as the first shard downloads.", flush=True)

    accepted = 0
    for row in dataset:
        value = row.get(source.text_field) if isinstance(row, dict) else None
        if not isinstance(value, str):
            continue
        text = value.strip()
        if len(text) < min_chars or len(text) > max_chars:
            continue
        accepted += 1
        if accepted == 1:
            print(f"[{source.label}] first text row received; tokenization is starting.", flush=True)
        yield text


@dataclass
class ShardInfo:
    label: str
    path: str
    tokens: int
    bytes: int


class ShardWriter:
    def __init__(
        self,
        root: Path,
        label: str,
        dtype: np.dtype,
        tokens_per_shard: int,
        align_to: int,
        overwrite: bool,
    ) -> None:
        self.root = root
        self.label = label
        self.dtype = dtype
        self.tokens_per_shard = tokens_per_shard
        self.align_to = align_to
        self.overwrite = overwrite
        self.itemsize = dtype.itemsize
        self.label_dir = root / label
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.shards: list[ShardInfo] = []
        self._mmap: np.memmap | None = None
        self._path: Path | None = None
        self._index = 0
        self._pos = 0

    def write(self, token_ids: list[int] | np.ndarray) -> None:
        if len(token_ids) == 0:
            return
        array = np.asarray(token_ids, dtype=self.dtype)
        offset = 0
        while offset < len(array):
            if self._mmap is None:
                self._open_next()
            assert self._mmap is not None
            room = self.tokens_per_shard - self._pos
            count = min(room, len(array) - offset)
            self._mmap[self._pos : self._pos + count] = array[offset : offset + count]
            self._pos += count
            offset += count
            if self._pos == self.tokens_per_shard:
                self.close_current()

    def _open_next(self) -> None:
        self._path = self.label_dir / f"part-{self._index:05d}.npy"
        if self._path.exists() and not self.overwrite:
            raise SystemExit(f"{self._path} exists. Use --overwrite to replace it.")
        self._mmap = np.memmap(self._path, mode="w+", dtype=self.dtype, shape=(self.tokens_per_shard,))
        self._pos = 0
        self._index += 1

    def close_current(self) -> None:
        if self._mmap is None or self._path is None:
            return
        written = self._pos - (self._pos % self.align_to)
        self._mmap.flush()
        del self._mmap
        self._mmap = None
        if written <= 0:
            self._path.unlink(missing_ok=True)
        else:
            with self._path.open("ab") as f:
                f.truncate(written * self.itemsize)
            self.shards.append(
                ShardInfo(
                    label=self.label,
                    path=str(self._path.resolve()),
                    tokens=written,
                    bytes=written * self.itemsize,
                )
            )
        self._path = None
        self._pos = 0

    def close(self) -> list[ShardInfo]:
        self.close_current()
        return self.shards


def write_data_yaml(path: Path, dtype: np.dtype, shards: list[ShardInfo]) -> None:
    by_label: dict[str, list[ShardInfo]] = {}
    for shard in shards:
        by_label.setdefault(shard.label, []).append(shard)

    lines = [
        "# Generated by scripts/gen_real_data.py",
        "data:",
        f"  memmap_dtype: {dtype.name}",
        "  datasets:",
    ]
    for label in sorted(by_label):
        lines.append(f"    {label}:")
        for shard in by_label[label]:
            lines.append(f"      - {shard.path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def token_ids_with_eos(encoding: Any, eos_token_id: int) -> list[int]:
    ids = list(encoding.ids)
    if not ids or ids[-1] != eos_token_id:
        ids.append(eos_token_id)
    return ids


def build_source(
    source: SourceSpec,
    *,
    budget: int,
    source_index: int,
    args: argparse.Namespace,
    tokenizer: Any,
    dtype: np.dtype,
    eos_token_id: int,
) -> list[ShardInfo]:
    writer = ShardWriter(
        root=args.output_dir,
        label=source.label,
        dtype=dtype,
        tokens_per_shard=args.tokens_per_shard,
        align_to=args.align_to,
        overwrite=args.overwrite,
    )
    written = 0
    docs = 0
    started = time.time()

    texts = iter_hf_texts(
        source,
        cache_dir=args.hf_cache_dir,
        seed=args.seed + source_index,
        shuffle_buffer=args.shuffle_buffer,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        trust_remote_code=args.trust_remote_code,
    )

    print(f"\n[{source.label}] target={human_count(budget)} tokens from {source.path}/{source.name or '-'}")
    for batch in batched(texts, args.batch_size):
        encodings = tokenizer.encode_batch(batch)
        for encoding in encodings:
            ids = token_ids_with_eos(encoding, eos_token_id)
            if len(ids) < args.min_doc_tokens:
                continue
            remaining = budget - written
            if remaining <= 0:
                break
            if len(ids) > remaining:
                ids = ids[:remaining]
                if ids:
                    ids[-1] = eos_token_id
            writer.write(ids)
            written += len(ids)
            docs += 1
        if written >= budget:
            break
        if docs and docs % args.log_every_docs == 0:
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[{source.label}] docs={docs:,} tokens={written:,}/{budget:,} "
                f"rate={written / elapsed:,.0f} tok/s",
                flush=True,
            )

    shards = writer.close()
    trainable_tokens = sum(shard.tokens for shard in shards)
    if trainable_tokens < budget:
        raise SystemExit(
            f"{source.label} ended early: wrote {trainable_tokens:,} trainable tokens, "
            f"target was {budget:,}"
        )
    elapsed = max(time.time() - started, 1e-6)
    print(
        f"[{source.label}] done: docs={docs:,}, tokens={trainable_tokens:,}, "
        f"shards={len(shards)}, rate={trainable_tokens / elapsed:,.0f} tok/s"
    )
    return shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream, tokenize, and shard an OLMo/OLMoE raw memmap dataset."
    )
    parser.add_argument("--tokens", type=parse_count, default=parse_count("500M"), help="Total token budget.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/olmoe-router-500m")
    parser.add_argument("--tokens-per-shard", type=parse_count, default=parse_count("100M"))
    parser.add_argument("--align-to", type=int, default=4096, help="Usually model.max_sequence_length.")
    parser.add_argument("--dtype", default=None, help="Defaults to train config data.memmap_dtype or uint16.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer JSON path or HF tokenizer id.")
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--train-config", type=Path, default=None, help="Optional OLMo/OLMoE YAML config.")
    parser.add_argument("--sources-json", type=Path, default=None, help="Override the built-in dataset mix.")
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=6198)
    parser.add_argument("--shuffle-buffer", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--min-doc-tokens", type=int, default=16)
    parser.add_argument("--log-every-docs", type=int, default=1_000)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()

    train_config: dict[str, Any] = {}
    if args.train_config is not None:
        train_config = load_train_config(args.train_config)
        args.align_to = int(nested_get(train_config, ("model", "max_sequence_length"), args.align_to))
        if args.dtype is None:
            args.dtype = nested_get(train_config, ("data", "memmap_dtype"), None)
        if args.tokenizer is None:
            args.tokenizer = nested_get(train_config, ("tokenizer", "identifier"), None)
        if args.eos_token_id is None:
            args.eos_token_id = nested_get(train_config, ("model", "eos_token_id"), None)
        if args.vocab_size is None:
            args.vocab_size = nested_get(train_config, ("model", "vocab_size"), None)
    if args.eos_token_id is None:
        args.eos_token_id = 0

    args.dtype = args.dtype or "uint16"
    dtype = dtype_from_name(args.dtype)
    check_dtype_capacity(dtype, args.vocab_size)

    args.tokens_per_shard -= args.tokens_per_shard % args.align_to
    if args.tokens_per_shard <= 0:
        raise SystemExit("--tokens-per-shard must be at least --align-to after alignment")

    tokenizer_identifier = resolve_tokenizer_identifier(args.tokenizer)
    sources = normalize_weights(load_sources(args.sources_json))
    budgets = allocate_budgets(args.tokens, sources, args.align_to)
    aligned_total = sum(budgets.values())

    print("=== dataset plan ===")
    print(f"output_dir:       {args.output_dir}")
    print(f"tokens:           {aligned_total:,} ({human_count(aligned_total)})")
    print(f"tokens_per_shard: {args.tokens_per_shard:,}")
    print(f"align_to:         {args.align_to}")
    print(f"dtype:            {dtype.name}")
    print(f"tokenizer:        {tokenizer_identifier}")
    print(f"eos_token_id:     {args.eos_token_id}")
    print("sources:")
    for source in sources:
        print(
            f"  - {source.label}: weight={source.weight:.3f}, "
            f"budget={budgets[source.label]:,}, hf={source.path}, name={source.name}"
        )

    if args.dry_run:
        return

    if args.eos_token_id is None:
        raise SystemExit("Set --eos-token-id or pass --train-config with model.eos_token_id.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(tokenizer_identifier)
    tokenizer_vocab_size = tokenizer.get_vocab_size()
    check_dtype_capacity(dtype, tokenizer_vocab_size)
    if args.vocab_size is not None and args.vocab_size != tokenizer_vocab_size:
        raise SystemExit(
            f"vocab_size mismatch: train config has {args.vocab_size}, tokenizer has {tokenizer_vocab_size}"
        )

    all_shards: list[ShardInfo] = []
    for index, source in enumerate(sources):
        all_shards.extend(
            build_source(
                source,
                budget=budgets[source.label],
                source_index=index,
                args=args,
                tokenizer=tokenizer,
                dtype=dtype,
                eos_token_id=args.eos_token_id,
            )
        )

    manifest = {
        "created_by": "scripts/gen_real_data.py",
        "target_tokens_requested": args.tokens,
        "trainable_tokens": sum(shard.tokens for shard in all_shards),
        "dtype": dtype.name,
        "align_to": args.align_to,
        "tokens_per_shard": args.tokens_per_shard,
        "tokenizer": tokenizer_identifier,
        "eos_token_id": args.eos_token_id,
        "vocab_size": tokenizer_vocab_size,
        "sources": [asdict(source) | {"budget_tokens": budgets[source.label]} for source in sources],
        "shards": [asdict(shard) for shard in all_shards],
    }
    manifest_path = args.output_dir / "manifest.json"
    data_yaml_path = args.output_dir / "data_paths.yaml"
    write_manifest(manifest_path, manifest)
    write_data_yaml(data_yaml_path, dtype, all_shards)

    print("\n=== done ===")
    print(f"manifest:  {manifest_path}")
    print(f"data yaml: {data_yaml_path}")
    print("Use the generated YAML as the data section, or copy its data.datasets block into your train config.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    else:
        # Some datasets/pyarrow builds can abort during interpreter teardown after
        # a successful streaming run. All shards are already flushed before here.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
