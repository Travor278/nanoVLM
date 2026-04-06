import argparse
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from tests.worker_sharding_utils import (
    FakeStreamingDataset,
    LegacyVQADataset,
    VQADataset,
    make_vqa_dataset,
)


@dataclass
class SmokeStats:
    name: str
    warmup_s: float
    train_s: float
    scan_s: float
    trained_batches: int
    total_seen: int
    unique_seen: int
    duplicates: int

    @property
    def duplicate_factor(self):
        return self.total_seen / self.unique_seen if self.unique_seen else 0.0


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, labels):
        logits = self.proj(self.embedding(input_ids))
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_loader(dataset_cls, sample_ids, num_workers, batch_size, seq_length):
    vqa_dataset = make_vqa_dataset(dataset_cls, FakeStreamingDataset(sample_ids))
    train_dataset = ConstantLengthDataset(
        vqa_dataset,
        infinite=False,
        max_sample_length=16,
        seq_length=seq_length,
        num_of_sequences=batch_size * 4,
        queue_size=2,
        max_images_per_example=1,
        max_images_per_knapsack=8,
    )
    collator = VQACollator(vqa_dataset.tokenizer, max_length=seq_length)
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=device_is_cuda(),
        persistent_workers=False,
        drop_last=False,
        generator=generator,
    )


def device_is_cuda():
    return torch.cuda.is_available()


def extract_sample_ids(batch):
    sample_ids = []
    for sequence in batch["input_ids"]:
        tokens = sequence.tolist()
        for previous_token, current_token in zip(tokens, tokens[1:]):
            if previous_token == 1 and current_token not in (0, 1):
                sample_ids.append(int(current_token))
    return sample_ids


def run_training_steps(dataset_cls, sample_ids, num_workers, batch_size, seq_length, steps, device):
    loader = build_loader(dataset_cls, sample_ids, num_workers, batch_size, seq_length)
    iterator = iter(loader)

    synchronize(device)
    warmup_start = time.perf_counter()
    first_batch = next(iterator)
    synchronize(device)
    warmup_s = time.perf_counter() - warmup_start

    vocab_size = max(sample_ids) + 32
    model = TinyLanguageModel(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_batches = [first_batch, *list(islice(iterator, max(steps - 1, 0)))]

    synchronize(device)
    train_start = time.perf_counter()
    for batch in train_batches:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss = model(input_ids, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    train_s = time.perf_counter() - train_start

    return warmup_s, train_s, len(train_batches)


def scan_loader(dataset_cls, sample_ids, num_workers, batch_size, seq_length):
    loader = build_loader(dataset_cls, sample_ids, num_workers, batch_size, seq_length)
    seen_ids = []

    scan_start = time.perf_counter()
    for batch in loader:
        seen_ids.extend(extract_sample_ids(batch))
    scan_s = time.perf_counter() - scan_start
    return seen_ids, scan_s


def run_case(dataset_cls, sample_ids, num_workers, batch_size, seq_length, steps, device):
    warmup_s, train_s, trained_batches = run_training_steps(
        dataset_cls,
        sample_ids,
        num_workers,
        batch_size,
        seq_length,
        steps,
        device,
    )
    seen_ids, scan_s = scan_loader(
        dataset_cls,
        sample_ids,
        num_workers,
        batch_size,
        seq_length,
    )

    unique_seen = len(set(seen_ids))
    return SmokeStats(
        name=dataset_cls.__name__,
        warmup_s=warmup_s,
        train_s=train_s,
        scan_s=scan_s,
        trained_batches=trained_batches,
        total_seen=len(seen_ids),
        unique_seen=unique_seen,
        duplicates=len(seen_ids) - unique_seen,
    )


def print_stats(stats):
    print(f"{stats.name}:")
    print(f"  warmup_s:          {stats.warmup_s:.3f}")
    print(f"  train_s:           {stats.train_s:.3f}")
    print(f"  scan_s:            {stats.scan_s:.3f}")
    print(f"  trained_batches:   {stats.trained_batches}")
    print(f"  total_seen:        {stats.total_seen}")
    print(f"  unique_seen:       {stats.unique_seen}")
    print(f"  duplicates:        {stats.duplicates}")
    print(f"  duplicate_factor:  {stats.duplicate_factor:.2f}x")


def validate_stats(stats, expected_unique, expected_duplicate_factor):
    if stats.unique_seen != expected_unique:
        raise AssertionError(
            f"{stats.name} expected {expected_unique} unique samples, saw {stats.unique_seen}"
        )
    if round(stats.duplicate_factor, 2) != round(expected_duplicate_factor, 2):
        raise AssertionError(
            f"{stats.name} expected duplicate factor {expected_duplicate_factor:.2f}x, "
            f"saw {stats.duplicate_factor:.2f}x"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test the real training data path with multi-worker sharding."
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    multiprocessing.freeze_support()
    args = parse_args()
    device = resolve_device(args.device)
    torch.manual_seed(0)

    sample_ids = list(range(100, 100 + args.samples))
    print(
        "Running smoke test with "
        f"device={device}, workers={args.workers}, batch_size={args.batch_size}, "
        f"samples={args.samples}, seq_length={args.seq_length}, steps={args.steps}"
    )

    overall_start = time.perf_counter()
    legacy_stats = run_case(
        LegacyVQADataset,
        sample_ids,
        args.workers,
        args.batch_size,
        args.seq_length,
        args.steps,
        device,
    )
    current_stats = run_case(
        VQADataset,
        sample_ids,
        args.workers,
        args.batch_size,
        args.seq_length,
        args.steps,
        device,
    )
    total_s = time.perf_counter() - overall_start

    print_stats(legacy_stats)
    print_stats(current_stats)

    validate_stats(
        legacy_stats,
        expected_unique=len(sample_ids),
        expected_duplicate_factor=args.workers,
    )
    validate_stats(
        current_stats,
        expected_unique=len(sample_ids),
        expected_duplicate_factor=1.0,
    )

    print(f"total_runtime_s:    {total_s:.3f}")


if __name__ == "__main__":
    main()
