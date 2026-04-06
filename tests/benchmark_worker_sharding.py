import argparse
import multiprocessing
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.worker_sharding_utils import (
    FakeStreamingDataset,
    LegacyVQADataset,
    VQADataset,
    collect_ids_with_dataloader,
)


@dataclass
class BenchmarkStats:
    name: str
    total_processed: int
    unique_processed: int
    duplicates: int
    elapsed_s: float

    @property
    def raw_samples_per_s(self):
        return self.total_processed / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def effective_samples_per_s(self):
        return self.unique_processed / self.elapsed_s if self.elapsed_s else 0.0

    @property
    def duplicate_factor(self):
        return self.total_processed / self.unique_processed if self.unique_processed else 0.0


def run_case(dataset_cls, num_workers, samples_per_repeat, repeats):
    total_processed = 0
    unique_processed = set()

    start = time.perf_counter()
    for repeat in range(repeats):
        start_id = 2 + repeat * samples_per_repeat
        sample_ids = list(range(start_id, start_id + samples_per_repeat))
        seen_ids = collect_ids_with_dataloader(
            dataset_cls,
            FakeStreamingDataset(sample_ids),
            num_workers=num_workers,
        )
        total_processed += len(seen_ids)
        unique_processed.update(seen_ids)

    elapsed_s = time.perf_counter() - start
    return BenchmarkStats(
        name=dataset_cls.__name__,
        total_processed=total_processed,
        unique_processed=len(unique_processed),
        duplicates=total_processed - len(unique_processed),
        elapsed_s=elapsed_s,
    )


def print_stats(stats):
    print(f"{stats.name}:")
    print(f"  elapsed_s:            {stats.elapsed_s:.6f}")
    print(f"  total_processed:      {stats.total_processed}")
    print(f"  unique_processed:     {stats.unique_processed}")
    print(f"  duplicates:           {stats.duplicates}")
    print(f"  duplicate_factor:     {stats.duplicate_factor:.2f}x")
    print(f"  raw_samples_per_s:    {stats.raw_samples_per_s:.2f}")
    print(f"  effective_samples/s:  {stats.effective_samples_per_s:.2f}")


def main():
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(
        description="Benchmark real multi-worker DataLoader behavior for VQADataset.iter_for_worker()."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    print(
        f"Running benchmark with workers={args.workers}, samples={args.samples}, repeats={args.repeats}"
    )

    legacy_stats = run_case(
        LegacyVQADataset,
        num_workers=args.workers,
        samples_per_repeat=args.samples,
        repeats=args.repeats,
    )
    current_stats = run_case(
        VQADataset,
        num_workers=args.workers,
        samples_per_repeat=args.samples,
        repeats=args.repeats,
    )

    print_stats(legacy_stats)
    print_stats(current_stats)

    if legacy_stats.effective_samples_per_s:
        improvement = current_stats.effective_samples_per_s / legacy_stats.effective_samples_per_s
        print(f"effective_samples/s improvement: {improvement:.2f}x")


if __name__ == "__main__":
    main()
