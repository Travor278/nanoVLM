import argparse
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

stub_processors = types.ModuleType("data.processors")
stub_processors.get_image_string = lambda tokenizer, counts, mp_len: ""
sys.modules.setdefault("data.processors", stub_processors)

from data.datasets import VQADataset


class FakeTokenizer:
    image_token = "<image>"
    pad_token_id = 0

    def apply_chat_template(
        self, messages, tokenize=False, add_special_tokens=False, return_dict=False
    ):
        if not tokenize:
            return "".join(message["content"] for message in messages)

        tokens = []
        for message in messages:
            if message["role"] == "user":
                tokens.append(1)
            else:
                tokens.append(int(message["content"]))

        if return_dict:
            return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
        return tokens

    def encode(self, text):
        if not text:
            return []
        return [99]


class FakeStreamingDataset:
    def __init__(self, sample_ids):
        self.sample_ids = list(sample_ids)

    def __iter__(self):
        for sample_id in self.sample_ids:
            yield {
                "images": None,
                "texts": [{"user": f"user-{sample_id}", "assistant": str(sample_id)}],
            }

    def shard(self, num_shards, index):
        return FakeStreamingDataset(self.sample_ids[index::num_shards])


class LegacyVQADataset(VQADataset):
    def iter_for_worker(self):
        for data in self.dataset:
            yield self._process_data(data)


class FakeWorkerInfo:
    def __init__(self, worker_id, num_workers):
        self.id = worker_id
        self.num_workers = num_workers


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


def collect_seen_ids(dataset, worker_id, num_workers):
    worker_info = FakeWorkerInfo(worker_id, num_workers)
    with patch("torch.utils.data.get_worker_info", return_value=worker_info):
        return [int(batch["input_ids"][1].item()) for batch in dataset.iter_for_worker()]


def run_case(dataset_cls, num_workers, samples_per_repeat, repeats):
    tokenizer = FakeTokenizer()
    total_processed = 0
    unique_processed = set()

    start = time.perf_counter()
    for repeat in range(repeats):
        start_id = 2 + repeat * samples_per_repeat
        sample_ids = list(range(start_id, start_id + samples_per_repeat))
        dataset = dataset_cls(
            FakeStreamingDataset(sample_ids),
            tokenizer,
            image_processor=None,
            mp_image_token_length=1,
        )

        for worker_id in range(num_workers):
            seen_ids = collect_seen_ids(dataset, worker_id, num_workers)
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
    parser = argparse.ArgumentParser(
        description="CPU benchmark for worker sharding behavior in VQADataset.iter_for_worker()."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

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
