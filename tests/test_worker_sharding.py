import sys
import types
import unittest
from unittest.mock import patch


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


class FakeWorkerInfo:
    def __init__(self, worker_id, num_workers):
        self.id = worker_id
        self.num_workers = num_workers


def make_dataset(sample_ids):
    return VQADataset(
        FakeStreamingDataset(sample_ids),
        FakeTokenizer(),
        image_processor=None,
        mp_image_token_length=1,
    )


def collect_seen_ids(dataset, worker_id, num_workers):
    worker_info = FakeWorkerInfo(worker_id, num_workers)
    with patch("torch.utils.data.get_worker_info", return_value=worker_info):
        return [int(batch["input_ids"][1].item()) for batch in dataset.iter_for_worker()]


class TestWorkerSharding(unittest.TestCase):
    def test_streaming_dataset_is_not_duplicated_across_workers(self):
        sample_ids = list(range(2, 10))
        dataset = make_dataset(sample_ids)

        worker_zero_ids = collect_seen_ids(dataset, worker_id=0, num_workers=2)
        worker_one_ids = collect_seen_ids(dataset, worker_id=1, num_workers=2)

        self.assertEqual(
            len(worker_zero_ids) + len(worker_one_ids),
            len(sample_ids),
            f"Expected each base sample exactly once, saw worker 0={worker_zero_ids}, worker 1={worker_one_ids}",
        )
        self.assertEqual(
            set(worker_zero_ids).union(worker_one_ids),
            set(sample_ids),
            f"Expected all base samples to be covered, saw worker 0={worker_zero_ids}, worker 1={worker_one_ids}",
        )
        self.assertTrue(
            set(worker_zero_ids).isdisjoint(worker_one_ids),
            f"Expected worker shards to be disjoint, saw worker 0={worker_zero_ids}, worker 1={worker_one_ids}",
        )


if __name__ == "__main__":
    unittest.main()
