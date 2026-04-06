import sys
import types
from pathlib import Path

from torch.utils.data import DataLoader, IterableDataset

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


class _BaseFakeStreamingDataset:
    def __init__(self, sample_ids):
        self.sample_ids = list(sample_ids)

    def __iter__(self):
        for sample_id in self.sample_ids:
            yield {
                "images": None,
                "texts": [{"user": f"user-{sample_id}", "assistant": str(sample_id)}],
            }


class FakeStreamingDataset(_BaseFakeStreamingDataset):
    def shard(self, num_shards, index):
        return FakeStreamingDataset(self.sample_ids[index::num_shards])


class NonShardableStreamingDataset(_BaseFakeStreamingDataset):
    pass


class LegacyVQADataset(VQADataset):
    def iter_for_worker(self):
        for data in self.dataset:
            yield self._process_data(data)


class WorkerIterableDataset(IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        yield from self.dataset.iter_for_worker()


def make_vqa_dataset(dataset_cls, base_dataset):
    return dataset_cls(
        base_dataset,
        FakeTokenizer(),
        image_processor=None,
        mp_image_token_length=1,
    )


def collect_ids_with_dataloader(dataset_cls, base_dataset, num_workers):
    dataset = WorkerIterableDataset(make_vqa_dataset(dataset_cls, base_dataset))
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
    )
    return [int(sample["input_ids"][1].item()) for sample in loader]
