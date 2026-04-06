import unittest
from types import SimpleNamespace

import torch

from data.advanced_datasets import ConstantLengthDataset
from data.collators import VQACollator
from models.language_model import LanguageModel


class _FakeTokenizer:
    pad_token_id = 0


class _FakePackedDataset:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.mp_image_token_length = 1


def make_model():
    cfg = SimpleNamespace(
        lm_hidden_dim=64,
        lm_inter_dim=128,
        lm_rms_eps=1e-5,
        lm_re_base=10000.0,
        lm_max_position_embeddings=1024,
        lm_attn_scaling=1.0,
        lm_vocab_size=256,
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_dropout=0.0,
        lm_n_blocks=2,
        lm_use_tokens=True,
        lm_tie_weights=True,
    )
    model = LanguageModel(cfg)
    model.eval()
    return model


def make_sample(tokens):
    return {
        "input_ids": torch.tensor(tokens, dtype=torch.long),
        "labels": torch.tensor(tokens[:-1] + [-100], dtype=torch.long),
        "attention_mask": torch.tensor([1, 1, 1, 0], dtype=torch.bool),
        "images": [],
    }


class TestPackedSequenceMasking(unittest.TestCase):
    def test_pack_resets_position_ids_and_builds_document_mask(self):
        dataset = ConstantLengthDataset(
            _FakePackedDataset(),
            seq_length=8,
            num_of_sequences=1,
        )
        buffer = [
            make_sample([11, 12, 13, 0]),
            make_sample([31, 32, 33, 0]),
        ]

        packed = dataset._pack_one_group([0, 1], buffer, max_len=8)
        input_ids, labels, attention_mask, images, position_ids, document_ids = packed

        self.assertEqual(input_ids.tolist(), [11, 12, 13, 0, 31, 32, 33, 0])
        self.assertEqual(labels.tolist(), [11, 12, 13, -100, 31, 32, 33, -100])
        self.assertEqual(attention_mask.tolist(), [1, 1, 1, 0, 1, 1, 1, 0])
        self.assertEqual(position_ids.tolist(), [0, 1, 2, 3, 0, 1, 2, 3])
        self.assertEqual(document_ids.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(images, [])

    def test_reset_position_ids_plus_document_mask_isolates_sample_b(self):
        torch.manual_seed(0)
        model = make_model()

        packed_left = torch.tensor([[11, 12, 13, 0, 31, 32, 33, 0]], dtype=torch.long)
        packed_right = torch.tensor([[21, 22, 23, 0, 31, 32, 33, 0]], dtype=torch.long)
        padding_mask = torch.tensor([[1, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.long)
        reset_positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long)
        document_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)

        with torch.no_grad():
            baseline_left, _ = model(packed_left, attention_mask=padding_mask)
            baseline_right, _ = model(packed_right, attention_mask=padding_mask)
            fixed_left, _ = model(
                packed_left,
                attention_mask=padding_mask,
                position_ids=reset_positions,
                document_ids=document_ids,
            )
            fixed_right, _ = model(
                packed_right,
                attention_mask=padding_mask,
                position_ids=reset_positions,
                document_ids=document_ids,
            )

        sample_b_slice = slice(4, 7)
        baseline_delta = (
            baseline_left[:, sample_b_slice, :] - baseline_right[:, sample_b_slice, :]
        ).abs().max().item()
        fixed_delta = (
            fixed_left[:, sample_b_slice, :] - fixed_right[:, sample_b_slice, :]
        ).abs().max().item()

        self.assertGreater(baseline_delta, 1e-6)
        self.assertLess(fixed_delta, 1e-5)

    def test_collator_pads_packed_masks_and_position_ids(self):
        dataset = ConstantLengthDataset(
            _FakePackedDataset(),
            seq_length=8,
            num_of_sequences=1,
        )
        collator = VQACollator(_FakeTokenizer(), max_length=10)
        packed = dataset._pack_one_group(
            [0, 1],
            [
                make_sample([11, 12, 13, 0]),
                make_sample([31, 32, 33, 0]),
            ],
            max_len=8,
        )

        batch = [
            {
                "input_ids": packed[0],
                "labels": packed[1],
                "attention_mask": packed[2],
                "images": packed[3],
                "position_ids": packed[4],
                "document_ids": packed[5],
            }
        ]
        collated = collator(batch)

        self.assertEqual(tuple(collated["input_ids"].shape), (1, 10))
        self.assertEqual(tuple(collated["position_ids"].shape), (1, 10))
        self.assertEqual(tuple(collated["document_ids"].shape), (1, 10))
        self.assertEqual(tuple(collated["attention_mask"].shape), (1, 10))
        self.assertEqual(
            collated["position_ids"][0, -8:].tolist(),
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        self.assertEqual(collated["document_ids"][0, -8:].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])
        self.assertEqual(collated["document_ids"][0, :2].tolist(), [-1, -1])
        self.assertEqual(collated["attention_mask"][0, -8:].tolist(), [1, 1, 1, 0, 1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
