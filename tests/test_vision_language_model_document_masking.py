import unittest
import sys
import types

import torch

stub_processors = types.ModuleType("data.processors")
stub_processors.get_tokenizer = lambda *args, **kwargs: None
sys.modules.setdefault("data.processors", stub_processors)

from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


class FakeTokenizer:
    image_token_id = 999


class TestVisionLanguageModelDocumentMasking(unittest.TestCase):
    def test_forward_accepts_position_ids_and_document_ids(self):
        cfg = VLMConfig(
            vit_model_type="testing",
            vit_patch_size=16,
            vit_hidden_dim=48,
            vit_inter_dim=96,
            vit_n_heads=3,
            vit_n_blocks=1,
            vit_img_size=32,
            vit_dropout=0.0,
            lm_model_type="testing",
            lm_hidden_dim=64,
            lm_inter_dim=128,
            lm_rms_eps=1e-5,
            lm_re_base=10000.0,
            lm_max_position_embeddings=512,
            lm_attn_scaling=1.0,
            lm_vocab_size=128,
            lm_n_heads=4,
            lm_n_kv_heads=2,
            lm_dropout=0.0,
            lm_n_blocks=2,
            lm_use_tokens=False,
            lm_tie_weights=True,
            mp_pixel_shuffle_factor=2,
        )

        original_get_tokenizer = sys.modules["models.vision_language_model"].get_tokenizer
        sys.modules["models.vision_language_model"].get_tokenizer = lambda *args, **kwargs: FakeTokenizer()
        self.addCleanup(
            lambda: setattr(
                sys.modules["models.vision_language_model"],
                "get_tokenizer",
                original_get_tokenizer,
            )
        )

        model = VisionLanguageModel(cfg, load_backbone=False)
        model.eval()

        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len))
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 1, 1, 1, 0],
                [1, 1, 0, 1, 1, 0, 1, 1],
            ],
            dtype=torch.long,
        )
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3, 0, 1, 2, 3],
                [0, 1, 2, 0, 1, 2, 0, 1],
            ],
            dtype=torch.long,
        )
        document_ids = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 0, 1, 1, 1, 2, 2],
            ],
            dtype=torch.long,
        )
        labels = torch.randint(0, cfg.lm_vocab_size, (batch_size, seq_len))
        images = [[] for _ in range(batch_size)]

        logits, loss = model(
            input_ids,
            images,
            attention_mask=attention_mask,
            targets=labels,
            position_ids=position_ids,
            document_ids=document_ids,
        )

        self.assertEqual(tuple(logits.shape), (batch_size, seq_len, cfg.lm_vocab_size))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
