import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.language_model import LanguageModel


def make_test_config():
    return SimpleNamespace(
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


def build_inputs():
    pad_token_id = 0
    sample_a_left = [11, 12, 13]
    sample_a_right = [21, 22, 23]
    sample_b = [31, 32, 33]

    packed_left = torch.tensor(
        [[*sample_a_left, pad_token_id, *sample_b, pad_token_id]], dtype=torch.long
    )
    packed_right = torch.tensor(
        [[*sample_a_right, pad_token_id, *sample_b, pad_token_id]], dtype=torch.long
    )
    packed_padding_mask = torch.tensor(
        [[1, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.long
    )
    packed_reset_positions = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long
    )

    packed_document_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.long)

    sample_b_only = torch.tensor([[*sample_b, pad_token_id]], dtype=torch.long)
    sample_b_only_padding_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)
    sample_b_only_positions = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

    return {
        "packed_left": packed_left,
        "packed_right": packed_right,
        "packed_padding_mask": packed_padding_mask,
        "packed_reset_positions": packed_reset_positions,
        "packed_document_ids": packed_document_ids,
        "sample_b_only": sample_b_only,
        "sample_b_only_padding_mask": sample_b_only_padding_mask,
        "sample_b_only_positions": sample_b_only_positions,
        "sample_b_start": 4,
        "sample_b_length": len(sample_b),
    }


def run_model(model, input_ids, attention_mask, position_ids=None, document_ids=None):
    with torch.no_grad():
        logits, _ = model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            document_ids=document_ids,
            start_pos=0,
        )
    return logits


def max_abs_delta(lhs, rhs):
    return (lhs - rhs).abs().max().item()


def compare_mode(
    model,
    name,
    packed_left,
    packed_right,
    packed_attention_mask,
    sample_b_only,
    sample_b_only_attention_mask,
    sample_b_slice,
    packed_position_ids=None,
    sample_b_only_position_ids=None,
    packed_document_ids=None,
):
    packed_left_logits = run_model(
        model,
        packed_left,
        packed_attention_mask,
        position_ids=packed_position_ids,
        document_ids=packed_document_ids,
    )
    packed_right_logits = run_model(
        model,
        packed_right,
        packed_attention_mask,
        position_ids=packed_position_ids,
        document_ids=packed_document_ids,
    )
    sample_b_only_logits = run_model(
        model,
        sample_b_only,
        sample_b_only_attention_mask,
        position_ids=sample_b_only_position_ids,
    )

    packed_left_sample_b_logits = packed_left_logits[:, sample_b_slice, :]
    packed_right_sample_b_logits = packed_right_logits[:, sample_b_slice, :]
    sample_b_only_slice = sample_b_only_logits[:, : sample_b_slice.stop - sample_b_slice.start, :]

    prefix_delta = max_abs_delta(
        packed_left_sample_b_logits,
        packed_right_sample_b_logits,
    )
    packed_vs_single_delta = max_abs_delta(
        packed_left_sample_b_logits,
        sample_b_only_slice,
    )

    return {
        "name": name,
        "prefix_delta": prefix_delta,
        "packed_vs_single_delta": packed_vs_single_delta,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare candidate fixes for issue #201 on packed sequences."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = LanguageModel(make_test_config())
    model.eval()

    inputs = build_inputs()
    sample_b_slice = slice(
        inputs["sample_b_start"],
        inputs["sample_b_start"] + inputs["sample_b_length"],
    )

    results = [
        compare_mode(
            model,
            name="baseline_contiguous_positions",
            packed_left=inputs["packed_left"],
            packed_right=inputs["packed_right"],
            packed_attention_mask=inputs["packed_padding_mask"],
            sample_b_only=inputs["sample_b_only"],
            sample_b_only_attention_mask=inputs["sample_b_only_padding_mask"],
            sample_b_slice=sample_b_slice,
        ),
        compare_mode(
            model,
            name="reset_positions_only",
            packed_left=inputs["packed_left"],
            packed_right=inputs["packed_right"],
            packed_attention_mask=inputs["packed_padding_mask"],
            sample_b_only=inputs["sample_b_only"],
            sample_b_only_attention_mask=inputs["sample_b_only_padding_mask"],
            sample_b_slice=sample_b_slice,
            packed_position_ids=inputs["packed_reset_positions"],
            sample_b_only_position_ids=inputs["sample_b_only_positions"],
        ),
        compare_mode(
            model,
            name="reset_positions_plus_document_ids",
            packed_left=inputs["packed_left"],
            packed_right=inputs["packed_right"],
            packed_attention_mask=inputs["packed_padding_mask"],
            sample_b_only=inputs["sample_b_only"],
            sample_b_only_attention_mask=inputs["sample_b_only_padding_mask"],
            sample_b_slice=sample_b_slice,
            packed_position_ids=inputs["packed_reset_positions"],
            sample_b_only_position_ids=inputs["sample_b_only_positions"],
            packed_document_ids=inputs["packed_document_ids"],
        ),
    ]

    print("Issue #201 candidate-fix comparison")
    print(f"seed: {args.seed}")
    for result in results:
        print(result["name"] + ":")
        print(f"  prefix_delta:          {result['prefix_delta']:.6f}")
        print(f"  packed_vs_single_delta:{result['packed_vs_single_delta']:.6f}")

    baseline = results[0]
    reset_only = results[1]
    reset_plus_mask = results[2]

    if baseline["prefix_delta"] <= 1e-6:
        raise AssertionError("Baseline should show sample-B dependence on sample-A prefix.")
    if reset_only["prefix_delta"] <= 1e-6:
        raise AssertionError(
            "Resetting position ids alone should still leave cross-sample attention active."
        )
    if reset_plus_mask["prefix_delta"] >= 1e-5:
        raise AssertionError(
            "Adding a document mask should remove sample-B dependence on the sample-A prefix."
        )
    if reset_plus_mask["packed_vs_single_delta"] >= 1e-5:
        raise AssertionError(
            "With reset position ids and a document mask, packed sample B should match sample B alone."
        )

    print("comparison_status: OK")


if __name__ == "__main__":
    main()
