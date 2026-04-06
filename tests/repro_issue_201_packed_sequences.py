import argparse
import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

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


def build_packed_inputs():
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
    packed_attention_mask = torch.tensor(
        [[1, 1, 1, 0, 1, 1, 1, 0]], dtype=torch.long
    )

    sample_b_only = torch.tensor([[*sample_b, pad_token_id]], dtype=torch.long)
    sample_b_only_attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.long)

    return {
        "packed_left": packed_left,
        "packed_right": packed_right,
        "packed_attention_mask": packed_attention_mask,
        "sample_b_only": sample_b_only,
        "sample_b_only_attention_mask": sample_b_only_attention_mask,
        "sample_b_start": 4,
        "sample_b_length": len(sample_b),
    }


def run_and_capture_position_ids(model, input_ids, attention_mask):
    captured = {}
    original_forward = model.rotary_embd.forward

    def wrapped_forward(self, position_ids):
        captured["position_ids"] = position_ids.detach().clone()
        return original_forward(position_ids)

    model.rotary_embd.forward = MethodType(wrapped_forward, model.rotary_embd)
    try:
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask=attention_mask, start_pos=0)
    finally:
        model.rotary_embd.forward = original_forward

    return captured["position_ids"], logits


def max_abs_delta(lhs, rhs):
    return (lhs - rhs).abs().max().item()


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce issue #201 with packed sequences and contiguous position ids."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = LanguageModel(make_test_config())
    model.eval()

    inputs = build_packed_inputs()

    packed_position_ids, packed_left_logits = run_and_capture_position_ids(
        model,
        inputs["packed_left"],
        inputs["packed_attention_mask"],
    )
    _, packed_right_logits = run_and_capture_position_ids(
        model,
        inputs["packed_right"],
        inputs["packed_attention_mask"],
    )
    sample_b_only_position_ids, sample_b_only_logits = run_and_capture_position_ids(
        model,
        inputs["sample_b_only"],
        inputs["sample_b_only_attention_mask"],
    )

    sample_b_slice = slice(
        inputs["sample_b_start"],
        inputs["sample_b_start"] + inputs["sample_b_length"],
    )
    packed_left_sample_b_logits = packed_left_logits[:, sample_b_slice, :]
    packed_right_sample_b_logits = packed_right_logits[:, sample_b_slice, :]

    prefix_delta = max_abs_delta(
        packed_left_sample_b_logits,
        packed_right_sample_b_logits,
    )
    packed_vs_single_delta = max_abs_delta(
        packed_left_sample_b_logits,
        sample_b_only_logits[:, : inputs["sample_b_length"], :],
    )

    sample_b_start_position = packed_position_ids[0, inputs["sample_b_start"]].item()
    expected_contiguous_positions = list(range(inputs["packed_left"].shape[1]))

    print("Packed sequence reproduction for issue #201")
    print(f"seed: {args.seed}")
    print(f"packed_input_ids:            {inputs['packed_left'][0].tolist()}")
    print(
        f"packed_attention_mask:       {inputs['packed_attention_mask'][0].tolist()}"
    )
    print(f"captured_position_ids:       {packed_position_ids[0].tolist()}")
    print(
        f"sample_b_only_position_ids:  {sample_b_only_position_ids[0].tolist()}"
    )
    print(
        f"sample_b starts at position: {sample_b_start_position} "
        f"(packed) vs {sample_b_only_position_ids[0, 0].item()} (single sample)"
    )
    print(f"expected contiguous ids:     {expected_contiguous_positions}")
    print(f"max_delta_from_prefix_swap:  {prefix_delta:.6f}")
    print(f"max_delta_packed_vs_single:  {packed_vs_single_delta:.6f}")

    if packed_position_ids[0].tolist() != expected_contiguous_positions:
        raise AssertionError(
            "The reproduction expected contiguous packed position ids, but saw a different pattern."
        )
    if sample_b_start_position == 0:
        raise AssertionError(
            "The packed reproduction expected sample B to start at a non-zero position id."
        )
    if prefix_delta <= 1e-6:
        raise AssertionError(
            "Changing sample A should affect sample B logits if cross-sample attention is enabled."
        )
    if packed_vs_single_delta <= 1e-6:
        raise AssertionError(
            "Packed sample B should differ from sample B alone under the current training semantics."
        )

    print("reproduction_status: OK")


if __name__ == "__main__":
    main()
