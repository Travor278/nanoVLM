import argparse
import gc
import math
import sys
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.language_model import LanguageModel, _to_additive_attention_mask


def make_config(
    hidden_dim,
    inter_dim,
    vocab_size,
    n_heads,
    n_kv_heads,
    n_blocks,
):
    return SimpleNamespace(
        lm_hidden_dim=hidden_dim,
        lm_inter_dim=inter_dim,
        lm_rms_eps=1e-5,
        lm_re_base=10000.0,
        lm_max_position_embeddings=8192,
        lm_attn_scaling=1.0,
        lm_vocab_size=vocab_size,
        lm_n_heads=n_heads,
        lm_n_kv_heads=n_kv_heads,
        lm_dropout=0.0,
        lm_n_blocks=n_blocks,
        lm_use_tokens=True,
        lm_tie_weights=True,
    )


def format_bytes(num_bytes):
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GiB"


def build_inputs(batch_size, seq_len, doc_span, vocab_size, device):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)

    position_ids = torch.arange(seq_len, device=device).repeat(batch_size, 1)
    document_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)

    for batch_idx in range(batch_size):
        for start in range(0, seq_len, doc_span):
            end = min(start + doc_span, seq_len)
            position_ids[batch_idx, start:end] = torch.arange(end - start, device=device)
            document_ids[batch_idx, start:end] = start // doc_span

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "document_ids": document_ids,
    }


def persistent_extra_bytes(position_ids, document_ids):
    return (
        position_ids.numel() * position_ids.element_size()
        + document_ids.numel() * document_ids.element_size()
    )


def persistent_extra_bytes_from_shape(batch_size, seq_len, dtype=torch.long):
    element_size = torch.tensor([], dtype=dtype).element_size()
    return 2 * batch_size * seq_len * element_size


def theoretical_mask_bytes(batch_size, seq_len, q_dtype):
    same_document_bytes = batch_size * seq_len * seq_len
    additive_mask_bytes = batch_size * seq_len * seq_len * torch.tensor([], dtype=q_dtype).element_size()
    return same_document_bytes, additive_mask_bytes


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_mask_builder(device, batch_size, seq_len, q_dtype, steps, warmup, use_document_ids):
    q = torch.zeros((batch_size, 1, seq_len, 64), dtype=q_dtype, device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    document_ids = torch.arange(seq_len, device=device).div(128, rounding_mode="floor").repeat(batch_size, 1)

    for _ in range(warmup):
        _ = _to_additive_attention_mask(
            attention_mask,
            q,
            seq_len,
            seq_len,
            document_ids=document_ids if use_document_ids else None,
        )
    synchronize(device)

    times_ms = []
    peak_bytes = 0
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        mask = _to_additive_attention_mask(
            attention_mask,
            q,
            seq_len,
            seq_len,
            document_ids=document_ids if use_document_ids else None,
        )
        synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)
        if device.type == "cuda":
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))
        del mask
    return {
        "avg_ms": mean(times_ms),
        "peak_bytes": peak_bytes,
    }


def benchmark_training_mode(mode_name, device, cfg, inputs, steps, warmup):
    torch.manual_seed(0)
    model = LanguageModel(cfg).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    use_document_ids = mode_name == "document_ids"

    def run_step():
        optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        )
        with autocast_context:
            logits, _ = model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                position_ids=inputs["position_ids"] if use_document_ids else None,
                document_ids=inputs["document_ids"] if use_document_ids else None,
            )
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                inputs["labels"].reshape(-1),
            )
        loss.backward()
        optimizer.step()
        return loss.item()

    for _ in range(warmup):
        _ = run_step()
    synchronize(device)

    times_ms = []
    peak_bytes = 0
    losses = []
    for _ in range(steps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        losses.append(run_step())
        synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000)
        if device.type == "cuda":
            peak_bytes = max(peak_bytes, torch.cuda.max_memory_allocated(device))

    return {
        "avg_ms": mean(times_ms),
        "peak_bytes": peak_bytes,
        "avg_loss": mean(losses),
    }


def print_mode_stats(name, stats, baseline=None):
    print(f"{name}:")
    print(f"  avg_step_ms:      {stats['avg_ms']:.2f}")
    print(f"  peak_memory:      {format_bytes(stats['peak_bytes'])}")
    print(f"  avg_loss:         {stats['avg_loss']:.4f}")
    if baseline is not None and baseline["avg_ms"] > 0:
        print(f"  step_overhead:    {(stats['avg_ms'] / baseline['avg_ms'] - 1.0) * 100:.2f}%")
    if baseline is not None and baseline["peak_bytes"] > 0:
        print(f"  memory_overhead:  {(stats['peak_bytes'] / baseline['peak_bytes'] - 1.0) * 100:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the training-time cost of issue #201 document_ids masking."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--doc-span", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=960)
    parser.add_argument("--inter-dim", type=int, default=2560)
    parser.add_argument("--vocab-size", type=int, default=49152)
    parser.add_argument("--n-heads", type=int, default=15)
    parser.add_argument("--n-kv-heads", type=int, default=5)
    parser.add_argument("--n-blocks", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--mask-batch-size", type=int, default=2)
    parser.add_argument("--mask-seq-len", type=int, default=4096)
    parser.add_argument(
        "--train-mode",
        choices=("both", "baseline", "document_ids"),
        default="both",
    )
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    cfg = make_config(
        hidden_dim=args.hidden_dim,
        inter_dim=args.inter_dim,
        vocab_size=args.vocab_size,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        n_blocks=args.n_blocks,
    )
    inputs = build_inputs(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        doc_span=args.doc_span,
        vocab_size=args.vocab_size,
        device=device,
    )

    print("Issue #201 document_ids cost benchmark")
    print(
        f"device={device}, batch_size={args.batch_size}, seq_len={args.seq_len}, "
        f"doc_span={args.doc_span}, hidden_dim={args.hidden_dim}, blocks={args.n_blocks}"
    )
    print(
        "benchmark batch overhead from extra tensors: "
        f"{format_bytes(persistent_extra_bytes(inputs['position_ids'], inputs['document_ids']))}"
    )
    print(
        f"target config extra tensors @ B={args.mask_batch_size}, T={args.mask_seq_len}: "
        f"{format_bytes(persistent_extra_bytes_from_shape(args.mask_batch_size, args.mask_seq_len))}"
    )

    if not args.skip_training:
        baseline_stats = None
        document_stats = None

        if args.train_mode in ("both", "baseline"):
            baseline_stats = benchmark_training_mode(
                "baseline",
                device,
                cfg,
                inputs,
                steps=args.steps,
                warmup=args.warmup,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            print_mode_stats("baseline", baseline_stats)

        if args.train_mode in ("both", "document_ids"):
            document_stats = benchmark_training_mode(
                "document_ids",
                device,
                cfg,
                inputs,
                steps=args.steps,
                warmup=args.warmup,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            print_mode_stats("document_ids", document_stats, baseline=baseline_stats)

    q_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    same_document_bytes, additive_mask_bytes = theoretical_mask_bytes(
        args.mask_batch_size,
        args.mask_seq_len,
        q_dtype,
    )
    baseline_mask_stats = benchmark_mask_builder(
        device,
        batch_size=args.mask_batch_size,
        seq_len=args.mask_seq_len,
        q_dtype=q_dtype,
        steps=args.steps,
        warmup=args.warmup,
        use_document_ids=False,
    )
    document_mask_stats = benchmark_mask_builder(
        device,
        batch_size=args.mask_batch_size,
        seq_len=args.mask_seq_len,
        q_dtype=q_dtype,
        steps=args.steps,
        warmup=args.warmup,
        use_document_ids=True,
    )
    print("mask_only_estimate:")
    print(
        f"  theoretical_same_document_bool @ B={args.mask_batch_size}, T={args.mask_seq_len}: "
        f"{format_bytes(same_document_bytes)}"
    )
    print(
        f"  theoretical_additive_mask @ B={args.mask_batch_size}, T={args.mask_seq_len}: "
        f"{format_bytes(additive_mask_bytes)}"
    )
    print(
        f"  baseline_mask_build_avg_ms @ B={args.mask_batch_size}, "
        f"T={args.mask_seq_len}: {baseline_mask_stats['avg_ms']:.2f}"
    )
    print(
        f"  document_ids_mask_build_avg_ms @ B={args.mask_batch_size}, "
        f"T={args.mask_seq_len}: {document_mask_stats['avg_ms']:.2f}"
    )
    if device.type == "cuda":
        print(
            f"  baseline_mask_build_peak: {format_bytes(baseline_mask_stats['peak_bytes'])}"
        )
        print(
            f"  document_ids_mask_build_peak: {format_bytes(document_mask_stats['peak_bytes'])}"
        )


if __name__ == "__main__":
    main()
