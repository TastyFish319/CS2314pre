"""Optional GPU experiment; not run as part of the tutorial's theoretical analysis.

Compatibility baseline: Python 3.10/3.11, CUDA PyTorch, transformers==4.32.1,
accelerate, tiktoken, einops, transformers_stream_generator.
Review and pin the original Qwen remote code with --revision before running.
This script reports timing only: it does not measure FLOPs or HBM traffic.
"""
import argparse
import contextlib
import json
import statistics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", required=True, help="Reviewed Qwen/Qwen-7B commit SHA")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--length", type=int, default=1024)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if min(args.batch, args.length, args.steps, args.repeats) <= 0 or args.warmup < 0:
        p.error("batch, length, steps, repeats must be positive; warmup must be nonnegative")
    if len(args.revision) != 40 or any(c not in "0123456789abcdefABCDEF" for c in args.revision):
        p.error("--revision must be a full 40-character commit SHA, not main")
    return args


def main():
    args = parse_args()
    # Defer optional dependencies so --help works without a GPU environment.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required; no CPU/offload fallback is used.")
    device = torch.device("cuda:0")
    model_id = "Qwen/Qwen-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
        fp16=True, bf16=False, fp32=False, use_flash_attn=False,
    ).to(device).eval()
    token_id = tokenizer.encode("hello", add_special_tokens=False)[0]
    prompt = torch.full((args.batch, args.length), token_id, dtype=torch.long, device=device)

    @contextlib.contextmanager
    def nvtx(name):
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()

    def event():
        return torch.cuda.Event(enable_timing=True)

    @torch.inference_mode()
    def run_once(record):
        # No padding or prefix cache: each repetition starts at identical prompt length.
        if record:
            pre_start, pre_end = event(), event()
        with nvtx("prefill" if record else "warmup_prefill"):
            if record:
                pre_start.record()
            out = model(input_ids=prompt, use_cache=True, return_dict=True)
            cache = out.past_key_values
            current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if record:
                pre_end.record()
            del out
        pairs = []
        for _ in range(args.steps):
            # Input the selected token; append its K/V and predict the next token.
            if record:
                start, end = event(), event()
            with nvtx("decode_step" if record else "warmup_decode"):
                if record:
                    start.record()
                out = model(input_ids=current, past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
                current = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                if record:
                    end.record()
                    pairs.append((start, end))
                del out
        torch.cuda.synchronize(device)
        if record:
            return pre_start.elapsed_time(pre_end), [a.elapsed_time(b) for a, b in pairs]
        return None

    for _ in range(args.warmup):
        run_once(False)
    results = [run_once(True) for _ in range(args.repeats)]
    prefill = [row[0] for row in results]
    decode = [row[1] for row in results]
    mean_step = statistics.mean(x for row in decode for x in row)
    summary = {
        "notice": "CUDA-event stage timing from this run; not end-to-end service TTFT. FLOPs/HBM bytes not measured.",
        "model": model_id, "revision": args.revision,
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "transformers": transformers.__version__, "dtype": str(next(model.parameters()).dtype),
        "external_flash_attention": False,
        "attention_note": "Qwen remote implementation may use PyTorch SDPA; inspect actual kernel names.",
        "exact_loaded_parameter_numel": sum(p.numel() for p in model.parameters()),
        "batch": args.batch, "prompt_length": args.length, "decode_steps": args.steps,
        "history_length_before_decode_min": args.length,
        "history_length_before_decode_max": args.length + args.steps - 1,
        "input_kind": "synthetic repeated valid token ID; greedy selection; fixed steps, EOS ignored",
        "prefill_includes_first_argmax": True,
        "prefill_ms_each_repeat": prefill,
        "prefill_ms_median": statistics.median(prefill),
        "decode_ms_each_repeat_and_step": decode,
        "decode_TPOT_ms_mean": mean_step,
        "aggregate_decode_tokens_per_second_from_device_stage_time": args.batch * 1000 / mean_step,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
