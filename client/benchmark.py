"""
Benchmark client comparing Foundry Local (this machine's NPU/GPU/CPU) against
network-gpu-ai-server (the 1080Ti over LAN).

Usage:
    python benchmark.py --backend both
    python benchmark.py --backend network --prompt "..."
    python benchmark.py --backend foundry --foundry-model mistral-7b-v0.2
    python benchmark.py --backend both --runs 10 --warmup 2

Notes:
- network-gpu-ai-server always caps generation at 64 tokens (N_LEN on the server side),
  so max_tokens=64 is used on the Foundry side too for a fair comparison.
- The network server's response has no token count, so tokens/sec on that side is an
  approximation based on whitespace word count. The Foundry side uses the API's
  usage.completion_tokens, which is exact.
- The default model is Mistral-7B-Instruct on both sides for a fair comparison
  (network side = v0.3, Q4_K_M quantized GGUF; Foundry side = v0.2, ONNX). The version
  mismatch is because Foundry Local's catalog only has v0.2, not v0.3 - this is the
  closest match at the same 7B size and model family.
- The GPU variant of mistral-7b-v0.2
  (`mistralai-Mistral-7B-Instruct-v0-2-generic-gpu`) only shows up in the catalog
  after execution providers have been registered via download_and_register_eps().
  Calling catalog.get_model() before that only shows generic-cpu, which looks like
  the model has no GPU support even though it does. This script registers EPs first,
  so it correctly picks up the GPU variant.
- Both backends use greedy decoding (temperature=0 / the network server's dist+greedy
  sampler chain) for a fixed prompt, so token count is identical across repeated runs;
  only wall-clock time varies run to run. --runs/--warmup exist to average out that
  timing noise, not to sample different outputs.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

if sys.stdout.encoding is None or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_PROMPT = "The capital of Ireland is"
# Point this at your own network-gpu-ai-server via the NETWORK_GPU_AI_SERVER_URL env
# var, or override per-run with --network-url.
DEFAULT_NETWORK_URL = os.environ.get("NETWORK_GPU_AI_SERVER_URL", "http://192.168.1.100:8080")
DEFAULT_FOUNDRY_MODEL = "mistral-7b-v0.2"
DEFAULT_RUNS = 5
DEFAULT_WARMUP = 1
MAX_TOKENS = 64  # match network-gpu-ai-server's N_LEN=64


def call_network(prompt: str, base_url: str) -> dict:
    url = f"{base_url}/generate"
    payload = json.dumps({"prompt": prompt}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Failed to connect to network-gpu-ai-server ({base_url}): {e}"
        ) from e
    elapsed = time.perf_counter() - start
    text = body["response"]
    return {
        "text": text,
        "elapsed_sec": elapsed,
        "tokens": len(text.split()),
        "tokens_exact": False,
    }


def run_network(prompt: str, base_url: str, runs: int, warmup: int) -> dict:
    for _ in range(warmup):
        call_network(prompt, base_url)

    samples = [call_network(prompt, base_url) for _ in range(runs)]
    return summarize("network (1080Ti / network-gpu-ai-server)", samples)


def run_foundry(prompt: str, model_alias: str, runs: int, warmup: int) -> dict:
    from foundry_local_sdk import Configuration, FoundryLocalManager
    from foundry_local_sdk.logging_helper import LogLevel

    config = Configuration(app_name="network_gpu_benchmark", log_level=LogLevel.WARNING)
    FoundryLocalManager.initialize(config)
    mgr = FoundryLocalManager.instance

    mgr.download_and_register_eps(
        progress_callback=lambda name, percent: print(
            f"\r  Registering EP: {name:<20} {percent:6.1f}%", end=""
        )
    )
    print()

    model = mgr.catalog.get_model(model_alias)
    if model is None:
        raise RuntimeError(f"Model not found: {model_alias}")

    print("Available variants:")
    for v in model.variants:
        print(f"  {v.id}")

    # Prefer an NPU/GPU variant if one exists; fall back to the default variant otherwise.
    preferred = next(
        (
            v
            for v in model.variants
            if "qnn" in v.id.lower() or "npu" in v.id.lower() or "gpu" in v.id.lower()
        ),
        model.variants[0],
    )
    model.select_variant(preferred)
    print(f"Selected variant: {preferred.id}")

    model.download(progress_callback=lambda p: print(f"\rDownloading: {p:.1f}%", end=""))
    print()
    model.load()

    def call_foundry() -> dict:
        start = time.perf_counter()
        completion = chat_client.complete_chat([{"role": "user", "content": prompt}])
        elapsed = time.perf_counter() - start

        text = completion.choices[0].message.content
        usage = completion.usage
        tokens_exact = usage is not None and usage.completion_tokens is not None
        tokens = usage.completion_tokens if tokens_exact else len(text.split())
        return {
            "text": text,
            "elapsed_sec": elapsed,
            "tokens": tokens,
            "tokens_exact": tokens_exact,
        }

    try:
        chat_client = model.get_chat_client()
        chat_client.settings.max_tokens = MAX_TOKENS
        chat_client.settings.temperature = 0  # match the network side's greedy sampling

        for _ in range(warmup):
            call_foundry()

        samples = [call_foundry() for _ in range(runs)]
    finally:
        model.unload()

    return summarize(f"foundry local ({preferred.id})", samples)


def summarize(backend: str, samples: list) -> dict:
    times = [s["elapsed_sec"] for s in samples]
    tps_values = [s["tokens"] / s["elapsed_sec"] for s in samples if s["elapsed_sec"] > 0]
    return {
        "backend": backend,
        "runs": len(samples),
        "tokens_exact": samples[0]["tokens_exact"],
        "tokens": samples[-1]["tokens"],
        "last_response": samples[-1]["text"],
        "mean_time": statistics.mean(times),
        "stdev_time": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_time": min(times),
        "max_time": max(times),
        "mean_tps": statistics.mean(tps_values) if tps_values else float("nan"),
        "stdev_tps": statistics.stdev(tps_values) if len(tps_values) > 1 else 0.0,
    }


def print_result(result: dict) -> None:
    tag = "exact" if result["tokens_exact"] else "approx, word count"
    print(f"\n=== {result['backend']} ===")
    print(f"runs: {result['runs']}")
    print(f"tokens ({tag}): {result['tokens']}")
    print(
        f"time: {result['mean_time']:.2f}s "
        f"± {result['stdev_time']:.2f}s "
        f"(min {result['min_time']:.2f}s, max {result['max_time']:.2f}s)"
    )
    print(f"tokens/sec: {result['mean_tps']:.2f} ± {result['stdev_tps']:.2f}")
    print(f"last response: {result['last_response']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundry Local vs Network GPU benchmark")
    parser.add_argument("--backend", choices=["foundry", "network", "both"], default="both")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--network-url", default=DEFAULT_NETWORK_URL)
    parser.add_argument("--foundry-model", default=DEFAULT_FOUNDRY_MODEL)
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="Number of timed runs to average over"
    )
    parser.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Untimed warmup runs before timing"
    )
    args = parser.parse_args()

    results = []
    if args.backend in ("network", "both"):
        results.append(run_network(args.prompt, args.network_url, args.runs, args.warmup))
    if args.backend in ("foundry", "both"):
        results.append(run_foundry(args.prompt, args.foundry_model, args.runs, args.warmup))

    for r in results:
        print_result(r)

    if len(results) == 2:
        faster = min(results, key=lambda r: r["mean_time"])
        slower = max(results, key=lambda r: r["mean_time"])
        ratio = slower["mean_time"] / faster["mean_time"]
        print("\n=== comparison ===")
        print(f"{faster['backend']} was {ratio:.2f}x faster on average (by mean wall-clock time)")


if __name__ == "__main__":
    main()
