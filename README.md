# network-gpu-ai-sample

A minimal, self-built alternative to a local inference platform (like [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/)) that offloads LLM inference over the LAN to a machine with a real GPU.

## Why

Modern mobile laptops (especially ARM64 / Copilot+ PCs) have an NPU or a weak integrated GPU that isn't strong enough for larger local models. If you happen to have an older desktop with a proper discrete GPU sitting on your home network, you can point inference requests at it instead - basically the same idea as calling a cloud AI API, just over your LAN instead of the internet.

This repo is two small, independent pieces:

- **`server/`** - a Rust HTTP server (Axum + [llama-cpp-2](https://docs.rs/llama-cpp-2)) that loads a GGUF model with CUDA and serves it over `/generate`. This is the machine with the GPU (in the reference setup: a GTX 1080 Ti on Ubuntu).
- **`client/`** - a Python benchmark client that compares this network server's generation speed against [Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/) running on the local machine.

```
[client machine, e.g. a laptop]  --HTTP (LAN)-->  [server machine, has a GPU]
        send prompt                                     run inference
        receive response          <-----------------------  return result
```

Foundry Local and llama.cpp solve overlapping but different problems: Foundry Local is a full platform (model catalog, automatic hardware selection, download/lifecycle management, an OpenAI-compatible service layer). llama.cpp is just the inference engine - no catalog, no automatic hardware selection, you bring your own GGUF file and pick your backend (CUDA/Metal/Vulkan/CPU) at build time. `server/` is a small hand-built service layer around llama.cpp, roughly playing the role Foundry Local's service layer plays for ONNX Runtime.

## `server/` - the inference server

### What it does

- Loads a single GGUF model at startup and offloads all layers to the GPU (`with_n_gpu_layers(1000)`).
- Exposes two endpoints:
  - `GET /health` - liveness check, doesn't touch the model.
  - `POST /generate` - `{"prompt": "..."}` in, `{"response": "..."}` out. Generation is capped at 64 tokens (`N_LEN` in `src/main.rs`) and stops early on an end-of-generation token.
- Runs the model on one dedicated OS thread (the GPU/model state isn't `Send`/`Sync`, so it can't be shared via a `Mutex`); HTTP handlers talk to it over an `mpsc` channel. This serializes all GPU access - concurrent requests queue rather than run in parallel.
- Clears the KV cache before every request, since the context is reused across requests. Without this, the *second* request onward would fail (the KV cache would remember token positions from the previous request and reject the new one starting at position 0).

### Requirements

Built and tested on:

| Component | Version used |
|---|---|
| OS | Ubuntu 22.04 |
| GPU | NVIDIA GeForce GTX 1080 Ti (11GB VRAM, Pascal architecture) |
| CUDA Toolkit | 11.5 (the Ubuntu distro package, `nvidia-cuda-toolkit`) |
| NVIDIA driver | 535.309.01 |
| Rust | 1.97.0 (via rustup) |
| gcc | 11.4.0 (system default) + gcc-10/g++-10 (needed as `nvcc`'s host compiler, see below) |
| libclang | required for `llama-cpp-2`'s bindgen step |

Install the extra build dependencies:

```bash
sudo apt update
sudo apt install -y libclang-dev clang cmake gcc-10 g++-10
```

### Build quirks on an older CUDA/GPU setup

If you're on a similarly old CUDA Toolkit + GPU combo, you'll likely hit two build errors that `server/.cargo/config.toml` already works around:

1. **`nvcc` (CUDA 11.5) can't parse gcc 11 headers** - CUDA 11.5 only officially supports up to gcc 10 as its host compiler. Building `.cu` kernels with gcc 11 headers fails with errors like `parameter packs not expanded with '...'`. Fixed by pointing `nvcc` at gcc-10 via `CUDAHOSTCXX`.
2. **Linker can't find `cudart_static`** - if CUDA Toolkit was installed via the Ubuntu distro package (not NVIDIA's official installer), `libcudart_static.a` ends up in `/usr/lib/x86_64-linux-gnu/` instead of the usual `/usr/local/cuda/lib64/`, and the build script doesn't find it automatically. Fixed with an explicit `-L` passed through `RUSTFLAGS`.

Both fixes are baked into `server/.cargo/config.toml`, so `cargo build`/`cargo run` should pick them up automatically on this repo - no need to `export` anything by hand. **If you're on a different CUDA version, a newer GPU, or NVIDIA's official CUDA installer, you may need to adjust or remove these settings.**

### Model

The GGUF model file is **not** included in this repo (it's ~4.1GB and gitignored). Download it yourself:

- Model: [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) (Mistral AI, Apache 2.0 license)
- Quantization: `Q4_K_M` (a good default balance of size vs. quality)
- Recommended source: [bartowski/Mistral-7B-Instruct-v0.3-GGUF](https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF) on Hugging Face

Place the file here:

```
server/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf
```

(To use a different model, edit `MODEL_PATH` in `server/src/main.rs`.)

### Running the server

```bash
cd server
cargo build            # first build compiles llama.cpp itself via CMake - can take ~15-20 minutes
cargo run
```

On success you'll see:

```
OK: worker ready. n_vocab = ..., n_ctx = 2048
listening on 0.0.0.0:8080
```

Quick test:

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of Japan is"}'
```

### Keeping it running (systemd)

`cargo run` (or a plain `nohup ... &`) ties the process's lifetime to whatever shell/SSH session started it - close the session and it dies (`SIGHUP`). For a server you want to leave running, register it as a systemd service instead:

```ini
# /etc/systemd/system/network-gpu-ai-server.service
[Unit]
Description=Network GPU AI inference server
After=network.target

[Service]
Type=simple
User=<your-user>
WorkingDirectory=/path/to/network-gpu-ai-sample/server
ExecStart=/path/to/network-gpu-ai-sample/server/target/debug/network-gpu-ai-server
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now network-gpu-ai-server
sudo systemctl status network-gpu-ai-server
journalctl -u network-gpu-ai-server -f
```

This also survives a reboot (`enable`) and auto-restarts on crash (`Restart=on-failure`).

### Known limitations

- Ships as a debug build (`target/debug/`). Switching `ExecStart` to a `--release` build would improve throughput, at the cost of a much longer first build (llama.cpp gets recompiled in release mode too).
- No auth - anything that can reach port 8080 can use the model. Fine on a trusted LAN, not something to expose past your router.
- One request at a time by design (see "What it does" above) - this is a single-GPU, single-tenant toy server, not a production inference service.

## `client/` - the benchmark client

Compares this server's generation speed against [Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/) running on your own machine, using the same prompt, the same `max_tokens`, and greedy (deterministic) decoding on both sides.

### Setup

```bash
cd client
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
```

`foundry-local-sdk` is only needed if you want to run the `foundry` backend; it requires [Foundry Local](https://learn.microsoft.com/en-us/azure/foundry-local/get-started) to be installed on the machine you run the client from.

### Pointing at your server

The server URL defaults to a placeholder (`http://192.168.1.100:8080`). Point it at your actual server either way:

```bash
# one-off
python benchmark.py --network-url http://<your-server-ip>:8080

# or via env var
set NETWORK_GPU_AI_SERVER_URL=http://<your-server-ip>:8080   # Windows (cmd)
$env:NETWORK_GPU_AI_SERVER_URL="http://<your-server-ip>:8080" # Windows (PowerShell)
export NETWORK_GPU_AI_SERVER_URL=http://<your-server-ip>:8080 # Linux/macOS
```

### Usage

```bash
python benchmark.py --backend both                         # run both backends, compare
python benchmark.py --backend network --prompt "..."
python benchmark.py --backend foundry --foundry-model mistral-7b-v0.2
python benchmark.py --backend both --runs 10 --warmup 2     # more accurate averaging
```

| Flag | Default | Meaning |
|---|---|---|
| `--backend` | `both` | `foundry`, `network`, or `both` |
| `--prompt` | `"The capital of Ireland is"` | Prompt sent to whichever backend(s) are selected |
| `--network-url` | env var or placeholder | Base URL of the `network-gpu-ai-server` instance |
| `--foundry-model` | `mistral-7b-v0.2` | Foundry Local catalog alias to benchmark |
| `--runs` | `5` | Timed runs to average over |
| `--warmup` | `1` | Untimed warmup runs before timing starts |

### Sample output

```
=== network (1080Ti / network-gpu-ai-server) ===
runs: 5
tokens (approx, word count): 50
time: 1.11s ± 0.01s (min 1.10s, max 1.12s)
tokens/sec: 45.19 ± 0.31
last response: ...

=== foundry local (mistralai-Mistral-7B-Instruct-v0-2-generic-gpu:2) ===
runs: 5
tokens (exact): 64
time: 10.18s ± 0.02s (min 10.15s, max 10.22s)
tokens/sec: 6.29 ± 0.01
last response: ...

=== comparison ===
network (1080Ti / network-gpu-ai-server) was 9.20x faster on average (by mean wall-clock time)
```

(Numbers above are from the reference setup: an aging GTX 1080 Ti beating a Copilot+ PC's on-device GPU by roughly 9x on the same model. Your mileage will vary with your own hardware on both ends.)

### Notes

- `network-gpu-ai-server` always generates up to 64 tokens, so the client caps Foundry Local at `max_tokens=64` too for a fair comparison.
- The network server's response doesn't include a token count, so tokens/sec on that side is approximated from whitespace word count. The Foundry Local side uses the API's exact `usage.completion_tokens`.
- Foundry Local's catalog only hides GPU/NPU variants of a model *before* execution providers are registered (`download_and_register_eps()`). If you check `model.variants` too early, a model that does support your GPU can look CPU-only.
