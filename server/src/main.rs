use anyhow::Result;
use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use tokio::sync::{mpsc, oneshot};

const MODEL_PATH: &str = "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf";
const N_LEN: i32 = 64;

/// Initializes the llama.cpp execution backend. Backend init (CUDA/CPU, etc.) and the
/// thread pool are done globally, exactly once, inside this call. All later processing
/// goes through this backend.
fn init_backend() -> Result<LlamaBackend> {
    Ok(LlamaBackend::init()?)
}

/// Loads the model weights from a GGUF file.
/// with_n_gpu_layers(1000) means "offload up to 1000 layers to the GPU". The actual
/// model only has around 32-33 layers, so in practice this means "offload every layer"
/// (the request for extra layers beyond what the model has is simply ignored). This is
/// where the transfer to VRAM happens.
fn load_model(backend: &LlamaBackend, path: &str) -> Result<LlamaModel> {
    let params = LlamaModelParams::default().with_n_gpu_layers(1000);
    Ok(LlamaModel::load_from_file(backend, path, &params)?)
}

/// Creates the inference context (the state for one conversation session, including the
/// KV cache and compute buffers). n_ctx is the maximum number of tokens this context can
/// handle (prompt + generation combined); the larger it is, the more VRAM it uses.
fn create_context<'a>(model: &'a LlamaModel, backend: &LlamaBackend) -> Result<LlamaContext<'a>> {
    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));
    Ok(model.new_context(backend, ctx_params)?)
}

/// Converts a prompt string into the integer token sequence the model understands.
/// AddBos::Always always attaches the special BOS ("beginning of sentence") token at the
/// start; most models can't correctly recognize the start of context without it.
fn tokenize(model: &LlamaModel, prompt: &str) -> Result<Vec<LlamaToken>> {
    Ok(model.str_to_token(prompt, AddBos::Always)?)
}

/// Decodes the prompt's token sequence, then repeatedly samples and decodes one token at
/// a time to generate text. Stops when an EOG (end-of-generation) token appears or n_len
/// is reached.
fn generate(ctx: &mut LlamaContext, model: &LlamaModel, tokens: Vec<LlamaToken>, n_len: i32) -> Result<String> {
    let mut batch = LlamaBatch::new(512, 1);
    let last_index = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.iter()) {
        let is_last = i == last_index;
        batch.add(*token, i, &[0], is_last)?;
    }
    ctx.decode(&mut batch)?;

    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut output = String::new();
    let mut n_cur = batch.n_tokens();

    while n_cur <= n_len {
        let token = sampler.sample(ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        output.push_str(&model.token_to_piece(token, &mut decoder, true, None)?);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }

    Ok(output)
}

/// A single generation request handed from an HTTP handler to the worker thread. The
/// result is sent back to the caller (the HTTP handler) via a oneshot channel.
struct GenerateJob {
    prompt: String,
    respond_to: oneshot::Sender<Result<String>>,
}

/// The GPU/model state (LlamaModel, LlamaContext) doesn't implement Send/Sync, so it
/// can't be shared across threads. Rather than sharing it via a Mutex, this confines it
/// to a single dedicated OS thread and communicates with HTTP handlers only through an
/// mpsc channel. This serializes all GPU access onto that one thread.
fn spawn_worker(model_path: &'static str) -> mpsc::UnboundedSender<GenerateJob> {
    let (tx, mut rx) = mpsc::unbounded_channel::<GenerateJob>();

    std::thread::spawn(move || {
        let backend = init_backend().expect("backend init failed");
        let model = load_model(&backend, model_path).expect("model load failed");
        let mut ctx = create_context(&model, &backend).expect("context creation failed");
        println!(
            "OK: worker ready. n_vocab = {}, n_ctx = {}",
            model.n_vocab(),
            ctx.n_ctx()
        );

        while let Some(job) = rx.blocking_recv() {
            // Treat each request as an independent, single-turn generation. Since ctx is
            // reused across all requests (it lives on this dedicated thread), the KV
            // cache must be cleared every time - otherwise the new token sequence's
            // starting position (0) conflicts with the leftover position from the
            // previous request, and decode fails.
            ctx.clear_kv_cache();
            let result =
                tokenize(&model, &job.prompt).and_then(|tokens| generate(&mut ctx, &model, tokens, N_LEN));
            let _ = job.respond_to.send(result);
        }
    });

    tx
}

#[derive(Deserialize)]
struct GenerateRequest {
    prompt: String,
}

#[derive(Serialize)]
struct GenerateResponse {
    response: String,
}

/// Liveness endpoint. Doesn't touch the model at all - just confirms axum itself is up.
async fn health() -> &'static str {
    "ok"
}

/// POST /generate. Sends the request to the worker thread as a job and waits for the
/// generation result to come back over the oneshot channel. No GPU inference happens
/// here directly.
async fn generate_handler(
    State(tx): State<mpsc::UnboundedSender<GenerateJob>>,
    Json(req): Json<GenerateRequest>,
) -> Json<GenerateResponse> {
    let (respond_to, response_rx) = oneshot::channel();
    tx.send(GenerateJob {
        prompt: req.prompt,
        respond_to,
    })
    .expect("worker thread died");

    let response = response_rx
        .await
        .expect("worker thread dropped response")
        .expect("generation failed");

    Json(GenerateResponse { response })
}

#[tokio::main]
async fn main() -> Result<()> {
    let tx = spawn_worker(MODEL_PATH);

    let app = Router::new()
        .route("/health", get(health))
        .route("/generate", post(generate_handler))
        .with_state(tx);

    // Bind on 0.0.0.0 so the server is reachable from other machines on the LAN, not
    // just localhost.
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    println!("listening on {}", listener.local_addr()?);
    axum::serve(listener, app).await?;

    Ok(())
}
