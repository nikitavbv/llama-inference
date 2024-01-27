use {
    tracing::{Level, info, warn},
    tracing_subscriber::FmtSubscriber,
    tokio::net::TcpListener,
    serde::{Serialize, Deserialize},
    axum::{Router, routing::{get, post}, http::{StatusCode, Request}, response::IntoResponse, body::Body, extract::Json},
    candle_core::{Device, DType},
    candle_transformers::models::llama::{Llama, LlamaConfig},
};

// based on https://github.com/huggingface/candle/blob/main/candle-examples/examples/llama/main.rs

#[derive(Debug, Deserialize)]
struct ChatCompletionsRequest {
    messages: Vec<ChatMessage>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionsResponse {
    choices: Vec<ChatCompletion>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: ChatRole,
    content: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ChatRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Serialize)]
struct ChatCompletion {
    index: u32,
    message: ChatMessage,
}

#[tokio::main]
async fn main() {
    init_logging();

    let addr = "0.0.0.0:3000";
    load_model();

    info!("starting llama inference server on {}", addr);

    let app = Router::new()
        .route("/", get(root))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(not_found_handler);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn load_model() {
    use hf_hub::{api::sync::Api, Repo, RepoType};

    let device = Device::Cpu;
    let dtype = DType::F32;

    let hf_api = Api::new().unwrap();
    let hf_api = hf_api.repo(Repo::with_revision(
        "meta-llama/Llama-2-7b-hf".to_owned(),
        RepoType::Model,
        "main".to_owned()
    ));

    let tokenizer_filename = hf_api.get("tokenizer.json").unwrap();
    let config_filename = hf_api.get("config.json").unwrap();
    let filenames = hub_load_safetensors(&hf_api, "model.safetensors.index.json");

    let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename).unwrap()).unwrap();

}

async fn chat_completions(Json(request): Json<ChatCompletionsRequest>) -> Json<ChatCompletionsResponse> {
    Json(ChatCompletionsResponse {
        choices: vec![
            ChatCompletion {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: format!("echo: {:?}", request.messages.last().unwrap().content),
                },
            }
        ],
    })
}

async fn root() -> &'static str {
    "Hello, World!"
}

async fn not_found_handler(req: Request<Body>) -> impl IntoResponse {
    warn!(url=req.uri().to_string(), method=req.method().to_string(), "endpoint is not implemented");
    (StatusCode::NOT_FOUND, "endpoint not implemented\n")
}

// taken from: https://github.com/huggingface/candle/blob/main/candle-examples/src/lib.rs#L122
/// Loads the safetensors files for a model from the hub based on a json index file.
fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Vec<std::path::PathBuf> {
    let json_file = repo.get(json_file).unwrap();
    let json_file = std::fs::File::open(json_file).unwrap();
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).unwrap();
    let weight_map = match json.get("weight_map") {
        None => panic!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => panic!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    safetensors_files
        .iter()
        .map(|v| repo.get(v).unwrap())
        .collect::<Vec<_>>()
}

fn init_logging() {
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .init();
}
