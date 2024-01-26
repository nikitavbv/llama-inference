use {
    tracing::{Level, info, warn},
    tracing_subscriber::FmtSubscriber,
    tokio::net::TcpListener,
    serde::{Serialize, Deserialize},
    axum::{Router, routing::{get, post}, http::{StatusCode, Request}, response::IntoResponse, body::Body, extract::Json},
};

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
    info!("starting llama inference server on {}", addr);

    let app = Router::new()
        .route("/", get(root))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(not_found_handler);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
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

fn init_logging() {
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .init();
}
