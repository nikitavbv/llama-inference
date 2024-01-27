use {
    std::sync::Arc,
    tracing::{Level, info, warn},
    tracing_subscriber::FmtSubscriber,
    tokio::{net::TcpListener, sync::Mutex},
    serde::{Serialize, Deserialize},
    axum::{
        Router,
        routing::{get, post},
        http::{StatusCode, Request},
        response::IntoResponse,
        body::Body,
        extract::{State, Json},
    },
    crate::model::ChatModel,
};

mod model;

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

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<ChatModel>>,
}

#[tokio::main]
async fn main() {
    init_logging();

    let addr = "0.0.0.0:3000";
    let model = ChatModel::load();
    let state = AppState {
        model: Arc::new(Mutex::new(model)),
    };

    info!("starting llama inference server on {}", addr);

    let app = Router::new()
        .route("/", get(root))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(not_found_handler)
        .with_state(state);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn chat_completions(State(state): State<AppState>, Json(request): Json<ChatCompletionsRequest>) -> Json<ChatCompletionsResponse> {
    let model = state.model.lock().await;
    let completion = model.chat_completions(
        request.messages.into_iter()
            .map(|message| model::ChatMessage {
                role: match message.role {
                    ChatRole::System => model::ChatRole::System,
                    ChatRole::User => model::ChatRole::User,
                    ChatRole::Assistant => model::ChatRole::Assistant,
                },
                content: message.content,
            })
            .collect()
    );

    Json(ChatCompletionsResponse {
        choices: vec![
            ChatCompletion {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: completion.content,
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
