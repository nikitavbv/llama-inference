use {
    std::sync::Mutex,
    crate::model::ChatModel,
    async_openai::types::{
        ChatChoice,
        ChatCompletionResponseMessage,
        CreateChatCompletionResponse,
        Role,
        CreateChatCompletionRequest,
        ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageContent,
    },
    axum::{
        body::Body,
        extract::{Json, State},
        http::{Request, StatusCode},
        response::IntoResponse,
        routing::{get, post},
        Router,
    },
    metrics_util::layers::Layer,
    metrics_exporter_prometheus::PrometheusBuilder,
    rand::{distributions::Alphanumeric, Rng},
    std::{sync::Arc, time::{SystemTime, UNIX_EPOCH}},
    tokio::{net::TcpListener, join},
    tracing::{info, warn, Level},
    tracing_subscriber::FmtSubscriber,
    crate::{
        utils::PrefixLayer,
        model::{ChatMessage, ChatRole},
    },
};

mod model;
mod utils;

#[derive(Clone)]
struct AppState {
    model: Arc<Mutex<ChatModel>>,
}

#[tokio::main]
async fn main() {
    init_logging();

    let (recorder, prometheus_future) = PrometheusBuilder::new().build().unwrap();
    metrics::set_global_recorder(PrefixLayer.layer(recorder)).unwrap();

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
    let _ = join!(
        async {
            axum::serve(listener, app).await.unwrap()
        },
        prometheus_future,
    );
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<CreateChatCompletionRequest>,
) -> Json<CreateChatCompletionResponse> {
    let completion = tokio::task::spawn_blocking(move || {
        let mut model = state.model.lock().unwrap();

        model.chat_completions(
            request
                .messages
                .into_iter()
                .map(|message| match message {
                    ChatCompletionRequestMessage::System(message) => ChatMessage {
                        role: ChatRole::System,
                        content: message.content,
                    },
                    ChatCompletionRequestMessage::User(message) => ChatMessage {
                        role: ChatRole::User,
                        content: match message.content {
                            ChatCompletionRequestUserMessageContent::Text(text) => text,
                            other => panic!("message content not implemented: {:?}", other),
                        },
                    },
                    ChatCompletionRequestMessage::Assistant(message) => ChatMessage {
                        role: ChatRole::Assistant,
                        content: message.content.unwrap(),
                    },
                    other => panic!("message type not implemented: {:?}", other),
                })
                .collect(),
        )
    }).await.unwrap();

    Json(CreateChatCompletionResponse {
        id: response_id(),
        created: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as u32,
        model: "llama-7b".to_owned(),
        object: "chat.completion".to_owned(),

        choices: vec![ChatChoice {
            index: 0,
            #[allow(deprecated)] message: ChatCompletionResponseMessage {
                role: Role::Assistant,
                content: Some(completion.content),
                tool_calls: None,
                function_call: None,
            },
            finish_reason: None,
            logprobs: None,
        }],

        system_fingerprint: None,
        usage: None,
    })
}

async fn root() -> &'static str {
    "Hello, World!"
}

async fn not_found_handler(req: Request<Body>) -> impl IntoResponse {
    warn!(
        url = req.uri().to_string(),
        method = req.method().to_string(),
        "endpoint is not implemented"
    );
    (StatusCode::NOT_FOUND, "endpoint not implemented\n")
}

fn response_id() -> String {
    rand::thread_rng().sample_iter(&Alphanumeric).take(12).map(|b| b as char).collect()
}

fn init_logging() {
    FmtSubscriber::builder().with_max_level(Level::INFO).init();
}
