use {
    tracing::{Level, info},
    tracing_subscriber::FmtSubscriber,
    tokio::net::TcpListener,
    axum::{Router, routing::get, http::StatusCode, response::IntoResponse},
};

#[tokio::main]
async fn main() {
    init_logging();

    let addr = "0.0.0.0:3000";
    info!("starting llama inference server on {}", addr);

    let app = Router::new()
        .route("/", get(root))
        .fallback(not_found_handler);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn root() -> &'static str {
    "Hello, World!"
}

async fn not_found_handler() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, "endpoint not implemented\n")
}

fn init_logging() {
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .init();
}
