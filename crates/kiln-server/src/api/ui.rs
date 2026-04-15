use axum::{response::Html, routing::get, Router};

const UI_HTML: &str = include_str!("../ui.html");

async fn serve_ui() -> Html<&'static str> {
    Html(UI_HTML)
}

pub fn routes() -> Router<crate::state::AppState> {
    Router::new().route("/ui", get(serve_ui))
}
