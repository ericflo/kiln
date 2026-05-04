use axum::{
    response::{Html, Redirect},
    routing::get,
    Router,
};

const UI_HTML: &str = include_str!("../ui.html");

async fn serve_ui() -> Html<&'static str> {
    Html(UI_HTML)
}

async fn redirect_to_ui() -> Redirect {
    Redirect::to("/ui")
}

pub fn routes() -> Router<crate::state::AppState> {
    Router::new()
        .route("/", get(redirect_to_ui))
        .route("/ui", get(serve_ui))
}
