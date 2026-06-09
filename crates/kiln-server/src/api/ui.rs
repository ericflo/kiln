use axum::{
    Router,
    http::header,
    response::{Html, IntoResponse, Redirect},
    routing::get,
};

const UI_HTML: &str = include_str!("../ui.html");
// Vendored terminal renderer (xterm.js, MIT) for the embedded pi terminal.
// Served as separate routes so ui.html stays readable; everything is still
// compiled into the binary — the dashboard works fully offline.
const XTERM_JS: &str = include_str!("../vendor/xterm.js");
const XTERM_CSS: &str = include_str!("../vendor/xterm.css");
const XTERM_FIT_JS: &str = include_str!("../vendor/xterm-addon-fit.js");

async fn serve_ui() -> Html<&'static str> {
    Html(UI_HTML)
}

async fn redirect_to_ui() -> Redirect {
    Redirect::to("/ui")
}

async fn serve_xterm_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript")], XTERM_JS)
}

async fn serve_xterm_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css")], XTERM_CSS)
}

async fn serve_xterm_fit_js() -> impl IntoResponse {
    (
        [(header::CONTENT_TYPE, "application/javascript")],
        XTERM_FIT_JS,
    )
}

pub fn routes() -> Router<crate::state::AppState> {
    Router::new()
        .route("/", get(redirect_to_ui))
        .route("/ui", get(serve_ui))
        .route("/ui/vendor/xterm.js", get(serve_xterm_js))
        .route("/ui/vendor/xterm.css", get(serve_xterm_css))
        .route("/ui/vendor/xterm-addon-fit.js", get(serve_xterm_fit_js))
}
