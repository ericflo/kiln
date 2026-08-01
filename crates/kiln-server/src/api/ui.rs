use axum::{
    Router,
    http::{Uri, header},
    response::{IntoResponse, Redirect},
    routing::get,
};

const UI_INDEX_HTML: &str = include_str!("../ui/index.html");
const UI_STYLES_CSS: &str = include_str!("../ui/styles.css");
const UI_DEMO_JS: &str = include_str!("../ui/demo.js");
// Preserve one classic-script closure and one /ui/app.js response while the
// source is owned by behavior-specific files. Fragment order is execution
// order; each fragment is also valid JavaScript on its own for syntax checks.
const UI_APP_JS: &str = concat!(
    "(function() {\n'use strict';\n\n",
    include_str!("../ui/app/shell.js"),
    include_str!("../ui/app/adapters.js"),
    include_str!("../ui/app/training.js"),
    include_str!("../ui/app/playground.js"),
    include_str!("../ui/app/evaluations.js"),
    include_str!("../ui/app/command_palette.js"),
    include_str!("../ui/app/charts.js"),
    include_str!("../ui/app/adapter_drill.js"),
    include_str!("../ui/app/training_drill.js"),
    include_str!("../ui/app/playground_compare.js"),
    include_str!("../ui/app/terminal.js"),
    include_str!("../ui/app/distillation.js"),
    include_str!("../ui/app/agents.js"),
    include_str!("../ui/app/preflight.js"),
    include_str!("../ui/app/bootstrap.js"),
    "\n})();\n",
);
// Vendored terminal renderer (xterm.js, MIT) for the embedded pi terminal.
// Served as separate routes so index.html stays readable; everything is still
// compiled into the binary — the dashboard works fully offline.
const XTERM_JS: &str = include_str!("../ui/vendor/xterm.js");
const XTERM_CSS: &str = include_str!("../ui/vendor/xterm.css");
const XTERM_FIT_JS: &str = include_str!("../ui/vendor/xterm-addon-fit.js");

// UI typefaces — Inter Variable (100–900) + JetBrains Mono Variable (100–800).
// Two files, every weight, baked into the binary so the dashboard renders
// identically offline. (OFL-licensed; see THIRD_PARTY_LICENSES.md.)
const FONT_INTER: &[u8] = include_bytes!("../ui/fonts/InterVariable.woff2");
const FONT_JBMONO: &[u8] = include_bytes!("../ui/fonts/JetBrainsMonoVariable.ttf");

// Every /ui* response is marked no-cache: the assets are baked into the
// binary, so after a binary upgrade the browser must revalidate rather than
// pair a cached stale app.js with a new index.html. (When everything was
// inlined in a single document it was always self-consistent; this preserves
// that property across the split.)
const NO_CACHE: &str = "no-cache";

fn ui_asset(content_type: &'static str, body: &'static str) -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, NO_CACHE),
        ],
        body,
    )
}

async fn serve_ui() -> impl IntoResponse {
    ui_asset("text/html; charset=utf-8", UI_INDEX_HTML)
}

async fn serve_styles_css() -> impl IntoResponse {
    ui_asset("text/css", UI_STYLES_CSS)
}

async fn serve_demo_js() -> impl IntoResponse {
    ui_asset("application/javascript", UI_DEMO_JS)
}

async fn serve_app_js() -> impl IntoResponse {
    ui_asset("application/javascript", UI_APP_JS)
}

/// `/ui` → `/ui/`, preserving the query string (`/ui?demo=1` → `/ui/?demo=1`)
/// so demo mode and any future params survive the redirect. 307 keeps the
/// method intact. Relative asset URLs (./styles.css, ./app.js) require the
/// trailing-slash form so they resolve under /ui/.
async fn redirect_ui_to_slash(uri: Uri) -> impl IntoResponse {
    let target = match uri.query() {
        Some(query) => format!("/ui/?{query}"),
        None => "/ui/".to_string(),
    };
    (
        [(header::CACHE_CONTROL, NO_CACHE)],
        Redirect::temporary(&target),
    )
}

async fn redirect_root_to_ui() -> Redirect {
    Redirect::to("/ui/")
}

async fn serve_xterm_js() -> impl IntoResponse {
    ui_asset("application/javascript", XTERM_JS)
}

async fn serve_xterm_css() -> impl IntoResponse {
    ui_asset("text/css", XTERM_CSS)
}

async fn serve_xterm_fit_js() -> impl IntoResponse {
    ui_asset("application/javascript", XTERM_FIT_JS)
}

fn font_asset(content_type: &'static str, body: &'static [u8]) -> impl IntoResponse {
    (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, NO_CACHE),
        ],
        body,
    )
}

async fn serve_font_inter() -> impl IntoResponse {
    font_asset("font/woff2", FONT_INTER)
}

async fn serve_font_jbmono() -> impl IntoResponse {
    font_asset("font/ttf", FONT_JBMONO)
}

pub fn routes() -> Router<crate::state::AppState> {
    Router::new()
        .route("/", get(redirect_root_to_ui))
        .route("/ui", get(redirect_ui_to_slash))
        .route("/ui/", get(serve_ui))
        .route("/ui/styles.css", get(serve_styles_css))
        .route("/ui/demo.js", get(serve_demo_js))
        .route("/ui/app.js", get(serve_app_js))
        .route("/ui/vendor/xterm.js", get(serve_xterm_js))
        .route("/ui/vendor/xterm.css", get(serve_xterm_css))
        .route("/ui/vendor/xterm-addon-fit.js", get(serve_xterm_fit_js))
        .route("/ui/fonts/InterVariable.woff2", get(serve_font_inter))
        .route(
            "/ui/fonts/JetBrainsMonoVariable.ttf",
            get(serve_font_jbmono),
        )
}
