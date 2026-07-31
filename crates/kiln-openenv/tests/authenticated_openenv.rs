use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use axum::{
    Json, Router,
    extract::{
        Path, State, WebSocketUpgrade,
        ws::{Message, WebSocket},
    },
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use futures::StreamExt;
use kiln_openenv::{OpenEnvAuthentication, OpenEnvClient, OpenEnvReward};
use serde_json::{Value, json};

const TOKEN: &str = "kiln-openenv-test-token";
const REFLECTED_HTTP_TOKEN: &str = "reflected-http-token-must-not-escape";
const REFLECTED_WEBSOCKET_TOKEN: &str = "reflected-websocket-token-must-not-escape";

#[derive(Clone)]
struct AuthState {
    websocket_authenticated: Arc<AtomicBool>,
}

fn authorized(headers: &HeaderMap) -> bool {
    matches!(
        bearer_token(headers),
        Some(TOKEN | REFLECTED_HTTP_TOKEN | REFLECTED_WEBSOCKET_TOKEN)
    )
}

fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

async fn health(headers: HeaderMap) -> StatusCode {
    if authorized(&headers) {
        StatusCode::OK
    } else {
        StatusCode::UNAUTHORIZED
    }
}

async fn metadata(headers: HeaderMap) -> Response {
    let description = if bearer_token(&headers) == Some(REFLECTED_HTTP_TOKEN) {
        REFLECTED_HTTP_TOKEN
    } else {
        "requires one bearer credential"
    };
    protected_json(
        &headers,
        json!({
            "name": "AuthenticatedEnvironment",
            "description": description,
            "version": "1"
        }),
    )
}

async fn schema(headers: HeaderMap) -> Response {
    protected_json(
        &headers,
        json!({
            "action": {
                "type": "object",
                "properties": {"answer": {"type": "integer"}},
                "required": ["answer"]
            },
            "observation": {
                "type": "object",
                "properties": {"question": {"type": "string"}}
            },
            "state": {"type": "object"}
        }),
    )
}

async fn environments(headers: HeaderMap) -> Response {
    protected_json(&headers, json!(["authenticated_env"]))
}

async fn openapi(headers: HeaderMap) -> Response {
    protected_json(
        &headers,
        json!({"openapi": "3.1.0", "info": {"title": "auth", "version": "1"}}),
    )
}

async fn task_splits(Path(environment): Path<String>, headers: HeaderMap) -> Response {
    assert_eq!(environment, "authenticated_env");
    protected_json(&headers, json!([{"name": "train", "type": "train"}]))
}

async fn task_post(
    Path((environment, operation)): Path<(String, String)>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    assert_eq!(environment, "authenticated_env");
    match operation.as_str() {
        "num_tasks" => {
            assert_eq!(body, json!({"split": "train"}));
            protected_json(&headers, json!({"num_tasks": 1}))
        }
        "task_range" => {
            assert_eq!(body, json!({"split": "train", "start": 0, "stop": 1}));
            let prompt = if bearer_token(&headers) == Some(REFLECTED_HTTP_TOKEN) {
                REFLECTED_HTTP_TOKEN
            } else {
                "2 + 2"
            };
            protected_json(
                &headers,
                json!({"tasks": [{"prompt": prompt, "answer": "4"}]}),
            )
        }
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

fn protected_json(headers: &HeaderMap, value: Value) -> Response {
    if authorized(headers) {
        Json(value).into_response()
    } else {
        (
            StatusCode::UNAUTHORIZED,
            headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or("missing authorization")
                .to_string(),
        )
            .into_response()
    }
}

async fn websocket(
    State(state): State<AuthState>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Response {
    if !authorized(&headers) {
        return (
            StatusCode::UNAUTHORIZED,
            headers
                .get(AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or("missing authorization")
                .to_string(),
        )
            .into_response();
    }
    state.websocket_authenticated.store(true, Ordering::Relaxed);
    let reflect_credential = bearer_token(&headers) == Some(REFLECTED_WEBSOCKET_TOKEN);
    upgrade
        .on_upgrade(move |socket| serve_episode(socket, reflect_credential))
        .into_response()
}

async fn serve_episode(mut socket: WebSocket, reflect_credential: bool) {
    while let Some(Ok(Message::Text(text))) = socket.next().await {
        let request: Value = serde_json::from_str(&text).unwrap();
        match request["type"].as_str() {
            Some("reset") => {
                socket
                    .send(Message::Text(
                        json!({
                            "type": "observation",
                            "data": {
                                "observation": {
                                    "question": if reflect_credential {
                                        REFLECTED_WEBSOCKET_TOKEN
                                    } else {
                                        "authenticated?"
                                    }
                                },
                                "reward": null,
                                "done": false
                            }
                        })
                        .to_string()
                        .into(),
                    ))
                    .await
                    .unwrap();
            }
            Some("close") => break,
            other => panic!("unexpected test OpenEnv message {other:?}"),
        }
    }
}

#[tokio::test]
async fn bearer_auth_covers_discovery_task_api_and_the_websocket_upgrade() {
    let websocket_authenticated = Arc::new(AtomicBool::new(false));
    let state = AuthState {
        websocket_authenticated: websocket_authenticated.clone(),
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/metadata", get(metadata))
        .route("/schema", get(schema))
        .route("/list_environments", get(environments))
        .route("/openapi.json", get(openapi))
        .route("/{environment}/splits", get(task_splits))
        .route("/{environment}/{operation}", post(task_post))
        .route("/ws", get(websocket))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let unauthenticated = OpenEnvClient::new(format!("http://{address}")).unwrap();
    assert!(unauthenticated.inspect().await.is_err());

    let wrong_token = "wrong-token-that-must-never-leak";
    let rejected = OpenEnvClient::new(format!("http://{address}"))
        .unwrap()
        .with_bearer_token(wrong_token)
        .unwrap();
    let discovery_error = rejected.metadata().await.unwrap_err().to_string();
    assert!(!discovery_error.contains(wrong_token), "{discovery_error}");
    assert!(discovery_error.contains("response body redacted"));
    let websocket_error = rejected.connect().await.unwrap_err().to_string();
    assert!(!websocket_error.contains(wrong_token), "{websocket_error}");
    assert!(websocket_error.contains("response body redacted"));

    let reflected_http = OpenEnvClient::new(format!("http://{address}"))
        .unwrap()
        .with_bearer_token(REFLECTED_HTTP_TOKEN)
        .unwrap();
    let reflected_http_error = reflected_http.metadata().await.unwrap_err().to_string();
    assert!(
        !reflected_http_error.contains(REFLECTED_HTTP_TOKEN),
        "{reflected_http_error}"
    );
    assert!(reflected_http_error.contains("reflected"));
    let reflected_task_error = reflected_http
        .task_catalog(None, Some("train"), 0, 1)
        .await
        .unwrap_err()
        .to_string();
    assert!(
        !reflected_task_error.contains(REFLECTED_HTTP_TOKEN),
        "{reflected_task_error}"
    );
    assert!(reflected_task_error.contains("reflected"));

    let reflected_websocket = OpenEnvClient::new(format!("http://{address}"))
        .unwrap()
        .with_bearer_token(REFLECTED_WEBSOCKET_TOKEN)
        .unwrap();
    let mut reflected_episode = reflected_websocket.connect().await.unwrap();
    let reflected_websocket_error = reflected_episode
        .reset(&json!({"seed": 8}))
        .await
        .unwrap_err()
        .to_string();
    assert!(
        !reflected_websocket_error.contains(REFLECTED_WEBSOCKET_TOKEN),
        "{reflected_websocket_error}"
    );
    assert!(reflected_websocket_error.contains("reflected"));

    let authenticated = OpenEnvClient::new(format!("http://{address}"))
        .unwrap()
        .with_bearer_token(TOKEN)
        .unwrap();
    let inspection = authenticated.inspect().await.unwrap();
    assert_eq!(
        inspection.identity.metadata.name,
        "AuthenticatedEnvironment"
    );
    assert_eq!(
        inspection.identity.authentication,
        OpenEnvAuthentication::Bearer
    );
    let task_catalog = authenticated
        .task_catalog(None, Some("train"), 0, 1)
        .await
        .unwrap();
    assert_eq!(task_catalog.num_tasks, Some(1));
    assert_eq!(
        task_catalog.tasks,
        [json!({"prompt": "2 + 2", "answer": "4"})]
    );
    let mut episode = authenticated.connect().await.unwrap();
    let reset = episode.reset(&json!({"seed": 7})).await.unwrap();
    assert_eq!(reset.observation["question"], "authenticated?");
    assert_eq!(reset.reward, OpenEnvReward::Null);
    episode.close().await.unwrap();
    assert!(websocket_authenticated.load(Ordering::Relaxed));

    server.abort();
}
