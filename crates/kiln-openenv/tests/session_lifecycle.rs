use std::{future::Future, time::Duration};

use futures::{SinkExt, StreamExt};
use kiln_openenv::{OpenEnvClient, OpenEnvClientError};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
    task::JoinHandle,
};
use tokio_tungstenite::{WebSocketStream, accept_async, tungstenite::Message};

type ServerSocket = WebSocketStream<TcpStream>;

async fn websocket_fixture<H, F>(handler: H) -> (String, JoinHandle<()>)
where
    H: FnOnce(ServerSocket) -> F + Send + 'static,
    F: Future<Output = ()> + Send + 'static,
{
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut health_stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        loop {
            let mut chunk = [0u8; 1024];
            let read = health_stream.read(&mut chunk).await.unwrap();
            assert_ne!(read, 0, "health request ended before its headers");
            request.extend_from_slice(&chunk[..read]);
            if request.windows(4).any(|window| window == b"\r\n\r\n") {
                break;
            }
        }
        assert!(request.starts_with(b"GET /health HTTP/1.1\r\n"));
        health_stream
            .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 0\r\nconnection: close\r\n\r\n")
            .await
            .unwrap();
        health_stream.shutdown().await.unwrap();

        let (stream, _) = listener.accept().await.unwrap();
        let socket = accept_async(stream).await.unwrap();
        handler(socket).await;
    });
    (format!("http://{address}"), server)
}

async fn receive_json(socket: &mut ServerSocket) -> Value {
    loop {
        match socket.next().await.unwrap().unwrap() {
            Message::Text(text) => return serde_json::from_str(&text).unwrap(),
            Message::Ping(payload) => socket.send(Message::Pong(payload)).await.unwrap(),
            Message::Pong(_) => {}
            other => panic!("expected client text frame, got {other:?}"),
        }
    }
}

fn observation(value: i64, done: bool) -> Message {
    Message::Text(
        json!({
            "type": "observation",
            "data": {
                "observation": {"value": value},
                "reward": value,
                "done": done
            }
        })
        .to_string()
        .into(),
    )
}

#[tokio::test]
async fn policy_work_pumps_ping_pong_without_consuming_the_next_exchange() {
    let (base_url, server) = websocket_fixture(|mut socket| async move {
        assert_eq!(
            receive_json(&mut socket).await,
            json!({"type": "reset", "data": {"seed": 7}})
        );
        socket.send(observation(0, false)).await.unwrap();
        socket
            .send(Message::Ping(b"policy-still-thinking".to_vec().into()))
            .await
            .unwrap();
        assert_eq!(
            socket.next().await.unwrap().unwrap(),
            Message::Pong(b"policy-still-thinking".to_vec().into())
        );
        assert_eq!(
            receive_json(&mut socket).await,
            json!({"type": "step", "data": {"answer": "42"}})
        );
        socket.send(observation(1, true)).await.unwrap();
        assert_eq!(receive_json(&mut socket).await, json!({"type": "close"}));
    })
    .await;

    let client = OpenEnvClient::new(base_url).unwrap();
    let mut session = client.connect().await.unwrap();
    session.reset(&json!({"seed": 7})).await.unwrap();
    let output = session
        .keep_alive_while(async {
            tokio::time::sleep(Duration::from_millis(200)).await;
            "ready"
        })
        .await
        .unwrap();
    assert_eq!(output, "ready");
    assert!(!session.is_closed());
    assert!(session.step(&json!({"answer": "42"})).await.unwrap().done);
    session.close().await.unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn policy_work_uses_read_only_state_exchanges_to_maintain_idle_sessions() {
    let (base_url, server) = websocket_fixture(|mut socket| async move {
        assert_eq!(
            receive_json(&mut socket).await,
            json!({"type": "reset", "data": {"seed": 9}})
        );
        socket.send(observation(0, false)).await.unwrap();

        let mut state_requests = 0;
        loop {
            let request = receive_json(&mut socket).await;
            match request.get("type").and_then(Value::as_str) {
                Some("state") => {
                    state_requests += 1;
                    socket
                        .send(Message::Text(
                            json!({"type": "state", "data": {"step_count": 0}})
                                .to_string()
                                .into(),
                        ))
                        .await
                        .unwrap();
                }
                Some("step") => {
                    assert_eq!(request["data"], json!({"answer": "42"}));
                    break;
                }
                other => panic!("unexpected OpenEnv request during policy work: {other:?}"),
            }
        }
        assert!(state_requests >= 2, "expected repeated state maintenance");
        socket.send(observation(1, true)).await.unwrap();
        assert_eq!(receive_json(&mut socket).await, json!({"type": "close"}));
    })
    .await;

    let client = OpenEnvClient::new(base_url)
        .unwrap()
        .with_request_timeout(Duration::from_millis(200));
    let mut session = client.connect().await.unwrap();
    session.reset(&json!({"seed": 9})).await.unwrap();
    let output = session
        .keep_alive_while(async {
            tokio::time::sleep(Duration::from_millis(170)).await;
            "ready"
        })
        .await
        .unwrap();
    assert_eq!(output, "ready");
    assert!(session.step(&json!({"answer": "42"})).await.unwrap().done);
    session.close().await.unwrap();
    server.await.unwrap();
}

#[tokio::test]
async fn timed_out_exchange_permanently_poisons_the_lock_step_session() {
    let (base_url, server) = websocket_fixture(|mut socket| async move {
        assert_eq!(
            receive_json(&mut socket).await,
            json!({"type": "reset", "data": {}})
        );
        tokio::time::sleep(Duration::from_millis(80)).await;
        let _ = socket.send(observation(0, false)).await;
    })
    .await;

    let client = OpenEnvClient::new(base_url)
        .unwrap()
        .with_request_timeout(Duration::from_millis(20));
    let mut session = client.connect().await.unwrap();
    assert!(matches!(
        session.reset(&json!({})).await.unwrap_err(),
        OpenEnvClientError::Timeout(_)
    ));
    assert!(session.is_closed());
    assert!(matches!(
        session.state().await.unwrap_err(),
        OpenEnvClientError::Closed
    ));
    drop(session);
    server.await.unwrap();
}

#[tokio::test]
async fn wrong_response_type_permanently_poisons_the_lock_step_session() {
    let (base_url, server) = websocket_fixture(|mut socket| async move {
        let _ = receive_json(&mut socket).await;
        socket
            .send(Message::Text(
                json!({"type": "state", "data": {"step_count": 0}})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    })
    .await;

    let client = OpenEnvClient::new(base_url).unwrap();
    let mut session = client.connect().await.unwrap();
    assert!(matches!(
        session.reset(&json!({})).await.unwrap_err(),
        OpenEnvClientError::UnexpectedResponse {
            expected: "observation",
            actual: "state"
        }
    ));
    assert!(session.is_closed());
    assert!(matches!(
        session.step(&json!({})).await.unwrap_err(),
        OpenEnvClientError::Closed
    ));
    drop(session);
    server.await.unwrap();
}

#[tokio::test]
async fn unsolicited_application_message_during_policy_work_is_terminal() {
    let (base_url, server) = websocket_fixture(|mut socket| async move {
        let _ = receive_json(&mut socket).await;
        socket.send(observation(0, false)).await.unwrap();
        socket.send(observation(99, false)).await.unwrap();
        tokio::time::sleep(Duration::from_millis(20)).await;
    })
    .await;

    let client = OpenEnvClient::new(base_url).unwrap();
    let mut session = client.connect().await.unwrap();
    session.reset(&json!({})).await.unwrap();
    assert!(matches!(
        session
            .keep_alive_while(tokio::time::sleep(Duration::from_secs(1)))
            .await
            .unwrap_err(),
        OpenEnvClientError::UnsolicitedApplicationMessage
    ));
    assert!(session.is_closed());
    assert!(matches!(
        session.state().await.unwrap_err(),
        OpenEnvClientError::Closed
    ));
    drop(session);
    server.await.unwrap();
}

#[tokio::test]
async fn recoverable_protocol_error_is_the_only_exchange_error_that_keeps_the_session_live() {
    let (base_url, server) = websocket_fixture(|mut socket| async move {
        let _ = receive_json(&mut socket).await;
        socket.send(observation(0, false)).await.unwrap();
        let _ = receive_json(&mut socket).await;
        socket
            .send(Message::Text(
                json!({
                    "type": "error",
                    "data": {
                        "code": "VALIDATION_ERROR",
                        "message": "answer must be a string",
                        "errors": []
                    }
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        let _ = receive_json(&mut socket).await;
        socket.send(observation(1, true)).await.unwrap();
    })
    .await;

    let client = OpenEnvClient::new(base_url).unwrap();
    let mut session = client.connect().await.unwrap();
    session.reset(&json!({})).await.unwrap();
    assert!(matches!(
        session.step(&json!({"answer": 42})).await.unwrap_err(),
        OpenEnvClientError::Protocol(_)
    ));
    assert!(!session.is_closed());
    assert!(session.step(&json!({"answer": "42"})).await.unwrap().done);
    drop(session);
    server.await.unwrap();
}
