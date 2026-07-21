use super::*;

async fn wait_for_actor_barrier(
    handle: &BatchingEngineHandle,
    adapter: bool,
    resize: bool,
) -> BatchingEngineSnapshot {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let snapshot = handle.cached_snapshot();
        if snapshot.actor_barrier_adapter_active == adapter
            && snapshot.actor_barrier_resize_active == resize
        {
            return snapshot;
        }
        assert!(
            Instant::now() < deadline,
            "actor barrier did not publish: {snapshot:?}"
        );
        tokio::task::yield_now().await;
    }
}

#[tokio::test]
async fn adapter_swap_waits_for_active_request_and_pauses_admission() {
    let events = Arc::new(StdMutex::new(Vec::new()));
    let (forward, release) = GatedForward::new(events.clone());
    let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 8);

    let mut active = handle.enqueue(request(100, 3)).await.unwrap();
    release.send(()).unwrap();
    assert!(matches!(
        active.recv().await,
        Some(EngineEvent::Token { .. })
    ));

    let swap_events = events.clone();
    let swap_task = {
        let handle = handle.clone();
        tokio::spawn(async move {
            handle
                .swap_adapter(Box::new(move || {
                    swap_events.lock().unwrap().push("swap".to_string());
                    Ok(())
                }))
                .await
        })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    release.send(()).unwrap();
    assert!(matches!(
        active.recv().await,
        Some(EngineEvent::Token { .. })
    ));
    wait_for_actor_barrier(&handle, true, false).await;
    let mut waiting = handle.enqueue(request(200, 1)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    release.send(()).unwrap();
    release.send(()).unwrap();
    swap_task.await.unwrap().unwrap();
    for response in [&mut active, &mut waiting] {
        loop {
            match response.recv().await {
                Some(EngineEvent::Done { .. }) => break,
                Some(EngineEvent::Token { .. }) => {}
                Some(EngineEvent::Error(error)) => panic!("request failed: {error}"),
                None => panic!("request channel closed before completion"),
            }
        }
    }

    let log = events.lock().unwrap().clone();
    let swap_idx = log.iter().position(|event| event == "swap").unwrap();
    let waiting_idx = log.iter().position(|event| event == "prepare:200").unwrap();
    assert!(
        swap_idx < waiting_idx,
        "waiting request preceded swap: {log:?}"
    );
    assert_eq!(
        log[..swap_idx]
            .iter()
            .filter(|event| event.as_str() == "decode")
            .count(),
        3,
        "active request did not drain before swap: {log:?}"
    );
    let completed = handle.cached_snapshot();
    assert!(!completed.actor_barrier_adapter_active);
    assert!(!completed.actor_barrier_resize_active);
    handle.stop().await.unwrap();
}

#[tokio::test]
async fn kv_resize_waits_for_active_request_and_pauses_admission() {
    let events = Arc::new(StdMutex::new(Vec::new()));
    let (forward, release) = GatedForward::new(events.clone());
    let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 8);

    let mut active = handle.enqueue(request(100, 3)).await.unwrap();
    release.send(()).unwrap();
    assert!(matches!(
        active.recv().await,
        Some(EngineEvent::Token { .. })
    ));

    let resize_task = {
        let handle = handle.clone();
        tokio::spawn(async move { handle.resize_kv(17).await })
    };
    tokio::time::sleep(Duration::from_millis(20)).await;
    release.send(()).unwrap();
    assert!(matches!(
        active.recv().await,
        Some(EngineEvent::Token { .. })
    ));
    wait_for_actor_barrier(&handle, false, true).await;
    let mut waiting = handle.enqueue(request(200, 1)).await.unwrap();
    tokio::time::sleep(Duration::from_millis(50)).await;

    assert!(!resize_task.is_finished(), "resize ran before active drain");
    let before_drain = events.lock().unwrap().clone();
    assert!(!before_drain.iter().any(|event| event == "resize:17"));
    assert!(!before_drain.iter().any(|event| event == "prepare:200"));

    release.send(()).unwrap();
    release.send(()).unwrap();
    assert_eq!(resize_task.await.unwrap().unwrap(), 17);
    for response in [&mut active, &mut waiting] {
        loop {
            match response.recv().await {
                Some(EngineEvent::Done { .. }) => break,
                Some(EngineEvent::Token { .. }) => {}
                Some(EngineEvent::Error(error)) => panic!("request failed: {error}"),
                None => panic!("request channel closed before completion"),
            }
        }
    }

    let log = events.lock().unwrap().clone();
    let resize_idx = log
        .iter()
        .position(|event| event == "resize:17")
        .expect("resize ran");
    assert!(log.iter().any(|event| event == "resize_reason:maintenance"));
    assert!(
        log.iter()
            .any(|event| event == "resize_barrier_wait:positive")
    );
    let waiting_idx = log.iter().position(|event| event == "prepare:200").unwrap();
    assert!(
        resize_idx < waiting_idx,
        "waiting request preceded resize: {log:?}"
    );
    assert_eq!(
        log[..resize_idx]
            .iter()
            .filter(|event| event.as_str() == "decode")
            .count(),
        3,
        "active request did not drain before resize: {log:?}"
    );
    let completed = handle.cached_snapshot();
    assert!(!completed.actor_barrier_adapter_active);
    assert!(!completed.actor_barrier_resize_active);
    handle.stop().await.unwrap();
}

#[tokio::test]
async fn dropping_final_handle_stops_actor_and_delivery_worker() {
    let events = Arc::new(StdMutex::new(Vec::new()));
    let (forward, release) = GatedForward::new(events);
    let handle = BatchingEngineHandle::start_with_options(Arc::new(forward), 1);
    let req = request(101, 4);
    let cancel = req.cancel.clone();
    let mut response = handle.enqueue(req).await.unwrap();

    let forward_deadline = Instant::now() + Duration::from_secs(2);
    loop {
        let snapshot = handle.cached_snapshot();
        if snapshot.current_batch_size == 1 {
            break;
        }
        assert!(
            Instant::now() < forward_deadline,
            "request did not enter its gated forward: {snapshot:?}"
        );
        tokio::task::yield_now().await;
    }

    drop(handle);
    release.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(2), async {
        while response.recv().await.is_some() {}
    })
    .await
    .expect("dropping the final strong handle must tear down both threads");
    assert!(cancel.is_cancelled());
}

#[test]
fn closed_response_channel_is_counted_without_backpressure() {
    let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
    let forward = Arc::new(MockForward::default());
    let mut actor = test_actor(
        rx,
        forward,
        8,
        false,
        1,
        false,
        ResponseDeliveryPolicy::default(),
    );
    let (response_tx, response_rx) = mpsc::channel(1);
    drop(response_rx);
    let req = request(101, 2);
    push_test_active(
        &mut actor,
        req,
        response_tx,
        DecodeSlot::Mock {
            next_token: 101,
            generated_tokens: Vec::new(),
        },
    );

    actor.submit_token_delivery(
        0,
        111,
        EngineTokenTiming::ready(Instant::now(), TokenPhaseDurations::default()),
    );
    settle_active_deliveries(&mut actor);
    assert!(actor.active.is_empty());
    assert_eq!(actor.snapshot.response_channel_closed, 1);
    assert_eq!(actor.snapshot.response_backpressure_events, 0);
    assert_eq!(actor.snapshot.response_backpressure_wait_ms, 0);
    assert_eq!(actor.snapshot.response_stall_evictions, 0);
}

#[test]
fn delivered_token_carries_response_ready_timestamp() {
    let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
    let forward = Arc::new(MockForward::default());
    let mut actor = test_actor(
        rx,
        forward,
        8,
        false,
        1,
        false,
        ResponseDeliveryPolicy::default(),
    );
    let (response_tx, mut response_rx) = mpsc::channel(1);
    push_test_active(
        &mut actor,
        request(101, 2),
        response_tx,
        DecodeSlot::Mock {
            next_token: 101,
            generated_tokens: Vec::new(),
        },
    );

    let before = Instant::now();
    let ready_at = Instant::now();
    actor.submit_token_delivery(
        0,
        111,
        EngineTokenTiming::ready(ready_at, TokenPhaseDurations::default()),
    );
    let after = Instant::now();
    match response_rx.blocking_recv() {
        Some(EngineEvent::Token { token, timing }) => {
            assert_eq!(token, 111);
            assert!(timing.ready_at >= before);
            assert!(timing.ready_at <= after);
            assert!(timing.producer_delivered_at.is_some());
        }
        other => panic!("expected timed token, got {other:?}"),
    }
}

#[test]
fn decode_backend_phases_follow_owned_ready_tokens_only() {
    let mut backend_phases = BackendPhaseDurations::default();
    backend_phases.observe_gpu_lock_wait(Duration::from_millis(7));
    backend_phases.observe_synchronization(Duration::from_millis(11));
    backend_phases.observe_graph_capture(Duration::from_millis(13));
    backend_phases.observe_graph_replay(Duration::from_millis(17));
    let forward = Arc::new(MockForward {
        reported_backend_phases: backend_phases,
        ..MockForward::default()
    });
    let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
    let mut actor = test_actor(
        rx,
        forward,
        8,
        false,
        1,
        false,
        ResponseDeliveryPolicy::default(),
    );
    let (response_a_tx, mut response_a_rx) = mpsc::channel(2);
    push_test_active(
        &mut actor,
        request(101, 2),
        response_a_tx,
        DecodeSlot::Mock {
            next_token: 101,
            generated_tokens: Vec::new(),
        },
    );
    let (response_b_tx, mut response_b_rx) = mpsc::channel(2);
    push_test_active(
        &mut actor,
        request(201, 2),
        response_b_tx,
        DecodeSlot::Mock {
            next_token: 201,
            generated_tokens: Vec::new(),
        },
    );
    let (response_c_tx, _response_c_rx) = mpsc::channel(2);
    push_test_active(
        &mut actor,
        request(301, 2),
        response_c_tx,
        DecodeSlot::Mock {
            next_token: 301,
            generated_tokens: Vec::new(),
        },
    );
    actor.active[2].delivery_state = ActiveDeliveryState::InFlight { sequence: 0 };
    let non_ready_phases = actor.active[2].token_phase_durations;

    assert_eq!(actor.run_decode_batch(), 2);
    for (expected_token, response_rx) in [(111, &mut response_a_rx), (211, &mut response_b_rx)] {
        match response_rx.blocking_recv() {
            Some(EngineEvent::Token { token, timing }) => {
                assert_eq!(token, expected_token);
                assert_eq!(timing.phases_since_previous_token.backend, backend_phases);
            }
            other => panic!("expected timed token, got {other:?}"),
        }
    }
    assert_eq!(actor.active[2].token_phase_durations, non_ready_phases);
}

#[test]
fn admission_backend_phases_follow_new_and_already_active_requests() {
    let mut admission_phases = BackendPhaseDurations::default();
    admission_phases.observe_training(Duration::from_millis(7));
    let forward = Arc::new(MockForward {
        reported_admission_phases: admission_phases,
        ..MockForward::default()
    });
    let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
    let mut actor = test_actor(
        rx,
        forward,
        2,
        false,
        1,
        false,
        ResponseDeliveryPolicy::default(),
    );
    let (active_tx, mut active_rx) = mpsc::channel(4);
    push_test_active(
        &mut actor,
        request(101, 1),
        active_tx,
        DecodeSlot::Mock {
            next_token: 101,
            generated_tokens: Vec::new(),
        },
    );
    let (queued_tx, mut queued_rx) = mpsc::channel(4);
    queue_test_request(&mut actor, request(201, 1), queued_tx);

    assert_eq!(actor.admit_waiting(), false);
    assert_eq!(actor.run_decode_batch(), 2);
    for response in [&mut active_rx, &mut queued_rx] {
        match response.blocking_recv() {
            Some(EngineEvent::Token { timing, .. }) => assert_eq!(
                timing.phases_since_previous_token.backend.training,
                Some(Duration::from_millis(7))
            ),
            other => panic!("expected token carrying admission phases, got {other:?}"),
        }
    }
}

#[test]
fn resumable_prefill_backend_phases_reach_the_next_token() {
    let mut prefill_phases = BackendPhaseDurations::default();
    prefill_phases.observe_trim(Duration::from_millis(9));
    let forward = Arc::new(SyntheticPrefillForward {
        reported_prefill_phases: prefill_phases,
        ..SyntheticPrefillForward::default()
    });
    let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
    let mut actor = test_actor(
        rx,
        forward.clone(),
        1,
        false,
        1,
        false,
        ResponseDeliveryPolicy::default(),
    );
    let mut responses = push_synthetic_prefill_rows(&mut actor, &forward, &[(101, 1)]);

    assert!(actor.run_prefill_budget(64));
    assert_eq!(actor.run_decode_batch(), 1);
    match responses[0].blocking_recv() {
        Some(EngineEvent::Token { timing, .. }) => assert_eq!(
            timing.phases_since_previous_token.backend.trim,
            Some(Duration::from_millis(9))
        ),
        other => panic!("expected token carrying prefill phases, got {other:?}"),
    }
}

#[test]
fn exclusive_barrier_phases_follow_queued_requests_not_active_requests() {
    let forward = Arc::new(MockForward {
        resize_delay: Duration::from_millis(5),
        ..MockForward::default()
    });
    let (_tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
    let mut actor = test_actor(
        rx,
        forward,
        2,
        false,
        1,
        false,
        ResponseDeliveryPolicy::default(),
    );
    let (active_tx, mut active_rx) = mpsc::channel(4);
    push_test_active(
        &mut actor,
        request(101, 1),
        active_tx,
        DecodeSlot::Mock {
            next_token: 101,
            generated_tokens: Vec::new(),
        },
    );
    let (queued_tx, mut queued_rx) = mpsc::channel(4);
    queue_test_request(&mut actor, request(201, 1), queued_tx);

    let (resize_reply, resize_rx) = oneshot::channel();
    actor.handle_command(EngineCommand::ResizeKv {
        target_blocks: 17,
        reason: KvResizeReason::Maintenance,
        enqueued_at: Instant::now(),
        reply: resize_reply,
    });
    let (adapter_reply, adapter_rx) = oneshot::channel();
    actor.handle_command(EngineCommand::SwapAdapter {
        swap: Box::new(|| {
            thread::sleep(Duration::from_millis(5));
            Ok(())
        }),
        reply: adapter_reply,
    });

    assert_eq!(actor.run_decode_batch(), 1);
    match active_rx.blocking_recv() {
        Some(EngineEvent::Token { timing, .. }) => {
            assert_eq!(timing.phases_since_previous_token.backend.resize, None);
            assert_eq!(timing.phases_since_previous_token.backend.adapter, None);
        }
        other => panic!("expected active token, got {other:?}"),
    }

    actor.run_pending_exclusive_mutations_at_barrier();
    assert_eq!(resize_rx.blocking_recv().unwrap(), Ok(17));
    assert_eq!(adapter_rx.blocking_recv().unwrap(), Ok(()));
    assert_eq!(actor.admit_waiting(), false);
    assert_eq!(actor.run_decode_batch(), 1);
    match queued_rx.blocking_recv() {
        Some(EngineEvent::Token { timing, .. }) => {
            let backend = timing.phases_since_previous_token.backend;
            assert!(backend.resize >= Some(Duration::from_millis(5)));
            assert!(backend.adapter >= Some(Duration::from_millis(5)));
        }
        other => panic!("expected queued token, got {other:?}"),
    }
}
