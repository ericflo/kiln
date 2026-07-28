use super::*;

struct StarvingGrowForward {
    inner: MockForward,
    starve_once: std::sync::atomic::AtomicBool,
}

impl DecodeForward for StarvingGrowForward {
    fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
        self.inner.prepare_request(req)
    }

    fn grow_for_decode(&self, slots: &mut [&mut DecodeSlot]) -> Result<Vec<usize>> {
        if slots.len() > 1
            && self
                .starve_once
                .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(vec![0]);
        }
        Ok(Vec::new())
    }

    fn forward_decode(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<Vec<TokenId>> {
        self.inner.forward_decode(slots, sampling)
    }

    fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
        self.inner.accept_token(slot, token)
    }

    fn finish_request(
        &self,
        slot: DecodeSlot,
        finish_reason: FinishReason,
    ) -> Result<DecodeForwardOutput> {
        self.inner.finish_request(slot, finish_reason)
    }
}

#[tokio::test]
async fn transient_kv_growth_starvation_defers_without_truncation() {
    let forward = Arc::new(StarvingGrowForward {
        inner: MockForward::default(),
        starve_once: std::sync::atomic::AtomicBool::new(true),
    });
    let handle = BatchingEngineHandle::start_with_options(forward, 8);

    let rx_a = handle.enqueue(request(100, 5)).await.unwrap();
    let rx_b = handle.enqueue(request(200, 5)).await.unwrap();
    let outcome = |mut rx: mpsc::Receiver<EngineEvent>| async move {
        loop {
            match rx.recv().await {
                Some(EngineEvent::Done { output }) => break Ok(output),
                Some(EngineEvent::Error(error)) => break Err(error),
                Some(_) => {}
                None => break Err("closed".to_string()),
            }
        }
    };
    let (a, b) = tokio::join!(
        tokio::time::timeout(Duration::from_secs(10), outcome(rx_a)),
        tokio::time::timeout(Duration::from_secs(10), outcome(rx_b)),
    );
    let a = a.unwrap().expect("temporarily starved request completes");
    let b = b.unwrap().expect("survivor completes");
    assert_eq!(a.completion_tokens, 5);
    assert_eq!(b.completion_tokens, 5);
    handle.stop().await.unwrap();
}

struct ExhaustedGrowForward {
    inner: MockForward,
}

impl DecodeForward for ExhaustedGrowForward {
    fn prepare_request(&self, req: &EngineRequest) -> Result<DecodeSlot> {
        self.inner.prepare_request(req)
    }

    fn grow_for_decode(&self, slots: &mut [&mut DecodeSlot]) -> Result<Vec<usize>> {
        Ok((0..slots.len()).collect())
    }

    fn forward_decode(
        &self,
        slots: &mut [&mut DecodeSlot],
        sampling: &[SamplingParams],
    ) -> Result<Vec<TokenId>> {
        self.inner.forward_decode(slots, sampling)
    }

    fn accept_token(&self, slot: &mut DecodeSlot, token: TokenId) -> Result<usize> {
        self.inner.accept_token(slot, token)
    }

    fn finish_request(
        &self,
        slot: DecodeSlot,
        finish_reason: FinishReason,
    ) -> Result<DecodeForwardOutput> {
        self.inner.finish_request(slot, finish_reason)
    }
}

#[tokio::test]
async fn total_kv_growth_exhaustion_fails_explicitly() {
    let handle = BatchingEngineHandle::start_with_options(
        Arc::new(ExhaustedGrowForward {
            inner: MockForward::default(),
        }),
        8,
    );
    let mut response = handle.enqueue(request(100, 5)).await.unwrap();
    let terminal = tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match response.recv().await {
                Some(EngineEvent::Error(error)) => break error,
                Some(EngineEvent::Done { output }) => {
                    panic!("capacity exhaustion returned false success: {output:?}")
                }
                Some(_) => {}
                None => panic!("response closed without an explicit error"),
            }
        }
    })
    .await
    .expect("capacity exhaustion must not spin forever");
    assert!(terminal.contains("exhausted before the requested output length"));
    handle.stop().await.unwrap();
}
