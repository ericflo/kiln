use super::*;

pub(super) fn cache_value_from_response(
    resp: &ChatCompletionResponse,
) -> Option<DeterministicCompletionCacheValue> {
    let choice = resp.choices.first()?;
    Some(DeterministicCompletionCacheValue {
        text: response_content_for_cache(
            &choice.message.content,
            choice.message.reasoning_content.as_deref(),
        ),
        reasoning_content: choice.message.reasoning_content.clone(),
        tool_calls: choice.message.tool_calls.clone(),
        finish_reason: choice.finish_reason.clone(),
        completion_tokens: resp.usage.completion_tokens,
        thinking_budget_status: choice.thinking_budget,
    })
}

pub(super) fn store_deterministic_completion(
    state: &AppState,
    key: DeterministicCompletionCacheKey,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = cache_value_from_response(resp) else {
        return;
    };
    state
        .completion_cache
        .lock()
        .unwrap()
        .insert_complete_value(key, value);
}

pub(super) fn complete_deterministic_completion_owner(
    state: &AppState,
    key: DeterministicCompletionCacheKey,
    claim_id: DeterministicCacheClaimId,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = cache_value_from_response(resp) else {
        state.completion_cache.lock().unwrap().fail(&key, claim_id);
        return;
    };
    state
        .completion_cache
        .lock()
        .unwrap()
        .complete(key, claim_id, value);
}

pub(super) fn fail_deterministic_completion_owner(
    state: &AppState,
    key: &DeterministicCompletionCacheKey,
    claim_id: DeterministicCacheClaimId,
) {
    state.completion_cache.lock().unwrap().fail(key, claim_id);
}

pub(super) async fn wait_for_deterministic_completion(
    mut receiver: tokio::sync::watch::Receiver<DeterministicCompletionInFlightState>,
) -> Option<DeterministicCompletionCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicCompletionInFlightState::Pending => {}
            DeterministicCompletionInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}

pub(super) struct ChatRequestCacheOwnerGuard {
    cache: std::sync::Arc<std::sync::Mutex<DeterministicChatRequestCache>>,
    key: DeterministicCacheKey,
    claim_id: DeterministicCacheClaimId,
    active: bool,
}

impl ChatRequestCacheOwnerGuard {
    pub(super) fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicChatRequestCache>>,
        key: DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> Self {
        Self {
            cache,
            key,
            claim_id,
            active: true,
        }
    }

    pub(super) fn complete(mut self, value: DeterministicChatRequestCacheValue) {
        self.cache
            .lock()
            .unwrap()
            .complete(self.key.clone(), self.claim_id, value);
        self.active = false;
    }

    pub(super) fn matches_key(&self, key: &DeterministicCacheKey) -> bool {
        &self.key == key
    }
}

impl Drop for ChatRequestCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key, self.claim_id);
        }
    }
}

pub(super) async fn wait_for_deterministic_chat_request(
    mut receiver: tokio::sync::watch::Receiver<DeterministicChatRequestInFlightState>,
) -> Option<DeterministicChatRequestCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicChatRequestInFlightState::Pending => {}
            DeterministicChatRequestInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}

pub(super) struct ChatChoicesCacheOwnerGuard {
    cache: std::sync::Arc<std::sync::Mutex<DeterministicChatChoicesCache>>,
    key: DeterministicCacheKey,
    claim_id: DeterministicCacheClaimId,
    active: bool,
}

impl ChatChoicesCacheOwnerGuard {
    pub(super) fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicChatChoicesCache>>,
        key: DeterministicCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> Self {
        Self {
            cache,
            key,
            claim_id,
            active: true,
        }
    }

    pub(super) fn complete(mut self, value: DeterministicChatChoicesCacheValue) {
        self.cache
            .lock()
            .unwrap()
            .complete(self.key.clone(), self.claim_id, value);
        self.active = false;
    }

    pub(super) fn matches_key(&self, key: &DeterministicCacheKey) -> bool {
        &self.key == key
    }
}

impl Drop for ChatChoicesCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key, self.claim_id);
        }
    }
}

pub(super) async fn wait_for_deterministic_chat_choices(
    mut receiver: tokio::sync::watch::Receiver<DeterministicChatChoicesInFlightState>,
) -> Option<DeterministicChatChoicesCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicChatChoicesInFlightState::Pending => {}
            DeterministicChatChoicesInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}

pub(super) fn finish_chat_request_cache(
    state: &AppState,
    key: Option<DeterministicCacheKey>,
    owner: Option<ChatRequestCacheOwnerGuard>,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = chat_request_cache_value_from_response(resp) else {
        return;
    };
    finish_chat_request_cache_value(state, key, owner, value);
}

pub(super) fn finish_chat_request_cache_value(
    state: &AppState,
    key: Option<DeterministicCacheKey>,
    owner: Option<ChatRequestCacheOwnerGuard>,
    value: DeterministicChatRequestCacheValue,
) {
    if let Some(owner) = owner {
        owner.complete(value);
    } else if let Some(key) = key {
        state.chat_request_cache.lock().unwrap().insert(key, value);
    }
}

pub(super) fn finish_chat_choices_cache(
    state: &AppState,
    key: Option<DeterministicCacheKey>,
    owner: Option<ChatChoicesCacheOwnerGuard>,
    resp: &ChatCompletionResponse,
) {
    let Some(value) = chat_choices_cache_value_from_response(resp) else {
        return;
    };
    if let Some(owner) = owner {
        owner.complete(value);
    } else if let Some(key) = key {
        state.chat_choices_cache.lock().unwrap().insert(key, value);
    }
}

pub(super) struct BatchCacheOwnerGuard {
    cache: std::sync::Arc<std::sync::Mutex<DeterministicBatchCache>>,
    key: DeterministicBatchCacheKey,
    claim_id: DeterministicCacheClaimId,
    active: bool,
}

impl BatchCacheOwnerGuard {
    pub(super) fn new(
        cache: std::sync::Arc<std::sync::Mutex<DeterministicBatchCache>>,
        key: DeterministicBatchCacheKey,
        claim_id: DeterministicCacheClaimId,
    ) -> Self {
        Self {
            cache,
            key,
            claim_id,
            active: true,
        }
    }

    pub(super) fn complete(mut self, value: DeterministicBatchCacheValue) {
        self.cache
            .lock()
            .unwrap()
            .complete(self.key.clone(), self.claim_id, value);
        self.active = false;
    }

    pub(super) fn matches_key(&self, key: &DeterministicBatchCacheKey) -> bool {
        &self.key == key
    }
}

impl Drop for BatchCacheOwnerGuard {
    fn drop(&mut self) {
        if self.active {
            self.cache.lock().unwrap().fail(&self.key, self.claim_id);
        }
    }
}

pub(super) async fn wait_for_deterministic_batch(
    mut receiver: tokio::sync::watch::Receiver<DeterministicBatchInFlightState>,
) -> Option<DeterministicBatchCacheValue> {
    loop {
        match receiver.borrow().clone() {
            DeterministicBatchInFlightState::Pending => {}
            DeterministicBatchInFlightState::Ready(value) => return value,
        }

        if receiver.changed().await.is_err() {
            return None;
        }
    }
}
