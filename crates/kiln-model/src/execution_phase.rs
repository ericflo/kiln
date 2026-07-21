#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
use std::cell::RefCell;
use std::time::Duration;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
use std::time::Instant;

use kiln_core::token::TokenId;
use kiln_tensor::Device;

/// Tokens and request-correlatable backend work from one paged decode step.
///
/// A phase remains `None` when its boundary is fused with the transformer
/// forward or otherwise cannot be measured without inventing attribution.
#[derive(Debug)]
pub struct ProfiledPagedDecodeStep<T> {
    pub tokens: Vec<T>,
    pub sampling_duration: Option<Duration>,
    pub readback_duration: Option<Duration>,
    pub graph_capture_duration: Option<Duration>,
    pub graph_replay_duration: Option<Duration>,
}

/// Invocation-owned backend work carried with a direct model event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamBackendPhaseDurations {
    pub sampling: Option<Duration>,
    pub readback: Option<Duration>,
    pub graph_capture: Option<Duration>,
    pub graph_replay: Option<Duration>,
    pub synchronization: Option<Duration>,
}

impl StreamBackendPhaseDurations {
    fn add_observation(total: &mut Option<Duration>, duration: Duration) {
        *total = Some(total.unwrap_or(Duration::ZERO).saturating_add(duration));
    }

    pub(crate) fn observe_sampling(&mut self, duration: Duration) {
        Self::add_observation(&mut self.sampling, duration);
    }

    pub(crate) fn observe_readback(&mut self, duration: Duration) {
        Self::add_observation(&mut self.readback, duration);
    }

    pub(crate) fn observe_graph_capture(&mut self, duration: Duration) {
        Self::add_observation(&mut self.graph_capture, duration);
    }

    pub(crate) fn observe_graph_replay(&mut self, duration: Duration) {
        Self::add_observation(&mut self.graph_replay, duration);
    }

    pub(crate) fn observe_synchronization(&mut self, duration: Duration) {
        Self::add_observation(&mut self.synchronization, duration);
    }
}

#[derive(Debug)]
pub(crate) struct ProfiledDirectDecodeStep {
    pub(crate) token: TokenId,
    pub(crate) backend_phases: StreamBackendPhaseDurations,
}

impl ProfiledDirectDecodeStep {
    pub(crate) fn without_distinct_backend_phase(token: TokenId) -> Self {
        Self {
            token,
            backend_phases: StreamBackendPhaseDurations::default(),
        }
    }
}

pub(crate) fn add_profiled_duration(total: &mut Option<Duration>, duration: Duration) {
    *total = Some(total.unwrap_or(Duration::ZERO).saturating_add(duration));
}

pub(crate) fn add_profiled_sampling_tail(
    sampling_total: &mut Option<Duration>,
    readback_total: &mut Option<Duration>,
    total: Duration,
    readback: Option<Duration>,
) {
    if let Some(readback) = readback {
        add_profiled_duration(sampling_total, total.saturating_sub(readback));
        add_profiled_duration(readback_total, readback);
    } else {
        add_profiled_duration(sampling_total, total);
    }
}

pub(crate) fn merge_profiled_graph_paged_step<T>(
    profiled: ProfiledGraphValue<ProfiledPagedDecodeStep<T>>,
) -> ProfiledPagedDecodeStep<T> {
    let mut step = profiled.value;
    if let Some(duration) = profiled.capture_duration {
        add_profiled_duration(&mut step.graph_capture_duration, duration);
    }
    if let Some(duration) = profiled.replay_duration {
        add_profiled_duration(&mut step.graph_replay_duration, duration);
    }
    step
}

pub(crate) fn merge_profiled_graph_direct_step(
    profiled: ProfiledGraphValue<ProfiledDirectDecodeStep>,
) -> ProfiledDirectDecodeStep {
    let mut step = profiled.value;
    if let Some(duration) = profiled.capture_duration {
        step.backend_phases.observe_graph_capture(duration);
    }
    if let Some(duration) = profiled.replay_duration {
        step.backend_phases.observe_graph_replay(duration);
    }
    step
}

fn merge_profiled_readback_paged_step<T>(
    mut step: ProfiledPagedDecodeStep<T>,
    readback_duration: Option<Duration>,
) -> ProfiledPagedDecodeStep<T> {
    if let Some(duration) = readback_duration {
        if let Some(sampling) = step.sampling_duration.as_mut() {
            *sampling = sampling.saturating_sub(duration);
        }
        add_profiled_duration(&mut step.readback_duration, duration);
    }
    step
}

fn merge_profiled_readback_direct_step(
    mut step: ProfiledDirectDecodeStep,
    readback_duration: Option<Duration>,
) -> ProfiledDirectDecodeStep {
    if let Some(duration) = readback_duration {
        if let Some(sampling) = step.backend_phases.sampling.as_mut() {
            *sampling = sampling.saturating_sub(duration);
        }
        step.backend_phases.observe_readback(duration);
    }
    step
}

pub(crate) fn profile_paged_decode_invocation<T>(
    operation: impl FnOnce() -> anyhow::Result<ProfiledPagedDecodeStep<T>>,
) -> anyhow::Result<ProfiledPagedDecodeStep<T>> {
    let profiled = profile_readback_invocation(|| profile_graph_invocation(operation))?;
    Ok(merge_profiled_readback_paged_step(
        merge_profiled_graph_paged_step(profiled.value),
        profiled.readback_duration,
    ))
}

pub(crate) fn profile_direct_decode_invocation(
    operation: impl FnOnce() -> anyhow::Result<ProfiledDirectDecodeStep>,
) -> anyhow::Result<ProfiledDirectDecodeStep> {
    let profiled = profile_readback_invocation(|| profile_graph_invocation(operation))?;
    Ok(merge_profiled_readback_direct_step(
        merge_profiled_graph_direct_step(profiled.value),
        profiled.readback_duration,
    ))
}

#[derive(Debug)]
pub(crate) struct ProfiledReadbackValue<T> {
    pub(crate) value: T,
    pub(crate) readback_duration: Option<Duration>,
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
thread_local! {
    static ACTIVE_READBACK_INVOCATIONS: RefCell<Vec<Option<Duration>>> = const {
        RefCell::new(Vec::new())
    };
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
struct ReadbackInvocationScope {
    active: bool,
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
impl ReadbackInvocationScope {
    fn begin() -> Self {
        ACTIVE_READBACK_INVOCATIONS.with(|scopes| scopes.borrow_mut().push(None));
        Self { active: true }
    }

    fn finish(mut self) -> Option<Duration> {
        self.active = false;
        ACTIVE_READBACK_INVOCATIONS.with(|scopes| {
            scopes
                .borrow_mut()
                .pop()
                .expect("readback invocation scope stack must be balanced")
        })
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
impl Drop for ReadbackInvocationScope {
    fn drop(&mut self) {
        if self.active {
            ACTIVE_READBACK_INVOCATIONS.with(|scopes| {
                scopes.borrow_mut().pop();
            });
        }
    }
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
pub(crate) fn profile_readback_invocation<T>(
    operation: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<ProfiledReadbackValue<T>> {
    let scope = ReadbackInvocationScope::begin();
    let result = operation();
    let readback_duration = scope.finish();
    result.map(|value| ProfiledReadbackValue {
        value,
        readback_duration,
    })
}

#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
)))]
pub(crate) fn profile_readback_invocation<T>(
    operation: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<ProfiledReadbackValue<T>> {
    operation().map(|value| ProfiledReadbackValue {
        value,
        readback_duration: None,
    })
}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
pub(crate) fn observe_profiled_readback(duration: Duration) {
    ACTIVE_READBACK_INVOCATIONS.with(|scopes| {
        if let Some(active) = scopes.borrow_mut().last_mut() {
            *active = Some(active.unwrap_or(Duration::ZERO).saturating_add(duration));
        }
    });
}

#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
)))]
pub(crate) fn observe_profiled_readback(_duration: Duration) {}

#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
))]
pub(crate) fn profile_accelerator_readback<T, E>(
    device: Device,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    let started = (device.is_gpu()
        && ACTIVE_READBACK_INVOCATIONS.with(|scopes| !scopes.borrow().is_empty()))
    .then(Instant::now);
    let result = operation();
    if let Some(started) = started {
        observe_profiled_readback(started.elapsed());
    }
    result
}

#[cfg(not(any(
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "vulkan",
    test
)))]
pub(crate) fn profile_accelerator_readback<T, E>(
    _device: Device,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    operation()
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GraphPhaseDurations {
    pub(crate) capture: Option<Duration>,
    pub(crate) replay: Option<Duration>,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl GraphPhaseDurations {
    fn add(total: &mut Option<Duration>, duration: Duration) {
        *total = Some(total.unwrap_or(Duration::ZERO).saturating_add(duration));
    }

    fn observe(&mut self, phase: GraphPhase, duration: Duration) {
        match phase {
            GraphPhase::Capture => Self::add(&mut self.capture, duration),
            GraphPhase::Replay => Self::add(&mut self.replay, duration),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ProfiledGraphValue<T> {
    pub(crate) value: T,
    pub(crate) capture_duration: Option<Duration>,
    pub(crate) replay_duration: Option<Duration>,
}

impl<T> ProfiledGraphValue<T> {
    pub(crate) fn without_graph_work(value: T) -> Self {
        Self {
            value,
            capture_duration: None,
            replay_duration: None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) enum GraphPhase {
    Capture,
    Replay,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
thread_local! {
    static ACTIVE_GRAPH_INVOCATIONS: RefCell<Vec<GraphPhaseDurations>> = const {
        RefCell::new(Vec::new())
    };
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
struct GraphInvocationScope {
    active: bool,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl GraphInvocationScope {
    fn begin() -> Self {
        ACTIVE_GRAPH_INVOCATIONS.with(|scopes| scopes.borrow_mut().push(Default::default()));
        Self { active: true }
    }

    fn finish(mut self) -> GraphPhaseDurations {
        self.active = false;
        ACTIVE_GRAPH_INVOCATIONS.with(|scopes| {
            scopes
                .borrow_mut()
                .pop()
                .expect("graph invocation scope stack must be balanced")
        })
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl Drop for GraphInvocationScope {
    fn drop(&mut self) {
        if self.active {
            ACTIVE_GRAPH_INVOCATIONS.with(|scopes| {
                scopes.borrow_mut().pop();
            });
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) fn profile_graph_invocation<T>(
    operation: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<ProfiledGraphValue<T>> {
    let scope = GraphInvocationScope::begin();
    let result = operation();
    let phases = scope.finish();
    result.map(|value| ProfiledGraphValue {
        value,
        capture_duration: phases.capture,
        replay_duration: phases.replay,
    })
}

#[cfg(not(any(feature = "cuda", feature = "metal", test)))]
pub(crate) fn profile_graph_invocation<T>(
    operation: impl FnOnce() -> anyhow::Result<T>,
) -> anyhow::Result<ProfiledGraphValue<T>> {
    operation().map(ProfiledGraphValue::without_graph_work)
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn graph_profile_active() -> bool {
    ACTIVE_GRAPH_INVOCATIONS.with(|scopes| !scopes.borrow().is_empty())
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
fn observe_graph_phase(phase: GraphPhase, duration: Duration) {
    ACTIVE_GRAPH_INVOCATIONS.with(|scopes| {
        if let Some(active) = scopes.borrow_mut().last_mut() {
            active.observe(phase, duration);
        }
    });
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
pub(crate) struct GraphPhaseTimer {
    phase: GraphPhase,
    started: Option<Instant>,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl GraphPhaseTimer {
    pub(crate) fn start(phase: GraphPhase) -> Self {
        Self {
            phase,
            started: graph_profile_active().then(Instant::now),
        }
    }
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl Drop for GraphPhaseTimer {
    fn drop(&mut self) {
        if let Some(started) = self.started {
            observe_graph_phase(self.phase, started.elapsed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_profile_keeps_unsupported_phases_null() {
        let profiled = profile_graph_invocation(|| Ok(7_u32)).unwrap();
        assert_eq!(profiled.value, 7);
        assert_eq!(profiled.capture_duration, None);
        assert_eq!(profiled.replay_duration, None);
    }

    #[test]
    fn profiled_duration_preserves_unavailable_and_accumulates_observations() {
        let mut duration = None;
        assert_eq!(duration, None);
        add_profiled_duration(&mut duration, Duration::from_millis(7));
        add_profiled_duration(&mut duration, Duration::from_millis(11));
        assert_eq!(duration, Some(Duration::from_millis(18)));
    }

    #[test]
    fn sampling_tail_separates_existing_readback_without_double_counting() {
        let mut sampling = None;
        let mut readback = None;
        add_profiled_sampling_tail(
            &mut sampling,
            &mut readback,
            Duration::from_millis(25),
            Some(Duration::from_millis(7)),
        );
        add_profiled_sampling_tail(
            &mut sampling,
            &mut readback,
            Duration::from_millis(10),
            None,
        );
        assert_eq!(sampling, Some(Duration::from_millis(28)));
        assert_eq!(readback, Some(Duration::from_millis(7)));

        let mut zero_sampling = None;
        let mut zero_readback = None;
        add_profiled_sampling_tail(
            &mut zero_sampling,
            &mut zero_readback,
            Duration::ZERO,
            Some(Duration::ZERO),
        );
        assert_eq!(zero_sampling, Some(Duration::ZERO));
        assert_eq!(zero_readback, Some(Duration::ZERO));
    }

    #[test]
    fn direct_stream_phases_preserve_measured_zero_and_accumulate() {
        let mut phases = StreamBackendPhaseDurations::default();
        phases.observe_sampling(Duration::ZERO);
        phases.observe_sampling(Duration::from_millis(9));
        phases.observe_readback(Duration::ZERO);
        phases.observe_readback(Duration::from_millis(2));
        phases.observe_graph_capture(Duration::ZERO);
        phases.observe_graph_capture(Duration::from_millis(7));
        phases.observe_graph_replay(Duration::from_millis(3));
        phases.observe_synchronization(Duration::from_millis(4));
        assert_eq!(phases.sampling, Some(Duration::from_millis(9)));
        assert_eq!(phases.readback, Some(Duration::from_millis(2)));
        assert_eq!(phases.graph_capture, Some(Duration::from_millis(7)));
        assert_eq!(phases.graph_replay, Some(Duration::from_millis(3)));
        assert_eq!(phases.synchronization, Some(Duration::from_millis(4)));
    }

    #[test]
    fn graph_profile_sums_only_the_current_invocation() {
        let profiled = profile_graph_invocation(|| {
            observe_graph_phase(GraphPhase::Capture, Duration::from_millis(3));
            observe_graph_phase(GraphPhase::Capture, Duration::from_millis(5));
            observe_graph_phase(GraphPhase::Replay, Duration::ZERO);
            Ok(())
        })
        .unwrap();
        assert_eq!(profiled.capture_duration, Some(Duration::from_millis(8)));
        assert_eq!(profiled.replay_duration, Some(Duration::ZERO));
    }

    #[test]
    fn graph_phase_timer_records_only_inside_a_profile() {
        let profiled = profile_graph_invocation(|| {
            let phase = GraphPhaseTimer::start(GraphPhase::Replay);
            std::thread::yield_now();
            drop(phase);
            Ok(())
        })
        .unwrap();
        assert!(profiled.replay_duration.is_some());

        let phase = GraphPhaseTimer::start(GraphPhase::Capture);
        drop(phase);
        let next = profile_graph_invocation(|| Ok(())).unwrap();
        assert_eq!(next.capture_duration, None);
    }

    #[test]
    fn nested_graph_profiles_do_not_contaminate_their_parent() {
        let outer = profile_graph_invocation(|| {
            observe_graph_phase(GraphPhase::Capture, Duration::from_millis(2));
            let inner = profile_graph_invocation(|| {
                observe_graph_phase(GraphPhase::Replay, Duration::from_millis(11));
                Ok(())
            })?;
            assert_eq!(inner.replay_duration, Some(Duration::from_millis(11)));
            Ok(())
        })
        .unwrap();
        assert_eq!(outer.capture_duration, Some(Duration::from_millis(2)));
        assert_eq!(outer.replay_duration, None);
    }

    #[test]
    fn failed_graph_profile_does_not_leak_scope() {
        let failed = profile_graph_invocation::<()>(|| anyhow::bail!("expected"));
        assert!(failed.is_err());
        let next = profile_graph_invocation(|| Ok(())).unwrap();
        assert_eq!(next.capture_duration, None);
        assert_eq!(next.replay_duration, None);
    }

    #[test]
    fn readback_profile_preserves_null_zero_and_sums_current_invocation() {
        let unsupported = profile_readback_invocation(|| Ok(7_u32)).unwrap();
        assert_eq!(unsupported.value, 7);
        assert_eq!(unsupported.readback_duration, None);

        let profiled = profile_readback_invocation(|| {
            observe_profiled_readback(Duration::ZERO);
            observe_profiled_readback(Duration::from_millis(5));
            Ok(11_u32)
        })
        .unwrap();
        assert_eq!(profiled.value, 11);
        assert_eq!(profiled.readback_duration, Some(Duration::from_millis(5)));
    }

    #[test]
    fn accelerator_readback_times_only_gpu_operations_inside_a_scope() {
        let profiled = profile_readback_invocation(|| {
            profile_accelerator_readback(Device::Cpu, || Ok::<_, anyhow::Error>(()))?;
            profile_accelerator_readback(Device::Cuda(0), || {
                std::thread::yield_now();
                Ok::<_, anyhow::Error>(())
            })?;
            Ok(())
        })
        .unwrap();
        assert!(profiled.readback_duration.is_some());

        profile_accelerator_readback(Device::Cuda(0), || Ok::<_, anyhow::Error>(())).unwrap();
        let next = profile_readback_invocation(|| Ok(())).unwrap();
        assert_eq!(next.readback_duration, None);
    }

    #[test]
    fn nested_readback_profiles_do_not_contaminate_their_parent() {
        let outer = profile_readback_invocation(|| {
            observe_profiled_readback(Duration::from_millis(2));
            let inner = profile_readback_invocation(|| {
                observe_profiled_readback(Duration::from_millis(11));
                Ok(())
            })?;
            assert_eq!(inner.readback_duration, Some(Duration::from_millis(11)));
            Ok(())
        })
        .unwrap();
        assert_eq!(outer.readback_duration, Some(Duration::from_millis(2)));
    }

    #[test]
    fn failed_readback_profile_does_not_leak_scope() {
        let failed = profile_readback_invocation::<()>(|| {
            observe_profiled_readback(Duration::from_millis(3));
            anyhow::bail!("expected")
        });
        assert!(failed.is_err());
        let next = profile_readback_invocation(|| Ok(())).unwrap();
        assert_eq!(next.readback_duration, None);
    }

    #[test]
    fn decode_invocation_separates_readback_from_sampling_without_double_counting() {
        let paged = profile_paged_decode_invocation(|| {
            observe_profiled_readback(Duration::from_millis(7));
            Ok(ProfiledPagedDecodeStep {
                tokens: vec![17_u32],
                sampling_duration: Some(Duration::from_millis(25)),
                readback_duration: Some(Duration::from_millis(3)),
                graph_capture_duration: None,
                graph_replay_duration: None,
            })
        })
        .unwrap();
        assert_eq!(paged.sampling_duration, Some(Duration::from_millis(18)));
        assert_eq!(paged.readback_duration, Some(Duration::from_millis(10)));

        let direct = profile_direct_decode_invocation(|| {
            observe_profiled_readback(Duration::ZERO);
            let mut step = ProfiledDirectDecodeStep::without_distinct_backend_phase(23);
            step.backend_phases
                .observe_sampling(Duration::from_millis(9));
            Ok(step)
        })
        .unwrap();
        assert_eq!(
            direct.backend_phases.sampling,
            Some(Duration::from_millis(9))
        );
        assert_eq!(direct.backend_phases.readback, Some(Duration::ZERO));
    }

    #[test]
    fn request_graph_profile_merge_preserves_null_zero_and_existing_work() {
        let paged = merge_profiled_graph_paged_step(ProfiledGraphValue {
            value: ProfiledPagedDecodeStep {
                tokens: vec![17_u32],
                sampling_duration: None,
                readback_duration: None,
                graph_capture_duration: Some(Duration::from_millis(2)),
                graph_replay_duration: None,
            },
            capture_duration: Some(Duration::from_millis(3)),
            replay_duration: Some(Duration::ZERO),
        });
        assert_eq!(paged.tokens, vec![17]);
        assert_eq!(paged.graph_capture_duration, Some(Duration::from_millis(5)));
        assert_eq!(paged.graph_replay_duration, Some(Duration::ZERO));

        let direct = merge_profiled_graph_direct_step(ProfiledGraphValue {
            value: ProfiledDirectDecodeStep::without_distinct_backend_phase(23),
            capture_duration: None,
            replay_duration: Some(Duration::ZERO),
        });
        assert_eq!(direct.token, 23);
        assert_eq!(direct.backend_phases.graph_capture, None);
        assert_eq!(direct.backend_phases.graph_replay, Some(Duration::ZERO));
    }
}
