//! NaN/Inf diagnostics for explicit autograd tape scopes.
//!
//! Anomaly detection is selected when a [`crate::Tape`] is created through
//! [`crate::TapeOptions`]. The policy is immutable for that tape, so one
//! training request cannot change another request's behavior through process
//! state. When enabled, [`crate::Tape::backward`] scans every backward-op
//! gradient and traps at the first producing tape node.

/// Panic with a stable, structured anomaly-detection message.
///
/// `node_index` is the offset into `Tape::nodes()` and `op_name` matches the
/// offending [`crate::BackwardOp::name`] value.
#[track_caller]
pub fn anomaly_panic(node_index: usize, op_name: &str, detail: &str) -> ! {
    panic!(
        "kiln_autograd: anomaly detected at tape position {node_index} \
         (op `{op_name}`): {detail}. \
         Set detect_anomaly=false in the training job config to disable this trap."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "kiln_autograd: anomaly detected at tape position 7")]
    fn anomaly_panic_contains_position_and_op_name() {
        anomaly_panic(7, "test/some_op", "grad[3] is NaN");
    }

    #[test]
    #[should_panic(expected = "op `test/some_op`")]
    fn anomaly_panic_contains_op_name_in_backticks() {
        anomaly_panic(0, "test/some_op", "grad[0] is +Inf");
    }
}
