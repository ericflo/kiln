//! LLM-as-judge scorer.
//!
//! Asks another model to grade the completion. The crate-level scorer only
//! formats the judge prompt and parses the score out of the reply; the
//! generation itself is dispatched through a `JudgeRunner` so the kiln-eval
//! crate stays GPU-free and the kiln-server crate can plug in the live model.

use regex::RegexBuilder;

use crate::result::EvalOutcomeKind;
use crate::scorers::{JudgeRunner, ScorerError};
use crate::suite::EvalExample;

pub fn default_judge_template() -> String {
    "You are a strict grader. The user asked:\n\n{question}\n\nReference answer:\n\n{target}\n\n\
Model answer:\n\n{answer}\n\n\
On a scale of 0 to 1, how correct is the model answer? Output ONLY the score, e.g. `Score: 1` for fully correct or `Score: 0` for incorrect. Half-credit is allowed.\n\nScore:"
        .to_string()
}

pub fn default_judge_regex() -> String {
    r"(?i)score[:\s]*([01](?:\.\d+)?)".to_string()
}

pub(super) fn score(
    example: &EvalExample,
    completion_text: &str,
    judge_adapter: Option<&str>,
    template: &str,
    score_regex: &str,
    judge_runner: &dyn JudgeRunner,
) -> Result<(f32, EvalOutcomeKind, Option<String>), ScorerError> {
    let prompt = render_template(example, completion_text, template);
    let re = RegexBuilder::new(score_regex)
        .build()
        .map_err(|e| ScorerError::InvalidRegex {
            pattern: score_regex.to_string(),
            msg: format!("{e}"),
        })?;
    let Some(reply) = judge_runner.judge(judge_adapter, &prompt) else {
        return Ok((
            0.0,
            EvalOutcomeKind::Invalid,
            Some("judge runner unavailable".into()),
        ));
    };
    let Some(caps) = re.captures(&reply) else {
        return Ok((
            0.0,
            EvalOutcomeKind::Invalid,
            Some(format!("score not found in judge reply: `{}`", truncate(&reply, 80))),
        ));
    };
    let group = caps.get(1).ok_or(ScorerError::InvalidRegex {
        pattern: score_regex.to_string(),
        msg: "regex must contain a capture group".into(),
    })?;
    let raw: f32 = match group.as_str().parse() {
        Ok(v) => v,
        Err(_) => {
            return Ok((
                0.0,
                EvalOutcomeKind::Invalid,
                Some(format!("captured `{}` is not a number", group.as_str())),
            ));
        }
    };
    let clamped = raw.clamp(0.0, 1.0);
    let kind = if clamped >= 0.5 {
        EvalOutcomeKind::Pass
    } else {
        EvalOutcomeKind::Fail
    };
    Ok((clamped, kind, Some(format!("judge score {clamped:.2}"))))
}

fn render_template(example: &EvalExample, completion_text: &str, template: &str) -> String {
    let question = example
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.as_str())
        .or_else(|| example.messages.last().map(|m| m.content.as_str()))
        .unwrap_or("");
    let target = example.target.as_deref().unwrap_or("");
    template
        .replace("{question}", question)
        .replace("{target}", target)
        .replace("{answer}", completion_text)
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let head: String = s.chars().take(max).collect();
        format!("{head}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::EvalChatMessage;

    struct MockJudge {
        reply: String,
    }
    impl JudgeRunner for MockJudge {
        fn judge(&self, _adapter: Option<&str>, _prompt: &str) -> Option<String> {
            Some(self.reply.clone())
        }
    }

    fn ex() -> EvalExample {
        EvalExample {
            id: None,
            messages: vec![EvalChatMessage {
                role: "user".into(),
                content: "What is the capital of France?".into(),
            }],
            target: Some("Paris".into()),
            aliases: Vec::new(),
            tags: Vec::new(),
            metadata: None,
            scorer: None,
            generation: None,
            weight: 1.0,
        }
    }

    #[test]
    fn passes_when_judge_returns_one() {
        let runner = MockJudge {
            reply: "Score: 1".into(),
        };
        let (s, kind, _) = score(
            &ex(),
            "Paris",
            None,
            &default_judge_template(),
            &default_judge_regex(),
            &runner,
        )
        .unwrap();
        assert_eq!(s, 1.0);
        assert_eq!(kind, EvalOutcomeKind::Pass);
    }

    #[test]
    fn fails_below_threshold() {
        let runner = MockJudge {
            reply: "Score: 0.2".into(),
        };
        let (_, kind, _) = score(
            &ex(),
            "Lyon",
            None,
            &default_judge_template(),
            &default_judge_regex(),
            &runner,
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Fail);
    }

    #[test]
    fn invalid_when_reply_unparseable() {
        let runner = MockJudge {
            reply: "I refuse to answer.".into(),
        };
        let (_, kind, _) = score(
            &ex(),
            "Paris",
            None,
            &default_judge_template(),
            &default_judge_regex(),
            &runner,
        )
        .unwrap();
        assert_eq!(kind, EvalOutcomeKind::Invalid);
    }

    #[test]
    fn placeholders_replace_correctly() {
        let template = "Q={question}|T={target}|A={answer}";
        let rendered = render_template(&ex(), "Paris", template);
        assert_eq!(rendered, "Q=What is the capital of France?|T=Paris|A=Paris");
    }
}
