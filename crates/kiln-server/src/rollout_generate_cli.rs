use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use regex::Regex;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct RolloutGenerateOptions {
    pub url: String,
    pub adapter: String,
    pub thinking: bool,
    pub tasks: PathBuf,
    pub seeds: usize,
    pub seed_start: u64,
    pub request_template: PathBuf,
    pub scorer: PathBuf,
    pub output: PathBuf,
    pub summary_output: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RolloutGenerateSummary {
    pub adapter: Option<String>,
    pub adapter_label: String,
    pub url: String,
    pub thinking_enabled: bool,
    pub task_count: usize,
    pub seeds: usize,
    pub seed_start: u64,
    pub group_count: usize,
    pub completion_count: usize,
    pub tasks_path: String,
    pub request_template_path: String,
    pub scorer_path: String,
    pub output_path: String,
    pub summary_output_path: String,
    pub tasks_sha256: String,
    pub request_template_sha256: String,
    pub scorer_sha256: Option<String>,
    pub stats: RolloutGenerateStats,
    pub warnings: Vec<String>,
    pub completions: Vec<RolloutCompletionSummary>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RolloutGenerateStats {
    pub mean_reward: f64,
    pub min_reward: Option<f64>,
    pub max_reward: Option<f64>,
    pub total_prompt_tokens: usize,
    pub total_completion_tokens: usize,
    pub total_tokens: usize,
    pub mean_client_latency_ms: f64,
    pub max_client_latency_ms: f64,
    pub mean_server_latency_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutCompletionSummary {
    pub task_index: usize,
    pub task_id: String,
    pub seed: u64,
    pub reward: f64,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub client_latency_ms: f64,
    pub server_total_latency_ms: Option<f64>,
    pub finish_reason: String,
    pub content_chars: usize,
    pub response_id: Option<String>,
}

#[derive(Debug)]
struct RolloutTask {
    index: usize,
    id: String,
    value: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct RolloutChatMessage {
    role: String,
    content: String,
}

#[derive(Debug)]
struct AdapterSelection {
    request_value: Option<String>,
    label: String,
}

#[derive(Debug)]
struct RewardScore {
    reward: f64,
    raw: Value,
}

#[derive(Debug, Clone, Serialize)]
struct TokenUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

pub async fn run_rollout_generate(
    options: RolloutGenerateOptions,
) -> Result<RolloutGenerateSummary> {
    if options.seeds == 0 {
        anyhow::bail!("--seeds must be greater than zero");
    }

    let adapter = parse_adapter_selection(&options.adapter);
    let started = Instant::now();
    let tasks_bytes = std::fs::read(&options.tasks)
        .with_context(|| format!("reading tasks JSONL {}", options.tasks.display()))?;
    let template_bytes = std::fs::read(&options.request_template).with_context(|| {
        format!(
            "reading request template {}",
            options.request_template.display()
        )
    })?;
    let scorer_sha256 = sha256_file(&options.scorer).ok();
    let tasks = parse_tasks_jsonl(&tasks_bytes)
        .with_context(|| format!("parsing tasks JSONL {}", options.tasks.display()))?;
    let template: Value = serde_json::from_slice(&template_bytes).with_context(|| {
        format!(
            "parsing request template {}",
            options.request_template.display()
        )
    })?;

    if tasks.is_empty() {
        anyhow::bail!("tasks JSONL contained no tasks");
    }

    let client = reqwest::Client::new();
    let mut writer = std::io::BufWriter::new(
        std::fs::File::create(&options.output)
            .with_context(|| format!("creating {}", options.output.display()))?,
    );
    let mut summaries = Vec::with_capacity(tasks.len() * options.seeds);
    let mut warnings = Vec::new();

    for task in &tasks {
        let mut group_messages: Option<Vec<RolloutChatMessage>> = None;
        let mut completions = Vec::with_capacity(options.seeds);

        for seed_index in 0..options.seeds {
            let seed = options.seed_start.wrapping_add(seed_index as u64);
            let request = render_rollout_request(
                &template,
                &task.value,
                seed,
                adapter.request_value.as_deref(),
                options.thinking,
            )?;
            let request_messages = extract_messages_for_group(&request)?;
            match &group_messages {
                Some(existing) if existing != &request_messages => {
                    anyhow::bail!(
                        "request template rendered different messages for task {} across seeds; GRPO groups require one shared prompt",
                        task.id
                    );
                }
                None => group_messages = Some(request_messages),
                _ => {}
            }

            let request_started = Instant::now();
            let response = post_chat_completion(&client, &options.url, &request)
                .await
                .with_context(|| {
                    format!(
                        "completion failed for task {} seed {} adapter {}",
                        task.id, seed, adapter.label
                    )
                })?;
            let client_latency_ms = request_started.elapsed().as_secs_f64() * 1000.0;
            let content = extract_chat_content(&response);
            let finish_reason = extract_finish_reason(&response);
            let usage = extract_usage(&response);
            let server_total_latency_ms = response
                .pointer("/metadata/performance/total_latency_ms")
                .and_then(Value::as_f64);

            if usage.total_tokens == 0 {
                warnings.push(format!(
                    "task {} seed {} response did not include nonzero usage token counts",
                    task.id, seed
                ));
            }
            if server_total_latency_ms.is_none() {
                warnings.push(format!(
                    "task {} seed {} response did not include metadata.performance.total_latency_ms",
                    task.id, seed
                ));
            }

            let scorer_input = json!({
                "task_index": task.index,
                "task_id": task.id,
                "task": task.value,
                "seed": seed,
                "adapter": adapter.request_value,
                "adapter_label": adapter.label,
                "thinking_enabled": options.thinking,
                "request": request,
                "response": response,
                "content": content,
                "finish_reason": finish_reason,
                "usage": usage,
                "latency": {
                    "client_ms": client_latency_ms,
                    "server_total_ms": server_total_latency_ms
                }
            });
            let score = run_scorer(&options.scorer, &scorer_input)
                .with_context(|| format!("scoring task {} seed {}", task.id, seed))?;

            let completion_summary = RolloutCompletionSummary {
                task_index: task.index,
                task_id: task.id.clone(),
                seed,
                reward: score.reward,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                client_latency_ms,
                server_total_latency_ms,
                finish_reason: finish_reason.clone(),
                content_chars: content.chars().count(),
                response_id: response
                    .get("id")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            };
            summaries.push(completion_summary.clone());

            completions.push(json!({
                "text": content,
                "reward": score.reward,
                "trajectory": [{
                    "role": "assistant",
                    "content": content,
                    "kind": "action"
                }],
                "metadata": {
                    "task_index": task.index,
                    "task_id": task.id,
                    "seed": seed,
                    "adapter": adapter.request_value,
                    "adapter_label": adapter.label,
                    "thinking_enabled": options.thinking,
                    "usage": usage,
                    "latency": {
                        "client_ms": client_latency_ms,
                        "server_total_ms": server_total_latency_ms
                    },
                    "performance": response.pointer("/metadata/performance").cloned(),
                    "finish_reason": finish_reason,
                    "response_id": response.get("id").cloned(),
                    "scorer_output": score.raw
                }
            }));
        }

        let group = json!({
            "messages": group_messages.unwrap_or_default(),
            "completions": completions,
            "metadata": {
                "source": "kiln rollout-generate",
                "task_index": task.index,
                "task_id": task.id,
                "adapter": adapter.request_value,
                "adapter_label": adapter.label,
                "thinking_enabled": options.thinking,
                "seed_start": options.seed_start,
                "seeds": options.seeds
            }
        });
        writeln!(writer, "{}", serde_json::to_string(&group)?)
            .with_context(|| format!("writing {}", options.output.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing {}", options.output.display()))?;

    let stats = summarize_completions(&summaries);
    let summary = RolloutGenerateSummary {
        adapter: adapter.request_value,
        adapter_label: adapter.label,
        url: options.url,
        thinking_enabled: options.thinking,
        task_count: tasks.len(),
        seeds: options.seeds,
        seed_start: options.seed_start,
        group_count: tasks.len(),
        completion_count: summaries.len(),
        tasks_path: options.tasks.display().to_string(),
        request_template_path: options.request_template.display().to_string(),
        scorer_path: options.scorer.display().to_string(),
        output_path: options.output.display().to_string(),
        summary_output_path: options.summary_output.display().to_string(),
        tasks_sha256: sha256_bytes(&tasks_bytes),
        request_template_sha256: sha256_bytes(&template_bytes),
        scorer_sha256,
        stats,
        warnings,
        completions: summaries,
    };

    let mut summary_value = serde_json::to_value(&summary)?;
    if let Some(obj) = summary_value.as_object_mut() {
        obj.insert(
            "wall_clock_ms".to_string(),
            json!(started.elapsed().as_secs_f64() * 1000.0),
        );
    }
    std::fs::write(
        &options.summary_output,
        serde_json::to_vec_pretty(&summary_value)?,
    )
    .with_context(|| format!("writing {}", options.summary_output.display()))?;

    Ok(summary)
}

fn parse_adapter_selection(raw: &str) -> AdapterSelection {
    let trimmed = raw.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("base")
        || trimmed.eq_ignore_ascii_case("none")
        || trimmed.eq_ignore_ascii_case("null")
    {
        AdapterSelection {
            request_value: None,
            label: "base".to_string(),
        }
    } else {
        AdapterSelection {
            request_value: Some(trimmed.to_string()),
            label: trimmed.to_string(),
        }
    }
}

fn parse_tasks_jsonl(bytes: &[u8]) -> Result<Vec<RolloutTask>> {
    let mut tasks = Vec::new();
    let text = std::str::from_utf8(bytes).context("tasks file is not valid UTF-8")?;
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .with_context(|| format!("invalid JSON on line {}", line_index + 1))?;
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| (tasks.len() + 1).to_string());
        tasks.push(RolloutTask {
            index: tasks.len(),
            id,
            value,
        });
    }
    Ok(tasks)
}

fn render_rollout_request(
    template: &Value,
    task: &Value,
    seed: u64,
    adapter: Option<&str>,
    thinking: bool,
) -> Result<Value> {
    let mut value = render_template_value(template, task, seed, adapter, thinking)?;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("request template must render to a JSON object"))?;
    obj.insert("seed".to_string(), json!(seed));
    obj.insert(
        "adapter".to_string(),
        adapter.map_or(Value::Null, |name| Value::String(name.to_string())),
    );
    obj.insert("stream".to_string(), Value::Bool(false));
    obj.insert("include_performance".to_string(), Value::Bool(true));
    let kwargs = obj
        .entry("chat_template_kwargs".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let kwargs = kwargs
        .as_object_mut()
        .ok_or_else(|| anyhow!("chat_template_kwargs must be a JSON object"))?;
    kwargs.insert("enable_thinking".to_string(), Value::Bool(thinking));
    Ok(value)
}

fn render_template_value(
    value: &Value,
    task: &Value,
    seed: u64,
    adapter: Option<&str>,
    thinking: bool,
) -> Result<Value> {
    match value {
        Value::String(s) => render_template_string(s, task, seed, adapter, thinking),
        Value::Array(items) => items
            .iter()
            .map(|item| render_template_value(item, task, seed, adapter, thinking))
            .collect(),
        Value::Object(map) => {
            let mut rendered = Map::new();
            for (key, item) in map {
                rendered.insert(
                    key.clone(),
                    render_template_value(item, task, seed, adapter, thinking)?,
                );
            }
            Ok(Value::Object(rendered))
        }
        other => Ok(other.clone()),
    }
}

fn render_template_string(
    template: &str,
    task: &Value,
    seed: u64,
    adapter: Option<&str>,
    thinking: bool,
) -> Result<Value> {
    let re = Regex::new(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}").expect("valid placeholder regex");
    if let Some(caps) = re.captures(template)
        && caps.get(0).is_some_and(|m| m.as_str() == template)
    {
        let name = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        if let Some(value) = placeholder_value(name, task, seed, adapter, thinking) {
            return Ok(value);
        }
    }

    let mut rendered = String::with_capacity(template.len());
    let mut cursor = 0;
    for caps in re.captures_iter(template) {
        let full = caps.get(0).unwrap();
        rendered.push_str(&template[cursor..full.start()]);
        let name = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        if let Some(value) = placeholder_value(name, task, seed, adapter, thinking) {
            rendered.push_str(&placeholder_to_string(&value));
        } else {
            rendered.push_str(full.as_str());
        }
        cursor = full.end();
    }
    rendered.push_str(&template[cursor..]);
    Ok(Value::String(rendered))
}

fn placeholder_value(
    name: &str,
    task: &Value,
    seed: u64,
    adapter: Option<&str>,
    thinking: bool,
) -> Option<Value> {
    match name {
        "seed" => Some(json!(seed)),
        "adapter" | "adapter_name" => adapter.map_or(Some(Value::Null), |name| {
            Some(Value::String(name.to_string()))
        }),
        "adapter_label" => Some(Value::String(
            adapter.map_or_else(|| "base".to_string(), str::to_string),
        )),
        "thinking" | "thinking_enabled" => Some(Value::Bool(thinking)),
        _ => {
            if let Some(path) = name.strip_prefix("task.") {
                task_path_value(task, path).cloned()
            } else {
                task_path_value(task, name).cloned()
            }
        }
    }
}

fn task_path_value<'a>(task: &'a Value, path: &str) -> Option<&'a Value> {
    let mut current = task;
    for segment in path.split('.') {
        current = match current {
            Value::Object(map) => map.get(segment)?,
            _ => return None,
        };
    }
    Some(current)
}

fn placeholder_to_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

fn extract_messages_for_group(request: &Value) -> Result<Vec<RolloutChatMessage>> {
    let messages = request
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("rendered request must contain a messages array"))?;
    if messages.is_empty() {
        anyhow::bail!("rendered request messages array is empty");
    }

    messages
        .iter()
        .enumerate()
        .map(|(idx, message)| {
            let obj = message
                .as_object()
                .ok_or_else(|| anyhow!("messages[{idx}] must be an object"))?;
            let role = obj
                .get("role")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("messages[{idx}].role must be a string"))?
                .to_string();
            let content = message_content_to_string(obj.get("content"))
                .with_context(|| format!("messages[{idx}].content"))?;
            Ok(RolloutChatMessage { role, content })
        })
        .collect()
}

fn message_content_to_string(value: Option<&Value>) -> Result<String> {
    match value {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for part in parts {
                let Some(obj) = part.as_object() else {
                    continue;
                };
                if obj.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(text) = obj.get("text").and_then(Value::as_str)
                {
                    out.push_str(text);
                }
            }
            Ok(out)
        }
        Some(other) => anyhow::bail!("must be a string, null, or text-part array, got {other}"),
    }
}

async fn post_chat_completion(client: &reqwest::Client, url: &str, body: &Value) -> Result<Value> {
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .json(body)
        .send()
        .await?;
    let status = resp.status();
    let body: Value = resp.json().await?;
    if !status.is_success() {
        anyhow::bail!("{}", render_http_error(status, &body));
    }
    Ok(body)
}

fn extract_chat_content(response: &Value) -> String {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn extract_finish_reason(response: &Value) -> String {
    response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("finish_reason"))
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string()
}

fn extract_usage(response: &Value) -> TokenUsage {
    let usage = response.get("usage").unwrap_or(&Value::Null);
    TokenUsage {
        prompt_tokens: usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        completion_tokens: usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        total_tokens: usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
    }
}

fn run_scorer(path: &Path, input: &Value) -> Result<RewardScore> {
    let mut child = Command::new(path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("starting scorer {}", path.display()))?;

    {
        let stdin = child.stdin.as_mut().context("opening scorer stdin")?;
        serde_json::to_writer(&mut *stdin, input).context("writing scorer input")?;
        stdin.write_all(b"\n").context("finishing scorer input")?;
    }

    let output = child
        .wait_with_output()
        .with_context(|| format!("waiting for scorer {}", path.display()))?;
    if !output.status.success() {
        anyhow::bail!(
            "scorer {} exited with {}; stderr: {}",
            path.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let stdout = String::from_utf8(output.stdout).context("scorer stdout was not UTF-8")?;
    parse_reward_stdout(stdout.trim())
}

fn parse_reward_stdout(stdout: &str) -> Result<RewardScore> {
    if stdout.is_empty() {
        anyhow::bail!("scorer produced no stdout");
    }

    let raw = match serde_json::from_str::<Value>(stdout) {
        Ok(value) => value,
        Err(_) => {
            let reward: f64 = stdout
                .parse()
                .with_context(|| format!("scorer stdout was not JSON or a number: {stdout:?}"))?;
            ensure_finite_reward(reward)?;
            return Ok(RewardScore {
                reward,
                raw: json!(reward),
            });
        }
    };

    let reward = match &raw {
        Value::Number(number) => number
            .as_f64()
            .ok_or_else(|| anyhow!("scorer numeric output was not finite"))?,
        Value::Object(map) => first_number(map, &["reward", "score", "value"])
            .ok_or_else(|| anyhow!("scorer JSON must include numeric reward, score, or value"))?,
        _ => anyhow::bail!("scorer JSON output must be a number or object"),
    };
    ensure_finite_reward(reward)?;
    Ok(RewardScore { reward, raw })
}

fn first_number(map: &Map<String, Value>, names: &[&str]) -> Option<f64> {
    names.iter().find_map(|name| map.get(*name)?.as_f64())
}

fn ensure_finite_reward(reward: f64) -> Result<()> {
    if reward.is_finite() {
        Ok(())
    } else {
        anyhow::bail!("scorer reward must be finite")
    }
}

fn summarize_completions(completions: &[RolloutCompletionSummary]) -> RolloutGenerateStats {
    if completions.is_empty() {
        return RolloutGenerateStats::default();
    }

    let rewards: Vec<f64> = completions.iter().map(|item| item.reward).collect();
    let latencies: Vec<f64> = completions
        .iter()
        .map(|item| item.client_latency_ms)
        .collect();
    let server_latencies: Vec<f64> = completions
        .iter()
        .filter_map(|item| item.server_total_latency_ms)
        .collect();

    RolloutGenerateStats {
        mean_reward: mean(&rewards),
        min_reward: rewards.iter().copied().reduce(f64::min),
        max_reward: rewards.iter().copied().reduce(f64::max),
        total_prompt_tokens: completions.iter().map(|item| item.prompt_tokens).sum(),
        total_completion_tokens: completions.iter().map(|item| item.completion_tokens).sum(),
        total_tokens: completions.iter().map(|item| item.total_tokens).sum(),
        mean_client_latency_ms: mean(&latencies),
        max_client_latency_ms: latencies.into_iter().fold(0.0, f64::max),
        mean_server_latency_ms: (!server_latencies.is_empty()).then(|| mean(&server_latencies)),
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn render_http_error(status: StatusCode, body: &Value) -> String {
    if let Some(error) = body.get("error")
        && let Some(message) = error.get("message").and_then(Value::as_str)
    {
        return format!("{status}: {message}");
    }
    format!("{status}: {body}")
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    Ok(sha256_bytes(&bytes))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{}", hex_digest(hasher.finalize().as_slice()))
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::Router;
    use axum::extract::State;
    use axum::routing::post;
    use std::sync::{Arc, Mutex};

    #[test]
    fn render_rollout_request_forces_adapter_seed_thinking_and_performance() {
        let template = json!({
            "model": "Qwen3.5-4B",
            "messages": [{"role": "user", "content": "Question {{prompt}} seed {{seed}} {{adapter_label}} {{thinking_enabled}}"}],
            "chat_template_kwargs": {"enable_thinking": true, "custom": "kept"},
            "stream": true,
            "max_tokens": "{{max_tokens}}"
        });
        let task = json!({"prompt": "2+2?", "max_tokens": 3});

        let request = render_rollout_request(&template, &task, 7, Some("cap"), false).unwrap();
        assert_eq!(request["adapter"], "cap");
        assert_eq!(request["seed"], 7);
        assert_eq!(request["stream"], false);
        assert_eq!(request["include_performance"], true);
        assert_eq!(
            request["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(false)
        );
        assert_eq!(request["chat_template_kwargs"]["custom"], "kept");
        assert_eq!(request["max_tokens"], 3);
        assert_eq!(
            request["messages"][0]["content"],
            "Question 2+2? seed 7 cap false"
        );

        let base = render_rollout_request(&template, &task, 8, None, true).unwrap();
        assert!(base["adapter"].is_null());
        assert_eq!(
            base["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(true)
        );
    }

    #[test]
    fn scorer_reward_accepts_number_and_object() {
        assert_eq!(parse_reward_stdout("0.25").unwrap().reward, 0.25);
        assert_eq!(
            parse_reward_stdout(r#"{"reward":0.75,"note":"ok"}"#)
                .unwrap()
                .reward,
            0.75
        );
        assert!(parse_reward_stdout(r#"{"lift":1.0}"#).is_err());
    }

    #[tokio::test]
    async fn run_rollout_generate_writes_trainer_compatible_jsonl() {
        #[derive(Clone, Default)]
        struct Calls(Arc<Mutex<Vec<Value>>>);

        async fn chat(
            State(calls): State<Calls>,
            axum::Json(body): axum::Json<Value>,
        ) -> axum::Json<Value> {
            calls.0.lock().unwrap().push(body.clone());
            let adapter = body
                .get("adapter")
                .and_then(Value::as_str)
                .unwrap_or("base");
            let seed = body.get("seed").and_then(Value::as_u64).unwrap_or(0);
            axum::Json(json!({
                "id": format!("chatcmpl-{seed}"),
                "model": "Qwen3.5-4B",
                "choices": [{
                    "message": {"role": "assistant", "content": format!("{adapter} answer {seed}")},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
                "metadata": {
                    "thinking_enabled": false,
                    "thinking_mode": "non_reasoning",
                    "thinking_source": "request",
                    "final_content_empty": false,
                    "reasoning_folded_into_content": false,
                    "performance": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "ttft_ms": 1.0,
                        "total_latency_ms": 4.0,
                        "decode_tokens_per_sec": 500.0,
                        "adapter_used": adapter,
                        "thinking_mode": "non_reasoning",
                        "finish_reason": "stop"
                    }
                }
            }))
        }

        let calls = Calls::default();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .with_state(calls.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let tmp = tempfile::tempdir().unwrap();
        let tasks = tmp.path().join("tasks.jsonl");
        let template = tmp.path().join("request.json");
        let scorer = tmp.path().join("score_one.py");
        let output = tmp.path().join("rollouts.jsonl");
        let summary_output = tmp.path().join("rollout_summary.json");
        std::fs::write(&tasks, r#"{"id":"t1","prompt":"say hi"}"#).unwrap();
        std::fs::write(
            &template,
            serde_json::to_vec(&json!({
                "model": "Qwen3.5-4B",
                "messages": [{"role": "user", "content": "{{prompt}}"}],
                "max_tokens": 1
            }))
            .unwrap(),
        )
        .unwrap();
        std::fs::write(
            &scorer,
            "#!/usr/bin/env python3\nimport json,sys\np=json.load(sys.stdin)\nprint(json.dumps({'reward': 1.0 if p['content'].startswith('cap') else 0.0}))\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&scorer).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&scorer, perms).unwrap();
        }

        let summary = run_rollout_generate(RolloutGenerateOptions {
            url: format!("http://{addr}"),
            adapter: "cap".to_string(),
            thinking: false,
            tasks,
            seeds: 2,
            seed_start: 10,
            request_template: template,
            scorer,
            output: output.clone(),
            summary_output: summary_output.clone(),
        })
        .await
        .unwrap();

        assert_eq!(summary.group_count, 1);
        assert_eq!(summary.completion_count, 2);
        assert_eq!(summary.stats.total_tokens, 14);
        assert_eq!(summary.stats.total_completion_tokens, 4);
        assert_eq!(summary.stats.mean_reward, 1.0);
        assert!(summary_output.exists());

        let line = std::fs::read_to_string(&output).unwrap();
        let group_value: Value = serde_json::from_str(line.trim()).unwrap();
        let group: kiln_train::GrpoGroup = serde_json::from_value(group_value.clone()).unwrap();
        assert_eq!(group.messages.len(), 1);
        assert_eq!(group.messages[0].content, "say hi");
        assert_eq!(group.completions.len(), 2);
        assert_eq!(group.completions[0].reward, 1.0);
        assert_eq!(group.completions[0].trajectory[0].content, "cap answer 10");
        assert_eq!(
            group_value["completions"][0]["metadata"]["usage"]["total_tokens"],
            7
        );
        assert_eq!(
            group_value["completions"][0]["metadata"]["latency"]["server_total_ms"],
            4.0
        );

        let observed = calls.0.lock().unwrap().clone();
        assert_eq!(observed.len(), 2);
        assert_eq!(observed[0]["adapter"], "cap");
        assert_eq!(observed[0]["seed"], 10);
        assert_eq!(observed[1]["seed"], 11);
        assert_eq!(
            observed[0]["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(false)
        );
        assert_eq!(observed[0]["include_performance"], true);
        assert_eq!(observed[0]["stream"], false);

        server.abort();
    }
}
