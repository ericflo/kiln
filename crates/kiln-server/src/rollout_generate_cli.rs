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

use crate::cli::ThinkingBudgetArg;

#[derive(Debug, Clone)]
pub struct RolloutGenerateOptions {
    pub url: String,
    pub adapter: String,
    pub thinking: bool,
    pub thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    pub thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
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
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_requested_thinking_budget"
    )]
    pub thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_requested_thinking_budget"
    )]
    pub thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
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
    #[serde(default)]
    pub total_sampled_action_tokens: usize,
    #[serde(default)]
    pub total_forced_action_tokens: usize,
}

fn deserialize_requested_thinking_budget<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<ThinkingBudgetArg<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    let value = <Option<T> as serde::Deserialize>::deserialize(deserializer)?;
    Ok(Some(value.map_or(
        ThinkingBudgetArg::Unlimited,
        ThinkingBudgetArg::Limited,
    )))
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rollout_provenance_schema: Option<String>,
    #[serde(default)]
    pub sampled_action_tokens: usize,
    #[serde(default)]
    pub forced_action_tokens: usize,
}

#[derive(Debug)]
struct RolloutTask {
    index: usize,
    id: String,
    value: Value,
}

type RolloutChatMessage = kiln_train::ChatMessage;

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

#[derive(Debug, Clone, Copy, Serialize)]
struct TokenUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug)]
struct ValidatedRolloutResponse {
    content: String,
    finish_reason: String,
    usage: TokenUsage,
    provenance: kiln_train::RolloutProvenanceV1,
    sampled_action_tokens: usize,
    forced_action_tokens: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct RolloutGroupIdentity {
    prompt_token_ids: Vec<u32>,
    behavior_policy: kiln_train::RolloutBehaviorPolicyIdentityV1,
    tokenizer: kiln_train::RolloutTokenizerIdentityV1,
    template_invocation: kiln_train::RolloutChatTemplateInvocationV1,
    sampling: kiln_train::RolloutSamplingConfigV1,
    generation_backend: String,
}

impl RolloutGroupIdentity {
    fn from_provenance(provenance: &kiln_train::RolloutProvenanceV1) -> Self {
        Self {
            prompt_token_ids: provenance.input_token_ids[..provenance.prompt_token_count].to_vec(),
            behavior_policy: provenance.behavior_policy.clone(),
            tokenizer: provenance.tokenizer.clone(),
            template_invocation: provenance.template_invocation.clone(),
            sampling: provenance.sampling.clone(),
            generation_backend: provenance.generation_backend.clone(),
        }
    }
}

fn bind_rollout_group_identity(
    group_identity: &mut Option<RolloutGroupIdentity>,
    current: RolloutGroupIdentity,
    task_id: &str,
) -> Result<()> {
    match group_identity {
        Some(existing) => anyhow::ensure!(
            existing == &current,
            "rollout identity changed across seeds for task {task_id}; exact prompt tokens, behavior policy, tokenizer, template invocation, sampling controls, and generation backend must remain stable within one GRPO group"
        ),
        None => *group_identity = Some(current),
    }
    Ok(())
}

pub async fn run_rollout_generate(
    options: RolloutGenerateOptions,
) -> Result<RolloutGenerateSummary> {
    if options.seeds == 0 {
        anyhow::bail!("--seeds must be greater than zero");
    }
    if options.output == options.summary_output {
        anyhow::bail!("--output and --summary-output must name different files");
    }
    validate_thinking_budget_applicability(
        options.thinking,
        options.thinking_budget_tokens,
        options.thinking_budget_ms,
        None,
    )?;

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
    validate_thinking_budget_applicability(
        options.thinking,
        options.thinking_budget_tokens,
        options.thinking_budget_ms,
        Some(&template),
    )?;

    if tasks.is_empty() {
        anyhow::bail!("tasks JSONL contained no tasks");
    }

    let client = reqwest::Client::new();
    let mut staged_output = new_staged_file(&options.output)?;
    let mut writer = std::io::BufWriter::new(staged_output.as_file_mut());
    let mut summaries = Vec::with_capacity(tasks.len() * options.seeds);
    let mut warnings = Vec::new();

    for task in &tasks {
        let mut group_messages: Option<Vec<RolloutChatMessage>> = None;
        let mut group_rollout_identity: Option<RolloutGroupIdentity> = None;
        let mut completions = Vec::with_capacity(options.seeds);

        for seed_index in 0..options.seeds {
            let seed = options.seed_start.wrapping_add(seed_index as u64);
            let request = render_rollout_request(
                &template,
                &task.value,
                seed,
                adapter.request_value.as_deref(),
                options.thinking,
                options.thinking_budget_tokens,
                options.thinking_budget_ms,
            )?;
            let request_messages = extract_messages_for_group(&request)?;
            match &group_messages {
                Some(existing) if existing != &request_messages => {
                    anyhow::bail!(
                        "request template rendered different messages for task {} across seeds; GRPO groups require one shared prompt",
                        task.id
                    );
                }
                None => group_messages = Some(request_messages.clone()),
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
            let validated = validate_rollout_response(
                &response,
                seed,
                adapter.request_value.as_deref(),
                &request_messages,
            )
            .with_context(|| {
                format!(
                    "invalid rollout provenance for task {} seed {} adapter {}",
                    task.id, seed, adapter.label
                )
            })?;
            let ValidatedRolloutResponse {
                content,
                finish_reason,
                usage,
                provenance,
                sampled_action_tokens,
                forced_action_tokens,
            } = validated;
            let rollout_identity = RolloutGroupIdentity::from_provenance(&provenance);
            bind_rollout_group_identity(&mut group_rollout_identity, rollout_identity, &task.id)?;
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
                rollout_provenance_schema: Some(provenance.schema().to_string()),
                sampled_action_tokens,
                forced_action_tokens,
            };
            summaries.push(completion_summary.clone());

            completions.push(json!({
                "text": content,
                "reward": score.reward,
                "provenance": provenance,
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
    drop(writer);
    staged_output
        .as_file()
        .sync_all()
        .with_context(|| format!("syncing staged {}", options.output.display()))?;

    let stats = summarize_completions(&summaries);
    let summary = RolloutGenerateSummary {
        adapter: adapter.request_value,
        adapter_label: adapter.label,
        url: options.url,
        thinking_enabled: options.thinking,
        thinking_budget_tokens: options.thinking_budget_tokens,
        thinking_budget_ms: options.thinking_budget_ms,
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
    let staged_summary = stage_bytes(
        &options.summary_output,
        &serde_json::to_vec_pretty(&summary_value)?,
    )?;
    persist_staged_file(staged_output, &options.output)?;
    persist_staged_file(staged_summary, &options.summary_output)?;

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
    thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
) -> Result<Value> {
    validate_thinking_budget_applicability(
        thinking,
        thinking_budget_tokens,
        thinking_budget_ms,
        Some(template),
    )?;
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
    obj.insert("n".to_string(), json!(1));
    obj.insert("rollout_provenance".to_string(), Value::Bool(true));
    obj.insert("include_performance".to_string(), Value::Bool(true));
    if let Some(budget) = thinking_budget_tokens {
        obj.insert(
            "thinking_budget_tokens".to_string(),
            serde_json::to_value(budget)?,
        );
    }
    if let Some(budget) = thinking_budget_ms {
        obj.insert(
            "thinking_budget_ms".to_string(),
            serde_json::to_value(budget)?,
        );
    }
    let kwargs = obj
        .entry("chat_template_kwargs".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let kwargs = kwargs
        .as_object_mut()
        .ok_or_else(|| anyhow!("chat_template_kwargs must be a JSON object"))?;
    kwargs.insert("enable_thinking".to_string(), Value::Bool(thinking));
    Ok(value)
}

fn validate_thinking_budget_applicability(
    thinking: bool,
    thinking_budget_tokens: Option<ThinkingBudgetArg<usize>>,
    thinking_budget_ms: Option<ThinkingBudgetArg<u64>>,
    template: Option<&Value>,
) -> Result<()> {
    if thinking {
        return Ok(());
    }

    let mut inert_settings = Vec::new();
    if thinking_budget_tokens.is_some() {
        inert_settings.push("`--thinking-budget-tokens`");
    }
    if thinking_budget_ms.is_some() {
        inert_settings.push("`--thinking-budget-ms`");
    }
    if let Some(template) = template.and_then(Value::as_object) {
        if template.contains_key("thinking_budget_tokens") {
            inert_settings.push("request template field `thinking_budget_tokens`");
        }
        if template.contains_key("thinking_budget_ms") {
            inert_settings.push("request template field `thinking_budget_ms`");
        }
    }

    if inert_settings.is_empty() {
        return Ok(());
    }

    anyhow::bail!(
        "thinking budgets require `--thinking true`; thinking is disabled and would make {} inert. Enable thinking or remove the listed budget settings",
        inert_settings.join(", ")
    )
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
            serde_json::from_value::<RolloutChatMessage>(message.clone())
                .with_context(|| format!("messages[{idx}] must match the canonical chat schema"))
        })
        .collect()
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

fn validate_rollout_response(
    response: &Value,
    expected_seed: u64,
    expected_adapter: Option<&str>,
    expected_messages: &[RolloutChatMessage],
) -> Result<ValidatedRolloutResponse> {
    let choices = response
        .get("choices")
        .and_then(Value::as_array)
        .context("response.choices must be an array")?;
    anyhow::ensure!(
        choices.len() == 1,
        "response must contain exactly one choice, found {}",
        choices.len()
    );
    let choice = &choices[0];
    let content = choice
        .pointer("/message/content")
        .and_then(Value::as_str)
        .context("response choices[0].message.content must be a string")?
        .to_string();
    let finish_reason = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .context("response choices[0].finish_reason must be a string")?
        .to_string();
    let provenance_value = choice
        .get("rollout_provenance")
        .context("response choices[0].rollout_provenance is missing")?;
    let provenance: kiln_train::RolloutProvenanceV1 =
        serde_json::from_value(provenance_value.clone())
            .context("response choices[0].rollout_provenance is malformed")?;
    provenance
        .validate()
        .map_err(anyhow::Error::msg)
        .context("response rollout provenance is invalid")?;

    anyhow::ensure!(
        provenance.seed == expected_seed,
        "response rollout seed {} differs from requested seed {expected_seed}",
        provenance.seed
    );
    match (
        expected_adapter,
        provenance.behavior_policy.adapter.as_ref(),
    ) {
        (None, None) => {}
        (Some(expected), Some(actual)) if actual.name == expected => {}
        (None, Some(actual)) => anyhow::bail!(
            "response recorded adapter {:?} for an explicitly base-model rollout",
            actual.name
        ),
        (Some(expected), Some(actual)) => anyhow::bail!(
            "response recorded adapter {:?} instead of requested adapter {expected:?}",
            actual.name
        ),
        (Some(expected), None) => {
            anyhow::bail!("response omitted adapter identity for requested adapter {expected:?}")
        }
    }

    let prompt_messages_sha256 = kiln_train::rollout_prompt_messages_sha256(expected_messages)
        .map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        provenance.prompt_messages_sha256 == prompt_messages_sha256,
        "response rollout provenance belongs to different prompt messages"
    );
    let scored = kiln_train::ScoredRollout::legacy(content.clone(), 0.0);
    let scored_payload_sha256 =
        kiln_train::scored_rollout_payload_sha256(&scored).map_err(anyhow::Error::msg)?;
    anyhow::ensure!(
        provenance.scored_payload_sha256 == scored_payload_sha256,
        "response rollout provenance belongs to different completion content"
    );

    let generated_tokens = provenance
        .input_token_ids
        .len()
        .checked_sub(provenance.prompt_token_count)
        .context("response rollout prompt boundary exceeds its token sequence")?;
    anyhow::ensure!(
        provenance.action_tokens.len() == generated_tokens,
        "response rollout provenance enumerates {} actions for {generated_tokens} generated tokens",
        provenance.action_tokens.len()
    );
    for (offset, action) in provenance.action_tokens.iter().enumerate() {
        anyhow::ensure!(
            action.sequence_index == provenance.prompt_token_count + offset,
            "response rollout action {} is at sequence index {}, expected {}",
            offset,
            action.sequence_index,
            provenance.prompt_token_count + offset
        );
    }

    let usage = required_token_usage(response)?;
    anyhow::ensure!(
        usage.prompt_tokens == provenance.prompt_token_count,
        "response usage reports {} prompt tokens but provenance records {}",
        usage.prompt_tokens,
        provenance.prompt_token_count
    );
    anyhow::ensure!(
        usage.completion_tokens == provenance.action_tokens.len(),
        "response usage reports {} completion tokens but provenance records {} actions",
        usage.completion_tokens,
        provenance.action_tokens.len()
    );
    anyhow::ensure!(
        usage.total_tokens == usage.prompt_tokens + usage.completion_tokens,
        "response usage total {} differs from prompt {} + completion {}",
        usage.total_tokens,
        usage.prompt_tokens,
        usage.completion_tokens
    );

    let sampled_action_tokens = provenance.sampled_action_tokens().count();
    let forced_action_tokens = provenance.action_tokens.len() - sampled_action_tokens;
    Ok(ValidatedRolloutResponse {
        content,
        finish_reason,
        usage,
        provenance,
        sampled_action_tokens,
        forced_action_tokens,
    })
}

fn required_token_usage(response: &Value) -> Result<TokenUsage> {
    let usage = response
        .get("usage")
        .and_then(Value::as_object)
        .context("response.usage must be an object")?;
    let required = |name: &str| -> Result<usize> {
        let value = usage
            .get(name)
            .and_then(Value::as_u64)
            .with_context(|| format!("response.usage.{name} must be a non-negative integer"))?;
        usize::try_from(value)
            .with_context(|| format!("response.usage.{name} does not fit this platform"))
    };
    Ok(TokenUsage {
        prompt_tokens: required("prompt_tokens")?,
        completion_tokens: required("completion_tokens")?,
        total_tokens: required("total_tokens")?,
    })
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
        total_sampled_action_tokens: completions
            .iter()
            .map(|item| item.sampled_action_tokens)
            .sum(),
        total_forced_action_tokens: completions
            .iter()
            .map(|item| item.forced_action_tokens)
            .sum(),
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

fn destination_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn new_staged_file(destination: &Path) -> Result<tempfile::NamedTempFile> {
    tempfile::Builder::new()
        .prefix(".kiln-rollout-")
        .tempfile_in(destination_parent(destination))
        .with_context(|| format!("creating staged {}", destination.display()))
}

fn stage_bytes(destination: &Path, bytes: &[u8]) -> Result<tempfile::NamedTempFile> {
    let mut staged = new_staged_file(destination)?;
    staged
        .write_all(bytes)
        .with_context(|| format!("writing staged {}", destination.display()))?;
    staged
        .flush()
        .with_context(|| format!("flushing staged {}", destination.display()))?;
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("syncing staged {}", destination.display()))?;
    Ok(staged)
}

fn persist_staged_file(staged: tempfile::NamedTempFile, destination: &Path) -> Result<()> {
    staged.persist(destination).map_err(|error| {
        anyhow!(
            "publishing {} atomically: {}",
            destination.display(),
            error.error
        )
    })?;
    Ok(())
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

    const PROVENANCE_TEST_TEMPLATE: &str = "{% for message in messages %}{{ message.role }}:{{ message.content }}\n{% endfor %}assistant:";

    fn provenance_test_tokenizer() -> kiln_core::tokenizer::KilnTokenizer {
        let mut vocab = serde_json::Map::new();
        for (id, token) in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 :\n.-_|<>/"
                .chars()
                .enumerate()
        {
            vocab.insert(token.to_string(), json!(id));
        }
        let bytes = serde_json::to_vec(&json!({
            "version": "1.0",
            "model": {
                "type": "BPE",
                "vocab": vocab,
                "merges": []
            }
        }))
        .unwrap();
        kiln_core::tokenizer::KilnTokenizer::from_bytes(&bytes)
            .unwrap()
            .with_chat_template(PROVENANCE_TEST_TEMPLATE.to_string())
    }

    fn identity_hash(fill: char) -> String {
        format!("sha256:{}", fill.to_string().repeat(64))
    }

    fn provenance_for_test_response(
        request: &Value,
        content: &str,
    ) -> (kiln_train::RolloutProvenanceV1, usize) {
        let tokenizer = provenance_test_tokenizer();
        let messages = extract_messages_for_group(request).unwrap();
        let training_messages = messages;
        let core_messages = training_messages.clone();
        let template_kwargs = request
            .get("chat_template_kwargs")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let prompt = tokenizer
            .apply_chat_template_full_with_options(
                &core_messages,
                request
                    .get("tools")
                    .and_then(Value::as_array)
                    .map(Vec::as_slice),
                request.get("tool_choice"),
                kiln_core::tokenizer::ChatTemplateOptions {
                    template_kwargs: template_kwargs.clone(),
                },
            )
            .unwrap();
        let mut input_token_ids = tokenizer.encode(&prompt).unwrap();
        let prompt_token_count = input_token_ids.len();
        input_token_ids.push(0);
        let adapter = request.get("adapter").and_then(Value::as_str).map(|name| {
            kiln_train::RolloutAdapterIdentityV1 {
                name: name.to_string(),
                content_sha256: identity_hash('b'),
            }
        });
        let scored = kiln_train::ScoredRollout::legacy(content.to_string(), 0.0);
        let provenance = kiln_train::RolloutProvenanceV1::new(
            input_token_ids,
            prompt_token_count,
            kiln_train::rollout_prompt_messages_sha256(&training_messages).unwrap(),
            kiln_train::scored_rollout_payload_sha256(&scored).unwrap(),
            vec![kiln_train::RolloutActionTokenV1::sampled(
                prompt_token_count,
                0,
                -0.25,
            )],
            kiln_train::RolloutBehaviorPolicyIdentityV1 {
                served_model_id: "Qwen3.5-4B".to_string(),
                base_model_sha256: identity_hash('a'),
                adapter,
                inference_config_sha256: identity_hash('c'),
                implementation: "kiln-test".to_string(),
            },
            kiln_train::RolloutTokenizerIdentityV1 {
                vocab_sha256: tokenizer.vocab_identity_sha256(),
                config_sha256: tokenizer.tokenizer_config_sha256().unwrap(),
                chat_template_sha256: tokenizer.chat_template_sha256().unwrap(),
            },
            kiln_train::RolloutSamplingConfigV1 {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                max_tokens: 1,
                repetition_penalty: 1.0,
                presence_penalty: 1.5,
                frequency_penalty: 0.0,
                stop: Vec::new(),
                thinking_budget: None,
            },
            request.get("seed").and_then(Value::as_u64).unwrap(),
            "cpu",
        )
        .unwrap()
        .with_template_invocation(kiln_train::RolloutChatTemplateInvocationV1 {
            tools: request
                .get("tools")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default(),
            tool_choice: request.get("tool_choice").cloned(),
            template_kwargs,
        })
        .unwrap();
        (provenance, prompt_token_count)
    }

    #[test]
    fn render_rollout_request_forces_adapter_seed_thinking_and_performance() {
        let template = json!({
            "model": "Qwen3.5-4B",
            "messages": [{"role": "user", "content": "Question {{prompt}} seed {{seed}} {{adapter_label}} {{thinking_enabled}}"}],
            "chat_template_kwargs": {"enable_thinking": true, "custom": "kept"},
            "thinking_budget_tokens": 64,
            "thinking_budget_ms": 750,
            "stream": true,
            "max_tokens": "{{max_tokens}}"
        });
        let task = json!({"prompt": "2+2?", "max_tokens": 3});

        let request = render_rollout_request(
            &template,
            &task,
            7,
            Some("cap"),
            true,
            Some(ThinkingBudgetArg::Limited(96)),
            Some(ThinkingBudgetArg::Limited(1500)),
        )
        .unwrap();
        assert_eq!(request["adapter"], "cap");
        assert_eq!(request["seed"], 7);
        assert_eq!(request["stream"], false);
        assert_eq!(request["n"], 1);
        assert_eq!(request["rollout_provenance"], true);
        assert_eq!(request["include_performance"], true);
        assert_eq!(
            request["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(true)
        );
        assert_eq!(request["chat_template_kwargs"]["custom"], "kept");
        assert_eq!(request["max_tokens"], 3);
        assert_eq!(request["thinking_budget_tokens"], 96);
        assert_eq!(request["thinking_budget_ms"], 1500);
        assert_eq!(
            request["messages"][0]["content"],
            "Question 2+2? seed 7 cap true"
        );

        let base = render_rollout_request(&template, &task, 8, None, true, None, None).unwrap();
        assert!(base["adapter"].is_null());
        assert_eq!(
            base["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(true)
        );
        assert_eq!(base["thinking_budget_tokens"], 64);
        assert_eq!(base["thinking_budget_ms"], 750);

        let unlimited = render_rollout_request(
            &template,
            &task,
            9,
            None,
            true,
            Some(ThinkingBudgetArg::Unlimited),
            Some(ThinkingBudgetArg::Unlimited),
        )
        .unwrap();
        assert!(unlimited["thinking_budget_tokens"].is_null());
        assert!(unlimited["thinking_budget_ms"].is_null());
    }

    fn contract_budget_arg<T>(case: &Value, dimension: &str) -> Option<ThinkingBudgetArg<T>>
    where
        T: serde::de::DeserializeOwned,
    {
        let setting = &case[dimension];
        match setting["state"].as_str().unwrap() {
            "inherit" => None,
            "unlimited" => Some(ThinkingBudgetArg::Unlimited),
            "limit" => Some(ThinkingBudgetArg::Limited(
                serde_json::from_value(setting["value"].clone()).unwrap(),
            )),
            state => panic!("unknown thinking-budget contract state {state:?}"),
        }
    }

    #[test]
    fn rollout_request_runs_request_scope_budget_contract_matrix() {
        let contract: Value = serde_json::from_str(include_str!(
            "../../../contracts/thinking-budget-v1.conformance.json"
        ))
        .unwrap();
        assert_eq!(contract["contract_version"], 1);
        let template = json!({
            "model": "Qwen3.5-4B",
            "messages": [{"role": "user", "content": "{{prompt}}"}]
        });
        let task = json!({"prompt": "contract check"});

        for case in contract["resolution_cases"].as_array().unwrap() {
            if case["scope"] != "request" {
                continue;
            }
            let name = case["name"].as_str().unwrap();
            let tokens = contract_budget_arg::<usize>(case, "tokens");
            let time_ms = contract_budget_arg::<u64>(case, "time");
            let rendered =
                render_rollout_request(&template, &task, 1, None, true, tokens, time_ms).unwrap();

            for field in ["thinking_budget_tokens", "thinking_budget_ms"] {
                assert_eq!(
                    rendered.get(field),
                    case["request"].get(field),
                    "{name} {field} wire shape"
                );
            }

            let disabled =
                render_rollout_request(&template, &task, 1, None, false, tokens, time_ms);
            if tokens.is_some() || time_ms.is_some() {
                let error = disabled.unwrap_err().to_string();
                assert!(
                    error.contains("thinking budgets require `--thinking true`"),
                    "{name}: {error}"
                );
            } else {
                let disabled = disabled.unwrap();
                assert_eq!(
                    disabled["chat_template_kwargs"]["enable_thinking"], false,
                    "{name}"
                );
            }
        }
    }

    #[test]
    fn disabled_thinking_rejects_template_budget_fields() {
        let template = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_budget_tokens": 64,
            "thinking_budget_ms": null
        });
        let error = render_rollout_request(&template, &json!({}), 0, None, false, None, None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("request template field `thinking_budget_tokens`"));
        assert!(error.contains("request template field `thinking_budget_ms`"));
        assert!(error.contains("Enable thinking or remove the listed budget settings"));
    }

    #[tokio::test]
    async fn rollout_rejects_disabled_budget_flags_before_file_io() {
        let tmp = tempfile::tempdir().unwrap();
        let output = tmp.path().join("must-not-exist.jsonl");
        let summary_output = tmp.path().join("must-not-exist.summary.json");
        let error = run_rollout_generate(RolloutGenerateOptions {
            url: "http://127.0.0.1:1".to_string(),
            adapter: "base".to_string(),
            thinking: false,
            thinking_budget_tokens: Some(ThinkingBudgetArg::Limited(0)),
            thinking_budget_ms: None,
            tasks: tmp.path().join("missing.tasks.jsonl"),
            seeds: 1,
            seed_start: 0,
            request_template: tmp.path().join("missing.request.json"),
            scorer: tmp.path().join("missing.scorer"),
            output: output.clone(),
            summary_output: summary_output.clone(),
        })
        .await
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("`--thinking-budget-tokens`"));
        assert!(!error.contains("reading tasks JSONL"));
        assert!(!output.exists());
        assert!(!summary_output.exists());
    }

    #[tokio::test]
    async fn rollout_rejects_disabled_template_budget_before_output_or_network() {
        let tmp = tempfile::tempdir().unwrap();
        let tasks = tmp.path().join("tasks.jsonl");
        let template = tmp.path().join("request.json");
        let output = tmp.path().join("must-not-exist.jsonl");
        let summary_output = tmp.path().join("must-not-exist.summary.json");
        std::fs::write(&tasks, r#"{"prompt":"hello"}"#).unwrap();
        std::fs::write(
            &template,
            r#"{"messages":[{"role":"user","content":"{{prompt}}"}],"thinking_budget_ms":null}"#,
        )
        .unwrap();

        let error = run_rollout_generate(RolloutGenerateOptions {
            url: "http://127.0.0.1:1".to_string(),
            adapter: "base".to_string(),
            thinking: false,
            thinking_budget_tokens: None,
            thinking_budget_ms: None,
            tasks,
            seeds: 1,
            seed_start: 0,
            request_template: template,
            scorer: tmp.path().join("missing.scorer"),
            output: output.clone(),
            summary_output: summary_output.clone(),
        })
        .await
        .unwrap_err()
        .to_string();

        assert!(error.contains("request template field `thinking_budget_ms`"));
        assert!(!output.exists());
        assert!(!summary_output.exists());
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

    #[test]
    fn rollout_response_validation_binds_seed_prompt_content_adapter_and_usage() {
        let request = render_rollout_request(
            &json!({
                "messages": [
                    {"role": "user", "content": "use the prior result"},
                    {
                        "role": "assistant",
                        "content": null,
                        "name": "lookup",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"}
                        }]
                    },
                    {
                        "role": "tool",
                        "content": [{"type": "text", "text": "ready"}],
                        "name": "lookup",
                        "tool_call_id": "call_1"
                    },
                    {"role": "user", "content": "say hi"}
                ],
                "max_tokens": 1
            }),
            &json!({}),
            10,
            Some("cap"),
            false,
            None,
            None,
        )
        .unwrap();
        let messages = extract_messages_for_group(&request).unwrap();
        assert_eq!(messages[1].content, "");
        assert_eq!(messages[1].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(messages[2].content, "ready");
        assert_eq!(messages[2].name.as_deref(), Some("lookup"));
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
        let content = "cap answer 10";
        let (provenance, prompt_tokens) = provenance_for_test_response(&request, content);
        let mut response = json!({
            "choices": [{
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "rollout_provenance": provenance
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1
            }
        });
        validate_rollout_response(&response, 10, Some("cap"), &messages).unwrap();

        let mut wrong_messages = messages.clone();
        wrong_messages[2].tool_call_id = Some("call_other".into());
        let error = validate_rollout_response(&response, 10, Some("cap"), &wrong_messages)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different prompt messages"), "{error}");

        response["choices"][0]["rollout_provenance"]["seed"] = json!(11);
        let error = validate_rollout_response(&response, 10, Some("cap"), &messages)
            .unwrap_err()
            .to_string();
        assert!(error.contains("differs from requested seed"), "{error}");

        response["choices"][0]["rollout_provenance"]["seed"] = json!(10);
        response["choices"][0]["message"]["content"] = json!("different output");
        let error = validate_rollout_response(&response, 10, Some("cap"), &messages)
            .unwrap_err()
            .to_string();
        assert!(error.contains("different completion content"), "{error}");

        response["choices"][0]
            .as_object_mut()
            .unwrap()
            .remove("rollout_provenance");
        let error = validate_rollout_response(&response, 10, Some("cap"), &messages)
            .unwrap_err()
            .to_string();
        assert!(error.contains("rollout_provenance is missing"), "{error}");
    }

    #[test]
    fn rollout_group_identity_rejects_same_name_adapter_revision_drift() {
        let request = render_rollout_request(
            &json!({
                "messages": [{"role": "user", "content": "say hi"}],
                "max_tokens": 1
            }),
            &json!({}),
            10,
            Some("cap"),
            false,
            None,
            None,
        )
        .unwrap();
        let (provenance, _) = provenance_for_test_response(&request, "cap answer 10");
        let identity = RolloutGroupIdentity::from_provenance(&provenance);
        let mut group = None;
        bind_rollout_group_identity(&mut group, identity.clone(), "t1").unwrap();
        bind_rollout_group_identity(&mut group, identity.clone(), "t1").unwrap();

        let mut drifted = identity;
        drifted
            .behavior_policy
            .adapter
            .as_mut()
            .unwrap()
            .content_sha256 = identity_hash('d');
        let error = bind_rollout_group_identity(&mut group, drifted, "t1")
            .unwrap_err()
            .to_string();
        assert!(error.contains("identity changed across seeds"), "{error}");
    }

    #[tokio::test]
    async fn missing_rollout_provenance_never_scores_or_publishes_partial_output() {
        async fn chat(axum::Json(body): axum::Json<Value>) -> axum::Json<Value> {
            assert_eq!(body["rollout_provenance"], true);
            axum::Json(json!({
                "id": "chatcmpl-missing-provenance",
                "choices": [{
                    "message": {"role": "assistant", "content": "unbound"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }))
        }

        let app = Router::new().route("/v1/chat/completions", post(chat));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });

        let tmp = tempfile::tempdir().unwrap();
        let tasks = tmp.path().join("tasks.jsonl");
        let template = tmp.path().join("request.json");
        let output = tmp.path().join("rollouts.jsonl");
        let summary_output = tmp.path().join("summary.json");
        std::fs::write(&tasks, r#"{"prompt":"say hi"}"#).unwrap();
        std::fs::write(
            &template,
            r#"{"messages":[{"role":"user","content":"{{prompt}}"}],"max_tokens":1}"#,
        )
        .unwrap();
        std::fs::write(&output, "previous complete dataset\n").unwrap();
        std::fs::write(&summary_output, "previous summary\n").unwrap();

        let error = run_rollout_generate(RolloutGenerateOptions {
            url: format!("http://{addr}"),
            adapter: "base".to_string(),
            thinking: false,
            thinking_budget_tokens: None,
            thinking_budget_ms: None,
            tasks,
            seeds: 1,
            seed_start: 0,
            request_template: template,
            scorer: tmp.path().join("must-not-run.scorer"),
            output: output.clone(),
            summary_output: summary_output.clone(),
        })
        .await
        .unwrap_err();
        let error = format!("{error:#}");

        assert!(error.contains("rollout_provenance is missing"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&output).unwrap(),
            "previous complete dataset\n"
        );
        assert_eq!(
            std::fs::read_to_string(&summary_output).unwrap(),
            "previous summary\n"
        );
        assert!(std::fs::read_dir(tmp.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".kiln-rollout-")
        }));
        server.abort();
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
            let content = format!("{adapter} answer {seed}");
            let (provenance, prompt_tokens) = provenance_for_test_response(&body, &content);
            axum::Json(json!({
                "id": format!("chatcmpl-{seed}"),
                "model": "Qwen3.5-4B",
                "choices": [{
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                    "rollout_provenance": provenance
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 1,
                    "total_tokens": prompt_tokens + 1
                },
                "metadata": {
                    "thinking_enabled": true,
                    "thinking_mode": "reasoning",
                    "thinking_source": "request",
                    "final_content_empty": false,
                    "reasoning_folded_into_content": false,
                    "performance": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 1,
                        "ttft_ms": 1.0,
                        "total_latency_ms": 4.0,
                        "decode_tokens_per_sec": 500.0,
                        "adapter_used": adapter,
                        "thinking_mode": "reasoning",
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
            thinking: true,
            thinking_budget_tokens: Some(ThinkingBudgetArg::Limited(96)),
            thinking_budget_ms: Some(ThinkingBudgetArg::Limited(1500)),
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
        assert_eq!(summary.stats.total_completion_tokens, 2);
        assert_eq!(summary.stats.total_sampled_action_tokens, 2);
        assert_eq!(summary.stats.total_forced_action_tokens, 0);
        assert_eq!(summary.stats.mean_reward, 1.0);
        assert_eq!(
            summary.thinking_budget_tokens,
            Some(ThinkingBudgetArg::Limited(96))
        );
        assert_eq!(
            summary.thinking_budget_ms,
            Some(ThinkingBudgetArg::Limited(1500))
        );
        assert!(summary_output.exists());

        let summary_json: Value =
            serde_json::from_slice(&std::fs::read(&summary_output).unwrap()).unwrap();
        assert_eq!(summary_json["thinking_budget_tokens"], 96);
        assert_eq!(summary_json["thinking_budget_ms"], 1500);
        let decoded: RolloutGenerateSummary = serde_json::from_value(summary_json.clone()).unwrap();
        assert_eq!(
            decoded.thinking_budget_tokens,
            Some(ThinkingBudgetArg::Limited(96))
        );

        let mut unlimited_summary = summary_json.clone();
        unlimited_summary["thinking_budget_tokens"] = Value::Null;
        let decoded: RolloutGenerateSummary = serde_json::from_value(unlimited_summary).unwrap();
        assert_eq!(
            decoded.thinking_budget_tokens,
            Some(ThinkingBudgetArg::Unlimited)
        );

        let mut inherited_summary = summary_json.clone();
        inherited_summary
            .as_object_mut()
            .unwrap()
            .remove("thinking_budget_tokens");
        let decoded: RolloutGenerateSummary = serde_json::from_value(inherited_summary).unwrap();
        assert_eq!(decoded.thinking_budget_tokens, None);

        let line = std::fs::read_to_string(&output).unwrap();
        let group_value: Value = serde_json::from_str(line.trim()).unwrap();
        let group: kiln_train::GrpoGroup = serde_json::from_value(group_value.clone()).unwrap();
        assert_eq!(group.messages.len(), 1);
        assert_eq!(group.messages[0].content, "say hi");
        assert_eq!(group.completions.len(), 2);
        assert_eq!(group.completions[0].reward, 1.0);
        assert_eq!(group.completions[0].text, "cap answer 10");
        assert!(group.completions[0].trajectory.is_empty());
        assert!(group.completions[0].provenance.is_some());
        let mut recorded = kiln_train::GrpoConfig::default();
        recorded.behavior_policy = kiln_train::BehaviorPolicy::Recorded;
        kiln_train::trainer::validate_grpo_group_policy_data(
            &group,
            &recorded,
            &provenance_test_tokenizer(),
        )
        .unwrap();
        let expected_total_tokens = group.completions[0]
            .provenance
            .as_ref()
            .unwrap()
            .prompt_token_count
            + 1;
        assert_eq!(
            group_value["completions"][0]["metadata"]["usage"]["total_tokens"],
            expected_total_tokens
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
        assert_eq!(observed[0]["thinking_budget_tokens"], 96);
        assert_eq!(observed[0]["thinking_budget_ms"], 1500);
        assert_eq!(
            observed[0]["chat_template_kwargs"]["enable_thinking"],
            Value::Bool(true)
        );
        assert_eq!(observed[0]["include_performance"], true);
        assert_eq!(observed[0]["stream"], false);
        assert_eq!(observed[0]["n"], 1);
        assert_eq!(observed[0]["rollout_provenance"], true);

        server.abort();
    }
}
