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
pub struct EvalAdapterOptions {
    pub url: String,
    pub adapter: String,
    pub tasks: PathBuf,
    pub seeds: usize,
    pub request_template: PathBuf,
    pub scorer: PathBuf,
    pub output: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalAdapterSummary {
    pub adapter: String,
    pub url: String,
    pub task_count: usize,
    pub seeds: usize,
    pub pair_count: usize,
    pub tasks_path: String,
    pub request_template_path: String,
    pub scorer_path: String,
    pub output_path: String,
    pub tasks_sha256: String,
    pub request_template_sha256: String,
    pub scorer_sha256: Option<String>,
    pub adapter_hashes: Vec<AdapterHashRecord>,
    pub config_hashes: Option<Value>,
    pub warnings: Vec<String>,
    pub stats: EvalAdapterStats,
    pub results: Vec<EvalAdapterPairResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AdapterHashRecord {
    pub name: String,
    pub path: Option<String>,
    pub adapter_model_sha256: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalAdapterStats {
    pub mean_lift: f64,
    pub stdev_lift: f64,
    pub zero_count: usize,
    pub base_mean: Option<f64>,
    pub adapter_mean: Option<f64>,
    pub wall_clock_ms: f64,
    pub mean_pair_wall_clock_ms: f64,
    pub max_pair_wall_clock_ms: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EvalAdapterPairResult {
    pub task_index: usize,
    pub task_id: String,
    pub seed: u64,
    pub base_score: Option<f64>,
    pub adapter_score: Option<f64>,
    pub lift: f64,
    pub zero_lift: bool,
    pub wall_clock_ms: f64,
    pub base_content: String,
    pub adapter_content: String,
    pub scorer_output: Value,
}

#[derive(Debug)]
struct EvalTask {
    index: usize,
    id: String,
    value: Value,
}

#[derive(Debug)]
struct PairScore {
    base_score: Option<f64>,
    adapter_score: Option<f64>,
    lift: f64,
    raw: Value,
}

#[derive(Debug, Serialize)]
struct ScorerInput<'a> {
    task_index: usize,
    task_id: &'a str,
    task: &'a Value,
    seed: u64,
    adapter: &'a str,
    base: ScorerCompletion<'a>,
    candidate: ScorerCompletion<'a>,
}

#[derive(Debug, Serialize)]
struct ScorerCompletion<'a> {
    adapter: Option<&'a str>,
    content: &'a str,
    response: &'a Value,
}

#[derive(Debug)]
struct ModelStateSnapshot {
    config_hashes: Option<Value>,
    adapter_hashes: Vec<AdapterHashRecord>,
    warning: Option<String>,
}

pub async fn run_eval_adapter(options: EvalAdapterOptions) -> Result<EvalAdapterSummary> {
    if options.seeds == 0 {
        anyhow::bail!("--seeds must be greater than zero");
    }

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
    let template: Value = serde_json::from_slice(&template_bytes)
        .with_context(|| format!("parsing request template {}", options.request_template.display()))?;

    if tasks.is_empty() {
        anyhow::bail!("tasks JSONL contained no tasks");
    }

    let client = reqwest::Client::new();
    let mut results = Vec::with_capacity(tasks.len() * options.seeds);
    let mut warnings = Vec::new();

    for task in &tasks {
        for seed_index in 0..options.seeds {
            let seed = seed_index as u64;
            let pair_started = Instant::now();

            let base_request = render_chat_request(&template, &task.value, seed, None)?;
            let adapter_request =
                render_chat_request(&template, &task.value, seed, Some(&options.adapter))?;

            let base_response = post_chat_completion(&client, &options.url, &base_request)
                .await
                .with_context(|| {
                    format!(
                        "base completion failed for task {} seed {}",
                        task.id, seed
                    )
                })?;
            let adapter_response = post_chat_completion(&client, &options.url, &adapter_request)
                .await
                .with_context(|| {
                    format!(
                        "adapter completion failed for task {} seed {}",
                        task.id, seed
                    )
                })?;

            let base_content = extract_chat_content(&base_response);
            let adapter_content = extract_chat_content(&adapter_response);
            let score = run_scorer(
                &options.scorer,
                &ScorerInput {
                    task_index: task.index,
                    task_id: &task.id,
                    task: &task.value,
                    seed,
                    adapter: &options.adapter,
                    base: ScorerCompletion {
                        adapter: None,
                        content: &base_content,
                        response: &base_response,
                    },
                    candidate: ScorerCompletion {
                        adapter: Some(&options.adapter),
                        content: &adapter_content,
                        response: &adapter_response,
                    },
                },
            )
            .with_context(|| format!("scoring task {} seed {}", task.id, seed))?;

            let lift = score.lift;
            results.push(EvalAdapterPairResult {
                task_index: task.index,
                task_id: task.id.clone(),
                seed,
                base_score: score.base_score,
                adapter_score: score.adapter_score,
                lift,
                zero_lift: lift.abs() <= f64::EPSILON,
                wall_clock_ms: pair_started.elapsed().as_secs_f64() * 1000.0,
                base_content,
                adapter_content,
                scorer_output: score.raw,
            });
        }
    }

    let mut model_state = fetch_model_state(&client, &options.url, &options.adapter).await;
    if let Some(warning) = model_state.warning.take() {
        warnings.push(warning);
    }

    let stats = summarize_results(&results, started.elapsed().as_secs_f64() * 1000.0);
    if stats.stdev_lift >= stats.mean_lift.abs() && stats.stdev_lift > 0.0 {
        warnings.push(format!(
            "lift stdev ({:.6}) is comparable to or larger than mean lift ({:.6}); run more seeds/tasks before overclaiming",
            stats.stdev_lift, stats.mean_lift
        ));
    }

    let summary = EvalAdapterSummary {
        adapter: options.adapter,
        url: options.url,
        task_count: tasks.len(),
        seeds: options.seeds,
        pair_count: results.len(),
        tasks_path: options.tasks.display().to_string(),
        request_template_path: options.request_template.display().to_string(),
        scorer_path: options.scorer.display().to_string(),
        output_path: options.output.display().to_string(),
        tasks_sha256: sha256_bytes(&tasks_bytes),
        request_template_sha256: sha256_bytes(&template_bytes),
        scorer_sha256,
        adapter_hashes: model_state.adapter_hashes,
        config_hashes: model_state.config_hashes,
        warnings,
        stats,
        results,
    };

    let output_bytes = serde_json::to_vec_pretty(&summary)?;
    std::fs::write(&options.output, output_bytes)
        .with_context(|| format!("writing {}", options.output.display()))?;

    Ok(summary)
}

fn parse_tasks_jsonl(bytes: &[u8]) -> Result<Vec<EvalTask>> {
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
        tasks.push(EvalTask {
            index: tasks.len(),
            id,
            value,
        });
    }
    Ok(tasks)
}

fn render_chat_request(
    template: &Value,
    task: &Value,
    seed: u64,
    adapter: Option<&str>,
) -> Result<Value> {
    let mut value = render_template_value(template, task, seed, adapter)?;
    let obj = value
        .as_object_mut()
        .ok_or_else(|| anyhow!("request template must render to a JSON object"))?;
    obj.insert("seed".to_string(), json!(seed));
    obj.insert(
        "adapter".to_string(),
        adapter.map_or(Value::Null, |name| Value::String(name.to_string())),
    );
    obj.entry("include_performance".to_string())
        .or_insert(Value::Bool(true));
    obj.entry("include_config_hashes".to_string())
        .or_insert(Value::Bool(true));
    Ok(value)
}

fn render_template_value(
    value: &Value,
    task: &Value,
    seed: u64,
    adapter: Option<&str>,
) -> Result<Value> {
    match value {
        Value::String(s) => render_template_string(s, task, seed, adapter),
        Value::Array(items) => items
            .iter()
            .map(|item| render_template_value(item, task, seed, adapter))
            .collect(),
        Value::Object(map) => {
            let mut rendered = Map::new();
            for (key, item) in map {
                rendered.insert(
                    key.clone(),
                    render_template_value(item, task, seed, adapter)?,
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
) -> Result<Value> {
    let re = Regex::new(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}").expect("valid placeholder regex");
    if let Some(caps) = re.captures(template)
        && caps.get(0).is_some_and(|m| m.as_str() == template)
    {
        let name = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        if let Some(value) = placeholder_value(name, task, seed, adapter) {
            return Ok(value);
        }
    }

    let mut rendered = String::with_capacity(template.len());
    let mut cursor = 0;
    for caps in re.captures_iter(template) {
        let full = caps.get(0).unwrap();
        rendered.push_str(&template[cursor..full.start()]);
        let name = caps.get(1).map(|m| m.as_str()).unwrap_or_default();
        if let Some(value) = placeholder_value(name, task, seed, adapter) {
            rendered.push_str(&placeholder_to_string(&value));
        } else {
            rendered.push_str(full.as_str());
        }
        cursor = full.end();
    }
    rendered.push_str(&template[cursor..]);
    Ok(Value::String(rendered))
}

fn placeholder_value(name: &str, task: &Value, seed: u64, adapter: Option<&str>) -> Option<Value> {
    match name {
        "seed" => Some(json!(seed)),
        "adapter" | "adapter_name" => adapter.map_or(Some(Value::Null), |name| {
            Some(Value::String(name.to_string()))
        }),
        "adapter_label" => Some(Value::String(
            adapter.map_or_else(|| "base".to_string(), str::to_string),
        )),
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

fn run_scorer(path: &Path, input: &ScorerInput<'_>) -> Result<PairScore> {
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
    parse_scorer_stdout(stdout.trim())
}

fn parse_scorer_stdout(stdout: &str) -> Result<PairScore> {
    if stdout.is_empty() {
        anyhow::bail!("scorer produced no stdout");
    }

    let raw = match serde_json::from_str::<Value>(stdout) {
        Ok(value) => value,
        Err(_) => {
            let lift: f64 = stdout
                .parse()
                .with_context(|| format!("scorer stdout was not JSON or a number: {stdout:?}"))?;
            return Ok(PairScore {
                base_score: None,
                adapter_score: None,
                lift,
                raw: json!(lift),
            });
        }
    };

    match &raw {
        Value::Number(number) => {
            let lift = number
                .as_f64()
                .ok_or_else(|| anyhow!("scorer numeric output was not finite"))?;
            Ok(PairScore {
                base_score: None,
                adapter_score: None,
                lift,
                raw,
            })
        }
        Value::Object(map) => {
            let base_score = first_number(map, &["base_score", "score_base"]);
            let adapter_score = first_number(
                map,
                &["adapter_score", "candidate_score", "score_adapter"],
            );
            let lift = first_number(map, &["lift", "delta", "score"])
                .or_else(|| Some(adapter_score? - base_score?))
                .ok_or_else(|| {
                    anyhow!(
                        "scorer JSON must include lift/delta/score or both base_score and adapter_score"
                    )
                })?;
            Ok(PairScore {
                base_score,
                adapter_score,
                lift,
                raw,
            })
        }
        _ => anyhow::bail!("scorer JSON output must be a number or object"),
    }
}

fn first_number(map: &Map<String, Value>, names: &[&str]) -> Option<f64> {
    names.iter().find_map(|name| map.get(*name)?.as_f64())
}

fn summarize_results(results: &[EvalAdapterPairResult], wall_clock_ms: f64) -> EvalAdapterStats {
    let lifts: Vec<f64> = results.iter().map(|result| result.lift).collect();
    let mean_lift = mean(&lifts);
    let stdev_lift = sample_stdev(&lifts, mean_lift);
    let zero_count = results.iter().filter(|result| result.zero_lift).count();
    let base_scores: Vec<f64> = results
        .iter()
        .filter_map(|result| result.base_score)
        .collect();
    let adapter_scores: Vec<f64> = results
        .iter()
        .filter_map(|result| result.adapter_score)
        .collect();
    let pair_times: Vec<f64> = results
        .iter()
        .map(|result| result.wall_clock_ms)
        .collect();

    EvalAdapterStats {
        mean_lift,
        stdev_lift,
        zero_count,
        base_mean: (!base_scores.is_empty()).then(|| mean(&base_scores)),
        adapter_mean: (!adapter_scores.is_empty()).then(|| mean(&adapter_scores)),
        wall_clock_ms,
        mean_pair_wall_clock_ms: mean(&pair_times),
        max_pair_wall_clock_ms: pair_times.into_iter().fold(0.0, f64::max),
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_stdev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance =
        values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

async fn fetch_model_state(
    client: &reqwest::Client,
    url: &str,
    adapter: &str,
) -> ModelStateSnapshot {
    match client
        .get(format!("{url}/v1/debug/model-state"))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let body = resp.json::<Value>().await.unwrap_or_else(|_| json!({}));
            let config_hashes = body.get("config_hashes").cloned();
            let adapter_hashes = body
                .pointer("/adapters/loaded_adapters")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|entry| {
                    let name = entry.get("name")?.as_str()?.to_string();
                    Some(AdapterHashRecord {
                        name,
                        path: entry
                            .get("path")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                        adapter_model_sha256: entry
                            .get("adapter_model_sha256")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    })
                })
                .collect();
            ModelStateSnapshot {
                config_hashes,
                adapter_hashes,
                warning: None,
            }
        }
        Ok(resp) => {
            let status = resp.status();
            let health = fetch_health_hashes(client, url).await;
            ModelStateSnapshot {
                config_hashes: health,
                adapter_hashes: Vec::new(),
                warning: Some(format!(
                    "could not record adapter hash for `{adapter}` because /v1/debug/model-state returned {status}; run the server with --eval-mode or KILN_DEBUG_ENDPOINTS=1"
                )),
            }
        }
        Err(err) => {
            let health = fetch_health_hashes(client, url).await;
            ModelStateSnapshot {
                config_hashes: health,
                adapter_hashes: Vec::new(),
                warning: Some(format!(
                    "could not record adapter hash for `{adapter}` because /v1/debug/model-state was unreachable: {err}"
                )),
            }
        }
    }
}

async fn fetch_health_hashes(client: &reqwest::Client, url: &str) -> Option<Value> {
    let resp = client.get(format!("{url}/health")).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body = resp.json::<Value>().await.ok()?;
    body.get("config_hashes").cloned()
}

fn render_http_error(status: StatusCode, body: &Value) -> String {
    if let Some(error) = body.get("error") {
        if let Some(message) = error.get("message").and_then(Value::as_str) {
            return format!("{status}: {message}");
        }
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
    use axum::routing::{get, post};
    use std::sync::{Arc, Mutex};

    #[test]
    fn render_template_substitutes_task_seed_and_forces_adapter() {
        let template = json!({
            "model": "Qwen3.5-4B",
            "messages": [
                {"role": "user", "content": "Question {{question}} seed {{seed}} adapter {{adapter_label}}"},
                "{{task.extra_messages}}"
            ],
            "max_tokens": "{{max_tokens}}"
        });
        let task = json!({
            "id": "task-a",
            "question": "2+2?",
            "max_tokens": 3,
            "extra_messages": [{"role": "system", "content": "be short"}]
        });

        let base = render_chat_request(&template, &task, 2, None).unwrap();
        assert_eq!(base["adapter"], Value::Null);
        assert_eq!(base["seed"], 2);
        assert_eq!(base["max_tokens"], 3);
        assert_eq!(
            base["messages"][0]["content"],
            "Question 2+2? seed 2 adapter base"
        );
        assert_eq!(base["messages"][1][0]["role"], "system");

        let adapter = render_chat_request(&template, &task, 2, Some("cap")).unwrap();
        assert_eq!(adapter["adapter"], "cap");
        assert_eq!(
            adapter["messages"][0]["content"],
            "Question 2+2? seed 2 adapter cap"
        );
    }

    #[test]
    fn scorer_output_derives_lift_and_stats_warning_inputs() {
        let score = parse_scorer_stdout(r#"{"base_score":0.25,"adapter_score":0.75}"#).unwrap();
        assert_eq!(score.base_score, Some(0.25));
        assert_eq!(score.adapter_score, Some(0.75));
        assert_eq!(score.lift, 0.5);

        let results = vec![
            EvalAdapterPairResult {
                task_index: 0,
                task_id: "a".to_string(),
                seed: 0,
                base_score: Some(0.0),
                adapter_score: Some(1.0),
                lift: 1.0,
                zero_lift: false,
                wall_clock_ms: 10.0,
                base_content: "base".to_string(),
                adapter_content: "adapter".to_string(),
                scorer_output: json!({}),
            },
            EvalAdapterPairResult {
                task_index: 0,
                task_id: "a".to_string(),
                seed: 1,
                base_score: Some(0.5),
                adapter_score: Some(0.5),
                lift: 0.0,
                zero_lift: true,
                wall_clock_ms: 30.0,
                base_content: "base".to_string(),
                adapter_content: "adapter".to_string(),
                scorer_output: json!({}),
            },
        ];
        let stats = summarize_results(&results, 45.0);
        assert_eq!(stats.mean_lift, 0.5);
        assert!((stats.stdev_lift - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert_eq!(stats.zero_count, 1);
        assert_eq!(stats.base_mean, Some(0.25));
        assert_eq!(stats.adapter_mean, Some(0.75));
        assert_eq!(stats.mean_pair_wall_clock_ms, 20.0);
        assert_eq!(stats.max_pair_wall_clock_ms, 30.0);
    }

    #[tokio::test]
    async fn run_eval_adapter_writes_summary_with_pairing_and_hashes() {
        #[derive(Clone, Default)]
        struct Calls(Arc<Mutex<Vec<Value>>>);

        async fn chat(State(calls): State<Calls>, axum::Json(body): axum::Json<Value>) -> axum::Json<Value> {
            calls.0.lock().unwrap().push(body.clone());
            let adapter = body.get("adapter").and_then(Value::as_str).unwrap_or("base");
            axum::Json(json!({
                "id": "chatcmpl-test",
                "model": "Qwen3.5-4B",
                "choices": [{
                    "message": {"role": "assistant", "content": format!("{adapter} answer")},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2}
            }))
        }

        async fn debug_state() -> axum::Json<Value> {
            axum::Json(json!({
                "config_hashes": {
                    "tokenizer_config_hash": "sha256:tok",
                    "chat_template_hash": "sha256:tmpl",
                    "model_config_hash": "sha256:model",
                    "kiln_env_config_hash": "sha256:env"
                },
                "adapters": {
                    "loaded_adapters": [{
                        "name": "cap",
                        "path": "/models/Qwen3.5-4B/adapters/cap",
                        "adapter_model_sha256": "sha256:adapter"
                    }]
                }
            }))
        }

        async fn health() -> axum::Json<Value> {
            axum::Json(json!({"config_hashes": {}}))
        }

        let calls = Calls::default();
        let app = Router::new()
            .route("/v1/chat/completions", post(chat))
            .route("/v1/debug/model-state", get(debug_state))
            .route("/health", get(health))
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
        let output = tmp.path().join("eval_summary.json");
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
            "#!/usr/bin/env python3\nimport json,sys\np=json.load(sys.stdin)\nprint(json.dumps({'base_score': 0.0 if p['base']['content'].startswith('base') else -1.0, 'adapter_score': 1.0 if p['candidate']['content'].startswith('cap') else -1.0}))\n",
        )
        .unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(&scorer).unwrap().permissions();
            perms.set_mode(0o755);
            std::fs::set_permissions(&scorer, perms).unwrap();
        }

        let summary = run_eval_adapter(EvalAdapterOptions {
            url: format!("http://{addr}"),
            adapter: "cap".to_string(),
            tasks,
            seeds: 1,
            request_template: template,
            scorer,
            output: output.clone(),
        })
        .await
        .unwrap();

        assert_eq!(summary.pair_count, 1);
        assert_eq!(summary.stats.mean_lift, 1.0);
        assert_eq!(summary.adapter_hashes[0].adapter_model_sha256.as_deref(), Some("sha256:adapter"));
        assert!(output.exists());
        let observed = calls.0.lock().unwrap().clone();
        assert_eq!(observed.len(), 2);
        assert!(observed[0]["adapter"].is_null());
        assert_eq!(observed[1]["adapter"], "cap");
        assert_eq!(observed[0]["seed"], observed[1]["seed"]);

        server.abort();
    }
}
