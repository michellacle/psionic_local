use std::{
    env,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitCode},
};

use psionic_models::{
    GgufDecoderAdapterLoader, PromptMessage, PromptMessageRole, PromptRenderOptions,
};
use psionic_runtime::{StructuredOutputRequest, StructuredOutputValue, DEFAULT_PENALTY_LOOKBACK};
use psionic_serve::{
    CudaGgufQwen35TextGenerationService, GenerationOptions, GenerationRequest,
    Qwen35CudaDecodeOutputMetrics, TextGenerationExecutor,
};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let config = BenchConfig::parse(env::args().skip(1))?;
    match config.backend {
        BenchBackend::Psionic => run_psionic_benchmark(&config),
        BenchBackend::Ollama => run_ollama_benchmark(&config),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchBackend {
    Psionic,
    Ollama,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BenchDecodeMode {
    Greedy,
    Sample,
}

#[derive(Clone, Debug)]
struct BenchConfig {
    backend: BenchBackend,
    model_path: PathBuf,
    ollama_model: Option<String>,
    ollama_base_url: String,
    prompt: String,
    max_output_tokens: usize,
    repeats: usize,
    decode_mode: BenchDecodeMode,
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    min_p: Option<f32>,
    typical_p: Option<f32>,
    mirostat: Option<u8>,
    mirostat_tau: Option<f32>,
    mirostat_eta: Option<f32>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    seed: Option<u64>,
    structured_output: Option<BenchStructuredOutput>,
    jsonl_out: Option<PathBuf>,
    row_label: Option<String>,
    model_alias: Option<String>,
}

#[derive(Clone, Debug)]
enum BenchStructuredOutput {
    JsonObject,
    JsonSchema { name: Option<String>, schema: Value },
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            backend: BenchBackend::Psionic,
            model_path: PathBuf::new(),
            ollama_model: None,
            ollama_base_url: String::from("http://127.0.0.1:11434"),
            prompt: String::from("Explain what Psionic is in one sentence."),
            max_output_tokens: 256,
            repeats: 3,
            decode_mode: BenchDecodeMode::Greedy,
            temperature: None,
            top_k: None,
            top_p: None,
            min_p: None,
            typical_p: None,
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            repeat_penalty: None,
            repeat_last_n: None,
            presence_penalty: None,
            frequency_penalty: None,
            seed: None,
            structured_output: None,
            jsonl_out: None,
            row_label: None,
            model_alias: None,
        }
    }
}

impl BenchConfig {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut config = Self::default();
        let raw_args = args.collect::<Vec<_>>();
        let mut json_object = false;
        let mut json_schema_file: Option<PathBuf> = None;
        let mut json_schema_name: Option<String> = None;
        if raw_args.is_empty() {
            return Err(usage());
        }
        if !raw_args[0].starts_with("--") {
            return Self::parse_legacy(raw_args);
        }
        let mut index = 0;
        while index < raw_args.len() {
            let argument = &raw_args[index];
            match argument.as_str() {
                "--backend" => {
                    config.backend = match next_arg(&raw_args, &mut index, "--backend")?.as_str() {
                        "psionic" => BenchBackend::Psionic,
                        "ollama" => BenchBackend::Ollama,
                        value => {
                            return Err(format!(
                                "invalid --backend `{value}`; expected `psionic` or `ollama`"
                            ));
                        }
                    };
                }
                "--model-path" => {
                    config.model_path =
                        PathBuf::from(next_arg(&raw_args, &mut index, "--model-path")?);
                }
                "--ollama-model" => {
                    config.ollama_model = Some(next_arg(&raw_args, &mut index, "--ollama-model")?);
                }
                "--ollama-base-url" => {
                    config.ollama_base_url = next_arg(&raw_args, &mut index, "--ollama-base-url")?;
                }
                "--prompt" => {
                    config.prompt = next_arg(&raw_args, &mut index, "--prompt")?;
                }
                "--max-output-tokens" => {
                    config.max_output_tokens = parse_arg(
                        &next_arg(&raw_args, &mut index, "--max-output-tokens")?,
                        "--max-output-tokens",
                    )?;
                }
                "--repeats" => {
                    config.repeats =
                        parse_arg(&next_arg(&raw_args, &mut index, "--repeats")?, "--repeats")?;
                }
                "--decode" => {
                    config.decode_mode = match next_arg(&raw_args, &mut index, "--decode")?.as_str()
                    {
                        "greedy" => BenchDecodeMode::Greedy,
                        "sample" => BenchDecodeMode::Sample,
                        value => {
                            return Err(format!(
                                "invalid --decode `{value}`; expected `greedy` or `sample`"
                            ));
                        }
                    };
                }
                "--temperature" => {
                    config.temperature = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--temperature")?,
                        "--temperature",
                    )?);
                }
                "--top-k" => {
                    config.top_k = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--top-k")?,
                        "--top-k",
                    )?);
                }
                "--top-p" => {
                    config.top_p = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--top-p")?,
                        "--top-p",
                    )?);
                }
                "--min-p" => {
                    config.min_p = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--min-p")?,
                        "--min-p",
                    )?);
                }
                "--typical-p" => {
                    config.typical_p = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--typical-p")?,
                        "--typical-p",
                    )?);
                }
                "--mirostat" => {
                    config.mirostat = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--mirostat")?,
                        "--mirostat",
                    )?);
                }
                "--mirostat-tau" => {
                    config.mirostat_tau = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--mirostat-tau")?,
                        "--mirostat-tau",
                    )?);
                }
                "--mirostat-eta" => {
                    config.mirostat_eta = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--mirostat-eta")?,
                        "--mirostat-eta",
                    )?);
                }
                "--repeat-penalty" => {
                    config.repeat_penalty = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--repeat-penalty")?,
                        "--repeat-penalty",
                    )?);
                }
                "--repeat-last-n" => {
                    config.repeat_last_n = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--repeat-last-n")?,
                        "--repeat-last-n",
                    )?);
                }
                "--presence-penalty" => {
                    config.presence_penalty = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--presence-penalty")?,
                        "--presence-penalty",
                    )?);
                }
                "--frequency-penalty" => {
                    config.frequency_penalty = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--frequency-penalty")?,
                        "--frequency-penalty",
                    )?);
                }
                "--seed" => {
                    config.seed = Some(parse_arg(
                        &next_arg(&raw_args, &mut index, "--seed")?,
                        "--seed",
                    )?);
                }
                "--json-object" => {
                    json_object = true;
                }
                "--json-schema-file" => {
                    json_schema_file = Some(PathBuf::from(next_arg(
                        &raw_args,
                        &mut index,
                        "--json-schema-file",
                    )?));
                }
                "--json-schema-name" => {
                    json_schema_name = Some(next_arg(&raw_args, &mut index, "--json-schema-name")?);
                }
                "--jsonl-out" => {
                    config.jsonl_out = Some(PathBuf::from(next_arg(
                        &raw_args,
                        &mut index,
                        "--jsonl-out",
                    )?));
                }
                "--row-label" => {
                    config.row_label = Some(next_arg(&raw_args, &mut index, "--row-label")?);
                }
                "--model-alias" => {
                    config.model_alias = Some(next_arg(&raw_args, &mut index, "--model-alias")?);
                }
                "--greedy" => {
                    config.decode_mode = BenchDecodeMode::Greedy;
                }
                "--sample" => {
                    config.decode_mode = BenchDecodeMode::Sample;
                }
                "--help" | "-h" => {
                    return Err(usage());
                }
                value => {
                    return Err(format!("unknown argument `{value}`\n\n{}", usage()));
                }
            }
            index += 1;
        }
        config.structured_output =
            parse_structured_output(json_object, json_schema_file, json_schema_name)?;
        config.validate()?;
        Ok(config)
    }

    fn parse_legacy(args: Vec<String>) -> Result<Self, String> {
        let mut config = Self::default();
        config.model_path = PathBuf::from(args.first().cloned().ok_or_else(usage)?);
        if let Some(prompt) = args.get(1) {
            config.prompt = prompt.clone();
        }
        if let Some(max_output_tokens) = args.get(2) {
            config.max_output_tokens = parse_arg(max_output_tokens, "max_output_tokens")?;
        }
        if let Some(repeats) = args.get(3) {
            config.repeats = parse_arg(repeats, "repeats")?;
        }
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.model_path.as_os_str().is_empty() {
            return Err(format!("missing --model-path\n\n{}", usage()));
        }
        if self.repeats == 0 {
            return Err(String::from("--repeats must be at least 1"));
        }
        if matches!(self.backend, BenchBackend::Ollama) && self.ollama_model.is_none() {
            return Err(String::from(
                "missing --ollama-model for `--backend ollama`",
            ));
        }
        Ok(())
    }

    fn effective_temperature(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.temperature,
            BenchDecodeMode::Sample => Some(self.temperature.unwrap_or(0.8)),
        }
    }

    fn effective_top_k(&self) -> Option<usize> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.top_k,
            BenchDecodeMode::Sample => Some(self.top_k.unwrap_or(40)),
        }
    }

    fn effective_top_p(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.top_p,
            BenchDecodeMode::Sample => Some(self.top_p.unwrap_or(0.9)),
        }
    }

    fn effective_min_p(&self) -> Option<f32> {
        self.min_p.filter(|min_p| min_p.is_finite() && *min_p > 0.0)
    }

    fn effective_typical_p(&self) -> Option<f32> {
        self.typical_p
            .filter(|typical_p| typical_p.is_finite() && *typical_p > 0.0 && *typical_p < 1.0)
    }

    fn effective_mirostat(&self) -> Option<u8> {
        self.mirostat.filter(|value| matches!(value, 1 | 2))
    }

    fn effective_mirostat_tau(&self) -> Option<f32> {
        self.effective_mirostat()
            .map(|_| self.mirostat_tau.unwrap_or(5.0).max(0.0))
    }

    fn effective_mirostat_eta(&self) -> Option<f32> {
        self.effective_mirostat()
            .map(|_| self.mirostat_eta.unwrap_or(0.1).max(0.0))
    }

    fn effective_repeat_penalty(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.repeat_penalty,
            BenchDecodeMode::Sample => Some(self.repeat_penalty.unwrap_or(1.0)),
        }
    }

    fn effective_repeat_last_n(&self) -> Option<i32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.repeat_last_n,
            BenchDecodeMode::Sample => Some(
                self.repeat_last_n
                    .unwrap_or(DEFAULT_PENALTY_LOOKBACK as i32),
            ),
        }
    }

    fn effective_presence_penalty(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.presence_penalty,
            BenchDecodeMode::Sample => Some(self.presence_penalty.unwrap_or(0.0)),
        }
    }

    fn effective_frequency_penalty(&self) -> Option<f32> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.frequency_penalty,
            BenchDecodeMode::Sample => Some(self.frequency_penalty.unwrap_or(0.0)),
        }
    }

    fn effective_seed(&self) -> Option<u64> {
        match self.decode_mode {
            BenchDecodeMode::Greedy => self.seed,
            BenchDecodeMode::Sample => Some(self.seed.unwrap_or(42)),
        }
    }

    fn ollama_format_payload(&self) -> Option<Value> {
        match self.structured_output.as_ref() {
            Some(BenchStructuredOutput::JsonObject) => Some(Value::String(String::from("json"))),
            Some(BenchStructuredOutput::JsonSchema { schema, .. }) => Some(schema.clone()),
            None => None,
        }
    }
}

fn run_psionic_benchmark(config: &BenchConfig) -> Result<(), String> {
    let rendered = render_prompt(&config.model_path, &config.prompt)?;
    let jsonl_base = build_jsonl_base(config, &rendered);
    let mut service = CudaGgufQwen35TextGenerationService::from_gguf_path(&config.model_path)
        .map_err(|error| format!("failed to load qwen35 cuda service: {error}"))?;
    let descriptor = service.model_descriptor().clone();

    let warmup = GenerationRequest::new_text(
        String::from("warmup"),
        descriptor.clone(),
        None,
        rendered.text.clone(),
        build_generation_options(
            config,
            min_warmup_tokens(config.max_output_tokens),
            &rendered.stop_sequences,
        ),
    );
    let _ = service
        .generate(&warmup)
        .map_err(|error| format!("warmup generation failed: {error}"))?;

    let mut decode_tok_s_values = Vec::with_capacity(config.repeats);
    for run_index in 0..config.repeats {
        let request = GenerationRequest::new_text(
            format!("bench-{run_index}"),
            descriptor.clone(),
            None,
            rendered.text.clone(),
            build_generation_options(config, config.max_output_tokens, &rendered.stop_sequences),
        );
        let response = service
            .generate(&request)
            .map_err(|error| format!("benchmark generation failed: {error}"))?;
        let output_tokens = response
            .metrics
            .eval_count
            .unwrap_or(response.output.tokens.len());
        let decode_ns = response.metrics.eval_duration_ns.unwrap_or(0);
        let prompt_ns = response.metrics.prompt_eval_duration_ns.unwrap_or(0);
        let total_ns = response.metrics.total_duration_ns.unwrap_or(0);
        let decode_tok_s = tokens_per_second(output_tokens, decode_ns);
        decode_tok_s_values.push(decode_tok_s);
        let output_metrics_struct =
            qwen35_output_metrics_report(response.metrics.qwen35_cuda_decode.as_ref());
        let output_metrics = format_qwen35_output_metrics(&output_metrics_struct);
        let structured_output = format_structured_output_report(
            response.provenance.as_ref(),
            response.output.structured.as_ref(),
        );
        let termination_reason = termination_reason_label(response.termination);
        println!(
            "backend=psionic run={} decode_mode={} prompt_tokens={} output_tokens={} prompt_s={:.6} decode_s={:.6} total_s={:.6} decode_tok_s={:.2} termination_reason={} {} {} output={}",
            run_index + 1,
            bench_decode_mode_label(config.decode_mode),
            response.metrics.prompt_eval_count.unwrap_or(0),
            output_tokens,
            nanos_to_seconds(prompt_ns),
            nanos_to_seconds(decode_ns),
            nanos_to_seconds(total_ns),
            decode_tok_s,
            termination_reason,
            output_metrics,
            structured_output,
            response.output.text.replace('\n', "\\n"),
        );

        let mut record = jsonl_base.clone();
        let map = record
            .as_object_mut()
            .ok_or_else(|| String::from("invalid jsonl base record"))?;
        map.insert(
            String::from("record_type"),
            Value::String(String::from("run")),
        );
        map.insert(String::from("run"), Value::Number((run_index + 1).into()));
        map.insert(
            String::from("prompt_tokens"),
            Value::Number(response.metrics.prompt_eval_count.unwrap_or(0).into()),
        );
        map.insert(
            String::from("output_tokens"),
            Value::Number(output_tokens.into()),
        );
        map.insert(
            String::from("prompt_s"),
            serde_json::json!(nanos_to_seconds(prompt_ns)),
        );
        map.insert(
            String::from("decode_s"),
            serde_json::json!(nanos_to_seconds(decode_ns)),
        );
        map.insert(
            String::from("total_s"),
            serde_json::json!(nanos_to_seconds(total_ns)),
        );
        map.insert(
            String::from("decode_tok_s"),
            serde_json::json!(decode_tok_s),
        );
        map.insert(
            String::from("termination_reason"),
            Value::String(String::from(termination_reason)),
        );
        map.insert(
            String::from("qwen35_output_modes"),
            serde_json::json!(output_metrics_struct.output_modes),
        );
        map.insert(
            String::from("qwen35_readback_bytes"),
            serde_json::json!(output_metrics_struct.readback_bytes),
        );
        map.insert(
            String::from("qwen35_raw_logits"),
            serde_json::json!(output_metrics_struct.raw_logits_materialized),
        );
        append_jsonl_record(config.jsonl_out.as_ref(), &record)?;
    }

    let mean_decode_tok_s = mean(&decode_tok_s_values);
    println!("backend=psionic mean_decode_tok_s={:.2}", mean_decode_tok_s);
    let mut summary = jsonl_base.clone();
    let map = summary
        .as_object_mut()
        .ok_or_else(|| String::from("invalid jsonl base record"))?;
    map.insert(
        String::from("record_type"),
        Value::String(String::from("summary")),
    );
    map.insert(
        String::from("repeats"),
        Value::Number(config.repeats.into()),
    );
    map.insert(
        String::from("mean_decode_tok_s"),
        serde_json::json!(mean_decode_tok_s),
    );
    map.insert(
        String::from("stddev_decode_tok_s"),
        serde_json::json!(stddev(&decode_tok_s_values)),
    );
    map.insert(
        String::from("min_decode_tok_s"),
        serde_json::json!(decode_tok_s_values
            .iter()
            .fold(f64::INFINITY, |acc, v| acc.min(*v))),
    );
    map.insert(
        String::from("max_decode_tok_s"),
        serde_json::json!(decode_tok_s_values
            .iter()
            .fold(f64::NEG_INFINITY, |acc, v| acc.max(*v))),
    );
    append_jsonl_record(config.jsonl_out.as_ref(), &summary)?;
    Ok(())
}

fn run_ollama_benchmark(config: &BenchConfig) -> Result<(), String> {
    let rendered = render_prompt(&config.model_path, &config.prompt)?;
    let jsonl_base = build_jsonl_base(config, &rendered);
    let client = Client::builder()
        .build()
        .map_err(|error| format!("failed to build Ollama HTTP client: {error}"))?;
    let ollama_model = config
        .ollama_model
        .as_ref()
        .ok_or_else(|| String::from("missing Ollama model alias"))?;

    let _ = ollama_generate(
        &client,
        &config.ollama_base_url,
        ollama_model,
        &rendered,
        config,
        min_warmup_tokens(config.max_output_tokens),
    )?;

    let mut decode_tok_s_values = Vec::with_capacity(config.repeats);
    for run_index in 0..config.repeats {
        let response = ollama_generate(
            &client,
            &config.ollama_base_url,
            ollama_model,
            &rendered,
            config,
            config.max_output_tokens,
        )?;
        let output_tokens = response.eval_count.unwrap_or(0);
        let decode_ns = response.eval_duration.unwrap_or(0);
        let prompt_ns = response.prompt_eval_duration.unwrap_or(0);
        let total_ns = response.total_duration.unwrap_or(0);
        let decode_tok_s = tokens_per_second(output_tokens, decode_ns);
        decode_tok_s_values.push(decode_tok_s);
        let finish_reason = response.done_reason.as_deref().unwrap_or("none");
        println!(
            "backend=ollama run={} decode_mode={} prompt_tokens={} output_tokens={} prompt_s={:.6} decode_s={:.6} total_s={:.6} decode_tok_s={:.2} finish_reason={} output={}",
            run_index + 1,
            bench_decode_mode_label(config.decode_mode),
            response.prompt_eval_count.unwrap_or(0),
            output_tokens,
            nanos_to_seconds(prompt_ns),
            nanos_to_seconds(decode_ns),
            nanos_to_seconds(total_ns),
            decode_tok_s,
            finish_reason,
            response.response.replace('\n', "\\n"),
        );
        let mut record = jsonl_base.clone();
        let map = record
            .as_object_mut()
            .ok_or_else(|| String::from("invalid jsonl base record"))?;
        map.insert(
            String::from("record_type"),
            Value::String(String::from("run")),
        );
        map.insert(String::from("run"), Value::Number((run_index + 1).into()));
        map.insert(
            String::from("prompt_tokens"),
            Value::Number(response.prompt_eval_count.unwrap_or(0).into()),
        );
        map.insert(
            String::from("output_tokens"),
            Value::Number(output_tokens.into()),
        );
        map.insert(
            String::from("prompt_s"),
            serde_json::json!(nanos_to_seconds(prompt_ns)),
        );
        map.insert(
            String::from("decode_s"),
            serde_json::json!(nanos_to_seconds(decode_ns)),
        );
        map.insert(
            String::from("total_s"),
            serde_json::json!(nanos_to_seconds(total_ns)),
        );
        map.insert(
            String::from("decode_tok_s"),
            serde_json::json!(decode_tok_s),
        );
        map.insert(
            String::from("finish_reason"),
            response
                .done_reason
                .as_ref()
                .map_or(Value::Null, |value| Value::String(value.clone())),
        );
        map.insert(
            String::from("ollama_done"),
            response.done.map_or(Value::Null, Value::Bool),
        );
        append_jsonl_record(config.jsonl_out.as_ref(), &record)?;
    }

    let mean_decode_tok_s = mean(&decode_tok_s_values);
    println!("backend=ollama mean_decode_tok_s={:.2}", mean_decode_tok_s);
    let mut summary = jsonl_base.clone();
    let map = summary
        .as_object_mut()
        .ok_or_else(|| String::from("invalid jsonl base record"))?;
    map.insert(
        String::from("record_type"),
        Value::String(String::from("summary")),
    );
    map.insert(
        String::from("repeats"),
        Value::Number(config.repeats.into()),
    );
    map.insert(
        String::from("mean_decode_tok_s"),
        serde_json::json!(mean_decode_tok_s),
    );
    map.insert(
        String::from("stddev_decode_tok_s"),
        serde_json::json!(stddev(&decode_tok_s_values)),
    );
    map.insert(
        String::from("min_decode_tok_s"),
        serde_json::json!(decode_tok_s_values
            .iter()
            .fold(f64::INFINITY, |acc, v| acc.min(*v))),
    );
    map.insert(
        String::from("max_decode_tok_s"),
        serde_json::json!(decode_tok_s_values
            .iter()
            .fold(f64::NEG_INFINITY, |acc, v| acc.max(*v))),
    );
    append_jsonl_record(config.jsonl_out.as_ref(), &summary)?;
    Ok(())
}

fn build_generation_options(
    config: &BenchConfig,
    max_output_tokens: usize,
    stop_sequences: &[String],
) -> GenerationOptions {
    let mut options = match config.decode_mode {
        BenchDecodeMode::Greedy => GenerationOptions::greedy(max_output_tokens),
        BenchDecodeMode::Sample => GenerationOptions::sample(max_output_tokens),
    };
    options.stop_sequences = stop_sequences.to_vec();
    options.temperature = config.effective_temperature();
    options.top_k = config.effective_top_k();
    options.top_p = config.effective_top_p();
    options.min_p = config.effective_min_p();
    options.typical_p = config.effective_typical_p();
    options.mirostat = config.effective_mirostat();
    options.mirostat_tau = config.effective_mirostat_tau();
    options.mirostat_eta = config.effective_mirostat_eta();
    options.repeat_penalty = config.effective_repeat_penalty();
    options.repeat_last_n = config.effective_repeat_last_n();
    options.presence_penalty = config.effective_presence_penalty();
    options.frequency_penalty = config.effective_frequency_penalty();
    options.seed = config.effective_seed();
    options.structured_output = match config.structured_output.as_ref() {
        Some(BenchStructuredOutput::JsonObject) => Some(StructuredOutputRequest::JsonObject),
        Some(BenchStructuredOutput::JsonSchema { name, schema }) => {
            Some(StructuredOutputRequest::JsonSchema {
                name: name.clone(),
                schema: schema.clone(),
            })
        }
        None => None,
    };
    options
}

fn render_prompt(model_path: &Path, prompt: &str) -> Result<RenderedPrompt, String> {
    let adapter = GgufDecoderAdapterLoader
        .load_path(model_path)
        .map_err(|error| format!("failed to load GGUF metadata: {error}"))?;
    let renderer = adapter.prompt_renderer();
    let rendered = renderer
        .render_with_options(
            None,
            &[PromptMessage::new(
                PromptMessageRole::User,
                prompt.to_string(),
            )],
            true,
            &PromptRenderOptions::default(),
        )
        .map_err(|error| format!("failed to render qwen35 prompt: {error}"))?;
    Ok(RenderedPrompt {
        text: rendered.text,
        stop_sequences: rendered.stop_sequences,
    })
}

#[derive(Clone, Debug, Serialize)]
struct Qwen35OutputMetricsReport {
    output_modes: Vec<String>,
    readback_bytes: u64,
    raw_logits_materialized: bool,
}

fn qwen35_output_metrics_report(
    metrics: Option<&Qwen35CudaDecodeOutputMetrics>,
) -> Qwen35OutputMetricsReport {
    let Some(metrics) = metrics else {
        return Qwen35OutputMetricsReport {
            output_modes: Vec::new(),
            readback_bytes: 0,
            raw_logits_materialized: false,
        };
    };
    let output_modes = metrics
        .output_modes
        .iter()
        .map(|mode| match mode {
            psionic_serve::Qwen35CudaDecodeOutputMode::ArgmaxOnly => String::from("argmax_only"),
            psionic_serve::Qwen35CudaDecodeOutputMode::TopKCandidates { top_k } => {
                format!("top_k_candidates:{top_k}")
            }
            psionic_serve::Qwen35CudaDecodeOutputMode::SparseLogits { token_count } => {
                format!("sparse_logits:{token_count}")
            }
            psionic_serve::Qwen35CudaDecodeOutputMode::RawLogits => String::from("raw_logits"),
        })
        .collect::<Vec<_>>();
    Qwen35OutputMetricsReport {
        output_modes,
        readback_bytes: metrics.readback_bytes,
        raw_logits_materialized: metrics.raw_logits_materialized,
    }
}

fn format_qwen35_output_metrics(metrics: &Qwen35OutputMetricsReport) -> String {
    let output_modes = metrics.output_modes.join(",");
    format!(
        "qwen35_output_modes=[{}] qwen35_readback_bytes={} qwen35_raw_logits={}",
        output_modes, metrics.readback_bytes, metrics.raw_logits_materialized
    )
}

fn format_structured_output_report(
    provenance: Option<&psionic_serve::GenerationProvenance>,
    structured_value: Option<&StructuredOutputValue>,
) -> String {
    let Some(report) = provenance.and_then(|provenance| provenance.structured_output.as_ref())
    else {
        return String::from(
            "structured_output_mode=none structured_output_parser=none structured_output_kind=none structured_output_value=none",
        );
    };
    let value = match structured_value {
        Some(value) => {
            serde_json::to_string(value).unwrap_or_else(|_| String::from("\"<unserializable>\""))
        }
        None => String::from("none"),
    };
    format!(
        "structured_output_mode={} structured_output_parser={} structured_output_kind={} structured_output_value={}",
        report.mode.label(),
        report.parser.label(),
        report.kind.label(),
        value
    )
}

fn bench_decode_mode_label(mode: BenchDecodeMode) -> &'static str {
    match mode {
        BenchDecodeMode::Greedy => "greedy",
        BenchDecodeMode::Sample => "sample",
    }
}

fn min_warmup_tokens(max_output_tokens: usize) -> usize {
    max_output_tokens.min(16).max(1)
}

fn next_arg(args: &[String], index: &mut usize, flag: &str) -> Result<String, String> {
    let value_index = index.saturating_add(1);
    let value = args
        .get(value_index)
        .cloned()
        .ok_or_else(|| format!("missing value for `{flag}`"))?;
    *index = value_index;
    Ok(value)
}

fn parse_structured_output(
    json_object: bool,
    json_schema_file: Option<PathBuf>,
    json_schema_name: Option<String>,
) -> Result<Option<BenchStructuredOutput>, String> {
    let selected_modes = usize::from(json_object) + usize::from(json_schema_file.is_some());
    if selected_modes > 1 {
        return Err(String::from(
            "structured output accepts at most one of `--json-object` or `--json-schema-file`",
        ));
    }
    match json_schema_file {
        Some(path) => {
            let raw = fs::read_to_string(&path).map_err(|error| {
                format!(
                    "failed to read JSON schema file `{}`: {error}",
                    path.display()
                )
            })?;
            let schema = serde_json::from_str::<Value>(&raw).map_err(|error| {
                format!(
                    "failed to parse JSON schema file `{}`: {error}",
                    path.display()
                )
            })?;
            Ok(Some(BenchStructuredOutput::JsonSchema {
                name: json_schema_name,
                schema,
            }))
        }
        None if json_object => {
            if json_schema_name.is_some() {
                return Err(String::from(
                    "`--json-schema-name` requires `--json-schema-file`",
                ));
            }
            Ok(Some(BenchStructuredOutput::JsonObject))
        }
        None => {
            if json_schema_name.is_some() {
                return Err(String::from(
                    "`--json-schema-name` requires `--json-schema-file`",
                ));
            }
            Ok(None)
        }
    }
}

fn parse_arg<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse::<T>()
        .map_err(|error| format!("invalid {name} `{value}`: {error}"))
}

fn tokens_per_second(tokens: usize, duration_ns: u64) -> f64 {
    if tokens == 0 || duration_ns == 0 {
        return 0.0;
    }
    tokens as f64 / nanos_to_seconds(duration_ns)
}

fn nanos_to_seconds(duration_ns: u64) -> f64 {
    duration_ns as f64 / 1_000_000_000.0
}

fn usage() -> String {
    String::from(
        "usage:\n  cargo run -p psionic-serve --example qwen35_cuda_bench -- <model.gguf> [prompt] [max_output_tokens] [repeats]\n  cargo run -p psionic-serve --example qwen35_cuda_bench -- --backend psionic --model-path <model.gguf> [--model-alias qwen3.5:0.8b] [--row-label sampled_topk40] [--decode greedy|sample] [--temperature 0.8] [--top-k 40] [--top-p 0.9] [--min-p 0.05] [--typical-p 0.5] [--mirostat 1|2] [--mirostat-tau 5.0] [--mirostat-eta 0.1] [--repeat-penalty 1.0] [--repeat-last-n 64] [--presence-penalty 0.0] [--frequency-penalty 0.0] [--seed 42] [--json-object | --json-schema-file schema.json [--json-schema-name summary]] [--jsonl-out run.jsonl] [--prompt <text>] [--max-output-tokens 128] [--repeats 3]\n  cargo run -p psionic-serve --example qwen35_cuda_bench -- --backend ollama --model-path <model.gguf> --ollama-model qwen3.5:0.8b [--model-alias qwen3.5:0.8b] [--row-label sampled_topk40] [--ollama-base-url http://127.0.0.1:11434] [--decode greedy|sample] [--temperature 0.8] [--top-k 40] [--top-p 0.9] [--min-p 0.05] [--typical-p 0.5] [--mirostat 1|2] [--mirostat-tau 5.0] [--mirostat-eta 0.1] [--repeat-penalty 1.0] [--repeat-last-n 64] [--presence-penalty 0.0] [--frequency-penalty 0.0] [--seed 42] [--json-object | --json-schema-file schema.json [--json-schema-name summary]] [--jsonl-out run.jsonl] [--prompt <text>] [--max-output-tokens 128] [--repeats 3]",
    )
}

#[derive(Clone, Debug)]
struct RenderedPrompt {
    text: String,
    stop_sequences: Vec<String>,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
    #[serde(default)]
    done: Option<bool>,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<usize>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<usize>,
    #[serde(default)]
    eval_duration: Option<u64>,
}

fn ollama_generate(
    client: &Client,
    base_url: &str,
    model: &str,
    rendered: &RenderedPrompt,
    config: &BenchConfig,
    max_output_tokens: usize,
) -> Result<OllamaGenerateResponse, String> {
    let mut options = serde_json::json!({
        "num_predict": max_output_tokens,
    });
    if let Some(temperature) = config.effective_temperature() {
        options["temperature"] = serde_json::json!(temperature);
    }
    if let Some(top_k) = config.effective_top_k() {
        options["top_k"] = serde_json::json!(top_k);
    }
    if let Some(top_p) = config.effective_top_p() {
        options["top_p"] = serde_json::json!(top_p);
    }
    if let Some(min_p) = config.effective_min_p() {
        options["min_p"] = serde_json::json!(min_p);
    }
    if let Some(typical_p) = config.effective_typical_p() {
        options["typical_p"] = serde_json::json!(typical_p);
    }
    if let Some(mirostat) = config.effective_mirostat() {
        options["mirostat"] = serde_json::json!(mirostat);
    }
    if let Some(mirostat_tau) = config.effective_mirostat_tau() {
        options["mirostat_tau"] = serde_json::json!(mirostat_tau);
    }
    if let Some(mirostat_eta) = config.effective_mirostat_eta() {
        options["mirostat_eta"] = serde_json::json!(mirostat_eta);
    }
    if let Some(repeat_penalty) = config.effective_repeat_penalty() {
        options["repeat_penalty"] = serde_json::json!(repeat_penalty);
    }
    if let Some(repeat_last_n) = config.effective_repeat_last_n() {
        options["repeat_last_n"] = serde_json::json!(repeat_last_n);
    }
    if let Some(presence_penalty) = config.effective_presence_penalty() {
        options["presence_penalty"] = serde_json::json!(presence_penalty);
    }
    if let Some(frequency_penalty) = config.effective_frequency_penalty() {
        options["frequency_penalty"] = serde_json::json!(frequency_penalty);
    }
    if let Some(seed) = config.effective_seed() {
        options["seed"] = serde_json::json!(seed);
    }
    if !rendered.stop_sequences.is_empty() {
        options["stop"] = serde_json::json!(rendered.stop_sequences);
    }
    let mut payload = serde_json::json!({
        "model": model,
        "prompt": rendered.text,
        "raw": true,
        "stream": false,
        "think": false,
        "options": options,
    });
    if let Some(format) = config.ollama_format_payload() {
        payload["format"] = format;
    }
    let url = format!("{}/api/generate", base_url.trim_end_matches('/'));
    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .map_err(|error| format!("failed to call Ollama generate endpoint: {error}"))?;
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .unwrap_or_else(|_| String::from("<unreadable response body>"));
        return Err(format!(
            "Ollama generate request failed with {status}: {body}"
        ));
    }
    response
        .json::<OllamaGenerateResponse>()
        .map_err(|error| format!("failed to decode Ollama generate response: {error}"))
}

fn build_jsonl_base(config: &BenchConfig, rendered: &RenderedPrompt) -> Value {
    let mut base = Map::new();
    base.insert(
        String::from("schema_version"),
        Value::String(String::from("qwen35_bench_run_v2")),
    );
    base.insert(
        String::from("backend"),
        Value::String(match config.backend {
            BenchBackend::Psionic => String::from("psionic"),
            BenchBackend::Ollama => String::from("ollama"),
        }),
    );
    base.insert(
        String::from("decode_mode"),
        Value::String(String::from(bench_decode_mode_label(config.decode_mode))),
    );
    base.insert(
        String::from("model_path"),
        Value::String(config.model_path.display().to_string()),
    );
    base.insert(
        String::from("model_alias"),
        config
            .model_alias
            .as_ref()
            .map_or(Value::Null, |value| Value::String(value.clone())),
    );
    base.insert(
        String::from("row_label"),
        config
            .row_label
            .as_ref()
            .map_or(Value::Null, |value| Value::String(value.clone())),
    );
    base.insert(
        String::from("ollama_model"),
        config
            .ollama_model
            .as_ref()
            .map_or(Value::Null, |value| Value::String(value.clone())),
    );
    base.insert(
        String::from("git_commit"),
        detect_git_commit().map_or(Value::Null, Value::String),
    );
    base.insert(
        String::from("request_contract"),
        serde_json::json!({
            "prompt": config.prompt,
            "stop_sequences": rendered.stop_sequences,
            "max_output_tokens": config.max_output_tokens,
            "decode_mode": bench_decode_mode_label(config.decode_mode),
            "temperature": config.effective_temperature(),
            "top_k": config.effective_top_k(),
            "top_p": config.effective_top_p(),
            "min_p": config.effective_min_p(),
            "typical_p": config.effective_typical_p(),
            "mirostat": config.effective_mirostat(),
            "mirostat_tau": config.effective_mirostat_tau(),
            "mirostat_eta": config.effective_mirostat_eta(),
            "repeat_penalty": config.effective_repeat_penalty(),
            "repeat_last_n": config.effective_repeat_last_n(),
            "presence_penalty": config.effective_presence_penalty(),
            "frequency_penalty": config.effective_frequency_penalty(),
            "seed": config.effective_seed(),
            "structured_output": match config.structured_output.as_ref() {
                Some(BenchStructuredOutput::JsonObject) => Value::String(String::from("json_object")),
                Some(BenchStructuredOutput::JsonSchema { name, schema }) => serde_json::json!({
                    "type": "json_schema",
                    "name": name,
                    "schema": schema
                }),
                None => Value::Null,
            }
        }),
    );
    Value::Object(base)
}

fn append_jsonl_record(path: Option<&PathBuf>, record: &Value) -> Result<(), String> {
    let Some(path) = path else {
        return Ok(());
    };
    let line = serde_json::to_string(record)
        .map_err(|error| format!("failed to serialize jsonl benchmark record: {error}"))?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| format!("failed to open jsonl output `{}`: {error}", path.display()))?;
    writeln!(file, "{line}")
        .map_err(|error| format!("failed to write jsonl output `{}`: {error}", path.display()))
}

fn detect_git_commit() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let value = String::from_utf8(output.stdout).ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(String::from(trimmed))
    }
}

fn termination_reason_label(reason: psionic_serve::TerminationReason) -> &'static str {
    match reason {
        psionic_serve::TerminationReason::EndOfSequence => "end_of_sequence",
        psionic_serve::TerminationReason::MaxOutputTokens => "max_output_tokens",
        psionic_serve::TerminationReason::ContextLimit => "context_limit",
        psionic_serve::TerminationReason::Cancelled => "cancelled",
        psionic_serve::TerminationReason::Disconnected => "disconnected",
        psionic_serve::TerminationReason::Error => "error",
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let avg = mean(values);
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - avg;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}
