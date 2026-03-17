use std::{
    collections::BTreeMap,
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use psionic_mlx_bench::{
    MlxBenchWorkspace, MlxBenchmarkSuite, MlxBenchmarkSuiteSpec, MlxProviderResponse,
    MlxServedBenchmarkProvider, MlxTextBenchmarkProvider,
};
use psionic_mlx_vlm::MlxVlmServedEndpoint;
use serde::{Deserialize, Serialize};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::Usage(message)) => {
            let _ = writeln!(io::stdout(), "{message}");
            ExitCode::SUCCESS
        }
        Err(CliError::Message(message)) => {
            let _ = writeln!(io::stderr(), "{message}");
            ExitCode::FAILURE
        }
    }
}

#[derive(Debug)]
enum CliError {
    Usage(String),
    Message(String),
}

fn run() -> Result<(), CliError> {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        return Err(CliError::Usage(usage()));
    };
    match command.as_str() {
        "build-suite" => run_build_suite(args),
        "run-text-fixture" => run_text_fixture(args),
        "run-served-fixture" => run_served_fixture(args),
        "-h" | "--help" => Err(CliError::Usage(usage())),
        other => Err(CliError::Message(format!(
            "unrecognized subcommand `{other}`\n\n{}",
            usage()
        ))),
    }
}

fn run_build_suite(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_build_suite(args).map_err(CliError::Message)?;
    let suite = load_suite(parsed.spec_json.as_path()).map_err(CliError::Message)?;
    let output = SuiteOutput {
        benchmark_package: suite.benchmark_package().clone(),
        suite_manifest: suite.manifest().clone(),
    };
    write_json_output(&output, parsed.json_out).map_err(CliError::Message)
}

fn run_text_fixture(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_text_fixture(args).map_err(CliError::Message)?;
    let suite = load_suite(parsed.spec_json.as_path()).map_err(CliError::Message)?;
    let responses =
        load_fixture_responses(parsed.responses_json.as_path()).map_err(CliError::Message)?;
    let mut provider = MlxTextBenchmarkProvider::new(
        "fixture-text-provider",
        vec![String::from("fixture-backed local text benchmark adapter")],
        move |request| {
            responses
                .get(&request.request.request_id)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "missing fixture response for `{}`",
                        request.request.request_id
                    )
                })
        },
    )
    .map_err(|error| CliError::Message(format!("failed to build provider: {error}")))?;
    let receipt = suite
        .execute(
            &mut provider,
            psionic_eval::BenchmarkExecutionMode::OperatorSimulation,
        )
        .map_err(|error| CliError::Message(format!("benchmark run failed: {error}")))?;
    write_json_output(&receipt, parsed.json_out).map_err(CliError::Message)
}

fn run_served_fixture(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_served_fixture(args).map_err(CliError::Message)?;
    let suite = load_suite(parsed.spec_json.as_path()).map_err(CliError::Message)?;
    let responses =
        load_fixture_responses(parsed.responses_json.as_path()).map_err(CliError::Message)?;
    let mut provider = MlxServedBenchmarkProvider::new(
        "fixture-served-provider",
        parsed.model_reference,
        parsed.endpoint,
        vec![String::from("fixture-backed served benchmark adapter")],
        move |request| {
            responses
                .get(&request.request_id)
                .cloned()
                .ok_or_else(|| format!("missing fixture response for `{}`", request.request_id))
        },
    )
    .map_err(|error| CliError::Message(format!("failed to build provider: {error}")))?;
    let receipt = suite
        .execute(
            &mut provider,
            psionic_eval::BenchmarkExecutionMode::Validator,
        )
        .map_err(|error| CliError::Message(format!("benchmark run failed: {error}")))?;
    write_json_output(&receipt, parsed.json_out).map_err(CliError::Message)
}

#[derive(Clone, Debug)]
struct BuildSuiteArgs {
    spec_json: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct TextFixtureArgs {
    spec_json: PathBuf,
    responses_json: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct ServedFixtureArgs {
    spec_json: PathBuf,
    responses_json: PathBuf,
    model_reference: String,
    endpoint: MlxVlmServedEndpoint,
    json_out: Option<PathBuf>,
}

fn parse_build_suite(args: impl IntoIterator<Item = String>) -> Result<BuildSuiteArgs, String> {
    let mut spec_json = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--spec-json" => spec_json = Some(PathBuf::from(next_value(&mut args, "--spec-json")?)),
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(spec_json) = spec_json else {
        return Err(format!("missing required `--spec-json`\n\n{}", usage()));
    };
    Ok(BuildSuiteArgs {
        spec_json,
        json_out,
    })
}

fn parse_text_fixture(args: impl IntoIterator<Item = String>) -> Result<TextFixtureArgs, String> {
    let mut spec_json = None;
    let mut responses_json = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--spec-json" => spec_json = Some(PathBuf::from(next_value(&mut args, "--spec-json")?)),
            "--responses-json" => {
                responses_json = Some(PathBuf::from(next_value(&mut args, "--responses-json")?))
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(spec_json) = spec_json else {
        return Err(format!("missing required `--spec-json`\n\n{}", usage()));
    };
    let Some(responses_json) = responses_json else {
        return Err(format!(
            "missing required `--responses-json`\n\n{}",
            usage()
        ));
    };
    Ok(TextFixtureArgs {
        spec_json,
        responses_json,
        json_out,
    })
}

fn parse_served_fixture(
    args: impl IntoIterator<Item = String>,
) -> Result<ServedFixtureArgs, String> {
    let mut spec_json = None;
    let mut responses_json = None;
    let mut model_reference = None;
    let mut endpoint = MlxVlmServedEndpoint::Responses;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--spec-json" => spec_json = Some(PathBuf::from(next_value(&mut args, "--spec-json")?)),
            "--responses-json" => {
                responses_json = Some(PathBuf::from(next_value(&mut args, "--responses-json")?))
            }
            "--model-reference" => {
                model_reference = Some(next_value(&mut args, "--model-reference")?)
            }
            "--endpoint" => {
                endpoint = parse_endpoint(next_value(&mut args, "--endpoint")?.as_str())?;
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(spec_json) = spec_json else {
        return Err(format!("missing required `--spec-json`\n\n{}", usage()));
    };
    let Some(responses_json) = responses_json else {
        return Err(format!(
            "missing required `--responses-json`\n\n{}",
            usage()
        ));
    };
    let Some(model_reference) = model_reference else {
        return Err(format!(
            "missing required `--model-reference`\n\n{}",
            usage()
        ));
    };
    Ok(ServedFixtureArgs {
        spec_json,
        responses_json,
        model_reference,
        endpoint,
        json_out,
    })
}

fn parse_endpoint(value: &str) -> Result<MlxVlmServedEndpoint, String> {
    match value {
        "responses" => Ok(MlxVlmServedEndpoint::Responses),
        "chat" => Ok(MlxVlmServedEndpoint::ChatCompletions),
        other => Err(format!(
            "invalid --endpoint value `{other}` (expected responses or chat)\n\n{}",
            usage()
        )),
    }
}

fn load_suite(path: &std::path::Path) -> Result<MlxBenchmarkSuite, String> {
    let json = fs::read_to_string(path)
        .map_err(|error| format!("failed to read suite spec `{}`: {error}", path.display()))?;
    let spec: MlxBenchmarkSuiteSpec = serde_json::from_str(&json)
        .map_err(|error| format!("failed to decode suite spec `{}`: {error}", path.display()))?;
    MlxBenchWorkspace::default()
        .build_suite(&spec)
        .map_err(|error| format!("failed to build suite: {error}"))
}

fn load_fixture_responses(
    path: &std::path::Path,
) -> Result<BTreeMap<String, MlxProviderResponse>, String> {
    let json = fs::read_to_string(path).map_err(|error| {
        format!(
            "failed to read fixture responses `{}`: {error}",
            path.display()
        )
    })?;
    let file: FixtureResponseFile = serde_json::from_str(&json).map_err(|error| {
        format!(
            "failed to decode fixture responses `{}`: {error}",
            path.display()
        )
    })?;
    let mut responses = BTreeMap::new();
    for entry in file.responses {
        if responses
            .insert(entry.request_id.clone(), entry.response)
            .is_some()
        {
            return Err(format!(
                "fixture responses repeated request_id `{}`",
                entry.request_id
            ));
        }
    }
    Ok(responses)
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for `{flag}`"))
}

fn write_json_output<T: Serialize>(value: &T, json_out: Option<PathBuf>) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON output: {error}"))?;
    if let Some(path) = json_out {
        fs::write(&path, format!("{json}\n"))
            .map_err(|error| format!("failed to write `{}`: {error}", path.display()))
    } else {
        let mut stdout = io::stdout();
        writeln!(stdout, "{json}").map_err(|error| format!("failed to write stdout: {error}"))
    }
}

fn usage() -> String {
    String::from(
        "usage:\n  psionic-mlx-bench build-suite --spec-json <path> [--json-out <path>]\n  psionic-mlx-bench run-text-fixture --spec-json <path> --responses-json <path> [--json-out <path>]\n  psionic-mlx-bench run-served-fixture --spec-json <path> --responses-json <path> --model-reference <ref> [--endpoint responses|chat] [--json-out <path>]",
    )
}

#[derive(Clone, Debug, Serialize)]
struct SuiteOutput {
    benchmark_package: psionic_eval::BenchmarkPackage,
    suite_manifest: psionic_mlx_bench::MlxBenchmarkSuiteManifest,
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureResponseFile {
    responses: Vec<FixtureResponseEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureResponseEntry {
    request_id: String,
    response: MlxProviderResponse,
}

#[cfg(test)]
mod tests {
    use super::{parse_build_suite, parse_endpoint, parse_served_fixture, parse_text_fixture};
    use psionic_mlx_vlm::MlxVlmServedEndpoint;
    use std::path::PathBuf;

    #[test]
    fn parse_build_suite_requires_spec_path() {
        let parsed = parse_build_suite(["--spec-json".to_string(), "suite.json".to_string()])
            .expect("build-suite parse");
        assert_eq!(parsed.spec_json, PathBuf::from("suite.json"));
        assert!(parsed.json_out.is_none());
    }

    #[test]
    fn parse_text_fixture_accepts_output_path() {
        let parsed = parse_text_fixture([
            "--spec-json".to_string(),
            "suite.json".to_string(),
            "--responses-json".to_string(),
            "responses.json".to_string(),
            "--json-out".to_string(),
            "receipt.json".to_string(),
        ])
        .expect("text fixture parse");
        assert_eq!(parsed.spec_json, PathBuf::from("suite.json"));
        assert_eq!(parsed.responses_json, PathBuf::from("responses.json"));
        assert_eq!(parsed.json_out, Some(PathBuf::from("receipt.json")));
    }

    #[test]
    fn parse_served_fixture_accepts_chat_endpoint() {
        let parsed = parse_served_fixture([
            "--spec-json".to_string(),
            "suite.json".to_string(),
            "--responses-json".to_string(),
            "responses.json".to_string(),
            "--model-reference".to_string(),
            "hf:openagents/demo".to_string(),
            "--endpoint".to_string(),
            "chat".to_string(),
        ])
        .expect("served fixture parse");
        assert_eq!(parsed.endpoint, MlxVlmServedEndpoint::ChatCompletions);
        assert_eq!(parsed.model_reference, "hf:openagents/demo");
    }

    #[test]
    fn parse_endpoint_rejects_unknown_values() {
        let error = parse_endpoint("other").expect_err("invalid endpoint");
        assert!(error.contains("invalid --endpoint value"));
    }
}
