use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use psionic_mlx_vlm::{MlxVlmMessage, MlxVlmServedEndpoint, MlxVlmWorkspace};

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
        "shape" => run_shape(args),
        "plan-request" => run_plan_request(args),
        "-h" | "--help" => Err(CliError::Usage(usage())),
        other => Err(CliError::Message(format!(
            "unrecognized subcommand `{other}`\n\n{}",
            usage()
        ))),
    }
}

fn run_shape(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_shape(args).map_err(CliError::Message)?;
    let messages = load_messages(&parsed.messages_json).map_err(CliError::Message)?;
    let report = MlxVlmWorkspace::default()
        .project_messages(&parsed.family, &messages)
        .map_err(|error| CliError::Message(format!("projection failed: {error}")))?;
    write_json_output(&report, parsed.json_out).map_err(CliError::Message)
}

fn run_plan_request(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_plan(args).map_err(CliError::Message)?;
    let messages = load_messages(&parsed.messages_json).map_err(CliError::Message)?;
    let plan = MlxVlmWorkspace::default()
        .plan_request(
            &parsed.family,
            &parsed.model_reference,
            parsed.endpoint,
            &messages,
        )
        .map_err(|error| CliError::Message(format!("request planning failed: {error}")))?;
    write_json_output(&plan, parsed.json_out).map_err(CliError::Message)
}

#[derive(Clone, Debug)]
struct ShapeArgs {
    family: String,
    messages_json: PathBuf,
    json_out: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct PlanArgs {
    family: String,
    model_reference: String,
    endpoint: MlxVlmServedEndpoint,
    messages_json: PathBuf,
    json_out: Option<PathBuf>,
}

fn parse_shape(args: impl IntoIterator<Item = String>) -> Result<ShapeArgs, String> {
    let mut family = None;
    let mut messages_json = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--family" => family = Some(next_value(&mut args, "--family")?),
            "--messages-json" => {
                messages_json = Some(PathBuf::from(next_value(&mut args, "--messages-json")?))
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(family) = family else {
        return Err(format!("missing required `--family`\n\n{}", usage()));
    };
    let Some(messages_json) = messages_json else {
        return Err(format!("missing required `--messages-json`\n\n{}", usage()));
    };
    Ok(ShapeArgs {
        family,
        messages_json,
        json_out,
    })
}

fn parse_plan(args: impl IntoIterator<Item = String>) -> Result<PlanArgs, String> {
    let mut family = None;
    let mut model_reference = None;
    let mut endpoint = MlxVlmServedEndpoint::Responses;
    let mut messages_json = None;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--family" => family = Some(next_value(&mut args, "--family")?),
            "--model-reference" => {
                model_reference = Some(next_value(&mut args, "--model-reference")?)
            }
            "--endpoint" => {
                endpoint = match next_value(&mut args, "--endpoint")?.as_str() {
                    "responses" => MlxVlmServedEndpoint::Responses,
                    "chat" => MlxVlmServedEndpoint::ChatCompletions,
                    other => {
                        return Err(format!(
                            "invalid --endpoint value `{other}` (expected responses or chat)\n\n{}",
                            usage()
                        ))
                    }
                };
            }
            "--messages-json" => {
                messages_json = Some(PathBuf::from(next_value(&mut args, "--messages-json")?))
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(family) = family else {
        return Err(format!("missing required `--family`\n\n{}", usage()));
    };
    let Some(model_reference) = model_reference else {
        return Err(format!(
            "missing required `--model-reference`\n\n{}",
            usage()
        ));
    };
    let Some(messages_json) = messages_json else {
        return Err(format!("missing required `--messages-json`\n\n{}", usage()));
    };
    Ok(PlanArgs {
        family,
        model_reference,
        endpoint,
        messages_json,
        json_out,
    })
}

fn load_messages(path: &PathBuf) -> Result<Vec<MlxVlmMessage>, String> {
    let json = fs::read_to_string(path)
        .map_err(|error| format!("failed to read `{}`: {error}", path.display()))?;
    serde_json::from_str(&json).map_err(|error| format!("invalid message JSON: {error}"))
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for `{flag}`"))
}

fn write_json_output<T: serde::Serialize>(
    value: &T,
    json_out: Option<PathBuf>,
) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON: {error}"))?;
    match json_out {
        Some(path) => fs::write(path, format!("{json}\n"))
            .map_err(|error| format!("failed to write JSON output: {error}"))?,
        None => {
            let mut stdout = io::stdout().lock();
            stdout
                .write_all(json.as_bytes())
                .map_err(|error| format!("failed to write JSON output: {error}"))?;
            stdout
                .write_all(b"\n")
                .map_err(|error| format!("failed to terminate JSON output: {error}"))?;
        }
    }
    Ok(())
}

fn usage() -> String {
    String::from(
        "usage:\n  psionic-mlx-vlm shape --family <processor> --messages-json <path> [--json-out <path>]\n  psionic-mlx-vlm plan-request --family <processor> --model-reference <ref> [--endpoint responses|chat] --messages-json <path> [--json-out <path>]",
    )
}

#[cfg(test)]
mod tests {
    use super::{parse_plan, parse_shape, usage};

    #[test]
    fn parse_shape_requires_family() {
        let error = parse_shape(["--messages-json", "/tmp/messages.json"].into_iter().map(String::from))
            .expect_err("missing family");
        assert!(error.contains("missing required `--family`"));
        assert!(error.contains(&usage()));
    }

    #[test]
    fn parse_plan_accepts_chat_endpoint() {
        let parsed = parse_plan(
            [
                "--family",
                "omni",
                "--model-reference",
                "hf:mlx-community/omni",
                "--endpoint",
                "chat",
                "--messages-json",
                "/tmp/messages.json",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("parsed");
        assert_eq!(parsed.model_reference, "hf:mlx-community/omni");
    }
}
