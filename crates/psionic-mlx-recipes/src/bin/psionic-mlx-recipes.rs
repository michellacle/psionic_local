use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use psionic_environments::EnvironmentPackageKey;
use psionic_mlx_recipes::{MlxAdapterRecipe, MlxRecipeConfig, MlxRecipeMethod, MlxRecipeWorkspace};

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
        "plan" => run_plan(args),
        "methods" => run_methods(),
        "-h" | "--help" => Err(CliError::Usage(usage())),
        other => Err(CliError::Message(format!(
            "unrecognized subcommand `{other}`\n\n{}",
            usage()
        ))),
    }
}

fn run_plan(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_plan(args).map_err(CliError::Message)?;
    let mut config = MlxRecipeConfig::new(
        parsed.run_id,
        parsed.cluster_id,
        parsed.checkpoint_family,
        parsed.environment,
        parsed.method,
    )
    .map_err(|error| CliError::Message(format!("invalid recipe config: {error}")))?;
    config.budget = psionic_train::TrainingLoopBudget::new(
        parsed.max_steps,
        parsed.steps_per_window,
        parsed.windows_per_cadence,
    )
    .map_err(|error| CliError::Message(format!("invalid budget: {error}")))?;
    if let Some(adapter) = parsed.adapter {
        config = config.with_adapter(adapter);
    }
    let plan = MlxRecipeWorkspace::default()
        .plan(&config)
        .map_err(|error| CliError::Message(format!("planning failed: {error}")))?;
    write_json_output(&plan, parsed.json_out).map_err(CliError::Message)
}

fn run_methods() -> Result<(), CliError> {
    write_json_output(&MlxRecipeWorkspace::default().methods(), None).map_err(CliError::Message)
}

#[derive(Clone, Debug)]
struct PlanArgs {
    run_id: String,
    cluster_id: String,
    checkpoint_family: String,
    environment: EnvironmentPackageKey,
    method: MlxRecipeMethod,
    adapter: Option<MlxAdapterRecipe>,
    max_steps: u64,
    steps_per_window: u64,
    windows_per_cadence: u64,
    json_out: Option<PathBuf>,
}

fn parse_plan(args: impl IntoIterator<Item = String>) -> Result<PlanArgs, String> {
    let mut run_id = None;
    let mut cluster_id = String::from("cluster-mlx");
    let mut checkpoint_family = String::from("mlx.recipe.checkpoint");
    let mut environment = None;
    let mut method = None;
    let mut adapter_rank = None;
    let mut adapter_alpha = None;
    let mut adapter_quantization = None;
    let mut max_steps = 64_u64;
    let mut steps_per_window = 8_u64;
    let mut windows_per_cadence = 4_u64;
    let mut json_out = None;
    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--run-id" => run_id = Some(next_value(&mut args, "--run-id")?),
            "--cluster-id" => cluster_id = next_value(&mut args, "--cluster-id")?,
            "--checkpoint-family" => {
                checkpoint_family = next_value(&mut args, "--checkpoint-family")?
            }
            "--environment" => {
                environment = Some(parse_environment_ref(next_value(
                    &mut args,
                    "--environment",
                )?)?)
            }
            "--method" => method = Some(parse_method(next_value(&mut args, "--method")?.as_str())?),
            "--adapter-rank" => {
                adapter_rank = Some(
                    next_value(&mut args, "--adapter-rank")?
                        .parse()
                        .map_err(|error| format!("invalid --adapter-rank value: {error}"))?,
                )
            }
            "--adapter-alpha" => {
                adapter_alpha = Some(
                    next_value(&mut args, "--adapter-alpha")?
                        .parse()
                        .map_err(|error| format!("invalid --adapter-alpha value: {error}"))?,
                )
            }
            "--adapter-quantization" => {
                adapter_quantization = Some(next_value(&mut args, "--adapter-quantization")?)
            }
            "--max-steps" => {
                max_steps = next_value(&mut args, "--max-steps")?
                    .parse()
                    .map_err(|error| format!("invalid --max-steps value: {error}"))?;
            }
            "--steps-per-window" => {
                steps_per_window = next_value(&mut args, "--steps-per-window")?
                    .parse()
                    .map_err(|error| format!("invalid --steps-per-window value: {error}"))?;
            }
            "--windows-per-cadence" => {
                windows_per_cadence = next_value(&mut args, "--windows-per-cadence")?
                    .parse()
                    .map_err(|error| format!("invalid --windows-per-cadence value: {error}"))?;
            }
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => return Err(format!("unrecognized argument `{other}`\n\n{}", usage())),
        }
    }
    let Some(run_id) = run_id else {
        return Err(format!("missing required `--run-id`\n\n{}", usage()));
    };
    let Some(environment) = environment else {
        return Err(format!("missing required `--environment`\n\n{}", usage()));
    };
    let Some(method) = method else {
        return Err(format!("missing required `--method`\n\n{}", usage()));
    };
    let adapter = adapter_rank
        .zip(adapter_alpha)
        .map(|(rank, alpha)| MlxAdapterRecipe {
            method,
            rank,
            alpha,
            quantization: adapter_quantization,
        });
    Ok(PlanArgs {
        run_id,
        cluster_id,
        checkpoint_family,
        environment,
        method,
        adapter,
        max_steps,
        steps_per_window,
        windows_per_cadence,
        json_out,
    })
}

fn parse_method(value: &str) -> Result<MlxRecipeMethod, String> {
    match value {
        "sft" => Ok(MlxRecipeMethod::Sft),
        "lora" => Ok(MlxRecipeMethod::Lora),
        "dora" => Ok(MlxRecipeMethod::Dora),
        "qlora" => Ok(MlxRecipeMethod::Qlora),
        "dpo" => Ok(MlxRecipeMethod::Dpo),
        "cpo" => Ok(MlxRecipeMethod::Cpo),
        "orpo" => Ok(MlxRecipeMethod::Orpo),
        "grpo" => Ok(MlxRecipeMethod::Grpo),
        "online_dpo" => Ok(MlxRecipeMethod::OnlineDpo),
        "xpo" => Ok(MlxRecipeMethod::Xpo),
        "ppo" => Ok(MlxRecipeMethod::Ppo),
        other => Err(format!("invalid --method value `{other}`\n\n{}", usage())),
    }
}

fn parse_environment_ref(value: String) -> Result<EnvironmentPackageKey, String> {
    let Some((environment_ref, version)) = value.split_once('@') else {
        return Err(format!(
            "invalid --environment value `{value}` (expected <ref>@<version>)\n\n{}",
            usage()
        ));
    };
    Ok(EnvironmentPackageKey::new(environment_ref, version))
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
        "usage:\n  psionic-mlx-recipes methods\n  psionic-mlx-recipes plan --run-id <id> --environment <ref@version> --method <method> [--cluster-id <id>] [--checkpoint-family <family>] [--adapter-rank <n> --adapter-alpha <f> [--adapter-quantization <label>]] [--max-steps <n>] [--steps-per-window <n>] [--windows-per-cadence <n>] [--json-out <path>]",
    )
}

#[cfg(test)]
mod tests {
    use super::{parse_method, parse_plan, usage};
    use psionic_mlx_recipes::MlxRecipeMethod;

    #[test]
    fn parse_method_accepts_online_dpo() {
        assert_eq!(
            parse_method("online_dpo").expect("method"),
            MlxRecipeMethod::OnlineDpo
        );
    }

    #[test]
    fn parse_plan_requires_environment() {
        let error = parse_plan(
            ["--run-id", "run-1", "--method", "sft"]
                .into_iter()
                .map(String::from),
        )
        .expect_err("missing environment");
        assert!(error.contains("missing required `--environment`"));
        assert!(error.contains(&usage()));
    }
}
