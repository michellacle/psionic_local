use std::{
    env, fs,
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
};

use psionic_catalog::{OllamaRegistryPullOptions, RegistryScheme};
use psionic_mlx_catalog::{MlxCatalogRoots, MlxCatalogWorkspace, MlxMetadataTrustMode, MlxRemoteMetadataPolicy};
use serde::Serialize;

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
        "resolve" => run_resolve(args),
        "discover-ollama" => run_discover_ollama(args),
        "discover-hf" => run_discover_hf(args),
        "load-text" => run_load_text(args),
        "pull-ollama" => run_pull_ollama(args),
        "-h" | "--help" => Err(CliError::Usage(usage())),
        other => Err(CliError::Message(format!(
            "unrecognized subcommand `{other}`\n\n{}",
            usage()
        ))),
    }
}

fn run_resolve(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_common_with_reference(args).map_err(CliError::Message)?;
    let workspace = parsed.workspace();
    let report = workspace
        .resolve(&parsed.reference, &parsed.policy())
        .map_err(|error| CliError::Message(format!("failed to resolve `{}`: {error}", parsed.reference)))?;
    write_json_output(&report, parsed.json_out()).map_err(CliError::Message)
}

fn run_discover_ollama(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_common(args).map_err(CliError::Message)?;
    let discovery = parsed
        .workspace()
        .discover_ollama_models()
        .map_err(|error| CliError::Message(format!("failed to discover ollama models: {error}")))?;
    write_json_output(&discovery, parsed.json_out).map_err(CliError::Message)
}

fn run_discover_hf(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_common_with_repo(args).map_err(CliError::Message)?;
    let snapshots = parsed
        .workspace()
        .discover_hugging_face_snapshots(&parsed.repo_id, &parsed.policy())
        .map_err(|error| CliError::Message(format!("failed to discover HF snapshots: {error}")))?;
    write_json_output(&snapshots, parsed.json_out()).map_err(CliError::Message)
}

fn run_load_text(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_common_with_reference(args).map_err(CliError::Message)?;
    let report = parsed
        .workspace()
        .load_text_report(&parsed.reference, &parsed.policy())
        .map_err(|error| CliError::Message(format!("failed to load text runtime: {error}")))?;
    write_json_output(&report, parsed.json_out()).map_err(CliError::Message)
}

fn run_pull_ollama(args: impl IntoIterator<Item = String>) -> Result<(), CliError> {
    let parsed = parse_pull_ollama(args).map_err(CliError::Message)?;
    let report = parsed
        .workspace()
        .pull_ollama_model(
            &parsed.reference,
            OllamaRegistryPullOptions {
                scheme: parsed.registry_scheme,
                auth: None,
                user_agent: Some(String::from("psionic-mlx-catalog")),
            },
        )
        .map_err(|error| CliError::Message(format!("failed to pull ollama model: {error}")))?;
    write_json_output(&report, parsed.common.json_out.clone()).map_err(CliError::Message)
}

#[derive(Clone, Debug)]
struct CommonArgs {
    ollama_root: Option<PathBuf>,
    hf_root: Option<PathBuf>,
    allow_processor_metadata: bool,
    allow_template_metadata: bool,
    json_out: Option<PathBuf>,
}

impl CommonArgs {
    fn workspace(&self) -> MlxCatalogWorkspace {
        MlxCatalogWorkspace::new(MlxCatalogRoots {
            ollama_models_root: self
                .ollama_root
                .clone()
                .unwrap_or_else(psionic_mlx_catalog::default_ollama_models_root),
            hugging_face_hub_root: self
                .hf_root
                .clone()
                .unwrap_or_else(psionic_mlx_catalog::default_hugging_face_hub_root),
        })
    }

    fn policy(&self) -> MlxRemoteMetadataPolicy {
        MlxRemoteMetadataPolicy {
            processor_metadata: if self.allow_processor_metadata {
                MlxMetadataTrustMode::AllowDigestBoundLocalCache
            } else {
                MlxMetadataTrustMode::Refuse
            },
            template_metadata: if self.allow_template_metadata {
                MlxMetadataTrustMode::AllowDigestBoundLocalCache
            } else {
                MlxMetadataTrustMode::Refuse
            },
        }
    }
}

#[derive(Clone, Debug)]
struct CommonWithReferenceArgs {
    common: CommonArgs,
    reference: String,
}

impl CommonWithReferenceArgs {
    fn workspace(&self) -> MlxCatalogWorkspace {
        self.common.workspace()
    }

    fn policy(&self) -> MlxRemoteMetadataPolicy {
        self.common.policy()
    }

    fn json_out(&self) -> Option<PathBuf> {
        self.common.json_out.clone()
    }
}

#[derive(Clone, Debug)]
struct CommonWithRepoArgs {
    common: CommonArgs,
    repo_id: String,
}

impl CommonWithRepoArgs {
    fn workspace(&self) -> MlxCatalogWorkspace {
        self.common.workspace()
    }

    fn policy(&self) -> MlxRemoteMetadataPolicy {
        self.common.policy()
    }

    fn json_out(&self) -> Option<PathBuf> {
        self.common.json_out.clone()
    }
}

#[derive(Clone, Debug)]
struct PullOllamaArgs {
    common: CommonArgs,
    reference: String,
    registry_scheme: RegistryScheme,
}

impl PullOllamaArgs {
    fn workspace(&self) -> MlxCatalogWorkspace {
        self.common.workspace()
    }
}

fn parse_common(args: impl IntoIterator<Item = String>) -> Result<CommonArgs, String> {
    let mut ollama_root = None;
    let mut hf_root = None;
    let mut allow_processor_metadata = false;
    let mut allow_template_metadata = false;
    let mut json_out = None;

    let mut args = args.into_iter();
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--ollama-root" => ollama_root = Some(PathBuf::from(next_value(&mut args, "--ollama-root")?)),
            "--hf-root" => hf_root = Some(PathBuf::from(next_value(&mut args, "--hf-root")?)),
            "--allow-processor-metadata" => allow_processor_metadata = true,
            "--allow-template-metadata" => allow_template_metadata = true,
            "--json-out" => json_out = Some(PathBuf::from(next_value(&mut args, "--json-out")?)),
            "-h" | "--help" => return Err(usage()),
            other => {
                return Err(format!(
                    "unrecognized argument `{other}`\n\n{}",
                    usage()
                ))
            }
        }
    }

    Ok(CommonArgs {
        ollama_root,
        hf_root,
        allow_processor_metadata,
        allow_template_metadata,
        json_out,
    })
}

fn parse_common_with_reference(
    args: impl IntoIterator<Item = String>,
) -> Result<CommonWithReferenceArgs, String> {
    let mut raw_args = Vec::new();
    let mut reference = None;
    let mut iter = args.into_iter();
    while let Some(argument) = iter.next() {
        if argument == "--reference" {
            reference = Some(next_value(&mut iter, "--reference")?);
        } else {
            raw_args.push(argument);
        }
    }
    let Some(reference) = reference else {
        return Err(format!("missing required `--reference`\n\n{}", usage()));
    };
    Ok(CommonWithReferenceArgs {
        common: parse_common(raw_args)?,
        reference,
    })
}

fn parse_common_with_repo(args: impl IntoIterator<Item = String>) -> Result<CommonWithRepoArgs, String> {
    let mut raw_args = Vec::new();
    let mut repo_id = None;
    let mut iter = args.into_iter();
    while let Some(argument) = iter.next() {
        if argument == "--repo" {
            repo_id = Some(next_value(&mut iter, "--repo")?);
        } else {
            raw_args.push(argument);
        }
    }
    let Some(repo_id) = repo_id else {
        return Err(format!("missing required `--repo`\n\n{}", usage()));
    };
    Ok(CommonWithRepoArgs {
        common: parse_common(raw_args)?,
        repo_id,
    })
}

fn parse_pull_ollama(args: impl IntoIterator<Item = String>) -> Result<PullOllamaArgs, String> {
    let mut raw_args = Vec::new();
    let mut reference = None;
    let mut registry_scheme = RegistryScheme::Https;
    let mut iter = args.into_iter();
    while let Some(argument) = iter.next() {
        match argument.as_str() {
            "--reference" => reference = Some(next_value(&mut iter, "--reference")?),
            "--registry-scheme" => {
                registry_scheme = match next_value(&mut iter, "--registry-scheme")?.as_str() {
                    "https" => RegistryScheme::Https,
                    "http" => RegistryScheme::Http,
                    other => {
                        return Err(format!(
                            "invalid --registry-scheme value `{other}` (expected http or https)\n\n{}",
                            usage()
                        ))
                    }
                };
            }
            other => raw_args.push(String::from(other)),
        }
    }
    let Some(reference) = reference else {
        return Err(format!("missing required `--reference`\n\n{}", usage()));
    };
    Ok(PullOllamaArgs {
        common: parse_common(raw_args)?,
        reference,
        registry_scheme,
    })
}

fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for `{flag}`"))
}

fn write_json_output<T: Serialize>(value: &T, output: Option<PathBuf>) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value)
        .map_err(|error| format!("failed to serialize JSON output: {error}"))?;
    if let Some(path) = output {
        fs::write(&path, format!("{json}\n"))
            .map_err(|error| format!("failed to write `{}`: {error}", path.display()))?;
        return Ok(());
    }
    let mut stdout = io::stdout().lock();
    stdout
        .write_all(json.as_bytes())
        .map_err(|error| format!("failed to write stdout: {error}"))?;
    stdout
        .write_all(b"\n")
        .map_err(|error| format!("failed to terminate stdout JSON: {error}"))
}

fn usage() -> String {
    String::from(
        "usage:\n  psionic-mlx-catalog resolve --reference <ref> [--ollama-root <path>] [--hf-root <path>] [--allow-processor-metadata] [--allow-template-metadata] [--json-out <path>]\n  psionic-mlx-catalog discover-ollama [--ollama-root <path>] [--json-out <path>]\n  psionic-mlx-catalog discover-hf --repo <owner/repo> [--hf-root <path>] [--allow-processor-metadata] [--allow-template-metadata] [--json-out <path>]\n  psionic-mlx-catalog load-text --reference <ref> [common flags]\n  psionic-mlx-catalog pull-ollama --reference <model> [--registry-scheme http|https] [--ollama-root <path>] [--json-out <path>]",
    )
}

#[cfg(test)]
mod tests {
    use super::{parse_common_with_reference, parse_common_with_repo, parse_pull_ollama};
    use psionic_catalog::RegistryScheme;

    #[test]
    fn parse_resolve_accepts_metadata_flags() {
        let parsed = parse_common_with_reference(
            vec![
                "--reference",
                "hf:mlx-community/tiny-qwen",
                "--allow-template-metadata",
                "--hf-root",
                "/tmp/hf",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("resolve args");
        assert_eq!(parsed.reference, "hf:mlx-community/tiny-qwen");
        assert!(parsed.common.allow_template_metadata);
        assert_eq!(
            parsed.common.hf_root.as_ref().map(|value| value.to_string_lossy().into_owned()),
            Some(String::from("/tmp/hf"))
        );
    }

    #[test]
    fn parse_discover_hf_requires_repo() {
        let error = parse_common_with_repo(Vec::<String>::new()).expect_err("missing repo");
        assert!(error.contains("missing required `--repo`"));
    }

    #[test]
    fn parse_pull_ollama_accepts_scheme_override() {
        let parsed = parse_pull_ollama(
            vec![
                "--reference",
                "qwen2",
                "--registry-scheme",
                "http",
            ]
            .into_iter()
            .map(String::from),
        )
        .expect("pull args");
        assert_eq!(parsed.reference, "qwen2");
        assert_eq!(parsed.registry_scheme, RegistryScheme::Http);
    }
}
