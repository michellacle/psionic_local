use std::{env, path::PathBuf};

use psionic_train::{
    execute_parameter_golf_submission_runtime_entrypoint, ParameterGolfSubmissionRuntimeError,
    ParameterGolfSubmissionRuntimeOutcome,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), ParameterGolfSubmissionRuntimeError> {
    let manifest_arg = env::args()
        .nth(1)
        .unwrap_or_else(|| String::from("runtime/parameter_golf_submission_runtime.json"));
    let root = env::current_dir().map_err(|error| ParameterGolfSubmissionRuntimeError::Read {
        path: String::from("."),
        error,
    })?;
    let manifest_path = if PathBuf::from(&manifest_arg).is_absolute() {
        PathBuf::from(&manifest_arg)
    } else {
        root.join(&manifest_arg)
    };
    match execute_parameter_golf_submission_runtime_entrypoint(&root, &manifest_path)? {
        ParameterGolfSubmissionRuntimeOutcome::LocalReference(receipt) => {
            println!(
                "psionic_non_record_submission_runtime run_id={} runtime_posture={}",
                receipt.run_id, receipt.runtime_posture
            );
            println!(
                "final_int8_zlib_roundtrip_exact val_loss:{:.8} val_bpb:{:.8}",
                receipt.executed_validation.mean_loss, receipt.executed_validation.bits_per_byte
            );
            println!(
                "runtime_consistency bytes_code={} bytes_total={} model_bytes_match_submission={} model_bytes_match_accounting={}",
                receipt.matches_accounting_code_bytes,
                receipt.matches_accounting_total_bytes,
                receipt.matches_submission_model_bytes,
                receipt.matches_accounting_model_bytes,
            );
            Ok(())
        }
        ParameterGolfSubmissionRuntimeOutcome::Distributed8xH100Bringup {
            report_path,
            report,
        } => {
            eprintln!(
                "parameter golf submission runtime does not ship a distributed 8xH100 trainer payload yet; wrote distributed bring-up report to {} with disposition {:?}",
                report_path,
                report.disposition
            );
            eprintln!(
                "psionic_parameter_golf_distributed_8xh100_bringup matching_h100_device_count={} machine_contract_satisfied={}",
                report.matching_h100_device_count, report.machine_contract_satisfied
            );
            if let Some(refusal) = report.refusal {
                eprintln!(
                    "refusal subject={:?} detail={}",
                    refusal.subject, refusal.detail
                );
            }
            Err(ParameterGolfSubmissionRuntimeError::ExecutionMode {
                message: String::from(
                    "the exported folder now ships a Rust-owned distributed 8xH100 bring-up path, but the real distributed trainer payload still has not landed",
                ),
            })
        }
    }
}
