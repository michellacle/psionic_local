use psionic_train::{
    compiled_agent_learning_receipt_ledger_fixture_path, compiled_agent_replay_bundle_fixture_path,
    write_compiled_agent_learning_receipt_ledger, write_compiled_agent_replay_bundle,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ledger_path = compiled_agent_learning_receipt_ledger_fixture_path();
    let ledger = write_compiled_agent_learning_receipt_ledger(&ledger_path)?;
    let bundle_path = compiled_agent_replay_bundle_fixture_path();
    let bundle = write_compiled_agent_replay_bundle(&bundle_path)?;
    println!(
        "wrote compiled-agent learning ledger={} digest={}",
        ledger_path.display(),
        ledger.ledger_digest
    );
    println!(
        "wrote compiled-agent replay bundle={} digest={}",
        bundle_path.display(),
        bundle.bundle_digest
    );
    Ok(())
}
