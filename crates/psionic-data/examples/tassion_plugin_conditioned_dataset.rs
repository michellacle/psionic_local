use psionic_data::{
    tassion_plugin_conditioned_dataset_bundle_path, write_tassion_plugin_conditioned_dataset_bundle,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output_path = tassion_plugin_conditioned_dataset_bundle_path();
    let bundle = write_tassion_plugin_conditioned_dataset_bundle(&output_path)?;
    println!(
        "wrote {} with digest {}",
        output_path.display(),
        bundle.bundle_digest
    );
    println!("dataset identity: {}", bundle.stable_dataset_identity);
    println!("split rows: {}", bundle.split_rows.len());
    Ok(())
}
