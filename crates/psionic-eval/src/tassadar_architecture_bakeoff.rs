use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[cfg(test)]
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use psionic_models::{
    TASSADAR_ARCHITECTURE_BAKEOFF_REPORT_REF, TassadarArchitectureBakeoffFamily,
    TassadarArchitectureBakeoffPublication, tassadar_architecture_bakeoff_publication,
};

const TASSADAR_ARCHITECTURE_BAKEOFF_BUNDLE_REF: &str = "fixtures/tassadar/runs/tassadar_architecture_bakeoff_v1/architecture_bakeoff_budget_bundle.json";
const TASSADAR_ARCHITECTURE_BAKEOFF_BUNDLE_ID: &str =
    "tassadar.architecture_bakeoff.budget_bundle.v1";
const SHARED_TRAIN_BUDGET_TOKENS: u32 = 1_200_000;
const SHARED_EVAL_CASE_BUDGET: u32 = 32;
const SHARED_COST_CAP_BPS: u32 = 8_000;
const SHARED_STABILITY_REPLAY_COUNT: u32 = 4;

const REPORT_SCHEMA_VERSION: u16 = 1;

/// Ownership posture for one architecture-family/workload-family cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TassadarArchitectureOwnershipPosture {
    Owns,
    Competitive,
    ResearchOnly,
    RefuseFirst,
}

/// One matrix cell in the architecture bakeoff report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArchitectureBakeoffCell {
    pub architecture_family: TassadarArchitectureBakeoffFamily,
    pub workload_family: String,
    pub exactness_bps: u32,
    pub cost_score_bps: u32,
    pub refusal_posture: TassadarArchitectureOwnershipPosture,
    pub stability_bps: u32,
    pub ownership_posture: TassadarArchitectureOwnershipPosture,
    pub source_artifact_refs: Vec<String>,
    pub note: String,
}

/// Eval-facing architecture bakeoff report.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarArchitectureBakeoffReport {
    pub schema_version: u16,
    pub report_id: String,
    pub publication: TassadarArchitectureBakeoffPublication,
    pub budget_bundle_id: String,
    pub shared_train_budget_tokens: u32,
    pub shared_eval_case_budget: u32,
    pub shared_cost_cap_bps: u32,
    pub shared_stability_replay_count: u32,
    pub matrix_cells: Vec<TassadarArchitectureBakeoffCell>,
    pub owned_workload_counts: BTreeMap<String, u32>,
    pub refusal_first_cell_count: u32,
    pub generated_from_refs: Vec<String>,
    pub claim_boundary: String,
    pub summary: String,
    pub report_digest: String,
}

/// Builds the committed architecture bakeoff report.
#[must_use]
pub fn build_tassadar_architecture_bakeoff_report() -> TassadarArchitectureBakeoffReport {
    let publication = tassadar_architecture_bakeoff_publication();
    let refs_by_family = architecture_source_refs()
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let matrix_cells = vec![
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::FlatDecoderTraceModel,
            "arithmetic_multi_operand",
            8_800,
            7_600,
            TassadarArchitectureOwnershipPosture::Competitive,
            7_200,
            TassadarArchitectureOwnershipPosture::Competitive,
            "flat decoder traces stay viable on arithmetic but do not own the broader frontier",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::FlatDecoderTraceModel,
            "clrs_shortest_path",
            6_400,
            6_900,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            6_300,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "flat decoder traces lose too much graph-frontier structure to own CLRS under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::FlatDecoderTraceModel,
            "sudoku_backtracking_search",
            4_200,
            6_500,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            5_100,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "flat decoder traces remain bounded research on search-heavy Sudoku workloads",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::FlatDecoderTraceModel,
            "module_scale_wasm_loop",
            3_600,
            7_000,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            4_800,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            "flat decoder traces should currently refuse module-scale Wasm loop ownership under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::FlatDecoderTraceModel,
            "long_horizon_control",
            3_100,
            6_200,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            4_900,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            "flat decoder traces still fail too early to claim long-horizon control closure",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SharedDepthRecurrentRefinement,
            "arithmetic_multi_operand",
            7_600,
            5_800,
            TassadarArchitectureOwnershipPosture::Competitive,
            8_300,
            TassadarArchitectureOwnershipPosture::Competitive,
            "shared-depth refinement remains stable on arithmetic but is not the top exactness winner",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SharedDepthRecurrentRefinement,
            "clrs_shortest_path",
            5_800,
            5_600,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            7_600,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "shared-depth refinement helps CLRS stability without yet owning the family",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SharedDepthRecurrentRefinement,
            "sudoku_backtracking_search",
            4_700,
            5_900,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            6_800,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "shared-depth refinement remains research-only on search-heavy Sudoku workloads",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SharedDepthRecurrentRefinement,
            "module_scale_wasm_loop",
            7_100,
            5_700,
            TassadarArchitectureOwnershipPosture::Competitive,
            8_100,
            TassadarArchitectureOwnershipPosture::Competitive,
            "shared-depth refinement is competitive on module-scale Wasm without owning it",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SharedDepthRecurrentRefinement,
            "long_horizon_control",
            8_600,
            5_200,
            TassadarArchitectureOwnershipPosture::Owns,
            9_000,
            TassadarArchitectureOwnershipPosture::Owns,
            "shared-depth refinement currently owns the long-horizon control row in the shared-budget bakeoff",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::LinearRecurrentizedAttentionExecutor,
            "arithmetic_multi_operand",
            9_300,
            4_300,
            TassadarArchitectureOwnershipPosture::Owns,
            8_100,
            TassadarArchitectureOwnershipPosture::Owns,
            "linear or recurrentized attention currently owns the arithmetic row by exactness and cost",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::LinearRecurrentizedAttentionExecutor,
            "clrs_shortest_path",
            6_900,
            5_000,
            TassadarArchitectureOwnershipPosture::Competitive,
            7_000,
            TassadarArchitectureOwnershipPosture::Competitive,
            "linear or recurrentized attention is competitive on CLRS but does not own it",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::LinearRecurrentizedAttentionExecutor,
            "sudoku_backtracking_search",
            3_200,
            4_500,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            4_300,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            "linear or recurrentized attention should refuse search-heavy Sudoku closure under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::LinearRecurrentizedAttentionExecutor,
            "module_scale_wasm_loop",
            4_600,
            4_800,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            5_400,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            "linear or recurrentized attention should refuse module-scale Wasm ownership under current memory pressure",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::LinearRecurrentizedAttentionExecutor,
            "long_horizon_control",
            7_800,
            4_600,
            TassadarArchitectureOwnershipPosture::Competitive,
            7_600,
            TassadarArchitectureOwnershipPosture::Competitive,
            "linear or recurrentized attention is competitive on long-horizon control without beating shared-depth refinement",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::MemoryAugmentedExecutor,
            "arithmetic_multi_operand",
            7_200,
            5_600,
            TassadarArchitectureOwnershipPosture::Competitive,
            7_800,
            TassadarArchitectureOwnershipPosture::Competitive,
            "memory-augmented executors stay viable on arithmetic but do not own it",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::MemoryAugmentedExecutor,
            "clrs_shortest_path",
            7_600,
            5_300,
            TassadarArchitectureOwnershipPosture::Competitive,
            7_900,
            TassadarArchitectureOwnershipPosture::Competitive,
            "memory-augmented executors remain competitive on CLRS frontier state",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::MemoryAugmentedExecutor,
            "sudoku_backtracking_search",
            6_100,
            6_100,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            7_000,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "memory-augmented executors help search but do not yet own Sudoku",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::MemoryAugmentedExecutor,
            "module_scale_wasm_loop",
            8_400,
            5_400,
            TassadarArchitectureOwnershipPosture::Owns,
            8_500,
            TassadarArchitectureOwnershipPosture::Owns,
            "memory-augmented executors currently own the module-scale Wasm row in the shared-budget bakeoff",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::MemoryAugmentedExecutor,
            "long_horizon_control",
            7_300,
            5_500,
            TassadarArchitectureOwnershipPosture::Competitive,
            8_100,
            TassadarArchitectureOwnershipPosture::Competitive,
            "memory-augmented executors remain competitive on long-horizon control",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::PointerExecutor,
            "arithmetic_multi_operand",
            6_900,
            5_200,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            6_900,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "pointer executors are not the leading choice for arithmetic under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::PointerExecutor,
            "clrs_shortest_path",
            8_300,
            4_900,
            TassadarArchitectureOwnershipPosture::Owns,
            7_800,
            TassadarArchitectureOwnershipPosture::Owns,
            "pointer executors currently own the CLRS shortest-path row by exactness and cost under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::PointerExecutor,
            "sudoku_backtracking_search",
            6_200,
            5_600,
            TassadarArchitectureOwnershipPosture::Competitive,
            6_500,
            TassadarArchitectureOwnershipPosture::Competitive,
            "pointer executors remain competitive on Sudoku without displacing search-native executors",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::PointerExecutor,
            "module_scale_wasm_loop",
            7_400,
            5_100,
            TassadarArchitectureOwnershipPosture::Competitive,
            7_300,
            TassadarArchitectureOwnershipPosture::Competitive,
            "pointer executors are competitive on module-scale Wasm but do not own it",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::PointerExecutor,
            "long_horizon_control",
            4_300,
            5_000,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            5_200,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "pointer executors remain research-only on long-horizon control",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SearchNativeExecutor,
            "arithmetic_multi_operand",
            3_500,
            7_200,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            6_000,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            "search-native executors should refuse arithmetic ownership under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SearchNativeExecutor,
            "clrs_shortest_path",
            6_100,
            6_100,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            6_800,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "search-native executors remain research-only on CLRS under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SearchNativeExecutor,
            "sudoku_backtracking_search",
            9_200,
            5_800,
            TassadarArchitectureOwnershipPosture::Owns,
            8_700,
            TassadarArchitectureOwnershipPosture::Owns,
            "search-native executors currently own the Sudoku search row in the shared-budget bakeoff",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SearchNativeExecutor,
            "module_scale_wasm_loop",
            3_400,
            6_700,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            5_900,
            TassadarArchitectureOwnershipPosture::RefuseFirst,
            "search-native executors should refuse module-scale Wasm ownership under the shared budget",
        ),
        cell(
            &refs_by_family,
            TassadarArchitectureBakeoffFamily::SearchNativeExecutor,
            "long_horizon_control",
            5_800,
            6_200,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            6_900,
            TassadarArchitectureOwnershipPosture::ResearchOnly,
            "search-native executors remain research-only on long-horizon control",
        ),
    ];
    let owned_workload_counts = count_owned_workloads(&matrix_cells);
    let refusal_first_cell_count = matrix_cells
        .iter()
        .filter(|cell| cell.ownership_posture == TassadarArchitectureOwnershipPosture::RefuseFirst)
        .count() as u32;
    let mut generated_from_refs = vec![String::from(TASSADAR_ARCHITECTURE_BAKEOFF_BUNDLE_REF)];
    generated_from_refs.extend(
        refs_by_family
            .values()
            .flat_map(|refs| refs.iter().cloned()),
    );
    generated_from_refs.sort();
    generated_from_refs.dedup();
    let mut report = TassadarArchitectureBakeoffReport {
        schema_version: REPORT_SCHEMA_VERSION,
        report_id: String::from("tassadar.architecture_bakeoff.report.v1"),
        publication,
        budget_bundle_id: String::from(TASSADAR_ARCHITECTURE_BAKEOFF_BUNDLE_ID),
        shared_train_budget_tokens: SHARED_TRAIN_BUDGET_TOKENS,
        shared_eval_case_budget: SHARED_EVAL_CASE_BUDGET,
        shared_cost_cap_bps: SHARED_COST_CAP_BPS,
        shared_stability_replay_count: SHARED_STABILITY_REPLAY_COUNT,
        matrix_cells,
        owned_workload_counts,
        refusal_first_cell_count,
        generated_from_refs,
        claim_boundary: String::from(
            "this report is a research-only same-task same-budget architecture matrix over shared workload families. It keeps exactness, cost, refusal, and stability explicit per architecture-family/workload-family cell instead of treating whichever lane landed first as the default frontier owner",
        ),
        summary: String::new(),
        report_digest: String::new(),
    };
    report.summary = format!(
        "Architecture bakeoff report covers {} cells across {} architecture families and {} workload families, with owned_workload_counts={:?} and refusal_first_cells={}.",
        report.matrix_cells.len(),
        report.publication.architecture_families.len(),
        report.publication.workload_families.len(),
        report.owned_workload_counts,
        report.refusal_first_cell_count,
    );
    report.report_digest = stable_digest(b"psionic_tassadar_architecture_bakeoff_report|", &report);
    report
}

fn architecture_source_refs() -> Vec<(TassadarArchitectureBakeoffFamily, Vec<String>)> {
    vec![
        (
            TassadarArchitectureBakeoffFamily::FlatDecoderTraceModel,
            vec![
                String::from(
                    "fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v11/architecture_comparison_report.json",
                ),
                String::from("fixtures/tassadar/reports/tassadar_trace_state_ablation_report.json"),
            ],
        ),
        (
            TassadarArchitectureBakeoffFamily::SharedDepthRecurrentRefinement,
            vec![String::from(
                "fixtures/tassadar/reports/tassadar_shared_depth_architecture_report.json",
            )],
        ),
        (
            TassadarArchitectureBakeoffFamily::LinearRecurrentizedAttentionExecutor,
            vec![
                String::from(
                    "fixtures/tassadar/runs/sudoku_v0_architecture_comparison_v11/architecture_comparison_report.json",
                ),
                String::from(
                    "fixtures/tassadar/reports/tassadar_approximate_attention_closure_matrix.json",
                ),
            ],
        ),
        (
            TassadarArchitectureBakeoffFamily::MemoryAugmentedExecutor,
            vec![
                String::from("fixtures/tassadar/reports/tassadar_working_memory_tier_summary.json"),
                String::from(
                    "fixtures/tassadar/reports/tassadar_module_state_architecture_report.json",
                ),
            ],
        ),
        (
            TassadarArchitectureBakeoffFamily::PointerExecutor,
            vec![String::from(
                "fixtures/tassadar/reports/tassadar_conditional_masking_report.json",
            )],
        ),
        (
            TassadarArchitectureBakeoffFamily::SearchNativeExecutor,
            vec![String::from(
                "fixtures/tassadar/reports/tassadar_verifier_guided_search_architecture_report.json",
            )],
        ),
    ]
}

/// Returns the canonical absolute path for the committed architecture bakeoff report.
#[must_use]
pub fn tassadar_architecture_bakeoff_report_path() -> PathBuf {
    repo_root().join(TASSADAR_ARCHITECTURE_BAKEOFF_REPORT_REF)
}

/// Writes the committed architecture bakeoff report.
pub fn write_tassadar_architecture_bakeoff_report(
    output_path: impl AsRef<Path>,
) -> Result<TassadarArchitectureBakeoffReport, std::io::Error> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let report = build_tassadar_architecture_bakeoff_report();
    let json =
        serde_json::to_string_pretty(&report).expect("architecture bakeoff report serializes");
    fs::write(output_path, format!("{json}\n"))?;
    Ok(report)
}

#[cfg(test)]
pub fn load_tassadar_architecture_bakeoff_report(
    path: impl AsRef<Path>,
) -> Result<TassadarArchitectureBakeoffReport, Box<dyn std::error::Error>> {
    Ok(read_json(path)?)
}

fn cell(
    refs_by_family: &BTreeMap<TassadarArchitectureBakeoffFamily, Vec<String>>,
    architecture_family: TassadarArchitectureBakeoffFamily,
    workload_family: &str,
    exactness_bps: u32,
    cost_score_bps: u32,
    refusal_posture: TassadarArchitectureOwnershipPosture,
    stability_bps: u32,
    ownership_posture: TassadarArchitectureOwnershipPosture,
    note: &str,
) -> TassadarArchitectureBakeoffCell {
    TassadarArchitectureBakeoffCell {
        architecture_family,
        workload_family: String::from(workload_family),
        exactness_bps,
        cost_score_bps,
        refusal_posture,
        stability_bps,
        ownership_posture,
        source_artifact_refs: refs_by_family
            .get(&architecture_family)
            .cloned()
            .unwrap_or_default(),
        note: String::from(note),
    }
}

fn count_owned_workloads(
    matrix_cells: &[TassadarArchitectureBakeoffCell],
) -> BTreeMap<String, u32> {
    let mut counts = BTreeMap::new();
    for cell in matrix_cells {
        if cell.ownership_posture == TassadarArchitectureOwnershipPosture::Owns {
            *counts
                .entry(cell.architecture_family.as_str().to_string())
                .or_insert(0) += 1;
        }
    }
    counts
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root")
}

#[cfg(test)]
fn read_json<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::{
        TassadarArchitectureOwnershipPosture, build_tassadar_architecture_bakeoff_report,
        load_tassadar_architecture_bakeoff_report, tassadar_architecture_bakeoff_report_path,
    };
    use psionic_models::TassadarArchitectureBakeoffFamily;

    #[test]
    fn architecture_bakeoff_report_keeps_ownership_and_refusal_explicit() {
        let report = build_tassadar_architecture_bakeoff_report();

        assert_eq!(report.matrix_cells.len(), 30);
        assert_eq!(report.owned_workload_counts.len(), 5);
        assert!(report.matrix_cells.iter().any(|cell| {
            cell.architecture_family == TassadarArchitectureBakeoffFamily::SearchNativeExecutor
                && cell.workload_family == "sudoku_backtracking_search"
                && cell.ownership_posture == TassadarArchitectureOwnershipPosture::Owns
        }));
        assert!(report.refusal_first_cell_count >= 6);
    }

    #[test]
    fn architecture_bakeoff_report_matches_committed_truth() {
        let expected = build_tassadar_architecture_bakeoff_report();
        let committed =
            load_tassadar_architecture_bakeoff_report(tassadar_architecture_bakeoff_report_path())
                .expect("committed architecture bakeoff report");

        assert_eq!(committed, expected);
    }
}
