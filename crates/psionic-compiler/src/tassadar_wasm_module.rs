use std::convert::TryFrom;

use psionic_ir::{
    TassadarNormalizedWasmConstExpr, TassadarNormalizedWasmDataMode,
    TassadarNormalizedWasmInstruction, TassadarNormalizedWasmModule,
    TassadarNormalizedWasmModuleError, TassadarNormalizedWasmValueType,
    encode_tassadar_normalized_wasm_module, parse_tassadar_normalized_wasm_module,
};
use psionic_runtime::{
    TassadarCompilerToolchainIdentity, TassadarCpuReferenceRunner, TassadarExecutionRefusal,
    TassadarInstruction, TassadarProgram, TassadarProgramArtifact, TassadarProgramArtifactError,
    TassadarProgramSourceIdentity, TassadarProgramSourceKind, TassadarTraceAbi,
    TassadarWasmProfile, tassadar_trace_abi_for_profile_id,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

const TASSADAR_WASM_MODULE_ARTIFACT_BUNDLE_SCHEMA_VERSION: u16 = 1;
const TASSADAR_WASM_MODULE_COMPILER_FAMILY: &str = "tassadar_wasm_module_lowering";
const TASSADAR_WASM_MODULE_COMPILER_VERSION: &str = "v1";
const TASSADAR_WASM_MODULE_BUNDLE_CLAIM_BOUNDARY: &str = "bounded normalized Wasm module lowering compiles exported zero-parameter functions from the current straight-line core module slice into runnable Tassadar program artifacts; calls, structured control flow, dynamic memory addresses, multi-memory, byte-addressed memory ABI closure, and arbitrary Wasm remain out of scope";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum PendingValue {
    Const(i32),
    Local(u32),
    StackValue,
}

/// Exact execution manifest paired with one lowered module export artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarWasmModuleExportExecutionManifest {
    /// Stable export name.
    pub export_name: String,
    /// Stable function index.
    pub function_index: u32,
    /// Expected emitted outputs on the CPU reference lane.
    pub expected_outputs: Vec<i32>,
    /// Expected final memory image on the CPU reference lane.
    pub expected_final_memory: Vec<i32>,
    /// Stable digest over the manifest.
    pub execution_digest: String,
}

impl TassadarWasmModuleExportExecutionManifest {
    fn new(
        export_name: impl Into<String>,
        function_index: u32,
        expected_outputs: Vec<i32>,
        expected_final_memory: Vec<i32>,
    ) -> Self {
        let mut manifest = Self {
            export_name: export_name.into(),
            function_index,
            expected_outputs,
            expected_final_memory,
            execution_digest: String::new(),
        };
        manifest.execution_digest = stable_digest(
            b"tassadar_wasm_module_export_execution_manifest|",
            &manifest,
        );
        manifest
    }
}

/// One lowered export artifact from the normalized Wasm module lane.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarWasmModuleExportArtifact {
    /// Stable export name.
    pub export_name: String,
    /// Stable function index.
    pub function_index: u32,
    /// Runnable runtime-facing artifact.
    pub program_artifact: TassadarProgramArtifact,
    /// Digest-bound expected execution manifest.
    pub execution_manifest: TassadarWasmModuleExportExecutionManifest,
}

/// Digest-bound artifact bundle produced from one normalized Wasm module.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarWasmModuleArtifactBundle {
    /// Stable schema version.
    pub schema_version: u16,
    /// Stable bundle identifier.
    pub bundle_id: String,
    /// Coarse claim class.
    pub claim_class: String,
    /// Plain-language claim boundary.
    pub claim_boundary: String,
    /// Shared source identity for all lowered exports.
    pub source_identity: TassadarProgramSourceIdentity,
    /// Shared lowering toolchain identity.
    pub toolchain_identity: TassadarCompilerToolchainIdentity,
    /// Normalized module IR bound into the bundle.
    pub normalized_module: TassadarNormalizedWasmModule,
    /// Ordered lowered function exports.
    pub lowered_exports: Vec<TassadarWasmModuleExportArtifact>,
    /// Stable digest over the bundle.
    pub bundle_digest: String,
}

impl TassadarWasmModuleArtifactBundle {
    fn new(
        bundle_id: impl Into<String>,
        source_identity: TassadarProgramSourceIdentity,
        toolchain_identity: TassadarCompilerToolchainIdentity,
        normalized_module: TassadarNormalizedWasmModule,
        lowered_exports: Vec<TassadarWasmModuleExportArtifact>,
    ) -> Self {
        let mut bundle = Self {
            schema_version: TASSADAR_WASM_MODULE_ARTIFACT_BUNDLE_SCHEMA_VERSION,
            bundle_id: bundle_id.into(),
            claim_class: String::from("execution truth / compiled bounded exactness"),
            claim_boundary: String::from(TASSADAR_WASM_MODULE_BUNDLE_CLAIM_BOUNDARY),
            source_identity,
            toolchain_identity,
            normalized_module,
            lowered_exports,
            bundle_digest: String::new(),
        };
        bundle.bundle_digest = stable_digest(b"tassadar_wasm_module_artifact_bundle|", &bundle);
        bundle
    }
}

/// Failure while lowering one normalized Wasm module into runnable runtime
/// artifacts.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TassadarWasmModuleArtifactBundleError {
    /// Parsing or validating the normalized module failed.
    #[error(transparent)]
    Module(#[from] TassadarNormalizedWasmModuleError),
    /// The selected runtime profile has no published trace ABI.
    #[error("no trace ABI is published for Wasm module lowering target `{profile_id}`")]
    UnsupportedTraceAbi {
        /// Runtime profile id.
        profile_id: String,
    },
    /// The module exported no functions to lower.
    #[error("normalized module `{module_digest}` exports no functions")]
    NoFunctionExports {
        /// Module digest.
        module_digest: String,
    },
    /// One function export pointed at an imported function rather than a body.
    #[error("export `{export_name}` points at imported function {function_index}")]
    ExportedImportUnsupported {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
    },
    /// The bounded runtime lane does not yet accept Wasm function parameters.
    #[error(
        "export `{export_name}` function {function_index} declares {param_count} params, but the bounded runtime lane only admits zero-parameter module exports today"
    )]
    UnsupportedParamCount {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Declared param count.
        param_count: usize,
    },
    /// The bounded runtime lane only accepts zero or one `i32` result.
    #[error(
        "export `{export_name}` function {function_index} declares unsupported results {result_types:?}"
    )]
    UnsupportedResultTypes {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Declared result types.
        result_types: Vec<String>,
    },
    /// One lowered local type falls outside the current i32-only runtime.
    #[error(
        "export `{export_name}` function {function_index} declares unsupported local type `{local_type}`"
    )]
    UnsupportedLocalType {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Unsupported local type.
        local_type: String,
    },
    /// One local index does not fit inside the current runtime instruction surface.
    #[error(
        "export `{export_name}` function {function_index} references local {local_index}, but the bounded runtime only encodes locals up to {max_supported}"
    )]
    UnsupportedLocalIndex {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Referenced local index.
        local_index: u32,
        /// Maximum encodable local index.
        max_supported: u8,
    },
    /// One memory layout is outside the current runtime representation.
    #[error("unsupported Wasm memory shape: {detail}")]
    UnsupportedMemoryShape {
        /// Human-readable detail.
        detail: String,
    },
    /// One data segment is outside the current runtime representation.
    #[error("unsupported data segment {data_index}: {detail}")]
    UnsupportedDataSegment {
        /// Data segment index.
        data_index: u32,
        /// Human-readable detail.
        detail: String,
    },
    /// One memory instruction relied on a dynamic address.
    #[error(
        "export `{export_name}` function {function_index} used a dynamic memory address for `{opcode}`; byte-addressed memory ABI closure remains out of scope"
    )]
    UnsupportedDynamicMemoryAddress {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Opcode mnemonic.
        opcode: String,
    },
    /// One memory immediate or address shape is unsupported.
    #[error(
        "export `{export_name}` function {function_index} used unsupported memory form for `{opcode}`: {detail}"
    )]
    UnsupportedMemoryImmediate {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Opcode mnemonic.
        opcode: String,
        /// Human-readable detail.
        detail: String,
    },
    /// One call instruction reached the lowering boundary before call frames land.
    #[error(
        "export `{export_name}` function {function_index} calls function {target_function_index}, but call-frame support is not yet landed"
    )]
    UnsupportedCall {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Callee function index.
        target_function_index: u32,
    },
    /// One parsed Wasm instruction is still outside the bounded runtime lowering slice.
    #[error(
        "export `{export_name}` function {function_index} uses unsupported instruction `{opcode}` for the current runtime lowering slice"
    )]
    UnsupportedInstruction {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Unsupported opcode mnemonic.
        opcode: String,
    },
    /// One drop instruction could not be represented by the bounded runtime.
    #[error(
        "export `{export_name}` function {function_index} requires `drop` over a materialized stack value"
    )]
    UnsupportedDrop {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
    },
    /// One lowered function violated the expected straight-line stack discipline.
    #[error(
        "export `{export_name}` function {function_index} violated bounded stack discipline: {detail}"
    )]
    InvalidStackState {
        /// Export name.
        export_name: String,
        /// Function index.
        function_index: u32,
        /// Human-readable detail.
        detail: String,
    },
    /// One runtime validation or execution refusal occurred after lowering.
    #[error(transparent)]
    Execution(#[from] TassadarExecutionRefusal),
    /// Final program-artifact assembly failed validation.
    #[error(transparent)]
    ProgramArtifact(#[from] TassadarProgramArtifactError),
}

/// Parses one Wasm binary and lowers its exported bounded module functions into
/// runnable runtime artifacts.
pub fn compile_tassadar_wasm_binary_module_to_artifact_bundle(
    source_name: impl Into<String>,
    wasm_bytes: &[u8],
    profile: &TassadarWasmProfile,
) -> Result<TassadarWasmModuleArtifactBundle, TassadarWasmModuleArtifactBundleError> {
    let normalized_module = parse_tassadar_normalized_wasm_module(wasm_bytes)?;
    compile_tassadar_normalized_wasm_module_to_artifact_bundle_with_source(
        source_name.into(),
        stable_bytes_digest(wasm_bytes),
        normalized_module,
        profile,
    )
}

/// Lowers one normalized Wasm module into runnable runtime artifacts.
pub fn compile_tassadar_normalized_wasm_module_to_artifact_bundle(
    source_name: impl Into<String>,
    normalized_module: &TassadarNormalizedWasmModule,
    profile: &TassadarWasmProfile,
) -> Result<TassadarWasmModuleArtifactBundle, TassadarWasmModuleArtifactBundleError> {
    let wasm_bytes = encode_tassadar_normalized_wasm_module(normalized_module)?;
    compile_tassadar_normalized_wasm_module_to_artifact_bundle_with_source(
        source_name.into(),
        stable_bytes_digest(&wasm_bytes),
        normalized_module.clone(),
        profile,
    )
}

fn compile_tassadar_normalized_wasm_module_to_artifact_bundle_with_source(
    source_name: String,
    source_digest: String,
    normalized_module: TassadarNormalizedWasmModule,
    profile: &TassadarWasmProfile,
) -> Result<TassadarWasmModuleArtifactBundle, TassadarWasmModuleArtifactBundleError> {
    normalized_module.validate_internal_consistency()?;
    let Some(trace_abi) = tassadar_trace_abi_for_profile_id(profile.profile_id.as_str()) else {
        return Err(TassadarWasmModuleArtifactBundleError::UnsupportedTraceAbi {
            profile_id: profile.profile_id.clone(),
        });
    };
    validate_memory_shape(&normalized_module)?;
    let source_identity = TassadarProgramSourceIdentity::new(
        TassadarProgramSourceKind::WasmBinary,
        source_name,
        source_digest,
    );
    let toolchain_identity = TassadarCompilerToolchainIdentity::new(
        TASSADAR_WASM_MODULE_COMPILER_FAMILY,
        TASSADAR_WASM_MODULE_COMPILER_VERSION,
        profile.profile_id.clone(),
    )
    .with_pipeline_features(vec![
        String::from("normalized_module_ir"),
        String::from("export_only_lowering"),
        String::from("straight_line_only"),
    ]);

    let function_exports = normalized_module
        .exports
        .iter()
        .filter(|export| export.kind == psionic_ir::TassadarNormalizedWasmExportKind::Function)
        .cloned()
        .collect::<Vec<_>>();
    if function_exports.is_empty() {
        return Err(TassadarWasmModuleArtifactBundleError::NoFunctionExports {
            module_digest: normalized_module.module_digest.clone(),
        });
    }

    let base_memory = initial_memory_image(&normalized_module)?;
    let lowered_exports = function_exports
        .into_iter()
        .map(|export| {
            lower_function_export(
                &normalized_module,
                &export.export_name,
                export.index,
                &base_memory,
                &source_identity,
                &toolchain_identity,
                profile,
                &trace_abi,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(TassadarWasmModuleArtifactBundle::new(
        format!(
            "tassadar.wasm_module.{}.artifact_bundle.v1",
            &normalized_module.module_digest[..12]
        ),
        source_identity,
        toolchain_identity,
        normalized_module,
        lowered_exports,
    ))
}

fn lower_function_export(
    module: &TassadarNormalizedWasmModule,
    export_name: &str,
    function_index: u32,
    base_memory: &[i32],
    source_identity: &TassadarProgramSourceIdentity,
    toolchain_identity: &TassadarCompilerToolchainIdentity,
    profile: &TassadarWasmProfile,
    trace_abi: &TassadarTraceAbi,
) -> Result<TassadarWasmModuleExportArtifact, TassadarWasmModuleArtifactBundleError> {
    let function = module
        .functions
        .iter()
        .find(|function| function.function_index == function_index)
        .ok_or_else(
            || TassadarWasmModuleArtifactBundleError::InvalidStackState {
                export_name: export_name.to_string(),
                function_index,
                detail: String::from(
                    "export referenced a missing function after module validation",
                ),
            },
        )?;
    let body = function.body.as_ref().ok_or_else(|| {
        TassadarWasmModuleArtifactBundleError::ExportedImportUnsupported {
            export_name: export_name.to_string(),
            function_index,
        }
    })?;
    let signature = &module.types[function.type_index as usize];
    if !signature.params.is_empty() {
        return Err(
            TassadarWasmModuleArtifactBundleError::UnsupportedParamCount {
                export_name: export_name.to_string(),
                function_index,
                param_count: signature.params.len(),
            },
        );
    }
    if signature.results.len() > 1
        || signature
            .results
            .iter()
            .any(|result| *result != TassadarNormalizedWasmValueType::I32)
    {
        return Err(
            TassadarWasmModuleArtifactBundleError::UnsupportedResultTypes {
                export_name: export_name.to_string(),
                function_index,
                result_types: signature
                    .results
                    .iter()
                    .map(|result| format!("{result:?}"))
                    .collect(),
            },
        );
    }
    for local in &body.locals {
        if *local != TassadarNormalizedWasmValueType::I32 {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedLocalType {
                    export_name: export_name.to_string(),
                    function_index,
                    local_type: format!("{local:?}"),
                },
            );
        }
    }

    let local_count = body.locals.len();
    let mut runtime_instructions = Vec::new();
    let mut stack = Vec::<PendingValue>::new();
    let mut max_slot = base_memory.len().saturating_sub(1);
    let mut terminated = false;

    for instruction in &body.instructions {
        if terminated {
            return Err(TassadarWasmModuleArtifactBundleError::InvalidStackState {
                export_name: export_name.to_string(),
                function_index,
                detail: String::from("instructions continued after explicit return"),
            });
        }
        match instruction {
            TassadarNormalizedWasmInstruction::I32Const { value } => {
                stack.push(PendingValue::Const(*value));
            }
            TassadarNormalizedWasmInstruction::LocalGet { local_index } => {
                validate_local_index(export_name, function_index, *local_index, local_count)?;
                stack.push(PendingValue::Local(*local_index));
            }
            TassadarNormalizedWasmInstruction::LocalSet { local_index } => {
                validate_local_index(export_name, function_index, *local_index, local_count)?;
                let value = pop_stack_value(export_name, function_index, &mut stack, "local.set")?;
                materialize_value(
                    export_name,
                    function_index,
                    value,
                    &mut runtime_instructions,
                    local_count,
                )?;
                runtime_instructions.push(TassadarInstruction::LocalSet {
                    local: u8::try_from(*local_index)
                        .expect("validated local index should fit in u8"),
                });
            }
            TassadarNormalizedWasmInstruction::LocalTee { local_index } => {
                validate_local_index(export_name, function_index, *local_index, local_count)?;
                let value = pop_stack_value(export_name, function_index, &mut stack, "local.tee")?;
                materialize_value(
                    export_name,
                    function_index,
                    value,
                    &mut runtime_instructions,
                    local_count,
                )?;
                let local =
                    u8::try_from(*local_index).expect("validated local index should fit in u8");
                runtime_instructions.push(TassadarInstruction::LocalSet { local });
                runtime_instructions.push(TassadarInstruction::LocalGet { local });
                stack.push(PendingValue::StackValue);
            }
            TassadarNormalizedWasmInstruction::I32Add
            | TassadarNormalizedWasmInstruction::I32Sub
            | TassadarNormalizedWasmInstruction::I32Mul
            | TassadarNormalizedWasmInstruction::I32LtS => {
                let right = pop_stack_value(
                    export_name,
                    function_index,
                    &mut stack,
                    instruction.mnemonic(),
                )?;
                let left = pop_stack_value(
                    export_name,
                    function_index,
                    &mut stack,
                    instruction.mnemonic(),
                )?;
                materialize_value(
                    export_name,
                    function_index,
                    left,
                    &mut runtime_instructions,
                    local_count,
                )?;
                materialize_value(
                    export_name,
                    function_index,
                    right,
                    &mut runtime_instructions,
                    local_count,
                )?;
                runtime_instructions.push(match instruction {
                    TassadarNormalizedWasmInstruction::I32Add => TassadarInstruction::I32Add,
                    TassadarNormalizedWasmInstruction::I32Sub => TassadarInstruction::I32Sub,
                    TassadarNormalizedWasmInstruction::I32Mul => TassadarInstruction::I32Mul,
                    TassadarNormalizedWasmInstruction::I32LtS => TassadarInstruction::I32Lt,
                    _ => unreachable!("filtered above"),
                });
                stack.push(PendingValue::StackValue);
            }
            TassadarNormalizedWasmInstruction::I32Shl => {
                return Err(
                    TassadarWasmModuleArtifactBundleError::UnsupportedInstruction {
                        export_name: export_name.to_string(),
                        function_index,
                        opcode: String::from("i32.shl"),
                    },
                );
            }
            TassadarNormalizedWasmInstruction::I32Load {
                offset,
                memory_index,
                ..
            } => {
                let address = pop_stack_value(export_name, function_index, &mut stack, "i32.load")?;
                let slot = resolve_memory_slot(
                    export_name,
                    function_index,
                    "i32.load",
                    address,
                    *offset,
                    *memory_index,
                )?;
                max_slot = max_slot.max(usize::from(slot));
                runtime_instructions.push(TassadarInstruction::I32Load { slot });
                stack.push(PendingValue::StackValue);
            }
            TassadarNormalizedWasmInstruction::I32Store {
                offset,
                memory_index,
                ..
            } => {
                let value = pop_stack_value(export_name, function_index, &mut stack, "i32.store")?;
                let address =
                    pop_stack_value(export_name, function_index, &mut stack, "i32.store")?;
                let slot = resolve_memory_slot(
                    export_name,
                    function_index,
                    "i32.store",
                    address,
                    *offset,
                    *memory_index,
                )?;
                max_slot = max_slot.max(usize::from(slot));
                materialize_value(
                    export_name,
                    function_index,
                    value,
                    &mut runtime_instructions,
                    local_count,
                )?;
                runtime_instructions.push(TassadarInstruction::I32Store { slot });
            }
            TassadarNormalizedWasmInstruction::Call {
                function_index: target,
            } => {
                return Err(TassadarWasmModuleArtifactBundleError::UnsupportedCall {
                    export_name: export_name.to_string(),
                    function_index,
                    target_function_index: *target,
                });
            }
            TassadarNormalizedWasmInstruction::Drop => {
                let value = pop_stack_value(export_name, function_index, &mut stack, "drop")?;
                if matches!(value, PendingValue::StackValue) {
                    return Err(TassadarWasmModuleArtifactBundleError::UnsupportedDrop {
                        export_name: export_name.to_string(),
                        function_index,
                    });
                }
            }
            TassadarNormalizedWasmInstruction::Return => {
                emit_return(
                    export_name,
                    function_index,
                    &signature.results,
                    &mut stack,
                    &mut runtime_instructions,
                    local_count,
                )?;
                terminated = true;
            }
        }
    }

    if !terminated {
        emit_return(
            export_name,
            function_index,
            &signature.results,
            &mut stack,
            &mut runtime_instructions,
            local_count,
        )?;
    }

    let mut initial_memory = base_memory.to_vec();
    if max_slot >= initial_memory.len() {
        initial_memory.resize(max_slot + 1, 0);
    }
    let program_id = format!(
        "tassadar.wasm_module.{}.{}.program.v1",
        &module.module_digest[..12],
        sanitize_label(export_name)
    );
    let validated_program = TassadarProgram::new(
        program_id.clone(),
        profile,
        local_count,
        initial_memory.len(),
        runtime_instructions,
    )
    .with_initial_memory(initial_memory);
    let runner = TassadarCpuReferenceRunner::for_profile(profile.clone()).ok_or_else(|| {
        TassadarWasmModuleArtifactBundleError::UnsupportedTraceAbi {
            profile_id: profile.profile_id.clone(),
        }
    })?;
    let execution = runner.execute(&validated_program)?;
    let program_artifact = TassadarProgramArtifact::new(
        format!(
            "tassadar.wasm_module.{}.{}.artifact.v1",
            &module.module_digest[..12],
            sanitize_label(export_name)
        ),
        source_identity.clone(),
        toolchain_identity.clone(),
        profile,
        trace_abi,
        validated_program,
    )?;
    Ok(TassadarWasmModuleExportArtifact {
        export_name: export_name.to_string(),
        function_index,
        program_artifact,
        execution_manifest: TassadarWasmModuleExportExecutionManifest::new(
            export_name,
            function_index,
            execution.outputs,
            execution.final_memory,
        ),
    })
}

fn validate_memory_shape(
    module: &TassadarNormalizedWasmModule,
) -> Result<(), TassadarWasmModuleArtifactBundleError> {
    if module.memories.len() > 1 {
        return Err(
            TassadarWasmModuleArtifactBundleError::UnsupportedMemoryShape {
                detail: format!(
                    "module declares {} memories, but the bounded runtime still exposes one memory image at most",
                    module.memories.len()
                ),
            },
        );
    }
    for memory in &module.memories {
        if memory.memory_type.memory64 {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedMemoryShape {
                    detail: format!("memory {} is memory64", memory.memory_index),
                },
            );
        }
        if memory.memory_type.shared {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedMemoryShape {
                    detail: format!("memory {} is shared", memory.memory_index),
                },
            );
        }
        if memory.memory_type.page_size_log2.is_some() {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedMemoryShape {
                    detail: format!("memory {} uses a custom page size", memory.memory_index),
                },
            );
        }
    }
    Ok(())
}

fn initial_memory_image(
    module: &TassadarNormalizedWasmModule,
) -> Result<Vec<i32>, TassadarWasmModuleArtifactBundleError> {
    let mut memory = Vec::<i32>::new();
    for segment in &module.data_segments {
        let (memory_index, offset) = match &segment.mode {
            TassadarNormalizedWasmDataMode::Passive => continue,
            TassadarNormalizedWasmDataMode::Active {
                memory_index,
                offset_expr,
            } => match offset_expr {
                TassadarNormalizedWasmConstExpr::I32Const { value } if *value >= 0 => {
                    (*memory_index, *value as u64)
                }
                TassadarNormalizedWasmConstExpr::I32Const { value } => {
                    return Err(
                        TassadarWasmModuleArtifactBundleError::UnsupportedDataSegment {
                            data_index: segment.data_index,
                            detail: format!("negative offset {value}"),
                        },
                    );
                }
                other => {
                    return Err(
                        TassadarWasmModuleArtifactBundleError::UnsupportedDataSegment {
                            data_index: segment.data_index,
                            detail: format!("non-constant offset {other:?}"),
                        },
                    );
                }
            },
        };
        if memory_index != 0 {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedDataSegment {
                    data_index: segment.data_index,
                    detail: format!("memory index {memory_index}"),
                },
            );
        }
        if offset % 4 != 0 {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedDataSegment {
                    data_index: segment.data_index,
                    detail: format!("byte offset {offset} is not word aligned"),
                },
            );
        }
        if segment.bytes.len() % 4 != 0 {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedDataSegment {
                    data_index: segment.data_index,
                    detail: format!("byte length {} is not a multiple of 4", segment.bytes.len()),
                },
            );
        }
        let start_slot = (offset / 4) as usize;
        let end_slot = start_slot + (segment.bytes.len() / 4);
        if end_slot > memory.len() {
            memory.resize(end_slot, 0);
        }
        for (slot_offset, chunk) in segment.bytes.chunks_exact(4).enumerate() {
            memory[start_slot + slot_offset] =
                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
    }
    Ok(memory)
}

fn validate_local_index(
    export_name: &str,
    function_index: u32,
    local_index: u32,
    local_count: usize,
) -> Result<(), TassadarWasmModuleArtifactBundleError> {
    let local_limit = usize::from(u8::MAX);
    if local_index as usize >= local_count {
        return Err(TassadarWasmModuleArtifactBundleError::InvalidStackState {
            export_name: export_name.to_string(),
            function_index,
            detail: format!(
                "local {} is out of range for {} locals",
                local_index, local_count
            ),
        });
    }
    if local_index as usize > local_limit {
        return Err(
            TassadarWasmModuleArtifactBundleError::UnsupportedLocalIndex {
                export_name: export_name.to_string(),
                function_index,
                local_index,
                max_supported: u8::MAX,
            },
        );
    }
    Ok(())
}

fn pop_stack_value(
    export_name: &str,
    function_index: u32,
    stack: &mut Vec<PendingValue>,
    opcode: &str,
) -> Result<PendingValue, TassadarWasmModuleArtifactBundleError> {
    stack.pop().ok_or_else(
        || TassadarWasmModuleArtifactBundleError::InvalidStackState {
            export_name: export_name.to_string(),
            function_index,
            detail: format!("stack underflow while lowering `{opcode}`"),
        },
    )
}

fn materialize_value(
    export_name: &str,
    function_index: u32,
    value: PendingValue,
    instructions: &mut Vec<TassadarInstruction>,
    local_count: usize,
) -> Result<(), TassadarWasmModuleArtifactBundleError> {
    match value {
        PendingValue::Const(value) => instructions.push(TassadarInstruction::I32Const { value }),
        PendingValue::Local(local_index) => {
            validate_local_index(export_name, function_index, local_index, local_count)?;
            instructions.push(TassadarInstruction::LocalGet {
                local: u8::try_from(local_index).expect("validated local index should fit in u8"),
            });
        }
        PendingValue::StackValue => {}
    }
    Ok(())
}

fn resolve_memory_slot(
    export_name: &str,
    function_index: u32,
    opcode: &str,
    address: PendingValue,
    offset: u64,
    memory_index: u32,
) -> Result<u8, TassadarWasmModuleArtifactBundleError> {
    if memory_index != 0 {
        return Err(
            TassadarWasmModuleArtifactBundleError::UnsupportedMemoryImmediate {
                export_name: export_name.to_string(),
                function_index,
                opcode: opcode.to_string(),
                detail: format!("memory index {memory_index}"),
            },
        );
    }
    let base = match address {
        PendingValue::Const(value) if value >= 0 => value as u64,
        PendingValue::Const(value) => {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedMemoryImmediate {
                    export_name: export_name.to_string(),
                    function_index,
                    opcode: opcode.to_string(),
                    detail: format!("negative address {value}"),
                },
            );
        }
        _ => {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedDynamicMemoryAddress {
                    export_name: export_name.to_string(),
                    function_index,
                    opcode: opcode.to_string(),
                },
            );
        }
    };
    let absolute = base.saturating_add(offset);
    if absolute % 4 != 0 {
        return Err(
            TassadarWasmModuleArtifactBundleError::UnsupportedMemoryImmediate {
                export_name: export_name.to_string(),
                function_index,
                opcode: opcode.to_string(),
                detail: format!("absolute byte address {absolute} is not word aligned"),
            },
        );
    }
    let slot = absolute / 4;
    u8::try_from(slot).map_err(|_| {
        TassadarWasmModuleArtifactBundleError::UnsupportedMemoryImmediate {
            export_name: export_name.to_string(),
            function_index,
            opcode: opcode.to_string(),
            detail: format!("slot {slot} exceeds u8 runtime address space"),
        }
    })
}

fn emit_return(
    export_name: &str,
    function_index: u32,
    results: &[TassadarNormalizedWasmValueType],
    stack: &mut Vec<PendingValue>,
    runtime_instructions: &mut Vec<TassadarInstruction>,
    local_count: usize,
) -> Result<(), TassadarWasmModuleArtifactBundleError> {
    match results {
        [] => {
            if !stack.is_empty() {
                return Err(TassadarWasmModuleArtifactBundleError::InvalidStackState {
                    export_name: export_name.to_string(),
                    function_index,
                    detail: format!(
                        "implicit or explicit void return left {} values on the stack",
                        stack.len()
                    ),
                });
            }
        }
        [TassadarNormalizedWasmValueType::I32] => {
            let result = pop_stack_value(export_name, function_index, stack, "return")?;
            materialize_value(
                export_name,
                function_index,
                result,
                runtime_instructions,
                local_count,
            )?;
            if !stack.is_empty() {
                return Err(TassadarWasmModuleArtifactBundleError::InvalidStackState {
                    export_name: export_name.to_string(),
                    function_index,
                    detail: format!(
                        "return left {} extra values below the final result",
                        stack.len()
                    ),
                });
            }
            runtime_instructions.push(TassadarInstruction::Output);
        }
        other => {
            return Err(
                TassadarWasmModuleArtifactBundleError::UnsupportedResultTypes {
                    export_name: export_name.to_string(),
                    function_index,
                    result_types: other.iter().map(|value| format!("{value:?}")).collect(),
                },
            );
        }
    }
    runtime_instructions.push(TassadarInstruction::Return);
    Ok(())
}

fn sanitize_label(label: &str) -> String {
    let mut sanitized = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    while sanitized.contains("__") {
        sanitized = sanitized.replace("__", "_");
    }
    sanitized.trim_matches('_').to_string()
}

fn stable_bytes_digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use psionic_ir::{
        encode_tassadar_normalized_wasm_module, tassadar_seeded_multi_function_module,
    };
    use psionic_runtime::{
        TassadarCpuReferenceRunner, TassadarWasmProfile, tassadar_canonical_wasm_binary_path,
    };

    use super::{
        TassadarWasmModuleArtifactBundleError,
        compile_tassadar_normalized_wasm_module_to_artifact_bundle,
        compile_tassadar_wasm_binary_module_to_artifact_bundle,
    };

    #[test]
    fn wasm_module_bundle_refuses_parametrized_canonical_micro_kernel() {
        let wasm_bytes =
            std::fs::read(tassadar_canonical_wasm_binary_path()).expect("canonical wasm binary");
        let error = compile_tassadar_wasm_binary_module_to_artifact_bundle(
            "fixtures/tassadar/wasm/tassadar_micro_wasm_kernel.wasm",
            &wasm_bytes,
            &TassadarWasmProfile::article_i32_compute_v1(),
        )
        .expect_err("canonical kernel should stay outside zero-parameter lowering");
        assert!(
            matches!(
                error,
                TassadarWasmModuleArtifactBundleError::UnsupportedParamCount { .. }
            ),
            "{error:?}"
        );
    }

    #[test]
    fn wasm_module_bundle_lowers_multi_function_exports_and_matches_cpu_reference() {
        let module = tassadar_seeded_multi_function_module().expect("seeded module should build");
        let bundle = compile_tassadar_normalized_wasm_module_to_artifact_bundle(
            "seeded://tassadar/wasm/multi_function_v1",
            &module,
            &TassadarWasmProfile::article_i32_compute_v1(),
        )
        .expect("seeded module should lower");
        assert_eq!(bundle.lowered_exports.len(), 2);

        for artifact in &bundle.lowered_exports {
            let execution = TassadarCpuReferenceRunner::for_program(
                &artifact.program_artifact.validated_program,
            )
            .expect("lowered program should select a runner")
            .execute(&artifact.program_artifact.validated_program)
            .expect("lowered program should execute exactly");
            assert_eq!(
                execution.outputs,
                artifact.execution_manifest.expected_outputs
            );
            assert_eq!(
                execution.final_memory,
                artifact.execution_manifest.expected_final_memory
            );
        }
        assert_eq!(
            bundle
                .lowered_exports
                .iter()
                .map(|artifact| (
                    artifact.export_name.as_str(),
                    artifact.execution_manifest.expected_outputs.clone()
                ))
                .collect::<std::collections::BTreeMap<_, _>>(),
            std::collections::BTreeMap::from([("local_double", vec![14]), ("pair_sum", vec![5]),])
        );
    }

    #[test]
    fn wasm_binary_roundtrip_parse_and_lower_stays_machine_legible() {
        let module = tassadar_seeded_multi_function_module().expect("seeded module should build");
        let bytes =
            encode_tassadar_normalized_wasm_module(&module).expect("seeded module should encode");
        let bundle = compile_tassadar_wasm_binary_module_to_artifact_bundle(
            "seeded://tassadar/wasm/multi_function_v1",
            &bytes,
            &TassadarWasmProfile::article_i32_compute_v1(),
        )
        .expect("seeded bytes should lower");
        assert_eq!(bundle.normalized_module, module);
        assert_eq!(
            bundle.normalized_module.exported_function_names(),
            vec![String::from("pair_sum"), String::from("local_double")]
        );
    }
}
