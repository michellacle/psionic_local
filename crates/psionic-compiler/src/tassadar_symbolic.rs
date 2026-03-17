use std::collections::BTreeMap;

use psionic_ir::{
    TassadarSymbolicBinaryOp, TassadarSymbolicExpr, TassadarSymbolicLoweringOpcode,
    TassadarSymbolicOperand, TassadarSymbolicProgram, TassadarSymbolicProgramError,
};
use psionic_runtime::{
    TassadarCpuReferenceRunner, TassadarExecutionRefusal, TassadarInstruction, TassadarOpcode,
    TassadarProgram, tassadar_wasm_profile_for_id,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

/// One lowered runtime program instantiated from the bounded symbolic IR.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TassadarLoweredSymbolicProgram {
    /// Stable symbolic source-program identifier.
    pub symbolic_program_id: String,
    /// Stable digest of the symbolic program.
    pub symbolic_program_digest: String,
    /// Stable digest of the concrete input assignments used for lowering.
    pub input_assignment_digest: String,
    /// Lowered runtime-visible Tassadar program.
    pub validated_program: TassadarProgram,
}

impl TassadarLoweredSymbolicProgram {
    /// Returns a stable digest over the lowered symbolic/runtime pair.
    #[must_use]
    pub fn stable_digest(&self) -> String {
        stable_digest(b"tassadar_lowered_symbolic_program|", self)
    }
}

/// Lowering failure for the bounded symbolic IR.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TassadarSymbolicLoweringError {
    /// One symbolic parse/validation/evaluation check failed.
    #[error(transparent)]
    Symbolic(#[from] TassadarSymbolicProgramError),
    /// The symbolic program targeted an unknown runtime profile.
    #[error("unsupported symbolic runtime profile `{profile_id}`")]
    UnsupportedProfile {
        /// Unsupported profile identifier.
        profile_id: String,
    },
    /// The symbolic program required one opcode the selected profile does not support.
    #[error("symbolic lowering opcode `{opcode}` is unsupported by runtime profile `{profile_id}`")]
    UnsupportedLoweringOpcode {
        /// Unsupported opcode mnemonic.
        opcode: String,
        /// Runtime profile identifier.
        profile_id: String,
    },
    /// The symbolic program lowered into too many runtime locals for the selected profile.
    #[error(
        "symbolic lowering requires {required} locals, but runtime profile `{profile_id}` supports at most {max_supported}"
    )]
    TooManyLoweredLocals {
        /// Required lowered local count.
        required: usize,
        /// Runtime profile local limit.
        max_supported: usize,
        /// Runtime profile identifier.
        profile_id: String,
    },
    /// The lowered runtime program failed validation or execution on the CPU reference lane.
    #[error(transparent)]
    Execution(#[from] TassadarExecutionRefusal),
}

/// Lowers one bounded symbolic program into a concrete runtime program over the
/// supplied input assignments.
pub fn lower_tassadar_symbolic_program(
    program: &TassadarSymbolicProgram,
    input_assignments: &BTreeMap<String, i32>,
) -> Result<TassadarLoweredSymbolicProgram, TassadarSymbolicLoweringError> {
    program.validate()?;
    let Some(profile) = tassadar_wasm_profile_for_id(program.profile_id.as_str()) else {
        return Err(TassadarSymbolicLoweringError::UnsupportedProfile {
            profile_id: program.profile_id.clone(),
        });
    };

    for opcode in program.required_lowering_opcodes() {
        let runtime_opcode = runtime_opcode(opcode);
        if !profile.supports(runtime_opcode) {
            return Err(TassadarSymbolicLoweringError::UnsupportedLoweringOpcode {
                opcode: runtime_opcode.mnemonic().to_string(),
                profile_id: profile.profile_id.clone(),
            });
        }
    }

    let required_locals = program.required_lowered_locals();
    if required_locals > profile.max_locals {
        return Err(TassadarSymbolicLoweringError::TooManyLoweredLocals {
            required: required_locals,
            max_supported: profile.max_locals,
            profile_id: profile.profile_id.clone(),
        });
    }

    let local_map = symbolic_local_map(program);
    let initial_memory = program.initial_memory_image(input_assignments)?;
    let mut instructions = Vec::new();

    for statement in &program.statements {
        match statement {
            psionic_ir::TassadarSymbolicStatement::Let { name, expr } => {
                lower_expr(program, expr, &local_map, &mut instructions)?;
                instructions.push(TassadarInstruction::LocalSet {
                    local: *local_map
                        .get(name.as_str())
                        .expect("validated symbolic binding should have one lowered local"),
                });
            }
            psionic_ir::TassadarSymbolicStatement::Store { slot, value } => {
                lower_operand(program, value, &local_map, &mut instructions)?;
                instructions.push(TassadarInstruction::I32Store { slot: *slot });
            }
            psionic_ir::TassadarSymbolicStatement::Output { value } => {
                lower_operand(program, value, &local_map, &mut instructions)?;
                instructions.push(TassadarInstruction::Output);
            }
        }
    }
    instructions.push(TassadarInstruction::Return);

    let input_assignment_digest =
        stable_digest(b"tassadar_symbolic_input_assignments|", input_assignments);
    let lowered_program_id = format!(
        "{}.lowered.{}",
        program.program_id,
        &input_assignment_digest[..12]
    );
    let validated_program = TassadarProgram::new(
        lowered_program_id,
        &profile,
        required_locals,
        program.memory_slots,
        instructions,
    )
    .with_initial_memory(initial_memory);

    let runner = TassadarCpuReferenceRunner::for_program(&validated_program)?;
    let _execution = runner.execute(&validated_program)?;

    Ok(TassadarLoweredSymbolicProgram {
        symbolic_program_id: program.program_id.clone(),
        symbolic_program_digest: program.stable_digest(),
        input_assignment_digest,
        validated_program,
    })
}

fn symbolic_local_map(program: &TassadarSymbolicProgram) -> BTreeMap<&str, u8> {
    let mut local_map = BTreeMap::new();
    let mut next_local = 0_u8;
    for statement in &program.statements {
        if let psionic_ir::TassadarSymbolicStatement::Let { name, .. } = statement {
            local_map.insert(name.as_str(), next_local);
            next_local = next_local.saturating_add(1);
        }
    }
    local_map
}

fn lower_expr(
    program: &TassadarSymbolicProgram,
    expr: &TassadarSymbolicExpr,
    local_map: &BTreeMap<&str, u8>,
    instructions: &mut Vec<TassadarInstruction>,
) -> Result<(), TassadarSymbolicLoweringError> {
    match expr {
        TassadarSymbolicExpr::Operand { operand } => {
            lower_operand(program, operand, local_map, instructions)?;
        }
        TassadarSymbolicExpr::Binary { op, left, right } => {
            lower_operand(program, left, local_map, instructions)?;
            lower_operand(program, right, local_map, instructions)?;
            instructions.push(match op {
                TassadarSymbolicBinaryOp::Add => TassadarInstruction::I32Add,
                TassadarSymbolicBinaryOp::Sub => TassadarInstruction::I32Sub,
                TassadarSymbolicBinaryOp::Mul => TassadarInstruction::I32Mul,
                TassadarSymbolicBinaryOp::Lt => TassadarInstruction::I32Lt,
            });
        }
    }
    Ok(())
}

fn lower_operand(
    program: &TassadarSymbolicProgram,
    operand: &TassadarSymbolicOperand,
    local_map: &BTreeMap<&str, u8>,
    instructions: &mut Vec<TassadarInstruction>,
) -> Result<(), TassadarSymbolicLoweringError> {
    match operand {
        TassadarSymbolicOperand::Name { name } => {
            if let Some(local) = local_map.get(name.as_str()) {
                instructions.push(TassadarInstruction::LocalGet { local: *local });
            } else if let Some(slot) = program.input_slot(name.as_str()) {
                instructions.push(TassadarInstruction::I32Load { slot });
            } else {
                return Err(
                    TassadarSymbolicProgramError::UnknownValue { name: name.clone() }.into(),
                );
            }
        }
        TassadarSymbolicOperand::Const { value } => {
            instructions.push(TassadarInstruction::I32Const { value: *value });
        }
        TassadarSymbolicOperand::MemorySlot { slot } => {
            instructions.push(TassadarInstruction::I32Load { slot: *slot });
        }
    }
    Ok(())
}

fn runtime_opcode(opcode: TassadarSymbolicLoweringOpcode) -> TassadarOpcode {
    match opcode {
        TassadarSymbolicLoweringOpcode::I32Const => TassadarOpcode::I32Const,
        TassadarSymbolicLoweringOpcode::I32Load => TassadarOpcode::I32Load,
        TassadarSymbolicLoweringOpcode::I32Store => TassadarOpcode::I32Store,
        TassadarSymbolicLoweringOpcode::LocalGet => TassadarOpcode::LocalGet,
        TassadarSymbolicLoweringOpcode::LocalSet => TassadarOpcode::LocalSet,
        TassadarSymbolicLoweringOpcode::I32Add => TassadarOpcode::I32Add,
        TassadarSymbolicLoweringOpcode::I32Sub => TassadarOpcode::I32Sub,
        TassadarSymbolicLoweringOpcode::I32Mul => TassadarOpcode::I32Mul,
        TassadarSymbolicLoweringOpcode::I32Lt => TassadarOpcode::I32Lt,
        TassadarSymbolicLoweringOpcode::Output => TassadarOpcode::Output,
        TassadarSymbolicLoweringOpcode::Return => TassadarOpcode::Return,
    }
}

fn stable_digest<T: Serialize>(prefix: &[u8], value: &T) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(serde_json::to_vec(value).unwrap_or_default());
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use psionic_ir::tassadar_symbolic_program_examples;
    use psionic_runtime::run_tassadar_exact_parity;

    use super::{TassadarSymbolicLoweringError, lower_tassadar_symbolic_program};

    #[test]
    fn symbolic_programs_lower_and_match_cpu_reference_truth() {
        for example in tassadar_symbolic_program_examples() {
            let lowered =
                lower_tassadar_symbolic_program(&example.program, &example.input_assignments)
                    .expect("example should lower");
            let parity = run_tassadar_exact_parity(&lowered.validated_program)
                .expect("lowered example should execute");

            parity.require_exact().expect("fixture parity should hold");
            assert_eq!(parity.reference.outputs, example.expected_outputs);
            assert_eq!(parity.reference.final_memory, example.expected_final_memory);
        }
    }

    #[test]
    fn symbolic_lowering_refuses_profiles_that_do_not_support_required_opcodes() {
        let mut example = tassadar_symbolic_program_examples()
            .into_iter()
            .find(|example| example.case_id == "parity_two_bits")
            .expect("parity example");
        example.program.profile_id = String::from("tassadar.wasm.core_i32.v1");

        let error = lower_tassadar_symbolic_program(&example.program, &example.input_assignments)
            .expect_err("lt should stay explicit as unsupported on core_i32_v1");

        assert_eq!(
            error,
            TassadarSymbolicLoweringError::UnsupportedLoweringOpcode {
                opcode: String::from("i32.lt"),
                profile_id: String::from("tassadar.wasm.core_i32.v1"),
            }
        );
    }

    #[test]
    fn symbolic_lowering_requires_declared_input_assignments() {
        let example = tassadar_symbolic_program_examples()
            .into_iter()
            .next()
            .expect("addition example");
        let error = lower_tassadar_symbolic_program(&example.program, &BTreeMap::new())
            .expect_err("missing input assignments should stay explicit");

        assert_eq!(
            error,
            TassadarSymbolicLoweringError::Symbolic(
                psionic_ir::TassadarSymbolicProgramError::MissingInputAssignment {
                    input: String::from("lhs"),
                }
            )
        );
    }
}
