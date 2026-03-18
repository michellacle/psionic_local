#!/usr/bin/env python3
"""Generate the committed Parameter Golf oracle parity fixture.

This script extracts the current `load_data_shard`, `load_validation_tokens`,
and `build_sentencepiece_luts` functions from the local `parameter-golf`
checkout, runs them on a small synthetic fixture, and writes the resulting
reference artifact used by `PGOLF-103`.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch


class TorchShim:
    """Small compatibility shim for local torch builds without uint16 support."""

    Tensor = torch.Tensor
    device = torch.device
    int64 = torch.int64
    bool = torch.bool
    int16 = torch.int16

    @staticmethod
    def tensor(*args, **kwargs):
        return torch.tensor(*args, **kwargs)

    @staticmethod
    def cat(tensors, *args, **kwargs):
        return torch.cat(tensors, *args, **kwargs)

    @staticmethod
    def from_numpy(array):
        if array.dtype == np.uint16:
            return torch.tensor(array.astype(np.int32, copy=False), dtype=torch.int32)
        return torch.from_numpy(array)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the committed Parameter Golf oracle parity fixture"
    )
    parser.add_argument(
        "--parameter-golf-root",
        default=str(Path("~/code/parameter-golf").expanduser()),
        help="Local parameter-golf checkout root.",
    )
    parser.add_argument(
        "--output",
        default="fixtures/parameter_golf/parity/parameter_golf_oracle_parity_fixture.json",
        help="Output fixture path.",
    )
    return parser


def extract_functions(path: Path, names: set[str], torch_mod) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace = {
        "np": np,
        "torch": torch_mod,
        "Tensor": torch.Tensor,
        "Path": Path,
        "math": math,
        "glob": glob,
        "spm": type("spm", (), {"SentencePieceProcessor": object}),
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
            exec(code, namespace)
    return {name: namespace[name] for name in names}


def shard_bytes(tokens: list[int]) -> bytes:
    output = bytearray(256 * 4)
    output[0:4] = int(20240520).to_bytes(4, "little", signed=True)
    output[4:8] = int(1).to_bytes(4, "little", signed=True)
    output[8:12] = int(len(tokens)).to_bytes(4, "little", signed=True)
    for token in tokens:
        output.extend(int(token).to_bytes(2, "little", signed=False))
    return bytes(output)


class StubSentencePiece:
    def __init__(self) -> None:
        self.entries = {
            0: ("<unk>", "unknown"),
            1: ("<s>", "control"),
            2: ("▁hello", "normal"),
            3: ("world", "normal"),
            4: ("<0x41>", "byte"),
            5: ("<unused>", "unused"),
        }

    def vocab_size(self) -> int:
        return len(self.entries)

    def is_control(self, token_id: int) -> bool:
        return self.entries[token_id][1] == "control"

    def is_unknown(self, token_id: int) -> bool:
        return self.entries[token_id][1] == "unknown"

    def is_unused(self, token_id: int) -> bool:
        return self.entries[token_id][1] == "unused"

    def is_byte(self, token_id: int) -> bool:
        return self.entries[token_id][1] == "byte"

    def id_to_piece(self, token_id: int) -> str:
        return self.entries[token_id][0]


def luts_to_json_torch(luts) -> dict[str, list]:
    return {
        "base_bytes_lut": luts[0].to(torch.int64).tolist(),
        "has_leading_space_lut": [bool(value) for value in luts[1].tolist()],
        "is_boundary_token_lut": [bool(value) for value in luts[2].tolist()],
    }


def luts_to_json_numpy(luts) -> dict[str, list]:
    return {
        "base_bytes_lut": [int(value) for value in luts[0].tolist()],
        "has_leading_space_lut": [bool(value) for value in luts[1].tolist()],
        "is_boundary_token_lut": [bool(value) for value in luts[2].tolist()],
    }


def byte_count(
    previous_token_ids: list[int], target_token_ids: list[int], luts: dict[str, list]
) -> int:
    total = 0
    for previous_id, target_id in zip(previous_token_ids, target_token_ids):
        total += int(luts["base_bytes_lut"][target_id])
        if luts["has_leading_space_lut"][target_id] and not luts["is_boundary_token_lut"][
            previous_id
        ]:
            total += 1
    return total


def main() -> None:
    args = build_parser().parse_args()
    parameter_golf_root = Path(args.parameter_golf_root).expanduser().resolve()
    train_path = parameter_golf_root / "train_gpt.py"
    mlx_path = parameter_golf_root / "train_gpt_mlx.py"

    function_names = {"load_data_shard", "load_validation_tokens", "build_sentencepiece_luts"}
    train_funcs = extract_functions(train_path, function_names, TorchShim)
    mlx_funcs = extract_functions(mlx_path, function_names, torch)

    validation_shards = [
        ("fineweb_val_000000.bin", [1, 2, 3, 2, 4]),
        ("fineweb_val_000001.bin", [2, 5, 3, 1, 2]),
    ]
    seq_len = 4

    with tempfile.TemporaryDirectory() as tempdir:
        temp_root = Path(tempdir)
        for name, tokens in validation_shards:
            (temp_root / name).write_bytes(shard_bytes(tokens))
        pattern = str(temp_root / "fineweb_val_*.bin")
        train_val_tokens = (
            train_funcs["load_validation_tokens"](pattern, seq_len).to(torch.int64).tolist()
        )
        mlx_val_tokens = mlx_funcs["load_validation_tokens"](pattern, seq_len).tolist()

    stub_sp = StubSentencePiece()
    train_luts = train_funcs["build_sentencepiece_luts"](stub_sp, 8, torch.device("cpu"))
    mlx_luts = mlx_funcs["build_sentencepiece_luts"](stub_sp, 8)
    train_luts_json = luts_to_json_torch(train_luts)
    mlx_luts_json = luts_to_json_numpy(mlx_luts)

    previous_token_ids = train_val_tokens[:-1][:6]
    target_token_ids = train_val_tokens[1:][:6]
    logits = [
        [2.0, 0.5, -1.0, 0.0, -0.5, 1.0],
        [0.2, 1.5, 0.1, -0.3, 0.7, -0.8],
        [1.2, -0.7, 0.3, 2.1, 0.0, -1.4],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [3.0, 1.0, -2.0, 0.5, 0.0, -1.0],
        [-0.2, 0.4, 1.7, -1.1, 0.3, 0.2],
    ]
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    target_tensor = torch.tensor(target_token_ids, dtype=torch.int64)
    val_loss = torch.nn.functional.cross_entropy(
        logits_tensor, target_tensor, reduction="mean"
    ).item()

    train_byte_count = byte_count(previous_token_ids, target_token_ids, train_luts_json)
    mlx_byte_count = byte_count(previous_token_ids, target_token_ids, mlx_luts_json)
    train_val_bpb = (val_loss / math.log(2.0)) * (len(target_token_ids) / train_byte_count)
    mlx_val_bpb = (val_loss / math.log(2.0)) * (len(target_token_ids) / mlx_byte_count)

    fixture = {
        "fixture_id": "parameter_golf_oracle_parity_v1",
        "seq_len": seq_len,
        "tokenizer_vocab_size": 8,
        "validation_shards": [
            {
                "file_name": name,
                "tokens": tokens,
                "file_hex": shard_bytes(tokens).hex(),
            }
            for name, tokens in validation_shards
        ],
        "sentencepiece_entries": [
            {"token_id": token_id, "piece": piece, "kind": kind}
            for token_id, (piece, kind) in sorted(stub_sp.entries.items())
        ],
        "loss_fixture": {
            "prev_token_ids": previous_token_ids,
            "target_token_ids": target_token_ids,
            "logits": logits,
        },
        "oracles": {
            "train_gpt.py": {
                "validation_tokens": train_val_tokens,
                "luts": train_luts_json,
                "val_loss": val_loss,
                "byte_count": train_byte_count,
                "val_bpb": train_val_bpb,
            },
            "train_gpt_mlx.py": {
                "validation_tokens": mlx_val_tokens,
                "luts": mlx_luts_json,
                "val_loss": val_loss,
                "byte_count": mlx_byte_count,
                "val_bpb": mlx_val_bpb,
            },
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
