#!/usr/bin/env python
"""Convert old MedKGAT-named checkpoints to the current MKGSurv naming.

The project saves model checkpoints as state_dict objects, so Python class names
are not serialized in the standard valid_Best.pth files. This script still
normalizes old path names, common metadata strings, and known state_dict key
prefixes so released checkpoints match the current project naming.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch


STATE_KEY_REPLACEMENTS = (
    ("fusion_fusion_block", "fusion_module.fusion_block"),
    ("medkgat_fusion", "mkgsurv_fusion"),
    ("medkgatfusion", "mkgsurvfusion"),
    ("MedKGATFusion", "MKGSurvFusion"),
    ("MedKGAT", "MKGSurv"),
)

STRING_REPLACEMENTS = (
    ("medkgat_fusion", "mkgsurv_fusion"),
    ("medkgatfusion", "mkgsurvfusion"),
    ("MedKGATFusion", "MKGSurvFusion"),
    ("MedKGAT", "MKGSurv"),
)


def apply_replacements(value: str, replacements: tuple[tuple[str, str], ...]) -> tuple[str, int]:
    changed = 0
    current = value
    for old, new in replacements:
        if old in current:
            changed += current.count(old)
            current = current.replace(old, new)
    return current, changed


def convert_nested_strings(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        return apply_replacements(value, STRING_REPLACEMENTS)
    if isinstance(value, dict):
        changed = 0
        out = {}
        for key, item in value.items():
            new_key, key_changed = convert_nested_strings(key)
            new_item, item_changed = convert_nested_strings(item)
            out[new_key] = new_item
            changed += key_changed + item_changed
        return out, changed
    if isinstance(value, list):
        changed = 0
        out = []
        for item in value:
            new_item, item_changed = convert_nested_strings(item)
            out.append(new_item)
            changed += item_changed
        return out, changed
    if isinstance(value, tuple):
        changed = 0
        out = []
        for item in value:
            new_item, item_changed = convert_nested_strings(item)
            out.append(new_item)
            changed += item_changed
        return tuple(out), changed
    return value, 0


def convert_state_dict(state_dict: dict[str, Any]) -> tuple[OrderedDict[str, Any], int]:
    converted: OrderedDict[str, Any] = OrderedDict()
    renamed = 0
    for key, value in state_dict.items():
        new_key, key_changes = apply_replacements(key, STATE_KEY_REPLACEMENTS)
        if new_key in converted:
            raise ValueError(f"Duplicate key after conversion: {key!r} -> {new_key!r}")
        converted[new_key] = value
        renamed += 1 if new_key != key else 0
        renamed += max(0, key_changes - (1 if new_key != key else 0))
    return converted, renamed


def iter_checkpoint_paths(input_root: Path, include_extra_pth: bool) -> list[Path]:
    if include_extra_pth:
        paths = sorted(input_root.rglob("*.pth"))
    else:
        paths = sorted(input_root.rglob("valid_Best.pth"))
    return [path for path in paths if path.is_file()]


def destination_for(input_path: Path, input_root: Path, output_root: Path, in_place: bool) -> Path:
    if in_place:
        return input_path

    rel = input_path.relative_to(input_root)
    rel_string, _ = apply_replacements(str(rel), STRING_REPLACEMENTS)
    return output_root / rel_string


def convert_checkpoint(input_path: Path, output_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(input_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected checkpoint dict, got {type(checkpoint).__name__}: {input_path}")
    if "model" not in checkpoint or not isinstance(checkpoint["model"], dict):
        raise KeyError(f"Checkpoint has no model state_dict: {input_path}")

    converted_checkpoint = copy.copy(checkpoint)
    converted_model, renamed_keys = convert_state_dict(checkpoint["model"])
    converted_checkpoint["model"] = converted_model

    metadata_changes = 0
    for key, value in list(converted_checkpoint.items()):
        if key == "model":
            continue
        new_value, changed = convert_nested_strings(value)
        converted_checkpoint[key] = new_value
        metadata_changes += changed

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_checkpoint, output_path)

    return {
        "source": str(input_path),
        "destination": str(output_path),
        "epoch": converted_checkpoint.get("epoch"),
        "state_dict_keys": len(converted_model),
        "renamed_keys": renamed_keys,
        "metadata_string_replacements": metadata_changes,
    }


def infer_modalities(run_id: str) -> str:
    if "run001" in run_id:
        return "genomics-genomics,image-pathology"
    return "all"


def verify_checkpoint_load(checkpoint_path: Path, repo_root: Path) -> dict[str, Any]:
    code_path = repo_root / "Code"
    if str(code_path) not in sys.path:
        sys.path.insert(0, str(code_path))

    from argparse import Namespace
    from datasets import GetDataset
    from modules.model import GetModel

    parts = checkpoint_path.parts
    fold_name = checkpoint_path.parent.name
    run_dir = checkpoint_path.parent.parent.name
    dataset_name = checkpoint_path.parent.parent.parent.name
    run_id = run_dir.split("+", 1)[0]
    fold = int(fold_name.replace("Fold", ""))

    args = Namespace(
        ckpt_path=str(checkpoint_path.parents[3]),
        log_path="../Results",
        load_pth_path=None,
        points_save_path=None,
        view_groups_attention_path=None,
        dataset=dataset_name,
        modalities=infer_modalities(run_id),
        debug_mode=False,
        fold=fold,
        do_mixup=False,
        knowledge_source="kimi",
        use_medical_knowledge=True,
        knowledge_type="all",
        model_task=dataset_name,
        decode_task="surv_pred",
        image_aggregater="panther",
        fusion_type="mkgsurv_fusion",
        with_multimodal_align=False,
        with_multimodal_vib=False,
        num_layers=3,
        kl_loss_weight=7,
        runs_id=run_id,
        acc_step=1,
        gpu_id="0",
        seed=109,
        learning_rate=5e-5,
        weight_decay=1e-4,
        num_epochs=60,
        batch_size=64,
        optimizer="AdamW",
        scheduler="CosineAnnealingLR",
        use_amp=False,
        use_ddp=False,
        freezed_backbone=False,
        finetune=False,
        continue_training=False,
        draw_kaplan_meier=False,
    )

    dataset = GetDataset("test", args)
    model = GetModel(args, dataset)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)

    return {
        "checkpoint": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="Checkpoints", help="Root containing original checkpoints.")
    parser.add_argument(
        "--output-root",
        default="Checkpoints_mkgsurv_converted",
        help="Root for converted checkpoints when not using --in-place.",
    )
    parser.add_argument("--in-place", action="store_true", help="Overwrite checkpoints in --input-root.")
    parser.add_argument(
        "--include-extra-pth",
        action="store_true",
        help="Convert every .pth file. By default only standard valid_Best.pth files are converted.",
    )
    parser.add_argument("--verify-load", action="store_true", help="Strictly load converted checkpoints with current code.")
    parser.add_argument("--report-path", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / "Code").is_dir() and (path / "Checkpoints").is_dir():
            return path
    raise FileNotFoundError(f"Could not find project root from {start}")


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).resolve())
    input_root = (repo_root / args.input_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    paths = iter_checkpoint_paths(input_root, args.include_extra_pth)
    if not paths:
        raise FileNotFoundError(f"No checkpoint files found under {input_root}")

    report: dict[str, Any] = {
        "input_root": str(input_root),
        "output_root": str(input_root if args.in_place else output_root),
        "in_place": args.in_place,
        "checkpoint_count": len(paths),
        "converted": [],
        "verification": [],
    }

    for input_path in paths:
        output_path = destination_for(input_path, input_root, output_root, args.in_place)
        item = convert_checkpoint(input_path, output_path)
        report["converted"].append(item)
        print(
            "CONVERTED "
            f"{input_path} -> {output_path} "
            f"renamed_keys={item['renamed_keys']} "
            f"metadata_replacements={item['metadata_string_replacements']}",
            flush=True,
        )

    if args.verify_load:
        for item in report["converted"]:
            result = verify_checkpoint_load(Path(item["destination"]), repo_root)
            report["verification"].append(result)
            print(f"VERIFIED {result['checkpoint']}", flush=True)

    report_path = Path(args.report_path) if args.report_path else (output_root / "conversion_report.json")
    if args.in_place and args.report_path is None:
        report_path = input_root / "conversion_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"REPORT {report_path}", flush=True)


if __name__ == "__main__":
    main()
