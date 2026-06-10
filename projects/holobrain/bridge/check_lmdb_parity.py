from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "configs"))

from configs.config_bridge_dataset import build_transforms, dataset_config
from configs.config_holobrain_gd_common_bridge import config as train_config
from robo_orchard_lab.dataset.robotwin.robotwin_lmdb_dataset import (
    RoboTwinLmdbDataset,
)
from robo_orchard_lab.utils.build import build


RAW_KEYS = [
    "uuid",
    "step_index",
    "text",
    "imgs",
    "depths",
    "intrinsic",
    "T_world2cam",
    "T_base2world",
    "joint_state",
]

TRANSFORMED_KEYS = [
    "uuid",
    "text",
    "imgs",
    "depths",
    "image_wh",
    "projection_mat",
    "hist_robot_state",
    "pred_robot_state",
    "joint_scale_shift",
    "joint_mask",
]


@dataclass
class CompareResult:
    name: str
    ok: bool
    detail: str


def _clone_data(data: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(data)


def _build_transform_pipeline(mode: str) -> list[Any]:
    data_config = dataset_config["bridge_tfds_train"]
    transforms = build_transforms(
        train_config,
        mode,
        data_config["kinematics_config"],
        data_config["T_base2world"],
        data_config["scale_shift"],
        data_config["num_joint"],
    )
    return [build(t) if t is not None else None for t in transforms]


def _apply_transforms(data: dict[str, Any], transforms: list[Any]) -> dict[str, Any]:
    data = _clone_data(data)
    for transform in transforms:
        if transform is None:
            continue
        data = transform(data)
    return data


def _to_numpy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _compare_value(name: str, left: Any, right: Any) -> CompareResult:
    left = _to_numpy(left)
    right = _to_numpy(right)

    if isinstance(left, str) or isinstance(right, str):
        ok = left == right
        return CompareResult(name, ok, f"left={left!r}, right={right!r}")

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        left = np.asarray(left)
        right = np.asarray(right)

    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape:
            return CompareResult(
                name,
                False,
                f"shape mismatch: {left.shape} vs {right.shape}",
            )
        if np.array_equal(left, right):
            return CompareResult(name, True, f"exact match {left.shape}")
        if np.issubdtype(left.dtype, np.number) and np.issubdtype(right.dtype, np.number):
            diff = np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))
            ok = np.allclose(left, right, atol=1e-6, rtol=1e-6)
            return CompareResult(
                name,
                ok,
                f"allclose={ok}, shape={left.shape}, max_abs_diff={diff:.8g}",
            )
        return CompareResult(name, False, f"array mismatch for dtype {left.dtype}")

    ok = left == right
    return CompareResult(name, ok, f"left={left!r}, right={right!r}")


def _summarize(results: list[CompareResult]) -> tuple[bool, list[str]]:
    ok = all(r.ok for r in results)
    lines = []
    for result in results:
        status = "OK" if result.ok else "FAIL"
        lines.append(f"{status:4} {result.name}: {result.detail}")
    return ok, lines


def _build_raw_lmdb_dataset(lmdb_root: str, max_episodes: int | None) -> RoboTwinLmdbDataset:
    data_config = dataset_config["bridge_tfds_train"]
    return RoboTwinLmdbDataset(
        paths=[lmdb_root],
        transforms=[],
        lazy_init=False,
        num_episode=max_episodes,
        cam_names=data_config["cam_names"],
        T_base2world=data_config["T_base2world"],
        dataset_name="bridge_lmdb",
    )


def _select_indices(dataset_len: int, requested: list[int] | None) -> list[int]:
    if requested:
        return requested
    candidates = [0]
    if dataset_len > 2:
        candidates.append(dataset_len // 2)
    if dataset_len > 1:
        candidates.append(dataset_len - 1)
    return sorted(set(candidates))


def _check_index(
    index: int,
    left_dataset: RoboTwinLmdbDataset,
    right_dataset: RoboTwinLmdbDataset,
    transforms: list[Any],
) -> dict[str, Any]:
    left_raw = left_dataset[index]
    right_raw = right_dataset[index]

    raw_results = []
    for key in RAW_KEYS:
        raw_results.append(_compare_value(key, left_raw[key], right_raw[key]))

    left_ee_state = _to_numpy(left_raw.get("ee_state"))
    right_ee_state = _to_numpy(right_raw.get("ee_state"))
    if isinstance(left_ee_state, np.ndarray) and isinstance(right_ee_state, np.ndarray):
        raw_results.append(_compare_value("ee_state", left_ee_state, right_ee_state))

    left_transformed = _apply_transforms(left_raw, transforms)
    right_transformed = _apply_transforms(right_raw, transforms)

    transformed_results = []
    for key in TRANSFORMED_KEYS:
        transformed_results.append(
            _compare_value(key, left_transformed[key], right_transformed[key])
        )

    raw_ok, raw_lines = _summarize(raw_results)
    transformed_ok, transformed_lines = _summarize(transformed_results)
    return {
        "index": index,
        "raw_ok": raw_ok,
        "transformed_ok": transformed_ok,
        "raw_results": raw_lines,
        "transformed_results": transformed_lines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-lmdb-root", type=str, required=True)
    parser.add_argument("--right-lmdb-root", type=str, required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Global dataset indices to compare. Defaults to 0/middle/last.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="validation",
        choices=["training", "validation", "deploy"],
    )
    args = parser.parse_args()

    left_dataset = _build_raw_lmdb_dataset(args.left_lmdb_root, args.max_episodes)
    right_dataset = _build_raw_lmdb_dataset(args.right_lmdb_root, args.max_episodes)

    if len(left_dataset) != len(right_dataset):
        raise SystemExit(
            f"dataset length mismatch: left={len(left_dataset)} "
            f"right={len(right_dataset)}"
        )

    transforms = _build_transform_pipeline(args.mode)
    indices = _select_indices(len(left_dataset), args.indices)
    results = [
        _check_index(
            index=index,
            left_dataset=left_dataset,
            right_dataset=right_dataset,
            transforms=transforms,
        )
        for index in indices
    ]

    overall_ok = all(item["raw_ok"] and item["transformed_ok"] for item in results)
    output = {
        "overall_ok": overall_ok,
        "dataset_length": len(left_dataset),
        "checked_indices": indices,
        "results": results,
    }
    print(json.dumps(output, indent=2))

    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
