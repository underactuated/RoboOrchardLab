from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


IMAGE_KEYS = {"image_0", "image_1", "image_2", "image_3"}


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _summarize_array(value: np.ndarray, include_values: bool) -> dict[str, Any]:
    value = np.asarray(value)
    summary: dict[str, Any] = {
        "type": "ndarray",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    if value.size == 0:
        summary["values"] = []
        return summary
    if include_values:
        summary["values"] = value.tolist()
    elif np.issubdtype(value.dtype, np.number):
        summary["min"] = value.min().item()
        summary["max"] = value.max().item()
    return summary


def _sanitize(
    value: Any,
    include_values: bool,
    strip_images: bool,
) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if strip_images and key in IMAGE_KEYS:
                continue
            output[key] = _sanitize(
                item,
                include_values=include_values,
                strip_images=strip_images,
            )
        return output

    if isinstance(value, (list, tuple)):
        return [
            _sanitize(
                item,
                include_values=include_values,
                strip_images=strip_images,
            )
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return _summarize_array(value, include_values=include_values)

    return _normalize_scalar(value)


def inspect_episode(
    data_dir: str,
    split: str,
    episode_index: int,
    include_values: bool,
    num_steps: int | None,
) -> dict[str, Any]:
    import tensorflow_datasets as tfds

    ds = tfds.load(
        "bridge_dataset",
        data_dir=data_dir,
        split=f"{split}[{episode_index}:{episode_index + 1}]",
        shuffle_files=False,
    )
    try:
        raw_episode = next(iter(tfds.as_numpy(ds)))
    except StopIteration as exc:
        raise IndexError(f"Episode index {episode_index} not found in {split}") from exc

    episode_metadata = _sanitize(
        raw_episode.get("episode_metadata", {}),
        include_values=True,
        strip_images=False,
    )
    raw_steps = list(raw_episode.get("steps", []))
    total_num_steps = len(raw_steps)
    if num_steps is not None:
        raw_steps = raw_steps[:num_steps]

    steps = []
    for step_index, step in enumerate(raw_steps):
        step_clean = _sanitize(
            step,
            include_values=include_values,
            strip_images=True,
        )
        step_clean["__step_index__"] = step_index
        steps.append(step_clean)

    return {
        "episode_index": episode_index,
        "num_steps": len(raw_steps),
        "total_num_steps": total_num_steps,
        "episode_metadata": episode_metadata,
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/jerry/datasets/openx/resize_224",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument(
        "--episode-start",
        type=int,
        default=None,
        help="Start episode index for a range (inclusive).",
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        default=None,
        help="End episode index for a range (exclusive).",
    )
    parser.add_argument(
        "--include-values",
        action="store_true",
        help="Include full ndarray values for non-image arrays.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Only include the first N steps from the episode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path. Prints to stdout if omitted.",
    )
    args = parser.parse_args()

    using_range = args.episode_start is not None or args.episode_end is not None
    if using_range:
        if args.episode_index is not None:
            raise ValueError(
                "--episode-index cannot be used together with "
                "--episode-start/--episode-end"
            )
        if args.episode_start is None or args.episode_end is None:
            raise ValueError(
                "Both --episode-start and --episode-end are required when "
                "inspecting a range."
            )
        if args.episode_end <= args.episode_start:
            raise ValueError("--episode-end must be greater than --episode-start.")
        num_steps = 1 if args.num_steps is None else args.num_steps
        report = [
            inspect_episode(
                data_dir=args.data_dir,
                split=args.split,
                episode_index=episode_index,
                include_values=args.include_values,
                num_steps=num_steps,
            )
            for episode_index in range(args.episode_start, args.episode_end)
        ]
    else:
        if args.episode_index is None:
            raise ValueError(
                "Either --episode-index or --episode-start/--episode-end is required."
            )
        report = inspect_episode(
            data_dir=args.data_dir,
            split=args.split,
            episode_index=args.episode_index,
            include_values=args.include_values,
            num_steps=args.num_steps,
        )

    text = json.dumps(report, indent=2, sort_keys=False)
    if args.output is None:
        print(text)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
