from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow_datasets as tfds


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _normalize_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _episode_np(example: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        key: _normalize_scalar(value)
        for key, value in example["episode_metadata"].items()
    }

    steps = list(example["steps"])
    if not steps:
        raise ValueError("Encountered empty Bridge episode.")

    images = {name: [] for name in ["image_0", "image_1", "image_2", "image_3"]}
    states = []
    actions = []
    language = ""

    for step in steps:
        obs = step["observation"]
        for name in images:
            images[name].append(np.asarray(obs[name]))
        states.append(np.asarray(obs["state"], dtype=np.float32))
        actions.append(np.asarray(step["action"], dtype=np.float32))
        if not language:
            language = _normalize_text(step.get("language_instruction"))

    episode_id = metadata.get("episode_id")
    if episode_id is None:
        episode_id = metadata.get("file_path", "unknown_episode")

    output = {
        "episode_id": np.asarray(str(episode_id)),
        "language_instruction": np.asarray(language),
        "states": np.stack(states, axis=0),
        "actions": np.stack(actions, axis=0),
        "num_steps": np.asarray(len(steps), dtype=np.int32),
        "episode_metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for name, frames in images.items():
        output[name] = np.stack(frames, axis=0)
    return output


def export_split(
    data_dir: str,
    split: str,
    output_dir: Path,
    max_episodes: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = tfds.load(
        "bridge_dataset",
        data_dir=data_dir,
        split=split,
        shuffle_files=False,
    )

    count = 0
    for example in tfds.as_numpy(ds):
        episode = _episode_np(example)
        episode_id = str(episode["episode_id"].item())
        filename = output_dir / f"episode_{count:06d}_{episode_id}.npz"
        np.savez_compressed(filename, **episode)
        count += 1
        if count % 10 == 0:
            print(f"exported {count} episodes to {output_dir}")
        if max_episodes is not None and count >= max_episodes:
            break

    print(f"done: exported {count} episodes to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/jerry/datasets/openx/resize_224",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./bridge_npz_export"),
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    args = parser.parse_args()

    export_split(
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
