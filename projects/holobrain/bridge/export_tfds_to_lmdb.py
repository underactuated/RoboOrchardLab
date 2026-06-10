from __future__ import annotations

import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Iterator
import sys

import cv2
import lmdb
import numpy as np

logger = logging.getLogger(__name__)


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


class SimpleLmdb:
    def __init__(
        self,
        uri: str,
        writable: bool = True,
        commit_step: int = 1,
        map_size: int | None = None,
        encoding_mode: str = "utf-8",
    ):
        self.uri = uri
        self.writable = writable
        self.commit_step = int(commit_step)
        self.encoding_mode = encoding_mode
        if map_size is None:
            map_size = 1024**4 if writable else 10485760
        self.env = lmdb.open(
            uri,
            map_size=map_size,
            meminit=False,
            map_async=True,
            sync=False,
            readonly=not writable,
            lock=not writable,
        )
        self.txn = self.env.begin(write=writable)
        self.put_idx = 0

    def write(self, idx: int | str, record: Any, commit: bool = False):
        key = f"{idx}".encode(self.encoding_mode)
        payload = pickle.dumps(record, protocol=4)
        self.txn.put(key, payload)
        self.put_idx += 1
        if (self.put_idx % self.commit_step == 0) or commit:
            self.txn.commit()
            self.txn = self.env.begin(write=self.writable)

    def close(self):
        if self.txn is not None:
            self.txn.commit()
            self.txn = None
        if self.env is not None:
            self.env.sync()
            self.env.close()
            self.env = None


class BridgeTfdsToLmdbPacker:
    """
    Convert Bridge TFDS episodes directly into RoboTwin-style LMDB.

    This mirrors the currently validated Bridge mapping:
    - use only image_0 as a single shoulder-camera view
    - duplicate single-arm 7D state into dual-arm 14D joint state
    - use constant 0.6m depth
    - use fixed D435 intrinsics
    - keep identity extrinsics/base transform placeholders for now
    """

    SINGLE_VIEW_SOURCE = "image_0"
    CAMERA_TARGET_NAMES = ["front_camera"]

    D435_FX = 282.0
    D435_FY = 282.0
    D435_CX = 112.0
    D435_CY = 112.0

    def __init__(
        self,
        data_dir: str,
        split: str,
        output_path: str,
        episode_start: int | None = None,
        episode_end: int | None = None,
        max_episodes: int | None = None,
        task_name: str = "bridge",
        constant_depth: float = 0.6,
        simulation: bool = False,
        image_ext: str = ".png",
        commit_step: int = 500,
        **kwargs,
    ):
        self.data_dir = str(data_dir)
        self.split = str(split)
        self.output_path = str(output_path)
        self.episode_start = 0 if episode_start is None else int(episode_start)
        self.episode_end = (
            None if episode_end is None else int(episode_end)
        )
        self.max_episodes = max_episodes
        self.task_name = str(task_name)
        self.constant_depth = float(constant_depth)
        self.simulation = bool(simulation)
        self.image_ext = str(image_ext).lower()
        self.commit_step = int(commit_step)
        self.lmdb_kwargs = kwargs
        if self.image_ext not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"Unsupported image extension: {self.image_ext}")
        if self.episode_start < 0:
            raise ValueError("--episode-start must be >= 0")
        if self.episode_end is not None and self.episode_end <= self.episode_start:
            raise ValueError("--episode-end must be greater than --episode-start")
        if self.max_episodes is not None and self.max_episodes <= 0:
            raise ValueError("--max-episodes must be > 0")

    def _iter_episodes(self) -> Iterator[dict[str, Any]]:
        import tensorflow_datasets as tfds

        ds = tfds.load(
            "bridge_dataset",
            data_dir=self.data_dir,
            split=self._effective_split(),
            shuffle_files=False,
        )
        count = 0
        for example in tfds.as_numpy(ds):
            yield self._episode_np(example)
            count += 1
            if self.max_episodes is not None and count >= self.max_episodes:
                break

    def _effective_split(self) -> str:
        if self.episode_end is None:
            if self.episode_start == 0:
                return self.split
            return f"{self.split}[{self.episode_start}:]"
        return f"{self.split}[{self.episode_start}:{self.episode_end}]"

    def _init_lmdbs(self):
        for name in ["index", "meta", "image", "depth"]:
            uri = os.path.join(self.output_path, name)
            os.makedirs(uri, exist_ok=True)
            setattr(
                self,
                f"{name}_pack_file",
                SimpleLmdb(
                    uri=uri,
                    writable=True,
                    commit_step=self.commit_step,
                ),
            )

    def close(self):
        self.index_pack_file.close()
        self.meta_pack_file.close()
        self.image_pack_file.close()
        self.depth_pack_file.close()

    def write_index(self, index: int | str, index_data: dict[str, Any]):
        self.index_pack_file.write(index, index_data)

    def __call__(self):
        self._init_lmdbs()
        self._pack()

    def _pack(self):
        num_valid_ep = 0
        for episode_index, episode in enumerate(self._iter_episodes()):
            uuid = str(episode["episode_id"])
            num_steps = int(episode["num_steps"])
            logger.info(
                "start process [%s] %s",
                episode_index + 1,
                uuid,
            )
            if num_steps == 0:
                logger.warning("skip empty episode %s", uuid)
                continue

            first_imgs = self._stack_images(episode, 0)
            intrinsic = self._make_intrinsic_dict()
            extrinsic = self._make_extrinsic_dict()
            joint_positions = np.stack(
                [self._bridge_state_to_dual_arm(state) for state in episode["states"]],
                axis=0,
            )
            cartesian_positions = np.zeros((num_steps, 0), dtype=np.float64)

            for step_index in range(num_steps):
                imgs = self._stack_images(episode, step_index)
                depths = self._make_constant_depth(imgs.shape[1], imgs.shape[2])
                for cam_idx, cam_name in enumerate(self.CAMERA_TARGET_NAMES):
                    image_bytes = self._encode_image(imgs[cam_idx])
                    depth_bytes = self._encode_depth(depths)
                    self.image_pack_file.write(
                        f"{uuid}/{cam_name}/{step_index}",
                        image_bytes,
                    )
                    self.depth_pack_file.write(
                        f"{uuid}/{cam_name}/{step_index}",
                        depth_bytes,
                    )

            self.meta_pack_file.write(f"{uuid}/camera_names", self.CAMERA_TARGET_NAMES)
            self.meta_pack_file.write(f"{uuid}/extrinsic", extrinsic)
            self.meta_pack_file.write(f"{uuid}/intrinsic", intrinsic)
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/joint_positions",
                joint_positions,
            )
            self.meta_pack_file.write(
                f"{uuid}/observation/robot_state/cartesian_position",
                cartesian_positions,
            )
            self.meta_pack_file.write(f"{uuid}/instructions", episode["language_instruction"])
            self.meta_pack_file.write(
                f"{uuid}/meta_data",
                {
                    "uuid": uuid,
                    "task_name": self.task_name,
                    "source": "bridge_tfds",
                    "episode_metadata": episode["episode_metadata"],
                },
            )

            index_data = dict(
                uuid=uuid,
                task_name=self.task_name,
                num_steps=num_steps,
                simulation=self.simulation,
            )
            self.write_index(episode_index, index_data)
            num_valid_ep += 1
            logger.info(
                "finish process [%s] %s, num_steps:%s",
                episode_index + 1,
                uuid,
                num_steps,
            )

        self.index_pack_file.write("__len__", num_valid_ep, commit=True)
        self.close()

    def _episode_np(self, example: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            key: _normalize_scalar(value)
            for key, value in example["episode_metadata"].items()
        }

        steps = list(example["steps"])
        if not steps:
            raise ValueError("Encountered empty Bridge episode.")

        images = {self.SINGLE_VIEW_SOURCE: []}
        states = []
        actions = []
        language = ""

        for step in steps:
            obs = step["observation"]
            images[self.SINGLE_VIEW_SOURCE].append(np.asarray(obs[self.SINGLE_VIEW_SOURCE]))
            states.append(np.asarray(obs["state"], dtype=np.float32))
            actions.append(np.asarray(step["action"], dtype=np.float32))
            if not language:
                language = _normalize_text(step.get("language_instruction"))

        episode_id = metadata.get("episode_id")
        if episode_id is None:
            episode_id = metadata.get("file_path", "unknown_episode")

        return {
            "episode_id": str(episode_id),
            "language_instruction": language,
            "states": np.stack(states, axis=0),
            "actions": np.stack(actions, axis=0),
            "num_steps": len(steps),
            "episode_metadata": metadata,
            self.SINGLE_VIEW_SOURCE: np.stack(images[self.SINGLE_VIEW_SOURCE], axis=0),
        }

    def _stack_images(self, episode: dict[str, Any], step_index: int) -> np.ndarray:
        image_0 = episode[self.SINGLE_VIEW_SOURCE][step_index]
        return np.expand_dims(image_0, axis=0)

    def _bridge_state_to_dual_arm(self, state7: np.ndarray) -> np.ndarray:
        state7 = np.asarray(state7, dtype=np.float64)
        if state7.shape != (7,):
            raise ValueError(f"Expected Bridge state shape (7,), got {state7.shape}")
        arm = state7[:6].tolist()
        gripper = float(state7[6])
        return np.asarray(arm + [gripper] + arm + [gripper], dtype=np.float64)

    def _make_intrinsic_dict(self) -> dict[str, np.ndarray]:
        intrinsic = np.eye(3, dtype=np.float64)
        intrinsic[0, 0] = self.D435_FX
        intrinsic[1, 1] = self.D435_FY
        intrinsic[0, 2] = self.D435_CX
        intrinsic[1, 2] = self.D435_CY
        return {
            cam_name: intrinsic.copy()
            for cam_name in self.CAMERA_TARGET_NAMES
        }

    def _make_extrinsic_dict(self) -> dict[str, np.ndarray]:
        return {
            cam_name: np.eye(4, dtype=np.float64)
            for cam_name in self.CAMERA_TARGET_NAMES
        }

    def _make_constant_depth(self, height: int, width: int) -> np.ndarray:
        depth_m = np.full((height, width), self.constant_depth, dtype=np.float32)
        return np.round(depth_m * 1000.0).astype(np.uint16)

    def _encode_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        success, encoded = cv2.imencode(self.image_ext, image)
        if not success:
            raise RuntimeError(f"Failed to encode image with {self.image_ext}")
        return encoded

    def _encode_depth(self, depth_mm: np.ndarray) -> np.ndarray:
        success, encoded = cv2.imencode(".png", depth_mm)
        if not success:
            raise RuntimeError("Failed to encode depth image as PNG")
        return encoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/jerry/datasets/openx/resize_224",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--task-name", type=str, default="bridge")
    parser.add_argument("--constant-depth", type=float, default=0.6)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--commit-step", type=int, default=500)
    parser.add_argument("--image-ext", type=str, default=".png")
    args = parser.parse_args()

    BridgeTfdsToLmdbPacker(
        data_dir=args.data_dir,
        split=args.split,
        output_path=args.output_path,
        episode_start=args.episode_start,
        episode_end=args.episode_end,
        max_episodes=args.max_episodes,
        task_name=args.task_name,
        constant_depth=args.constant_depth,
        simulation=args.simulation,
        commit_step=args.commit_step,
        image_ext=args.image_ext,
    )()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    main()
