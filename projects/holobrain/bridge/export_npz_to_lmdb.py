from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

from robo_orchard_lab.dataset.lmdb.base_lmdb_dataset import (
    BaseLmdbManipulationDataPacker,
)
from robo_orchard_lab.utils import log_basic_config

try:
    from .npz_backend import NpzBridgeBackend
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from bridge.npz_backend import NpzBridgeBackend

logger = logging.getLogger(__name__)


class BridgeNpzToLmdbPacker(BaseLmdbManipulationDataPacker):
    """
    Convert exported Bridge NPZ episodes into RobotWin-style LMDB.

    The output intentionally mirrors the current Bridge adapter path so that:
    - state[7] is duplicated into a dual-arm 14D joint trajectory
    - only image_0 is used as a single shoulder-camera view
    - depth is a constant synthetic 0.6m map
    - D435 intrinsics are fixed from the current hand-estimated values
    - world/camera extrinsics are identity
    """

    SINGLE_VIEW_SOURCE = "image_0"

    CAMERA_TARGET_NAMES = ["front_camera"]

    D435_FX = 282.0
    D435_FY = 282.0
    D435_CX = 112.0
    D435_CY = 112.0

    def __init__(
        self,
        input_path: str,
        output_path: str,
        max_episodes: int | None = None,
        task_name: str = "bridge",
        constant_depth: float = 0.6,
        simulation: bool = False,
        image_ext: str = ".png",
        **kwargs,
    ):
        super().__init__(input_path=input_path, output_path=output_path, **kwargs)
        self.backend = NpzBridgeBackend(root=input_path, max_episodes=max_episodes)
        self.task_name = str(task_name)
        self.constant_depth = float(constant_depth)
        self.simulation = bool(simulation)
        self.image_ext = str(image_ext).lower()
        if self.image_ext not in {".png", ".jpg", ".jpeg"}:
            raise ValueError(f"Unsupported image extension: {self.image_ext}")

    def _pack(self):
        num_valid_ep = 0
        for episode_index in range(len(self.backend)):
            episode = self.backend.get_episode(episode_index)
            uuid = str(episode.episode_id)
            num_steps = len(episode.steps)
            logger.info(
                "start process [%s/%s] %s",
                episode_index + 1,
                len(self.backend),
                uuid,
            )
            if num_steps == 0:
                logger.warning("skip empty episode %s", uuid)
                continue

            first_imgs = self._stack_images(episode.steps[0])
            intrinsic = self._make_intrinsic_dict(first_imgs)
            extrinsic = self._make_extrinsic_dict()
            joint_positions = np.stack(
                [self._bridge_state_to_dual_arm(step.state) for step in episode.steps],
                axis=0,
            )
            # Preserve parity with the current Bridge adapter path, which does
            # not provide ee_state/cartesian supervision.
            cartesian_positions = np.zeros((num_steps, 0), dtype=np.float64)
            instructions = self._normalize_text(
                episode.steps[0].language_instruction if episode.steps else ""
            )

            for step_index, step in enumerate(episode.steps):
                imgs = self._stack_images(step)
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
            self.meta_pack_file.write(f"{uuid}/instructions", instructions)
            self.meta_pack_file.write(
                f"{uuid}/meta_data",
                {
                    "uuid": uuid,
                    "task_name": self.task_name,
                    "source": "bridge_npz",
                    "source_path": str(self.backend.paths[episode_index]),
                    "episode_metadata": episode.raw_metadata,
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
                "finish process [%s/%s] %s, num_steps:%s",
                episode_index + 1,
                len(self.backend),
                uuid,
                num_steps,
            )

        self.index_pack_file.write("__len__", num_valid_ep, commit=True)
        self.close()

    def _stack_images(self, step) -> np.ndarray:
        image_0 = step.images[self.SINGLE_VIEW_SOURCE]
        return np.expand_dims(image_0, axis=0)

    def _bridge_state_to_dual_arm(self, state7: np.ndarray) -> np.ndarray:
        state7 = np.asarray(state7, dtype=np.float64)
        if state7.shape != (7,):
            raise ValueError(f"Expected Bridge state shape (7,), got {state7.shape}")
        arm = state7[:6].tolist()
        gripper = float(state7[6])
        return np.asarray(arm + [gripper] + arm + [gripper], dtype=np.float64)

    def _make_intrinsic_dict(self, imgs: np.ndarray) -> dict[str, np.ndarray]:
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

    def _normalize_text(self, value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--task_name", type=str, default="bridge")
    parser.add_argument("--constant_depth", type=float, default=0.6)
    parser.add_argument("--simulation", action="store_true")
    parser.add_argument("--commit_step", type=int, default=500)
    parser.add_argument("--image_ext", type=str, default=".png")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    BridgeNpzToLmdbPacker(
        input_path=str(input_path),
        output_path=args.output_path,
        max_episodes=args.max_episodes,
        task_name=args.task_name,
        constant_depth=args.constant_depth,
        simulation=args.simulation,
        commit_step=args.commit_step,
        image_ext=args.image_ext,
    )()


if __name__ == "__main__":
    log_basic_config(
        format="%(asctime)s %(levelname)s:%(lineno)d %(message)s",
        level=logging.INFO,
    )
    main()
