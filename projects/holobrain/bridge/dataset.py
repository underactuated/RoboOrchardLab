from __future__ import annotations

from typing import Any

import numpy as np

from .backend import BridgeEpisodeBackend, BridgeEpisodeStep


class BridgeHolobrainDataset:
    """
    Convert canonical Bridge episodes into RobotWin-like pre-transform samples.

    This wrapper intentionally keeps the Bridge data access backend separate
    from the Holobrain-facing sample mapping.
    """

    SINGLE_VIEW_SOURCE = "image_0"

    CAMERA_TARGET_NAMES = ["front_camera"]

    D435_FX = 282.0
    D435_FY = 282.0
    D435_CX = 112.0
    D435_CY = 112.0

    def __init__(
        self,
        backend: BridgeEpisodeBackend,
        constant_depth: float = 0.6,
    ):
        self.backend = backend
        self.constant_depth = float(constant_depth)
        self.index_map = self._build_index_map()

    def _build_index_map(self) -> list[tuple[int, int]]:
        index_map: list[tuple[int, int]] = []
        for episode_index in range(len(self.backend)):
            episode = self.backend.get_episode(episode_index)
            for step_index in range(len(episode.steps)):
                index_map.append((episode_index, step_index))
        return index_map

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, step_index = self.index_map[index]
        episode = self.backend.get_episode(episode_index)
        step = episode.steps[step_index]

        joint_state_trajectory = np.stack(
            [self._bridge_state_to_dual_arm(s.state) for s in episode.steps],
            axis=0,
        )

        imgs = self._stack_images(step)
        sample = {
            "uuid": episode.episode_id,
            "step_index": step_index,
            "text": self._normalize_language(step.language_instruction),
            "cam_names": list(self.CAMERA_TARGET_NAMES),
            "imgs": imgs,
            "depths": self._make_constant_depth(imgs),
            "intrinsic": self._make_intrinsics(imgs),
            "T_world2cam": self._make_world_to_cam(imgs.shape[0]),
            "T_base2world": self._make_base_to_world(),
            "joint_state": joint_state_trajectory,
        }
        return sample

    def _stack_images(self, step: BridgeEpisodeStep) -> np.ndarray:
        image_0 = step.images[self.SINGLE_VIEW_SOURCE]
        return np.expand_dims(image_0, axis=0)

    def _bridge_state_to_dual_arm(self, state7: np.ndarray) -> np.ndarray:
        state7 = np.asarray(state7, dtype=np.float64)
        if state7.shape != (7,):
            raise ValueError(
                f"Expected Bridge state shape (7,), got {state7.shape}"
            )
        arm = state7[:6].tolist()
        gripper = float(state7[6])
        return np.asarray(arm + [gripper] + arm + [gripper], dtype=np.float64)

    def _make_constant_depth(self, imgs: np.ndarray) -> np.ndarray:
        num_cams, height, width = imgs.shape[:3]
        return np.full(
            (num_cams, height, width),
            self.constant_depth,
            dtype=np.float32,
        )

    def _make_intrinsics(self, imgs: np.ndarray) -> np.ndarray:
        num_cams = imgs.shape[0]
        intrinsics = []
        for _ in range(num_cams):
            intrinsic = np.eye(4, dtype=np.float64)
            intrinsic[0, 0] = self.D435_FX
            intrinsic[1, 1] = self.D435_FY
            intrinsic[0, 2] = self.D435_CX
            intrinsic[1, 2] = self.D435_CY
            intrinsics.append(intrinsic)
        return np.stack(intrinsics, axis=0)

    def _make_world_to_cam(self, num_cams: int) -> np.ndarray:
        return np.stack(
            [np.eye(4, dtype=np.float64) for _ in range(num_cams)],
            axis=0,
        )

    def _make_base_to_world(self) -> np.ndarray:
        return np.eye(4, dtype=np.float64)

    def _normalize_language(self, value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)
