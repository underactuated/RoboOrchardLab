from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .backend import BridgeEpisode, BridgeEpisodeBackend, BridgeEpisodeStep


class NpzBridgeBackend(BridgeEpisodeBackend):
    """
    TF-free Bridge backend reading one exported `.npz` file per episode.
    """

    def __init__(
        self,
        root: str,
        max_episodes: int | None = None,
        cache_size: int = 8,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Bridge export root does not exist: {self.root}")
        self.paths = sorted(self.root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz files found under {self.root}")
        if max_episodes is not None:
            self.paths = self.paths[:max_episodes]
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, BridgeEpisode] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def get_episode(self, episode_index: int) -> BridgeEpisode:
        if episode_index < 0 or episode_index >= len(self):
            raise IndexError(
                f"episode_index {episode_index} out of range [0, {len(self)})"
            )
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached

        episode = self._load_episode(self.paths[episode_index])
        self._cache[episode_index] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def _load_episode(self, path: Path) -> BridgeEpisode:
        arr = np.load(path, allow_pickle=False)
        states = arr["states"]
        actions = arr["actions"]
        image_0 = arr["image_0"]
        image_1 = arr["image_1"]
        image_2 = arr["image_2"]
        image_3 = arr["image_3"]
        language_instruction = self._normalize_text(
            arr["language_instruction"].item()
        )
        episode_id = str(arr["episode_id"].item())
        metadata = json.loads(arr["episode_metadata_json"].item())

        num_steps = int(arr["num_steps"].item())
        steps = []
        for step_index in range(num_steps):
            steps.append(
                BridgeEpisodeStep(
                    images={
                        "image_0": image_0[step_index],
                        "image_1": image_1[step_index],
                        "image_2": image_2[step_index],
                        "image_3": image_3[step_index],
                    },
                    state=np.asarray(states[step_index], dtype=np.float64),
                    action=np.asarray(actions[step_index], dtype=np.float64),
                    language_instruction=language_instruction,
                    raw_metadata={"path": str(path), "step_index": step_index},
                )
            )
        return BridgeEpisode(
            episode_id=episode_id,
            steps=steps,
            raw_metadata=metadata,
        )

    def _normalize_text(self, value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)
