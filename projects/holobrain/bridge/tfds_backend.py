from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from .backend import BridgeEpisode, BridgeEpisodeBackend, BridgeEpisodeStep


class LocalTfdsBridgeBackend(BridgeEpisodeBackend):
    """
    TFDS-backed Bridge episode backend.

    This backend reads the local Bridge dataset from a TFDS-style directory and
    converts each episode into the canonical BridgeEpisode / BridgeEpisodeStep
    structures used by BridgeHolobrainDataset.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_episodes: int | None = None,
        cache_size: int = 8,
    ):
        import tensorflow_datasets as tfds

        self.data_dir = data_dir
        self.split = split
        self.max_episodes = max_episodes
        self.cache_size = int(cache_size)
        self._tfds = tfds
        self._cache: OrderedDict[int, BridgeEpisode] = OrderedDict()

        builder = tfds.builder("bridge_dataset", data_dir=self.data_dir)
        split_info = builder.info.splits[self.split]
        self._num_episodes = split_info.num_examples
        if self.max_episodes is not None:
            self._num_episodes = min(self._num_episodes, self.max_episodes)

    def __len__(self) -> int:
        return self._num_episodes

    def get_episode(self, episode_index: int) -> BridgeEpisode:
        if episode_index < 0 or episode_index >= len(self):
            raise IndexError(
                f"episode_index {episode_index} out of range [0, {len(self)})"
            )
        cached = self._cache.get(episode_index)
        if cached is not None:
            self._cache.move_to_end(episode_index)
            return cached

        episode = self._load_episode(episode_index)
        self._cache[episode_index] = episode
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return episode

    def _load_episode(self, episode_index: int) -> BridgeEpisode:
        ds = self._tfds.load(
            "bridge_dataset",
            data_dir=self.data_dir,
            split=f"{self.split}[{episode_index}:{episode_index + 1}]",
            shuffle_files=False,
        )
        try:
            raw_episode = next(iter(self._tfds.as_numpy(ds)))
        except StopIteration as exc:
            raise IndexError(
                f"Failed to load episode at index {episode_index}"
            ) from exc
        episode = raw_episode
        metadata = self._normalize_metadata(episode["episode_metadata"])

        episode_id = metadata.get("episode_id")
        if episode_id is None:
            episode_id = metadata.get("file_path", f"episode_{episode_index}")
        episode_id = str(episode_id)

        steps = [self._normalize_step(step) for step in episode["steps"]]
        return BridgeEpisode(
            episode_id=episode_id,
            steps=steps,
            raw_metadata=metadata,
        )

    def _normalize_step(self, step: dict[str, Any]) -> BridgeEpisodeStep:
        observation = step["observation"]
        images = {
            key: np.asarray(observation[key])
            for key in ["image_0", "image_1", "image_2", "image_3"]
        }
        state = np.asarray(observation["state"], dtype=np.float64)
        action = np.asarray(step["action"], dtype=np.float64)
        language_instruction = self._normalize_text(
            step.get("language_instruction", b"")
        )

        raw_metadata = {
            key: self._normalize_scalar(value)
            for key, value in step.items()
            if key not in {"observation", "action", "language_instruction"}
        }
        return BridgeEpisodeStep(
            images=images,
            state=state,
            action=action,
            language_instruction=language_instruction,
            raw_metadata=raw_metadata,
        )

    def _normalize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {
            key: self._normalize_scalar(value) for key, value in metadata.items()
        }

    def _normalize_scalar(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _normalize_text(self, value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)
