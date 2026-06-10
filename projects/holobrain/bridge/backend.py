from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class BridgeEpisodeStep:
    """Canonical Bridge step representation returned by a backend."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    action: np.ndarray
    language_instruction: str
    raw_metadata: dict[str, Any] | None = None


@dataclass
class BridgeEpisode:
    """Canonical Bridge episode representation returned by a backend."""

    episode_id: str
    steps: list[BridgeEpisodeStep]
    raw_metadata: dict[str, Any] | None = None


class BridgeEpisodeBackend(Protocol):
    """Replaceable Bridge data source."""

    def __len__(self) -> int:
        ...

    def get_episode(self, episode_index: int) -> BridgeEpisode:
        ...
