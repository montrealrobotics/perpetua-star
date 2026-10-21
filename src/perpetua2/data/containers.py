from typing import List, Optional, Union

from dataclasses import dataclass, field

import jax.numpy as jnp


@dataclass
class Sequences:
    """This dataclass is used to store the sequences used to train the filters"""
    name: str
    id: int
    persistence_states: List[jnp.ndarray] = field(default_factory=list)
    persistence_times: List[jnp.ndarray] = field(default_factory=list)
    emergence_states: List[jnp.ndarray] = field(default_factory=list)
    emergence_times: List[jnp.ndarray] = field(default_factory=list)

    def add_persistence_sequence(self, states: jnp.ndarray, times: jnp.ndarray):
        self.persistence_states.append(states)
        self.persistence_times.append(times)

    def add_emergence_sequence(self, states: jnp.ndarray, times: jnp.ndarray):
        self.emergence_states.append(states)
        self.emergence_times.append(times)

    def get_by_filter(self, key: str):
        if key not in ("persistence", "emergence"):
            raise ValueError(f"Unknown filter '{key}'")
        return getattr(self, f"{key}_states"), getattr(self, f"{key}_times")

    def to_jax(self):
        """convert containers to jax"""
        self.persistence_states = [jnp.asarray(arr) for arr in self.persistence_states]
        self.persistence_times = [jnp.asarray(arr) for arr in self.persistence_times]
        self.emergence_states = [jnp.asarray(arr) for arr in self.emergence_states]
        self.emergence_times = [jnp.asarray(arr) for arr in self.emergence_times]


@dataclass
class Receptacle:
    name: str
    id: int
    states: jnp.ndarray
    timestamps: jnp.ndarray
    gt_states: Optional[jnp.ndarray] = None
    gt_timestamps: Optional[jnp.ndarray] = None


@dataclass
class Pickupable:
    name: str
    id: int
    receptacles: List[Union[Receptacle, Sequences]] = field(default_factory=list)

    def get_by_name(self, name: str) -> Optional[Union[Receptacle, Sequences]]:
        for receptacle in self.receptacles:
            if receptacle.name == name:
                return receptacle
        return None

    def __getitem__(self, index: int) -> Union[Receptacle, Sequences]:
        return self.receptacles[index]

    def __iter__(self):
        for receptacle in self.receptacles:
            yield receptacle

    @property
    def n_receptacles(self) -> int:
        return len(self.receptacles)

    def add_receptacle(self, receptacle: Receptacle):
        self.receptacles.append(receptacle)


@dataclass
class Scene:
    name: str = "default"
    pickupables: List[Pickupable] = field(default_factory=list)

    @property
    def n_pickupables(self) -> int:
        return len(self.pickupables)

    def add_pickupable(self, pickupable: Pickupable):
        self.pickupables.append(pickupable)

    def get_by_name(self, name: str) -> Optional[Pickupable]:
        for pickupable in self.pickupables:
            if pickupable.name == name:
                return pickupable
        return None

    def __getitem__(self, index: int) -> Pickupable:
        return self.pickupables[index]

    def __iter__(self):
        for pickupable in self.pickupables:
            yield pickupable
