from typing import List, Union, Dict, Optional, Any
import dill as pickle

import jax.lax
import jax.numpy as jnp
from flax import struct
from jax.tree_util import Partial
from jax.typing import ArrayLike

from perpetua2.utils.jax_utils import logdiff
from perpetua2.filters.Fremen import FremenPrior
from perpetua2.utils.priors import PiecewisePrior

import logging

logger = logging.getLogger(__name__)


@struct.dataclass
class FilterState:
    # This is set at the start
    log_s: Partial
    params: Dict[str, Union[float, ArrayLike]]
    initialization_time: float

    # Update function modifies this
    last_observation_time: float
    log_likelihood: float = 0.0
    log_lower_evidence_sum: Union[float, ArrayLike] = 0.0
    log_conditional_evidence: Union[float, ArrayLike] = 0.0
    log_lower_evidence_sum_is_none: bool = True
    decay_rate: float = 1.0
    n_samples: int = 0

    # used for mixtures
    pi: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([1.0]))
    # Used only when resetting the mixture of perpetua filter
    pi_init: jnp.ndarray = struct.field(default_factory=lambda: jnp.array([1.0]))

    def __getitem__(self, item):
        return FilterState(
            log_s=self.log_s,
            params={key: value[item] for key, value in self.params.items()},
            initialization_time=self.initialization_time,
            last_observation_time=self.last_observation_time,
            log_likelihood=self.log_likelihood,
            log_lower_evidence_sum=self.log_lower_evidence_sum[item][None],
            log_conditional_evidence=self.log_conditional_evidence[item][None],
            log_lower_evidence_sum_is_none=self.log_lower_evidence_sum_is_none,
            pi=self.pi[item][None],
            decay_rate=self.decay_rate,
            n_samples=self.n_samples,
        )

    @property
    def log_joint_evidence(self):
        return self.log_conditional_evidence + jnp.log(self.pi)

    @property
    def log_evidence(self):
        return jax.scipy.special.logsumexp(self.log_joint_evidence)

    @property
    def log_tau(self):
        return self.log_joint_evidence - self.log_evidence

    @staticmethod
    def create(
        log_s: Partial,
        params: Dict[str, float],
        decay_rate: float = 1.0,
        initialization_time: float = 0.0,
        pi=jnp.ones(1),
        n_samples: int = 0,
    ):
        self = FilterState(log_s, params, initialization_time, last_observation_time=initialization_time, pi=pi)

        self = self.replace(
            log_lower_evidence_sum=jnp.zeros_like(pi),
            log_conditional_evidence=jnp.zeros_like(pi),
            pi_init=pi,
            decay_rate=decay_rate,
            n_samples=n_samples,
        )
        return self

    @property
    def num_components(self):
        return self.pi.shape[0]

    @property
    def shifted_log_survival_function(self):
        return lambda t: self.log_s(t - self.initialization_time, **self.params)

    @property
    def shifted_logdF(self):
        return lambda t1, t0: logdiff(self.shifted_log_survival_function(t0), self.shifted_log_survival_function(t1))

    def reset(self, initialization_time: Optional[float] = 0.0):
        return self.replace(
            initialization_time=initialization_time,
            last_observation_time=initialization_time,
            log_likelihood=0.0,
            log_lower_evidence_sum=jnp.zeros_like(self.pi_init),
            log_conditional_evidence=jnp.zeros_like(self.pi_init),
            log_lower_evidence_sum_is_none=True,
            pi=self.pi_init,
            pi_init=self.pi_init,
        )

    @property
    def state_dict(self) -> Dict[str, Any]:
        return {
            "params": self.params,
            "n_samples": self.n_samples,
            "pi": self.pi_init,
        }


@struct.dataclass
class PerpetuaState:
    initialization_time: float
    last_observation_time: float

    delta_low: float
    delta_high: float
    num_steps: int = struct.field(pytree_node=False)  # Make num_steps static

    ef_state: FilterState
    pf_state: FilterState

    # Mixing parameters
    eps: float = 1e-1

    initial_state: int = 1
    current_state: bool = None  # note: set in code to be ``initial_state'' in the create func
    switch_counter: int = 1
    weight: Union[ArrayLike, float] = 1.0

    @staticmethod
    def create(
        ef_state: FilterState,
        pf_state: FilterState,
        initialization_time: float = 0.0,
        delta_high: float = 0.95,
        delta_low: float = 0.05,
        num_steps: int = 1,
        epsilon: float = 1e-1,
    ):
        ret = PerpetuaState(
            delta_low=delta_low,
            delta_high=delta_high,
            initialization_time=initialization_time,
            last_observation_time=initialization_time,
            num_steps=num_steps,
            ef_state=ef_state,
            pf_state=pf_state,
        )
        return ret.replace(current_state=ret.initial_state, eps=epsilon)


@struct.dataclass
class PerpetuaStarState:
    initialization_time: float
    last_observation_time: float
    max_period: float
    # Switching prior
    switching_state: Union[FremenPrior, PiecewisePrior]

    ef_state: FilterState
    pf_state: FilterState
    alpha: float = 1.0  # Evidence annealing
    reset_counter: int = 0  # Counts the number of resets after reaching t > max_period
    # (Receptacle) metadata
    id: int = struct.field(pytree_node=False, default=-1)
    name: str = struct.field(pytree_node=False, default="")

    @property
    def switching_prior(self):
        return lambda t: self.switching_state.function(self.switching_state.state, t, **self.switching_state.kwargs)

    @staticmethod
    def create(
        ef_state: FilterState,
        pf_state: FilterState,
        switching_state: Union[FremenPrior, PiecewisePrior],
        alpha: float = 1.0,
        max_period: float = jnp.inf,
        id: int = -1,
        name: str = "",
        initialization_time: float = 0.0,
    ):
        out = PerpetuaStarState(
            initialization_time=initialization_time,
            last_observation_time=initialization_time,
            max_period=max_period,
            switching_state=switching_state,
            ef_state=ef_state,
            pf_state=pf_state,
            alpha=alpha,
            id=id,
            name=name,
        )
        return out

    def reset(self, initialization_time: Optional[float] = 0.0):
        return self.replace(
            initialization_time=initialization_time,
            last_observation_time=initialization_time,
            ef_state=self.ef_state.reset(initialization_time),
            pf_state=self.pf_state.reset(initialization_time),
            reset_counter=0,
        )


@struct.dataclass
class Object:
    # The list of states (one per receptacle).
    states: List[PerpetuaStarState | FilterState]
    # Metadata for the semi-static object itself (e.g., "Mug")
    name: str = struct.field(pytree_node=False)
    id: str = struct.field(pytree_node=False)

    @property
    def num_receptacles(self):
        return len(self.states)

    @property
    def receptacle_names(self) -> List[str]:
        return [s.name for s in self.states]

    @property
    def get_receptacles(self) -> List[PerpetuaStarState | FilterState]:
        if len(self.states) == 0:
            logger.warning(f"Object {self.name} has no receptacles.")
            return None
        elif len(self.states) > 1:
            logger.warning(f"Object {self.name} has multiple receptacles. Returning a list with all states.")
            return self.states
        return self.states[0]

    def update_alpha(self, new_alpha: float):
        new_states = []
        for state in self.states:
            new_states.append(state.replace(alpha=new_alpha))
        return self.replace(states=new_states)

    def update_decay_rate(self, new_decay_rate: float):
        # Call before giving the model any new data
        new_states = []
        for state in self.states:
            new_pf_state = state.pf_state.replace(decay_rate=new_decay_rate)
            new_ef_state = state.ef_state.replace(decay_rate=new_decay_rate)
            new_states.append(state.replace(pf_state=new_pf_state, ef_state=new_ef_state))
        return self.replace(states=new_states)

    def __iter__(self):
        # Iterate over receptacle states
        for state in self.states:
            yield state

    def select(self, target: Union[str, List[str]]):
        """
        Selects state(s) by receptacle name.
        - Input 'str' -> Returns single PerpetuaStarState
        - Input 'List[str]' -> Returns new Object (subset)
        """
        # Normalize input to list to share logic
        target_names = [target] if isinstance(target, str) else target

        # This ensures we return states in the same order as target_names
        state_map = {s.name: s for s in self.states}

        matched_states = []
        missing = []

        for name in target_names:
            if name in state_map:
                matched_states.append(state_map[name])
            else:
                missing.append(name)

        if missing:
            logger.warning(f"Receptacles not found: {missing} in {self.name}, returning available states.")
            raise AssertionError(f"Receptacles not found: {missing} in {self.name}")

        # Return subset Object if input was list
        return self.replace(states=matched_states)

    def __getitem__(self, index: Union[int, slice]):
        """Slicing is now much cleaner since we only have one list."""
        if isinstance(index, slice):
            return self.replace(states=self.states[index])
        elif isinstance(index, int):
            # Return a new Object containing just that one state
            return self.replace(states=[self.states[index]])
        else:
            raise TypeError("Index must be int or slice")

    def add_receptacle(self, state: PerpetuaStarState):
        """
        Adds a new receptacle.
        Updates the passed 'state' with the correct metadata before storing.
        """
        # Check if it already exists
        new_receptacle = state.name
        existing_names = self.receptacle_names
        if new_receptacle in existing_names:
            logger.info(
                f"Receptacle {new_receptacle} already exists in object {self.name}. Updating existing receptacle."
            )
            return self.update_receptacle(state, new_receptacle)

        new_states = self.states + [state]
        return self.replace(states=new_states)

    def update_receptacle(self, state: PerpetuaStarState):
        """Updates the state of an existing receptacle."""
        # Find index
        try:
            # We construct the list of names just for the lookup
            idx = self.receptacle_names.index(state.name)
        except ValueError:
            logger.warning(f"Receptacle {state.name} not found in object {self.name}. No update performed.")
            return self

        new_states = list(self.states)
        new_states[idx] = state

        return self.replace(states=new_states)

    def create(states: List[PerpetuaStarState], name: str, id):
        return Object(states=states, name=name, id=id)

    def dump(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)
