from typing import Tuple, Dict, Union, Optional, List

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jax.tree_util import Partial

from perpetua2.utils.filter_state import FilterState, Object
from perpetua2.filters import PersistenceFilter


def init(
    log_s: Partial,
    params: Dict[str, ArrayLike],
    pi: ArrayLike,
    decay_rate: float = 1.0,
    initialization_time: float = 0.0,
    n_samples: int = 0,
) -> FilterState:
    return FilterState.create(log_s, params, decay_rate, initialization_time, pi=pi, n_samples=n_samples)


@jax.jit
def update(
    mixture_state: FilterState, detector_output: bool, observation_time: float, P_M: float, P_F: float
) -> FilterState:
    """
    Updates the mixture filter by incorporating a new detector output.

    Args:
        mixture_state (FilterState): The current state of the mixture filter.
        detector_output (bool): A Boolean value output by the detector indicating whether the given feature was detected.
        observation_time (float): The timestamp for the detection.
        P_M (float): A floating-point value in the range [0, 1] indicating the missed detection probability
        of the feature detector.
        P_F (float): A floating-point value in the range [0, 1] indicating the false alarm probability of the feature
        detector.

    Returns:
        FilterState: The updated state of the mixture filter.
    """

    return PersistenceFilter.update(mixture_state, detector_output, observation_time, P_M, P_F)


@jax.jit
def predict(mixture_state: FilterState, prediction_time: ArrayLike) -> Tuple[Array, Array]:
    """
    Predict the mixture persistence posterior probability of the mixture component with the largest posterior weight
    p(X_t = 1 | C_k^*, Y_{1:N}), and return the probability of all other components p(X_t = 1 | C_k, Y_{1:N}) for all k.

    Args:
        mixture_state (FilterState): The current state of the mixture filter.
        prediction_time (ArrayLike): A floating-point value or array of values in the range
        [last_observation_time, infinity) indicating the time(s) t for which to compute the posterior.

    Returns:
        Tuple[Array, Array]: Two JAX arrays. The first array contains the component with the larsets posterior weight
        with shape (1, prediction_time.size). The second array contains the conditional posterior probabilities
        with shape (n_components, prediction_time.size).
    """

    predictions = PersistenceFilter.predict(mixture_state, prediction_time)
    return jnp.atleast_2d(predictions[jnp.argmax(mixture_state.log_tau)]), predictions


def object_update(
    state: Object,
    detector_output: jnp.ndarray,
    observation_time: jnp.ndarray,
    p_m: float,
    p_f: float,
    receptacle_names: Optional[Union[str, List[str]]] = None,
) -> Object:
    """Update function for an object, wrapping the underlying update function."""
    if receptacle_names is None:
        subset = state
    else:
        subset = state.get_receptacle_by_name(receptacle_names)

    # Get names and validate sizes
    query_names = subset.receptacle_names
    assert (
        len(query_names) == detector_output.shape[0] == observation_time.shape[0]
    ), "Length of receptacle_names, detector_output, and observation_time must match."

    # Not vectorized: each receptacle can have a different number of mixture components
    for r_id in range(subset.num_receptacles):
        receptacle_state = update(
            subset.states[r_id],
            detector_output[r_id],
            observation_time[r_id],
            p_m,
            p_f,
        )
        state = state.update_receptacle(receptacle_state, query_names[r_id])

    return state


def object_predict(
    state: Object,
    prediction_time: jnp.array,
    receptacle_names: Optional[Union[str, List[str]]] = None,
) -> jnp.ndarray:
    """Predict function for an object, wrapping the underlying predict function."""
    if receptacle_names is None:
        subset = state
    else:
        subset = state.get_receptacle_by_name(receptacle_names)

    # Get names and validate sizes
    out = jnp.zeros((subset.num_receptacles, 1))
    for r_id in range(subset.num_receptacles):
        belief, _ = predict(
            subset.states[r_id],
            prediction_time,
        )
        out = out.at[r_id].set(belief)

    return out
