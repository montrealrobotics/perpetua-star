from typing import List, Optional, Tuple, Dict, Union

import jax
from jax.tree_util import Partial
import jax.numpy as jnp

from perpetua2.utils.jax_utils import clip_prob
from perpetua2.utils.filter_state import PerpetuaStarState, Object
from perpetua2.filters import MixtureEmergenceFilters, MixturePersistenceFilters, Fremen


def init(
    log_s_persistence: Partial,
    log_s_emergence: Partial,
    params_persistence: Dict[str, jnp.array],
    params_emergence: Dict[str, jnp.array],
    pi_persistence: jnp.array,
    pi_emergence: jnp.array,
    switching_state: Fremen.FremenPrior,
    decay_rate: float = 1.0,
    alpha: float = 1.0,
    max_period: float = jnp.inf,
    initialization_time: float = 0.0,
    n_samples_persistence: int = 0,
    n_samples_emergence: int = 0,
    id: int = -1,
    name: str = "",
) -> PerpetuaStarState:
    """
    Initializes the Perpetua* filter state.

    Args:
        log_s_persistence (Partial): The log survival function for persistence.
        log_s_emergence (Partial): The log survival function for emergence.
        params_persistence (Dict[str, jnp.array]): Parameters for the persistence filter.
        params_emergence (Dict[str, jnp.array]): Parameters for the emergence filter.
        pi_persistence (jnp.array): Initial probabilities for persistence components.
        pi_emergence (jnp.array): Initial probabilities for emergence components.
        switching_state (FremenPrior | PiecewisePrior): The switching prior.
        decay_rate (float, optional): The decay rate for the filters. Default is 1.0.
        alpha (float, optional): Evidence annealing factor. Default is 1.0.
        max_period (float, optional): Maximum period before reset. Default is jnp.inf.
        initialization_time (float, optional): The time at which the filter is initialized. Default is 0.0.
        n_samples_persistence (int, optional): Number of data points the persistence mixture was fitted on. Default is 0.
        n_samples_emergence (int, optional): Number of data points the emergence mixture was fitted on. Default is 0.
        id (int, optional): Receptacle id. Default is -1.
        name (str, optional): Receptacle name. Default is "".

    Returns:
        PerpetuaStarState: The initialized state of the Perpetua* filter.
    """
    pf_state = MixturePersistenceFilters.init(
        log_s_persistence,
        params_persistence,
        pi_persistence,
        decay_rate,
        initialization_time,
        n_samples=n_samples_persistence,
    )
    ef_state = MixtureEmergenceFilters.init(
        log_s_emergence,
        params_emergence,
        pi_emergence,
        decay_rate,
        initialization_time,
        n_samples=n_samples_emergence,
    )
    return PerpetuaStarState.create(
        ef_state=ef_state,
        pf_state=pf_state,
        switching_state=switching_state,
        alpha=alpha,
        max_period=max_period,
        initialization_time=initialization_time,
        id=id,
        name=name,
    )


@jax.jit
def _update(
    state: PerpetuaStarState,
    detector_output: bool,
    observation_time: float,
    p_m: float,
    p_f: float,
) -> PerpetuaStarState:
    """
    Updates Perpetua by incorporating a new detector output.

    Args:
        state (PerpetuaState): The current state of the Perpetua filter.
        detector_output (bool): A Boolean value output by the detector indicating whether the given feature was detected.
        observation_time (float): The timestamp for the detection.
        p_m (float): A floating-point value in the range [0, 1] indicating the missed detection probability
                     of the feature detector.
        p_f (float): A floating-point value in the range [0, 1] indicating the false alarm probability of the feature
                     detector.

    Returns:
        PerpetuaState: The updated state of the Perpetua filter.
    """
    # Check if we need to reset Perpetua (max period exceeded)
    state = update_internal_state(state, observation_time)

    # Update persistence filter
    state = state.replace(
        pf_state=MixturePersistenceFilters.update(
            state.pf_state, detector_output, observation_time, p_m, p_f
        )
    )

    # Update emergence filter
    state = state.replace(
        ef_state=MixtureEmergenceFilters.update(
            state.ef_state, detector_output, observation_time, p_m, p_f
        )
    )
    # Store last observation time
    state = state.replace(last_observation_time=observation_time)
    return state


@jax.jit
def update(
    state: PerpetuaStarState,
    detector_output: bool,
    observation_time: float,
    p_m: float,
    p_f: float,
) -> PerpetuaStarState:
    # Only update if the observation time is > last observation time
    def perform_update(state, args):
        detector_output, observation_time = args
        return _update(state, detector_output, observation_time, p_m, p_f)

    def no_update(state, args):
        return state

    return jax.lax.cond(
        observation_time >= state.last_observation_time,
        perform_update,
        no_update,
        state,
        (detector_output, observation_time),
    )


@jax.jit
def predict(
    state: PerpetuaStarState, prediction_time: jnp.array
) -> Tuple[jnp.array, jnp.array, jnp.array, jnp.array, jnp.array]:
    emergence_state = state.ef_state
    persistence_state = state.pf_state
    # Avoid querying the past
    query_time = jnp.maximum(prediction_time, state.last_observation_time)
    # Predict belief for each model
    emergence_bel, _ = MixtureEmergenceFilters.predict(emergence_state, query_time)
    persistence_bel, _ = MixturePersistenceFilters.predict(
        persistence_state, query_time
    )

    belief = jnp.concatenate([emergence_bel.T, persistence_bel.T], axis=1)

    # Predict p(M) using Fremen - p(M=1) is persistence!
    model_prior = state.switching_prior(query_time)
    log_model_prior = jnp.stack([jnp.log(model_prior), jnp.log1p(-model_prior)], axis=1)

    # Get log evidence
    log_emergence_evidence, log_persistence_evidence = (
        emergence_state.log_evidence,
        persistence_state.log_evidence,
    )
    log_evidence = jnp.stack([log_emergence_evidence, log_persistence_evidence], axis=0)
    log_evidence = jnp.broadcast_to(log_evidence[None, :], log_model_prior.shape)

    # Compute log-evidence annealing
    dt = query_time - state.last_observation_time
    annealing_t = jnp.exp(-state.alpha * dt)
    log_evidence = log_evidence * annealing_t[:, None]

    # Compute posterior over model
    log_p_m = log_model_prior + log_evidence
    log_p_m = log_p_m - jax.scipy.special.logsumexp(log_p_m, axis=1, keepdims=True)

    # Weighted combination of beliefs
    final_belief = (jnp.exp(log_p_m) * belief).sum(axis=1)
    final_belief = clip_prob(final_belief)

    return final_belief, belief, log_evidence, jnp.log(model_prior), log_p_m


@jax.jit
def update_internal_state(state, observation_time) -> PerpetuaStarState:
    """Update internal state by resetting if max_period has been exceeded."""

    def no_reset_state(state, n_periods):
        return state

    def perform_reset(state, n_periods):
        # Reset in the next period
        init_time = (n_periods + state.reset_counter) * state.max_period
        old_pf_state = state.pf_state
        old_ef_state = state.ef_state
        pf_state = MixturePersistenceFilters.init(
            old_pf_state.log_s,
            old_pf_state.params,
            old_pf_state.pi,
            old_pf_state.decay_rate,
            init_time,
            old_pf_state.n_samples,
        )
        ef_state = MixtureEmergenceFilters.init(
            old_ef_state.log_s,
            old_ef_state.params,
            old_ef_state.pi,
            old_ef_state.decay_rate,
            init_time,
            old_ef_state.n_samples,
        )
        return state.replace(
            pf_state=pf_state,
            ef_state=ef_state,
            reset_counter=state.reset_counter + n_periods.astype(int),
            initialization_time=init_time,
            last_observation_time=init_time,
        )

    n_periods = (observation_time - state.initialization_time) // state.max_period
    return jax.lax.cond(
        n_periods > 0,
        perform_reset,
        no_reset_state,
        state,
        n_periods,
    )


################################################################################
### Methods for updating and predicting object state using Perpetua* ###
################################################################################


def select_object_subset(
    state: Object,
    receptacle_names: Optional[Union[str, List[str]]] = None,
) -> Object:
    """Select a subset of the object based on receptacle names."""
    if receptacle_names is None:
        subset = state
    else:
        subset = state.select(receptacle_names)

    return subset


def object_update(
    state: Object,
    detector_output: jnp.ndarray,
    observation_time: jnp.ndarray,
    p_m: float,
    p_f: float,
    receptacle_names: Optional[Union[str, List[str]]] = None,
) -> Object:
    """Update function for an object, wrapping the underlying update function."""
    subset = select_object_subset(state, receptacle_names)
    # Get names and validate sizes
    assert (
        len(subset.receptacle_names)
        == detector_output.shape[0]
        == observation_time.shape[0]
    ), "Length of receptacle_names, detector_output, and observation_time must match."

    for r_id, receptacle_state in enumerate(subset):
        new_state = update(
            receptacle_state,
            detector_output[r_id],
            observation_time[r_id],
            p_m,
            p_f,
        )
        state = state.update_receptacle(new_state)

    return state


def multistep_update(
    state: Object,
    detector_output: jnp.ndarray,
    observation_time: jnp.ndarray,
    p_m: float,
    p_f: float,
    receptacle_name: str,
) -> Object:
    """Multi-step update function for an object, wrapping the underlying update function."""
    assert isinstance(receptacle_name, str), "receptacle_name must be a string."
    subset = select_object_subset(state, receptacle_name)
    receptacle = subset.get_receptacles

    def scan_body_fun(carry_state, input_slice):
        d_out, obs_time = input_slice
        new_state = update(carry_state, d_out, obs_time, p_m=p_m, p_f=p_f)
        return new_state, None

    updated_receptacle, _ = jax.lax.scan(
        scan_body_fun, receptacle, (detector_output, observation_time)
    )
    state = state.update_receptacle(updated_receptacle)

    return state


def object_predict(
    state: Object,
    prediction_time: jnp.array,
    receptacle_names: Optional[Union[str, List[str]]] = None,
) -> jnp.ndarray:
    """Predict function for an object, wrapping the underlying predict function."""
    subset = select_object_subset(state, receptacle_names)

    # Get names and validate sizes
    out = jnp.zeros((subset.num_receptacles, 1))
    for r_id, receptacle_state in enumerate(subset):
        belief, _, _, _, _ = predict(
            receptacle_state,
            prediction_time,
        )
        out = out.at[r_id].set(belief)

    return out
