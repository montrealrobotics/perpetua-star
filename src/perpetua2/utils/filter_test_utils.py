from typing import Tuple, Callable, List, Dict
from jax import Array
from jax.typing import ArrayLike

import numpy as np
from scipy.stats import bernoulli
from tqdm import tqdm

from perpetua2.filters import MixtureEmergenceFilters, MixturePersistenceFilters, Perpetua, PerpetuaStar, Fremen
from perpetua2.utils.filter_state import Object
from perpetua2.utils.metrics import report_metrics
from perpetua2.data.containers import Receptacle
from jax.tree_util import Partial

import jax
import jax.numpy as jnp
from functools import partial


# Aggregate results


@jax.jit
def run_mixture_persistence_filters(
    obs_arr: ArrayLike,
    t_arr: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
    pi: ArrayLike,
    query_times: ArrayLike,
    log_s: List[Partial],
    params: Dict[str, float],
    decay_rate: float = 1.0,
    init_time: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Constructs and runs a Mixture of Persistence Filters on a given sequence of timestamped observations with specified error rates,
    returning the persistence probabilities and beliefs at the specified query times.

    Args:
        obs_arr (ArrayLike): A sequence of binary observations (0 or 1) corresponding to the times in t_arr.
        t_arr (ArrayLike): A sequence of timestamps corresponding to the observations in obs_arr.
        p_m (ArrayLike): Missed detection probability.
        p_f (ArrayLike): False alarm probability.
        pi (ArrayLike): Mixture weights for the persistence filters.
        query_times (ArrayLike): A sequence of query times at which to compute the persistence probabilities.
        log_s (List[Partial]): A list of partials with the survival functions for each persistence filter's component.
        params (Dict[str, float]): A dictionary of parameters for the filters.
        init_time (float, optional): The initialization time of the filter. Defaults to 0.0.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The persistence probability of the largest component and the conditional posterior
         of each component at the at the query_times.
    """
    # Initialize Mixture Persistence Filters
    filter_state = MixturePersistenceFilters.init(log_s, params, pi, decay_rate, init_time)

    # Container for probabilities and beliefs
    probs = jnp.zeros((len(pi), query_times.shape[0]))
    beliefs = jnp.zeros((1, query_times.shape[0]))
    evidence = jnp.zeros((1, query_times.shape[0]))

    # Predict for the query times before the first observation
    mask = query_times < t_arr[0]
    predicted_beliefs, predicted_probs = MixturePersistenceFilters.predict(filter_state, query_times)
    probs = jnp.where(mask[None, :], predicted_probs, probs)
    beliefs = jnp.where(mask, predicted_beliefs, beliefs)
    evidence = jnp.where(mask, filter_state.log_evidence, evidence)

    # Update the state with the first observation
    filter_state = MixturePersistenceFilters.update(filter_state, obs_arr[0], t_arr[0], p_m, p_f)

    def scan_body(carry, inputs, obs, t, queries, p_m, p_f):
        state, probs, beliefs, evidence = carry
        i, j = inputs

        # Predict for query times in the current range
        query_mask = (queries >= t[i]) & (queries < t[j])
        predicted_beliefs, predicted_probs = MixturePersistenceFilters.predict(state, queries)
        probs = jnp.where(query_mask[None, :], predicted_probs, probs)
        beliefs = jnp.where(query_mask, predicted_beliefs, beliefs)
        evidence = jnp.where(query_mask, state.log_evidence, evidence)

        # Update the filter with the current observation
        new_state = MixturePersistenceFilters.update(state, obs[j], t[j], p_m, p_f)

        return (new_state, probs, beliefs, evidence), None

    # Stack inputs
    inputs = (jnp.arange(len(t_arr) - 1), jnp.arange(len(t_arr) - 1) + 1)
    # Run the filter update and prediction process
    (filter_state, probs, beliefs, evidence), _ = jax.lax.scan(
        partial(scan_body, obs=obs_arr, t=t_arr, queries=query_times, p_m=p_m, p_f=p_f),
        (filter_state, probs, beliefs, evidence),
        inputs,
    )

    # Predict probabilities for query times after the last observation
    mask_after_last = query_times >= t_arr[-1]
    final_beliefs, final_probs = MixturePersistenceFilters.predict(filter_state, query_times)
    probs = jnp.where(mask_after_last[None, :], final_probs, probs)
    beliefs = jnp.where(mask_after_last, final_beliefs, beliefs)
    evidence = jnp.where(mask_after_last, filter_state.log_evidence, evidence)

    return jnp.concatenate((probs, beliefs), axis=0), evidence


def run_mixture_emergence_filters(
    obs_arr: ArrayLike,
    t_arr: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
    pi: ArrayLike,
    query_times: ArrayLike,
    log_s: List[Partial],
    params: Dict[str, float],
    decay_rate: float = 1.0,
    init_time: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Constructs and runs a Mixture of Emergence Filters on a given sequence of timestamped observations with specified error rates,
    returning the persistence probabilities and beliefs at the specified query times.

    Args:
        obs_arr (ArrayLike): A sequence of binary observations (0 or 1) corresponding to the times in t_arr.
        t_arr (ArrayLike): A sequence of timestamps corresponding to the observations in obs_arr.
        p_m (ArrayLike): Missed detection probability.
        p_f (ArrayLike): False alarm probability.
        pi (ArrayLike): Mixture weights for the emergence filters.
        query_times (ArrayLike): A sequence of query times at which to compute the persistence probabilities.
        log_s (List[Partial]): A list of partials with the survival functions for each emergence filter's component.
        params (Dict[str, float]): A dictionary of parameters for the filters.
        init_time (float, optional): The initialization time of the filter. Defaults to 0.0.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: The persistence probabilities and beliefs at the query_times.
    """
    # Initialize Mixture Emergence Filters
    filter_state = MixtureEmergenceFilters.init(log_s, params, pi, decay_rate, init_time)

    # Container for probabilities and beliefs
    probs = jnp.zeros((len(pi), query_times.shape[0]))
    beliefs = jnp.zeros((1, query_times.shape[0]))
    evidence = jnp.zeros((1, query_times.shape[0]))

    # Record the predictions for the first query times
    # Determine the first relevant observation index and mask based on init_time
    # Is this jittable? Runtime value in first_index
    first_index, mask = jax.lax.cond(
        t_arr[0] > init_time,
        lambda _: (0, query_times < t_arr[0]),
        lambda _: (jnp.argmax(init_time < t_arr), query_times < t_arr[jnp.argmax(init_time < t_arr)]),
        operand=None,
    )
    predicted_beliefs, predicted_probs = MixtureEmergenceFilters.predict(filter_state, query_times)
    probs = jnp.where(mask[None, :], predicted_probs, probs)
    beliefs = jnp.where(mask, predicted_beliefs, beliefs)
    evidence = jnp.where(mask, filter_state.log_evidence, evidence)

    # Update the state with the first observation
    filter_state = MixtureEmergenceFilters.update(filter_state, obs_arr[first_index], t_arr[first_index], p_m, p_f)

    def scan_body(carry, inputs, obs, t, queries, p_m, p_f):
        state, probs, beliefs, evidence = carry
        i, j = inputs

        # Predict for query times in the current range
        query_mask = (queries >= t[i]) & (queries < t[j])
        predicted_beliefs, predicted_probs = MixtureEmergenceFilters.predict(state, queries)
        probs = jnp.where(query_mask[None, :], predicted_probs, probs)
        beliefs = jnp.where(query_mask, predicted_beliefs, beliefs)
        evidence = jnp.where(query_mask, state.log_evidence, evidence)

        # Update the filter with the current observation
        new_state = MixtureEmergenceFilters.update(state, obs[j], t[j], p_m, p_f)

        return (new_state, probs, beliefs, evidence), None

    # Stack inputs
    inputs = (jnp.arange(first_index, len(t_arr) - 1), jnp.arange(first_index, len(t_arr) - 1) + 1)
    # Run the filter update and prediction process
    (filter_state, probs, beliefs, evidence), _ = jax.lax.scan(
        partial(scan_body, obs=obs_arr, t=t_arr, queries=query_times, p_m=p_m, p_f=p_f),
        (filter_state, probs, beliefs, evidence),
        inputs,
    )

    # Predict probabilities for query times after the last observation
    mask_after_last = query_times >= t_arr[-1]
    final_beliefs, final_probs = MixtureEmergenceFilters.predict(filter_state, query_times)
    probs = jnp.where(mask_after_last[None, :], final_probs, probs)
    beliefs = jnp.where(mask_after_last, final_beliefs, beliefs)
    evidence = jnp.where(mask_after_last, filter_state.log_evidence, evidence)

    return jnp.concatenate((probs, beliefs), axis=0), evidence


def run_perpetua(
    obs_arr: ArrayLike,
    t_arr: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
    pi_persistence: float,
    pi_emergence: float,
    query_times: ArrayLike,
    log_s_persistence: Callable[[float], Array],
    log_s_emergence: Callable[[float], Array],
    params_persistence: Dict[str, float],
    params_emergence: Dict[str, float],
    delta_high: float,
    delta_low: float,
    init_time: float = 0.0,
    num_steps: int = 1,
    eps: float = 1e-1,
) -> Tuple[Array, Array, Array]:
    """
    Constructs and runs a Perpetua model on a given sequence of timestamped observations with specified error rates,
    returning the persistence probabilities, states and weights at the specified query times.

    Args:
        obs_arr (ArrayLike): A sequence of binary observations (0 or 1) corresponding to the times in t_arr.
        t_arr (ArrayLike): A sequence of timestamps corresponding to the observations in obs_arr.
        p_m (ArrayLike): Missed detection probability.
        p_f (ArrayLike): False alarm probability.
        pi_persistence (float): Mixture weight for the persistence filter.
        pi_emergence (float): Mixture weight for the emergence filter.
        query_times (ArrayLike): A sequence of query times at which to query perpetua.
        log_s_persistence (Callable[[float], Array]): The logarithm of the survival function for the persistence filter.
        log_s_emergence (Callable[[float], Array]): The logarithm of the survival function for the emergence filter.
        params_persistence (Dict[str, float]): A dictionary of parameters for the persistence filter.
        params_emergence (Dict[str, float]): A dictionary of parameters for the emergence filter.
        delta_high (float): The high threshold for the belief to activate the persistence filter.
        delta_low (float): The low threshold for the belief to activate the emergence filter.
        init_time (float, optional): The initialization time of the filter. Defaults to 0.0.
        num_steps (int, optional): The number of steps for the filter. Used to approximate the switching state. Defaults to 1.
        eps (float, optional): A small value to prevent numerical instability. Defaults to 1e-1.

    Returns:
        Tuple[Array, Array, Array]: The persistence probabilities, states and weights at the query_times.
    """
    # Ensure PM and PF are arrays of the same length as obs_arr
    p_m = jnp.repeat(p_m, len(obs_arr)) if isinstance(p_m, (float, int)) else p_m
    p_f = jnp.repeat(p_f, len(obs_arr)) if isinstance(p_f, (float, int)) else p_f

    # Construct perpetua model
    f = Perpetua
    state = f.init(
        Partial(log_s_persistence),
        Partial(log_s_emergence),
        params_persistence,
        params_emergence,
        pi_persistence,
        pi_emergence,
        delta_low,
        delta_high,
        init_time,
        num_steps,
        eps,
    )

    ### Predict ###
    # Get the indices of all query_times prior to the first observation
    mask = query_times < t_arr[0]
    # Record the predictions for these query times
    probs, states, weights = [], [], []
    if np.any(mask):
        p, s, w = f.predict(state, query_times[mask])
        probs.append(p)
        states.append(s)
        weights.append(w)

    # UPDATE:
    state = f.update(state, obs_arr[0].item(), t_arr[0].item(), p_m[0].item(), p_f[0].item())

    for i in tqdm(range(len(t_arr) - 1), desc="Running Perpetua", unit="Iter"):
        ### Predict ###
        # Get the indices of all query_times in [t[i], t[i+1])
        mask = (query_times >= t_arr[i]) & (query_times < t_arr[i + 1])

        if np.any(mask):
            p, s, w = f.predict(state, query_times[mask])
            probs.append(p)
            states.append(s)
            weights.append(w)
        ### Update ###
        state = f.update(
            state,
            obs_arr[i + 1].item(),
            t_arr[i + 1].item(),
            p_m[i + 1].item(),
            p_f[i + 1].item(),
        )

    # Predict probabilities for query times after the last observation
    mask = query_times >= t_arr[-1]
    if jnp.any(mask):
        p, s, w = f.predict(state, query_times[mask])
        probs.append(p)
        states.append(s)
        weights.append(w)

    return jnp.concatenate(probs, axis=0), jnp.concatenate(states, axis=0), jnp.concatenate(weights, axis=0)


def run_perpetua_star(
    obs_arr: ArrayLike,
    t_arr: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
    pi_persistence: float,
    pi_emergence: float,
    query_times: ArrayLike,
    log_s_persistence: Callable[[float], Array],
    log_s_emergence: Callable[[float], Array],
    params_persistence: Dict[str, float],
    params_emergence: Dict[str, float],
    switching_state: Fremen.FremenState,
    decay_rate: float = 1.0,
    alpha: float = 1.0,
    period: float = jnp.inf,
    id : int = -1,
    name: str = "",
    init_time: float = 0.0,
) -> Tuple[Array, Array, Array]:

    # Construct perpetua model
    f = PerpetuaStar
    state = f.init(
        Partial(log_s_persistence),
        Partial(log_s_emergence),
        params_persistence,
        params_emergence,
        pi_persistence,
        pi_emergence,
        switching_state,
        decay_rate,
        alpha,
        period,
        init_time,
        id=id,
        name=name,
    )

    ### Predict ###
    # Get the indices of all query_times prior to the first observation
    mask = query_times < t_arr[0]
    # Record the predictions for these query times
    outputs = {"belief": [], "preds": [], "log_evidence": [], "log_prior": [], "log_posterior": []}
    if np.any(mask):
        belief, preds, log_evidence, log_prior, log_posterior = f.predict(state, query_times[mask])
        outputs["belief"].append(belief)
        outputs["preds"].append(preds)
        outputs["log_evidence"].append(log_evidence)
        outputs["log_prior"].append(log_prior)
        outputs["log_posterior"].append(log_posterior)

    # UPDATE:
    state = f.update(state, obs_arr[0].item(), t_arr[0].item(), p_m, p_f)

    for i in tqdm(range(len(t_arr) - 1), desc="Running Perpetua*", unit="Iter"):
        ### Predict ###
        # Get the indices of all query_times in [t[i], t[i+1])
        mask = (query_times >= t_arr[i]) & (query_times < t_arr[i + 1])

        if np.any(mask):
            belief, preds, log_evidence, log_prior, log_posterior = f.predict(state, query_times[mask])
            outputs["belief"].append(belief)
            outputs["preds"].append(preds)
            outputs["log_evidence"].append(log_evidence)
            outputs["log_prior"].append(log_prior)
            outputs["log_posterior"].append(log_posterior)
        ### Update ###
        state = f.update(
            state,
            obs_arr[i + 1].item(),
            t_arr[i + 1].item(),
            p_m,
            p_f,
        )

    # Predict probabilities for query times after the last observation
    mask = query_times >= t_arr[-1]
    if jnp.any(mask):
        belief, preds, log_evidence, log_prior, log_posterior = f.predict(state, query_times[mask])
        outputs["belief"].append(belief)
        outputs["preds"].append(preds)
        outputs["log_evidence"].append(log_evidence)
        outputs["log_prior"].append(log_prior)
        outputs["log_posterior"].append(log_posterior)

    outputs = {k: jnp.concatenate(v, axis=0) for k, v in outputs.items()}

    return outputs


def select_best_fremen_order(
    train_receptacle: Receptacle,
    val_receptacle: Receptacle,
    num_periodicities: int,
    fremen_period: int,
    max_order: int,
    threshold: float = 0.5,
) -> int:
    pass

    # Fit FreMen model on training data
    state = Fremen.FremenState.create(num_periodicities=num_periodicities, max_period=fremen_period)
    state = Fremen.add(state, train_receptacle.timestamps, train_receptacle.states)

    # Find best order on validation data
    best_mae = np.inf
    best_order = 1
    for order in range(1, max_order):
        val_prediction = [Fremen.estimate(state, val_receptacle.timestamps, orderi=order)]
        # Compute Metrics
        metrics = report_metrics(
            [val_receptacle.states],
            val_prediction,
            val_prediction,
            [val_receptacle.timestamps],
            threshold=threshold,
            table_title=f"FreMen@{order} {train_receptacle.name}",
            initial=False,
            verbose=False,
        )
        # Compute best order based on validation set's MAE
        val_mae = float(metrics.rows[1][1])
        if val_mae < best_mae:
            best_mae = val_mae
            best_order = order

    return best_order


def compute_receptacle_predictions(
    receptacle: Receptacle, state: Object, p_m: jnp.ndarray, p_f: jnp.ndarray
) -> jnp.array:
    # Get predict and update functions
    pred_fn = PerpetuaStar.object_predict
    update_fn = PerpetuaStar.object_update
    r_name = state.receptacle_names[0]
    # Get observation and time arrays
    obs_t = receptacle.timestamps
    obs = receptacle.states
    query_times = receptacle.gt_timestamps
    # Initialize output container
    output = list()
    # Get the indices of all query_times prior to the first observation
    mask = query_times < obs_t[0]
    # Record of the predictions for these query times
    if jnp.any(mask):
        bel = pred_fn(state, query_times[mask], r_name)
        output.append(bel)

    # UPDATE:
    state = update_fn(state, jnp.atleast_1d(obs[0]), jnp.atleast_1d(obs_t[0]), p_m, p_f, r_name)

    for i in tqdm(range(len(obs_t) - 1), desc=f"Predicting {state.name} @ {r_name}", unit="Iter"):
        ### Predict ###
        mask = (query_times >= obs_t[i]) & (query_times < obs_t[i + 1])
        if np.any(mask):
            bel = pred_fn(state, query_times[mask], r_name)
            output.append(bel)
        ### Update ###
        state = update_fn(state, jnp.atleast_1d(obs[i + 1]), jnp.atleast_1d(obs_t[i + 1]), p_m, p_f, r_name)

    # Predict probabilities for query times after the last observation
    mask = query_times >= obs_t[-1]
    if jnp.any(mask):
        bel = pred_fn(state, query_times[mask], r_name)
        output.append(bel)
    return jnp.concatenate(output, axis=0)


def sample_observation_times(lambda_r: float, lambda_o: float, p_n: float, simulation_length: float) -> np.ndarray:
    """
    Sample a set of feature observation times according to a "bursty" Markov switching process that simulates
    random revisitations of a patrolling entity.

    Code adapted from: https://github.com/david-m-rosen/Persistence-Filter

    Args:
        lambda_r (float): The rate parameter for the exponentially-distributed inter-visitation time intervals.
        lambda_o (float): The rate parameter for the exponentially-distributed time intervals between each
                          reobservation during a single revisitation.
        p_n (float): The probability of leaving the area after each observation of the feature; the expected number
                     of observations per revisitation is 1/p_n.
        simulation_length (float): The total duration of the simulation.

    Returns:
        np.ndarray: An array of observation times that fall within the simulation length.
    """
    current_time = 0.0
    observation_times = np.array([])  # Initially empty array

    while current_time < simulation_length:
        # Sample the number of observations we will obtain on this revisit
        N = np.random.geometric(p_n)

        # Sample the inter_observation_times for this revisit
        inter_observation_times = np.random.exponential(1.0 / lambda_o, N)

        observation_times = np.append(
            observation_times,
            np.repeat(current_time, N) + np.cumsum(inter_observation_times),
        )

        # Sample a revisitation interval
        revisit_interval = np.random.exponential(1.0 / lambda_r)

        # Advance the current time
        current_time = observation_times[-1] + revisit_interval

    # Return all of the observation times that fall within the specified interval
    return observation_times[observation_times <= simulation_length]


def generate_observations(survival_time: float, observation_times: np.ndarray, P_M: float, P_F: float) -> np.ndarray:
    """
    Generate a sequence of binary observations at specified times with given error rates.

    Code adapted from: https://github.com/david-m-rosen/Persistence-Filter

    Args:
        survival_time (float): The time at which the feature ceases to be present.
        observation_times (np.ndarray): An array of timestamps at which observations are made.
        P_M (float): The probability of a missed detection (false negative).
        P_F (float): The probability of a false alarm (false positive).

    Returns:
        np.ndarray: An array of binary observations (0 or 1) at the specified observation times.
    """
    # A function to randomly sample an observation, conditioned upon whether or not the feature is still present
    def sample_obs(v):
        return bernoulli.rvs(1 - P_M) if v else bernoulli.rvs(P_F)

    obs_binary = np.fromiter(map(sample_obs, observation_times <= survival_time), dtype=np.float32)
    return obs_binary


def generate_presence_observations(
    emergence_time: float, observation_times: np.ndarray, P_M: float, P_F: float
) -> np.ndarray:
    """
    Generate a sequence of binary observations at specified times with given error rates and
    a specified presence time.

    Args:
        emergence_time (float): The time at which the feature appears.
        observation_times (np.ndarray): An array of timestamps at which observations are made.
        P_M (float): The probability of a missed detection (false negative).
        P_F (float): The probability of a false alarm (false positive).

    Returns:
        np.ndarray: An array of binary observations (0 or 1) at the specified observation times.
    """
    # A function to randomly sample an observation, conditioned upon whether or not the feature has been born
    def sample_obs(v):
        return bernoulli.rvs(1 - P_M) if v else bernoulli.rvs(P_F)

    obs_binary = np.fromiter(map(sample_obs, observation_times >= emergence_time), dtype=np.float32)
    return obs_binary


def compute_querytimes(observation_times: List[ArrayLike]) -> Tuple[Array, Array]:
    """
    Obtain the query times (largest sequence) and the number of datapoints used during training.
    """
    longest_sequence = max([obs.max() for obs in observation_times])
    num_datapoints = sum([len(obs) for obs in observation_times])
    dt = min([min(obs[1:] - obs[:-1]) for obs in observation_times])
    return jnp.arange(0, longest_sequence, dt), num_datapoints
