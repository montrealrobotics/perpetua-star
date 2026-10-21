from typing import List, Optional, Tuple, Union, Dict, Callable
from functools import partial

from jax import Array
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from jax.tree_util import Partial
from perpetua2.utils.filter_state import FilterState

import perpetua2.utils.priors as priors
from perpetua2.learning.learning_utils import (
    compute_log_likelihood,
    create_filter,
    compute_marginals,
    print_results,
    perturb,
    pad_sequences,
)
from perpetua2.utils.metrics import compute_aic, compute_bic
from perpetua2.utils.jax_utils import logdiff
from perpetua2.utils.viz import Plotter
from perpetua2.utils.logger import logger


@partial(jax.jit, static_argnames=["filter"])
def process_sequence(
    filter: str,
    obs_times: ArrayLike,
    obs: ArrayLike,
    pi_: ArrayLike,
    lambda_: ArrayLike,
    state: FilterState,
    p_m: ArrayLike,
    p_f: ArrayLike,
):
    """
    Process a single sequence of observations using the specified filter.

    Args:
        filter (str): The filter type to use.
        obs_times (ArrayLike): Array of observation times.
        obs (ArrayLike): Array of observations.
        pi_ (ArrayLike): Mixing coefficients.
        lambda_ (ArrayLike): Rate parameters for each component.
        state (FilterState): The initial state of the filter.
        p_m (ArrayLike): Probability of missed detection.
        p_f (ArrayLike): Probability of false alarm.

    Returns:
        Tuple[Array, Array]: The expected survival times (psi_j) and the expected component assignments (phi_j).
    """
    n_components = len(pi_)

    # Initialize filter (simulating Filter class with JAX functionality)
    log_sum_psi = jnp.zeros((obs_times.shape[0] + 1, n_components))

    ####### E Step #######
    # Compute log-likelihood for t_0 = 0
    ll_function = jax.vmap(compute_log_likelihood, in_axes=(None, 0, None, None, None, None))
    log_likelihoods = ll_function(filter, jnp.insert(obs_times, 0, 0.0), obs, obs_times, p_m, p_f)

    ### Compute E_q[Z_kj T_j] for all k and j ###
    # For time t_0 = 0 and t_1 = t_1
    log_term1 = jnp.log(1 / lambda_)
    log_terms = -lambda_ * obs_times[:, jnp.newaxis] + jnp.log(obs_times[:, jnp.newaxis] + (1 / lambda_))
    log_expected_p0 = logdiff(log_term1, log_terms[0])
    log_sum_psi = log_sum_psi.at[0].set(log_likelihoods[0] + log_expected_p0)

    # Compute log \sum_{i=1}^{N-1} p(Y_{1:N} | T) \int_{t_i}^{t_{i+1}} T p(T) dT
    log_expected_ps = logdiff(log_terms[:-1], log_terms[1:])
    log_sum_psi = log_sum_psi.at[1:-1].set(log_likelihoods[1:-1][:, jnp.newaxis] + log_expected_ps)

    # Find index of the last valid entry, or -1 if all NaN
    last_idx = jnp.max(jnp.where(~jnp.isnan(obs_times), jnp.arange(obs_times.size), -1))
    last_valid_mask = jnp.arange(obs_times.size + 1) == last_idx + 1
    # Compute log p(Y_{1:N} | t_N) \int_{t_N}^{t_{N+1}} T p(T) dT
    log_sum_psi = jnp.where(
        last_valid_mask[:, jnp.newaxis], log_likelihoods[1 + last_idx] + log_terms[last_idx], log_sum_psi
    )
    # Mask with -inf where obs_times is nan
    log_sum_psi = jnp.where(jnp.isnan(jnp.insert(obs_times, 0, 0.0))[:, jnp.newaxis], -jnp.inf, log_sum_psi)

    # Compute expected survival time under p(T | Y_{1:N})
    total_log_sum_psi = logsumexp(log_sum_psi, axis=0)
    total_log_sum_psi += jnp.log(pi_)

    # Normalize and compute results
    log_evidence, log_joint_evidence = compute_marginals(filter, state, obs, obs_times, p_m, p_f)
    psi_j = jnp.exp(total_log_sum_psi - log_evidence)

    ### Compute E_q[Z_kj] for all k and j ###
    phi_j = jnp.exp(log_joint_evidence - log_evidence)
    return psi_j, phi_j


@partial(jax.jit, static_argnames=["filter"])
def em_step(
    filter: str,
    observation_times: List[ArrayLike],
    y_bool: List[ArrayLike],
    state: FilterState,
    pi_hat: ArrayLike,
    lambda_hat: ArrayLike,
    p_m: ArrayLike,
    p_f: ArrayLike,
) -> Union[Array, Array]:
    """
    Perform the E-step and M-step in the EM algorithm for the mixture model with exponential prior.

    Args:
        filter (str): The filter type to use.
        observation_times (List[ArrayLike]): List of arrays of observation times for each sequence.
        y_bool (List[ArrayLike]): List of binary observation arrays for each sequence.
        state (FilterState): The initial state of the filter.
        pi_hat (ArrayLike): Initial mixing coefficients.
        lambda_hat (ArrayLike): Initial rate parameters for each component.
        p_m (ArrayLike): Probability of missed detection.
        p_f (ArrayLike): Probability of false alarm.

    Returns:
        Tuple[Array, Array]: Updated mixing coefficients (pi_bar) and rate parameters (lambda_bar).
    """

    e_step_func = jax.vmap(process_sequence, in_axes=(None, 0, 0, None, None, None, None, None))
    psi, phi = e_step_func(
        filter,
        observation_times,
        y_bool,
        pi_hat,
        lambda_hat,
        state,
        p_m,
        p_f,
    )

    ####### M Step #######
    pi_bar = jnp.sum(phi, axis=0) / jnp.sum(phi)
    lambda_bar = jnp.sum(phi, axis=0) / jnp.sum(psi, axis=0)

    # Clip the parameters to prevent NaN values
    return jnp.clip(pi_bar, 1e-10, 1.0), jnp.clip(lambda_bar, 1e-10, 1e15)


def fit(
    p_m: ArrayLike,
    p_f: float,
    lambda_init: ArrayLike,
    pi_init: ArrayLike,
    filter: str,
    prior: str,
    delta: float,
    n_iter: int,
    y_bool: List[ArrayLike],
    observation_times: List[ArrayLike],
    plotter: Optional[Plotter] = None,
    exp_name: str = "mixture_exponential",
    **kwargs
) -> Tuple[Array, Array]:
    """
    Fit the mixture model with exponential prior using the EM algorithm.

    Args:
        p_m (ArrayLike): Probability of missed detection.
        p_f (float): Probability of false alarm.
        lambda_init (ArrayLike): Initial rate parameters for each component.
        pi_init (ArrayLike): Initial mixing coefficients.
        filter (str): The filter type to use.
        prior (str): The prior distribution type.
        delta (float): Convergence threshold for the EM algorithm.
        n_iter (int): Maximum number of iterations for the EM algorithm.
        plotter (Plotter): Plotter object for visualizing results.
        y_bool (List[ArrayLike]): List of binary observation arrays for each sequence.
        observation_times (List[ArrayLike]): List of arrays of observation times for each sequence.
        exp_name (str): Experiment name for logging and plotting.

    Returns:
        Tuple[Array, Array, float]: Final rate parameters, mixing coefficients and log-evidence.
    """

    # Params and initialization
    start = True
    m = 1
    # Initialize params
    lambda_hat = lambda_init
    pi_hat = pi_init
    # Compute metrics at initialization
    log_s = Partial(priors.log_survival_exponential)
    loss_t_1 = -jnp.inf
    # Store the loss values
    losses = [loss_t_1]
    print_results({"lambda_": lambda_hat, "pi": pi_hat}, m, loss_t_1, prior=prior)
    padded_obs: ArrayLike = pad_sequences(y_bool, jnp.nan)
    padded_times: ArrayLike = pad_sequences(observation_times, jnp.nan)
    batched_compute_marginals = jax.vmap(compute_marginals, in_axes=(None, None, 0, 0, None, None))

    lambda_best, pi_best, best_loss = lambda_hat, pi_hat, loss_t_1
    while start and m < n_iter:
        state = create_filter(filter, log_s, {"lambda_": lambda_hat}, pi_hat, initialization_time=0.0)
        pi_bar, lambda_bar = em_step(filter, padded_times, padded_obs, state, pi_hat, lambda_hat, p_m, p_f)

        state = create_filter(filter, log_s, {"lambda_": lambda_bar}, pi_bar, initialization_time=0.0)
        log_evidences, _ = batched_compute_marginals(filter, state, padded_obs, padded_times, p_m, p_f)
        loss_t = jnp.sum(log_evidences)
        # Increase iteration and if the loss does not improve, try again without updating params
        m += 1
        # Update lambda if iteration was successful and print results
        lambda_hat, pi_hat = lambda_bar, pi_bar
        lambda_best, pi_best, best_loss = (
            (lambda_hat, pi_hat, loss_t)
            if loss_t > best_loss and not jnp.isnan(loss_t)
            else (lambda_best, pi_best, best_loss)
        )

        ### Stopping criteria ###
        if jnp.abs(loss_t - loss_t_1) < delta or jnp.isnan(loss_t):
            logger.info("=" * 50)
            logger.info(f"Stopping the algorithm at iteration {m}")
            logger.info("=" * 50)
            start = False

        # Assign current loss and iterate again
        losses.append(loss_t)
        loss_t_1 = loss_t

    lambda_hat, pi_hat = lambda_best, pi_best
    print_results({"lambda_": lambda_hat, "pi": pi_hat}, "Final", best_loss, prior=prior)

    if plotter:
        plotter.plot_loss_curve(
            jnp.arange(len(losses)) + 1, -jnp.array(losses), title="Loss", f_name=f"{exp_name}_loss_curve"
        )
    return lambda_hat, pi_hat, best_loss


def select_and_refine_model(
    fit: Callable,
    query_times: jnp.ndarray,
    num_datapoints: int,
    params_range: range = range(2, 5),
    perturbation_scale: jnp.ndarray = jnp.array([0.01, 0.1]),
    num_retrain_attempts: int = 5,
    key: random.PRNGKey = random.PRNGKey(42),
    prune_threshold: float = 1e-2,
) -> Dict[str, jnp.ndarray]:
    """
    Selects the optimal model parameters (lambda, pi) based on AIC/BIC and refines them with perturbation.

    Args:
        fit (Callable): Function to fit the model.
        query_times (jnp.ndarray): Query times used to initial model parameters.
        num_datapoints (int): Number of data points for BIC computation.
        params_range (range): Range of number of components to evaluate.
        perturbation_scale (jnp.ndarray): Scale of perturbations for retraining.
        num_retrain_attempts (int): Number of retraining attempts.
        key (random.PRNGKey): Random key.

    Returns:
        Dict[str, jnp.ndarray]: Dictionary with best parameters and evidence.
    """
    best_lambda, best_pi = None, None
    best_aic = jnp.inf
    best_bic = jnp.inf
    best_evidence = -jnp.inf

    # Model selection loop
    for n in params_range:
        quantile_positions = jnp.linspace(0, 1, n + 2)[1:-1]
        quantiles = jnp.quantile(query_times, quantile_positions)
        lambda_init = 1 / quantiles
        pi_init = jnp.ones_like(lambda_init) / n

        # Fit model
        lambda_em, pi_em, final_evidence = fit(lambda_init=lambda_init, pi_init=pi_init)

        # Compute AIC and BIC
        aic = compute_aic(final_evidence, 2 * n)
        bic = compute_bic(num_datapoints, final_evidence, 2 * n)

        # Update best parameters based on AIC
        if aic < best_aic:
            best_lambda, best_pi = lambda_em, pi_em
            best_aic = aic
            best_bic = bic
            best_evidence = final_evidence

        logger.info("=" * 50)
        logger.info(f"components={n}, AIC={aic}, BIC={bic}, NLL={final_evidence}")
        logger.info(f"λ={lambda_em}, π={pi_em}")
        logger.info("=" * 50)
        # This operation can allocate A LOT of memory, so we clear the cache after each iteration
        jax.clear_caches()

    # Check if there exists a component with a low probability and prune it
    if jnp.any(best_pi < prune_threshold):
        mask = best_pi > prune_threshold
        best_pi, best_lambda = best_pi[mask], best_lambda[mask]
        best_pi = best_pi / jnp.sum(best_pi)
        best_aic, best_bic, best_evidence = jnp.inf, jnp.inf, -jnp.inf
        logger.info("=" * 50)
        logger.info(f"Pruned {len(mask) - mask.sum()} components with weight below {prune_threshold}")
        logger.info("=" * 50)

    # Refinement loop
    optimal_n_components = len(best_lambda)
    for _ in range(num_retrain_attempts):
        quantile_positions = jnp.linspace(0, 1, optimal_n_components + 2)[1:-1]
        quantiles = jnp.quantile(query_times, quantile_positions)

        # Perturb parameters
        lambda_init, key = perturb(1 / quantiles, perturbation_scale[0], key, min_value=jnp.finfo(jnp.float32).tiny)
        pi_init, key = perturb(jnp.ones_like(lambda_init) / optimal_n_components, perturbation_scale[1], key)
        pi_init /= jnp.sum(pi_init)

        # Refit model
        lambda_em, pi_em, final_evidence = fit(lambda_init=lambda_init, pi_init=pi_init)
        # Compute AIC and BIC
        aic = compute_aic(final_evidence, 2 * n)
        bic = compute_bic(num_datapoints, final_evidence, 2 * n)

        # Update best parameters if evidence improves
        if final_evidence > best_evidence:
            best_lambda, best_pi = lambda_em, pi_em
            best_aic = aic
            best_bic = bic
            best_evidence = final_evidence

    return {
        "lambda": best_lambda,
        "pi": best_pi,
        "aic": best_aic,
        "bic": best_bic,
        "evidence": best_evidence,
    }
