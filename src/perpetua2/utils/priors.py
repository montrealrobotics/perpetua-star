from typing import Any, Dict
from jax import Array
from jax.typing import ArrayLike

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.tree_util import Partial
from flax import struct

from perpetua2.utils.jax_utils import clip_log_prob, clip_prob


@jax.jit
def log_s_exponential_single(t: ArrayLike, lambda_: float) -> Array:
    """
    Compute the log survival function of an exponential distribution for a single input.

    Args:
        t (float): Single time at which to evaluate the survival function.
        lambda_ (float): The rate parameter of the exponential distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    log_survival = -lambda_ * t

    return clip_log_prob(log_survival)


def log_survival_exponential(t: ArrayLike, lambda_: float) -> Array:
    """
    Compute the log survival function of an exponential distribution for multiple inputs.

    Args:
        t (ArrayLike): Times at which to evaluate the survival function.
        lambda_ (float): The rate parameter of the exponential distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Vectorize the computation over multiple inputs
    times = jnp.atleast_1d(t)
    return jax.vmap(log_s_exponential_single, in_axes=(0, None))(times, lambda_).squeeze()


@jax.jit
def log_s_weibull_single(t: ArrayLike, k: float, lambda_: float) -> Array:
    """
    Compute the log survival function of a Weibull distribution for a single input.

    Args:
        t (float): Single time at which to evaluate the survival function.
        k (float): The shape parameter of the Weibull distribution.
        lambda_ (float): The scale parameter of the Weibull distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    epsilon = 1e-10
    log_t_lambda = jnp.log(t + epsilon) - jnp.log(lambda_ + epsilon)
    log_term = k * log_t_lambda

    # Clamp log_term to avoid extreme values that lead to 0 or 1 when exponentiated
    log_term = jnp.clip(log_term, -jnp.log(jnp.finfo(jnp.float32).max), jnp.log(jnp.finfo(jnp.float32).max))
    log_survival = -jnp.exp(log_term)

    return clip_log_prob(log_survival)


def log_survival_weibull(t: ArrayLike, k: float, lambda_: float) -> Array:
    """
    Compute the log survival function of a Weibull distribution for multiple inputs.

    Args:
        t (ArrayLike): Times at which to evaluate the survival function.
        k (float): The shape parameter of the Weibull distribution.
        lambda_ (float): The scale parameter of the Weibull distribution.

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Vectorize the computation over multiple inputs
    times = jnp.atleast_1d(t)
    return jax.vmap(log_s_weibull_single, in_axes=(0, None, None))(times, k, lambda_).squeeze()


@jax.jit
def log_s_lognormal_single(t: ArrayLike, logmu: float, std: float) -> Array:
    """
    Compute the log survival function of a log-normal distribution for a single input.

    Args:
        t (float): Single time at which to evaluate the survival function.
        logmu (float): The mean of the underlying normal distribution (in log space).
        std (float): The standard deviation of the underlying normal distribution (in log space).

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Compute the standard normal survival argument
    z = (jnp.log(t) - logmu) / std

    # Use the standard normal survival function
    log_survival = norm.logsf(z)

    return clip_log_prob(log_survival)


def log_survival_lognormal(t: ArrayLike, logmu: float, std: float) -> Array:
    """
    Compute the log survival function of a log-normal distribution for multiple inputs.

    Args:
        t (ArrayLike): Times at which to evaluate the survival function.
        logmu (float): The mean of the underlying normal distribution (in log space).
        std (float): The standard deviation of the underlying normal distribution (in log space).

    Returns:
        Array: The log survival function evaluated at t.
    """
    # Vectorize the computation over multiple inputs
    times = jnp.atleast_1d(t)
    return jax.vmap(log_s_lognormal_single, in_axes=(0, None, None))(times, logmu, std).squeeze()


@jax.jit
def per_time_cumulative_hazard(ti, starts, ends):
    # For each time ti, compute overlap with each interval [start, end]
    clipped = jnp.clip(ti, starts, ends)  # shape (N,)
    overlap = jnp.maximum(0.0, clipped - starts)
    return jnp.sum(overlap)


def cumulative_step_hazard(t, persistence_times, persistence_width, amplitude=1.0):
    """
    Computes the exact cumulative hazard for a step hazard function.

    Parameters:
        t: scalar or 1D array of times
        persistence_times: 1D array of center times
        persistence_width: 1D array of widths

    Returns:
        Array of cumulative hazard values at each time t
    """
    t = jnp.atleast_1d(t)

    starts = persistence_times - persistence_width / 2.0
    ends = persistence_times + persistence_width / 2.0

    # Vectorize over time points
    vmap_over_t = jax.vmap(per_time_cumulative_hazard, in_axes=(0, None, None), out_axes=0)
    vmap_over_c = jax.vmap(vmap_over_t, in_axes=(None, 0, 0), out_axes=0)
    cumulative_hazard = vmap_over_c(t, starts, ends)  # shape (M, T)

    return cumulative_hazard.squeeze() * amplitude  # shape (T,)


def log_survival_step(t, persistence_times, persistence_width, amplitude: int = 1):
    """
    Computes the survival function based on the cumulative hazard.

    Parameters:
        t: scalar or array of times
        persistence_times: array of center times for hazard bumps
        persistence_width: array of widths for each hazard bump

    Returns:
        Survival function values at each time in t
    """
    cumulative_hazard = cumulative_step_hazard(t, persistence_times, persistence_width, amplitude)
    return -cumulative_hazard  # survival function S(t) = exp(-H(t))


@jax.jit
def per_timestep_piecewise(t, intervals: ArrayLike, period: float) -> Array:
    starts = intervals[:, 0]
    ends = intervals[:, 1]
    present = jnp.any(jnp.logical_and(starts <= t % period, t % period <= ends))
    return clip_prob(present.astype(t.dtype))


def piecewise_prior(
    intervals: ArrayLike,  # [[start, end], ...]
    t: ArrayLike,
    period: float,
) -> Array:
    """
    Presence probability of an object at the given times under a periodic schedule.

    Args:
        intervals (ArrayLike): Presence intervals [[start, end], ...] within one period.
        t (ArrayLike): Query time(s).
        period (float): Period of the schedule.

    Returns:
        Array: Presence probability at each time, clipped away from 0 and 1.
    """
    # Vectorize the computation over multiple inputs
    times = jnp.atleast_1d(t)
    return jax.vmap(per_timestep_piecewise, in_axes=(0, None, None))(times, intervals, period)


def get_prior_function(prior) -> Partial:
    """
    Returns log survival function.
    """
    if prior == "exponential":
        return Partial(log_survival_exponential)
    elif prior == "lognorm":
        return Partial(log_survival_lognormal)
    else:
        raise NotImplementedError(f"Prior '{prior}' is not implemented.")

@struct.dataclass
class PiecewisePrior:
    state: ArrayLike
    kwargs: Dict[str, Any] = struct.field(pytree_node=False)
    function: Partial
