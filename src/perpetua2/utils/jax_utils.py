from typing import Tuple, Callable
from jax import Array
from jax.typing import ArrayLike

from functools import partial
from jax.tree_util import Partial
import jax.lax
import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes


info = jnp.finfo(dtypes.canonicalize_dtype(jnp.float32))
posinf = info.max
neginf = info.min

LOWER_LOGSPACE_BOUND = jnp.log(1e-10)
LOWER_PSPACE_BOUND = jax.lax.exp(LOWER_LOGSPACE_BOUND)
HIGHER_PSPACE_BOUND = 0.999999
HIGHER_LOGSPACE_BOUND = jnp.log(HIGHER_PSPACE_BOUND)


def bool_ifelse(boolean_or_boolfloat: ArrayLike, if_true: ArrayLike, if_false: ArrayLike) -> Array:
    boolean_or_boolfloat = jnp.asarray(boolean_or_boolfloat).astype(jnp.float32)

    return boolean_or_boolfloat * if_true + (1 - boolean_or_boolfloat) * if_false


@jax.jit
def logsumexp(a: ArrayLike, b: ArrayLike) -> Array:
    """
    Computes a stable logsumexp for two arrays element-wise:
    log(exp(a) + exp(b)) in a numerically stable way.

    Args:
        a (jnp.ndarray): First array of values.
        b (jnp.ndarray): Second array of values.

    Returns:
        jnp.ndarray: The result of the stable logsumexp for each element of a and b.
    """
    # Compute the maximum value for stability
    max_val = jnp.maximum(a, b)

    # Compute the logsumexp using the stable formula
    return jnp.log(jnp.exp(a - max_val) + jnp.exp(b - max_val)) + max_val


# Recursive function to traverse and convert lists to JAX arrays
def convert_lists_to_arrays(d):
    if isinstance(d, dict):
        # Recursively apply to dictionary values
        return {k: convert_lists_to_arrays(v) for k, v in d.items()}
    elif isinstance(d, list) or isinstance(d, np.ndarray):
        # Convert lists to JAX arrays
        return jnp.array(d)
    else:
        # Return value as-is for non-list, non-dict types
        return d


def ntn_logspace(arr: ArrayLike) -> Array:
    return jnp.nan_to_num(arr, nan=LOWER_LOGSPACE_BOUND, neginf=LOWER_LOGSPACE_BOUND, posinf=HIGHER_LOGSPACE_BOUND)


def ntn_pspace(arr: ArrayLike) -> Array:
    return jnp.nan_to_num(arr, nan=LOWER_PSPACE_BOUND, neginf=LOWER_PSPACE_BOUND, posinf=HIGHER_PSPACE_BOUND)


def logdiff(logx: ArrayLike, logy: ArrayLike) -> Array:
    """
    Compute the logarithm of the difference of two numbers in a numerically stable way.

    Args:
        logx (ArrayLike): The logarithm of the first number.
        logy (ArrayLike): The logarithm of the second number.

    Returns:
        Array: The logarithm of the difference of the two numbers.

    """

    iseq = jnp.allclose(jnp.exp(logy - logx), jnp.ones_like(logx))

    ifeq = LOWER_LOGSPACE_BOUND
    ifnoteq = ntn_logspace(logx + jnp.log1p(-jnp.exp(logy - logx)))
    return bool_ifelse(iseq, ifeq, ifnoteq)


def clip_log_prob(prob: ArrayLike) -> Array:
    """
    Clip the log probability to prevent numerical underflow/overflow.

    Args:
        prob (ArrayLike): The log probability to clip.

    Returns:
        Array: The clipped log probability.

    """
    return jnp.clip(prob, LOWER_LOGSPACE_BOUND, HIGHER_LOGSPACE_BOUND)


def clip_prob(prob: jnp.ndarray) -> jnp.ndarray:
    return jnp.clip(prob, LOWER_PSPACE_BOUND, HIGHER_PSPACE_BOUND)


@partial(jax.jit, static_argnums=(1,))
def create_interpolated_array(points: ArrayLike, num_steps: int) -> Tuple[Array, Callable]:
    """
    This function takes an array of points and generates a new array where the values are equally spaced by the specified num_steps.
    The new array will pass through each of the original points.

    Args:
        points (list or np.ndarray): The original array of points.
        num_steps (int): Number of interpolated points between consecutive points.

    Returns:
        np.ndarray: A new array with equally spaced values that passes through the given points.
    """

    points_shifted_left = jnp.concatenate([points[1:], jnp.atleast_1d(points[0])])

    def foreach(p1, p2):
        interpolate = jnp.linspace(p1, p2, num_steps, endpoint=False)
        return interpolate

    interpolated = jax.vmap(foreach)(points[:-1], points_shifted_left[:-1])
    interpolated = interpolated.reshape((points.shape[0] - 1) * num_steps)
    interpolated = jnp.concatenate([interpolated, points[-1][None]])

    def collect(interpolated_array):
        last_element = interpolated_array[-1][None]
        # Substract the element added at the beginning and the last element
        outshape = points.shape[0] - 2
        interpolated_array = interpolated_array[num_steps:-1].reshape(
            outshape, num_steps, *interpolated_array.shape[1:]
        )
        return jnp.concatenate([interpolated_array[:, 0], last_element]).squeeze()

    return interpolated, Partial(collect)
