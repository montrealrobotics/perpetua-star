# Code adopted from https://github.com/Raziel90/pyfremen/blob/master/fremen/tests/test_fremen.py
import jax.numpy as jnp
import jax
from flax import struct
from functools import partial
from jax.tree_util import Partial
from typing import Dict, Any

from perpetua2.utils.jax_utils import clip_prob


@struct.dataclass
class FremenState:
    last_observation_time: float

    states: jnp.array
    balances: jnp.array
    amplitudes: jnp.array
    phases: jnp.array
    periods: jnp.array

    num_periods: int
    max_period: int

    gain: float
    measurements: int

    @staticmethod
    def create(num_periodicities=100, max_period=7 * 24 * 3600):
        return FremenState(
            last_observation_time=-1.0,
            num_periods=num_periodicities,
            max_period=max_period,
            gain=0.5,
            measurements=0,
            states=jnp.zeros(num_periodicities, dtype=complex),
            balances=jnp.zeros(num_periodicities, dtype=complex),
            amplitudes=jnp.zeros(num_periodicities),
            phases=jnp.zeros(num_periodicities),
            periods=max_period / jnp.arange(1, num_periodicities + 1),
        )


@struct.dataclass
class FremenPrior:
    state: FremenState
    kwargs: Dict[str, Any] = struct.field(pytree_node=False)
    function: Partial

    @property
    def orderi(self) -> Dict[str, Any]:
        return self.kwargs['orderi']


@jax.jit
def add(fremen_state, times, observations):
    """
    Adds the observations y made at time x to the model. This can be called multiple times to add further observations incrementally.

    Args:
        fremen_state (FremenState): The current state of the model.
        times (jnp.ndarray): Times of the observations.
        observations (jnp.ndarray): Observations at the corresponding times.

    Returns:
        FremenState: Updated FremenState with the new observations.
    """
    old_gain = fremen_state.gain
    new_gain = observations.sum()
    updated_gain = (fremen_state.gain * fremen_state.measurements + new_gain) / (
        fremen_state.measurements + times.shape[0]
    )

    updated_balances = jax.lax.cond(
        old_gain > 0,
        lambda _: fremen_state.balances * (updated_gain / old_gain),
        lambda _: jnp.zeros_like(fremen_state.balances),
        operand=None,
    )

    angles = times * 2 * jnp.pi / fremen_state.periods.reshape(-1, 1)
    updated_states = fremen_state.states + (observations * jnp.cos(angles) + 1j * observations * jnp.sin(angles)).sum(
        axis=1
    )
    updated_balances += (jnp.cos(angles) * updated_gain + 1j * jnp.sin(angles) * updated_gain).sum(axis=1)

    updated_measurements = fremen_state.measurements + times.shape[0]
    temp = updated_states - updated_balances
    updated_amplitudes = jnp.abs(temp) / updated_measurements
    updated_amplitudes = jnp.maximum(updated_amplitudes, 0.0)  # Ensure non-negative amplitudes
    updated_phases = jnp.angle(temp)

    order = jnp.argsort(updated_amplitudes)[::-1]

    # Reorder the arrays based on the sorted amplitudes
    sorted_states = updated_states[order]
    sorted_balances = updated_balances[order]
    sorted_amplitudes = updated_amplitudes[order]
    sorted_phases = updated_phases[order]
    sorted_periods = fremen_state.periods[order]

    # Return the updated FremenState
    return FremenState(
        states=sorted_states,
        balances=sorted_balances,
        amplitudes=sorted_amplitudes,
        phases=sorted_phases,
        periods=sorted_periods,
        num_periods=fremen_state.num_periods,
        max_period=fremen_state.max_period,
        gain=updated_gain,
        measurements=updated_measurements,
        last_observation_time=times[-1],
    )


@partial(jax.jit, static_argnames=["orderi"])
def estimate(fremen_state, times, orderi, norm_ampl=False):
    """
    Estimates the reconstructed signal at given times using the static component and the top `orderi` harmonics.

    Args:
        fremen_state (FremenState): The current state of the model.
        times (jnp.ndarray): Input times for reconstruction.
        orderi (int): Number of harmonics to use in the reconstruction.
        norm_ampl (bool, optional): If True, normalize the amplitudes of the harmonics to sum to 0.5. Defaults to False.

    Returns:
        jnp.ndarray: The estimated values of the reconstructed signal at the given times.
    """

    norm_ampl = jax.lax.cond(
        norm_ampl,
        lambda _: fremen_state.amplitudes[:orderi] / 2 / fremen_state.amplitudes[:orderi].sum(),
        lambda _: fremen_state.amplitudes[:orderi],
        operand=None,
    )
    # Calculate the harmonic estimates
    estimates = (
        2
        * norm_ampl.reshape(-1, 1)
        * jnp.cos(
            times / fremen_state.periods[:orderi].reshape(-1, 1) * 2 * jnp.pi
            - fremen_state.phases[:orderi].reshape(-1, 1)
        )
    )

    # Sum the estimates and apply the gain
    probs = fremen_state.gain + jnp.sum(estimates, axis=0)

    # Clip the probabilities to stay within [0, 1]
    probs = clip_prob(probs)

    return probs


def estimate_prior(fremen_state, times, orderi, norm_ampl=False):
    return estimate(fremen_state, times, orderi=orderi, norm_ampl=norm_ampl)


