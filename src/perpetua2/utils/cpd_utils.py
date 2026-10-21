from typing import List

import numpy as np
import ruptures as rpt

from jax.typing import ArrayLike
import jax.numpy as jnp

from perpetua2.data.containers import Scene, Sequences, Pickupable

import logging

# A logger for this file
logger = logging.getLogger(__name__)


class TimeWeightedBernoulliCost:
    """Bernoulli likelihood cost accounting for uneven time intervals."""

    def __init__(self, signal, time_intervals, min_size: int = 5):
        self.signal = signal
        self.time_intervals = time_intervals
        self.min_size = min_size

    def error(self, start, end):
        segment = self.signal[start:end]
        intervals = self.time_intervals[start:end]
        if len(segment) == 0:
            return 0
        p = np.sum(segment * intervals) / np.sum(intervals)  # Weighted mean
        epsilon = 1e-10
        return -np.sum(intervals * (segment * np.log(p + epsilon) + (1 - segment) * np.log(1 - p + epsilon)))


def detect_change_points(observations, times, min_size: int = 2, pentalty: float = 5.0) -> List[float]:
    # Apply the binary cost function
    custom_cost = TimeWeightedBernoulliCost(observations, times, min_size=min_size)
    # Use PELT with the custom cost
    algo = rpt.Pelt(custom_cost=custom_cost).fit(observations)
    # Detect change points
    change_points = algo.predict(pen=pentalty)

    return change_points


def extract_sequences(
    sequences: Sequences,
    change_points: List[float],
    observations: ArrayLike,
    obs_times: ArrayLike,
) -> Sequences:
    start = 0
    n_change_points = len(change_points)
    for i in range(0, n_change_points - 1):
        middle = change_points[i]
        end = change_points[i + 1]
        # Extract first and second block
        first_block = observations[start:middle]
        # Validate if the cycle is death or birth
        mode_first_block = (jnp.median(first_block) >= 0.5).astype(int)
        # Offset to start observations from zero (Perpetua) or not (Perpetua*)
        offset = obs_times[start:end].min()
        # If the mode of the first block is one, then it is a persistence sequence
        if mode_first_block == 1:
            sequences.add_persistence_sequence(observations[start:end], obs_times[start:end] - offset)
        # Otherwise it is a emergence sequence
        else:
            sequences.add_emergence_sequence(observations[start:end], obs_times[start:end] - offset)
        start = middle

    # If there is only one change point, it means the entire sequence is either persistence or emergence
    if n_change_points == 1:
        mode_block = (jnp.median(observations[0:-1]) >= 0.5).astype(int)
        offset = obs_times.min()
        # These cases are for standard Perpetua
        if mode_block == 1:
            sequences.add_persistence_sequence(observations, obs_times - offset)
        else:
            sequences.add_emergence_sequence(observations, obs_times - offset)

    return sequences


def extract_periodic_sequences(
    sequences: Sequences,
    change_points: List[float],
    observations: ArrayLike,
    obs_times: ArrayLike,
) -> Sequences:
    start = 0
    n_change_points = len(change_points)
    first_collected = (
        False  # Ensure that if feature starts as persistence, we still collect an emergence sequence at the start
    )
    for i in range(0, n_change_points - 1):
        middle = change_points[i]
        end = change_points[i + 1]
        # Extract first and second block
        first_block = observations[start:middle]
        # Validate if the cycle is death or birth
        mode_first_block = (jnp.median(first_block) >= 0.5).astype(int)
        # If the mode of the first block is one, then it is a persistence sequence
        if mode_first_block == 1:
            sequences.add_persistence_sequence(observations[start:end], obs_times[start:end])
            if not first_collected:
                sequences.add_emergence_sequence(observations[start:middle], obs_times[start:middle])
                first_collected = True
        # Otherwise it is a emergence sequence
        else:
            sequences.add_emergence_sequence(observations[start:end], obs_times[start:end])
            if not first_collected:
                sequences.add_persistence_sequence(observations[start:middle], obs_times[start:middle])
                first_collected = True
        start = middle

    # If there is only one change point, it means the entire sequence is either persistence or emergence
    if n_change_points == 1:
        # This case is for Perpetua*
        sequences.add_persistence_sequence(observations, obs_times)
        sequences.add_emergence_sequence(observations, obs_times)

    return sequences


def compute_sequences(scene, min_size=5, penalty=3) -> Scene:
    """
    Compute the sequences of observations from the dictionary of observations.
    """
    new_scene = Scene(name=scene.name)
    for pickupable in scene:
        new_pickupable = Pickupable(id=pickupable.id, name=pickupable.name)
        for receptacle in pickupable:
            sequences = Sequences(id=receptacle.id, name=receptacle.name)
            # Compute change times for observations
            logger.info(f"Computing sequence for the {pickupable.name} on {receptacle.name}...")
            obs, t = np.array(receptacle.states), np.array(receptacle.timestamps)
            cp_times_obs = detect_change_points(
                obs,
                t,
                min_size=min_size,
                pentalty=penalty,
            )
            # Extract the death sequences
            sequences = extract_sequences(sequences, cp_times_obs, obs, t)
            sequences.to_jax()
            new_pickupable.add_receptacle(sequences)
        new_scene.add_pickupable(new_pickupable)
    return new_scene


def compute_periodic_sequences(scene: Scene, period: float, min_size=5, penalty=3) -> Scene:
    """
    Identify and segment periodic observation sequences within a scene.

    This function analyzes the dictionary of observations in a given scene
    to detect periodic sequences based on a specified period. It uses
    changepoint detection techniques to segment the data into intervals that
    reflect stable or repeating patterns.

    Parameters
    ----------
    scene : Scene
        The input scene object containing the observations to analyze.
    period : float
        The expected temporal or spatial period of repetition.
    min_size : int, optional, default=5
         This parameter controls the minimum number of samples required
         between two consecutive change points, or the minimum length of
         any given segment. It helps to ensure that detected segments are
         of a meaningful length and can reduce sensitivity to minor
         fluctuations or noise in the data by preventing change points
         from being placed too close together.
    penalty : float, optional, default=3
        Regularization term controlling the number of changepoints.
        Higher values result in fewer detected changepoints. The optimal
        value depends on the data characteristics; See discussion:
        https://github.com/deepcharles/ruptures/issues/275
    """
    new_scene = Scene(name=scene.name)
    for pickupable in scene:
        new_pickupable = Pickupable(id=pickupable.id, name=pickupable.name)
        for receptacle in pickupable:
            sequences = Sequences(id=receptacle.id, name=receptacle.name)
            # Split observations into periods
            periods = jnp.floor(receptacle.timestamps / period)
            unique_periods = jnp.unique(periods)
            logger.info(f"Computing sequence for the {pickupable.name} on {receptacle.name}...")
            for p in unique_periods:
                logger.info(f"Computing sequence for day {p}...")
                # Map to numpy, it works faster
                obs = np.array(receptacle.states)[periods == p]
                t = np.array(receptacle.timestamps)[periods == p] % period
                # Compute change times for observations
                cp_times_obs = detect_change_points(
                    obs,
                    t,
                    min_size=min_size,
                    pentalty=penalty,
                )
                # Extract the death sequences
                sequences = extract_periodic_sequences(sequences, cp_times_obs, obs, t)
            # Map everything to jax
            sequences.to_jax()
            new_pickupable.add_receptacle(sequences)
        new_scene.add_pickupable(new_pickupable)
    return new_scene
