import datetime
from typing import List, Dict, Optional

import os
from copy import deepcopy

from jax import Array
import jax.numpy as jnp

from perpetua2.utils.filter_test_utils import run_perpetua, run_perpetua_star
from perpetua2.utils.cpd_utils import compute_periodic_sequences, compute_sequences
from perpetua2.data.io_utils import read_pickle, save_pickle, read_json
from perpetua2.data.containers import Pickupable, Receptacle, Scene
from perpetua2.llm.prompts import get_weekly_prompt
import logging

# A logger for this file
logger = logging.getLogger(__name__)

OBJ_LOCATIONS_PAIRS = [
    ("TV remote", "Is in drawer?", "Passive Location (disappear when used)"),
    ("Plant", "Is on windowsill?", "Passive Location (disappear when used)"),
    ("Car keys", "Is hanging on wall?", "Passive Location (disappear when used)"),
    ("Family photo", "Is on shelf?", "Mostly Static"),
    ("Coffee Mug", "Is on desk?", "Active Location (appear when used)"),
    ("TV", "Is in living room?", "Mostly Static"),
    ("Guitar", "Inside case?", "Passive Location (disappear when used)"),
    ("Couch", "Is in living room?", "Mostly Static"),
]


def process_schedule(schedule: List[Dict]) -> Dict[int, List[List[int]]]:
    """
    Convert weekly schedule into absolute minutes from start of week.

    Args:
        schedule: list of dictionaries with "id" and daily intervals.

    Returns:
        Dict mapping id -> list of [start, end] intervals in minutes.
    """
    days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

    def time_to_minutes(time_str: str) -> int:
        """Convert HH:MM to minutes since start of the day."""
        hours, minutes = map(int, time_str.split(":"))
        return hours * 60 + minutes

    result = {}
    for item in schedule:
        item_id = item["id"]
        intervals = []
        for day_idx, day in enumerate(days_of_week):
            if day not in item:
                continue
            base_minutes = day_idx * 24 * 60
            for interval in item[day]:
                start = base_minutes + time_to_minutes(interval["start"])
                end = base_minutes + time_to_minutes(interval["end"])
                intervals.append([start, end])
        result[item_id] = jnp.array(intervals)
    return result


def predict_weekly_schedule(
    scene,
    final_params,
    p_id,
    p_m,
    p_f,
    log_s,
    params_persistence,
    params_emergence,
    delta_high,
    delta_low,
    num_steps,
    eps,
    switching_state,
    decay_rate,
    alpha,
    perpetua_period,
    perpetua_star: bool = False,
):
    r_id = 0  # This dataset just has one receptacle per pickupable
    out = {}
    if perpetua_star:
        out_dict = run_perpetua_star(
            scene[p_id][r_id].states,
            scene[p_id][r_id].timestamps,
            p_m,
            p_f,
            final_params[p_id]["persistence"]["pi"],
            final_params[p_id]["emergence"]["pi"],
            scene[p_id][r_id].gt_timestamps,
            log_s,
            log_s,
            params_persistence,
            params_emergence,
            switching_state=switching_state,
            decay_rate=decay_rate,
            alpha=alpha,
            period=perpetua_period,
        )
        out["prediction"] = out_dict["belief"]
        out["log_prior"] = out_dict["log_prior"]
        out["log_evidence"] = out_dict["log_evidence"]
        out["log_posterior"] = out_dict["log_posterior"]
    else:
        # Run mixture perpetua filters
        probs, _, weights = run_perpetua(
            scene[p_id][r_id].states,
            scene[p_id][r_id].timestamps,
            p_m,
            p_f,
            final_params[p_id]["persistence"]["pi"],
            final_params[p_id]["emergence"]["pi"],
            scene[p_id][r_id].gt_timestamps,
            log_s,
            log_s,
            params_persistence,
            params_emergence,
            delta_high=delta_high,
            delta_low=delta_low,
            num_steps=num_steps,
            eps=eps,
        )

        # Pick best component
        out["prediction"] = probs[jnp.arange(probs.shape[0]), jnp.argmax(weights, axis=1)]

    return out


def read_observations(observations_list: List, ground_truth_list: List, n_pickupables: int, scene_name: str) -> Scene:
    """
    Read the observations from the json file.

    Args:
        observations_list (List): List of observations from the json file.
        ground_truth_list (List): Ground-truth states of every object from the json file.
        n_pickupables (int): Number of objects in the environment.
        scene_name (str): Name of the returned scene.

    Returns:
        Scene: Scene object containing the observations, observation times and pickupables ids.
    """
    res = {i: {"obs": [], "obs_t": []} for i in range(n_pickupables)}
    # Populate first observations and observation times
    for observations in observations_list:
        for observation in observations:
            obs, t, id = observation
            res[int(id)]["obs"].append(obs)
            res[int(id)]["obs_t"].append(t)

    scene = Scene(name=scene_name)
    ground_truth_array = jnp.array(ground_truth_list)
    # Convert lists to arrays
    for i in range(n_pickupables):
        mask = ground_truth_array[:, :, -1] == i
        pickupable = Pickupable(name=f"pickupable_{i}", id=i)
        # This dataset just has one receptacle per pickupable
        receptacle = Receptacle(
            name=f"receptacle_{0}",
            id=0,
            states=jnp.array(res[i]["obs"]),
            timestamps=jnp.array(res[i]["obs_t"]),
            gt_states=ground_truth_array[mask][:, 0],
            gt_timestamps=ground_truth_array[mask][:, 1],
        )
        pickupable.add_receptacle(receptacle)
        scene.add_pickupable(pickupable)
    return scene


def train_val_split(scene: Scene, val_time: float = 10080):
    """Split the train set into train and val"""
    train_scene = deepcopy(scene)
    val_scene = None
    # If val time is -1, then do not split the data
    if val_time != -1:
        val_scene = deepcopy(scene)
        # Assert that there is enough data to split
        min_duration = jnp.inf
        for p_id in range(scene.n_pickupables):
            for r_id in range(scene[p_id].n_receptacles):
                timestamps = scene[p_id][r_id].timestamps
                # Calculate actual duration
                rec_start = timestamps.min()
                rec_end = timestamps.max()
                rec_duration = rec_end - rec_start

                # Track the shortest duration in the dataset
                min_duration = jnp.minimum(rec_duration, min_duration)
        assert (
            min_duration > val_time
        ), f"Not enough data to split, min duration: {min_duration}, val_time: {val_time}. Consider collecting more data or updating val_time."
        # Split actual data
        for p_id in range(scene.n_pickupables):
            for r_id in range(scene[p_id].n_receptacles):
                pickupable = scene[p_id]
                receptacle = pickupable[r_id]
                obs = receptacle.states
                obs_t = receptacle.timestamps
                gt_obs = receptacle.gt_states
                gt_t = receptacle.gt_timestamps
                # Split the data
                split_index_obs = jnp.searchsorted(obs_t, obs_t.max() - val_time)
                split_index_gt = jnp.searchsorted(gt_t, gt_t.max() - val_time)
                val_scene[p_id][r_id].states = obs[split_index_obs:]
                val_scene[p_id][r_id].timestamps = obs_t[split_index_obs:]
                val_scene[p_id][r_id].gt_states = gt_obs[split_index_gt:]
                val_scene[p_id][r_id].gt_timestamps = gt_t[split_index_gt:]
                # Update the training data
                train_scene[p_id][r_id].states = obs[:split_index_obs]
                train_scene[p_id][r_id].timestamps = obs_t[:split_index_obs]
                train_scene[p_id][r_id].gt_states = gt_obs[:split_index_gt]
                train_scene[p_id][r_id].gt_timestamps = gt_t[:split_index_gt]
    return train_scene, val_scene


def mask_test_data(
    scene: Scene,
    mask_periodic: bool,
    mask_weekends: bool,
) -> Dict[int, Dict[str, Array]]:
    """Mask the test set in a periodic way or during weekends"""
    masked_scene = deepcopy(scene)
    # Remove all weekend observations
    if mask_weekends:
        logger.info("Masking weekends in the test set...")
        # The -1 removes the firs minute of Saturday
        cut_time = 5 * 24 * 60 - 1  # Five days in minutes
        for p_id in range(scene.n_pickupables):
            r_id = 0  # This dataset just has one receptacle per pickupable
            mask = scene[p_id][r_id].timestamps <= cut_time
            masked_scene[p_id][r_id].states = scene[p_id][r_id].states[mask]
            masked_scene[p_id][r_id].timestamps = scene[p_id][r_id].timestamps[mask]
            masked_scene[p_id][r_id].gt_states = scene[p_id][r_id].gt_states
            masked_scene[p_id][r_id].gt_timestamps = scene[p_id][r_id].gt_timestamps
    elif mask_periodic:
        logger.info("Masking periodically in the test set...")
        # Remove observations periodically from 8am to 4pm every day
        # The -1 removes the firs minute of (non-inclusive)
        start, end, period = 60 * 8 - 1, 60 * 16 - 1, 60 * 24 - 1
        for p_id in range(scene.n_pickupables):
            r_id = 0  # This dataset just has one receptacle per pickupable
            mask = jnp.logical_and(
                scene[p_id][r_id].timestamps % period >= start, scene[p_id][r_id].timestamps % period < end
            )
            masked_scene[p_id][r_id].states = scene[p_id][r_id].states[~mask]
            masked_scene[p_id][r_id].timestamps = scene[p_id][r_id].timestamps[~mask]
            masked_scene[p_id][r_id].gt_states = scene[p_id][r_id].gt_states
            masked_scene[p_id][r_id].gt_timestamps = scene[p_id][r_id].gt_timestamps
    else:
        logger.info("No masking applied to the test set...")

    return masked_scene


def prepare_data(
    data_path: str,
    train_file: str,
    test_file: str,
    val_time: float,
    mask_periodic: bool,
    mask_weekends: bool,
    n_landmarks: int,
    period: Optional[float] = None,
):
    """
    Process room data and compute train sequences to train model
    """
    train_obs_file = os.path.join(data_path, "processed", f"{train_file}.pkl")
    test_obs_file = os.path.join(data_path, "processed", f"{test_file}.pkl")
    sequence_file = os.path.join(
        data_path, "processed", f"sequence_periodic_{train_file}.pkl" if period else f"sequence_{train_file}.pkl"
    )
    files = [train_obs_file, test_obs_file, sequence_file]

    if all(os.path.isfile(f) for f in files):
        logger.info("Loading data...")
        train_scene, test_scene, sequences = map(read_pickle, files)
    else:
        logger.info("Processing data...")
        # Read raw data
        train_data = read_json(os.path.join(data_path, f"{train_file}.json"))
        test_data = read_json(os.path.join(data_path, f"{test_file}.json"))
        train_scene = read_observations(
            train_data["binary_measurements"], train_data["gt_binary_measurements"], n_landmarks, "train_scene"
        )
        test_scene = read_observations(
            test_data["binary_measurements"], test_data["gt_binary_measurements"], n_landmarks, "test_scene"
        )
        if period:
            sequences = compute_periodic_sequences(train_scene, period=period)
        else:
            sequences = compute_sequences(train_scene)
        # Save data so it does not have to be processed again next time
        save_pickle(train_obs_file, train_scene)
        save_pickle(test_obs_file, test_scene)
        save_pickle(sequence_file, sequences)
        del train_data, test_data

    prior_train_scene, prior_val_scene = train_val_split(train_scene, val_time=val_time)
    test_scene = mask_test_data(test_scene, mask_periodic, mask_weekends)

    return train_scene, test_scene, sequences, prior_train_scene, prior_val_scene


def prepare_data_llm(scene: Scene, skip_every: int = 10) -> Scene:
    out_scene = deepcopy(scene)

    def minutes_to_datetime(minutes: jnp.ndarray, start_day: str = "monday") -> jnp.ndarray:
        """
        Convert minutes (from Monday 00:00) into datetime objects.
        """
        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        start_idx = days_of_week.index(start_day.lower())
        start_date = datetime.datetime(2023, 1, 2 + start_idx)  # pick arbitrary Monday
        return [(start_date + datetime.timedelta(minutes=int(m))).strftime("%a: %H:%M") for m in minutes]

    for pickupable in out_scene:
        for receptacle in pickupable:
            receptacle.states = receptacle.states[::skip_every].astype(jnp.int16).tolist()
            receptacle.timestamps = minutes_to_datetime(receptacle.timestamps[::skip_every])
            # Make these none to save some space
            receptacle.gt_states = None
            receptacle.gt_timestamps = None

    return out_scene


def parse_llm_time_str_to_min(schedule: List[List[str]]) -> Array:
    """Convert 'Day HH:MM' to minutes from Monday 00:00."""
    day_to_offset = {
        "Mon:": 0,
        "Tue:": 24 * 60,
        "Wed:": 2 * 24 * 60,
        "Thu:": 3 * 24 * 60,
        "Fri:": 4 * 24 * 60,
        "Sat:": 5 * 24 * 60,
        "Sun:": 6 * 24 * 60,
    }

    def str_to_min(interval: List[str]) -> int:
        start, end = interval.split(" - ")

        def time_str_to_min(time_str: str) -> int:
            day, hm = time_str.split()
            hour, minute = map(int, hm.split(":"))
            return day_to_offset[day] + hour * 60 + minute

        return [time_str_to_min(start), time_str_to_min(end)]

    minute_intervals = jnp.array([str_to_min(interval) for interval in schedule])
    return minute_intervals


def compute_llm_schedule(
    scene: Scene, model: str, llm_dict_schedule: Dict[str, jnp.ndarray] = None
) -> Dict[str, Array]:
    """Compute LLM schedule from string representation to minutes."""
    for p_id in range(scene.n_pickupables):
        for r_id in range(scene[p_id].n_receptacles):
            prompt = get_weekly_prompt(
                object_name=OBJ_LOCATIONS_PAIRS[p_id][0],
                object_location=OBJ_LOCATIONS_PAIRS[p_id][1],
                object_type=OBJ_LOCATIONS_PAIRS[p_id][2],
            )
            if llm_dict_schedule[p_id].get(r_id) is None:
                from perpetua2.llm.query import query_weekly_schedule

                llm_schedule = query_weekly_schedule(prompt, model=model)
                llm_dict_schedule[p_id][r_id] = llm_schedule.copy()
            else:
                llm_schedule = llm_dict_schedule[p_id][r_id].copy()
    return llm_dict_schedule
