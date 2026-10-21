import logging
from typing import List, Dict

from pathlib import Path
from natsort import natsorted
import jax.numpy as jnp
from jax import random

from perpetua2.data.containers import Pickupable, Receptacle, Scene
from perpetua2.data.weekly_schedule import train_val_split
from perpetua2.data.io_utils import save_json
from perpetua2.utils.cpd_utils import compute_periodic_sequences
from perpetua2.utils.filter_state import Object
from perpetua2.utils.filter_test_utils import compute_receptacle_predictions

try:
    from semistaticsim.datawrangling.sssd import load_sssd, path_2_parts
except ImportError as e:
    raise ImportError("perpetua2.data.procthor needs SemiStaticSim: pip install 'perpetua2[procthor]'") from e

logger = logging.getLogger(__name__)


def dump_object_data(objects: List[Object], path: str = "object_data.json"):
    res_dict = dict()
    for obj in objects:
        obj_name = obj.name
        obj_id = obj.id
        res_dict[obj_id] = dict(name=obj_name, receptacles=dict())
        for receptacle in obj:
            receptacle_name = receptacle.name
            r_id = receptacle.id
            persistence_params = receptacle.pf_state.state_dict
            emergence_params = receptacle.ef_state.state_dict
            orderi = receptacle.switching_state.orderi
            res_dict[obj_id]["receptacles"][r_id] = {
                "receptacle_name": receptacle_name,
                "persistence_params": persistence_params,
                "emergence_params": emergence_params,
                "orderi": orderi,
            }
    save_json(path, res_dict)


def compute_scene_predictions(
    scene: Scene,
    objects: List[Object],
    p_m: jnp.array,
    p_f: jnp.array,
    plotter=None,
):
    out_dict = dict()
    for pickupable in scene:
        p_id = pickupable.id
        p_name = pickupable.name
        # Get object with matching ID
        obj = [obj for obj in objects if obj.id == p_id][0]
        out_dict[p_id] = dict(name=p_name, receptacles=dict())
        for receptacle in pickupable:
            r_name = receptacle.name
            r_id = receptacle.id
            obj_receptacle = obj.select(r_name)
            prediction = compute_receptacle_predictions(
                receptacle,
                obj_receptacle,
                p_m=p_m,
                p_f=p_f,
            )
            out_dict[p_id]["receptacles"][r_id] = {"prediction": prediction, "receptacle_name": r_name}
        # Plot results if plotter is provided
        if plotter is not None:
            plotter.plot_pickupable(pickupable, out_dict[p_id], f_name=f"prediction_{p_name}")
    return out_dict


def read_data(folder: str, indices: List[int] = None) -> List[str] | List[str] | Dict[str, jnp.ndarray]:
    """Reads pickupables, receptacles and observations"""
    folder_path = Path(folder)
    if indices is not None:
        indices = [int(i) for i in indices]
        run_folders = natsorted([folder_path / f"run_{i}" for i in indices])
    else:
        run_folders = natsorted(folder_path.glob("run_*"))
    assert len(run_folders) > 0, f"No runs found in {folder}"
    logger.info(f"Found {len(run_folders)} runs in {folder}")
    assignments: List[jnp.ndarray] = []
    timestamps: List[jnp.ndarray] = []
    for i, run_folder in enumerate(run_folders):
        sssd = load_sssd(run_folder)
        for batch in sssd.get_generator_of_selves():
            assignments.append(batch._assignment)
            timestamps.append(batch._timestamp)
    # Stack all runs data
    raw_data = {
        "assignments": jnp.concatenate(assignments, axis=0),
        "timestamps": jnp.concatenate(timestamps, axis=0),
    }
    # Using the last sssd, read pickupable and receptacle names
    metadata = {
        "pickupable_names": sssd.pickupables_in_scene,
        "receptacle_names": sssd.receptacles_in_scene,
        "pickupables_to_receptacles": sssd.pickupable_to_receptacle,
        "receptacles_oobb": sssd.receptacles_oobb,
    }

    return raw_data, metadata


def process_wide_assignments(data: Dict[str, jnp.ndarray], receptacle_names: List[str], p_id: int) -> jnp.ndarray:
    """Convert wide assignments to single assignment per timestamp"""
    n_receptacles = data["assignments"].shape[2]
    assignments = data["assignments"][:, p_id, :]
    timestamps = data["timestamps"]
    out = dict()
    for r_id in range(n_receptacles):
        # Get positions where the pickupable is either present or absent in the receptacle
        mask = jnp.logical_or(assignments[:, r_id] == 0, assignments[:, r_id] == 1)
        # Do not populate data if there are no valid timestamps
        if jnp.sum(mask) == 0 or receptacle_names[r_id] == "OOB_FAKE_RECEPTACLE":
            continue
        out[r_id] = {"timestamps": timestamps[mask], "states": assignments[:, r_id][mask]}

    return out


def parse_assignments(
    raw_data: Dict[str, jnp.ndarray],
    metadata: Dict[str, List[str]],
    scene_name: str,
    noise: float = 0.0,
    key: random.PRNGKey = random.PRNGKey(0),
    populate_all: bool = True,
) -> List[Pickupable]:
    n_pickupables = len(metadata["pickupable_names"])
    # Initialize data container
    scene = Scene(name=scene_name)
    for p_id in range(n_pickupables):
        pickupable_data = process_wide_assignments(raw_data, metadata["receptacle_names"], p_id)
        receptacles_id = list(pickupable_data.keys())
        pickupable = Pickupable(name=metadata["pickupable_names"][p_id], id=p_id)
        for r_id in receptacles_id:
            receptacle_name = metadata["receptacle_names"][r_id]
            key, subkey = random.split(key)
            states = pickupable_data[r_id]["states"]
            noisy_states = random.bernoulli(
                subkey, p=jnp.clip(jnp.abs(states - noise), 0.0, 1.0), shape=states.shape
            ).astype(jnp.float32)
            receptacle = Receptacle(
                name=receptacle_name,
                id=r_id,
                states=noisy_states,
                timestamps=pickupable_data[r_id]["timestamps"],
                gt_states=states if populate_all else None,
                gt_timestamps=pickupable_data[r_id]["timestamps"] if populate_all else None,
            )
            pickupable.add_receptacle(receptacle)
        scene.add_pickupable(pickupable)
    return scene


def prepare_data(
    folder: str,
    period: float,
    noise: float,
    min_size: int,
    penalty: float,
    val_time: float,
    test_time: float,
    key: random.PRNGKey,
) -> Scene:
    """Prepares data for training and testing"""
    raw_data, metadata = read_data(folder)
    split, index, seed = path_2_parts(folder)
    scene_name = f"{split}_{index}_{seed}"
    # Parse the data into structured format
    full_scene = parse_assignments(raw_data, metadata, scene_name, noise=noise, key=key)
    # Split the scene into train/val/test
    train_scene, test_scene = train_val_split(full_scene, val_time=test_time)
    # Delete full scene to save memory
    del full_scene
    # Compute training sequences for each receptacle
    sequences = compute_periodic_sequences(train_scene, period=period, min_size=min_size, penalty=penalty)
    prior_train_scene, prior_val_scene = train_val_split(train_scene, val_time=val_time)

    return train_scene, test_scene, sequences, prior_train_scene, prior_val_scene

def get_data_slice(folder: str, current_time: int, get_all_week: bool = True, noise: float = 0.0, key: random.PRNGKey = random.PRNGKey(0)) -> Scene:
    '''Create a scene composed of data at a specific time slice'''
    # Get all past week data or only the last day
    interval = (24 * 7) if get_all_week else 24
    # Calculate the start time by aligning current_time to the interval grid
    start_time = (current_time // interval) * interval
    data_indices = list(range(start_time, current_time + 1))
    # Read and slice data once
    raw_data, metadata = read_data(folder, indices=data_indices)
    sliced_data = {
        "assignments": raw_data["assignments"],
        "timestamps": raw_data["timestamps"],
    }
    split, index, seed = path_2_parts(folder)
    scene_name = f"{split}_{index}_{seed}"
    return parse_assignments(sliced_data, metadata, scene_name, noise=noise, key=key, populate_all=False)


def read_gt_scene(path: str, key: random.PRNGKey = random.PRNGKey(0)) -> jnp.ndarray:
    sssd = load_sssd(path).take_by_global_timestep(None)
    data = {"assignments": sssd._assignment, "timestamps": sssd._timestamp}
    metadata = {"pickupable_names": sssd.pickupables_in_scene,
                "receptacle_names": sssd.receptacles_in_scene}
    split, index, seed = path_2_parts(path)
    scene_name = f"gt_{split}_{index}_{seed}"
    scene = parse_assignments(data, metadata, scene_name, noise=0.0, key=key, populate_all=False)
    return scene

def fetch_gt_assignment(gt_scene: Scene, pred_time: float, p_name: str, prediction: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Fetch groundtruth assignment for a given pickupable and receptacle name"""
    # Find pickupable
    pickupable = gt_scene.get_by_name(p_name)
    if pickupable is None:
        raise ValueError(f"Pickupable {p_name} not found in scene {gt_scene.name}")
    # Find receptacle
    gt_query = {}
    closest_idx = jnp.floor(pred_time).astype(jnp.int32)
    for receptacle in pickupable:
        gt_query[receptacle.name] = receptacle.states[closest_idx].item()
    if len(gt_query) == 0:
        raise ValueError(f"There is not receptacles for pickupable {p_name}")
    return gt_query
