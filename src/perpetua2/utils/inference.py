from typing import Tuple, Callable, List, Dict, Any, Optional, Union
from jax.typing import ArrayLike

from jax.tree_util import Partial
import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from omegaconf import DictConfig
from copy import deepcopy

from perpetua2.filters import Fremen
from perpetua2.filters.PerpetuaStar import init
from perpetua2.filters.Fremen import FremenState, FremenPrior
from perpetua2.filters.PerpetuaStar import multistep_update

from perpetua2.utils.filter_state import PerpetuaStarState, Object
from perpetua2.utils.priors import log_survival_exponential, log_survival_lognormal
from perpetua2.utils.filter_test_utils import compute_querytimes, select_best_fremen_order
from perpetua2.utils.viz import Plotter

from perpetua2.data.io_utils import format_params
from perpetua2.data.containers import Receptacle, Pickupable, Scene

import perpetua2.learning.mixture_lognormal as EMLogNormal
import perpetua2.learning.mixture_exponential as EMExponential
import logging

# A logger for this file
logger = logging.getLogger(__name__)


def make_prior_fn(prior_type: str):
    if prior_type == "exponential":
        return Partial(log_survival_exponential)
    elif prior_type == "lognorm":
        return Partial(log_survival_lognormal)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


def get_fitting_functions(prior, perturbation: float = 0.1) -> Tuple[Callable, Callable, ArrayLike, Partial]:
    """
    Returns Fit method, Select and Refine method, percentage of perturbation, and prior function.
    """
    if prior == "exponential":
        return (
            EMExponential.fit,
            EMExponential.select_and_refine_model,
            jnp.array([perturbation, perturbation]),
        )
    elif prior == "lognorm":
        return (
            EMLogNormal.fit,
            EMLogNormal.select_and_refine_model,
            jnp.array([perturbation, perturbation, perturbation]),
        )
    else:
        raise NotImplementedError(f"Prior '{prior}' is not implemented.")


def default_params(prior: str):
    """
    Returns default parameters for the specified prior. This
    is used when there are no sequences for a given filter and we initialize it
    so that it reaches its steady state after 1 time unit.
    """
    if prior == "exponential":
        return {"lambda_": jnp.array([1.0]), "pi": jnp.array([1.0])}
    else:
        return {
            "logmu": jnp.array([0.0]),
            "std": jnp.array([0.1]),
            "pi": jnp.array([1.0]),
        }


def _fit_single_filter(
    receptacle: Receptacle,
    filter_name: str,
    filter_config: DictConfig,
    train_config: DictConfig,
    key: random.PRNGKey,
    plotter: Optional[Plotter] = None,
) -> Tuple[Dict[str, Any], int]:
    obs, t = receptacle.get_by_filter(filter_name)
    if len(obs) == 0:
        logger.info(f"No data for {receptacle.name} and {filter_name} filter. Skipping training.")
        return default_params(train_config.prior), 0

    delta = 1e-6
    fit_method, model_selector, perturbation = get_fitting_functions(
        train_config.prior, train_config.perturbation_scale
    )
    if train_config.prior == "lognorm":
        model_selector = partial(model_selector, clip_log_means=train_config.get("clip_log_means", True))

    logger.info(f"Training receptacle {receptacle.name} - {filter_name.capitalize()}")
    query_times, n_datapoints = compute_querytimes(t)

    fit = partial(
        fit_method,
        p_m=filter_config.p_m,
        p_f=filter_config.p_f,
        filter=filter_name,
        prior=train_config.prior,
        delta=delta,
        n_iter=train_config.n_iter,
        y_bool=obs,
        observation_times=t,
        plotter=plotter,
        exp_name=f"{receptacle.name}_{filter_name}_learning_{train_config.prior}",
    )

    results = model_selector(
        fit,
        query_times,
        n_datapoints,
        range(train_config.min_components, train_config.max_components),
        perturbation,
        num_retrain_attempts=train_config.retrain_attempts,
        prune_threshold=train_config.prune_threshold,
        key=key,
    )

    return format_params(train_config.prior, results), n_datapoints


def fit_filters(
    receptacle: Receptacle,
    filter_config: DictConfig,
    train_config: DictConfig,
    plotter: Optional[Plotter] = None,
    key: random.PRNGKey = jax.random.PRNGKey(0),
) -> Dict[str, Any]:
    """Train persistence and emergence filters for a given receptacle."""

    key, subkey = random.split(key)
    persistence_params, p_points = _fit_single_filter(
        receptacle, "persistence", filter_config, train_config, key=subkey, plotter=plotter
    )

    key, subkey = random.split(key)
    emergence_params, e_points = _fit_single_filter(
        receptacle, "emergence", filter_config, train_config, key=subkey, plotter=plotter
    )

    return {
        "persistence": persistence_params,
        "emergence": emergence_params,
        "p_datapoints": p_points,
        "e_datapoints": e_points,
    }


def fit_fremen(
    # Fitting arguments
    train_receptacle: Receptacle,
    num_periodicities: int,
    fremen_period: int,
    orderi: Optional[int] = None,
    # Order selection arguments
    max_order: int = 50,
    train_orderi_receptacle: Optional[Receptacle] = None,
    val_orderi_receptacle: Optional[Receptacle] = None,
    threshold: float = 0.5,
) -> Union[FremenState, Dict[str, float], Partial]:

    # Select best order if not provided
    if orderi is None:
        orderi = select_best_fremen_order(
            train_orderi_receptacle, val_orderi_receptacle, num_periodicities, fremen_period, max_order, threshold
        )

    # Create Fremen state
    state = Fremen.FremenState.create(num_periodicities=num_periodicities, max_period=fremen_period)
    state = Fremen.add(state, train_receptacle.timestamps, train_receptacle.states)
    fremen_kwargs = {"orderi": orderi, "norm_ampl": False}
    fremen_function = Partial(Fremen.estimate_prior)
    return FremenPrior(state=state, kwargs=fremen_kwargs, function=fremen_function)


def fit_fremen_receptacle(
    # Data
    sequences: Receptacle,
    receptacle: Receptacle,
    # Filter params
    filter_config: DictConfig,
    # Training params
    train_config: DictConfig,
    # Fremen dataloaders
    prior_train_receptacle: Optional[Receptacle] = None,
    prior_val_receptacle: Optional[Receptacle] = None,
    # Receptacle metadata
    id: int = 0,
    name: str = "",
    # Other params
    key: random.PRNGKey = jax.random.PRNGKey(0),
    plotter: Optional[Plotter] = None,
) -> PerpetuaStarState:

    # ----------------------------------------
    # 1. Fit persistence & emergence filters
    # ----------------------------------------
    fitted = fit_filters(
        sequences,
        filter_config=filter_config,
        train_config=train_config,
        plotter=plotter,
        key=key,
    )
    persistence = fitted["persistence"]
    emergence = fitted["emergence"]
    # ----------------------------------------
    # 2. Fit Fremen switching prior
    # ----------------------------------------
    switching_state = fit_fremen(
        train_receptacle=receptacle,
        num_periodicities=filter_config.num_periodicities,
        fremen_period=filter_config.fremen_period,
        orderi=filter_config.orderi,
        max_order=filter_config.max_order,
        train_orderi_receptacle=prior_train_receptacle,
        val_orderi_receptacle=prior_val_receptacle,
        threshold=0.5,
    )
    # ----------------------------------------
    # 3. Select prior function
    # ----------------------------------------
    prior_fn = make_prior_fn(train_config.prior)

    # ----------------------------------------
    # 4. Build Perpetua state
    # ----------------------------------------
    pi_persistence = persistence.pop("pi")
    pi_emergence = emergence.pop("pi")
    perpetua_state = init(
        prior_fn,
        prior_fn,
        persistence,
        emergence,
        pi_persistence,
        pi_emergence,
        switching_state,
        decay_rate=filter_config.decay_rate,
        alpha=filter_config.alpha,
        max_period=filter_config.perpetua_period,
        initialization_time=0.0,
        n_samples_persistence=fitted["p_datapoints"],
        n_samples_emergence=fitted["e_datapoints"],
        id=id,
        name=name,
    )

    return perpetua_state


def fit_fremen_object(
    # Data
    sequences: Pickupable,
    pickupable: Pickupable,
    # Filter params
    filter_config: DictConfig,
    # Training params
    train_config: DictConfig,
    # Fremen dataloaders
    prior_train_pickupable: Optional[Pickupable] = None,
    prior_val_pickupable: Optional[Pickupable] = None,
    # Other params
    key: random.PRNGKey = jax.random.PRNGKey(0),
    plotter: Optional[Plotter] = None,
) -> Object:
    """Fits Perpetua* filters for all receptacles in an object."""
    obj_name = sequences.name
    obj_id = sequences.id
    states = list()
    for r_id in range(sequences.n_receptacles):
        logger.info(f"Fitting receptacle {sequences[r_id].name} for object {obj_name} (ID: {obj_id})")
        key, subkey = random.split(key)
        receptacle_state = fit_fremen_receptacle(
            sequences=sequences[r_id],
            receptacle=pickupable[r_id],
            filter_config=filter_config,
            train_config=train_config,
            prior_train_receptacle=prior_train_pickupable[r_id],
            prior_val_receptacle=prior_val_pickupable[r_id],
            id=sequences[r_id].id,
            name=sequences[r_id].name,
            key=subkey,
            plotter=plotter,
        )
        states.append(receptacle_state)

    object = Object.create(
        states=states,
        name=obj_name,
        id=obj_id,
    )
    return object


def fit_fremen_scene(
    # Data
    sequences: Scene,
    scene: Scene,
    # Filter params
    filter_config: DictConfig,
    # Training params
    train_config: DictConfig,
    # Fremen dataloaders
    prior_train_scene: Optional[Scene] = None,
    prior_val_scene: Optional[Scene] = None,
    # Other params
    key: random.PRNGKey = jax.random.PRNGKey(0),
    plotter: Optional[Plotter] = None,
) -> List[Object]:
    """Fits Perpetua* filters for all objects in a scene."""
    objects = list()
    for p_id in range(sequences.n_pickupables):
        logger.info(f"Fitting pickupable {sequences[p_id].name} (ID: {sequences[p_id].id})")
        key, subkey = random.split(key)
        prior_train_pickupable = None if prior_train_scene is None else prior_train_scene[p_id]
        prior_val_pickupable = None if prior_val_scene is None else prior_val_scene[p_id]
        obj = fit_fremen_object(
            sequences=sequences[p_id],
            pickupable=scene[p_id],
            filter_config=filter_config,
            train_config=train_config,
            prior_train_pickupable=prior_train_pickupable,
            prior_val_pickupable=prior_val_pickupable,
            key=subkey,
            plotter=plotter,
        )
        objects.append(obj)
        jax.clear_caches()
    return objects


def update_object(
    # Data
    update_scene: Scene,
    # Objects
    object: Object,
    # Meas. model
    p_m: float,
    p_f: float,
) -> List[Object]:
    """Updates Perpetua* filters for all objects in a scene with new observations."""
    updated_object = deepcopy(object)

    # Find data for this object
    target_data = None
    for pickupable in update_scene:
        if pickupable.name == updated_object.name:
            target_data = pickupable
            break

    if target_data is None:
        return updated_object

    for receptacle in target_data:
        r_name = receptacle.name
        updated_object = multistep_update(
            state=updated_object,
            detector_output=receptacle.states,
            observation_time=receptacle.timestamps,
            p_m=p_m,
            p_f=p_f,
            receptacle_name=r_name,
        )
    return updated_object
