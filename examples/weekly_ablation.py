import os
import re
from jax.tree_util import Partial

from collections import defaultdict
from copy import deepcopy
import jax
from jax import random
import numpy as np
import jax.numpy as jnp
import random as rnd

from perpetua2.utils.viz import WeeklyPlotter
from perpetua2.utils.logger import pretty_print_config
from perpetua2.utils.inference import fit_filters, fit_fremen
from perpetua2.utils.priors import piecewise_prior, get_prior_function, PiecewisePrior
from perpetua2.utils.jax_utils import convert_lists_to_arrays
from perpetua2.utils.metrics import compute_and_report_metrics

from perpetua2.learning.learning_utils import initialize_container, print_parameters

from perpetua2.data.io_utils import read_json, read_yaml, save_json
import perpetua2.data.weekly_schedule as data_utils

from omegaconf import DictConfig, OmegaConf
import logging

# A logger for this file
logger = logging.getLogger(__name__)


def model_name(cfg: DictConfig) -> str:
    return "perpetua_star" if cfg.models.get("perpetua_period", None) else "original"


def evaluation_method(cfg: DictConfig) -> str:
    if not cfg.models.get("perpetua_period", None):
        return "perpetua"
    if cfg.fremen:
        return "fremen"
    if cfg.gt_prior:
        return "perpetua_star_gt"
    if cfg.llm_prior:
        return "perpetua_star_llm"
    return "perpetua_star_fremen"


def save_run(output_dir: str, cfg: DictConfig, metrics: dict) -> None:
    OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"), resolve=True)
    save_json(os.path.join(output_dir, "metrics.json"), metrics)
    logger.info(f"Results saved in {output_dir}")


def main(cfg: DictConfig):
    pretty_print_config(cfg)
    fig_fmt = "png"

    # Access general configuration variables
    seed = cfg.seed
    n_landmarks = 8

    output_dir = os.path.join(cfg.results_dir, "train", f"{model_name(cfg)}_seed_{seed}")
    plotter = WeeklyPlotter(viz=cfg.viz, exp_name="weekly", res_path=output_dir, fig_fmt=fig_fmt)

    # Access model configuration variables
    delta_high, delta_low, p_m, p_f, eps, num_steps, decay_rate = (
        cfg.models.delta_high,
        cfg.models.delta_low,
        cfg.models.p_m,
        cfg.models.p_f,
        cfg.models.eps,
        cfg.models.num_steps,
        cfg.models.decay_rate,
    )
    alpha = cfg.models.get("alpha", None)
    perpetua_period = cfg.models.get("perpetua_period", None)
    num_periodicities = cfg.models.get("num_periodicities", None)
    fremen_period = cfg.models.get("fremen_period", None)
    max_order = cfg.models.get("max_order", None)
    prior = cfg.training.prior
    threshold = 0.5

    # Seed the experiment
    np.random.seed(seed)
    rnd.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    key = random.PRNGKey(seed)

    # Prepare data
    (
        train_scene,
        test_scene,
        sequences,
        prior_train_scene,
        prior_val_scene,
    ) = data_utils.prepare_data(
        cfg.data_path,
        cfg.train_file,
        cfg.test_file,
        cfg.val_time,
        cfg.mask_periodic,
        cfg.mask_weekends,
        n_landmarks,
        period=perpetua_period,
    )

    plotter.plot_schedule(test_scene, plt_gt=True, f_name="gt_test_schedule", title="Ground Truth Schedule")
    plotter.plot_environment(test_scene, f_name="feature_observations", interval=(0, 10080))

    # Define templates for each prior type
    final_params = initialize_container(prior, n_landmarks)
    log_s = get_prior_function(prior)

    # Compute the best fremen order if using Perpetua*
    orderi = defaultdict(dict) if perpetua_period else None

    test_pred_list = []
    r_id = 0  # We only have one receptacle per pickupable in this experiment
    for p_id in range(sequences.n_pickupables):
        logger.info("=" * 50)
        logger.info(f"Training landmark {sequences[p_id].name}")
        logger.info("=" * 50)
        key, subkey = random.split(key)
        out = fit_filters(
            sequences[p_id][r_id],
            cfg.models,
            cfg.training,
            plotter,
            subkey,
        )

        final_params[p_id]["persistence"] = out["persistence"]
        final_params[p_id]["emergence"] = out["emergence"]
        jax.clear_caches()

        # Predict using the computed params
        params_persistence = final_params[p_id]["persistence"].copy()
        params_emergence = final_params[p_id]["emergence"].copy()
        params_persistence.pop("pi")
        params_emergence.pop("pi")

        # Fit switching prior
        switching_state = None, None, None
        if perpetua_period:
            switching_state = fit_fremen(
                train_scene[p_id][r_id],
                num_periodicities,
                fremen_period,
                max_order=max_order,
                train_orderi_receptacle=prior_train_scene[p_id][r_id],
                val_orderi_receptacle=prior_val_scene[p_id][r_id],
                threshold=threshold,
            )
            orderi[p_id][r_id] = switching_state.kwargs["orderi"]

        # Process test predictions
        out_dict = data_utils.predict_weekly_schedule(
            test_scene,
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
            perpetua_star=bool(perpetua_period),
        )

        test_pred_list.append(out_dict["prediction"])

        # Clear jax caches to prevent OOM
        jax.clear_caches()

    # Define landmark indices for all and semi-static landmarks
    all_landmarks = range(n_landmarks)
    # Reuse the test scene container to plot the predicted schedule
    results_scene = deepcopy(test_scene)
    for p_id in range(sequences.n_pickupables):
        results_scene[p_id][r_id].gt_states = test_pred_list[p_id].copy()
    plotter.plot_schedule(results_scene, plt_gt=True, f_name="pred_test_schedule", title="Predicted Schedule")
    plotter.plot_results(test_scene, test_pred_list, f_name="test_predictions", interval=(0, 24 * 60 * 7))

    results = compute_and_report_metrics(test_scene, test_pred_list, all_landmarks, threshold, "Perpetua Test", "test")

    os.makedirs(cfg.models_dir, exist_ok=True)
    save_json(os.path.join(cfg.models_dir, f"{model_name(cfg)}_{seed}.json"), final_params)
    if orderi:
        save_json(os.path.join(cfg.models_dir, f"coefficients_{seed}.json"), orderi)
    logger.info(f"Fitted models saved in {cfg.models_dir}")
    save_run(output_dir, cfg, results)
    return 0


#################
## Test method ##
#################
def test(cfg: DictConfig):
    pretty_print_config(cfg)
    fig_fmt = "png"

    # Access general configuration variables
    seed = cfg.seed
    n_landmarks = 8

    output_dir = os.path.join(cfg.results_dir, cfg.test_file, evaluation_method(cfg))
    plotter = WeeklyPlotter(viz=cfg.viz, exp_name="weekly", res_path=output_dir, fig_fmt=fig_fmt)

    # Access model configuration variables
    delta_high, delta_low, p_m, p_f, eps, num_steps, decay_rate = (
        cfg.models.delta_high,
        cfg.models.delta_low,
        cfg.models.p_m,
        cfg.models.p_f,
        cfg.models.eps,
        cfg.models.num_steps,
        cfg.models.decay_rate,
    )
    alpha = cfg.models.get("alpha", None)
    perpetua_period = cfg.models.get("perpetua_period", None)
    num_periodicities = cfg.models.get("num_periodicities", None)
    fremen_period = cfg.models.get("fremen_period", None)

    threshold = 0.5

    # Seed the experiment
    np.random.seed(seed)
    rnd.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Prepare data
    (
        train_scene,
        test_scene,
        _,
        _,
        _,
    ) = data_utils.prepare_data(
        cfg.data_path,
        cfg.train_file,
        cfg.test_file,
        cfg.val_time,
        cfg.mask_periodic,
        cfg.mask_weekends,
        n_landmarks,
        period=perpetua_period,
    )
    train_llm_scene = data_utils.prepare_data_llm(train_scene, skip_every=cfg.skip_every)

    plotter.plot_schedule(test_scene, plt_gt=True, f_name="gt_test_schedule", title="Ground Truth Schedule")
    plotter.plot_environment(test_scene, f_name="feature_observations", interval=(0, 10080))

    final_params = read_json(os.path.join(cfg.params_dir, f"{model_name(cfg)}_{seed}.json"))
    final_params = {int(k): v for k, v in convert_lists_to_arrays(final_params).items()}
    orderi = read_json(os.path.join(cfg.params_dir, f"coefficients_{seed}.json"))
    orderi = {
        int(p_id): {int(r_id): order for r_id, order in r.items()}
        for p_id, r in convert_lists_to_arrays(orderi).items()
    }
    # Get GT schedule
    is_odd = bool(re.search(r"(^|_)odd(_|$)", cfg.test_file))
    schedule_file = os.path.join(cfg.data_path, "gt_schedule_odd.yaml" if is_odd else "gt_schedule.yaml")
    gt_schedule = read_yaml(schedule_file)
    gt_schedule = data_utils.process_schedule(gt_schedule)
    llm_schedule_path = os.path.join(cfg.llm_schedule_dir, f"llm_schedule_{seed}.json")
    if os.path.exists(llm_schedule_path):
        llm_dict_schedule = {int(k): {int(inner_k): inner_v for inner_k, inner_v in v.items()} for k, v in read_json(llm_schedule_path).items()}
    else:
        llm_dict_schedule = defaultdict(dict)
    llm_dict_schedule = (
        data_utils.compute_llm_schedule(train_llm_scene, cfg.model, llm_dict_schedule) if cfg.llm_prior else None
    )

    log_s = get_prior_function(cfg.training.prior)
    # Plot parameters
    print_parameters(final_params, scale=60)

    r_id = 0  # We only have one receptacle per pickupable in this experiment
    res_dict = defaultdict(dict)
    test_pred_list = []
    for p_id in range(n_landmarks):
        # Predict using the computed params
        params_persistence = final_params[p_id]["persistence"].copy()
        params_emergence = final_params[p_id]["emergence"].copy()
        params_persistence.pop("pi")
        params_emergence.pop("pi")

        # Fit switching prior
        switching_state = None
        if perpetua_period:
            if cfg.gt_prior:
                switching_state = PiecewisePrior(
                    state=gt_schedule[p_id],
                    kwargs={"period": 24 * 60 * 7},
                    function=Partial(piecewise_prior),
                )
            elif cfg.llm_prior:
                llm_schedule = llm_dict_schedule[p_id][r_id].copy()
                llm_schedule = data_utils.parse_llm_time_str_to_min(llm_schedule)
                switching_state = PiecewisePrior(
                    state=llm_schedule,
                    kwargs={"period": 24 * 60 * 7},
                    function=Partial(piecewise_prior),
                )
            else:
                switching_state = fit_fremen(
                    train_scene[p_id][r_id], num_periodicities, fremen_period, orderi=orderi[p_id][r_id]
                )

        # Process test predictions
        out_dict = data_utils.predict_weekly_schedule(
            test_scene,
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
            perpetua_star=bool(perpetua_period),
        )

        test_pred_list.append(out_dict["prediction"] if not cfg.fremen else jnp.exp(out_dict["log_prior"]))
        # If fremen, re-write prediction dict with log prior
        if cfg.fremen:
            out_dict["prediction"] = jnp.exp(out_dict["log_prior"])
        res_dict[p_id][r_id] = out_dict

        # Clear jax caches to prevent OOM
        jax.clear_caches()

    # Define landmark indices for all and semi-static landmarks
    all_landmarks = range(n_landmarks)
    # Create plotting results and plot results
    results_scene = deepcopy(test_scene)
    for p_id in range(n_landmarks):
        results_scene[p_id][r_id].gt_states = test_pred_list[p_id].copy()
    plotter.plot_schedule(results_scene, plt_gt=True, f_name="pred_test_schedule", title="Predicted Schedule")
    plotter.plot_results(test_scene, test_pred_list, f_name="test_predictions", interval=(0, 24 * 60 * 7))

    results = compute_and_report_metrics(test_scene, test_pred_list, all_landmarks, threshold, "Perpetua Test", "test")

    if cfg.llm_prior:
        save_json(os.path.join(output_dir, f"llm_schedule_{seed}.json"), llm_dict_schedule)

    # Store prediction assets, used for plotting later
    res_dict["scene"] = test_scene
    data_utils.save_pickle(os.path.join(output_dir, f"prediction_assets_{seed}.pkl"), res_dict)
    save_run(output_dir, cfg, results)
    return 0
