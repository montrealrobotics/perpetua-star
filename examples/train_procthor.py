import os
from copy import deepcopy
from pathlib import Path

from jax import random
import jax.numpy as jnp
import numpy as np
import random as rnd

from perpetua2.utils.viz import ProcthorPlotter
from perpetua2.utils.logger import pretty_print_config
from perpetua2.utils.inference import fit_fremen_scene
from perpetua2.data.procthor import prepare_data, dump_object_data, compute_scene_predictions

from omegaconf import DictConfig, OmegaConf
import logging

# A logger for this file
logger = logging.getLogger(__name__)


def main(cfg: DictConfig):
    pretty_print_config(cfg)
    fig_fmt = "png"

    # Check noise levels are the same
    assert cfg.models.p_m == cfg.models.p_f, "We add equal noise to false positives and false negatives"

    # Seed the experiment
    seed = cfg.seed
    np.random.seed(seed)
    rnd.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    key = random.PRNGKey(seed)

    plotter = ProcthorPlotter(viz=cfg.viz, exp_name="procthor", res_path=cfg.results_dir, fig_fmt=fig_fmt)
    key, subkey = random.split(key)
    train_scene, test_scene, sequences, prior_train_scene, prior_val_scene = prepare_data(
        cfg.sequence_path,
        cfg.models.perpetua_period,
        noise=cfg.models.p_m,
        min_size=5,
        penalty=3,
        val_time=cfg.val_time,
        test_time=cfg.test_time,
        key=subkey,
    )
    # Visualize the data
    for pickupable in train_scene:
        plotter.plot_environment(pickupable, f_name=f"Observations_{pickupable.name}", interval=(0, 10080))

    # Fit all objects in the scene
    key, subkey = random.split(key)
    objects = fit_fremen_scene(
        sequences=sequences,
        scene=train_scene,
        filter_config=cfg.models,
        train_config=cfg.training,
        prior_train_scene=prior_train_scene,
        prior_val_scene=prior_val_scene,
        plotter=plotter,
        key=key,
    )

    # Evaluate and plot results on the test scene
    compute_scene_predictions(
        objects=deepcopy(objects),
        scene=test_scene,
        p_m=jnp.array(cfg.models.p_m),
        p_f=jnp.array(cfg.models.p_f),
        plotter=plotter,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for obj in objects:
        obj.dump(str(output_dir / f"{obj.name.replace('|', '_')}_object.pkl"))
    logger.info(f"Saved {len(objects)} estimators to {output_dir}")

    dump_object_data(objects, path=os.path.join(cfg.results_dir, "object_data.json"))
    OmegaConf.save(cfg, os.path.join(cfg.results_dir, "config.yaml"), resolve=True)
    logger.info(f"Figures and parameter summary saved in {cfg.results_dir}")

    return 0
