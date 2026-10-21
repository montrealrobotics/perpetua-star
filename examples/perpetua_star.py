# Load necessary libraries
import numpy as np
import argparse
import os
import jax.numpy as jnp

from jax.tree_util import Partial

from perpetua2.utils import priors
from perpetua2.utils.filter_test_utils import run_perpetua_star
from perpetua2.utils.filter_test_utils import generate_observations, sample_observation_times, generate_presence_observations
from perpetua2.filters import Fremen
from perpetua2.utils.viz import Plotter
from perpetua2.utils.logger import setup_logger


def main(args):
    # Configure persistence filter measurement model options
    ### Data generation settings ###
    pm = 0.025  # Missed detection probability
    pf = 0.025  # False alarm probability
    lambda_r_standard = 1.0 / 5  # Standard revisitation rate
    lambda_o = 3.0  # Inter-observation rate
    p_n = 1.0 / 10.0  # p_N = probability of leaving after the last reobservation; expected # observations N = 1 / p_N
    simulation_length = 600  # Length of simulation in seconds
    sampling_period = 2.0  # Inter-sample period in seconds

    ### Model Parameters ###
    persistence_times = np.array([100, 350, 550])  # Survival times, in seconds
    emergence_times = np.array([200, 425, 650])  # Survival times, in seconds
    pi_persistence = np.ones(3) / 3  # Mixing coefficients
    pi_emergence = np.ones(3) / 3  # Mixing coefficients
    max_periodicities = 100  # Number of periodicities to consider
    orderi = 8  # Model order used during inference
    decay_rate = 0.99  # Decay rate for the filters
    # Evidence annealing factor exp(−λ⋅time)=ϵ
    # \lambda = -\frac{1}{time} \log(\epsilon) if time=100 units and eps=0.01, then λ=0.046
    alpha = 0.046
    perpetua_period = simulation_length  # Maximum period for Perpetua*
    fremen_period = simulation_length  # Period is equal to the simulation length

    ### Varia ###
    prior = args.prior
    viz = args.viz
    plotter = Plotter(viz=viz, exp_name="mixtures_evidence", fig_fmt="png")
    logger = setup_logger(os.path.join(plotter.res_path, "output.log"))

    if prior == "exponential":
        # Exponential prior settings
        params_persistence = {"lambda_": 1 / persistence_times}
        params_emergence = {"lambda_": 1 / emergence_times}
        log_s_persistence = Partial(priors.log_survival_exponential)
        log_s_emergence = Partial(priors.log_survival_exponential)
    elif prior == "lognorm":
        # Lognormal prior settings
        params_persistence = {"logmu": np.log(persistence_times), "std": np.array([0.05, 0.05, 0.025])}
        params_emergence = {"logmu": np.log(emergence_times), "std": np.array([0.05, 0.05, 0.025])}
        log_s_persistence = Partial(priors.log_survival_lognormal)
        log_s_emergence = Partial(priors.log_survival_lognormal)
    else:
        raise ValueError(f"Invalid prior option: {prior}")

    # Generate samples here
    x_sampling_times = np.arange(0, simulation_length, sampling_period)
    observation_times = sample_observation_times(lambda_r_standard, lambda_o, p_n, simulation_length)

    # Label each event as '0' (disappear) or '1' (appear)
    events = [(t, True) for t in persistence_times] + [(t, False) for t in emergence_times]
    events.sort(key=lambda x: x[0])  # sort by time

    # Fill intervals
    gt_states = np.zeros(len(x_sampling_times), dtype=bool)
    obs, obs_times = [], []
    prev_time = 0
    for time, event in events:
        state = event  # toggle to the new state from event
        temp_mask = np.where(np.logical_and(x_sampling_times >= prev_time, x_sampling_times < time), state, False)
        time_mask = np.where(np.logical_and(observation_times >= prev_time, observation_times < time), True, False)
        t_obs = observation_times[time_mask]
        # Obs
        if state:
            y_obs = generate_observations(time, t_obs, pm, pf)
        else:
            y_obs = generate_presence_observations(time, t_obs, pm, pf)
        obs.append(y_obs)
        obs_times.append(t_obs)
        gt_states = np.logical_or(gt_states, temp_mask)
        prev_time = time

    gt_states = np.where(gt_states, 1, 0)  # Convert to binary state
    gt_times = x_sampling_times
    obs = np.concatenate(obs)
    obs_times = np.concatenate(obs_times)

    # Duplicate the arrays to simulate a longer time period
    gt_times = np.concatenate([gt_times, gt_times + simulation_length])
    gt_states = np.concatenate([gt_states, gt_states])
    obs_times = np.concatenate([obs_times, obs_times + simulation_length])
    obs = np.concatenate([obs, obs])

    simulation_length *= 2
    query_times = np.arange(0, simulation_length, sampling_period)

    # Remove observations from t=150 to t=500
    fremen_obs_arr = jnp.array(obs.copy())
    fremen_t_arr = jnp.array(obs_times.copy().astype(np.float32))
    mask = np.logical_or(obs_times < 150, obs_times > 1000)
    obs = obs[mask]
    obs_times = obs_times[mask]

    ### Run the survival filter ###
    import time

    start_time = time.time()
    # Instantiate switching prior (using Fremen)
    fremen_state = Fremen.FremenState.create(
        num_periodicities=max_periodicities,
        max_period=fremen_period,
    )
    fremen_state = Fremen.add(fremen_state, fremen_t_arr, fremen_obs_arr)
    switching_state = Fremen.FremenPrior(
        state=fremen_state,
        kwargs={"orderi": orderi, "norm_ampl": False},
        function=Partial(Fremen.estimate_prior),
    )

    out_dict = run_perpetua_star(
        jnp.array(obs),
        jnp.array(obs_times.astype(np.float32)),
        jnp.array(pm),
        jnp.array(pf),
        jnp.array(pi_persistence),
        jnp.array(pi_emergence),
        jnp.array(query_times),
        log_s_persistence,
        log_s_emergence,
        params_persistence,
        params_emergence,
        switching_state=switching_state,
        decay_rate=decay_rate,
        alpha=alpha,
        period=perpetua_period,
        init_time=0.0,
    )
    belief, preds, log_evidence, log_prior, log_posterior = (
        out_dict["belief"],
        out_dict["preds"],
        out_dict["log_evidence"],
        out_dict["log_prior"],
        out_dict["log_posterior"],
    )
    belief.block_until_ready()
    preds.block_until_ready()
    log_evidence.block_until_ready()
    log_prior.block_until_ready()
    log_posterior.block_until_ready()
    end_time = time.time()
    logger.info(
        f"Time taken to run the mixture of Persistence and Emergence filters: {end_time - start_time:.4f} seconds"
    )
    # Compute the true persistence of the feature based on the survival and birth prior
    plotter.plot_perpetua_star(
        gt_times,
        gt_states,
        obs_times,
        obs,
        query_times,
        belief,
        mixture_preds=preds,
        log_evidence=log_evidence,
        log_prior=log_prior,
        log_model_posterior=log_posterior,
        location="center right",
        f_name="mixtures_evidence",
    )


if __name__ == "__main__":
    np.random.seed(0)  # Set the random seed for reproducibility
    # Argparse with different choose of prior
    parser = argparse.ArgumentParser(description="Run the Mixture of persistence filters with different prior options")
    parser.add_argument(
        "--prior",
        type=str,
        default="exponential",
        help="Choose the prior to use for the persistence filter, options are: exponential, lognorm, weibull",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold used to compute precision and recall",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        default=False,
        help="Enable vizualizations",
    )
    args = parser.parse_args()

    main(args)
