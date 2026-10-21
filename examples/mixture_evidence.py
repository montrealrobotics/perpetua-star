# Load necessary libraries
import numpy as np
import argparse
import os
import jax.numpy as jnp

from jax.tree_util import Partial

from perpetua2.utils import priors
from perpetua2.utils.filter_test_utils import run_mixture_persistence_filters, run_mixture_emergence_filters
from perpetua2.utils.filter_test_utils import generate_observations, sample_observation_times, generate_presence_observations
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
    decay_rate = 0.99  # Decay rate for the filters

    ### Model Parameters ###
    persistence_times = np.array([100, 350, 550])  # Survival times, in seconds
    emergence_times = np.array([200, 425, 650])  # Survival times, in seconds
    pi_persistence = np.ones(3) / 3  # Mixing coefficients
    pi_emergence = np.ones(3) / 3  # Mixing coefficients

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
    elif prior == "step":
        p_times = jnp.array([[50, 350, 550], [150, 200, 400], [20, 500, 550]])  # Survival times, in seconds
        e_times = jnp.array([[200, 425, 650], [100, 300, 500], [75, 350, 800]])  # Survival times, in seconds
        p_width = jnp.array([[20, 10, 30], [20, 10, 30], [20, 10, 30]])
        e_width = jnp.array([[20, 10, 10], [20, 10, 30], [20, 10, 30]])
        amplitude = 1 / 25
        # Step prior settings
        params_persistence = {"persistence_times": p_times, "persistence_width": p_width, "amplitude": amplitude}
        params_emergence = {"persistence_times": e_times, "persistence_width": e_width, "amplitude": amplitude}
        log_s_persistence = Partial(priors.log_survival_step)
        log_s_emergence = Partial(priors.log_survival_step)
    else:
        raise ValueError(f"Invalid prior option: {prior}")

    # The set of times to query the belief of the persistence filter and emergence prior
    query_times = np.arange(0, simulation_length, sampling_period)

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

    ### Run the survival filter ###
    import time

    start_time = time.time()
    persistence_belief, persistence_evidence = run_mixture_persistence_filters(
        jnp.array(obs),
        jnp.array(obs_times.astype(np.float32)),
        jnp.array(pm),
        jnp.array(pf),
        jnp.array(pi_persistence),
        jnp.array(query_times),
        log_s_persistence,
        params_persistence,
        decay_rate=decay_rate,
        init_time=0.0,
    )
    emergence_belief, emergence_evidence = run_mixture_emergence_filters(
        jnp.array(obs),
        jnp.array(obs_times.astype(np.float32)),
        jnp.array(pm),
        jnp.array(pf),
        jnp.array(pi_emergence),
        jnp.array(query_times),
        log_s_emergence,
        params_emergence,
        decay_rate=decay_rate,
        init_time=0.0,
    )
    persistence_belief.block_until_ready()
    persistence_evidence.block_until_ready()
    emergence_belief.block_until_ready()
    emergence_evidence.block_until_ready()
    end_time = time.time()
    logger.info(
        f"Time taken to run the mixture of Persistence and Emergence filters: {end_time - start_time:.4f} seconds"
    )
    evidence = jnp.concatenate((persistence_evidence, emergence_evidence), axis=0)
    assignments = jnp.argmax(evidence, axis=0)
    belief = jnp.concatenate((persistence_belief[-1:, :], emergence_belief[-1:, :]), axis=0)
    belief = belief[assignments, jnp.arange(belief.shape[1])]
    # Compute the true persistence of the feature based on the survival and birth prior
    plotter.plot_mixtures_evidence(
        gt_times,
        gt_states,
        obs_times,
        obs,
        query_times,
        belief,
        persistence_belief[-1],
        emergence_belief[-1],
        persistence_evidence[0],
        emergence_evidence[0],
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
        "--viz",
        action="store_true",
        default=False,
        help="Enable vizualizations",
    )
    args = parser.parse_args()

    main(args)
