"""Figure 4: predictions of each method for one object of the long-weekend test set."""
import argparse
import os
import pickle

from perpetua2.utils.viz import PlotterPaperResults

METHODS = [
    ("fremen", "FreMEn"),
    ("perpetua", "Perpetua"),
    ("perpetua_star_fremen", "Perpetua* (FreMEn)"),
    ("perpetua_star_llm", "Perpetua* (LLM)"),
    ("perpetua_star_gt", "Perpetua* (GT)"),
]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--results-dir", default="results/weekly")
parser.add_argument("--seed", type=int, default=698754, help="Training seed of the plotted run")
parser.add_argument("--test-file", default="test_noise01_1w_odd_seed_898462", help="Test set paired with --seed")
parser.add_argument("--object", type=int, default=6, help="Index of the plotted object (6: guitar)")
parser.add_argument("--output-dir", default="results/figures")
args = parser.parse_args()

r_id = 0
skip_every = 5

predictions = []
for method, _ in METHODS:
    with open(os.path.join(args.results_dir, args.test_file, method, f"prediction_assets_{args.seed}.pkl"), "rb") as f:
        prediction_assets = pickle.load(f)
    predictions.append(prediction_assets[args.object][r_id]["prediction"])

# Get x-axis info and ground truth
test_scene = prediction_assets["scene"]
receptacle = test_scene[args.object][r_id]

plotter = PlotterPaperResults(viz=False, exp_name="weekly_experiment", res_path=args.output_dir, fig_fmt="pdf")
plotter.plot_weekly_results(
    query_times=receptacle.gt_timestamps,
    true_state=receptacle.gt_states,
    observations=receptacle.states[::skip_every],
    obs_times=receptacle.timestamps[::skip_every],
    predictions=predictions,
    methods=[label for _, label in METHODS],
    interval=(0, 24 * 60 * 7),
    f_name="n_params_benchmark_prediction",
)
