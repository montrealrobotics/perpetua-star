"""Table I: mean and sample standard deviation over seeds of MAE, balanced accuracy and F1."""
import argparse
import glob
import json
import os

import numpy as np
from prettytable import PrettyTable

METHODS = [
    ("perpetua", "Perpetua"),
    ("fremen", "FreMEn"),
    ("perpetua_star_fremen", "Perpetua* (FM)"),
    ("perpetua_star_llm", "Perpetua* (LLM)"),
    ("perpetua_star_gt", "Perpetua* (GT)"),
]
METRICS = [("test/Mean Absolute Error", "MAE"), ("test/Balanced Acc", "B-Acc"), ("test/F1@0.5", "F1")]

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--results-dir", default="results/weekly")
args = parser.parse_args()

table = PrettyTable(["Method", "Seeds"] + [name for _, name in METRICS])
for method, label in METHODS:
    files = sorted(glob.glob(os.path.join(args.results_dir, "test_noise01_1w_odd_seed_*", method, "metrics.json")))
    if not files:
        table.add_row([label, 0] + ["-"] * len(METRICS))
        continue
    runs = [json.load(open(f)) for f in files]
    cells = []
    for key, _ in METRICS:
        values = np.array([run[key] for run in runs])
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        cells.append(f"{values.mean():.3f} ± {std:.3f}")
    table.add_row([label, len(runs)] + cells)
print(table)
