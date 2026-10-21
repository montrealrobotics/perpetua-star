# Perpetua*

Perpetua* estimates whether semi-static objects are still where they were last seen. It combines a mixture of persistence and emergence filters with a switching prior (FreMEn, an LLM-generated schedule, or a known schedule) and adapts online when an object's routine changes. It is the persistence estimator behind [Predictive Spatio-Temporal Scene Graphs for Semi-Static Scenes](https://montrealrobotics.ca/predictive-graphs/) (Predictive Graphs) and extends [Perpetua](https://montrealrobotics.ca/perpetua).

## Installation

Requires Python 3.12. With [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/montrealrobotics/perpetua-star.git
cd perpetua-star
uv sync
```

Or with pip: `pip install -e .`

Optional extras:

| Extra      | Needed for                                                          |
| ---------- | ------------------------------------------------------------------- |
| `cuda`     | Running JAX on an NVIDIA GPU                                        |
| `llm`      | Generating new LLM schedules with Gemini                            |
| `procthor` | Fitting estimators on SemiStaticSim data (`perpetua2.data.procthor`) |

Install them with `uv sync --extra <name>` or `pip install -e ".[<name>]"`. CPU is recommended (`export JAX_PLATFORMS=cpu`); it is faster for these filters.

## Examples

Each example simulates its own observations and saves figures and logs to `assets/<timestamp>/`.

```bash
# Mixtures of persistence / emergence filters (--prior exponential | lognorm | weibull)
python examples/mixture_persistence.py --prior lognorm
python examples/mixture_emergence.py --prior lognorm

# Model evidence of persistence vs. emergence mixtures (--prior exponential | lognorm | step)
python examples/mixture_evidence.py --prior lognorm

# Perpetua: switching between persistence and emergence filters
python examples/perpetua.py --prior lognorm --num_steps 10 --eps 0.1

# Perpetua*: Perpetua with a FreMEn switching prior (--prior exponential | lognorm)
python examples/perpetua_star.py --prior lognorm

# Learning mixture parameters with EM (--filter persistence | emergence)
python examples/mixture_learning_exponential.py --filter persistence --n_seq 20
python examples/mixture_learning_lognormal.py --filter persistence --n_seq 20
```

## Reproducing Table I and Figure 4

The benchmark has eight objects (four static, four semi-static) that follow a weekly schedule, observed with a noise of 0.1. Each of the five seeds has a four-week training set and a one-week test set where Monday and Friday follow the weekend schedule. The data is in `data/weekly_schedule/` and the paper's models are in `models/weekly_schedule/`.

To evaluate the paper's models (about three hours on 4 CPU cores):

```bash
scripts/reproduce_table1.sh
```

This prints Table I and saves Figure 4 to `results/weekly/figures/`. Each run's metrics and predictions are in `results/weekly/<test set>/<method>/`.

To retrain all models and evaluate them (20 to 40 minutes per model and seed):

```bash
scripts/train_weekly.sh
scripts/reproduce_table1.sh results/weekly/models results/weekly_refit
```

To retrain and evaluate a single seed:

```bash
scripts/train_weekly.sh 123456
SEEDS=123456 scripts/reproduce_table1.sh results/weekly/models results/weekly_refit
```

Retrained models differ slightly from the paper's depending on the CPU and library versions, but their results stay within the standard deviations of Table I.

### Generating new LLM schedules

The LLM prior uses the Gemini schedules saved in `models/weekly_schedule/`. To generate new ones, point `llm_schedule_dir` to an empty folder:

```bash
uv sync --extra llm
export GEMINI_API_KEY=<your key>
python entrypoint.py --config-name test_weekly seed=123456 train_file=train_noise01_4w_seed_123456 test_file=test_noise01_1w_odd_seed_265464 mask_weekends=true llm_prior=true llm_schedule_dir=<empty folder>
```

The schedules are saved next to the results as `llm_schedule_<seed>.json`. Set the model with `model=` (default `gemini-pro-latest`; the paper used `gemini-2.5-pro`). LLM answers vary, so results will differ from the paper's.

## Fitting estimators for Predictive Graphs

Predictive Graphs uses one Perpetua* estimator per object of a [SemiStaticSim](https://github.com/Velythyl/SemiStaticSim) layout. With the `procthor` extra installed (2.5 to 3.5 hours on 4 CPU cores):

```bash
scripts/train_procthor.sh <layout> <path to the layout's SemiStaticSim privileged folder>
```

The estimators are saved to `estimators/procthor/scene_<layout>/`. Add `output_dir=<folder>` to save them elsewhere.

## Using Perpetua* in another project

An `Object` holds one Perpetua* estimator per receptacle:

```python
import jax.numpy as jnp
from perpetua2.filters.PerpetuaStar import object_predict, object_update
from perpetua2.utils.filter_state import Object

obj = Object.load("tests/data/procthor_estimator.pkl")  # a fitted ProcTHOR estimator (hours)
receptacle = obj.receptacle_names[0]
belief = object_predict(obj, jnp.array([520.0]), receptacle)  # probability the object is on the receptacle
obj = object_update(obj, jnp.array([1.0]), jnp.array([521.0]), 0.05, 0.05, receptacle)  # new detection
```

## Tests

```bash
uv run pytest
```

## Citation

If you use this code, please cite:

```bibtex
@article{saavedra2026predictive,
  title   = {Predictive Spatio-Temporal Scene Graphs for Semi-Static Scenes},
  author  = {Saavedra-Ruiz, Miguel and Gauthier, Charlie and Gupta, Kumaraditya and Shahfar, Shima and
             Ellis, Kirsty and Parkison, Steven and Paull, Liam},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026}
}

@inproceedings{saavedra2025perpetua,
  title     = {Perpetua: Multi-Hypothesis Persistence Modeling for Semi-Static Environments},
  author    = {Saavedra-Ruiz, Miguel and Nashed, Samer and Gauthier, Charlie and Paull, Liam},
  booktitle = {2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages     = {1--8},
  year      = {2025}
}
```

## Acknowledgements

This code builds on the original [Perpetua implementation](https://github.com/montrealrobotics/perpetua-code). The FreMEn model is adapted from [pyfremen](https://github.com/Raziel90/pyfremen). The persistence filter (`perpetua2.filters.PersistenceFilter`) is a JAX port of [Persistence-Filter](https://github.com/david-m-rosen/Persistence-Filter), from which the observation simulation in the examples is also adapted.

## License

MIT. See [LICENSE](LICENSE).
