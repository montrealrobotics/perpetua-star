from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

from perpetua2.filters import PerpetuaStar
from perpetua2.utils import priors
from perpetua2.utils import filter_test_utils as ftu
from perpetua2.utils.filter_state import Object

DATA = Path(__file__).parent / "data"
PAPER_CODE_PREDICTIONS = DATA / "paper_code_predictions.npz"
PROCTHOR_ESTIMATOR = DATA / "procthor_estimator.pkl"

COMPONENTS = ("mixture_persistence", "mixture_emergence", "perpetua", "perpetua_star", "object")


def compute_predictions_on_fixed_inputs() -> dict:
    rng = np.random.default_rng(0)
    out = {}

    t = np.arange(1.0, 1000.0, 1.0)
    present = t <= 500
    obs = np.where(rng.random(t.size) < np.where(present, 0.99, 0.01), 1.0, 0.0).astype(np.float32)
    q = np.arange(0.0, 1000.0, 1.0)
    pi = jnp.array([0.5, 0.35, 0.15])
    surv = np.array([200.0, 500.0, 750.0])
    exp_params = {"lambda_": jnp.asarray(1 / surv)}
    logn_params = {"logmu": jnp.log(jnp.asarray(surv)), "std": jnp.array([0.05, 0.05, 0.025])}
    for name, log_s, params in [
        ("exp", Partial(priors.log_survival_exponential), exp_params),
        ("lognorm", Partial(priors.log_survival_lognormal), logn_params),
    ]:
        pers, ev = ftu.run_mixture_persistence_filters(
            jnp.asarray(obs), jnp.asarray(t), 0.01, 0.01, pi, jnp.asarray(q), log_s, params
        )
        out[f"mixture_persistence/{name}"] = np.asarray(pers)
        out[f"mixture_persistence/{name}_evidence"] = np.asarray(ev)
        emer, _ = ftu.run_mixture_emergence_filters(
            jnp.asarray(1 - obs), jnp.asarray(t), 0.01, 0.01, pi, jnp.asarray(q), log_s, params
        )
        out[f"mixture_emergence/{name}"] = np.asarray(emer)

    obj = Object.load(str(PROCTHOR_ESTIMATOR))
    r_name = obj.receptacle_names[0]
    rs = obj.select(r_name).get_receptacles
    pf, ef = rs.pf_state.state_dict, rs.ef_state.state_dict
    log_s = Partial(priors.log_survival_lognormal)
    th = np.arange(0.0, 336.0, 1.0)
    gt = ((th % 24) >= 8) & ((th % 24) < 18)
    oh = np.where(rng.random(th.size) < np.where(gt, 0.95, 0.05), 1.0, 0.0).astype(np.float32)
    qh = np.arange(0.0, 400.0, 0.5)
    probs, _, weights = ftu.run_perpetua(
        jnp.asarray(oh), jnp.asarray(th), 0.05, 0.05, pf["pi"], ef["pi"], jnp.asarray(qh), log_s, log_s,
        pf["params"], ef["params"], delta_high=0.95, delta_low=0.05, num_steps=5, eps=0.1,
    )
    out["perpetua/probs"], out["perpetua/weights"] = np.asarray(probs), np.asarray(weights)
    res = ftu.run_perpetua_star(
        jnp.asarray(oh), jnp.asarray(th), 0.05, 0.05, pf["pi"], ef["pi"], jnp.asarray(qh), log_s, log_s,
        pf["params"], ef["params"], switching_state=rs.switching_state, decay_rate=0.99, alpha=9.21034, period=24,
    )
    for k in ("belief", "log_prior", "log_evidence", "log_posterior"):
        out[f"perpetua_star/{k}"] = np.asarray(res[k])

    qo = np.arange(500.0, 700.0, 10.0)
    def predict(o):
        return np.stack([np.asarray(PerpetuaStar.object_predict(o, jnp.array([t]), r_name)) for t in qo])

    out["object/predict_initial"] = predict(obj)
    for tt, oo in zip(np.arange(504.0, 528.0, 1.0), oh[:24]):
        obj = PerpetuaStar.object_update(obj, jnp.atleast_1d(oo), jnp.atleast_1d(tt), 0.05, 0.05, r_name)
    out["object/predict_after_update"] = predict(obj)
    return out
