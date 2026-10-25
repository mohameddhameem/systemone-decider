"""Paired bootstrap over questions, and the cheapest correct option: shared by `slr` and `menu`."""

import numpy as np

from systemone_decider.evaluate_routing import aiq


def cheapest_correct_label(ok: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Index of the cheapest correct option per question; the cheapest option when none is correct."""
    return np.where(ok.any(axis=1), np.where(ok, cost, np.inf).argmin(axis=1), cost.argmin(axis=1))


def bootstrap(policies: dict, n: int, n_boot: int, seed: int, fixed_range: tuple[float, float]) -> dict:
    """Paired bootstrap over questions. `policies` maps a name to f(idx) -> points on the resampled rows."""
    rng = np.random.default_rng(seed)
    aiqs = {k: [] for k in policies}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, f in policies.items():
            aiqs[k].append(aiq(f(idx), *fixed_range))
    aiqs = {k: np.asarray(v) for k, v in aiqs.items()}
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in aiqs.items()}
    return {"n": n_boot, "seed": seed, "aiq_ci95": ci, "aiqs": aiqs}


def diff(boot: dict, a: str, b: str) -> dict:
    d = boot["aiqs"][a] - boot["aiqs"][b]
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}
