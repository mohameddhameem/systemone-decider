"""Evaluate routing policies on cached expert outputs.

For each policy, questions are escalated to the large expert in order of the policy's score,
at escalation rates from 0 to 1. Each rate gives one (mean cost, mean F1, mean EM) point.

Pre-generation policies (routers, oracle, random) pay for one expert per question. The
entropy baseline and the `+small` cascade routers decide after the small expert has answered, so
escalated questions pay for both experts.

Cost is in one of three units (--cost): `flops` (2 x parameters x tokens, in GFLOPs, so a 7B token
costs more than a 1.5B one), `tokens` (prompt + generated, model size ignored) or `seconds`
(measured wall-clock per question, averaged over each chunk). Parameter counts come from the run's
config.toml: an expert's `params` (billions), else the size in its model name.

The scorer (nano-jev) runs for every policy that uses the small expert's top-k paragraphs, so its
cost is reported separately (`scorer_cost`, using `scorer.params` from the config) rather than
added to the curves.

Router probabilities are temperature-scaled on calib; scaling keeps the ranking, so curves and AIQ
are unchanged, and the report gives ECE and reliability bins before and after.

Uncertainty (--bootstrap): a paired bootstrap over test questions. Each resample re-ranks the same
router scores, so the intervals cover test-set sampling, not router retraining. The report gives a
95% percentile interval for each policy's AIQ, for each router's AIQ minus the small-entropy and
random baselines', and for each operating point's F1 minus always-large's, with the share of
resamples in which the difference is <= 0 (a one-sided bootstrap p-value). AIQ takes the upper
hull of a curve, so even a random ranking scores at or above the random baseline's straight line:
the difference to `random` is biased upward on small test sets, while the difference between two
ranked policies (e.g. a router and small-entropy) is not.
"""

import argparse
import json
import logging
import pickle
import re
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from systemone_decider import provenance
from stair_common.cost import TABLE5_HOTPOTQA, UNIT_GFLOPS, budget

from systemone_decider.metrics import METRICS, metric_view, normalize_answer
from systemone_decider.router import LABELS, escalation_label, predict_escalation

log = logging.getLogger(__name__)

RATES = np.round(np.linspace(0, 1, 21), 2)
# agenttts: flops / one 3B pass of 128+64 tokens; table5: the HotpotQA SLR setup's fixed price per answering call
COST_UNITS = ("flops", "tokens", "seconds", "agenttts", "table5")
CPT_FRACTIONS = (0.5, 0.8)
CURVE_COLUMNS = ("small_f1", "large_f1", "small_em", "large_em", "small_cost", "large_cost", "small_uncertainty")
BASELINES = ("small-entropy (post)", "random")


def expert_params(cfg: dict) -> dict[str, float | None]:
    """Parameters in billions per expert: `params` if set, else parsed from e.g. "Qwen2.5-7B-Instruct"."""
    params = {}
    for expert in cfg["experts"]:
        if "params" in expert:
            params[expert["name"]] = float(expert["params"])
        else:
            m = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", expert.get("model", ""))
            params[expert["name"]] = float(m.group(1)) if m else None
    return params


def table5_price(model: str, params_b: float) -> float:
    """The SLR setup's price of one HotpotQA answering call: AgentTTS Table 5 for a model it lists, else the same
    formula at Table 4's answering lengths (stair_common.cost.budget, 256 + 64 tokens), with the nominal size in the
    model's name, as Table 5 counts (Qwen2.5-1.5B is 1.5, though a config may give its exact 1.54)."""
    for name, price in TABLE5_HOTPOTQA["answer"].items():
        if re.search(rf"{re.escape(name)}(?![\d.])", model):
            return float(price)
    nominal = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", model)
    return budget(float(nominal.group(1)) if nominal else params_b, 256, 64)


def judge_cost_in(unit: str, gflops):
    """A judge call's GFLOPs in a curve's unit; under table5, each call priced with stair_common.cost.budget."""
    if unit == "agenttts":
        return gflops / UNIT_GFLOPS
    if unit == "table5":
        return np.maximum(0.0, 3 * np.asarray(gflops, dtype=float) / UNIT_GFLOPS - 2)
    return gflops


def with_cost_unit(df: pd.DataFrame, unit: str, params: dict[str, float | None],
                   models: dict[str, str] | None = None) -> pd.DataFrame:
    """Copy of df whose small_cost/large_cost are in `unit` (cached costs are tokens). `table5` needs the experts'
    model names (`models`)."""
    df = df.copy()
    for name in ("small", "large"):
        if unit in ("flops", "agenttts", "table5") and params.get(name) is None:
            raise ValueError(f"no parameter count for expert '{name}': add `params = <billions>` to its "
                             "config or use --cost tokens")
        if unit in ("flops", "agenttts"):
            df[f"{name}_cost"] = 2 * params[name] * df[f"{name}_cost"] / (UNIT_GFLOPS if unit == "agenttts" else 1)
        elif unit == "table5":
            if not (models or {}).get(name):
                raise ValueError(f"cost table5 needs expert '{name}''s model name")
            df[f"{name}_cost"] = table5_price(models[name], params[name])
        elif unit == "seconds":
            df[f"{name}_cost"] = df[f"{name}_seconds"]
    return df


def outcome(df: pd.DataFrame, escalate: np.ndarray, post: bool) -> dict[str, float]:
    esc = escalate.astype(bool)
    f1 = np.where(esc, df["large_f1"], df["small_f1"])
    em = np.where(esc, df["large_em"], df["small_em"])
    cost = np.where(esc, df["large_cost"], df["small_cost"]).astype(float)
    if post:
        cost = cost + np.where(esc, df["small_cost"], 0)
    return {"rate": float(esc.mean()), "cost": float(cost.mean()), "f1": float(f1.mean()), "em": float(em.mean())}


def escalate_top(scores: np.ndarray, rate: float) -> np.ndarray:
    """Escalate the round(rate * n) highest-scoring questions (ties broken by position)."""
    n = round(rate * len(scores))
    esc = np.zeros(len(scores), dtype=bool)
    esc[np.argsort(-scores, kind="stable")[:n]] = True
    return esc


def as_arrays(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: df[c].to_numpy(dtype=float) for c in CURVE_COLUMNS}


def curve(df: pd.DataFrame | dict, scores: np.ndarray, post: bool = False) -> list[dict]:
    """outcome(df, escalate_top(scores, r), post) for each rate in RATES, from cumulative sums."""
    n = len(scores)
    order = np.argsort(-np.asarray(scores), kind="stable")
    out = []
    totals, gains = {}, {}
    for m in ("f1", "em", "cost"):
        small, large = np.asarray(df[f"small_{m}"], dtype=float), np.asarray(df[f"large_{m}"], dtype=float)
        delta = large if post and m == "cost" else large - small  # post: the small expert is always paid
        totals[m] = small.sum()
        gains[m] = np.concatenate([[0.0], np.cumsum(delta[order])])
    for r in RATES:
        k = round(r * n)
        out.append({"rate": k / n, **{m: float((totals[m] + gains[m][k]) / n) for m in ("cost", "f1", "em")}})
    return out


def random_curve(df: pd.DataFrame | dict) -> list[dict]:
    """Expected value of escalating a random fraction: a linear mix of the two experts."""
    n = len(df["small_f1"])
    s, l = outcome(df, np.zeros(n), False), outcome(df, np.ones(n), False)
    return [{k: (1 - r) * s[k] + r * l[k] for k in s} | {"rate": float(r)} for r in RATES]


def policy_curves(a: dict[str, np.ndarray], scores: dict[str, tuple[np.ndarray, bool]]) -> dict[str, list[dict]]:
    """Curves of the baselines and of each scored policy (name -> (scores, post))."""
    n = len(a["small_f1"])
    oracle = a["large_f1"] > a["small_f1"]
    curves = {
        "random": random_curve(a),
        "oracle": [outcome(a, np.zeros(n), False), outcome(a, oracle, False), outcome(a, np.ones(n), False)],
        "small-entropy (post)": curve(a, a["small_uncertainty"], post=True),
    }
    for name, (s, post) in scores.items():
        curves[name] = curve(a, s, post)
    return curves


def bootstrap(a: dict[str, np.ndarray], scores: dict[str, tuple[np.ndarray, bool]],
              operating: dict[str, tuple[np.ndarray, bool]], n_boot: int, seed: int,
              references: tuple[str, ...] = (), policy_costs: dict[str, np.ndarray] | None = None) -> dict:
    """Paired bootstrap over questions: AIQ intervals, AIQ differences, operating-point F1 vs always-large.

    `policy_costs` (policy -> its own per-question cost, e.g. a judge's) adds "aiq_diff_with_cost": the same
    differences with each such policy's curve shifted by its mean cost on the resampled questions (as
    `aiq_with_judge_cost`); policies without a cost enter unchanged.
    """
    rng = np.random.default_rng(seed)
    n = len(a["small_f1"])
    policy_costs = policy_costs or {}
    aiqs: dict[str, list[float]] = {}
    aiqs_cost: dict[str, list[float]] = {}
    op_gaps: dict[str, list[float]] = {name: [] for name in operating}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = {k: v[idx] for k, v in a.items()}
        curves = policy_curves(s, {name: (sc[idx], post) for name, (sc, post) in scores.items()})
        cmin, cmax = s["small_cost"].mean(), s["large_cost"].mean()
        for name, pts in curves.items():
            aiqs.setdefault(name, []).append(aiq(pts, cmin, cmax))
            extra = float(policy_costs[name][idx].mean()) if name in policy_costs else 0.0
            aiqs_cost.setdefault(name, []).append(
                aiq([{**p, "cost": p["cost"] + extra} for p in pts], cmin, cmax) if extra else aiqs[name][-1])
        for name, (esc, post) in operating.items():
            op_gaps[name].append(outcome(s, esc[idx], post)["f1"] - s["large_f1"].mean())

    def summary(values: list[float]) -> dict:
        v = np.asarray(values)
        return {"ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "p_le_0": float((v <= 0).mean())}

    aiqs_np = {name: np.asarray(v) for name, v in aiqs.items()}
    diffs = {f"{name} - {base}": summary(aiqs_np[name] - aiqs_np[base])
             for name in scores for base in (*BASELINES, *references) if base != name and base in aiqs_np}
    out = {"n": n_boot, "seed": seed, "aiq_ci95": {name: summary(v)["ci95"] for name, v in aiqs_np.items()},
           "aiq_diff": diffs, "op_f1_minus_large": {name: summary(v) for name, v in op_gaps.items()}}
    if policy_costs:
        cost_np = {name: np.asarray(v) for name, v in aiqs_cost.items()}
        out["aiq_ci95_with_cost"] = {name: summary(cost_np[name])["ci95"] for name in policy_costs if name in cost_np}
        out["aiq_diff_with_cost"] = {f"{name} - {base}": summary(cost_np[name] - cost_np[base])
                                     for name in scores for base in (*BASELINES, *references)
                                     if base != name and base in cost_np}
    return out


def aiq(points: list[dict], cmin: float, cmax: float) -> float:
    """RouterBench AIQ: mean F1 under the non-decreasing upper convex hull over [cmin, cmax]."""
    best: dict[float, float] = {}
    for p in points:  # keep the best F1 at each cost, so the hull has strictly increasing costs
        best[p["cost"]] = max(best.get(p["cost"], -np.inf), p["f1"])
    pts = sorted(best.items())
    hull: list[tuple[float, float]] = []
    for p in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(p)
    xs = np.array([h[0] for h in hull])
    ys = np.maximum.accumulate(np.array([h[1] for h in hull]))
    grid = np.linspace(cmin, cmax, 201)
    return float(np.interp(grid, xs, ys).mean())


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    return float(sum(
        abs(probs[idx == b].mean() - labels[idx == b].mean()) * (idx == b).mean()
        for b in range(bins) if (idx == b).any()
    ))


def _logit(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Temperature T minimising the NLL of sigmoid(logit(p) / T) on labels."""
    z, y = _logit(probs), labels.astype(float)

    def nll(log_t: float) -> float:
        q = np.clip(1 / (1 + np.exp(-z / np.exp(log_t))), 1e-12, 1 - 1e-12)
        return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-4, 4), method="bounded").x))


def scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    return 1 / (1 + np.exp(-_logit(probs) / temperature))


def reliability(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict]:
    """Per-bin mean predicted probability vs observed frequency: the data of a reliability diagram."""
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    return [{"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "count": int((idx == b).sum()),
             "mean_prob": float(probs[idx == b].mean()), "frac_positive": float(labels[idx == b].mean())}
            for b in range(bins) if (idx == b).any()]


def cpt(points: list[dict], f1_small: float, f1_large: float) -> dict:
    """CPT(x) (Ong et al., 2024): cheapest curve point recovering fraction x of the small-to-large F1 gap."""
    out = {}
    for frac in CPT_FRACTIONS:
        ok = [p for p in points if p["f1"] >= f1_small + frac * (f1_large - f1_small)]
        best = min(ok, key=lambda p: p["cost"]) if ok else None
        out[f"{round(frac * 100)}%"] = {"rate": best["rate"], "cost": best["cost"]} if best else None
    return out


def pick_threshold(calib: pd.DataFrame, scores: np.ndarray, max_drop: float) -> float:
    """Highest score threshold whose calib F1 is within max_drop (relative) of always-large."""
    target = (1 - max_drop) * calib["large_f1"].mean()
    for t in sorted(set(scores), reverse=True) + [-np.inf]:
        if outcome(calib, scores >= t, False)["f1"] >= target:  # F1 does not depend on post
            return float(t)
    return -np.inf


def answer_kind(df: pd.DataFrame) -> pd.Series:
    """yes/no vs span gold answers: for breakdowns only, never a router input."""
    return df["answer"].map(normalize_answer).isin({"yes", "no"}).map({True: "yes/no", False: "span"})


def by_group(df: pd.DataFrame, escalate: np.ndarray, groups: pd.Series, post: bool = False) -> dict:
    return {g: outcome(df[m], escalate[m.to_numpy()], post) for g, m in ((g, groups == g) for g in sorted(groups.unique()))}


def by_type(df: pd.DataFrame, escalate: np.ndarray, post: bool = False) -> dict:
    return by_group(df, escalate, df["type"], post)


def breakdowns(df: pd.DataFrame, escalate: np.ndarray, post: bool = False) -> dict:
    return {"by_type": by_type(df, escalate, post), "by_answer_kind": by_group(df, escalate, answer_kind(df), post)}


def evaluate(test: pd.DataFrame, calib: pd.DataFrame | None, routers: dict, tau: float, max_drop: float,
             n_boot: int = 0, seed: int = 0, references: tuple[str, ...] = (),
             policy_costs: dict[str, np.ndarray] | None = None) -> tuple[dict, pd.DataFrame]:
    n = len(test)
    oracle = (test["large_f1"] > test["small_f1"]).to_numpy()
    cmin, cmax = test["small_cost"].mean(), test["large_cost"].mean()
    a = as_arrays(test)
    scores_by_policy, operating = {}, {}
    report = {
        "n_test": n,
        "tau": tau,
        "escalate_label_rate": {label: float(escalation_label(test, tau, label).mean()) for label in LABELS},
        "gold_pair_in_top2": float(test["gold_pair_in_top2"].mean()),
        "always_small": outcome(test, np.zeros(n), False),
        "always_large": outcome(test, np.ones(n), False),
        "oracle": outcome(test, oracle, False),
        "routers": {},
    }
    for name, router in routers.items():
        scores = predict_escalation(router, test)
        post = router.get("post", False)  # cascade router: decides after the small expert
        scores_by_policy[f"router:{name}"] = (scores, post)
        label = router.get("label", "small-fails")  # routers.pkl from before labels were named
        labels = escalation_label(test, tau, label)
        entry = {"label": label, "post": post, "tuning": router.get("tuning"), "ece": ece(scores, labels),
                 "reliability": {"raw": reliability(scores, labels)}}
        if calib is not None:
            calib_scores = predict_escalation(router, calib)
            t = pick_threshold(calib, calib_scores, max_drop)
            esc = scores >= t
            operating[f"router:{name}"] = (esc, post)
            entry["operating_point"] = {"threshold": t, **outcome(test, esc, post), **breakdowns(test, esc, post)}
            temperature = fit_temperature(calib_scores, escalation_label(calib, tau, label))
            scaled = scale(scores, temperature)
            entry |= {"temperature": temperature, "ece_scaled": ece(scaled, labels)}
            entry["reliability"]["scaled"] = reliability(scaled, labels)
        report["routers"][name] = entry
    curves = policy_curves(a, scores_by_policy)
    report["aiq"] = {name: aiq(pts, cmin, cmax) for name, pts in curves.items()}
    f1_small, f1_large = report["always_small"]["f1"], report["always_large"]["f1"]
    report["cpt"] = {name: cpt(pts, f1_small, f1_large) for name, pts in curves.items()}
    for single, esc in (("always_small", np.zeros(n, bool)), ("always_large", np.ones(n, bool))):
        for key, value in breakdowns(test, esc).items():
            report.setdefault(key, {})[single] = value
    if n_boot:
        report["bootstrap"] = bootstrap(a, scores_by_policy, operating, n_boot, seed, references, policy_costs)

    rows = [{"policy": name, **p} for name, pts in curves.items() for p in pts]
    return report, pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-drop", type=float, default=0.01, help="allowed relative F1 drop vs always-large at the operating point")
    parser.add_argument("--cost", choices=COST_UNITS, default="flops", help="cost unit (see module docstring)")
    parser.add_argument("--metric", choices=METRICS, default="f1",
                        help="per-question score for labels, curves and AIQ: F1, or the standard lenient ruler (0/1)")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="file with test question ids to drop (e.g. ones nano-jev was tuned on); repeatable. "
                             "Writes report-clean.json and curves-clean.csv")
    parser.add_argument("--bootstrap", type=int, default=1000, help="paired bootstrap resamples (0 to skip)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    saved = pickle.loads((args.run / "routers.pkl").read_bytes())
    cfg = tomllib.loads((args.run / "config.toml").read_text())
    params, models = expert_params(cfg), {e["name"]: e.get("model", "") for e in cfg["experts"]}
    test = metric_view(pd.read_parquet(args.run / "test.parquet"), args.metric)
    if saved.get("metric", "f1") != args.metric:
        raise ValueError(f"routers.pkl was trained with metric {saved.get('metric', 'f1')!r}; rerun train-router --metric {args.metric}")
    excluded = set()
    for path in args.exclude:
        excluded |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if excluded:
        before = len(test)
        test = test[~test["id"].isin(excluded)].reset_index(drop=True)
        log.info(f"excluded {before - len(test)} of {before} test questions")
    calib_path = args.run / "calib.parquet"
    calib = metric_view(pd.read_parquet(calib_path), args.metric) if calib_path.exists() else None

    single = {}
    for unit in COST_UNITS:
        try:
            t = with_cost_unit(test, unit, params, models)
        except ValueError:
            continue
        single[unit] = {"small": float(t["small_cost"].mean()), "large": float(t["large_cost"].mean())}
    scorer_cost = None
    if "scorer_tokens" in test:
        tokens = float(test["scorer_tokens"].mean())
        scorer_params = cfg.get("scorer", {}).get("params")
        scorer_cost = {"tokens": tokens, "flops": 2 * scorer_params * tokens if scorer_params else None,
                       "truncated_question_rate": float((test["scorer_truncated"] > 0).mean())}
    test = with_cost_unit(test, args.cost, params, models)
    calib = with_cost_unit(calib, args.cost, params, models) if calib is not None else None

    report, curves = evaluate(test, calib, saved["routers"], saved["tau"], args.max_drop, args.bootstrap, args.seed)
    report = {"metric": args.metric, "cost_unit": args.cost, "expert_params_b": params, "excluded_files": [str(p) for p in args.exclude],
              **report, "single_expert_costs": single, "scorer_cost": scorer_cost}
    suffix = "-clean" if excluded else ""
    (args.run / f"report{suffix}.json").write_text(json.dumps(report, indent=1))
    curves.to_csv(args.run / f"curves{suffix}.csv", index=False)
    provenance.record(args.run, "eval-routing", report=f"report{suffix}.json", **provenance.environment())

    rates = " | ".join(f"{k} {v:.3f}" for k, v in report["escalate_label_rate"].items())
    log.info(f"test: {report['n_test']} questions | escalate label rate {rates} | gold pair in top-2 {report['gold_pair_in_top2']:.3f}")
    log.info(f"cost unit: {args.cost}")
    for name in ("always_small", "always_large", "oracle"):
        p = report[name]
        log.info(f"  {name:<28} F1 {p['f1']:.3f} EM {p['em']:.3f} cost {p['cost']:.0f} rate {p['rate']:.2f}")
    ci = report.get("bootstrap", {}).get("aiq_ci95", {})
    for name, value in report["aiq"].items():
        interval = f" [{ci[name][0]:.4f}, {ci[name][1]:.4f}]" if name in ci else ""
        log.info(f"  AIQ {name:<40} {value:.4f}{interval}")
    for name, d in report.get("bootstrap", {}).get("aiq_diff", {}).items():
        log.info(f"  AIQ {name:<64} 95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] p(<=0) {d['p_le_0']:.3f}")
    for name, entry in report["routers"].items():
        op = entry.get("operating_point")
        if op:
            log.info(f"  op {name:<40} F1 {op['f1']:.3f} cost {op['cost']:.0f} rate {op['rate']:.2f} | "
                     f"ECE {entry['ece']:.3f} -> {entry['ece_scaled']:.3f} at T={entry['temperature']:.2f}")
    log.info(f"Saved {args.run / f'report{suffix}.json'} and {args.run / f'curves{suffix}.csv'}")


if __name__ == "__main__":
    main()
