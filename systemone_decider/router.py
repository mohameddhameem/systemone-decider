"""Escalation routers: predict whether a question should go to the large expert.

Routers with the same label share the model (logistic regression), training data and tuning
procedure and differ only in input: `question` sees question text and lexical cues;
`question+evidence` also sees the paragraph-score features computed before any expert runs;
`evidence` sees only those features (an ablation). The regularisation strength C is chosen for
each router from the same grid by stratified 5-fold cross-validated log-loss, so no input set
wins by being better tuned.

The `+small` input sets are cascade routers: they also see the small expert's answer signals
(token entropy, length, whether it answered yes/no), so they decide after the small expert has
run and an escalated question pays for both experts. They compete with the small-entropy
baseline, not with the pre-generation routers.

Two labels are trained, because which one is right is part of the experiment:
`small-fails` (small F1 < tau) also escalates questions the large expert gets wrong too, paying
for nothing; `large-helps` (small F1 < tau and large F1 >= tau) escalates only when it pays off.
"""

import argparse
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from systemone_decider import provenance
from systemone_decider.evidence import tokenize
AUX_VERBS = frozenset("is are was were do does did can could has have had will would".split())
from systemone_decider.metrics import METRICS, metric_view, normalize_answer

log = logging.getLogger(__name__)

CUE_COLUMNS = ["q_starts_aux", "q_has_or", "q_comparative", "q_num_tokens", "q_num_capitalised"]
EVIDENCE_COLUMNS = ["ev_top1", "ev_top2_sum", "ev_margin_2_3", "ev_entropy", "ev_sufficiency"]
SMALL_COLUMNS = ["small_uncertainty", "small_entropy_max", "small_entropy_first", "small_generated_tokens",
                 "small_says_yesno"]
# input set -> (dense columns, whether the question text is used, whether it needs the small expert's answer)
ROUTER_INPUTS = {
    "question": (CUE_COLUMNS, True, False),
    "question+evidence": (CUE_COLUMNS + EVIDENCE_COLUMNS, True, False),
    "evidence": (EVIDENCE_COLUMNS, False, False),
    "evidence+small": (EVIDENCE_COLUMNS + SMALL_COLUMNS, False, True),
    "question+evidence+small": (CUE_COLUMNS + EVIDENCE_COLUMNS + SMALL_COLUMNS, True, True),
}
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
CV_FOLDS = 5
# Parallel CV workers. Each holds a copy of the data, so one per core on a 20-core machine ran out of
# memory next to a loaded judge; 8 is still fast. Results do not depend on it. Override with ROUTER_JOBS.
CV_JOBS = int(os.environ.get("ROUTER_JOBS", min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)))
COMPARATIVE = frozenset("both same first older younger earlier later more larger longer higher".split())


def add_question_cues(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    toks = df["question"].map(tokenize)
    df["q_starts_aux"] = toks.map(lambda t: float(bool(t) and t[0] in AUX_VERBS))
    df["q_has_or"] = toks.map(lambda t: float("or" in t))
    df["q_comparative"] = toks.map(lambda t: float(bool(COMPARATIVE & set(t))))
    df["q_num_tokens"] = toks.map(len).astype(float)
    df["q_num_capitalised"] = df["question"].str.count(r"\b[A-Z]").astype(float)
    return df


def answer_columns(prefix: str) -> list[str]:
    return [c.replace("small_", f"{prefix}_", 1) for c in SMALL_COLUMNS]


def add_answer_cues(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Signals from expert `prefix`'s own answer (NaN when the cache has no such expert)."""
    df = df.copy()
    if f"{prefix}_token_entropy" not in df:
        for c in answer_columns(prefix):
            df[c] = df.get(c, np.nan)
        return df
    steps = df[f"{prefix}_token_entropy"].map(lambda e: np.asarray(e, dtype=float))
    df[f"{prefix}_entropy_max"] = steps.map(lambda e: e.max() if e.size else np.nan)
    df[f"{prefix}_entropy_first"] = steps.map(lambda e: e[0] if e.size else np.nan)
    df[f"{prefix}_generated_tokens"] = df[f"{prefix}_generated_tokens"].astype(float)
    df[f"{prefix}_says_yesno"] = df[f"{prefix}_pred"].map(normalize_answer).isin({"yes", "no"}).astype(float)
    df[f"{prefix}_uncertainty"] = df[f"{prefix}_uncertainty"].replace([np.inf, -np.inf], np.nan)  # no tokens generated
    return df


def add_small_cues(df: pd.DataFrame) -> pd.DataFrame:
    """Signals from the small expert's own answer (NaN when the cache has no small expert)."""
    return add_answer_cues(df, "small")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    return add_small_cues(add_question_cues(df))


LABELS = ("small-fails", "large-helps")


def escalation_label(df: pd.DataFrame, tau: float, label: str) -> np.ndarray:
    """1 if the question should be escalated under `label` (see module docstring)."""
    small_wrong = df["small_f1"] < tau
    if label == "small-fails":
        return small_wrong.astype(int).to_numpy()
    if label == "large-helps":
        return (small_wrong & (df["large_f1"] >= tau)).astype(int).to_numpy()
    raise ValueError(f"unknown label: {label}")


def usable_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Drop columns that are entirely missing (e.g. sufficiency with the lexical scorer)."""
    return [c for c in columns if df[c].notna().any()]


def build_router(dense_columns: list[str], text: bool = True, C: float = 1.0) -> Pipeline:
    parts = [("dense", StandardScaler(), dense_columns)] if dense_columns else []
    if text:
        parts.insert(0, ("text", HashingVectorizer(ngram_range=(1, 2), n_features=2**18, alternate_sign=False,
                                                   norm="l2"), "question"))
    return Pipeline([("features", ColumnTransformer(parts)), ("clf", LogisticRegression(C=C, max_iter=2000))])


def fit_tuned(model: Pipeline, X: pd.DataFrame, y: np.ndarray, seed: int) -> tuple[Pipeline, dict]:
    """Fit with C chosen from C_GRID by stratified CV log-loss (C=1 when a class is too rare to fold)."""
    folds = min(CV_FOLDS, int(np.bincount(y).min()))
    if folds < 2:
        return model.fit(X, y), {"C": model.get_params()["clf__C"], "cv_folds": 0}
    search = GridSearchCV(model, {"clf__C": list(C_GRID)}, scoring="neg_log_loss",
                          cv=StratifiedKFold(folds, shuffle=True, random_state=seed), n_jobs=CV_JOBS)
    search.fit(X, y)
    return search.best_estimator_, {"C": search.best_params_["clf__C"], "cv_folds": folds,
                                    "cv_log_loss": -float(search.best_score_)}


def predict_escalation(router: dict, df: pd.DataFrame) -> np.ndarray:
    """The router's P(escalate) for each row, under its own label (see escalation_label).

    A router may instead name a precomputed `score_column` (e.g. a System One judge scored on GPU).
    """
    if "score_column" in router:
        return df[router["score_column"]].to_numpy(dtype=float)
    X = add_features(df)
    X[router["columns"]] = X[router["columns"]].fillna(0.0)
    return router["model"].predict_proba(X)[:, 1]


def train_routers(df: pd.DataFrame, tau: float, seed: int = 0) -> dict[str, dict]:
    """One router per (input set, label), named "<inputs>/<label>"."""
    X = add_features(df)
    routers = {}
    for label in LABELS:
        y = escalation_label(df, tau, label)
        if len(set(y)) < 2:
            log.warning(f"  skipping label {label}: all {len(y)} training labels are {y[0]}")
            continue
        for inputs, (columns, text, post) in ROUTER_INPUTS.items():
            cols = usable_columns(X, columns)
            if (not cols and not text) or (post and not set(cols) & set(SMALL_COLUMNS)):
                log.warning(f"  skipping {inputs}/{label}: no usable features")
                continue
            X[cols] = X[cols].fillna(0.0)
            model, tuning = fit_tuned(build_router(cols, text), X, y, seed)
            routers[f"{inputs}/{label}"] = {"model": model, "columns": cols, "label": label, "tuning": tuning,
                                            "post": post}
            log.info(f"  {inputs}/{label}: {len(cols)} dense features {cols} | {tuning}")
    if not routers:
        raise ValueError(f"no label has both classes on {len(df)} questions; change tau or add questions")
    return routers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.8, help="small expert counts as correct at F1 >= tau")
    parser.add_argument("--seed", type=int, default=0, help="seed for the cross-validation folds")
    parser.add_argument("--metric", choices=METRICS, default="f1",
                        help="score the labels use: F1, or the standard lenient ruler (0/1; use --tau 0.5)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = metric_view(pd.read_parquet(args.run / "router_train.parquet"), args.metric)
    rates = " | ".join(f"{label} {escalation_label(df, args.tau, label).mean():.3f}" for label in LABELS)
    log.info(f"router_train: {len(df)} questions, escalate rate at tau={args.tau}: {rates}")
    routers = train_routers(df, args.tau, args.seed)
    out = args.run / "routers.pkl"
    meta = {"n_train": len(df), "seed": args.seed, "metric": args.metric, "c_grid": C_GRID,
            "tuning": {name: r["tuning"] for name, r in routers.items()}}
    out.write_bytes(pickle.dumps({"tau": args.tau, "routers": routers, **meta}))
    provenance.record(args.run, "train-router", tau=args.tau, **meta, **provenance.environment())
    log.info(f"Saved {out}")


if __name__ == "__main__":
    main()
