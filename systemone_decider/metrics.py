"""Answer EM/F1 with the normalisation and yes/no rule of the official hotpot_evaluate_v1.py."""

import re
import string
from collections import Counter

_NO_PARTIAL_CREDIT = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in _NO_PARTIAL_CREDIT or ref in _NO_PARTIAL_CREDIT) and pred != ref:
        return 0.0
    pred_tokens, ref_tokens = pred.split(), ref.split()
    num_same = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


METRICS = ("f1", "lenient")


def text(answer) -> str:
    """A cached answer as a string: a missing one (None, NaN, pd.NA) is the empty answer, never "None" or "nan"."""
    import pandas as pd

    return "" if answer is None or (not isinstance(answer, str) and pd.isna(answer)) else str(answer)


def metric_view(df, metric: str):
    """df with `small_f1` / `large_f1` holding the chosen per-question metric, so every label, curve, AIQ and
    threshold downstream uses it. "lenient" is the standard ruler (stair_common.score.lenient, 0/1), kept in
    `small_lenient` / `large_lenient` too; reports must record the metric, as "f1" then means lenient accuracy."""
    if metric == "f1":
        return df
    if metric != "lenient":
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")
    from stair_common.score import lenient

    df = df.copy()
    for name in ("small", "large"):
        if f"{name}_lenient" not in df:
            df[f"{name}_lenient"] = [float(lenient(text(p), [str(g)])) for p, g in zip(df[f"{name}_pred"], df["answer"])]
        df[f"{name}_f1"] = df[f"{name}_lenient"].astype(float)
    return df
