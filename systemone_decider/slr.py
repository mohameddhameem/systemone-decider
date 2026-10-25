"""The Stage-Level Routing (SLR) Config B grid on HotpotQA, read at a pinned commit of the STAIR repo.

Config B is AgentTTS's menu A on all 7,405 dev questions. Retrieval: Qwen2.5-7B (weak) or Qwen2.5-72B-AWQ (strong)
selects passages from the 100-passage pool. Answering: Llama-3.2-3B (weak) or Llama-3.1-70B-AWQ (strong), under each
retrieval. So every question has four paths, named retrieval then answer: WW, WS, SW, SS (the SLR `em_standard` order).

The SLR setup (stair_common 0.3.0) fixes the slr-train / slr-val / slr-test split, the standard ruler and AgentTTS
Table 5 prices. A fixed policy runs only its path (always-SS = 929, strong-everything). A router runs weak-first, so an
escalated stage also pays for the weak call before it (stair_common.cost.SLR_PATH_PRICES); the router itself is not
charged. "His" below refers to the SLR setup's author, whose files these are.

    slr fetch   download the pinned files into `out`, checking sha256 (needs internet, no compute)
    slr verify  the ruler (stair_common's lenient gives his verdict on every question and path), the split (his
                splits.json, the grid and stair_common's slr-* parts agree), and his certified rows: thresholds
                re-selected on slr-val with his rule, then replayed on slr-test from his test scores
    slr ladder  verify, then always-* paths and the oracle on slr-val, slr-test and the clean test (without nano-jev's
                seen questions), at Table 5 prices, with measured AgentTTS units beside them: results/slr/report.json
"""

import argparse
import hashlib
import json
import logging
import tomllib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from stair_common import load_models
from stair_common.cost import SLR_PATH_PRICES, STRONG_EVERYTHING, TABLE5_HOTPOTQA, units
from stair_common.score import lenient
from stair_common.splits import load_split

from systemone_decider import bootstrap as ev
from systemone_decider import provenance

log = logging.getLogger(__name__)

PATHS = ["WW", "WS", "SW", "SS"]  # retrieval then answer: his em_standard order
RETRIEVER = {"W": "Qwen2.5-7B", "S": "Qwen2.5-72B"}
ANSWERER = {"W": "Llama-3.2-3B", "S": "Llama-3.1-70B"}
BRANCH = {"W": "weak", "S": "strong"}
GRID = "jonathan/runs/config_b"
SPLITS = "jonathan/lab_package/splits.json"
THRESHOLDS = [i / 20 for i in range(21)]  # his sweep grid, per stage: 21^3 = 9,261 points


def direct_price(path: str) -> float:
    """A fixed policy runs one retrieval and one answer, at Table 5 prices: always-SS is strong-everything (929)."""
    return TABLE5_HOTPOTQA["retrieve"][RETRIEVER[path[0]]] + TABLE5_HOTPOTQA["answer"][ANSWERER[path[1]]]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(plan: dict) -> None:
    """Download each pinned file that is missing or differs, then check every file's sha256."""
    out = Path(plan["out"])
    for name, sha in plan["files"].items():
        target = out / name
        if not (target.exists() and _sha(target) == sha):
            url = f"https://raw.githubusercontent.com/{plan['repo']}/{plan['commit']}/{name}"
            target.parent.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
            log.info(f"fetched {name}")
    check_files(plan)


def check_files(plan: dict) -> None:
    out = Path(plan["out"])
    for name, sha in plan["files"].items():
        if not (out / name).exists():
            raise FileNotFoundError(f"{out / name}: run `slr fetch` first")
        if (got := _sha(out / name)) != sha:
            raise ValueError(f"{name}: sha256 {got[:12]}, expected {sha[:12]} (commit {plan['commit'][:7]})")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _store(path: Path) -> dict:
    return {r["k"]: r for r in read_jsonl(path)}


def load_grid(plan: dict) -> pd.DataFrame:
    """One row per dev question in file order: idx, id, gold, his split, and per path its verdict (ok_), answer, price
    (a fixed policy's, Table 5) and measured AgentTTS units."""
    check_files(plan)
    root = Path(plan["out"]) / GRID
    dev = pd.read_parquet(plan["dev"], columns=["id", "answer"])
    grid = pd.DataFrame(read_jsonl(root / "router_data_standard.jsonl")).sort_values("idx").reset_index(drop=True)
    if grid["idx"].tolist() != list(range(len(dev))):
        raise ValueError(f"the grid has {len(grid)} rows, not one per dev question ({len(dev)})")
    df = pd.DataFrame({"idx": grid["idx"], "id": dev["id"], "gold": dev["answer"].astype(str), "split": grid["split"]})
    em = np.array(grid["em_standard"].tolist(), dtype=bool)
    answers = {a: _store(root / f"qa_{BRANCH[a]}.jsonl") for a in "WS"}
    meta = {(stage, x): _store(root / f"meta_{stage}_{BRANCH[x]}.jsonl") for stage in ("retr", "qa") for x in "WS"}
    menu = load_models()["A"]
    params = {("retr", x): menu["retriever"][RETRIEVER[x]]["params_b"] for x in "WS"}
    params |= {("qa", x): menu["answerer"][ANSWERER[x]]["params_b"] for x in "WS"}

    def call_units(stage: str, x: str, key: str) -> float:
        m = meta[(stage, x)][key]
        return units(params[(stage, x)], m["n_in"], m["n_out"])

    for j, path in enumerate(PATHS):
        r, a = path
        keys = [f"{i}|{BRANCH[r]}" for i in df["idx"]]
        df[f"ok_{path}"] = em[:, j]
        df[f"answer_{path}"] = [str(answers[a][k]["v"]) for k in keys]
        df[f"price_{path}"] = direct_price(path)
        df[f"units_{path}"] = [call_units("retr", r, str(i)) + call_units("qa", a, k) for i, k in zip(df["idx"], keys)]
    return df


def ruler_mismatches(df: pd.DataFrame) -> int:
    """Questions x paths where stair_common's lenient disagrees with his verdict."""
    return sum(int(lenient(answer, [gold]) != ok)
               for path in PATHS for answer, gold, ok in zip(df[f"answer_{path}"], df["gold"], df[f"ok_{path}"]))


def split_mismatches(df: pd.DataFrame, his: dict, parts: dict[str, list[str]]) -> list[str]:
    """Where his splits.json, the grid's split column and the shared slr-* parts disagree."""
    problems = []
    by_idx = {i: name for name, idxs in his["splits"].items() for i in idxs}
    if [by_idx.get(i) for i in df["idx"]] != df["split"].tolist():
        problems.append("the grid's split column differs from splits.json")
    for name, ids in parts.items():
        if set(df.loc[df["split"] == name, "id"]) != set(ids):
            problems.append(f"slr-{name} differs from his {name} part")
    return problems


def route(ret_pw: np.ndarray, qa_pw: dict[str, np.ndarray], t: list[float]) -> np.ndarray:
    """His routers' rule (configb_test_readout.py): keep weak retrieval if P(weak suffices) >= t[0], then keep the weak
    answer under that retrieval if its P >= t[1] (weak retrieval) or t[2] (strong). Returns the path index."""
    keep_ret = ret_pw >= t[0]
    keep_ans = np.where(keep_ret, qa_pw["W"] >= t[1], qa_pw["S"] >= t[2])
    return np.where(keep_ret, np.where(keep_ans, 0, 1), np.where(keep_ans, 2, 3))


def router_point(df: pd.DataFrame, choice: np.ndarray) -> dict:
    """Accuracy (%) and mean weak-first price of a router's path choices."""
    ok = ok_matrix(df)[np.arange(len(df)), choice]
    return {"lenient": float(100 * ok.mean()), "price": float(np.mean([SLR_PATH_PRICES[PATHS[c]] for c in choice]))}


def sweep(ok: np.ndarray, ret_pw: np.ndarray, qa_pw: dict[str, np.ndarray]) -> list[tuple]:
    """His threshold sweep on one split (configb_pair_sweep.py), in his order and arithmetic, so that ties break as his
    did: (accuracy, weak-first price / 929, t_ret, t_weak, t_strong) per point."""
    n, rows = len(ret_pw), np.arange(len(ret_pw))
    prices = np.array([SLR_PATH_PRICES[p] for p in PATHS])
    correct = ok.astype(int)
    points = []
    for tr in THRESHOLDS:
        for tw in THRESHOLDS:
            for ts in THRESHOLDS:
                c = route(ret_pw, qa_pw, [tr, tw, ts])
                points.append((int(correct[rows, c].sum()) / n, int(prices[c].sum()) / n / STRONG_EVERYTHING, tr, tw, ts))
    return points


def select(points: list[tuple], strong_accuracy: float) -> dict[str, list[float] | None]:
    """His two operating points: `peak`, the most accurate (Python's max over the tuples, so ties go to the dearer
    point), and `parity`, the first cheapest point at or above always-strong's accuracy (None if there is none)."""
    beats = [p for p in points if p[0] >= strong_accuracy]
    return {"peak": list(max(points)[2:]), "parity": list(min(beats, key=lambda p: p[1])[2:]) if beats else None}


def his_scores(plan: dict, ret: str, qa: str, idx: pd.Series, split: str = "test") -> tuple[np.ndarray, dict]:
    """His routers' scores, aligned to `idx`: P(weak retrieval suffices), and P(weak answer suffices) per branch. On
    test, `ret` / `qa` name test_scores files; on val, his run folders (their val_scores.jsonl)."""
    root = Path(plan["out"]) / GRID
    path = (lambda name: root / "test_scores" / f"{name}.jsonl") if split == "test" else \
           (lambda name: root / name / "val_scores.jsonl")
    r = {row["idx"]: row["pW"] for row in read_jsonl(path(ret))}
    q = {(row["idx"], row["ret_choice"]): row["pW"] for row in read_jsonl(path(qa))}
    return np.array([r[i] for i in idx]), {x: np.array([q[(i, x)] for i in idx]) for x in "WS"}


def ok_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[[f"ok_{p}" for p in PATHS]].to_numpy()


def certified(plan: dict, df: pd.DataFrame) -> list[dict]:
    """His certified sealed-test rows, recomputed: each with the value he reports and ours. For a router, its thresholds
    are first re-selected on slr-val from his val scores with his rule, and must be the ones he froze."""
    test = df[df["split"] == "test"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    rows = []
    for row in plan["certified"]:
        if "path" in row:
            got = {"lenient": float(100 * test[f"ok_{row['path']}"].mean()), "price": direct_price(row["path"])}
        else:
            ret_pw, qa_pw = his_scores(plan, *row["val"], val["idx"], split="val")
            strong = int(val["ok_SS"].sum()) / len(val)
            got = {"t_from_val": select(sweep(ok_matrix(val), ret_pw, qa_pw), strong)[row["rule"]]}
            ret_pw, qa_pw = his_scores(plan, row["ret"], row["qa"], test["idx"])
            got |= router_point(test, route(ret_pw, qa_pw, row["t"]))
        match = round(got["lenient"], 2) == row["lenient"] and round(got["price"], 1) == row["price"]
        rows.append({"name": row["name"], "reported": {"lenient": row["lenient"], "price": row["price"]}, "ours": got,
                     "match": match and got.get("t_from_val", row.get("t")) == row.get("t")})
    return rows


def verify(plan: dict, df: pd.DataFrame) -> dict:
    his = json.loads((Path(plan["out"]) / SPLITS).read_text(encoding="utf-8"))
    parts = {name: load_split("hotpotqa", f"slr-{name}") for name in ("train", "val", "test")}
    report = {"n": len(df), "ruler_mismatches": ruler_mismatches(df), "split_problems": split_mismatches(df, his, parts),
              "certified": certified(plan, df)}
    for row in report["certified"]:
        t = f" (t re-selected on val: {row['ours']['t_from_val']})" if "t_from_val" in row["ours"] else ""
        log.info(f"{row['name']:28s} his {row['reported']['lenient']:.2f} at {row['reported']['price']:.1f}, "
                 f"ours {row['ours']['lenient']:.2f} at {row['ours']['price']:.1f}{t}  "
                 f"{'ok' if row['match'] else 'MISMATCH'}")
    log.info(f"ruler mismatches: {report['ruler_mismatches']} of {4 * len(df)}; split problems: {report['split_problems']}")
    if report["ruler_mismatches"] or report["split_problems"] or not all(r["match"] for r in report["certified"]):
        raise ValueError("the SLR grid does not verify; see the log above")
    return report


def ladder(df: pd.DataFrame, seen: set[str]) -> dict:
    """Always-* paths and the oracle per split, at Table 5 prices (a fixed policy runs only its path), with the
    measured AgentTTS units of the same calls."""
    report = {}
    for name, mask in (("val", df["split"] == "val"), ("test", df["split"] == "test"),
                       ("test-clean", (df["split"] == "test") & ~df["id"].isin(seen))):
        part = df[mask]
        ok = part[[f"ok_{p}" for p in PATHS]].to_numpy()
        price = part[[f"price_{p}" for p in PATHS]].to_numpy()
        measured = part[[f"units_{p}" for p in PATHS]].to_numpy()
        choice = ev.cheapest_correct_label(ok, price)
        rows = np.arange(len(part))
        report[name] = {
            "n": int(mask.sum()),
            "always": {p: {"lenient": float(ok[:, k].mean()), "price": float(price[:, k].mean()),
                           "units": float(measured[:, k].mean())} for k, p in enumerate(PATHS)},
            "oracle": {"lenient": float(ok[rows, choice].mean()), "price": float(price[rows, choice].mean()),
                       "units": float(measured[rows, choice].mean())},
        }
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["fetch", "verify", "ladder", "all"])
    parser.add_argument("--plan", type=Path, default=Path("configs/slr.toml"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = tomllib.loads(args.plan.read_text(encoding="utf-8"))
    results = Path(plan["results"])
    if args.step in ("fetch", "all"):
        fetch(plan)
    if args.step == "fetch":
        return
    df = load_grid(plan)
    report = {"commit": plan["commit"], "verify": verify(plan, df)}
    if args.step == "verify":
        return
    report["ladder"] = ladder(df, set(load_split("hotpotqa", "nanojev_v1_seen")))
    for name, part in report["ladder"].items():
        row = "  ".join(f"{p} {v['lenient']:.3f}@{v['price']:.0f}" for p, v in part["always"].items())
        log.info(f"{name:10s} n={part['n']:5d}  {row}  oracle {part['oracle']['lenient']:.3f}@{part['oracle']['price']:.0f}")
    results.mkdir(parents=True, exist_ok=True)
    cols = ["idx", "id", "split"] + [f"{c}_{p}" for c in ("ok", "price", "units") for p in PATHS]
    for name in ("val", "test"):
        df.loc[df["split"] == name, cols].to_parquet(results / f"paths-{name}.parquet", index=False)
    (results / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    provenance.record(results, f"slr {args.step}", plan=str(args.plan), commit=plan["commit"], **provenance.environment())


if __name__ == "__main__":
    main()
