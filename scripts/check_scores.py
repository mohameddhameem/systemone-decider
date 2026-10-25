"""Check that the published folders reproduce the scores saved in training. Every repo push_hf.py staged
(build/hf/<repo>/, seeds 1-2 in seed1/ and seed2/) is loaded as a user loads it, with route.PairRouter or
route.MenuRouter.from_pretrained, and scores every test item. Each score must match the one saved when the judge was
trained (A-test.parquet or test.parquet in models/) to 1e-4 on a GPU; the saved scores are bf16 on a GPU, so a CPU
run is held to 1e-2. The check also counts the routing decisions that flip at the published thresholds.

    uv run python scripts/push_hf.py --org <org>          # stage first
    uv run python scripts/check_scores.py --org <org> [--only pair|menu] [--device cuda|cpu]

The pair needs models/pair/ and the HotpotQA parquet files (its passages are rebuilt from the dataset, as in training);
the menu needs models/menu/ and the SLR files (`slr fetch`). Results go to results/score_check.json.
"""

import argparse
import gc
import json
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import push_hf as release  # noqa: E402  (the repo names and the seed folders the release stages)

from systemone_decider import menu, slr  # noqa: E402
from systemone_decider.judge import Splits  # noqa: E402
from systemone_decider.route import MenuRouter, PairRouter  # noqa: E402
from systemone_decider.items import answer_item, retrieval_item  # noqa: E402

ROOT = release.ROOT
TOLERANCE = {"cuda": 1e-4, "cpu": 1e-2}
SPOT = 8  # questions also scored one call at a time through the menu router's own methods


def free(router) -> None:
    """Release a judge's GPU memory before the next seed loads."""
    import torch

    del router.judge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def folders(org: str, setting: str, size: str, inputs: str):
    stage = ROOT / "build" / "hf" / release.repo_name(org, setting, size, inputs).replace("/", "__")
    if not (stage / "systemone.json").exists():
        raise SystemExit(f"{stage}: not staged; run push_hf.py --org {org} first")
    for seed in (0, 1, 2):
        yield seed, str(stage), (None if seed == 0 else f"seed{seed}")


def check_pair(org: str, data_dir: Path, device: str | None) -> list[dict]:
    plan = tomllib.loads((ROOT / "configs" / "pair.toml").read_text(encoding="utf-8"))
    test = Splits({"A": ROOT / "models" / "pair" / "run"}, data_dir, plan.get("split_data")).get("A", "test")
    rows = [{"question": q, "passages": list(p), "answer": a}
            for q, p, a in zip(test["question"], test["passages"], test["small_pred"])]
    out = []
    for size in release.SIZES:
        for inputs in release.INPUTS:
            for seed, path, sub in folders(org, "pair", size, inputs):
                router = PairRouter.from_pretrained(path, subfolder=sub, device=device)
                ours = pd.DataFrame({"id": test["id"].to_numpy(), "ours": router.p_large_batch(rows)})
                saved = pd.read_parquet(release.seed_folder("pair", size, inputs, seed) / "A-test.parquet")
                m = ours.merge(saved, on="id", validate="1:1")
                flips = int(((m["ours"] >= router.threshold) != (m["score"] >= router.threshold)).sum())
                out.append({"setting": "pair", "size": size, "inputs": inputs, "seed": seed, "items": len(m),
                            "max_abs_diff": float((m["ours"] - m["score"]).abs().max()), "decisions_flipped": flips,
                            "device": router.judge.device})
                print(json.dumps(out[-1]), flush=True)
                free(router)
    return out


def check_menu(org: str, device: str | None) -> list[dict]:
    plan = tomllib.loads((ROOT / "configs" / "menu.toml").read_text(encoding="utf-8"))
    df = menu.load(plan)
    test = df[df["split"] == "test"].reset_index(drop=True)
    q = test["question"].tolist()
    built = {"ret": [retrieval_item(a, s) for a, s in zip(q, test["sel_W"])]}
    out = []
    for size in release.SIZES:
        for inputs in release.INPUTS:
            for x in "WS":  # the weak answer under retrieval x is path xW's
                built[f"qa_{x}"] = [answer_item(a, s, inputs, w) for a, s, w in
                                    zip(q, test[f"sel_{x}"], test[f"answer_{x}W"])]
            for seed, path, sub in folders(org, "menu", size, inputs):
                router = MenuRouter.from_pretrained(path, subfolder=sub, device=device)
                saved = pd.read_parquet(release.seed_folder("menu", size, inputs, seed) / "test.parquet")
                saved = saved.set_index("idx").loc[test["idx"]]
                ours = {k: router.judge.probabilities(its)[0][:, 0] for k, its in built.items()}  # P(yes)
                diff = max(float(np.abs(ours[k] - saved[k].to_numpy()).max()) for k in ours)
                spot = [router.p_weak_retrieval(q[i], test["sel_W"][i]) - saved["ret"].iloc[i] for i in range(SPOT)]
                spot += [router.p_weak_answer(q[i], test["sel_S"][i], test["answer_SW"][i]) - saved["qa_S"].iloc[i]
                         for i in range(SPOT)]
                t = router.thresholds
                route = lambda s: slr.route(s["ret"], {"W": s["qa_W"], "S": s["qa_S"]}, t)  # noqa: E731
                flips = int((route(ours) != route({k: saved[k].to_numpy() for k in ours})).sum())
                out.append({"setting": "menu", "size": size, "inputs": inputs, "seed": seed, "items": 3 * len(test),
                            "max_abs_diff": diff, "decisions_flipped": flips, "thresholds": t,
                            "single_call_max_abs_diff": float(np.abs(spot).max()), "device": router.judge.device})
                print(json.dumps(out[-1]), flush=True)
                free(router)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", required=True)
    p.add_argument("--only", choices=list(release.SETTINGS))
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "hotpotqa")
    p.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = p.parse_args()

    runs = []
    if args.only in (None, "pair"):
        runs += check_pair(args.org, args.data_dir, args.device)
    if args.only in (None, "menu"):
        runs += check_menu(args.org, args.device)
    device = runs[0]["device"].split(":")[0]
    tol, worst = TOLERANCE[device], max(r["max_abs_diff"] for r in runs)
    summary = {"device": device, "tolerance": tol, "max_abs_diff": worst,
               "max_decisions_flipped": max(r["decisions_flipped"] for r in runs), "runs": runs}
    (ROOT / "results" / "score_check.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if worst >= tol:
        sys.exit(f"score check failed on {device}: max diff {worst:.2e} >= {tol:g}")
    print(f"score check passed on {device}: max diff {worst:.2e} < {tol:g} over {len(runs)} folders; "
          f"at most {summary['max_decisions_flipped']} decisions flip")


if __name__ == "__main__":
    main()
