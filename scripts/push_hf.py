"""Stage (and, with --push, upload) the decider judges to the Hugging Face Hub.

Eight repos, one per (setting, size, inputs): <org>/systemone-decider-<size>-<inputs> for the pair setting, and
<org>/systemone-decider-<size>-<inputs>-menu-a for the two-stage menu. In each, seed 0 is the root and seeds 1-2 are in
seed1/ and seed2/. Every folder gets systemone.json (setting, inputs, thresholds) and training.json (its training
record). The root gets README.md, rendered from model-cards/<setting>.md with the numbers from results/.

    uv run python scripts/push_hf.py --org <hf-org>                  # dry run: stages into build/hf/<repo>/
    uv run python scripts/push_hf.py --org <hf-org> --cards-only     # only each README.md
    uv run python scripts/push_hf.py --org <hf-org> --push           # upload (needs `hf auth login`)

--only pair|menu restricts the run to one setting; --results <dir> reads results/<setting>/ from another folder. The
cards quote each default threshold on the test set, so even --cards-only needs models/ (the seeds' saved scores and the
pair's run/) and, for the menu, the SLR files (`slr fetch`).
"""

import argparse
import json
import os
import shutil
import tomllib
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

# Set SYSTEMONE_PARQUET_ENGINE=fastparquet where pyarrow's Parquet DLL is blocked (Windows App Control).
if os.environ.get("SYSTEMONE_PARQUET_ENGINE"):
    pd.set_option("io.parquet.engine", os.environ["SYSTEMONE_PARQUET_ENGINE"])

ROOT = Path(__file__).resolve().parents[1]
RELEASE = tomllib.loads((ROOT / "configs" / "release.toml").read_text(encoding="utf-8"))
# lora_rank is the base checkpoint's own LoRA, which fine-tuning keeps; stage_weights checks it on every seed.
# base_revision is the commit training started from (the only one in the cluster's Hub cache on 2026-10-25).
SIZES = {
    "2b": {"name": "2B", "base": "StrandsAgents/strands-decider-2B-hobson-v21", "torso": "Qwen/Qwen3.5-2B-Base",
           "params_b": 2.0, "tag": "2b-hobson-v21", "lora_rank": 16,
           "base_revision": "2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"},
    "e4b": {"name": "E4B", "base": "StrandsAgents/strands-decider-E4B-gemma4-v1-2610", "torso": "google/gemma-4-E4B-it",
            "params_b": 4.0, "tag": "e4b-gemma4-v1-2610", "lora_rank": 48,
            "base_revision": "1344d44397c457f3be34b8ff15c593b41248c6bf"},
}
INPUTS = ("qp", "qpa")
SETTINGS = {"pair": "pair-a", "menu": "menu-a"}  # setting -> its results/ folder
ENTROPY = "small-entropy (post)"
# The matched logistic-regression router at the judge's decision point (the paired intervals are against it), and the
# family it belongs to, whose best member the card also shows.
LR = {"qp": ("router:question+evidence/large-helps", "Logistic-regression router on the question and passages (before)"),
      "qpa": ("router:question+evidence+small/large-helps",
              "Logistic-regression cascade router on the question, passages and small answer (after)")}
LR_FAMILY = {"qp": ["router:question/large-helps", "router:question+evidence/large-helps", "router:evidence/large-helps"],
             "qpa": ["router:evidence+small/large-helps", "router:question+evidence+small/large-helps"]}
HIS = ("SLR SFT (label)", "SLR SFT (label) + reward")


def repo_name(org: str, setting: str, size: str, inputs: str) -> str:
    return f"{org}/systemone-decider-{size}-{inputs}" + ("-menu-a" if setting == "menu" else "")


def pair_judge(size: str, inputs: str) -> str:
    return f"{inputs}-large-helps-strands-decider-{SIZES[size]['tag']}-A"


def menu_router(size: str, inputs: str) -> str:
    return f"decider-{size} {inputs}"


def seed_folder(setting: str, size: str, inputs: str, seed: int) -> Path:
    if setting == "pair":
        return ROOT / "models" / "pair" / pair_judge(size, inputs) / f"seed{seed}"
    return ROOT / "models" / "menu" / f"decider-{size}" / inputs / f"seed{seed}"


# ---------------------------------------------------------------------------------------------
# Thresholds


def text(answer) -> str:
    return "" if answer is None or (not isinstance(answer, str) and pd.isna(answer)) else str(answer)


def lenient_view(df: pd.DataFrame) -> pd.DataFrame:
    """small_f1 / large_f1 holding lenient accuracy (the standard ruler), as the evaluation scores them."""
    from stair_common.score import lenient

    df = df.copy()
    for name in ("small", "large"):
        if f"{name}_lenient" not in df:
            df[f"{name}_lenient"] = [float(lenient(text(p), [str(g)])) for p, g in zip(df[f"{name}_pred"], df["answer"])]
        df[f"{name}_f1"] = df[f"{name}_lenient"].astype(float)
    return df


def pick_threshold(calib: pd.DataFrame, scores: np.ndarray, max_drop: float) -> float:
    """Highest score threshold whose calib accuracy is within max_drop (relative) of always-large's."""
    target = (1 - max_drop) * calib["large_f1"].mean()
    for t in sorted(set(scores), reverse=True) + [-np.inf]:
        if np.where(scores >= t, calib["large_f1"], calib["small_f1"]).mean() >= target:
            return float(t)
    return -np.inf


def pair_threshold(folder: Path, calib: pd.DataFrame) -> float:
    merged = calib.merge(pd.read_parquet(folder / "A-calib.parquet"), on="id", validate="1:1")
    return pick_threshold(merged, merged["score"].to_numpy(), RELEASE["pair"]["max_drop"])


def pair_run() -> dict:
    """The pair's calibration and test questions with both models' answers (lenient), and its expert config."""
    run = ROOT / "models" / "pair" / "run"
    return {"calib": lenient_view(pd.read_parquet(run / "calib.parquet")),
            "test": lenient_view(pd.read_parquet(run / "test.parquet")),
            "config": tomllib.loads((run / "config.toml").read_text(encoding="utf-8"))}


def pair_point(size: str, inputs: str, run: dict) -> dict:
    """Seed 0 at its published threshold on the test set, scored as the evaluation scores it (Table 5 prices; qpa pays
    the small model's call on every question), with always-large and always-small beside it."""
    from systemone_decider.evaluate_routing import expert_params, outcome, with_cost_unit
    from systemone_decider.judge import INPUTS as PAYS_SMALL

    folder, cfg = seed_folder("pair", size, inputs, 0), run["config"]
    t = pair_threshold(folder, run["calib"])
    plan = tomllib.loads((ROOT / "configs" / "pair.toml").read_text(encoding="utf-8"))
    models = {e["name"]: e.get("model", "") for e in cfg["experts"]}
    test = with_cost_unit(run["test"], plan["cost"], expert_params(cfg), models)
    test = test.merge(pd.read_parquet(folder / "A-test.parquet"), on="id", validate="1:1")
    every = np.ones(len(test), dtype=bool)
    return {"t": t, **outcome(test, test["score"].to_numpy() >= t, PAYS_SMALL[inputs]),
            "large": outcome(test, every, False), "small": outcome(test, ~every, False)}


def menu_grid() -> dict:
    """slr-val and slr-test: the question positions and each path's lenient verdict, from the pinned SLR files."""
    from systemone_decider import slr

    plan = tomllib.loads((ROOT / "configs" / "slr.toml").read_text(encoding="utf-8"))
    df = slr.load_grid(plan | {"out": ROOT / plan["out"], "dev": ROOT / plan["dev"]})
    return {s: (part["idx"], slr.ok_matrix(part)) for s in ("val", "test")
            for part in [df[df["split"] == s].reset_index(drop=True)]}


def menu_point(grid: dict, size: str, inputs: str, seed: int, t: list[float]) -> dict:
    """Thresholds `t` on slr-test: accuracy, weak-first price, and the judge's own calls priced as the evaluation does
    (the retrieval call, and the answer call on the branch taken)."""
    from stair_common.cost import budget
    from systemone_decider import slr
    from systemone_decider.menu import PRICES

    ids, ok = grid["test"]
    d = pd.read_parquet(seed_folder("menu", size, inputs, seed) / "test.parquet").set_index("idx").loc[ids]
    c = slr.route(d["ret"].to_numpy(), {"W": d["qa_W"].to_numpy(), "S": d["qa_S"].to_numpy()}, t)
    call = {k: np.array([budget(SIZES[size]["params_b"], int(n), 0) for n in d[f"tokens_{k}"]])
            for k in ("ret", "qa_W", "qa_S")}
    judge = call["ret"] + np.where(c < 2, call["qa_W"], call["qa_S"])
    return {"t": t, "lenient": float(ok[np.arange(len(c)), c].mean()), "price": float(PRICES[c].mean()),
            "judge_price": float(judge.mean())}


def near_strong(grid: dict, size: str, inputs: str, seed: int) -> dict:
    """The default menu thresholds: the cheapest weak-first point of the SLR setup's sweep on slr-val whose accuracy is
    within max_drop (relative) of always-strong's (its `parity` rule at a lower bar), and that point on slr-test."""
    from systemone_decider import slr

    ids, ok = grid["val"]
    d = pd.read_parquet(seed_folder("menu", size, inputs, seed) / "val.parquet").set_index("idx").loc[ids]
    points = slr.sweep(ok, d["ret"].to_numpy(), {"W": d["qa_W"].to_numpy(), "S": d["qa_S"].to_numpy()})
    bar = (1 - RELEASE["menu"]["max_drop"]) * ok[:, slr.PATHS.index("SS")].mean()
    return menu_point(grid, size, inputs, seed, slr.select(points, bar)["parity"])


def menu_thresholds(report: dict, grid: dict, size: str, inputs: str, seed: int) -> dict:
    entry = report["routers"][f"{menu_router(size, inputs)}#seed{seed}"]
    near = near_strong(grid, size, inputs, seed)["t"]
    return {"near_strong": near} | {rule: (entry[rule] or {}).get("t") for rule in ("peak", "parity")}


# ---------------------------------------------------------------------------------------------
# Cards


def num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def ci(c: list[float] | None) -> str:
    return "" if c is None else f"[{c[0]:.3f}, {c[1]:.3f}]"


def diff(mean: float, c: list[float] | None) -> str:
    if c is None:
        return f"{mean:+.3f}"
    word = "better" if c[0] > 0 else "worse" if c[1] < 0 else "not significant"
    return f"{mean:+.3f} (95% CI {c[0]:+.3f} to {c[1]:+.3f}, {word})"


def common_fields(org: str, setting: str, size: str, inputs: str) -> dict:
    s = SIZES[size]
    return {"repo": repo_name(org, setting, size, inputs), "size_name": s["name"], "inputs": inputs, "base": s["base"],
            "torso": s["torso"], "lora_rank": s["lora_rank"], "base_revision": s["base_revision"],
            "github": RELEASE["github"],
            "answer_arg": ", answer=small_answer" if inputs == "qpa" else "",
            "weak_answer_arg": ", weak_answer=weak_answer" if inputs == "qpa" else "",
            **{k: v for k, v in RELEASE[setting]["train"].items()}}


def fill(template: str, common: dict, fields: dict) -> str:
    """The card, refusing a field that would replace a common one (the LR AIQ once replaced the training lr)."""
    if clash := common.keys() & fields.keys():
        raise SystemExit(f"{template}: card fields {sorted(clash)} clash with the common ones")
    return (ROOT / "model-cards" / template).read_text(encoding="utf-8").format(**common, **fields)


def render_pair(org: str, size: str, inputs: str, report: dict, clean: dict, run: dict) -> str:
    name = pair_judge(size, inputs)
    judge, aiq, boot = report["judges"][f"judge/{name}"], report["aiq"], report["bootstrap"]
    key, lr_key = f"router:judge/{name}", LR[inputs][0]
    seed0, seed0_cost = judge["aiq_per_seed"][0], judge["aiq_per_seed_with_cost"][0]
    get = lambda d, k: d.get(k, {}).get("ci95") if isinstance(d.get(k), dict) else d.get(k)  # noqa: E731
    op, per_call = pair_point(size, inputs, run), judge["judge_cost"] / judge["judge_passes"]
    at = lambda o: f"{100 * o['f1']:.2f}% at a cost of {o['cost']:.2f} a question"  # noqa: E731
    fields = {
        "threshold": f"{op['t']:.4f}", "max_drop": f"{RELEASE['pair']['max_drop']:.0%}",
        "op": f"{at(op)}, plus {per_call:.2f} for the judge's call, sending {op['rate']:.1%} of questions to the large "
              f"model",
        "op_large": at(op["large"]), "op_small": at(op["small"]),
        "op_drop": f"{100 * (1 - op['f1'] / op['large']['f1']):.1f}%",
        "op_share": f"{(op['cost'] + per_call) / op['large']['cost']:.0%}",
        "inputs_text": {"qp": "the question and the small model's passages (before the small model answers)",
                        "qpa": "the question, the small model's passages and its answer"}[inputs],
        "n_test": f"{report['n_test']:,}", "n_clean": f"{clean['n_test']:,}",
        "seed0": num(seed0), "seed0_ci": ci(get(boot["aiq_ci95"], f"{key}#seed0")),
        "seed0_cost": num(seed0_cost), "seed0_cost_ci": ci(get(boot["aiq_ci95_with_cost"], f"{key}#seed0")),
        "per_call": f"{per_call:.2f}",
        "seeds": f"{judge['aiq_seed_mean']:.3f} ± {judge['aiq_seed_sd']:.3f}",
        "seeds_cost": f"{judge['aiq_seed_mean_with_cost']:.3f}",
        "ensemble": num(judge["aiq"]), "ensemble_cost": num(judge["aiq_with_judge_cost"]),
        "ensemble_clean": num(clean["judges"][f"judge/{name}"]["aiq"]),
        "entropy": num(aiq[ENTROPY]), "lr_name": LR[inputs][1], "lr_aiq": num(aiq[lr_key]),
        "lr_best": num(max(aiq[k] for k in LR_FAMILY[inputs])),
        "random": num(aiq["random"]), "oracle": num(aiq["oracle"]),
        "d_entropy": diff(seed0 - aiq[ENTROPY], get(boot["aiq_diff"], f"{key}#seed0 - {ENTROPY}")),
        "d_entropy_cost": diff(seed0_cost - aiq[ENTROPY], get(boot["aiq_diff_with_cost"], f"{key}#seed0 - {ENTROPY}")),
        "d_lr": diff(seed0 - aiq[lr_key], get(boot["aiq_diff"], f"{key}#seed0 - {lr_key}")),
        "d_lr_cost": diff(seed0_cost - aiq[lr_key], get(boot["aiq_diff_with_cost"], f"{key}#seed0 - {lr_key}")),
    }
    return fill("pair.md", common_fields(org, "pair", size, inputs), fields)


def point(entry: dict, rule: str) -> str:
    p = entry.get(rule)
    return "none" if not p else f"{100 * p['lenient']:.2f}% at a price of {p['price']:.1f} (thresholds {p['t']})"


def render_menu(org: str, size: str, inputs: str, report: dict, clean: dict, grid: dict) -> str:
    name = menu_router(size, inputs)
    r, seeds = report["routers"], [report["routers"][f"{name}#seed{k}"] for k in (0, 1, 2)]
    s0, near, strong = seeds[0], near_strong(grid, size, inputs, 0), report["fixed_paths"]["SS"]
    get = lambda d, k: (d.get(k) or {}).get("ci95") if isinstance(d.get(k), dict) else d.get(k)  # noqa: E731
    fields = {
        "credit": RELEASE["menu_credit"],
        "n_test": f"{report['n_test']:,}",
        "seed0": num(s0["aiq"]), "seed0_ci": ci(get(report["aiq_ci95"], f"{name}#seed0")),
        "seed0_price": num(s0["aiq_with_judge_price"]),
        "seeds": f"{np.mean([e['aiq'] for e in seeds]):.3f} ± {np.std([e['aiq'] for e in seeds], ddof=1):.3f}",
        "seeds_price": num(float(np.mean([e["aiq_with_judge_price"] for e in seeds]))),
        "ensemble": num(r[name]["aiq"]), "ensemble_price": num(r[name]["aiq_with_judge_price"]),
        "ensemble_clean": num(clean["routers"][name]["aiq"]),
        "his_label": num(r[HIS[0]]["aiq"]), "his_reward": num(r[HIS[1]]["aiq"]),
        "fixed": num(report["aiq_fixed_paths"]),
        "max_drop": f"{RELEASE['menu']['max_drop']:.0%}",
        "near": f"{100 * near['lenient']:.2f}% at a price of {near['price']:.1f}, plus {near['judge_price']:.1f} for the "
                f"judge's own calls (thresholds {near['t']})",
        "near_vs_strong": f"{100 * (1 - near['lenient'] / strong['lenient']):.1f}% below always-strong's accuracy, at "
                          f"{(near['price'] + near['judge_price']) / strong['price']:.0%} of its price",
        "strong": f"{100 * strong['lenient']:.2f}% at a price of {strong['price']:.1f}",
        "peak": point(s0, "peak"), "parity": point(s0, "parity"),
        "ens_peak": point(r[name], "peak"), "his_peak": point(r[HIS[0]], "peak"),
        "d_label": diff(s0["aiq"] - r[HIS[0]]["aiq"], get(report["aiq_diff"], f"{name}#seed0 - {HIS[0]}")),
        "d_reward": diff(s0["aiq"] - r[HIS[1]]["aiq"], get(report["aiq_diff"], f"{name}#seed0 - {HIS[1]}")),
        "d_label_price": diff(s0["aiq_with_judge_price"] - r[HIS[0]]["aiq"],
                              get(report["aiq_diff_with_judge_price"], f"{name}#seed0 +judge - {HIS[0]}")),
        "d_reward_price": diff(s0["aiq_with_judge_price"] - r[HIS[1]]["aiq"],
                               get(report["aiq_diff_with_judge_price"], f"{name}#seed0 +judge - {HIS[1]}")),
        "e_label": diff(r[name]["aiq"] - r[HIS[0]]["aiq"], get(report["aiq_diff"], f"{name} - {HIS[0]}")),
        "e_reward": diff(r[name]["aiq"] - r[HIS[1]]["aiq"], get(report["aiq_diff"], f"{name} - {HIS[1]}")),
        "e_label_price": diff(r[name]["aiq_with_judge_price"] - r[HIS[0]]["aiq"],
                              get(report["aiq_diff_with_judge_price"], f"{name} +judge - {HIS[0]}")),
        "e_reward_price": diff(r[name]["aiq_with_judge_price"] - r[HIS[1]]["aiq"],
                               get(report["aiq_diff_with_judge_price"], f"{name} +judge - {HIS[1]}")),
    }
    return fill("menu.md", common_fields(org, "menu", size, inputs), fields)


# ---------------------------------------------------------------------------------------------
# Staging


def stage_weights(stage: Path, setting: str, size: str, inputs: str, report: dict, calib: pd.DataFrame | None,
                  grid: dict | None) -> None:
    for seed in (0, 1, 2):
        src, dst = seed_folder(setting, size, inputs, seed), stage if seed == 0 else stage / f"seed{seed}"
        rank = json.loads((src / "model" / "lora" / "adapter_config.json").read_text())["r"]
        if rank != SIZES[size]["lora_rank"]:
            raise SystemExit(f"{src}: LoRA rank {rank}, but the card says {SIZES[size]['lora_rank']}")
        shutil.copytree(src / "model", dst, dirs_exist_ok=True)
        shutil.copy(src / "done.json", dst / "training.json")
        card = {"setting": setting, "inputs": inputs, "seed": seed, "base": SIZES[size]["base"],
                "base_revision": SIZES[size]["base_revision"],
                "params_b": SIZES[size]["params_b"], "metric": "lenient"}
        if setting == "pair":
            card |= {"options": ["small model", "large model"], "threshold": pair_threshold(src, calib)}
        else:
            card |= {"options": ["yes", "no"], "thresholds": menu_thresholds(report, grid, size, inputs, seed)}
        (dst / "systemone.json").write_text(json.dumps(card, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--org", required=True)
    p.add_argument("--only", choices=list(SETTINGS))
    p.add_argument("--results", type=Path, default=ROOT / "results")
    p.add_argument("--cards-only", action="store_true", help="stage (and with --push upload) only each README.md")
    p.add_argument("--push", action="store_true")
    args = p.parse_args()
    if args.push and args.only != "pair" and "TODO" in RELEASE["menu_credit"]:
        raise SystemExit("set menu_credit in configs/release.toml (agreed with the labels' author) before pushing menu")
    for setting in [args.only] if args.only else list(SETTINGS):
        folder = args.results / SETTINGS[setting]
        report = json.loads((folder / "report.json").read_text())
        clean = json.loads((folder / "report-clean.json").read_text())
        # the cards quote each default threshold's test point, so --cards-only needs the same inputs as staging
        run, grid = (pair_run(), None) if setting == "pair" else ({}, menu_grid())
        calib = run.get("calib")
        render = partial(render_pair, run=run) if setting == "pair" else partial(render_menu, grid=grid)
        for size in SIZES:
            for inputs in INPUTS:
                repo = repo_name(args.org, setting, size, inputs)
                stage = ROOT / "build" / ("hf-cards" if args.cards_only else "hf") / repo.replace("/", "__")
                shutil.rmtree(stage, ignore_errors=True)
                stage.mkdir(parents=True)
                if not args.cards_only:
                    stage_weights(stage, setting, size, inputs, report, calib, grid)
                (stage / "README.md").write_text(render(args.org, size, inputs, report, clean), encoding="utf-8")
                print(f"{repo}: staged in {stage}")
                if args.push:
                    from huggingface_hub import HfApi

                    api = HfApi()
                    if args.cards_only:
                        api.upload_file(path_or_fileobj=stage / "README.md", path_in_repo="README.md", repo_id=repo,
                                        commit_message="Model card")
                    else:
                        api.create_repo(repo, exist_ok=True)
                        api.upload_folder(folder_path=stage, repo_id=repo, commit_message="SystemOne-Decider judge")
                    print(f"  pushed https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
