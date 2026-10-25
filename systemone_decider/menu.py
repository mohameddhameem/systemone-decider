"""The menu setting: a typed-decision judge routing the two-stage SLR menu on HotpotQA (`slr`), weak first.

The comparison with the SLR routers is on their terms: the Config B grid (every question has all four paths), the SLR
labels and split (train on slr-train, choose thresholds on slr-val, report once on slr-test), weak-first Table 5 prices
and the two SLR threshold rules (peak, parity). Only the router differs. "His" below refers to the SLR setup's author.

The judge is `judge`'s typed-decision judge (`judge.make_judge`, `judge.fine_tune`): nano-jev, or a Strands Decider.
One model answers two typed yes/no questions, one per stage, so P(yes) is P(the weak stage suffices), his routers' pW:

    retrieval  the weak retriever's (Qwen2.5-7B) selected passages: do they hold what the answer needs?
               label: his ret_target (W -> yes)
    answer     under the chosen retrieval, that branch's passages (qp), or with the weak answer under it too (qpa):
               will the weak answer do? label: his qa_target for that branch (W -> yes)

Weak-first, as he prices it: the weak answer runs before the answer decision, so qpa reads it at no extra price; the
retrieval decision reads only the weak selection, as his does. `slr.route` turns the three scores into a path.

A `joint` judge set asks one 4-option question instead, "which is the cheapest pipeline that answers
this correctly?" (WW / WS / SW / SS), once, on what the retrieval decision reads (plus path WW's answer for qpa). Its
label is his two targets as one path; its four probabilities become the same three scores (`joint_scores`), so it is
swept, priced and compared exactly like the two-question judges. The plan's `contrasts` add paired intervals between
our own routers.

    menu-judge train --judge decider-2b --plan configs/menu.toml   # one judge set, every (inputs, seed); resumes
    menu-judge eval --plan configs/menu.toml                       # every judge set trained so far, his routers
    menu-judge train --judge decider-2b --out runs/menu-models --save-model   # a run that keeps the weights

Eval: for each router (each seed of each judge, the seeds' mean, and his two routers) the 9,261-point threshold sweep on
slr-val picks the peak and parity points, which are applied to slr-test. AIQ is over the test points of the val-Pareto
thresholds, at weak-first prices from 80 (all WW) to 1,009 (all SS), with paired bootstrap intervals. Router cost: none,
as in his rows; and, beside it, each judge call priced with `stair_common.cost.budget` at its measured tokens
(nano-jev's calls price at 0: they are cheaper than one 3B sample). His routers are priced the same way
(`his_router_price`): each call is his own prompt (router_prompts.py at the pinned commit, read as literals) rendered
through Qwen2.5-1.5B-Instruct's chat template, so `aiq_diff_both_priced` compares the two with both routers charged.
"""

import argparse
import ast
import json
import logging
import re
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from stair_common.cost import SLR_PATH_PRICES, budget
from stair_common.splits import load_split

from systemone_decider import bootstrap as ev
from systemone_decider import provenance, slr
from systemone_decider.evaluate_routing import aiq
from systemone_decider.items import clean_answer
from systemone_decider.judge import budget_split, fine_tune, judge_params_b, make_judge

log = logging.getLogger(__name__)

QUESTIONS = {
    "ret": "Do these passages hold what is needed to answer this question: {question}",
    "qp": "Will a small model answer this question correctly from these passages: {question}",
    "qpa": "Is the proposed answer correct? Question: {question}",
}
OPTIONS = ("yes", "no")  # option 0 = the weak stage suffices
SCORES = ["ret", "qa_W", "qa_S"]  # P(yes) per question: retrieval, answer under weak / strong retrieval
JOINT = ("Which is the cheapest pipeline that answers this question correctly: {question}",  # E6, options in slr.PATHS order
         ("weak retrieval, weak answer", "weak retrieval, strong answer", "strong retrieval, weak answer",
          "strong retrieval, strong answer"))
PRICE_RANGE = (float(SLR_PATH_PRICES["WW"]), float(SLR_PATH_PRICES["SS"]))
PRICES = np.array([SLR_PATH_PRICES[p] for p in slr.PATHS], dtype=float)
HIS_PROMPTS = "jonathan/lab_package/router_prompts.py"  # his prompt contract, one copy for every router he trained
HIS_ROUTER = "Qwen/Qwen2.5-1.5B-Instruct"  # the base of all his Config B routers (SFT and reward LoRA)
HIS_ROUTER_PARAMS_B = 1.5  # nominal size, as stair_common prices a model Table 5 does not list


# ---------------------------------------------------------------------------------------------
# Data


def selection(text: str) -> str:
    """A retriever's output without its "Query:" line: the selected references, "[k] title: text" per line."""
    return text.split("Selected references:", 1)[-1].strip() or "(no passages selected)"


def load(plan: dict) -> pd.DataFrame:
    """The SLR grid (slr.load_grid) plus what the judge reads and learns: the question, each retriever's
    selection, his labels."""
    slr_plan = tomllib.loads(Path(plan["slr_plan"]).read_text(encoding="utf-8"))
    df = slr.load_grid(slr_plan)
    root = Path(slr_plan["out"]) / slr.GRID
    df["question"] = pd.read_parquet(slr_plan["dev"], columns=["question"])["question"].to_numpy()
    for x in "WS":
        sel = {r["k"]: r["v"] for r in slr.read_jsonl(root / f"retr_{slr.BRANCH[x]}.jsonl")}
        df[f"sel_{x}"] = [selection(str(sel[str(i)])) for i in df["idx"]]
    ret = {r["idx"]: r["ret_target"] for r in slr.read_jsonl(root / "ret_labels" / "ret_labels.jsonl")}
    qa = {(r["idx"], r["ret_choice"]): r["qa_target"] for r in slr.read_jsonl(root / "qa_labels" / "qa_labels.jsonl")}
    df["y_ret"] = [ret.get(i) for i in df["idx"]]
    for x in "WS":
        df[f"y_qa_{x}"] = [qa.get((i, x)) for i in df["idx"]]
    return df


def items(df: pd.DataFrame, inputs: str) -> dict[str, list[dict]]:
    """Per score (ret, qa_W, qa_S), one typed item per question."""
    out = {"ret": [{"question": QUESTIONS["ret"].format(question=q), "options": OPTIONS, "state": s}
                   for q, s in zip(df["question"], df["sel_W"])]}
    for x in "WS":
        weak_answer = df[f"answer_{x}W"]  # the weak answer under retrieval x: path xW
        states = (df[f"sel_{x}"] if inputs == "qp" else
                  [f"Proposed answer: {clean_answer(a)[:200]}\n{s}" for a, s in zip(weak_answer, df[f"sel_{x}"])])
        out[f"qa_{x}"] = [{"question": QUESTIONS[inputs].format(question=q), "options": OPTIONS, "state": s}
                          for q, s in zip(df["question"], states)]
    return out


def labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """0 (yes: weak suffices) or 1, per score; his targets, which map every question to W or S."""
    out = {"ret": df["y_ret"], **{f"qa_{x}": df[f"y_qa_{x}"] for x in "WS"}}
    if any(v.isna().any() or not v.isin(["W", "S"]).all() for v in out.values()):
        raise ValueError("training questions need his W/S targets for both stages")
    return {k: (v == "S").astype(int).to_numpy() for k, v in out.items()}


def joint_items(df: pd.DataFrame, inputs: str) -> list[dict]:
    """E6: one 4-option typed question per question, asked once before anything escalates, so it reads only the weak
    selection (qp), and path WW's answer too (qpa), which weak-first prices already pay."""
    states = (df["sel_W"] if inputs == "qp" else
              [f"Proposed answer: {clean_answer(a)[:200]}\n{s}" for a, s in zip(df["answer_WW"], df["sel_W"])])
    return [{"question": JOINT[0].format(question=q), "options": JOINT[1], "state": s}
            for q, s in zip(df["question"], states)]


def joint_labels(df: pd.DataFrame) -> np.ndarray:
    """His two targets as one path (slr.PATHS index): the retrieval target, then the answer target under it."""
    y = labels(df)
    return np.where(y["ret"] == 0, y["qa_W"], 2 + y["qa_S"])


def joint_scores(p: np.ndarray) -> dict[str, np.ndarray]:
    """[questions, 4] path probabilities as the three scores slr.route thresholds: P(weak retrieval) = P(WW) + P(WS),
    then P(weak answer) under each retrieval, so the joint judge runs through the same sweep as the stage judges."""
    eps = 1e-12
    return {"ret": p[:, 0] + p[:, 1], "qa_W": p[:, 0] / (p[:, 0] + p[:, 1] + eps),
            "qa_S": p[:, 2] / (p[:, 2] + p[:, 3] + eps)}


# ---------------------------------------------------------------------------------------------
# Training


def judge_set(plan: dict, name: str) -> dict:
    return next(j for j in plan["judges"] if j["name"] == name)


def run_dir(plan: dict, spec: dict, inputs: str, seed: int, budget: int | None = None) -> Path:
    """<out>/<judge_dir>/<inputs>[/b<budget>]/seed<k>."""
    root = Path(plan["out"]) / spec["judge_dir"] / inputs
    return (root / f"b{budget}" if budget else root) / f"seed{seed}"


def run_train(df: pd.DataFrame, plan: dict, spec: dict, inputs: str, seed: int, device: str | None,
              budget: int | None = None, save_model: bool = False) -> None:
    """One judge seed. `budget` (a label budget) trains on that many questions, its dev hold-out included, nested
    and stratified by his retrieval label (judge.budget_split); a set's `train_scores` (E6) trains on those stages'
    items only, while every stage is still scored; a `joint` set (E6) asks the one 4-option question instead, one call
    a question (its tokens in tokens_ret, so the router pays that call once). `save_model` keeps the weights in
    <seed>/model/, as judge.run_train does."""
    target = run_dir(plan, spec, inputs, seed, budget)
    tag = f"b{budget} " if budget else ""
    if (target / "done.json").exists():
        log.info(f"{spec['name']} {inputs} {tag}seed {seed}: done")
        return
    hp = spec["train"]
    train = df[df["split"] == "train"].reset_index(drop=True)
    if budget:
        tr, dev = budget_split((train["y_ret"] == "S").astype(int).to_numpy(), budget, hp["dev_fraction"], seed)
        tr_q, dev_q = np.isin(np.arange(len(train)), tr), np.isin(np.arange(len(train)), dev)
    else:
        rng = np.random.default_rng(seed)
        dev_q = rng.random(len(train)) < hp["dev_fraction"]  # held out by question: a question's items stay together
        tr_q = ~dev_q
    joint = spec.get("joint", False)
    stages = ["joint"] if joint else spec.get("train_scores", SCORES)
    if joint:
        its, ys = {"joint": joint_items(train, inputs)}, {"joint": joint_labels(train)}
    else:
        its, ys = items(train, inputs), labels(train)
    pick = lambda mask: ([it for k in stages for it, m in zip(its[k], mask) if m],  # noqa: E731
                         np.concatenate([ys[k][mask] for k in stages]))
    (tr_items, tr_y), (dev_items, dev_y) = pick(tr_q), pick(dev_q)
    log.info(f"{spec['name']} {inputs} {tag}seed {seed}: {len(tr_items)} items ({int(tr_y.sum())} escalate), "
             f"{len(dev_items)} dev items")
    judge = make_judge(spec["init"], device, hp.get("max_length", 512), spec.get("score_batch"))
    start = time.perf_counter()
    history = fine_tune(judge, tr_items, tr_y, dev_items, dev_y, hp["epochs"], hp["lr"], hp["weight_decay"],
                        hp["warmup"], hp["batch_size"], seed, hp.get("micro_batch"), hp.get("min_steps", 0))
    train_seconds = time.perf_counter() - start
    target.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        cols = {"idx": part["idx"]}
        if joint:
            p, tokens = judge.probabilities(joint_items(part, inputs))
            cols |= joint_scores(p) | {"tokens_ret": tokens, "tokens_qa_W": np.zeros_like(tokens),
                                       "tokens_qa_S": np.zeros_like(tokens)}
            for j, path in enumerate(slr.PATHS):
                cols[f"p_{path}"] = p[:, j]
        else:
            for k, its_k in items(part, inputs).items():
                p, tokens = judge.probabilities(its_k)
                cols[k], cols[f"tokens_{k}"] = p[:, 0], tokens
        pd.DataFrame(cols).to_parquet(target / f"{split}.parquet", index=False)
    if save_model:
        judge.model.save_pretrained(target / "model")
        judge.tokenizer.save_pretrained(target / "model")
    (target / "done.json").write_text(json.dumps({
        "judge": spec["name"], "inputs": inputs, "seed": seed, "init": spec["init"], "train": hp, "budget": budget,
        "train_scores": list(stages), "n_train_questions": int(tr_q.sum()), "n_dev_questions": int(dev_q.sum()),
        "n_train_items": len(tr_items), "n_dev_items": len(dev_items), "history": history, "train_seconds": train_seconds,
        "init_revision": provenance.hub_revision(spec["init"]), "device": judge.device}, indent=1))


# ---------------------------------------------------------------------------------------------
# Evaluation


class Router:
    """One router's scores on val and test, and its judge's price per question (0 for his routers)."""

    def __init__(self, name: str, val: dict, test: dict, judge_price_test: np.ndarray | None = None):
        self.name, self.val, self.test = name, val, test
        self.judge_price = judge_price_test  # [questions, 3]: the ret call, the qa call under W, under S

    def choice(self, scores: dict, t) -> np.ndarray:
        return slr.route(scores["ret"], {"W": scores["qa_W"], "S": scores["qa_S"]}, t)

    def extra(self, choice: np.ndarray) -> np.ndarray:
        """The judge's own price: the retrieval call, and the answer call on the branch it took."""
        if self.judge_price is None:
            return np.zeros(len(choice))
        return self.judge_price[:, 0] + np.where(choice < 2, self.judge_price[:, 1], self.judge_price[:, 2])


def judge_routers(plan: dict, test_ids: pd.Series, val_ids: pd.Series) -> list[Router]:
    """Every trained (judge set, inputs): each seed, and the mean of the seeds (which pays every seed's calls)."""
    routers = []
    for spec in plan["judges"]:
        params = judge_params_b(spec["init"])
        for inputs in spec["inputs"]:
            for b in spec.get("budgets", [None]):
                label = f"{spec['name']} {inputs}" + (f" b{b}" if b else "")
                seeds = [s for s in spec["seeds"] if (run_dir(plan, spec, inputs, s, b) / "done.json").exists()]
                if not seeds:
                    log.info(f"{label}: not trained yet")
                    continue
                per = {}
                for s in seeds:
                    folder = run_dir(plan, spec, inputs, s, b)
                    val = pd.read_parquet(folder / "val.parquet").set_index("idx").loc[val_ids]
                    test = pd.read_parquet(folder / "test.parquet").set_index("idx").loc[test_ids]
                    price = np.stack([[budget(params, int(t), 0) for t in test[f"tokens_{k}"]] for k in SCORES], axis=1)
                    per[s] = ({k: val[k].to_numpy() for k in SCORES}, {k: test[k].to_numpy() for k in SCORES}, price)
                    routers.append(Router(f"{label}#seed{s}", *per[s]))
                if len(seeds) > 1:
                    mean = lambda i: {k: np.mean([per[s][i][k] for s in seeds], axis=0) for k in SCORES}  # noqa: E731
                    routers.append(Router(label, mean(0), mean(1), sum(per[s][2] for s in seeds)))
    return routers


def his_routers(plan: dict, slr_plan: dict, val_ids: pd.Series, test_ids: pd.Series,
                price: np.ndarray | None = None) -> list[Router]:
    """His routers from his frozen scores; `price` charges their calls ([test questions, 3]), None leaves them free."""
    out = []
    for row in plan["his_routers"]:
        scores = {}
        for split, names, ids in (("val", row["val"], val_ids), ("test", row["test"], test_ids)):
            ret, qa = slr.his_scores(slr_plan, *names, ids, split=split)
            scores[split] = {"ret": ret, "qa_W": qa["W"], "qa_S": qa["S"]}
        out.append(Router(row["name"], scores["val"], scores["test"], price))
    return out


def his_prompts(path: Path) -> dict:
    """The literal constants of his prompt contract (templates, CHAR_CAP, MAXLEN), read with `ast`: never imported."""
    out = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                pass  # not a literal
    return out


def his_router_calls(df: pd.DataFrame, prompts: dict) -> tuple[str, dict[str, list[str]]]:
    """His system prompt and, per score, the user turn of each question's router call: INSTR_RET2 on the weak
    selections, INSTR_QA1 on each retrieval's references with the weak answer under it. Passages are capped at
    CHAR_CAP as his format_selections / format_evidence do; they come from the retrievers' outputs, which can reword
    a pool passage, so the counts are close to his, not identical."""
    cap = prompts["CHAR_CAP"]
    system = prompts["SYS_CASCADE_TEMPLATE"].format(weak_name=prompts["WEAK_NAME_DEFAULT"],
                                                    strong_name=prompts["STRONG_NAME_DEFAULT"])

    def refs(sel: str) -> list[str]:
        return [m.group(1) + m.group(2)[:cap] for m in re.finditer(r"^(\[\d+\] )(.*)$", sel, re.MULTILINE)]

    calls = {k: [] for k in SCORES}
    for row in df.itertuples():
        weak = refs(row.sel_W)
        calls["ret"].append(prompts["INSTR_RET2"].format(q=row.question, n_sel=len(weak),
                                                        selections=" ".join(weak) or "(none)"))
        for x in "WS":
            calls[f"qa_{x}"].append(prompts["INSTR_QA1"].format(
                q=row.question, evidence="\n".join(refs(getattr(row, f"sel_{x}"))),
                weak_answer=getattr(row, f"answer_{x}W")))
    return system, calls


def his_router_price(df: pd.DataFrame, prompts: dict, tokenizer) -> np.ndarray:
    """[questions, 3]: each of his router calls priced like ours, `budget` at its prompt tokens (his routers read the
    next-token logits, so they generate none); the prompt is cut to MAXLEN tokens as his assemble() does."""
    system, calls = his_router_calls(df, prompts)

    def tokens(user: str) -> int:
        text = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                             tokenize=False, add_generation_prompt=True)
        return min(len(tokenizer(text, add_special_tokens=False)["input_ids"]), prompts["MAXLEN"])

    return np.array([[budget(HIS_ROUTER_PARAMS_B, tokens(calls[k][i]), 0) for k in SCORES] for i in range(len(df))])


def pareto(points: list[tuple]) -> list[tuple]:
    """The thresholds of the val points no other point beats on both accuracy and price."""
    best, out = -1.0, []
    for acc, cost, *t in sorted(points, key=lambda p: (p[1], -p[0])):
        if acc > best:
            best = acc
            out.append(tuple(t))
    return out


def evaluate(df: pd.DataFrame, routers: list[Router], plan: dict, keep: np.ndarray | None = None) -> dict:
    """Peak and parity points (thresholds from slr-val) and AIQ on slr-test; `keep` restricts the test questions."""
    val, test = df[df["split"] == "val"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    keep = np.ones(len(test), dtype=bool) if keep is None else keep
    ok_val, ok_test = slr.ok_matrix(val), slr.ok_matrix(test)[keep]
    strong = int(val["ok_SS"].sum()) / len(val)
    n, rows = int(keep.sum()), np.arange(int(keep.sum()))
    report = {"n_test": n, "price_range": PRICE_RANGE, "routers": {}}
    curves = {"fixed paths": lambda ix: [{"f1": float(ok_test[ix, k].mean()), "cost": slr.direct_price(p)}
                                         for k, p in enumerate(slr.PATHS)]}
    for r in routers:
        points = slr.sweep(ok_val, r.val["ret"], {"W": r.val["qa_W"], "S": r.val["qa_S"]})
        chosen = slr.select(points, strong)
        frontier = pareto(points)
        full = np.stack([r.choice(r.test, t) for t in frontier])  # [thresholds, test questions]
        choices, extras = full[:, keep], np.stack([r.extra(c) for c in full])[:, keep]
        entry = {}
        for rule, t in chosen.items():
            if t is None:
                entry[rule] = None
                continue
            c = r.choice(r.test, t)
            entry[rule] = {"t": t, "lenient": float(ok_test[rows, c[keep]].mean()), "price": float(PRICES[c[keep]].mean()),
                           "judge_price": float(r.extra(c)[keep].mean()),
                           "paths": {p: int((c[keep] == k).sum()) for k, p in enumerate(slr.PATHS)}}
        for tag, add in (("", False), (" +judge", True)):
            curves[r.name + tag] = (lambda ix, ch=choices, ex=extras, a=add:
                                    [{"f1": float(ok_test[ix, c[ix]].mean()), "cost": float((PRICES[c[ix]] + a * e[ix]).mean())}
                                     for c, e in zip(ch, ex)])
        entry["aiq"] = aiq(curves[r.name](rows), *PRICE_RANGE)
        entry["aiq_with_judge_price"] = aiq(curves[r.name + " +judge"](rows), *PRICE_RANGE)
        entry["frontier_size"] = len(frontier)
        report["routers"][r.name] = entry
    report["fixed_paths"] = {p: {"lenient": float(ok_test[:, k].mean()), "price": slr.direct_price(p)}
                             for k, p in enumerate(slr.PATHS)}
    choice = ev.cheapest_correct_label(ok_test, np.tile([slr.direct_price(p) for p in slr.PATHS], (n, 1)))
    report["oracle"] = {"lenient": float(ok_test[rows, choice].mean()),
                        "price": float(np.mean([slr.direct_price(slr.PATHS[c]) for c in choice]))}
    report["aiq_fixed_paths"] = aiq(curves["fixed paths"](rows), *PRICE_RANGE)
    boot = ev.bootstrap(curves, n, plan.get("bootstrap", 1000), 0, PRICE_RANGE)
    report["aiq_ci95"] = boot["aiq_ci95"]
    theirs = [r.name for r in routers if r.name.startswith("SLR ")] + ["fixed paths"]
    ours = [r.name for r in routers if not r.name.startswith("SLR ")]
    report["aiq_diff"] = {f"{a} - {b}": ev.diff(boot, a, b) for a in ours for b in theirs}
    report["aiq_diff_with_judge_price"] = {f"{a} +judge - {b}": ev.diff(boot, a + " +judge", b)
                                           for a in ours for b in theirs}
    report["aiq_diff_both_priced"] = {f"{a} +judge - {b} +judge": ev.diff(boot, a + " +judge", b + " +judge")
                                      for a in ours for b in theirs if b != "fixed paths"}
    # the plan's own pairs among our routers (E6: each stage-transfer or joint judge against the two-stage judge)
    report["aiq_diff_contrasts"] = {f"{a} - {b}": ev.diff(boot, a, b) for a, b in plan.get("contrasts", [])
                                    if a in curves and b in curves}
    return report


def run_eval(df: pd.DataFrame, plan: dict) -> dict:
    from transformers import AutoTokenizer

    slr_plan = tomllib.loads(Path(plan["slr_plan"]).read_text(encoding="utf-8"))
    val_ids, test_ids = (df.loc[df["split"] == s, "idx"].reset_index(drop=True) for s in ("val", "test"))
    test = df[df["split"] == "test"].set_index("idx").loc[test_ids].reset_index()
    prompts = his_prompts(Path(slr_plan["out"]) / HIS_PROMPTS)
    price = his_router_price(test, prompts, AutoTokenizer.from_pretrained(HIS_ROUTER))
    log.info(f"his router calls: {price.mean(axis=0).round(2).tolist()} a question (ret, qa under W, under S), "
             f"{HIS_ROUTER}@{provenance.hub_revision(HIS_ROUTER)}")
    routers = his_routers(plan, slr_plan, val_ids, test_ids, price) + judge_routers(plan, test_ids, val_ids)
    out = Path(plan["out"]) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    seen = set(load_split("hotpotqa", "nanojev_v1_seen"))
    clean = ~df.loc[df["split"] == "test", "id"].isin(seen).to_numpy()
    reports = {}
    for tag, keep in (("", None), ("-clean", clean)):
        reports[tag] = evaluate(df, routers, plan, keep)
        (out / f"report{tag}.json").write_text(json.dumps(reports[tag], indent=1) + "\n", encoding="utf-8")
    for name, e in reports[""]["routers"].items():
        pk, pa = e["peak"], e["parity"]
        log.info(f"{name:34s} AIQ {e['aiq']:.4f} | peak {100 * pk['lenient']:.2f} at {pk['price']:.1f}"
                 + (f" | parity {100 * pa['lenient']:.2f} at {pa['price']:.1f}" if pa else " | parity: none"))
    return reports[""]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "eval"])
    parser.add_argument("--plan", type=Path, default=Path("configs/menu.toml"))
    parser.add_argument("--judge", help="train: the judge set (a [[judges]] name; several, comma-separated)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--out", type=Path, help="the run folder instead of the plan's `out` (train and eval)")
    parser.add_argument("--save-model", action="store_true", help="train: also save the fine-tuned weights")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = tomllib.loads(args.plan.read_text(encoding="utf-8"))
    if args.out:
        plan["out"] = str(args.out)
    df = load(plan)
    out = Path(plan["out"])
    provenance.record(out, f"menu-judge {args.step}", plan=str(args.plan), judge=args.judge, save_model=args.save_model,
                      **provenance.environment())
    if args.step == "train":
        if not args.judge:
            raise SystemExit("train needs --judge <name>")
        for spec in [judge_set(plan, name) for name in args.judge.split(",")]:
            for inputs in spec["inputs"]:
                for b in spec.get("budgets", [None]):
                    for seed in spec["seeds"]:
                        run_train(df, plan, spec, inputs, seed, args.device, b, args.save_model)
    else:
        run_eval(df, plan)


if __name__ == "__main__":
    main()
