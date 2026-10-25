"""System One router: a typed-decision judge for small/large routing.

A judge is a nano-jev-style cross-encoder. For one typed question it scores every (question +
option, state) pair, and softmaxes the option scores together. Here the state is built from a
cached run's question, the small expert's passages (its top-k nano-jev paragraphs) and,
optionally, the small expert's answer:

    q    the question                         (decides before the small expert)
    qp   the passages                         (before)
    qpa  "Proposed answer: ..." + passages    (after the small expert: a cascade)
    qa   "Proposed answer: ..."               (after)

Subcommands (all read existing caches and run no LLM):

    zeroshot  nano-jev's `grounded` question and a custom "is the proposed answer correct?" question
              on the small expert's answer: escalation score P(no)
    train     fine-tune on "Which model should answer this question" {small model, large model},
              labelled from a pair's router_train caches; writes P(large model) for calib and test
              of every --score-pair
    eval      merge the judges' scores into a copy of a pair's caches and evaluate them with
              eval-routing's protocol, next to retrained logistic regression routers
    all       everything in a plan file (configs/pair.toml), skipping finished steps

Each judge also records its tokens per question, so its own cost (2 x params x tokens) can be
added to its cost-F1 curve.

A judge whose checkpoint is a Strands Decider (`systemone_decider.decider`) runs through
`decider.DeciderJudge`, with the same interface; `make_judge` picks the backend from the name. A plan's
`zero_shot_judge` may be one checkpoint (columns `zs_grounded`, `zs_correct`, in `zeroshot/<pair>`) or a
list (columns `zs_<name>_grounded`, ..., in `zeroshot/<name>/<pair>`), and `eval_dir` (default `eval`)
keeps a second plan's reports apart from the first's.
"""

import argparse
import gc
import hashlib
import json
import logging
import shutil
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from systemone_decider import provenance

from systemone_decider.evaluate_routing import aiq, evaluate, expert_params, judge_cost_in, with_cost_unit
from systemone_decider.evidence import paragraph_text, passages_state, resolve_checkpoint
from systemone_decider.decider import DeciderJudge, is_decider, params_b
from systemone_decider.metrics import metric_view
from systemone_decider.router import escalation_label, train_routers

log = logging.getLogger(__name__)

INPUTS = {"q": False, "qp": False, "qpa": True, "qa": True}  # input set -> needs the small expert's answer
HEADS = ("typed", "plain")  # the option-scoring head (nano-jev's), or the untyped ablation (Judge)
GRID_KEYS = ("init", "inputs", "head", "budget")  # a [[judges]] entry may list these; load_plan expands the product
ROUTE = ("Which model should answer this question: {question}", ("small model", "large model"))
GROUNDED = ("Is this claim supported by the context? Claim: {claim}", ("yes", "no"))  # nano-jev built-in
CLAIM = 'The answer to "{question}" is {answer}.'
CORRECT = ("Is the proposed answer correct? Question: {question} Proposed answer: {answer}", ("yes", "no"))
ZERO_SHOT = ("grounded", "correct")  # zero-shot questions; columns <prefix>_grounded, <prefix>_correct
JUDGE_PARAMS_B = 0.0334  # nano-jev / MiniLM-L12-H384, billions
SPLIT_DATA = {"router_train": "distractor_train.parquet", "calib": "distractor_train.parquet",
              "test": "distractor_validation.parquet"}  # a plan's [split_data] overrides (common: calib is dev)
JUDGE_COST_UNITS = ("flops", "tokens", "agenttts", "table5")
LR_REFERENCES = ("router:question+evidence/large-helps", "router:question+evidence+small/large-helps")


# ---------------------------------------------------------------------------------------------
# Data


def read_paragraphs(path: Path, ids: set[str], batch_size: int = 2048) -> dict[str, list[str]]:
    """id -> "title: text" paragraphs (run-experts' paragraphs_of) for the given ids only.

    Streams the file in batches: the full HotpotQA train file as pandas objects needs several GB
    and crashed a 15 GB WSL machine, while the ~21k questions the pairs use need ~100 MB.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    wanted = pa.array(sorted(ids), type=pa.string())
    out = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size,
                                                   columns=["id", "context_titles", "context_sentences"]):
        keep = batch.filter(pc.is_in(batch.column("id").cast(pa.string()), value_set=wanted))
        for qid, titles, sentences in zip(*(keep.column(c).to_pylist() for c in ("id", "context_titles", "context_sentences"))):
            out[qid] = [paragraph_text(t, s) for t, s in zip(titles, sentences)]
    missing = ids - out.keys()
    if missing:
        raise ValueError(f"{len(missing)} questions are not in {path}, e.g. {sorted(missing)[:3]}")
    return out


def load_pair_split(run: Path, data_dir: Path, split: str, paragraphs: dict[str, list[str]] | None = None,
                    split_data: dict[str, str] | None = None) -> pd.DataFrame:
    """A pair's cached split, plus the small expert's passages rebuilt from the dataset."""
    cache = pd.read_parquet(run / f"{split}.parquet")
    cfg = tomllib.loads((run / "config.toml").read_text())
    k = next(e for e in cfg["experts"] if e["name"] == "small").get("top_k")
    if paragraphs is None:
        paragraphs = read_paragraphs(data_dir / (split_data or SPLIT_DATA)[split], set(cache["id"]))
    passages = []
    for qid, scores in zip(cache["id"], cache["para_scores"]):
        order = np.argsort(-np.asarray(scores), kind="stable")  # run-experts' ranking
        passages.append([paragraphs[qid][i] for i in order][:k] if k else paragraphs[qid])
    cache["passages"] = passages
    return cache


def state(row, inputs: str) -> str:
    answer = f"Proposed answer: {row['small_pred']}"
    return {"q": row["question"], "qp": passages_state(row["passages"]),
            "qpa": f"{answer}\n{passages_state(row['passages'])}", "qa": answer}[inputs]


def route_items(df: pd.DataFrame, inputs: str) -> list[dict]:
    question, options = ROUTE
    return [{"question": question.format(question=r["question"]), "options": options, "state": state(r, inputs)}
            for _, r in df.iterrows()]


# ---------------------------------------------------------------------------------------------
# Model


class Judge:
    """A cross-encoder with one logit per (question + option, state) pair, softmaxed per item.

    `head="plain"` is the untyped ablation on the same backbone: one (question, state) pair per item and a fresh
    classifier head with one logit per option, so no option text is read and nothing of a typed head is reused.
    """

    def __init__(self, path: str, device: str | None = None, max_length: int = 512, score_batch: int | None = None,
                 head: str = "typed", n_options: int = 2):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
        self.torch, self.head = torch, head
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        local = resolve_checkpoint(path)
        self.tokenizer = AutoTokenizer.from_pretrained(local)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            local, num_labels=1 if head == "typed" else n_options, ignore_mismatched_sizes=True).to(self.device)
        calib = local / "calibration.json"
        self.temperatures = json.loads(calib.read_text()) if calib.exists() and head == "typed" else {}
        self.max_length = max_length
        self.score_batch = score_batch or (64 if self.device != "cpu" else 16)  # questions per inference batch

    def encode(self, items: list[dict]):
        if self.head == "plain":
            firsts, seconds = [it["question"] for it in items], [it["state"] for it in items]
        else:
            firsts = [f"question: {it['question']} option: {o}" for it in items for o in it["options"]]
            seconds = [it["state"] for it in items for _ in it["options"]]
        enc = self.tokenizer(firsts, seconds, truncation="only_second", max_length=self.max_length,
                             padding=True, return_tensors="pt")
        return enc

    def logits(self, items: list[dict]):
        """[items, options] logits (all items must have the same number of options)."""
        enc = self.encode(items).to(self.device)
        with self.torch.autocast(self.device if self.device != "cpu" else "cpu", dtype=self.torch.bfloat16,
                                 enabled=self.device != "cpu"):
            out = self.model(**enc).logits.float()
        if self.head == "plain":
            return out
        return out[:, 0].view(len(items), len(items[0]["options"]))

    def probabilities(self, items: list[dict], temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """[items, options] probabilities and the tokens each item processed (its cost)."""
        self.model.eval()
        batch_size = self.score_batch
        probs, tokens = [], []
        with self.torch.no_grad():
            for i in range(0, len(items), batch_size):
                batch = items[i: i + batch_size]
                enc = self.encode(batch)
                tokens += enc["attention_mask"].view(len(batch), -1).sum(dim=1).tolist()
                probs.append(self.torch.softmax(self.logits(batch) / temperature, dim=1).cpu().numpy())
        return np.concatenate(probs), np.asarray(tokens)


def make_judge(path: str, device: str | None = None, max_length: int = 512, score_batch: int | None = None,
               head: str = "typed"):
    """A Strands Decider or a nano-jev-style cross-encoder, by the checkpoint's name."""
    if is_decider(path):
        if head != "typed":
            raise ValueError("a Strands Decider has only its typed (pointer) head")
        return DeciderJudge(path, device, max_length, score_batch)
    return Judge(path, device, max_length, score_batch, head)


def budget_order(y: np.ndarray, seed: int) -> np.ndarray:
    """A seed's stratified order of the questions: every prefix keeps the label rate, so budgets nest (the 50 labels
    are inside the 100). Within each class the order is random; classes interleave by rank / class size."""
    rng = np.random.default_rng(seed)
    key = np.empty(len(y))
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        key[rng.permutation(idx)] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    return np.argsort(key, kind="stable")


def budget_split(y: np.ndarray, budget: int, dev_fraction: float, seed: int, min_dev: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """(train, dev) indices for a label budget: the seed's first `budget` questions, `dev_fraction` of them (at least
    `min_dev`) held out to pick the epoch. Every label the judge sees is inside the budget."""
    from sklearn.model_selection import train_test_split

    if budget > len(y):
        raise ValueError(f"budget {budget} > {len(y)} training questions")
    chosen = budget_order(y, seed)[:budget]
    n_dev = max(min_dev, int(round(dev_fraction * budget)))
    tr, dev = train_test_split(chosen, test_size=n_dev, stratify=y[chosen], random_state=seed)
    return tr, dev


def judge_params_b(path: str) -> float:
    return params_b(path) if is_decider(path) else JUDGE_PARAMS_B


def short_name(path: str) -> str:
    return path.split("/")[-1].split("@")[0].lower()


def zero_shot_judges(plan: dict) -> list[tuple[str, str, Path]]:
    """(checkpoint, column prefix, output folder under --out) for each zero-shot judge of a plan."""
    z = plan.get("zero_shot_judge")
    if not z:
        return []
    if isinstance(z, str):
        return [(z, "zs", Path("zeroshot"))]
    return [(path, f"zs_{short_name(path)}", Path("zeroshot") / short_name(path)) for path in z]


def accumulate(judge: Judge, items: list[dict], target, micro_batch: int) -> float:
    """Backpropagate the mean grouped-softmax cross-entropy of `items` in micro-batches; returns the loss.

    The gradient equals one pass over the whole batch, but activation memory is bounded by
    `micro_batch` questions: 16 questions x 2 options x 512 tokens need > 10 GB in float32 on CPU.
    """
    import torch

    total = 0.0
    for j in range(0, len(items), micro_batch):
        loss = torch.nn.functional.cross_entropy(judge.logits(items[j: j + micro_batch]),
                                                 target[j: j + micro_batch].to(judge.device))
        loss = loss * min(micro_batch, len(items) - j) / len(items)
        loss.backward()
        total += loss.item()
    return total


def fine_tune(judge: Judge, items: list[dict], labels: np.ndarray, dev_items: list[dict], dev_labels: np.ndarray,
              epochs: int, lr: float, weight_decay: float, warmup: float, batch_size: int, seed: int,
              micro_batch: int | None = None, min_steps: int = 0) -> list[dict]:
    """Grouped-softmax cross-entropy; keeps the epoch with the lowest dev NLL. Returns the history.

    `micro_batch` (default: the whole batch on GPU, 2 questions on CPU) only bounds memory. `min_steps` adds epochs
    until training takes at least that many optimizer steps (small label budgets: 40 questions are 3 steps an epoch).
    """
    import torch
    from transformers import get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = judge.model
    trainable = [p for p in model.parameters() if p.requires_grad]  # a decider trains only its LoRA and head
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    per_epoch = int(np.ceil(len(items) / batch_size))
    epochs = max(epochs, int(np.ceil(min_steps / per_epoch)))
    steps = epochs * per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(warmup * steps), steps)
    target = torch.tensor(np.asarray(labels), dtype=torch.long)
    micro_batch = micro_batch or (batch_size if judge.device != "cpu" else 2)
    history, best, best_state = [], np.inf, None
    for epoch in range(epochs):
        model.train()
        start, losses = time.perf_counter(), []
        order = rng.permutation(len(items))
        for i in range(0, len(order), batch_size):
            b = order[i: i + batch_size]
            opt.zero_grad()
            losses.append(accumulate(judge, [items[j] for j in b], target[b], micro_batch))
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
        probs, _ = judge.probabilities(dev_items)
        nll = float(-np.log(np.clip(probs[np.arange(len(dev_labels)), dev_labels], 1e-12, 1)).mean())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "dev_nll": nll,
                        "seconds": time.perf_counter() - start})
        log.info(f"    epoch {epoch + 1}: train loss {history[-1]['train_loss']:.4f} dev NLL {nll:.4f}")
        if nll < best:
            best, best_state = nll, {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    model.load_state_dict(best_state, strict=False)  # only the trained parameters change
    return history


# ---------------------------------------------------------------------------------------------
# Steps


def init_label(spec: dict) -> str:
    """A judge's starting point as its name shows it: `init_name` (a local folder), "nanojev", or the Hub model name."""
    return spec.get("init_name") or ("nanojev" if "nano-jev" in spec["init"]
                                     else spec["init"].split("/")[-1].split("@")[0].lower())


def judge_name(spec: dict) -> str:
    """inputs-label-init-pair, then -plain for the untyped head and -b<budget> for a label budget. `init_name` (from
    the plan's [init_names]) names a local checkpoint folder."""
    name = f"{spec['inputs']}-{spec['label']}-{init_label(spec)}-{spec['train_pair']}"
    if spec.get("head", "typed") != "typed":
        name += f"-{spec['head']}"
    if spec.get("budget"):
        name += f"-b{spec['budget']}"
    return name


def fingerprint(items: list[dict]) -> str:
    """Identity of a judge's inputs: pairs that share questions, passages and small answers score the same."""
    h = hashlib.sha1()
    for it in items:
        h.update(f"{it['question']}\x01{it['state']}\x00".encode())
    return h.hexdigest()


class Splits:
    """Loaded pair splits, cached across steps. The pairs share their questions, so each dataset
    file's paragraphs are read once (only for the questions needed) and reused by every pair."""

    def __init__(self, pairs: dict[str, Path], data_dir: Path, split_data: dict[str, str] | None = None):
        self.pairs, self.data_dir, self._cache = pairs, data_dir, {}
        self.split_data = {**SPLIT_DATA, **(split_data or {})}
        self._paragraphs: dict[str, dict[str, list[str]]] = {}

    def get(self, pair: str, split: str) -> pd.DataFrame:
        if (pair, split) not in self._cache:
            file = self.split_data[split]
            known = self._paragraphs.setdefault(file, {})
            ids = set(pd.read_parquet(self.pairs[pair] / f"{split}.parquet", columns=["id"])["id"])
            if ids - known.keys():
                known.update(read_paragraphs(self.data_dir / file, ids - known.keys()))
            self._cache[(pair, split)] = load_pair_split(self.pairs[pair], self.data_dir, split, known)
        return self._cache[(pair, split)]


def run_zeroshot(splits: Splits, pairs: list[str], judge_path: str, out: Path, device: str | None,
                 score_batch: int | None = None, prefix: str = "zs", folder: Path = Path("zeroshot")) -> None:
    judge, scored = None, {}  # fingerprint -> output frame, reused by pairs with the same small answers
    for pair in pairs:
        target = out / folder / pair
        if (target / "done.json").exists():
            log.info(f"zeroshot {pair}: done")
            continue
        target.mkdir(parents=True, exist_ok=True)
        for split in ("calib", "test"):
            df = splits.get(pair, split)
            ctx = [passages_state(p) for p in df["passages"]]
            grounded = [{"question": GROUNDED[0].format(claim=CLAIM.format(question=q, answer=a)), "options": GROUNDED[1],
                         "state": c} for q, a, c in zip(df["question"], df["small_pred"], ctx)]
            correct = [{"question": CORRECT[0].format(question=q, answer=a), "options": CORRECT[1], "state": c}
                       for q, a, c in zip(df["question"], df["small_pred"], ctx)]
            key = fingerprint(grounded + correct)
            if key in scored:
                log.info(f"zeroshot {pair}/{split}: same inputs as an earlier pair, scores reused")
            else:
                judge = judge or make_judge(judge_path, device, score_batch=score_batch)
                start = time.perf_counter()
                pg, tg = judge.probabilities(grounded, judge.temperatures.get("grounded", 1.0))
                pc, tc = judge.probabilities(correct)  # a new typed question: no calibrated temperature
                seconds = (time.perf_counter() - start) / len(df)
                scored[key] = pd.DataFrame({"id": df["id"], f"{prefix}_grounded": pg[:, 1], f"{prefix}_grounded_tokens": tg,
                                            f"{prefix}_correct": pc[:, 1], f"{prefix}_correct_tokens": tc})
                log.info(f"zeroshot {pair}/{split}: {len(df)} questions, {seconds * 1000:.1f} ms/question for both judges")
            scored[key].to_parquet(target / f"{split}.parquet")
        (target / "done.json").write_text(json.dumps({"judge": judge_path, "revision": provenance.hub_revision(judge_path),
                                                      "device": judge.device if judge else None}, indent=1))


def run_train(splits: Splits, spec: dict, seed: int, plan: dict, out: Path, device: str | None, save_model: bool,
              micro_batch: int | None = None, score_batch: int | None = None) -> None:
    from sklearn.model_selection import train_test_split

    name = judge_name(spec)
    target = out / "judges" / name / f"seed{seed}"
    if (target / "done.json").exists():
        log.info(f"train {name} seed {seed}: done")
        return
    hp = {**plan["train"], **spec.get("train", {})}  # a judge may override the plan's settings (a decider's LoRA)
    log.info(f"train {name} seed {seed}")
    df = splits.get(spec["train_pair"], "router_train")
    y = escalation_label(metric_view(df, plan.get("metric", "f1")), plan["tau"], spec["label"])
    if spec.get("budget"):  # a label budget: dev included, nested across budgets for one seed (budget_split)
        tr, dev = budget_split(y, spec["budget"], hp["dev_fraction"], seed)
    else:
        tr, dev = train_test_split(np.arange(len(df)), test_size=hp["dev_fraction"], stratify=y, random_state=seed)
    if hp.get("max_questions") and len(tr) > hp["max_questions"]:  # e.g. the CPU plan: a stratified subsample
        tr, _ = train_test_split(tr, train_size=hp["max_questions"], stratify=y[tr], random_state=seed)
    items = route_items(df, spec["inputs"])
    init = spec["init"].format(seed=seed)  # "{seed}": a frozen judge's per-seed folder
    judge = make_judge(init, device, hp["max_length"], score_batch, spec.get("head", "typed"))
    start = time.perf_counter()
    if spec.get("frozen"):  # an already fine-tuned judge (e.g. trained on another dataset), only scored here
        history, tr, dev = [], np.array([], dtype=int), np.array([], dtype=int)
    else:
        history = fine_tune(judge, [items[i] for i in tr], y[tr], [items[i] for i in dev], y[dev], hp["epochs"], hp["lr"],
                            hp["weight_decay"], hp["warmup"], hp["batch_size"], seed, micro_batch or hp.get("micro_batch"),
                            hp.get("min_steps", 0))
    train_seconds = time.perf_counter() - start
    target.mkdir(parents=True, exist_ok=True)
    scoring, scored = {}, {}  # fingerprint -> scores, reused by pairs whose inputs are identical
    for pair in plan["pairs"]:
        for split in ("calib", "test"):
            d = splits.get(pair, split)
            items = route_items(d, spec["inputs"])
            key = fingerprint(items)
            if key not in scored:
                start = time.perf_counter()
                p, tokens = judge.probabilities(items)
                scoring[f"{pair}/{split}"] = (time.perf_counter() - start) / len(d)
                scored[key] = pd.DataFrame({"id": d["id"], "score": p[:, 1], "tokens": tokens})
            else:
                scoring[f"{pair}/{split}"] = "reused"
            scored[key].to_parquet(target / f"{pair}-{split}.parquet")
    if save_model:
        judge.model.save_pretrained(target / "model")
        judge.tokenizer.save_pretrained(target / "model")
    (target / "done.json").write_text(json.dumps({
        "judge": name, **spec, "seed": seed, "tau": plan["tau"], "train": hp, "n_train": len(tr), "n_dev": len(dev),
        "micro_batch": micro_batch or hp.get("micro_batch"),
        "train_label_rate": float(y.mean()), "history": history, "train_seconds": train_seconds,
        "scoring_seconds_per_question": scoring, "init_revision": provenance.hub_revision(init), "init_path": init,
        "device": judge.device}, indent=1))


def judge_columns(out: Path, pair: str, split: str, plan: dict) -> tuple[pd.DataFrame, dict]:
    """Escalation-score columns of every finished judge for one pair and split, plus their specs."""
    cols, specs = {}, {}
    for path, prefix, folder in zero_shot_judges(plan):
        zs = out / folder / pair / f"{split}.parquet"
        if not zs.exists():
            continue
        z = pd.read_parquet(zs)
        if "id" in cols and not (z["id"].to_numpy() == np.asarray(cols["id"])).all():
            raise ValueError(f"{zs} has different questions than the other judges")
        cols["id"] = z["id"]
        for q in ZERO_SHOT:
            c = f"{prefix}_{q}"
            cols[c], cols[f"{c}_tokens"] = z[c].to_numpy(), z[f"{c}_tokens"].to_numpy()
            specs[c] = {"label": "small-fails", "post": True, "zero_shot": True, "init": path}
    for spec in plan["judges"]:
        name = judge_name(spec)
        files = sorted((out / "judges" / name).glob(f"seed*/{pair}-{split}.parquet"))
        if not files:
            continue
        seeds = [pd.read_parquet(f) for f in files]
        cols.setdefault("id", seeds[0]["id"])
        for f, s in zip(files, seeds):
            if not (s["id"].to_numpy() == np.asarray(cols["id"])).all():
                raise ValueError(f"{f} has different questions than the other judges")
            cols[f"judge/{name}#{f.parent.name}"] = s["score"].to_numpy()
        col = f"judge/{name}"
        cols[col] = np.mean([s["score"].to_numpy() for s in seeds], axis=0)  # mean over seeds
        cols[f"{col}_tokens"] = seeds[0]["tokens"].to_numpy()
        specs[col] = {"label": spec["label"], "post": INPUTS[spec["inputs"]], "zero_shot": False,
                      "seeds": [f.parent.name for f in files], **spec}
    return pd.DataFrame(cols), specs


def run_eval(pair: str, plan: dict, out: Path, unit: str, n_boot: int, seed: int, exclude: set[str],
             routers: dict | None = None) -> tuple[dict, dict]:
    """Evaluate one pair; `routers` are its logistic regression routers (trained here if not given)."""
    run = Path(plan["pairs"][pair])
    cfg = tomllib.loads((run / "config.toml").read_text())
    params, models = expert_params(cfg), {e["name"]: e.get("model", "") for e in cfg["experts"]}
    data, specs = {}, {}
    metric = plan.get("metric", "f1")
    for split in ("calib", "test"):
        df = metric_view(pd.read_parquet(run / f"{split}.parquet"), metric)
        cols, specs = judge_columns(out, pair, split, plan)
        if len(cols):
            df = df.merge(cols, on="id", how="left", validate="1:1")
        data[split] = with_cost_unit(df, unit, params, models)
    if exclude:
        data["test"] = data["test"][~data["test"]["id"].isin(exclude)].reset_index(drop=True)
    test, calib = data["test"], data["calib"]
    if test["ev_sufficiency"].notna().all() and calib["ev_sufficiency"].notna().all():
        for df in (test, calib):
            df["zs_sufficient"] = 1 - df["ev_sufficiency"]  # nano-jev's own `sufficient` question, P(no)
        specs["zs_sufficient"] = {"label": "small-fails", "post": False, "zero_shot": True}
    missing = [c for c in specs if test[c].isna().any() or calib[c].isna().any()]
    if missing:
        raise ValueError(f"judge scores missing for some questions: {missing}")

    lr_routers = routers or train_routers(metric_view(pd.read_parquet(run / "router_train.parquet"), metric),
                                          plan["tau"], seed)
    routers = dict(lr_routers)
    # a trained judge's score is the mean over its seeds, so it runs every seed's model; each seed is also a router of
    # its own (one model, one pass), which is what a single deployed judge costs
    seed_cols = {col: sorted(c for c in test.columns if c.startswith(f"{col}#")) for col in specs}
    for col, spec in specs.items():
        routers[col] = {"score_column": col, "label": spec["label"], "post": spec["post"], "tuning": None}
        for c in seed_cols[col]:
            routers[c] = {**routers[col], "score_column": c}
    policy_costs = {}  # each judge's own per-question cost, in the curves' unit, for the with-cost bootstrap
    for col, spec in specs.items():
        if f"{col}_tokens" in test and "init" in spec:
            gflops = 2 * judge_params_b(spec["init"]) * test[f"{col}_tokens"].to_numpy(dtype=float)
            one = judge_cost_in(unit, gflops)
            policy_costs[f"router:{col}"] = one * max(len(seed_cols[col]), 1)
            policy_costs |= {f"router:{c}": one for c in seed_cols[col]}
    report, curves = evaluate(test, calib, routers, plan["tau"], plan.get("max_drop", 0.01), n_boot, seed,
                              references=LR_REFERENCES + tuple(f"router:{c}" for c in plan.get("paired_references", [])),
                              policy_costs=policy_costs)

    cmin, cmax = test["small_cost"].mean(), test["large_cost"].mean()
    judges = {}
    for col, spec in specs.items():
        tokens = test.get(f"{col}_tokens")
        params = judge_params_b(spec["init"]) if "init" in spec else 0.0
        passes = max(len(seed_cols[col]), 1)
        per_call = 2 * params * tokens.to_numpy(dtype=float) if tokens is not None else np.zeros(1)  # sufficiency: paid
        gflops = float(per_call.mean()) * passes
        judge_cost = float(np.mean(judge_cost_in(unit, per_call))) * passes  # in the curves' unit
        pts = [p for p in report_curve(curves, f"router:{col}")]
        entry = {**spec, "aiq": report["aiq"][f"router:{col}"], "judge_gflops": gflops, "judge_params_b": params,
                 "judge_passes": passes, "judge_cost": judge_cost,
                 "aiq_with_judge_cost": aiq([{**p, "cost": p["cost"] + judge_cost} for p in pts], cmin, cmax)}
        if seed_cols[col]:
            per_seed = [report["aiq"][f"router:{c}"] for c in seed_cols[col]]
            with_cost = [aiq([{**p, "cost": p["cost"] + judge_cost / passes} for p in report_curve(curves, f"router:{c}")],
                             cmin, cmax) for c in seed_cols[col]]
            entry |= {"aiq_per_seed": per_seed, "aiq_seed_mean": float(np.mean(per_seed)),
                      "aiq_seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                      "aiq_per_seed_with_cost": with_cost, "aiq_seed_mean_with_cost": float(np.mean(with_cost))}
        judges[col] = entry
    report = {"pair": pair, "run": str(run), "metric": metric, "cost_unit": unit, "excluded": len(exclude), **report,
              "judges": judges}

    target = out / plan.get("eval_dir", "eval") / pair
    target.mkdir(parents=True, exist_ok=True)
    suffix = "-clean" if exclude else ""
    (target / f"report{suffix}.json").write_text(json.dumps(report, indent=1, default=str))
    curves.to_csv(target / f"curves{suffix}.csv", index=False)
    if not exclude:  # the merged judge scores, so the evaluation can be redone on CPU from results/
        for split, df in data.items():
            df[["id", *[c for c in df.columns if c.startswith(("judge/", "zs_"))]]].to_parquet(target / f"scores-{split}.parquet")
    provenance.record(target, "judge-eval", pair=pair, excluded=len(exclude), **provenance.environment())
    log.info(f"eval {pair}{suffix}: AIQ entropy {report['aiq']['small-entropy (post)']:.4f}")
    for col, e in sorted(judges.items(), key=lambda kv: -kv[1]["aiq"]):
        spread = f" (seeds {e['aiq_seed_mean']:.4f} ± {e['aiq_seed_sd']:.4f})" if "aiq_seed_mean" in e else ""
        log.info(f"  {col:<45} AIQ {e['aiq']:.4f}{spread} | +judge cost {e['aiq_with_judge_cost']:.4f} "
                 f"({e['judge_gflops']:.0f} GFLOPs)")
    return report, lr_routers


def report_curve(curves: pd.DataFrame, policy: str) -> list[dict]:
    return curves[curves["policy"] == policy].drop(columns="policy").to_dict("records")


# ---------------------------------------------------------------------------------------------
# CLI


def expand(spec: dict) -> list[dict]:
    """One [[judges]] entry -> one spec per combination of its list-valued GRID_KEYS (a label-budget grid)."""
    grid = [[(k, v) for v in spec[k]] for k in GRID_KEYS if isinstance(spec.get(k), list)]
    out = [dict(spec)]
    for options in grid:
        out = [{**s, k: v} for s in out for k, v in options]
    return out


def load_plan(path: Path) -> dict:
    plan = tomllib.loads(path.read_text())
    plan["pairs"] = {k: Path(v) for k, v in plan["pairs"].items()}
    plan["judges"] = [s for spec in plan["judges"] for s in expand(spec)]
    names = plan.get("init_names", {})  # local checkpoint folder -> short name (judge_name)
    for spec in plan["judges"]:
        if spec["init"] in names:
            spec["init_name"] = names[spec["init"]]
        if spec["inputs"] not in INPUTS:
            raise ValueError(f"unknown inputs {spec['inputs']!r}; expected one of {list(INPUTS)}")
        if spec.get("head", "typed") not in HEADS:
            raise ValueError(f"unknown head {spec['head']!r}; expected one of {HEADS}")
        if spec["train_pair"] not in plan["pairs"]:
            raise ValueError(f"judge {spec} trains on unknown pair {spec['train_pair']!r}")
    if len({judge_name(s) for s in plan["judges"]}) != len(plan["judges"]):
        raise ValueError("two judges of the plan share a name: give local inits an [init_names] entry")
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["zeroshot", "train", "eval", "all"])
    parser.add_argument("--plan", type=Path, default=Path("configs/pair.toml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/hotpotqa"))
    parser.add_argument("--out", type=Path, default=Path("runs/pair"))
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--only", action="append", default=[], help="train: judge names to run (default all); repeatable")
    parser.add_argument("--match", action="append", default=[],
                        help="train: run only judges whose name contains one of these substrings; repeatable")
    parser.add_argument("--save-model", action="store_true", help="train: also save the fine-tuned weights")
    parser.add_argument("--micro-batch", type=int,
                        help="train: questions per forward/backward pass (memory only; default: whole batch on GPU, 2 on CPU)")
    parser.add_argument("--score-batch", type=int, help="questions per inference batch (default 64 on GPU, 16 on CPU)")
    parser.add_argument("--cost", choices=JUDGE_COST_UNITS, help="cost unit (default: the plan's `cost`, else flops)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0, help="eval: seed for routers' CV and the bootstrap")
    parser.add_argument("--exclude", type=Path, action="append", default=[],
                        help="eval: test ids to drop (e.g. nanojev_validation_ids.txt); writes report-clean.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = load_plan(args.plan)
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.plan, args.out / "plan.toml")
    provenance.record(args.out, f"judge-{args.step}", plan=str(args.plan), **provenance.environment())
    splits = Splits(plan["pairs"], args.data_dir, plan.get("split_data"))
    args.cost = args.cost or plan.get("cost", "flops")
    if args.step in ("zeroshot", "all"):
        for path, prefix, folder in zero_shot_judges(plan):
            run_zeroshot(splits, list(plan["pairs"]), path, args.out, args.device, args.score_batch, prefix, folder)
    if args.step in ("train", "all"):
        if unknown := set(args.only) - {judge_name(spec) for spec in plan["judges"]}:
            raise SystemExit(f"--only names no judge of the plan: {sorted(unknown)}; "
                             f"its judges are {sorted(judge_name(s) for s in plan['judges'])}")
        for spec in plan["judges"]:
            if args.only and judge_name(spec) not in args.only:
                continue
            if args.match and not any(m in judge_name(spec) for m in args.match):
                continue
            for seed in spec.get("seeds", plan["seeds"]):
                run_train(splits, spec, seed, plan, args.out, args.device, args.save_model, args.micro_batch,
                          args.score_batch)
    if args.step in ("eval", "all"):
        del splits  # the passages are not needed to evaluate; free them before the routers' CV workers start
        gc.collect()
        exclude = set()
        for path in args.exclude:
            exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
        for pair in plan["pairs"]:
            _, lr = run_eval(pair, plan, args.out, args.cost, args.bootstrap, args.seed, set())
            if exclude:
                run_eval(pair, plan, args.out, args.cost, args.bootstrap, args.seed, exclude, lr)


if __name__ == "__main__":
    main()
