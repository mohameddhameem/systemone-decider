# SystemOne-Decider

Routers that decide, question by question, whether a weak language model is enough or a strong one should run. Each is
a [Strands Decider](https://github.com/strands-labs/strands-decider) (a 2B or a 4B LLM torso with a pointer head) whose
LoRA and head are fine-tuned to answer typed questions about the question, the passages and, optionally, the weak
model's answer.

Two settings, each with a 2B and an E4B judge, and each judge reading passages only (`qp`) or passages and the weak
answer (`qpa`): eight Hugging Face repos, each with three seeds.

| Setting | Decisions | Weak / strong | Repos |
|---|---|---|---|
| Pair | one: which model answers? | Qwen2.5-1.5B-Instruct on nano-jev's top 2 passages / Qwen2.5-7B-Instruct on all 10 | `mohameddhameem/systemone-decider-{2b,e4b}-{qp,qpa}` |
| Menu | two, weak first: retrieval, then the answer | retrieval Qwen2.5-7B / 72B-Instruct; answer Llama-3.2-3B / 3.1-70B-Instruct | `mohameddhameem/systemone-decider-{2b,e4b}-{qp,qpa}-menu-a` |

Both settings are trained and tested on the SLR split of HotpotQA's dev set: 4,814 training, 1,110 validation and 1,481
test questions, lenient scoring, AgentTTS Table 5 prices. Each model card has its results, against the small model's
entropy threshold and logistic-regression routers (pair) or the SLR setup's own fine-tuned routers (menu), with and
without the judge's own price.

## Usage

```
pip install git+https://github.com/mohameddhameem/systemone-decider
```

```python
from systemone_decider.route import PairRouter, MenuRouter

router = PairRouter.from_pretrained("mohameddhameem/systemone-decider-2b-qpa")
p = router.p_large(question, passages, answer=small_answer)   # P(large model)
use_large = router.route(question, passages, answer=small_answer) == "large"

menu = MenuRouter.from_pretrained("mohameddhameem/systemone-decider-e4b-qpa-menu-a")
branch = menu.retrieval(question, weak_selection)              # "weak" or "strong" retrieval
answerer = menu.answer(question, selection, branch, weak_answer=weak_answer)
```

`passages` are "title: text" paragraphs in the order the weak model read them. A menu selection is the retriever's
output, uncut: its "[k] title: text" references, after any "Query: … Selected references:" lines; the judge keeps what
follows the first "Selected references:", as training did. Each Hub folder's `systemone.json` names its setting, input set and thresholds; seeds 1
and 2 load with `subfolder="seed1"` or `"seed2"`. A menu folder holds three threshold rules, all chosen on validation:
`MenuRouter` uses `near_strong` (the cheapest point within 1% of always-strong's accuracy) unless given `rule="peak"`
or `rule="parity"`, the SLR setup's own.

The judges run the Strands Decider's torso (Qwen3.5-2B or Gemma 4 E4B), which loads from the Hub on first use; a GPU is
needed for useful speed. `strands-decider` is pinned to the commit the judges were trained with.

## Retraining

The published judges come from the plans in `configs/`, with three seeds each. Training needs a Linux GPU (the E4B
judge used an NVIDIA L40S at micro-batch 2).

```
uv sync --extra data
uv run prepare-hotpotqa                     # data/hotpotqa/distractor_{train,validation}.parquet
uv run slr fetch && uv run slr verify       # the SLR grid, split and labels, sha256-checked, from the STAIR repo
uv run menu-judge train --judge decider-2b --save-model && uv run menu-judge eval     # the menu setting
uv run judge train --plan configs/pair.toml --data-dir data/hotpotqa --out runs/pair --save-model   # the pair setting
```

- **The menu setting** rebuilds entirely from public files: the SLR setup's grid has every question's four paths
  answered.
- **The pair setting** also needs the two models' cached answers in `runs/pair-a` (router_train, calib and test
  parquet). This repository does not generate them.
- **The clean test set** leaves out the dev questions nano-jev saw: `stair_common`'s `hotpotqa/nanojev_v1_seen.txt`.

## Layout

| Path | Contents |
|---|---|
| `systemone_decider/` | `items.py` (the typed questions and states, as in training), `decider.py` (the judge), `route.py` (inference); `judge.py` (pair: train, score, evaluate), `menu.py` (menu: train, evaluate), `slr.py` (the SLR grid), `evaluate_routing.py`, `router.py`, `bootstrap.py` and helpers |
| `configs/` | `pair.toml`, `menu.toml`, `slr.toml` (the training and evaluation plans); `release.toml` (the cards) |
| `model-cards/` | the Hub card templates, one per setting |
| `scripts/push_hf.py` | stages the Hub repos and renders the cards from the evaluation reports |
| `scripts/check_scores.py` | loads every staged folder as a user would and checks it reproduces the saved test scores |
| `tests/` | `uv run pytest` (no model download) |

## Licence

Apache-2.0, as the Strands Decider checkpoints and their bases (Qwen3.5-2B, Gemma 4 E4B) are; see `NOTICE`. The training
data is HotpotQA, which is CC BY-SA 4.0. Whether share-alike terms apply to weights trained on that data is not settled,
and this project does not claim that they do not.
