---
license: apache-2.0
base_model: {base}
datasets:
  - hotpotqa/hotpot_qa
language:
  - en
tags:
  - llm-routing
  - strands-decider
  - lora
---

# SystemOne-Decider {size_name} {inputs}

A router between a small and a large language model for multi-hop questions: a [Strands Decider](https://huggingface.co/{base})
({size_name}) with its LoRA and pointer head fine-tuned to answer one typed question, *"Which model should answer this
question?"* Given {inputs_text}, it returns P(large model). Send the question to the large model when that probability is
at or above `threshold` in `systemone.json`, which is chosen on validation.

- **Small model:** Qwen2.5-1.5B-Instruct on nano-jev's top 2 HotpotQA paragraphs. **Large model:** Qwen2.5-7B-Instruct
  on all 10.
- **Label:** the large model is needed: the small one is wrong and the large one is right (lenient scoring).
- **Seeds:** seed 0 is in the root folder; seeds 1 and 2 are in `seed1/` and `seed2/`.
- **Base:** [{base}](https://huggingface.co/{base}), on [{torso}](https://huggingface.co/{torso}), which loads on first
  use.
- **Code:** [github.com/{github}](https://github.com/{github}) (`systemone_decider.route.PairRouter`).

## Results

The SLR split of HotpotQA's dev set: 4,814 training, 1,110 validation and {n_test} test questions. Cost is the AgentTTS
Table 5 price per answering call (Qwen2.5-1.5B 0.5, Qwen2.5-7B 10); the judge's own call is priced the same way. AIQ is
the area under the accuracy-vs-cost curve. Intervals are 95% paired bootstrap over the test questions.

| | AIQ | With its own cost |
|---|---|---|
| Seed 0 (root folder) | {seed0} {seed0_ci} | {seed0_cost} {seed0_cost_ci}, {per_call} a call |
| Single seeds, mean ± sd | {seeds} | {seeds_cost} |
| Three-seed ensemble (mean P) | {ensemble} | {ensemble_cost} (three calls) |
| Three-seed ensemble, the {n_clean} questions nano-jev never saw | {ensemble_clean} | |
| Entropy threshold on the small model | {entropy} | |
| {lr_name} | {lr_aiq} | |
| Best logistic-regression router at the same decision point | {lr_best} | |
| Random | {random} | |
| Oracle | {oracle} | |

Seed 0 minus each baseline:

| | Before its own cost | With it |
|---|---|---|
| Entropy threshold | {d_entropy} | {d_entropy_cost} |
| {lr_name} | {d_lr} | {d_lr_cost} |

At its published threshold ({threshold}), seed 0 on the test set: {op}. Always-large: {op_large}; always-small:
{op_small}. The threshold keeps accuracy within {max_drop} of always-large's on validation; on test, seed 0 lands
{op_drop} below it, at {op_share} of always-large's cost with the judge's call included.

## Use

```
pip install git+https://github.com/{github}
```

```python
from systemone_decider.route import PairRouter

# seed 0; subfolder="seed1" or "seed2" for the others
router = PairRouter.from_pretrained("{repo}")
# passages: ["title: text", ...], the small model's top 2
p = router.p_large(question, passages{answer_arg})
use_large = router.route(question, passages{answer_arg}) == "large"
```

The judge runs the Strands Decider's {size_name} torso, so a GPU is needed for useful speed.

## Training

The Strands Decider's LoRA (rank {lora_rank}) and pointer head, fine-tuned with grouped-softmax cross-entropy on the 4,814
training questions: lr {lr}, batch {batch_size}, up to {epochs} epochs, with the epoch of lowest NLL on a
{dev_fraction:.0%} hold-out kept. Training started from `{base}` at revision `{base_revision}`.
`training.json` has the full record and the loss per epoch.

## Limits

Trained and tested on HotpotQA with one model pair. The threshold is chosen on validation for this pair; re-pick it on
your own validation data before using another pair or dataset.

## Licence

Apache-2.0, as the Strands Decider and {torso} are. HotpotQA, the training data, is CC BY-SA 4.0. Whether its
share-alike terms apply to weights trained on it is not settled, and this card does not claim that they do not.
