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

# SystemOne-Decider {size_name} {inputs}, two-stage menu

A router for a two-stage question-answering pipeline, deciding weak first at each stage. It is a
[Strands Decider](https://huggingface.co/{base}) ({size_name}) with its LoRA and pointer head fine-tuned to answer two
yes/no questions; P(yes) is the probability that the weak stage suffices.

- **Retrieval:** Qwen2.5-7B-Instruct selects passages (weak) or Qwen2.5-72B-Instruct does (strong). The judge reads the
  weak selection: *"Do these passages hold what is needed to answer this question?"*
- **Answer:** Llama-3.2-3B-Instruct (weak) or Llama-3.1-70B-Instruct (strong), under the retrieval taken. `qp` reads that
  branch's passages (*"Will a small model answer this question correctly from these passages?"*); `qpa` also reads the
  weak answer (*"Is the proposed answer correct?"*).
- **Labels:** the per-stage targets of the Stage-Level Routing (SLR) setup ({credit}).
- **Seeds:** seed 0 is in the root folder; seeds 1 and 2 are in `seed1/` and `seed2/`.
- **Thresholds:** `systemone.json` holds three operating rules, each chosen on validation over the SLR setup's grid
  (steps of 0.05 per stage). `near_strong`, the default, is the cheapest point within {max_drop} of always-strong's
  accuracy. The SLR setup's own two are `peak`, the most accurate point, and `parity`, the cheapest point at or above
  always-strong's accuracy; for most seeds both are always-strong itself. So is `near_strong` for a seed whose scores
  crowd above 0.95.
- **Code:** [github.com/{github}](https://github.com/{github}) (`systemone_decider.route.MenuRouter`).

## Results

The SLR split of HotpotQA's dev set: 4,814 training, 1,110 validation and {n_test} test questions. Prices are AgentTTS
Table 5's, weak first: 80 (both stages weak) to 1,009 (both strong after both weak calls ran). AIQ is the area under
the accuracy-vs-price curve over the validation-chosen thresholds; the judge's own calls are priced the same way.

| | AIQ | With its own price |
|---|---|---|
| Seed 0 (root folder) | {seed0} {seed0_ci} | {seed0_price} |
| Single seeds, mean ± sd | {seeds} | {seeds_price} |
| Three-seed ensemble (mean P) | {ensemble} | {ensemble_price} (three calls a decision) |
| Three-seed ensemble, the questions nano-jev never saw | {ensemble_clean} | |
| SLR SFT router (label), Qwen2.5-1.5B LoRA | {his_label} | not charged |
| SLR SFT router (label) + reward | {his_reward} | not charged |
| The four fixed paths | {fixed} | |

Seed 0 and the three-seed ensemble, minus each SLR router. Single seeds vary, so read both:

| | Before its own price | With it |
|---|---|---|
| Seed 0 minus SLR SFT (label) | {d_label} | {d_label_price} |
| Seed 0 minus SLR SFT (label) + reward | {d_reward} | {d_reward_price} |
| Ensemble minus SLR SFT (label) | {e_label} | {e_label_price} |
| Ensemble minus SLR SFT (label) + reward | {e_reward} | {e_reward_price} |

Operating points on the test set (thresholds from validation):
- **Seed 0, near_strong (the default):** {near}: {near_vs_strong}.
- **Seed 0, peak:** {peak}. **Parity:** {parity}.
- **Three-seed ensemble, peak:** {ens_peak}.
- **SLR SFT (label), peak:** {his_peak}.
- **Always strong, no router:** {strong}.

## Use

```
pip install git+https://github.com/{github}
```

```python
from systemone_decider.route import MenuRouter

# seed 0, the near_strong thresholds; rule="peak" or rule="parity" for the SLR setup's own
router = MenuRouter.from_pretrained("{repo}")
branch = router.retrieval(question, weak_selection)          # "weak" or "strong": run the strong retriever if "strong"
answerer = router.answer(question, selection, branch{weak_answer_arg})   # under that branch's selected passages
```

The judge runs the Strands Decider's {size_name} torso, so a GPU is needed for useful speed.

## Training

One model learns both stages' questions: the Strands Decider's LoRA (rank {lora_rank}) and pointer head, fine-tuned with
grouped-softmax cross-entropy on the 4,814 training questions (three items each): lr {lr}, batch {batch_size}, up to
{epochs} epochs, with the epoch of lowest NLL on a {dev_fraction:.0%} hold-out of questions kept. Training started from
`{base}` at revision `{base_revision}`. `training.json` has the full record.

## Limits

Trained and tested on HotpotQA with one menu of models and one retrieval prompt. Its states are the retrievers' selected
references, "[k] title: text" per line. Re-pick the thresholds on your own validation data before using other models. The
{max_drop} of `near_strong` holds on validation, not necessarily on new questions: on test, seed 0 lands
{near_vs_strong}.

## Licence

Apache-2.0, as the Strands Decider and {torso} are. HotpotQA is CC BY-SA 4.0. Whether its share-alike terms apply to
weights trained on it is not settled, and this card does not claim that they do not.
