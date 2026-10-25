"""A Strands Decider (github.com/strands-labs/strands-decider, Apache-2.0) as a typed-decision judge.

A Strands Decider is an LLM torso (Qwen3.5-2B, or Gemma 4 E4B) whose text head is replaced by a pointer head: each option
of a typed question is scored from that option's own hidden state against the `<answer>` position. The published judges
fine-tune its LoRA (rank 16 on the 2B, 48 on the E4B) and the head; the folder they are saved in holds the LoRA, the
head and the tokenizer, and loads the base torso from the Hugging Face Hub on first use.

An item `{"question", "options", "state"}` becomes a decider `choice` question (instructions = the question, one option
per label, no descriptions) on that state, rendered and truncated exactly as the package's own engine does. The logits
carry the checkpoint's calibrated `choice` temperature.
"""

import logging
import re

import numpy as np

log = logging.getLogger(__name__)

# Billions of parameters per forward pass, for the judge's cost (2 x params x tokens GFLOPs). Gemma 4's
# E-models and the MoE count their effective (active) parameters, not their stored ones.
PARAMS_B = {"2B": 2.0, "E2B": 2.0, "E4B": 4.0, "12B": 12.0, "26B-A4B": 4.0}


def is_decider(path: str) -> bool:
    return "strands-decider" in str(path).lower()


def params_b(path: str) -> float:
    """The judge's parameters in billions, from the checkpoint name (strands-decider-<size>-...)."""
    m = re.search(r"strands-decider-(26B-A4B|E\d+B|\d+(?:\.\d+)?B)", str(path), re.IGNORECASE)
    if not m:
        raise ValueError(f"cannot read the model size from {path!r}")
    size = m.group(1).upper()
    return PARAMS_B[size] if size in PARAMS_B else float(size.rstrip("B"))


class DeciderJudge:
    """A Strands Decider judge: `probabilities(items)` per option, and the tokens each item read."""

    def __init__(self, path: str, device: str | None = None, max_length: int | None = None,
                 score_batch: int | None = None, size_b: float | None = None):
        import torch
        from strands_decider.infer import EngineConfig, SystemOneEngine
        from strands_decider.modeling import StrandsDeciderModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path
        self.engine = SystemOneEngine(StrandsDeciderModel.load(path),
                                      EngineConfig(device=self.device, use_prefix_cache=False))
        self.model = self.engine.model
        self.tokenizer = self.model.tokenizer
        cfg = self.model.config
        self.choice_temperature = float((cfg.temperature_by_kind or {}).get("choice", cfg.temperature))
        self.window = cfg.max_length  # the checkpoint's own window
        self.temperatures = {}  # no calibration beyond the checkpoint's own
        self.score_batch = score_batch or (16 if self.device != "cpu" else 2)
        self.params_b = size_b if size_b is not None else params_b(path)  # a published folder's name has no size
        for name, p in self.model.named_parameters():  # the LoRA loads frozen; fine-tuning trains it and the head
            p.requires_grad = "lora_" in name or name.startswith("head.")

    def _row(self, item: dict) -> tuple[list[int], list[int]]:
        """Token ids of one item's prompt and the positions of its options (pointer readout)."""
        from strands_decider.infer import _option_token_index
        from strands_decider.prompting import render_question, render_state
        from strands_decider.schema import ChoiceQuestion

        rq = render_question(ChoiceQuestion(instructions=item["question"], criteria={o: "" for o in item["options"]}))
        fitted = self.engine._fit(render_state(item["state"]), [rq.text])
        if len(fitted) == 3:  # the pinned main commit returns the offsets (the server is concurrent)
            s, q, offsets = fitted
        else:  # PyPI 0.1.0 keeps them on the engine
            (s, q), offsets = fitted, self.engine._last_offsets
        idx = _option_token_index(offsets[0], rq.option_spans, 0)
        return s + q[0], [len(s) + i for i in idx]

    def encode(self, items: list[dict]):
        torch = self.torch
        rows = [self._row(it) for it in items]
        width = max(len(ids) for ids, _ in rows)
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids = torch.tensor([r + [pad] * (width - len(r)) for r, _ in rows], device=self.device)
        mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r, _ in rows], device=self.device)
        opt_idx = torch.tensor([o for _, o in rows], dtype=torch.long, device=self.device)
        return ids, mask, opt_idx

    def logits(self, items: list[dict]):
        """[items, options] calibrated logits (all items must have the same number of options)."""
        torch = self.torch
        k = len(items[0]["options"])
        ids, mask, opt_idx = self.encode(items)
        out = self.model(input_ids=ids, attention_mask=mask, n_slots=torch.full((len(items),), k, device=self.device),
                         temperature=self.choice_temperature, opt_idx=opt_idx)
        return out["logits"][:, :k].float()

    def probabilities(self, items: list[dict], temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """[items, options] probabilities and the tokens each item processed (its cost)."""
        torch = self.torch
        self.model.eval()
        probs, tokens = [], []
        with torch.no_grad():
            for i in range(0, len(items), self.score_batch):
                batch = items[i: i + self.score_batch]
                tokens += [len(self._row(it)[0]) for it in batch]
                probs.append(torch.softmax(self.logits(batch) / temperature, dim=1).cpu().numpy())
        return np.concatenate(probs), np.asarray(tokens)


SMOKE_ITEMS = [
    {"question": "Which model should answer this question: Who directed the film Forbidden Quest?",
     "options": ("small model", "large model"),
     "state": "Forbidden Quest is a 2006 South Korean film written and directed by Kim Dae-woo."},
    {"question": "Is the proposed answer correct? Question: What is the capital of France? Proposed answer: Lyon",
     "options": ("yes", "no"), "state": "Paris is the capital and largest city of France."},
    {"question": "Which answer option should be used for this question: Which band released Oh, Inverted World?",
     "options": ("small, one answer", "small, ten fused", "large, one answer", "small, twenty fused", "small, forty fused"),
     "state": "Oh, Inverted World is the debut album by the American band The Shins, released in 2001."},
]


def smoke(path: str, device: str | None = None, batch: int = 3) -> dict:
    """The wrapper against the package's own engine on SMOKE_ITEMS: probabilities must agree and sum to 1."""
    from strands_decider.schema import ChoiceQuestion

    j = DeciderJudge(path, device, score_batch=batch)
    # one item at a time (the items have different option counts), then a padded batch of the two 2-option ones
    singles = [j.probabilities([it]) for it in SMOKE_ITEMS]
    ours, tokens = [p[0] for p, _ in singles], np.array([t[0] for _, t in singles])
    batched, _ = j.probabilities(SMOKE_ITEMS[:2])
    batch_gap = float(np.abs(batched - np.stack(ours[:2])).max())
    theirs = []
    for it in SMOKE_ITEMS:
        ans = j.engine.ask(it["state"], {"q": ChoiceQuestion(instructions=it["question"],
                                                             criteria={o: "" for o in it["options"]})}).answers["q"]
        theirs.append([ans.probabilities[o] for o in it["options"]])
    gap = max(float(np.abs(np.asarray(o) - np.asarray(t)).max()) for o, t in zip(ours, theirs))
    report = {"model": path, "device": j.device, "params_b": j.params_b, "choice_temperature": j.choice_temperature,
              "window": j.window, "tokens": tokens.tolist(), "max_abs_gap_vs_engine": gap,
              "max_abs_gap_batched_vs_single": batch_gap,
              "sums": [float(p.sum()) for p in ours], "ours": [p.round(4).tolist() for p in ours], "engine": theirs}
    # the engine rounds to 4 decimals; batching pads, so allow bf16-level drift
    report["passed"] = bool(gap < 0.02 and batch_gap < 0.02 and all(abs(s - 1) < 1e-4 for s in report["sums"]))
    return report


def download(path: str) -> list[str]:
    """Download a decider checkpoint and the base torso it loads on first use (login node: no model is loaded)."""
    import json

    from huggingface_hub import snapshot_download

    local = snapshot_download(path)
    from pathlib import Path

    cfg = next((Path(local) / n for n in ("strands_decider_config.json", "hobson_config.json") if (Path(local) / n).exists()), None)
    if cfg is None:
        raise FileNotFoundError(f"{path}: no strands_decider_config.json")
    base = json.loads(cfg.read_text())["base_model"]
    snapshot_download(base)
    return [path, base]


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Check the DeciderJudge wrapper against strands_decider's engine.")
    parser.add_argument("models", nargs="+", help="Strands Decider checkpoints (Hub ids or paths)")
    parser.add_argument("--device")
    parser.add_argument("--out", help="write the reports as JSON here")
    parser.add_argument("--download-only", action="store_true",
                        help="only download each checkpoint and its base torso (for the login node)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.download_only:
        for m in args.models:
            log.info(f"downloaded {download(m)}")
        return
    reports = [smoke(m, args.device) for m in args.models]
    text = json.dumps(reports, indent=1)
    log.info(text)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text)
    if not all(r["passed"] for r in reports):
        raise SystemExit("decider smoke test FAILED: the wrapper disagrees with the engine")


if __name__ == "__main__":
    main()
