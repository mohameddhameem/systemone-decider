"""Use a published decider judge.

Pair setting (one decision, before or after the small model answers):

    from systemone_decider.route import PairRouter
    router = PairRouter.from_pretrained("<org>/systemone-decider-2b-qp")       # seed 0; subfolder="seed1" for another
    p = router.p_large(question, passages)                                     # P(large model)
    use_large = router.route(question, passages) == "large"                    # p >= the published threshold

Menu setting (two decisions, weak first):

    from systemone_decider.route import MenuRouter
    router = MenuRouter.from_pretrained("<org>/systemone-decider-e4b-qpa-menu-a")
    if router.retrieval(question, weak_selection) == "strong": ...            # run the strong retriever instead
    router.answer(question, selection, branch, weak_answer=...)                # "weak" or "strong", under that branch

Each folder's `systemone.json` names its setting, input set and thresholds.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from systemone_decider.decider import DeciderJudge
from systemone_decider.items import answer_item, check_inputs, pair_item, retrieval_item

CARD = "systemone.json"


def resolve(path: str, subfolder: str | None = None) -> Path:
    """A local folder as is; otherwise a Hub id with an optional "@revision". Only `subfolder` (a seed) is downloaded."""
    if Path(path).is_dir():
        return Path(path) / subfolder if subfolder else Path(path)
    from huggingface_hub import snapshot_download

    repo_id, _, revision = path.partition("@")
    patterns = [f"{subfolder}/*"] if subfolder else ["*.json", "*.jinja", "*.pt", "lora/*"]
    local = Path(snapshot_download(repo_id, revision=revision or None, allow_patterns=patterns))
    return local / subfolder if subfolder else local


def read_card(folder: Path, setting: str) -> dict:
    card = json.loads((folder / CARD).read_text()) if (folder / CARD).exists() else {}
    if card.get("setting", setting) != setting:
        raise ValueError(f"{folder}: a {card['setting']} judge, not a {setting} one")
    return card


@dataclass
class PairRouter:
    judge: DeciderJudge
    inputs: str
    threshold: float = 0.5

    @classmethod
    def from_pretrained(cls, path: str, subfolder: str | None = None, device: str | None = None,
                        threshold: float | None = None) -> "PairRouter":
        folder = resolve(path, subfolder)
        card = read_card(folder, "pair")
        check_inputs(card.get("inputs", ""))
        judge = DeciderJudge(str(folder), device, size_b=card.get("params_b"))
        return cls(judge, card["inputs"], threshold if threshold is not None else card.get("threshold", 0.5))

    def p_large_batch(self, rows: list[dict]) -> list[float]:
        """rows: {"question", "passages", and "answer" for qpa}."""
        items = [pair_item(r["question"], r["passages"], self.inputs, r.get("answer")) for r in rows]
        probs, _ = self.judge.probabilities(items)
        return probs[:, 1].tolist()  # option 1 is "large model"

    def p_large(self, question: str, passages: list[str], answer: str | None = None) -> float:
        return self.p_large_batch([{"question": question, "passages": passages, "answer": answer}])[0]

    def route(self, question: str, passages: list[str], answer: str | None = None) -> str:
        return "large" if self.p_large(question, passages, answer) >= self.threshold else "small"


@dataclass
class MenuRouter:
    """Weak first: keep the weak retrieval if P(it suffices) >= t[0]; then, under the retrieval taken, keep the weak
    answer if P(it suffices) >= t[1] (weak retrieval) or t[2] (strong retrieval)."""

    judge: DeciderJudge
    inputs: str
    thresholds: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])

    @classmethod
    def from_pretrained(cls, path: str, subfolder: str | None = None, device: str | None = None,
                        rule: str = "near_strong", thresholds: list[float] | None = None) -> "MenuRouter":
        """`rule` picks the published thresholds: near_strong (within 1% of always-strong's accuracy), peak or parity."""
        folder = resolve(path, subfolder)
        card = read_card(folder, "menu")
        check_inputs(card.get("inputs", ""))
        rules = card.get("thresholds", {})
        if thresholds is None and rule not in rules:
            raise ValueError(f"{folder}: systemone.json has no {rule!r} thresholds (it has {sorted(rules)})")
        judge = DeciderJudge(str(folder), device, size_b=card.get("params_b"))
        return cls(judge, card["inputs"], thresholds or rules[rule])

    def p_weak_retrieval(self, question: str, weak_selection: str) -> float:
        probs, _ = self.judge.probabilities([retrieval_item(question, weak_selection)])
        return float(probs[0, 0])  # option 0 is "yes": the weak stage suffices

    def p_weak_answer(self, question: str, selection: str, weak_answer: str | None = None) -> float:
        probs, _ = self.judge.probabilities([answer_item(question, selection, self.inputs, weak_answer)])
        return float(probs[0, 0])

    def retrieval(self, question: str, weak_selection: str) -> str:
        return "weak" if self.p_weak_retrieval(question, weak_selection) >= self.thresholds[0] else "strong"

    def answer(self, question: str, selection: str, branch: str, weak_answer: str | None = None) -> str:
        """`branch` is the retrieval taken ("weak" or "strong"), `selection` its selected passages."""
        t = {"weak": self.thresholds[1], "strong": self.thresholds[2]}[branch]
        return "weak" if self.p_weak_answer(question, selection, weak_answer) >= t else "strong"
