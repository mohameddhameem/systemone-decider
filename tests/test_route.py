import json

import numpy as np
import pytest

from systemone_decider import items, route


class FakeJudge:
    """Scores without a model: P(option 0) is the next value of `p0`."""

    def __init__(self, path, device=None, size_b=None):
        self.path, self.size_b, self.seen = path, size_b, []

    def probabilities(self, its):
        self.seen += its
        p0 = np.array([FakeJudge.p0.pop(0) for _ in its])
        return np.stack([p0, 1 - p0], axis=1), np.full(len(its), 10)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setattr(route, "DeciderJudge", FakeJudge)

    def make(card):
        (tmp_path / "systemone.json").write_text(json.dumps(card))
        return str(tmp_path)
    return make


def test_pair_items_are_the_training_items():
    it = items.pair_item("Who?", ["A: a", "B: b"], "qpa", answer="Ann")
    assert it["question"] == "Which model should answer this question: Who?"
    assert it["options"] == ("small model", "large model")
    assert it["state"] == "Proposed answer: Ann\n[1] A: a\n[2] B: b"
    assert items.pair_item("Who?", ["A: a"], "qp")["state"] == "[1] A: a"
    with pytest.raises(ValueError, match="answer"):
        items.pair_item("Who?", ["A: a"], "qpa")


def test_menu_items_are_the_training_items():
    ret = items.retrieval_item("Who?", "Query: who?\nSelected references: [1] A: a")
    assert ret["state"] == "[1] A: a" and ret["options"] == ("yes", "no")
    assert ret["question"] == "Do these passages hold what is needed to answer this question: Who?"
    twice = "Query: who?\nSelected references:\n[1] A: a\nSelected references:\n[2] B: b"  # 26 test outputs repeat it
    assert items.retrieval_item("Who?", twice)["state"] == "[1] A: a\nSelected references:\n[2] B: b"  # cut once
    qpa =items.answer_item("Who?", "[1] A: a", "qpa", weak_answer="Ann.\nbecause")
    assert qpa["state"] == "Proposed answer: Ann\n[1] A: a"
    assert qpa["question"] == "Is the proposed answer correct? Question: Who?"
    qp = items.answer_item("Who?", "Selected references:", "qp")
    assert qp["state"] == "(no passages selected)"
    assert qp["question"] == "Will a small model answer this question correctly from these passages: Who?"


def test_the_routers_read_what_training_built():
    import pandas as pd

    from systemone_decider import judge, menu

    row = pd.DataFrame({"question": ["Who?"], "passages": [["A: a", "B: b"]], "small_pred": ["Ann"]})
    df = pd.DataFrame({"question": ["Who?"], "sel_W": ["[1] A: a"], "sel_S": ["[3] C: c"], "answer_WW": ["Ann.\nx"],
                       "answer_SW": ["Bob"]})
    for inputs in items.INPUTS:
        assert judge.route_items(row, inputs)[0] == items.pair_item("Who?", ["A: a", "B: b"], inputs, "Ann")
        trained = menu.items(df, inputs)
        assert trained["ret"][0] == items.retrieval_item("Who?", "[1] A: a")
        assert trained["qa_W"][0] == items.answer_item("Who?", "[1] A: a", inputs, "Ann.\nx")
        assert trained["qa_S"][0] == items.answer_item("Who?", "[3] C: c", inputs, "Bob")


def test_pair_router_escalates_at_its_threshold(folder):
    r = route.PairRouter.from_pretrained(folder({"setting": "pair", "inputs": "qp", "threshold": 0.3, "params_b": 2.0}))
    assert r.inputs == "qp" and r.judge.size_b == 2.0
    FakeJudge.p0 = [0.6, 0.8]  # P(large) = 0.4, then 0.2
    assert r.route("Who?", ["A: a"]) == "large" and r.route("Who?", ["A: a"]) == "small"


def test_menu_router_decides_weak_first(folder):
    card = {"setting": "menu", "inputs": "qpa",
            "thresholds": {"near_strong": [0.5, 0.7, 0.4], "peak": [1.0, 1.0, 1.0], "parity": [0.1, 0.1, 0.1]}}
    r = route.MenuRouter.from_pretrained(folder(card))
    assert r.thresholds == [0.5, 0.7, 0.4]
    FakeJudge.p0 = [0.6, 0.6, 0.6]
    assert r.retrieval("Who?", "[1] A: a") == "weak"
    assert r.answer("Who?", "[1] A: a", "weak", weak_answer="Ann") == "strong"  # 0.6 < 0.7
    assert r.answer("Who?", "[2] B: b", "strong", weak_answer="Ann") == "weak"  # 0.6 >= 0.4
    assert route.MenuRouter.from_pretrained(folder(card), rule="parity").thresholds == [0.1, 0.1, 0.1]
    with pytest.raises(ValueError, match="no 'near_strong' thresholds"):  # a card from before the default rule
        route.MenuRouter.from_pretrained(folder(card | {"thresholds": {"peak": [1.0, 1.0, 1.0]}}))


def test_a_judge_loads_only_in_its_own_setting(folder):
    with pytest.raises(ValueError, match="menu judge"):
        route.PairRouter.from_pretrained(folder({"setting": "menu", "inputs": "qp"}))
