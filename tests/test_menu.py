import numpy as np
import pandas as pd
import pytest

from systemone_decider import menu as e
from systemone_decider import slr


def _grid(ok_test, ok_val):
    """A grid with the columns evaluate() reads: verdicts per path, val first, then test."""
    ok = np.vstack([ok_val, ok_test]).astype(bool)
    split = ["val"] * len(ok_val) + ["test"] * len(ok_test)
    return pd.DataFrame({"idx": range(len(ok)), "id": [f"q{i}" for i in range(len(ok))], "split": split,
                         **{f"ok_{p}": ok[:, k] for k, p in enumerate(slr.PATHS)}})


def test_selection_drops_the_query_line():
    assert e.selection("Query: who?\nSelected references: [1] A: a\n[4] B: b") == "[1] A: a\n[4] B: b"
    assert e.selection("Query: who?\nSelected references:") == "(no passages selected)"


def test_labels_are_his_targets():
    df = pd.DataFrame({"y_ret": ["W", "S"], "y_qa_W": ["S", "W"], "y_qa_S": ["W", "W"]})
    assert {k: v.tolist() for k, v in e.labels(df).items()} == {"ret": [0, 1], "qa_W": [1, 0], "qa_S": [0, 0]}
    with pytest.raises(ValueError, match="targets"):
        e.labels(df.assign(y_ret=["W", None]))


def test_the_joint_question_learns_his_targets_as_one_path_and_routes_like_the_stages():
    df = pd.DataFrame({"y_ret": ["W", "W", "S", "S"], "y_qa_W": ["W", "S", "S", "W"], "y_qa_S": ["S", "W", "W", "S"]})
    assert e.joint_labels(df).tolist() == [0, 1, 2, 3]  # WW, WS, SW, SS: the answer target under his retrieval target
    p = np.full((4, 4), 0.1) + 0.6 * np.eye(4)  # each question's most likely path, 0.7 against 0.1 for the others
    s = e.joint_scores(p)
    assert np.allclose(s["ret"], [0.8, 0.8, 0.2, 0.2]) and np.allclose(s["qa_W"][:2], [0.875, 0.125])
    assert slr.route(s["ret"], {"W": s["qa_W"], "S": s["qa_S"]}, [0.5, 0.5, 0.5]).tolist() == [0, 1, 2, 3]
    states = e.joint_items(pd.DataFrame({"question": ["who?"], "sel_W": ["[1] A: a"], "answer_WW": ["Ann\nbecause"]}),
                           "qpa")
    assert states[0]["state"] == "Proposed answer: Ann\n[1] A: a" and len(states[0]["options"]) == 4


def test_a_retrain_can_keep_its_weights(tmp_path, monkeypatch):
    class Saved:
        def __init__(self, name):
            self.name = name

        def save_pretrained(self, path):
            path.mkdir(parents=True, exist_ok=True)
            (path / self.name).write_text("saved")

    class FakeJudge:
        device = "cpu"
        model, tokenizer = Saved("weights"), Saved("tokenizer")

        def probabilities(self, its):
            return np.full((len(its), 2), 0.5), np.full(len(its), 7)

    monkeypatch.setattr(e, "make_judge", lambda *a, **k: FakeJudge())
    monkeypatch.setattr(e, "fine_tune", lambda *a, **k: [{"epoch": 1}])
    monkeypatch.setattr(e.provenance, "hub_revision", lambda init: "rev")
    n = 12
    df = pd.DataFrame({"idx": range(n), "split": ["train"] * 8 + ["val"] * 2 + ["test"] * 2,
                       "question": [f"q{i}?" for i in range(n)], "sel_W": "[1] A: a", "sel_S": "[2] B: b",
                       "answer_WW": "x", "answer_SW": "y", "y_ret": ["W", "S"] * 6, "y_qa_W": "W", "y_qa_S": "S"})
    plan = {"out": str(tmp_path)}
    spec = {"name": "decider-2b", "judge_dir": "judge-decider-2b", "init": "strands-decider-2B",
            "train": {"dev_fraction": 0.25, "epochs": 1, "lr": 1e-4, "weight_decay": 0.0, "warmup": 0.0, "batch_size": 4}}
    e.run_train(df, plan, spec, "qpa", 0, "cpu", save_model=True)
    seed = tmp_path / "judge-decider-2b" / "qpa" / "seed0"
    assert (seed / "model" / "weights").exists() and (seed / "model" / "tokenizer").exists()
    assert (seed / "done.json").exists() and len(pd.read_parquet(seed / "test.parquet")) == 2
    e.run_train(df, plan, spec, "qp", 0, "cpu")  # without it, nothing is saved
    assert not (tmp_path / "judge-decider-2b" / "qp" / "seed0" / "model").exists()


def test_pareto_keeps_the_undominated_thresholds():
    points = [(0.5, 0.1, "a"), (0.6, 0.2, "b"), (0.55, 0.3, "c"), (0.6, 0.25, "d"), (0.7, 0.9, "e")]
    assert e.pareto(points) == [("a",), ("b",), ("e",)]


def test_router_pays_the_retrieval_call_and_the_answer_call_it_took():
    price = np.array([[1.0, 2.0, 30.0], [1.0, 2.0, 30.0]])  # ret call, qa under W, qa under S
    r = e.Router("ours", {}, {}, price)
    assert r.extra(np.array([1, 3])).tolist() == [3.0, 31.0]  # WS took the weak branch, SS the strong one
    assert e.Router("his", {}, {}).extra(np.array([0, 3])).tolist() == [0.0, 0.0]  # his routers: not charged


PROMPTS = '''"""docstring"""
CHAR_CAP = 5
MAXLEN   = 12
WEAK_NAME_DEFAULT   = "w"
STRONG_NAME_DEFAULT = "s"
SYS_CASCADE_TEMPLATE = (
    "route {weak_name} "
    "or {strong_name}"
)
INSTR_RET2 = "R {q} {n_sel}: {selections}"
INSTR_QA1 = "Q {q} | {evidence} | {weak_answer}"
def cap(text, n=CHAR_CAP):
    return text[:n]
'''


class _WordTokenizer:
    """One token per whitespace-separated word, with the chat template as plain concatenation."""

    def apply_chat_template(self, msgs, tokenize, add_generation_prompt):
        return " ".join(m["content"] for m in msgs) + " <assistant>"

    def __call__(self, text, add_special_tokens):
        return {"input_ids": text.split()}


def test_his_router_is_priced_on_his_own_prompts(tmp_path):
    path = tmp_path / "router_prompts.py"
    path.write_text(PROMPTS)
    prompts = e.his_prompts(path)
    assert prompts["SYS_CASCADE_TEMPLATE"] == "route {weak_name} or {strong_name}" and prompts["CHAR_CAP"] == 5
    assert "cap" not in prompts  # functions are skipped, never run
    df = pd.DataFrame({"question": ["who?"], "sel_W": ["[1] Alpha beta\n[4] Gamma\n\nSo the answer is A."],
                       "sel_S": ["[2] Delta"], "answer_WW": ["A"], "answer_SW": ["D"]})
    system, calls = e.his_router_calls(df, prompts)
    assert system == "route w or s"
    assert calls["ret"] == ["R who? 2: [1] Alpha [4] Gamma"]  # capped at 5 characters, the reasoning dropped
    assert calls["qa_W"] == ["Q who? | [1] Alpha\n[4] Gamma | A"] and calls["qa_S"] == ["Q who? | [2] Delta | D"]
    tokenizer = _WordTokenizer()
    price = e.his_router_price(df, prompts, tokenizer)
    words = len("route w or s".split()) + 1  # the system turn and the opened assistant turn
    assert price.tolist() == [[e.budget(e.HIS_ROUTER_PARAMS_B, min(words + len(c[0].split()), 12), 0)
                               for c in (calls["ret"], calls["qa_W"], calls["qa_S"])]]


def test_evaluate_compares_ours_with_his_on_his_terms():
    rng = np.random.default_rng(0)
    ok_val, ok_test = rng.random((40, 4)) < [0.5, 0.6, 0.6, 0.8], rng.random((60, 4)) < [0.5, 0.6, 0.6, 0.8]
    df = _grid(ok_test, ok_val)

    def oracle_scores(ok):  # route each question to its cheapest correct path, by weak-first price
        c = e.ev.cheapest_correct_label(ok, np.tile(e.PRICES, (len(ok), 1)))
        return {"ret": (c < 2).astype(float), "qa_W": (c == 0).astype(float), "qa_S": (c == 2).astype(float)}

    keep_weak = lambda n: {k: np.ones(n) for k in e.SCORES}  # noqa: E731
    routers = [e.Router("SLR perfect", oracle_scores(ok_val), oracle_scores(ok_test)),
               e.Router("ours#seed0", keep_weak(40), keep_weak(60), np.full((60, 3), 2.0))]
    report = e.evaluate(df, routers, {"bootstrap": 20})
    best = report["routers"]["SLR perfect"]["peak"]
    rows = np.arange(60)
    c = e.ev.cheapest_correct_label(ok_test, np.tile(e.PRICES, (60, 1)))
    assert best["lenient"] == pytest.approx(ok_test[rows, c].mean())  # a perfect router reaches every solvable question
    ours = report["routers"]["ours#seed0"]["peak"]
    assert ours["paths"]["WW"] == 60 and ours["price"] == 80 and ours["judge_price"] == 4.0
    assert report["routers"]["ours#seed0"]["aiq_with_judge_price"] <= report["routers"]["ours#seed0"]["aiq"]
    assert set(report["aiq_diff"]) == {"ours#seed0 - SLR perfect", "ours#seed0 - fixed paths"}
    assert report["aiq_diff"]["ours#seed0 - SLR perfect"]["ci95"][1] < 0
    assert set(report["aiq_diff_both_priced"]) == {"ours#seed0 +judge - SLR perfect +judge"}
    priced = e.Router("SLR perfect", oracle_scores(ok_val), oracle_scores(ok_test), np.full((60, 3), 50.0))
    both = e.evaluate(df, [priced, routers[1]], {"bootstrap": 20})
    his = both["routers"]["SLR perfect"]
    assert his["aiq_with_judge_price"] < his["aiq"] and his["peak"]["judge_price"] > 0  # his calls now cost something
    clean = e.evaluate(df, routers, {"bootstrap": 20}, keep=np.arange(60) < 30)
    assert clean["n_test"] == 30
