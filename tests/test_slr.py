import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from systemone_decider import slr

# six dev questions: two per split; verdicts per path [WW, WS, SW, SS]
GOLD = ["paris", "1961", "blue", "yes", "oslo", "red"]
EM = [[1, 1, 1, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 1]]
SPLIT = ["train", "train", "val", "val", "test", "test"]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def plan(tmp_path):
    out, grid = tmp_path / "runs", tmp_path / "runs" / slr.GRID
    pd.DataFrame({"id": [f"q{i}" for i in range(6)], "answer": GOLD}).to_parquet(tmp_path / "dev.parquet")
    _write_jsonl(grid / "router_data_standard.jsonl",
                 [{"idx": i, "split": SPLIT[i], "em_standard": EM[i]} for i in range(6)])
    for a, name in (("W", "weak"), ("S", "strong")):  # an answer is the gold when its path is correct
        _write_jsonl(grid / f"qa_{name}.jsonl",
                     [{"k": f"{i}|{b}", "v": GOLD[i] if EM[i][slr.PATHS.index(r + a)] else "wrong"}
                      for i in range(6) for r, b in (("W", "weak"), ("S", "strong"))])
        _write_jsonl(grid / f"meta_retr_{name}.jsonl", [{"k": str(i), "n_in": 2048, "n_out": 128} for i in range(6)])
        _write_jsonl(grid / f"meta_qa_{name}.jsonl",
                     [{"k": f"{i}|{b}", "n_in": 256, "n_out": 64} for i in range(6) for b in ("weak", "strong")])
    _write_jsonl(grid / "test_scores" / "ret_x.jsonl", [{"idx": 4, "pW": 0.9}, {"idx": 5, "pW": 0.1}])
    _write_jsonl(grid / "test_scores" / "qa_x.jsonl",
                 [{"idx": i, "ret_choice": x, "pW": p} for i, x, p in [(4, "W", 0.9), (4, "S", 0.9),
                                                                       (5, "W", 0.9), (5, "S", 0.2)]])
    _write_jsonl(grid / "ret_v" / "val_scores.jsonl", [{"idx": 2, "pW": 0.1}, {"idx": 3, "pW": 0.9}])
    _write_jsonl(grid / "qa_v" / "val_scores.jsonl",
                 [{"idx": i, "ret_choice": x, "pW": 0.5} for i in (2, 3) for x in "WS"])
    splits = {"n_total": 6, "splits": {s: [i for i in range(6) if SPLIT[i] == s] for s in ("train", "val", "test")}}
    (out / slr.SPLITS).parent.mkdir(parents=True, exist_ok=True)
    (out / slr.SPLITS).write_text(json.dumps(splits), encoding="utf-8")
    files = {str(p.relative_to(out).as_posix()): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in out.rglob("*") if p.is_file()}
    # On val, q2 is right only after strong retrieval and q3 never, so parity (val accuracy >= always-SS's 0.5 at the
    # lowest price) first reaches it at t = [0.15, 0, 0]. On test that keeps q4 weak (WW, right, 80) and sends q5 to
    # strong retrieval with the weak answer (SW, wrong, 894).
    certified = [{"name": "router", "ret": "ret_x", "qa": "qa_x", "val": ["ret_v", "qa_v"], "rule": "parity",
                  "t": [0.15, 0.0, 0.0], "lenient": 50.0, "price": 487.0},
                 {"name": "always-strong", "path": "SS", "lenient": 100.0, "price": 929.0}]
    return {"out": str(out), "dev": str(tmp_path / "dev.parquet"), "commit": "0" * 40, "files": files,
            "certified": certified}


def test_direct_and_weak_first_prices():
    assert [slr.direct_price(p) for p in slr.PATHS] == [80, 192, 817, 929]  # a fixed policy runs one path
    assert slr.direct_price("SS") == 929                                     # strong-everything


def test_route_follows_his_rule():
    ret = np.array([0.9, 0.9, 0.1, 0.1])
    qa = {"W": np.array([0.9, 0.1, 0.9, 0.9]), "S": np.array([0.0, 0.0, 0.9, 0.1])}
    assert slr.route(ret, qa, [0.5, 0.5, 0.5]).tolist() == [0, 1, 2, 3]  # WW, WS, SW, SS


def test_select_breaks_ties_as_he_did():
    ok = np.array([[0, 0, 1, 1], [0, 0, 0, 0]], dtype=bool)  # q0 right only after strong retrieval
    points = slr.sweep(ok, np.array([0.1, 0.9]), {"W": np.array([0.5, 0.5]), "S": np.array([0.5, 0.5])})
    assert len(points) == 21 ** 3
    chosen = slr.select(points, strong_accuracy=0.5)
    assert chosen["peak"] == [1.0, 1.0, 1.0]     # ties on accuracy go to the dearer point, then larger thresholds
    assert chosen["parity"] == [0.15, 0.0, 0.0]  # the first cheapest point at always-strong's accuracy
    assert slr.select(points, strong_accuracy=0.9)["parity"] is None


def test_grid_loads_verifies_and_replays(plan):
    df = slr.load_grid(plan)
    assert len(df) == 6 and df["id"].tolist() == [f"q{i}" for i in range(6)]
    assert df["units_WW"].iloc[0] == pytest.approx((7 * 2176 + 3 * 320) / 576)  # measured units, both calls
    parts = {s: [f"q{i}" for i in range(6) if SPLIT[i] == s] for s in ("train", "val", "test")}
    his = json.loads((slr.Path(plan["out"]) / slr.SPLITS).read_text())
    assert slr.ruler_mismatches(df) == 0
    assert slr.split_mismatches(df, his, parts) == []
    assert slr.split_mismatches(df, his, {**parts, "test": ["q4"]}) == ["slr-test differs from his test part"]
    rows = slr.certified(plan, df)
    assert all(row["match"] for row in rows) and rows[0]["ours"]["t_from_val"] == [0.15, 0.0, 0.0]
    plan["certified"][0]["t"] = [0.2, 0.0, 0.0]  # not what his rule picks on val
    assert not slr.certified(plan, df)[0]["match"]


def test_a_flipped_verdict_is_caught(plan):
    df = slr.load_grid(plan)
    df.loc[0, "ok_WW"] = False
    assert slr.ruler_mismatches(df) == 1


def test_a_changed_file_is_refused(plan):
    path = slr.Path(plan["out"]) / slr.GRID / "router_data_standard.jsonl"
    path.write_text(path.read_text() + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        slr.load_grid(plan)


def test_ladder_prices_fixed_paths_and_the_oracle(plan):
    df = slr.load_grid(plan)
    report = slr.ladder(df, seen={"q5"})
    test = report["test"]
    assert test["n"] == 2 and test["always"]["SS"] == {"lenient": 1.0, "price": 929.0,
                                                         "units": pytest.approx((72 * 2176 + 70 * 320) / 576)}
    assert test["oracle"] == pytest.approx({"lenient": 1.0, "price": (80 + 929) / 2,
                                            "units": ((7 * 2176 + 3 * 320) + (72 * 2176 + 70 * 320)) / 2 / 576})
    assert report["test-clean"]["n"] == 1  # q5 left out
