"""Core tests: generator, validator, metrics, axioms, knowledge bases and the mesh.

Run with:  python -m pytest -q
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis import axioms, fhir  # noqa: E402
from aegis.agents import Mesh, MeshConfig, run_single  # noqa: E402
from aegis.calibration import noisy_and, noisy_or  # noqa: E402
from aegis.knowledge import kb_overlap, operational_kb, reference_kb  # noqa: E402
from aegis.metrics import evaluate_record  # noqa: E402
from aegis.nlp import BASE_HEADERS, build_lexicon  # noqa: E402
from aegis.synth import PROFILES, Generator, make_record  # noqa: E402

SAMPLE = ROOT / "data" / "sample" / "test_sample_30.jsonl"


def records(n=60):
    gen = Generator(seed=123)
    return [make_record(gen, i, PROFILES[k]) for k in "ABC" for i in range(n // 3)]


def test_gold_bundles_are_valid_fhir():
    for r in records():
        v = fhir.validate_bundle(r["gold"])
        assert v["valid"], v["errors"][:3]


def test_validator_catches_common_llm_errors():
    r = records(3)[0]
    b = copy.deepcopy(r["gold"])
    res = [e["resource"] for e in b["entry"]]
    cond = next(x for x in res if x["resourceType"] == "Condition")
    cond["onsetDateTime"] = "03/14/2021"                 # not ISO
    med = next(x for x in res if x["resourceType"] == "MedicationRequest")
    med["dosage"] = med.pop("dosageInstruction", [])    # unknown element
    kinds = fhir.validate_bundle(b)["by_kind"]
    assert "invalid_datetime" in kinds and "unknown_element" in kinds


def test_gold_scores_perfectly():
    for r in records(15):
        gs = [frozenset(s["components"]) for s in r["gold_signals"]]
        e = evaluate_record(r, r["gold"], gs)
        assert e["acceptable"] and e["cf1"] == 1.0 and e["tf"] == 1.0 and e["fabricated"] == 0


def test_fabricated_timestamp_is_critical():
    r = next(x for x in records(60) if any(c["resource"].get("onsetString") for c in x["gold"]["entry"]))
    b = copy.deepcopy(r["gold"])
    for e in b["entry"]:
        if e["resource"].get("onsetString"):
            e["resource"].pop("onsetString")
            e["resource"]["onsetDateTime"] = "2020-01-01"
            break
    gs = [frozenset(s["components"]) for s in r["gold_signals"]]
    ev = evaluate_record(r, b, gs)
    assert ev["critical"].get("fabricated_timestamp", 0) >= 1 and not ev["acceptable"]


def test_partial_dates_are_intervals():
    assert axioms.le("2015", "2015-03-01") is True
    assert axioms.le("2016", "2015-12-31") is False


def test_operational_kb_differs_from_reference():
    ov = kb_overlap(operational_kb())
    assert 0.5 < ov["reference_coverage"] < 1.0 and ov["extra_rules"] > 0
    assert kb_overlap(operational_kb(coverage=1.0))["reference_coverage"] == 1.0


def test_noisy_and_is_weakest_link():
    import numpy as np
    sig = {"S": np.array([0.95]), "R": np.array([0.05]), "T": np.array([0.95]), "P": np.array([0.95])}
    assert noisy_and(sig, np.ones(4))[0] < 0.06
    assert noisy_or(sig, np.ones(4))[0] > 0.99


def test_mesh_beats_single_pass_on_sample():
    recs = [json.loads(l) for l in SAMPLE.open()]
    learned = json.loads((ROOT / "results" / "learned_partA.json").read_text())
    headers = dict(BASE_HEADERS)
    headers.update(learned["headers"])
    lex = build_lexicon("schema", [tuple(v) for v in learned["vocab"]])
    kb = operational_kb()
    mesh = Mesh(lex, lex, headers, kb)
    acc_mesh = acc_single = 0
    for r in recs:
        o = mesh.run(r["text"], MeshConfig())
        acc_mesh += evaluate_record(r, o["bundle"], o["alerts"])["acceptable"]
        s = run_single(r["text"], lex, headers, kb)
        acc_single += evaluate_record(r, s["bundle"], s["alerts"])["acceptable"]
        assert fhir.validate_bundle(o["bundle"])["valid"]
    assert acc_mesh >= acc_single and acc_mesh >= 0.8 * len(recs)


def test_provenance_policy_never_shifts_dates():
    recs = [json.loads(l) for l in SAMPLE.open()]
    learned = json.loads((ROOT / "results" / "learned_partA.json").read_text())
    lex = build_lexicon("schema", [tuple(v) for v in learned["vocab"]])
    mesh = Mesh(lex, lex, dict(BASE_HEADERS, **learned["headers"]), operational_kb())
    for r in recs:
        o = mesh.run(r["text"], MeshConfig())
        assert o["reports"]["T"].stats["shifted"] == 0
