"""Pharmacovigilance knowledge bases.

Two knowledge bases are kept deliberately separate:

* The **reference KB** is used only by the corpus generator to define the gold
  pharmacovigilance labels. No system reads it.
* The **operational KB** is what the Pharmacovigilance Agent (and every baseline
  that raises alerts) consults at run time. It is derived from the reference KB
  through a documented, seeded procedure that removes part of the rules
  (mostly rare combinations) and adds alert rules that the reference KB grades
  as minor. This mimics the incomplete and over-inclusive coverage of real drug
  knowledge bases and keeps the evaluation labels from coinciding with the
  rules the system applies. ``kb_overlap`` reports the agreement.

Rule semantics follow common interaction references (bleeding with
anticoagulant + antiplatelet/NSAID, hyperkalaemia with RAAS blockers +
potassium, QT prolongation, serotonin syndrome, CYP3A4 statin interactions,
beta-lactam allergy, NSAID contraindications in CKD/PUD, metformin with low
eGFR). Severity grades are simplified to major / moderate / minor.
"""
from __future__ import annotations

import random
from typing import Dict, Iterable, List

from .terminology import DRUG_BY_KEY

# (drug_a, drug_b, severity, rare)
_DDI = [
    ("warfarin", "aspirin", "major", False),
    ("warfarin", "ibuprofen", "major", False),
    ("warfarin", "naproxen", "major", False),
    ("warfarin", "ciprofloxacin", "moderate", False),
    ("warfarin", "levofloxacin", "moderate", False),
    ("warfarin", "fluconazole", "major", True),
    ("warfarin", "amiodarone", "major", False),
    ("warfarin", "tmp_smx", "major", False),
    ("warfarin", "clarithromycin", "moderate", True),
    ("apixaban", "aspirin", "moderate", False),
    ("apixaban", "clopidogrel", "moderate", False),
    ("apixaban", "ibuprofen", "moderate", True),
    ("enoxaparin", "ibuprofen", "moderate", True),
    ("heparin", "aspirin", "moderate", True),
    ("clopidogrel", "omeprazole", "moderate", False),
    ("simvastatin", "clarithromycin", "major", False),
    ("simvastatin", "amiodarone", "moderate", False),
    ("simvastatin", "diltiazem", "moderate", False),
    ("simvastatin", "verapamil", "moderate", True),
    ("atorvastatin", "clarithromycin", "moderate", True),
    ("lisinopril", "spironolactone", "moderate", False),
    ("losartan", "spironolactone", "moderate", False),
    ("lisinopril", "potassium_chloride", "moderate", False),
    ("spironolactone", "potassium_chloride", "major", False),
    ("lisinopril", "losartan", "moderate", True),
    ("digoxin", "amiodarone", "major", False),
    ("digoxin", "verapamil", "moderate", True),
    ("metoprolol", "verapamil", "major", True),
    ("metoprolol", "diltiazem", "moderate", False),
    ("carvedilol", "verapamil", "major", True),
    ("sertraline", "tramadol", "major", False),
    ("sertraline", "linezolid", "major", True),
    ("citalopram", "linezolid", "major", True),
    ("fluoxetine", "tramadol", "major", False),
    ("citalopram", "ondansetron", "moderate", False),
    ("citalopram", "haloperidol", "moderate", True),
    ("haloperidol", "ondansetron", "moderate", True),
    ("amiodarone", "levofloxacin", "major", True),
    ("methotrexate", "tmp_smx", "major", True),
    ("methotrexate", "ibuprofen", "moderate", True),
    ("oxycodone", "gabapentin", "moderate", False),
    ("morphine", "gabapentin", "moderate", False),
    ("tramadol", "ondansetron", "moderate", True),
    ("levofloxacin", "prednisone", "moderate", True),
    ("ciprofloxacin", "prednisone", "moderate", True),
    ("fluconazole", "citalopram", "moderate", True),
    ("clarithromycin", "amlodipine", "moderate", True),
    # graded minor by the reference: not gold signals
    ("aspirin", "lisinopril", "minor", False),
    ("omeprazole", "levothyroxine", "minor", False),
    ("digoxin", "furosemide", "minor", False),
    ("metformin", "furosemide", "minor", False),
    ("atorvastatin", "amlodipine", "minor", False),
]

# (allergen_key, target, target_is_class, severity, rare)
_ALLERGY = [
    ("penicillin", "penicillin", True, "major", False),
    ("penicillin", "cephalosporin", True, "minor", False),
    ("sulfonamide", "sulfonamide_abx", True, "major", False),
    ("sulfonamide", "loop", True, "minor", False),
    ("sulfonamide", "thiazide", True, "minor", False),
    ("aspirin_allergen", "nsaid", True, "major", False),
    ("aspirin_allergen", "nsaid_asa", True, "major", False),
    ("ibuprofen_allergen", "nsaid", True, "major", False),
    ("ibuprofen_allergen", "nsaid_asa", True, "moderate", True),
    ("codeine", "morphine", False, "moderate", True),
    ("codeine", "oxycodone", False, "minor", False),
    ("morphine_allergen", "morphine", False, "major", False),
    ("morphine_allergen", "oxycodone", False, "moderate", True),
]

# (condition_key, target, target_is_class, severity, rare)
_CONDITION = [
    ("ckd", "nsaid", True, "major", False),
    ("pud", "nsaid", True, "major", False),
    ("gi_bleed", "nsaid", True, "major", False),
    ("hf", "nsaid", True, "moderate", False),
    ("asthma", "propranolol", False, "major", False),
    ("asthma", "carvedilol", False, "moderate", True),
    ("asthma", "metoprolol", False, "minor", False),
    ("gi_bleed", "warfarin", False, "major", False),
    ("gi_bleed", "apixaban", False, "major", True),
    ("hf", "verapamil", False, "moderate", True),
    ("hf", "diltiazem", False, "moderate", True),
    ("epilepsy", "tramadol", False, "major", True),
    ("cirrhosis", "metformin", False, "moderate", True),
    ("t2dm", "prednisone", False, "minor", False),
]

# (lab_key, op, threshold, targets, target_is_class, severity, rare)
_LAB = [
    ("potassium", ">", 5.5, ["spironolactone", "potassium_chloride", "lisinopril", "losartan"], False, "major", False),
    ("egfr", "<", 30, ["metformin"], False, "major", False),
    ("potassium", "<", 3.5, ["digoxin"], False, "moderate", False),
    ("inr", ">", 4.0, ["warfarin"], False, "major", False),
    ("egfr", "<", 30, ["enoxaparin"], False, "moderate", True),
    ("egfr", "<", 30, ["nsaid"], True, "major", False),
]


def _drug_matches(drug_key: str, target: str, is_class: bool) -> bool:
    d = DRUG_BY_KEY.get(drug_key)
    if d is None:
        return False
    return d["cls"] == target if is_class else drug_key == target


def reference_kb() -> List[Dict]:
    rules = []
    for i, (a, b, sev, rare) in enumerate(_DDI):
        rules.append(dict(id=f"DDI{i:02d}", type="ddi", a=a, b=b, severity=sev, rare=rare))
    for i, (al, tgt, is_cls, sev, rare) in enumerate(_ALLERGY):
        rules.append(dict(id=f"ALG{i:02d}", type="allergy", allergen=al, target=tgt, is_class=is_cls,
                          severity=sev, rare=rare))
    for i, (cond, tgt, is_cls, sev, rare) in enumerate(_CONDITION):
        rules.append(dict(id=f"CND{i:02d}", type="condition", condition=cond, target=tgt, is_class=is_cls,
                          severity=sev, rare=rare))
    for i, (lab, op, thr, tgts, is_cls, sev, rare) in enumerate(_LAB):
        rules.append(dict(id=f"LAB{i:02d}", type="lab", lab=lab, op=op, threshold=thr, targets=tgts,
                          is_class=is_cls, severity=sev, rare=rare))
    return rules


def operational_kb(seed: int = 2026, rare_drop: float = 0.6, common_drop: float = 0.05,
                   promote_minor: Iterable[str] = ("ALG01", "ALG03", "DDI47", "DDI49", "CND06"),
                   coverage: float | None = None) -> List[Dict]:
    """Derive the operational KB from the reference KB.

    Alerting rules are those graded major/moderate. Each is dropped with
    probability ``rare_drop`` (rare combinations) or ``common_drop`` (common
    ones). Minor rules listed in ``promote_minor`` are added as alerts, which is
    a typical source of over-alerting. ``coverage`` overrides the drop rates and
    keeps that fraction of reference alerting rules (used in the KB-coverage
    sensitivity experiment; ``coverage=1.0`` reproduces the reference exactly
    and therefore represents the circular, upper-bound setting).
    """
    rng = random.Random(seed)
    ref = reference_kb()
    alerting = [r for r in ref if r["severity"] in ("major", "moderate")]
    kept = []
    if coverage is not None:
        order = sorted(alerting, key=lambda r: (not r["rare"], rng.random()))
        n_drop = int(round((1.0 - coverage) * len(alerting)))
        dropped = {r["id"] for r in order[:n_drop]}
        kept = [dict(r) for r in alerting if r["id"] not in dropped]
        promote = set() if coverage >= 1.0 else set(promote_minor)
    else:
        for r in alerting:
            p = rare_drop if r["rare"] else common_drop
            if rng.random() >= p:
                kept.append(dict(r))
        promote = set(promote_minor)
    for r in ref:
        if r["id"] in promote:
            rr = dict(r)
            rr["severity"] = "moderate"
            rr["promoted"] = True
            kept.append(rr)
    return kept


def kb_overlap(op_kb: List[Dict]) -> Dict:
    ref_alert = {r["id"] for r in reference_kb() if r["severity"] in ("major", "moderate")}
    op = {r["id"] for r in op_kb}
    inter = ref_alert & op
    return dict(reference_alert_rules=len(ref_alert), operational_rules=len(op),
                shared=len(inter), reference_coverage=len(inter) / max(len(ref_alert), 1),
                extra_rules=len(op - ref_alert),
                jaccard=len(inter) / max(len(ref_alert | op), 1))


def fire_rules(state: Dict, kb: List[Dict], min_severity=("major", "moderate")) -> List[Dict]:
    """Apply ``kb`` to a reconciled clinical state.

    ``state`` holds ``meds`` (active drug keys), ``allergies`` (verified
    allergen keys), ``conditions`` (active condition keys) and ``labs``
    (lab key -> latest value). Returns one signal per fired rule, keyed by the
    unordered pair of interacting components so that alerts from different
    systems can be matched to gold signals.
    """
    meds = sorted(set(state.get("meds", [])))
    allergies = set(state.get("allergies", []))
    conditions = set(state.get("conditions", []))
    labs = state.get("labs", {})
    out, seen = [], set()

    def emit(rule, comp):
        key = (rule["type"],) + tuple(sorted(comp))
        if key in seen:
            return
        seen.add(key)
        out.append(dict(rule_id=rule["id"], type=rule["type"], components=tuple(sorted(comp)),
                        key=key, severity=rule["severity"]))

    for r in kb:
        if r["severity"] not in min_severity:
            continue
        if r["type"] == "ddi":
            if r["a"] in meds and r["b"] in meds:
                emit(r, (r["a"], r["b"]))
        elif r["type"] == "allergy":
            if r["allergen"] in allergies:
                for m in meds:
                    if _drug_matches(m, r["target"], r["is_class"]):
                        emit(r, (r["allergen"], m))
        elif r["type"] == "condition":
            if r["condition"] in conditions:
                for m in meds:
                    if _drug_matches(m, r["target"], r["is_class"]):
                        emit(r, (r["condition"], m))
        elif r["type"] == "lab":
            v = labs.get(r["lab"])
            if v is None:
                continue
            hit = v > r["threshold"] if r["op"] == ">" else v < r["threshold"]
            if not hit:
                continue
            for m in meds:
                if any(_drug_matches(m, t, r["is_class"]) for t in r["targets"]):
                    emit(r, (r["lab"], m))
    return out
