"""Record-level evaluation against source-faithful gold bundles.

Every metric used in the paper is defined here:

* core clinical facts: conditions, medications, allergies (by code) and
  laboratory values (code + value); precision, recall, F1 (CF-F1);
* SPI (Semantic Preservation Index): share of gold facts reproduced with all
  key attributes correct (strict recall);
* clinical hallucination rate: emitted facts with no gold counterpart, i.e.
  facts not supported by, or contradicted by, the note;
* temporal fidelity (TF) over timestamp slots and the unsupported timestamp
  rate (UTR: precise dates that the note does not support);
* TCR over the temporal axioms (all / agent / held-out);
* FHIR validity;
* pharmacovigilance at signal and record level;
* MedBLEU: BLEU-4 over a canonical linearization of the bundle;
* critical / major / minor errors, bundle acceptability (no critical error)
  and a severity-weighted error score (SWES).
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Tuple

from . import axioms
from .fhir import resources_of, validate_bundle
from .terminology import (ALLERGEN_BY_CODE, COND_BY_CODE, DRUG_BY_CODE, DRUG_BY_KEY, FREQS, HIGH_ALERT, LAB_BY_CODE,
                          LOINC, ROUTES, RXNORM, SNOMED)
from .nlp import FREQ_MAP, ROUTE_MAP

ROUTE_BY_SNOMED = {v["snomed"]: k for k, v in ROUTES.items()}
UNIT_NORM = {"mg": "mg", "milligram": "mg", "ug": "ug", "mcg": "ug", "µg": "ug", "g": "g", "gram": "g",
             "[iu]": "[iU]", "iu": "[iU]", "units": "[iU]", "unit": "[iU]", "u": "[iU]", "meq": "meq",
             "{puff}": "{puff}", "puff": "{puff}", "puffs": "{puff}"}


def _code(cc, system):
    if not isinstance(cc, dict):
        return None
    for c in cc.get("coding", []) or []:
        if isinstance(c, dict) and c.get("system") == system and isinstance(c.get("code"), str):
            return c["code"]
    return None


def _first(x):
    return x[0] if isinstance(x, list) and x else (x if isinstance(x, dict) else {})


def norm_dose(q) -> Optional[Tuple[float, str]]:
    if not isinstance(q, dict) or not isinstance(q.get("value"), (int, float)) or isinstance(q.get("value"), bool):
        return None
    u = q.get("code") or q.get("unit")
    u = UNIT_NORM.get(str(u).lower(), str(u)) if u else None
    v = float(q["value"])
    if u == "g":
        v, u = v * 1000.0, "mg"
    if u == "ug":
        v, u = v / 1000.0, "mg"
    return round(v, 6), u


def per_day(dosage) -> Optional[float]:
    if not isinstance(dosage, dict):
        return None
    t = dosage.get("timing")
    if isinstance(t, dict):
        rep = t.get("repeat")
        if isinstance(rep, dict) and isinstance(rep.get("frequency"), (int, float)) and \
                isinstance(rep.get("period"), (int, float)) and rep.get("period"):
            f, p, u = rep["frequency"], rep["period"], rep.get("periodUnit")
            mult = {"h": 24.0, "min": 1440.0, "d": 1.0, "wk": 1 / 7.0, "mo": 1 / 30.0}.get(u)
            if mult is not None:
                return round(f * mult / p, 4)
        code = t.get("code")
        txt = code.get("text") if isinstance(code, dict) else None
        if not txt and isinstance(code, dict):
            txt = next((c.get("code") for c in code.get("coding", []) or [] if isinstance(c, dict)), None)
        if txt and str(txt).lower() in FREQ_MAP:
            f, p, u = FREQS[FREQ_MAP[str(txt).lower()]]["repeat"]
            return round(f * {"h": 24.0, "d": 1.0, "wk": 1 / 7.0}[u] / p, 4)
    txt = dosage.get("text")
    if isinstance(txt, str):
        for form in sorted(FREQ_MAP, key=len, reverse=True):
            if form in txt.lower():
                f, p, u = FREQS[FREQ_MAP[form]]["repeat"]
                return round(f * {"h": 24.0, "d": 1.0, "wk": 1 / 7.0}[u] / p, 4)
    return None


def route_key(dosage) -> Optional[str]:
    if not isinstance(dosage, dict) or not isinstance(dosage.get("route"), dict):
        return None
    r = dosage["route"]
    for c in r.get("coding", []) or []:
        if isinstance(c, dict):
            if c.get("code") in ROUTE_BY_SNOMED:
                return ROUTE_BY_SNOMED[c["code"]]
            if isinstance(c.get("code"), str) and c["code"].lower() in ROUTE_MAP:
                return ROUTE_MAP[c["code"].lower()]
            if isinstance(c.get("display"), str) and c["display"].lower() in ROUTE_MAP:
                return ROUTE_MAP[c["display"].lower()]
    t = r.get("text")
    if isinstance(t, str) and t.lower() in ROUTE_MAP:
        return ROUTE_MAP[t.lower()]
    return None


def _onset(r):
    if isinstance(r.get("onsetDateTime"), str):
        return r["onsetDateTime"]
    if isinstance(r.get("onsetString"), str):
        return ("text", r["onsetString"])
    return None


def bundle_facts(b) -> Dict:
    """Normalized clinical content of a bundle."""
    F = dict(condition={}, drug={}, allergen={}, lab=[], unknown=0, encounter={}, ids={})
    if not isinstance(b, dict):
        return F
    for r in resources_of(b):
        rt = r.get("resourceType")
        if rt == "Condition":
            c = COND_BY_CODE.get(_code(r.get("code"), SNOMED) or "")
            if not c:
                F["unknown"] += 1
                continue
            if c["key"] in F["condition"]:
                continue
            F["condition"][c["key"]] = dict(status=_code(r.get("clinicalStatus"),
                                                         "http://terminology.hl7.org/CodeSystem/condition-clinical"),
                                            onset=_onset(r), recorded=r.get("recordedDate"))
            F["ids"][f"Condition/{r.get('id')}"] = c["key"]
        elif rt == "MedicationRequest":
            d = DRUG_BY_CODE.get(_code(r.get("medicationCodeableConcept"), RXNORM) or "")
            if not d:
                F["unknown"] += 1
                continue
            if d["key"] in F["drug"]:
                continue
            dos = _first(r.get("dosageInstruction"))
            dq = None
            if isinstance(dos, dict):
                dr = _first(dos.get("doseAndRate"))
                if isinstance(dr, dict):
                    dq = dr.get("doseQuantity")
            F["drug"][d["key"]] = dict(status=r.get("status"), dose=norm_dose(dq), freq=per_day(dos),
                                       route=route_key(dos), prn=bool(dos.get("asNeededBoolean")) if isinstance(dos, dict) else False,
                                       authored=r.get("authoredOn"))
            F["ids"][f"MedicationRequest/{r.get('id')}"] = d["key"]
        elif rt == "AllergyIntolerance":
            a = ALLERGEN_BY_CODE.get(_code(r.get("code"), SNOMED) or "")
            if not a:
                F["unknown"] += 1
                continue
            if a["key"] in F["allergen"]:
                continue
            ver = _code(r.get("verificationStatus"), "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification")
            if ver == "refuted":
                continue
            rx = set()
            sev = None
            for x in r.get("reaction", []) or []:
                if isinstance(x, dict):
                    for m in x.get("manifestation", []) or []:
                        code = _code(m, SNOMED)
                        if code:
                            rx.add(code)
                    sev = sev or x.get("severity")
            F["allergen"][a["key"]] = dict(reactions=frozenset(rx), severity=sev, onset=_onset(r))
            F["ids"][f"AllergyIntolerance/{r.get('id')}"] = a["key"]
        elif rt == "Observation":
            lab = LAB_BY_CODE.get(_code(r.get("code"), LOINC) or "")
            if not lab:
                F["unknown"] += 1
                continue
            vq = r.get("valueQuantity")
            v = vq.get("value") if isinstance(vq, dict) else None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                F["lab"].append((lab["key"], round(float(v), 3), r.get("effectiveDateTime")))
                F["ids"][f"Observation/{r.get('id')}"] = lab["key"]
        elif rt == "Encounter":
            p = r.get("period") if isinstance(r.get("period"), dict) else {}
            F["encounter"] = dict(start=p.get("start"), end=p.get("end"), status=r.get("status"))
    return F


def state_from_facts(F: Dict) -> Dict:
    latest = {}
    for k, v, eff in sorted(F["lab"], key=lambda x: (x[2] or "")):
        latest[k] = v
    return dict(meds=[k for k, a in F["drug"].items() if a["status"] in ("active", None, "unknown")],
                allergies=list(F["allergen"]), conditions=[k for k, a in F["condition"].items() if a["status"] != "resolved"],
                labs=latest)


def alerts_from_detected_issues(b, F) -> List[frozenset]:
    out = []
    for r in resources_of(b, "DetectedIssue"):
        comps = set()
        for x in r.get("implicated", []) or []:
            if isinstance(x, dict) and isinstance(x.get("reference"), str):
                k = F["ids"].get(x["reference"])
                if k:
                    comps.add(k)
        if len(comps) >= 2:
            out.append(frozenset(comps))
    return out


# ---------------------------------------------------------------------------
# temporal slots
# ---------------------------------------------------------------------------
def _precision(s: str) -> int:
    return {4: 1, 7: 2}.get(len(s), 3)


def slot_outcome(g, p) -> Optional[str]:
    """Classify one timestamp slot. g/p: ISO string, ('text', s) or None."""
    if g is None and p is None:
        return None
    if isinstance(g, str):
        if isinstance(p, str):
            if p[:10] == g:
                return "correct"
            if p.startswith(g) and _precision(p[:10]) > _precision(g):
                return "fabricated"        # more precise than the note supports
            return "wrong"
        return "missing" if p is None else "wrong"
    if isinstance(g, tuple):               # vague wording in the note
        if isinstance(p, tuple):
            return "correct"
        return "fabricated" if isinstance(p, str) else "missing"
    # gold has no timestamp
    return "fabricated" if isinstance(p, str) else "unsupported_text"


# ---------------------------------------------------------------------------
# MedBLEU
# ---------------------------------------------------------------------------
_TYPE_ORDER = {"condition": 0, "drug": 1, "allergen": 2, "lab": 3}


def linearize(F: Dict) -> List[str]:
    toks: List[str] = []
    for k in sorted(F["condition"]):
        a = F["condition"][k]
        toks += ["Condition", k, str(a["status"]), _t(a["onset"])]
    for k in sorted(F["drug"]):
        a = F["drug"][k]
        toks += ["MedicationRequest", k, str(a["status"]), _t(a["dose"]), str(a["freq"]), str(a["route"]),
                 "prn" if a["prn"] else "sched", _t(a["authored"])]
    for k in sorted(F["allergen"]):
        a = F["allergen"][k]
        toks += ["AllergyIntolerance", k] + sorted(a["reactions"]) + [str(a["severity"]), _t(a["onset"])]
    for k, v, eff in sorted(F["lab"], key=lambda x: (x[0], x[1])):
        toks += ["Observation", k, str(v), _t(eff)]
    return toks


def _t(x):
    if x is None:
        return "none"
    if isinstance(x, tuple):
        return ":".join(str(v) for v in x)
    return str(x)


def bleu4(hyp: List[str], ref: List[str]) -> float:
    """Sentence BLEU-4, uniform weights, brevity penalty, add-one smoothing (Lin and Och, 2004)."""
    if not hyp:
        return 0.0
    logs = 0.0
    for n in range(1, 5):
        h = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        r = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        match = sum(min(c, r[g]) for g, c in h.items())
        total = max(sum(h.values()), 0)
        if n == 1:
            p = match / total if total else 0.0
            if p == 0:
                return 0.0
        else:
            p = (match + 1) / (total + 1)
        logs += math.log(p) / 4
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    return bp * math.exp(logs)


# ---------------------------------------------------------------------------
# record evaluation
# ---------------------------------------------------------------------------
def evaluate_record(rec: Dict, bundle, alerts: Optional[List[frozenset]] = None, validity: Optional[Dict] = None) -> Dict:
    """Evaluate one emitted bundle against the record's gold.

    ``alerts`` is the list of component sets the system raised; when None the
    system raised no alerts. ``validity`` may pass a precomputed validator result.
    """
    G = bundle_facts(rec["gold"])
    P = bundle_facts(bundle)
    v = validity if validity is not None else validate_bundle(bundle)
    out = {"valid": bool(v["valid"])}
    tp = fp = fn = 0
    crit, major, minor = Counter(), Counter(), Counter()
    strict = 0
    slots = Counter()
    for kind in ("condition", "drug", "allergen"):
        g, p = G[kind], P[kind]
        tpk = set(g) & set(p)
        tp += len(tpk)
        fp += len(set(p) - set(g))
        fn += len(set(g) - set(p))
        for k in set(p) - set(g):
            crit[f"hallucinated_{kind}"] += 1
        for k in set(g) - set(p):
            if kind == "condition":
                major["missed_condition"] += 1
            elif kind == "drug":
                (crit if g[k]["status"] == "active" else major)["missed_drug"] += 1
            else:
                crit["missed_allergen"] += 1
        for k in tpk:
            ga, pa = g[k], p[k]
            ok = True
            if kind == "condition":
                if ga["status"] != pa["status"]:
                    major["condition_status"] += 1
                    ok = False
                slots[slot_outcome(ga["onset"], pa["onset"])] += 1
            elif kind == "drug":
                if ga["status"] != pa["status"]:
                    crit["drug_status"] += 1
                    ok = False
                if pa["dose"] is None:
                    major["missing_dose"] += 1
                    ok = False
                elif ga["dose"] != pa["dose"]:
                    (crit if k in HIGH_ALERT else major)["dose_error"] += 1
                    ok = False
                if ga["freq"] != pa["freq"]:
                    major["frequency"] += 1
                    ok = False
                if ga["route"] and ga["route"] != pa["route"]:
                    minor["route"] += 1
                if ga["prn"] != pa["prn"]:
                    minor["prn"] += 1
                slots[slot_outcome(ga["authored"], pa["authored"])] += 1
            else:
                if ga["reactions"] and ga["reactions"] != pa["reactions"]:
                    major["allergy_reaction"] += 1
                    ok = False
                if ga["severity"] and ga["severity"] != pa["severity"]:
                    minor["allergy_severity"] += 1
                slots[slot_outcome(ga["onset"], pa["onset"])] += 1
            strict += ok
    # labs: multiset over (key, value); effective date as a slot
    gl = Counter((k, val) for k, val, _ in G["lab"])
    pl = Counter((k, val) for k, val, _ in P["lab"])
    ltp = sum((gl & pl).values())
    tp += ltp
    fp += sum((pl - gl).values())
    fn += sum((gl - pl).values())
    strict += ltp
    if sum((gl - pl).values()):
        major["missed_lab"] += sum((gl - pl).values())
    if sum((pl - gl).values()):
        minor["spurious_lab"] += sum((pl - gl).values())
    geff = {}
    for k, val, eff in G["lab"]:
        geff.setdefault((k, val), []).append(eff)
    for k, val, eff in P["lab"]:
        if geff.get((k, val)):
            slots[slot_outcome(geff[(k, val)].pop(0), eff)] += 1
    # unknown codes are hallucinated facts
    fp += P["unknown"]
    if P["unknown"]:
        crit["unknown_code"] += P["unknown"]
    ge, pe = G["encounter"], P["encounter"]
    for k in ("start", "end"):
        slots[slot_outcome(ge.get(k), pe.get(k))] += 1
    slots.pop(None, None)
    for s, n in slots.items():
        if s == "fabricated":
            crit["fabricated_timestamp"] += n
        elif s in ("wrong", "unsupported_text"):
            major["wrong_timestamp"] += n
        elif s == "missing":
            minor["missing_timestamp"] += n
    n_gold = sum(len(G[k]) for k in ("condition", "drug", "allergen")) + len(G["lab"])
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec_ = tp / (tp + fn) if tp + fn else 1.0
    out.update(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec_,
               cf1=2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0,
               spi=strict / n_gold if n_gold else 1.0, n_gold=n_gold,
               halluc=fp / (tp + fp) if tp + fp else 0.0, n_pred=tp + fp)
    n_slots = sum(slots.values())
    precise_pred = slots["correct"] + slots["wrong"] + slots["fabricated"]
    out.update(tf=slots["correct"] / n_slots if n_slots else 1.0, n_slots=n_slots,
               fabricated=slots["fabricated"], precise_pred=precise_pred, slot_counts=dict(slots))
    sat, napp = axioms.tcr(bundle, axioms.ALL_AXIOMS)
    sa, na = axioms.tcr(bundle, axioms.AGENT_AXIOMS)
    sh, nh = axioms.tcr(bundle, axioms.HELDOUT_AXIOMS)
    out.update(tcr_sat=sat, tcr_n=napp, tcr_agent_sat=sa, tcr_agent_n=na, tcr_held_sat=sh, tcr_held_n=nh)
    # pharmacovigilance
    gold_sig = [frozenset(s["components"]) for s in rec["gold_signals"]]
    gold_major = [frozenset(s["components"]) for s in rec["gold_signals"] if s["severity"] == "major"]
    al = list(alerts or [])
    matched_gold = [g for g in gold_sig if any(g <= a and len(a) <= 3 for a in al)]
    matched_alert = [a for a in al if any(g <= a for g in gold_sig) and len(a) <= 3]
    missed_major = [g for g in gold_major if not any(g <= a for a in al)]
    if missed_major:
        crit["missed_major_signal"] += len(missed_major)
    out.update(pv_tp=len(matched_gold), pv_fn=len(gold_sig) - len(matched_gold), pv_alerts=len(al),
               pv_fp=len(al) - len(matched_alert), pv_gold_pos=bool(gold_sig), pv_pred_pos=bool(al))
    if not v["valid"]:
        crit["fhir_invalid"] += 1
    out["critical"] = dict(crit)
    out["major"] = dict(major)
    out["minor"] = dict(minor)
    out["n_critical"] = sum(crit.values())
    out["acceptable"] = out["n_critical"] == 0
    out["swes"] = 3 * sum(crit.values()) + sum(major.values()) + 0.25 * sum(minor.values())
    out["medbleu"] = bleu4(linearize(P), linearize(G))
    return out


def empty_eval(rec: Dict) -> Dict:
    """Evaluation of an abstention (no bundle emitted), used for full-set views."""
    return evaluate_record(rec, {"resourceType": "Bundle", "type": "collection", "entry": []}, [],
                           {"valid": False, "errors": [], "by_kind": {}})
