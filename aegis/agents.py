"""Aegis-EHR agents, orchestrator and deterministic baselines.

The orchestrator is central: it routes work to the agents in a fixed
dependency order (Schema -> Reasoning -> Temporal -> Pharmacovigilance, with
Governance observing the stream), merges their repair proposals, aggregates
their calibrated confidences and decides AUTO / REVIEW / ABSTAIN. Agents
communicate only through typed reports (``AgentReport``), which is the
contract the MCP transport carries in a deployed system.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import axioms, fhir
from .knowledge import fire_rules
from .metrics import bundle_facts, state_from_facts
from .nlp import (EVENT_CONCEPTS, MED_SECTIONS, UNIT_MAP, Lexicon, Parse, TExpr, parse_date_token, parse_note)
from .terminology import (ALLERGEN_BY_KEY, COND_BY_KEY, DRUG_BY_KEY, FREQS, LAB_BY_KEY, REACTIONS, ROUTES, SNOMED)

UCUM_OF = {"mg": "mg", "ug": "ug", "g": "g", "[iU]": "[iU]", "meq": "meq", "{puff}": "{puff}"}
RAW_UNIT_TEXT = {"mg": "mg", "ug": "mcg", "g": "g", "[iU]": "units", "meq": "mEq", "{puff}": "puffs"}


@dataclass
class AgentReport:
    agent: str
    features: Dict[str, float] = field(default_factory=dict)
    raw_confidence: float = 1.0
    actions: List[Tuple[str, str]] = field(default_factory=list)   # (action, detail) with provenance


# ---------------------------------------------------------------------------
# Building bundles from a parse
# ---------------------------------------------------------------------------
def _reaction(window: str):
    low = window.lower()
    rx = None
    for r in REACTIONS:
        if any(f in low for f in r["forms"]):
            rx = r
            break
    sev = next((s for s in ("mild", "moderate", "severe") if s in low), None)
    if rx and rx["key"] == "anaphylaxis":
        sev = "severe"
    return rx, sev


def reconciled_meds(P: Parse, reconcile: bool) -> Dict[str, Dict]:
    """Per-drug attributes. With ``reconcile`` the latest statement wins and plan
    actions are applied; without it the first statement is used and actions are ignored."""
    meds: Dict[str, Dict] = {}
    order = ["meds", "plan", "dcmeds"]
    for info in P.meds:
        cur = meds.get(info.key)
        rec = dict(dose=info.dose, unit=info.unit, route=info.route, freq=info.freq, prn=info.prn,
                   texpr=info.texpr, section=info.section, status="active",
                   category="inpatient" if info.section == "plan" else "community")
        if cur is None:
            meds[info.key] = rec
            continue
        if reconcile:
            for k in ("dose", "unit", "route", "freq"):
                if rec[k] is not None and info.section in ("dcmeds", "plan"):
                    cur[k] = rec[k]
                elif cur[k] is None:
                    cur[k] = rec[k]
            if cur["texpr"] is None:
                cur["texpr"] = rec["texpr"]
        else:
            for k in ("dose", "unit", "route", "freq", "texpr"):
                if cur[k] is None:
                    cur[k] = rec[k]
    if reconcile:
        for act, key, val, unit in P.actions:
            if key not in meds:
                continue
            if act == "stop":
                meds[key]["status"] = "stopped"
            elif act == "hold":
                meds[key]["status"] = "on-hold"
            elif act == "dose" and val is not None:
                meds[key]["dose"] = val
                if unit:
                    meds[key]["unit"] = unit
    return meds


def _texpr_fields(t: Optional[TExpr], allow_text: bool, naive: bool = False):
    if t is None:
        return None, None
    if t.kind == "vague":
        return None, (t.text if allow_text else None)
    if naive and t.kind == "absolute":
        return t.text, None          # copied as written, no normalisation
    return t.value, None


def _dose_fields(m: Dict, naive: bool):
    dose, unit = m.get("dose"), m.get("unit")
    if dose is None:
        return None, None
    if unit is None:
        return None, None
    if naive:
        return dose, RAW_UNIT_TEXT.get(unit, unit)
    if unit == "g":
        return dose * 1000.0, "mg"
    return dose, UCUM_OF.get(unit, unit)


def build_bundle(P: Parse, text: str, mode: str, naive: bool = False) -> Tuple[Dict, Dict]:
    """Serialize a parse into a FHIR bundle. Returns (bundle, provenance map)."""
    full = mode == "full"
    res = [fhir.patient("p1", P.gender or "unknown")]
    status = "finished" if P.note_type == "dc" else "in-progress"
    start = P.admit.isoformat() if P.admit else None
    end = P.discharge.isoformat() if P.discharge else None
    res.append(fhir.encounter("e1", "p1", status, "IMP", start, end))
    recorded = P.note_date.isoformat() if P.note_date else None
    prov = {}
    # conditions
    seen = {}
    neg_only = set()
    for mt in P.mentions:
        if mt.kind != "condition":
            continue
        if mode in ("single", "full") and mt.assertion != "present":
            continue
        if mt.key not in seen:
            seen[mt.key] = dict(texpr=mt.texpr, historical=mt.historical, match=mt.match)
        else:
            if seen[mt.key]["texpr"] is None and mt.texpr is not None:
                seen[mt.key]["texpr"] = mt.texpr
            seen[mt.key]["historical"] |= mt.historical
    for i, (k, info) in enumerate(seen.items(), 1):
        c = COND_BY_KEY[k]
        clinical = "resolved" if full and info["historical"] and k in EVENT_CONCEPTS else "active"
        onset, onset_txt = _texpr_fields(info["texpr"], allow_text=full, naive=naive)
        r = fhir.condition(f"c{i}", "p1", c["snomed"], c["display"], clinical, onset, onset_txt, recorded, c["icd10"])
        res.append(r)
        prov[f"Condition/c{i}"] = info["match"]
    # medications
    meds = reconciled_meds(P, reconcile=full)
    for i, (k, m) in enumerate(meds.items(), 1):
        d = DRUG_BY_KEY[k]
        authored, _ = _texpr_fields(m["texpr"], allow_text=False, naive=naive)
        dose, unit = _dose_fields(m, naive)
        route = m["route"]
        freq = FREQS[m["freq"]]["repeat"] if m["freq"] and not naive else None
        r = fhir.medication_request(f"m{i}", "p1", d["rxcui"], d["display"], m["status"], m["category"], dose, unit,
                                    ROUTES[route]["snomed"] if route and not naive else None, freq, m["prn"], authored,
                                    ROUTES[route]["display"] if route and not naive else None,
                                    freq_text=(m["freq"] if naive and m["freq"] else None))
        if naive and route:
            r.setdefault("dosageInstruction", [{}])[0]["route"] = {"text": route}
        res.append(r)
    # allergies
    seen_a = {}
    for mt in P.mentions:
        if mt.kind != "allergen" or mt.section not in ("allergies", "narrative", "unknown"):
            continue
        if mode in ("single", "full") and mt.assertion != "present":
            continue
        if mt.key in seen_a:
            continue
        win = text[mt.end:mt.end + 60].split(";")[0]
        rx, sev = _reaction(win)
        seen_a[mt.key] = (rx, sev, mt.texpr)
    for i, (k, (rx, sev, t)) in enumerate(seen_a.items(), 1):
        a = ALLERGEN_BY_KEY[k]
        onset, onset_txt = _texpr_fields(t, allow_text=full, naive=naive)
        r = fhir.allergy(f"a{i}", "p1", a["snomed"], a["display"], [(rx["snomed"], rx["display"])] if rx else None,
                         sev if rx else None, onset)
        if onset_txt:
            r["onsetString"] = onset_txt
        res.append(r)
    # labs
    for i, (k, v, eff) in enumerate(P.labs, 1):
        lab = LAB_BY_KEY[k]
        val = int(v) if lab["decimals"] == 0 and float(v).is_integer() else v
        res.append(fhir.observation(f"o{i}", "p1", lab["loinc"], lab["display"], val, lab["unit"], eff))
    return fhir.bundle(res), prov


# ---------------------------------------------------------------------------
# Schema Agent: structural mapping, validation, structural repair
# ---------------------------------------------------------------------------
ALIASES = {"MedicationRequest": {"medication": "medicationCodeableConcept", "dosage": "dosageInstruction",
                                 "patient": "subject"},
           "Condition": {"patient": "subject", "onset": "onsetDateTime", "recorded": "recordedDate"},
           "AllergyIntolerance": {"subject": "patient", "onset": "onsetDateTime", "substance": "code"},
           "Observation": {"effective": "effectiveDateTime", "value": "valueQuantity", "patient": "subject"}}
UNIT_FIX = {"mcg": "ug", "units": "[iU]", "unit": "[iU]", "u": "[iU]", "iu": "[iU]", "meq": "meq", "puffs": "{puff}",
            "puff": "{puff}", "mmol/l": "mmol/L", "mg/dl": "mg/dL", "g/dl": "g/dL", "ml/min/1.73m2": "mL/min/{1.73_m2}",
            "miu/l": "m[IU]/L", "": None}


def _fix_date(s):
    if not isinstance(s, str):
        return None
    if fhir.valid_datetime(s):
        return s
    p = parse_date_token(s.strip().split("T")[0])
    if p and p[0]:
        return {"day": p[0].isoformat(), "month": f"{p[0].year:04d}-{p[0].month:02d}", "year": f"{p[0].year:04d}"}[p[1]]
    return None


def schema_repair(b, gender: Optional[str]) -> Tuple[Optional[Dict], List[Tuple[str, str]]]:
    """Structural repair of an external draft (e.g. an LLM bundle). Content is
    never invented: unknown elements are dropped or renamed to their R4 name,
    dates and units are re-serialized only when the original value parses, and
    resources that still fail validation are removed (and logged)."""
    actions = []
    if not isinstance(b, dict):
        return None, [("abstain", "draft is not a JSON object")]
    b = copy.deepcopy(b)
    if b.get("resourceType") != "Bundle" or not isinstance(b.get("entry"), list):
        return None, [("abstain", "draft is not a Bundle")]
    b["type"] = b.get("type") if b.get("type") in fhir.VS["bundle_type"] else "collection"
    entries = []
    for e in b["entry"]:
        r = e.get("resource") if isinstance(e, dict) else None
        if not isinstance(r, dict) or r.get("resourceType") not in fhir.SUPPORTED:
            actions.append(("drop", "entry without a supported resource"))
            continue
        rt = r["resourceType"]
        for old, new in ALIASES.get(rt, {}).items():
            if old in r and new not in r:
                val = r.pop(old)
                if new == "onsetDateTime" and isinstance(val, str) and not _fix_date(val):
                    r["onsetString"] = val
                else:
                    r[new] = val
                actions.append(("rename", f"{rt}.{old}->{new}"))
        allowed = fhir.DOMAIN["common"] | fhir.DOMAIN[rt]
        for k in list(r):
            if k not in allowed or r[k] in (None, "", [], {}):
                r.pop(k)
                actions.append(("drop_element", f"{rt}.{k}"))
        if not isinstance(r.get("id"), str) or not fhir._ID_RE.match(r.get("id", "")):
            r["id"] = f"x{len(entries) + 1}"
            actions.append(("reid", rt))
        for k in ("onsetDateTime", "recordedDate", "authoredOn", "effectiveDateTime", "abatementDateTime"):
            if k in r:
                f = _fix_date(r[k])
                if f is None:
                    if k == "onsetDateTime":
                        r["onsetString"] = str(r.pop(k))
                        actions.append(("date_to_text", f"{rt}.{k}"))
                    else:
                        r.pop(k)
                        actions.append(("drop_date", f"{rt}.{k}"))
                elif f != r[k]:
                    r[k] = f
                    actions.append(("normalize_date", f"{rt}.{k}"))
        if rt == "Encounter" and isinstance(r.get("period"), dict):
            for k in ("start", "end"):
                if k in r["period"]:
                    f = _fix_date(r["period"][k])
                    if f:
                        r["period"][k] = f
                    else:
                        r["period"].pop(k)
        if rt in ("Condition", "MedicationRequest", "Observation", "Encounter") and "subject" not in r:
            r["subject"] = {"reference": "Patient/p1"}
            actions.append(("add_subject", rt))
        if rt == "AllergyIntolerance" and "patient" not in r:
            r["patient"] = {"reference": "Patient/p1"}
        for refk in ("subject", "patient"):
            if isinstance(r.get(refk), dict) and str(r[refk].get("reference", "")).startswith("Patient/"):
                r[refk] = {"reference": "Patient/p1"}
        if rt == "Patient":
            r["id"] = "p1"
        _fix_units(r, actions)
        if rt == "MedicationRequest":
            if r.get("status") not in fhir.VS["mr_status"]:
                r["status"] = "unknown"
                actions.append(("status_unknown", "MedicationRequest.status"))
            if r.get("intent") not in fhir.VS["mr_intent"]:
                r["intent"] = "order"
                actions.append(("intent_default", "MedicationRequest.intent"))
        if rt == "Observation" and r.get("status") not in fhir.VS["obs_status"]:
            r["status"] = "unknown"
        if rt == "Encounter":
            if r.get("status") not in fhir.VS["encounter_status"]:
                r["status"] = "unknown"
            if not isinstance(r.get("class"), dict):
                r["class"] = {"system": fhir.CS_ACT, "code": "IMP"}
        entries.append({"fullUrl": f"{fhir.BASE}/{rt}/{r['id']}", "resource": r})
    # unique ids and one Patient
    seen, out = set(), []
    pats = [e for e in entries if e["resource"]["resourceType"] == "Patient"]
    if not pats:
        out.append({"fullUrl": f"{fhir.BASE}/Patient/p1", "resource": fhir.patient("p1", gender or "unknown")})
        seen.add("Patient/p1")
        actions.append(("add_patient", "missing Patient"))
    for e in entries:
        r = e["resource"]
        key = f"{r['resourceType']}/{r['id']}"
        if r["resourceType"] == "Patient" and "Patient/p1" in seen:
            continue
        while key in seen:
            r["id"] = r["id"] + "x"
            key = f"{r['resourceType']}/{r['id']}"
        e["fullUrl"] = f"{fhir.BASE}/{key}"
        seen.add(key)
        out.append(e)
    b["entry"] = out
    for k in list(b):
        if k not in fhir.DT_KEYS["Bundle"]:
            b.pop(k)
    # drop resources that still fail
    v = fhir.validate_bundle(b)
    if not v["valid"]:
        bad = set()
        for kind, msg in v["errors"]:
            if msg.startswith("entry["):
                bad.add(int(msg[6:msg.index("]")]))
        if bad:
            b["entry"] = [e for i, e in enumerate(b["entry"]) if i not in bad or e["resource"]["resourceType"] == "Patient"]
            actions += [("drop_resource", "failed validation after repair")] * len(bad)
            ids = {f"{e['resource']['resourceType']}/{e['resource']['id']}" for e in b["entry"]}
            for e in b["entry"]:
                r = e["resource"]
                if r["resourceType"] == "DetectedIssue":
                    r["implicated"] = [x for x in r.get("implicated", []) if x.get("reference") in ids]
                    if not r["implicated"]:
                        r.pop("implicated")
    return b, actions


def _fix_units(r, actions):
    def fixq(q):
        if not isinstance(q, dict):
            return
        u = q.get("code") or q.get("unit")
        if u is None:
            return
        uu = UNIT_FIX.get(str(u).lower(), str(u))
        if uu is None:
            return
        if uu not in fhir.UCUM_OK:
            return
        if q.get("code") != uu or q.get("system") != fhir.UCUM:
            q["code"], q["system"] = uu, fhir.UCUM
            actions.append(("normalize_unit", str(u)))
    if r.get("resourceType") == "Observation":
        fixq(r.get("valueQuantity"))
    for d in r.get("dosageInstruction", []) or []:
        if isinstance(d, dict):
            for x in d.get("doseAndRate", []) or []:
                if isinstance(x, dict):
                    fixq(x.get("doseQuantity"))


class SchemaAgent:
    name = "schema"

    def draft(self, text: str, lex: Lexicon, headers, naive: bool = False):
        P = parse_note(text, lex, "schema", headers)
        b, prov = build_bundle(P, text, "schema", naive=naive)
        return b, P

    def check(self, b, pre_valid_result, actions) -> AgentReport:
        v = fhir.validate_bundle(b) if b is not None else {"valid": False, "errors": [("x", "no bundle")]}
        n_err = len(pre_valid_result["errors"])
        dropped = sum(1 for a in actions if a[0] == "drop_resource")
        rep = AgentReport("schema", features=dict(pre_valid=float(pre_valid_result["valid"]), log_err=math.log1p(n_err),
                                                  n_dropped=float(dropped), post_valid=float(v["valid"]),
                                                  n_repairs=float(len(actions))),
                          raw_confidence=1.0 if pre_valid_result["valid"] else max(0.0, 1 - 0.1 * n_err),
                          actions=actions)
        rep.post = v
        return rep


# ---------------------------------------------------------------------------
# Reasoning Agent: semantic audit against the source, entity-level repair
# ---------------------------------------------------------------------------
class ReasoningAgent:
    name = "reasoning"

    def audit(self, b, F: Parse, text: str, repair: bool = True, lex: Optional[Lexicon] = None,
              grounding: bool = True) -> Tuple[Dict, AgentReport]:
        recoded = dropped_ungrounded = 0
        grounded_set = {(m.kind, m.key) for m in F.mentions}
        acts0 = []
        if repair and lex is not None:
            b, recoded = self._recode_from_display(b, lex, grounded_set, acts0)
        D = bundle_facts(b)
        present = {"condition": {}, "drug": {}, "allergen": {}}
        negated = {"condition": set(), "drug": set(), "allergen": set()}
        for mt in F.mentions:
            if mt.kind in present:
                if mt.assertion == "present" and (mt.kind != "allergen" or mt.section in ("allergies", "narrative", "unknown")) \
                        and (mt.kind != "drug" or mt.section in MED_SECTIONS):
                    present[mt.kind].setdefault(mt.key, mt)
                elif mt.assertion != "present":
                    negated[mt.kind].add(mt.key)
        ref_bundle, _ = build_bundle(F, text, "full")
        R = bundle_facts(ref_bundle)
        ref_res = {}
        for e in ref_bundle["entry"]:
            r = e["resource"]
            key = R["ids"].get(f"{r['resourceType']}/{r['id']}")
            if key:
                ref_res[(r["resourceType"], key)] = r
        out = copy.deepcopy(b)
        acts = []
        supported = contradicted = unverified = inserted = changed = 0
        kind_rt = {"condition": "Condition", "drug": "MedicationRequest", "allergen": "AllergyIntolerance"}
        keep = []
        for e in out["entry"]:
            r = e["resource"]
            rt = r["resourceType"]
            key = D["ids"].get(f"{rt}/{r.get('id')}")
            kind = next((k for k, v in kind_rt.items() if v == rt), None)
            if kind is None or key is None:
                keep.append(e)
                continue
            if key in present[kind]:
                supported += 1
                if repair and kind == "drug":
                    changed += self._reconcile_med(r, R["drug"].get(key), acts)
                if repair and kind == "condition":
                    want = R["condition"].get(key, {}).get("status")
                    cur = (r.get("clinicalStatus") or {}).get("coding", [{}])[0].get("code") if isinstance(r.get("clinicalStatus"), dict) else None
                    if want and cur != want:
                        r["clinicalStatus"] = fhir.cc(fhir.CS_CLINICAL, want, want.capitalize())
                        changed += 1
                        acts.append(("set_status", f"Condition {key} -> {want}"))
                if repair and kind == "allergen" and "reaction" not in r and ("AllergyIntolerance", key) in ref_res \
                        and "reaction" in ref_res[("AllergyIntolerance", key)]:
                    r["reaction"] = copy.deepcopy(ref_res[("AllergyIntolerance", key)]["reaction"])
                    changed += 1
                keep.append(e)
            elif key in negated[kind]:
                contradicted += 1
                acts.append(("remove_contradicted", f"{rt} {key}"))
                if not repair:
                    keep.append(e)
            else:
                unverified += 1
                if repair and grounding and (kind, key) not in grounded_set:
                    dropped_ungrounded += 1
                    acts.append(("drop_ungrounded", f"{rt} {key}"))
                    continue
                keep.append(e)
        out["entry"] = keep
        acts = acts0 + acts
        if repair:
            have = bundle_facts(out)
            n = 0
            for kind, rt in kind_rt.items():
                for key in present[kind]:
                    if key not in have[kind] and (rt, key) in ref_res:
                        r = copy.deepcopy(ref_res[(rt, key)])
                        for tf in ("onsetDateTime", "onsetString", "authoredOn"):
                            r.pop(tf, None)       # timestamps are the Temporal Agent's mandate
                        n += 1
                        r["id"] = f"r{n}"
                        out["entry"].append({"fullUrl": f"{fhir.BASE}/{rt}/{r['id']}", "resource": r})
                        inserted += 1
                        acts.append(("insert_missing", f"{rt} {key}"))
            # laboratory values: keep values found in the source, add missing ones
            src_labs = {(k, round(float(v), 3)) for k, v, _ in F.labs}
            new_entries = []
            for e in out["entry"]:
                r = e["resource"]
                if r["resourceType"] == "Observation":
                    lab = bundle_facts({"resourceType": "Bundle", "entry": [e]})["lab"]
                    if lab and (lab[0][0], lab[0][1]) not in src_labs and src_labs:
                        acts.append(("remove_unsupported_lab", lab[0][0]))
                        changed += 1
                        continue
                new_entries.append(e)
            out["entry"] = new_entries
            have_labs = {(k, v) for k, v, _ in bundle_facts(out)["lab"]}
            for (k, v, eff) in F.labs:
                if (k, round(float(v), 3)) not in have_labs:
                    lab = LAB_BY_KEY[k]
                    n += 1
                    r = fhir.observation(f"r{n}", "p1", lab["loinc"], lab["display"], v, lab["unit"], None)
                    out["entry"].append({"fullUrl": f"{fhir.BASE}/Observation/r{n}", "resource": r})
                    have_labs.add((k, round(float(v), 3)))
                    acts.append(("insert_lab", k))
                    inserted += 1
        n_draft = supported + contradicted + unverified
        n_m = max(1, len([m for m in F.mentions if m.kind in ("condition", "drug", "allergen")]))
        frac = lambda t: sum(1 for m in F.mentions if m.match == t) / n_m
        oov = F.oov_tokens / max(F.clinical_tokens, 1)
        feats = dict(sup_frac=supported / max(n_draft, 1), contradicted=float(contradicted), unverified=float(unverified),
                     recoded=float(recoded), ungrounded=float(dropped_ungrounded),
                     inserted=float(inserted), changed=float(changed), frac_learned=frac("learned"),
                     frac_fuzzy=frac("fuzzy"), frac_ambig=frac("ambiguous"), oov=oov,
                     unknown_headers=float(F.unknown_headers), n_draft=float(n_draft))
        mean_score = sum(m.score for m in F.mentions) / max(len(F.mentions), 1)
        raw = mean_score * (1 - oov) * (supported / max(n_draft, 1) if n_draft else 0.5)
        return out, AgentReport("reasoning", feats, raw, acts)

    @staticmethod
    def _recode_from_display(b, lex: Lexicon, grounded_set, acts):
        """If a coding's display names a different concept than its code, and only
        the displayed concept is found in the note, use the displayed concept."""
        from .nlp import norm, tokenize
        from .terminology import ALLERGEN_BY_CODE, COND_BY_CODE, DRUG_BY_CODE, RXNORM
        out = copy.deepcopy(b)
        n = 0
        spec = {"Condition": ("code", SNOMED, "condition", COND_BY_CODE, COND_BY_KEY, "snomed"),
                "MedicationRequest": ("medicationCodeableConcept", RXNORM, "drug", DRUG_BY_CODE, DRUG_BY_KEY, "rxcui"),
                "AllergyIntolerance": ("code", SNOMED, "allergen", ALLERGEN_BY_CODE, ALLERGEN_BY_KEY, "snomed")}
        for e in out.get("entry", []) or []:
            r = e.get("resource") if isinstance(e, dict) else None
            if not isinstance(r, dict) or r.get("resourceType") not in spec:
                continue
            field_, system, kind, by_code, by_key, code_attr = spec[r["resourceType"]]
            cc = r.get(field_)
            if not isinstance(cc, dict):
                continue
            for c in cc.get("coding", []) or []:
                if not isinstance(c, dict) or c.get("system") != system or not isinstance(c.get("display"), str):
                    continue
                cur = by_code.get(c.get("code", ""))
                cur_key = cur["key"] if cur else None
                toks = [norm(t) for t, _, _ in tokenize(c["display"])]
                m = lex.match(toks, 0)
                cands = m[1] if m and m[0] == len(toks) else [(k_, key_, "fuzzy", 1.0) for k_, key_, _ in lex.fuzzy(" ".join(toks))]
                disp = next((key_ for k_, key_, *_ in cands if k_ == kind), None)
                if disp and disp != cur_key and (kind, disp) in grounded_set and (kind, cur_key) not in grounded_set:
                    tgt = by_key[disp]
                    c["code"] = tgt[code_attr]
                    c["display"] = tgt["display"]
                    n += 1
                    acts.append(("recode_from_display", f"{r['resourceType']} {cur_key}->{disp}"))
        return out, n

    @staticmethod
    def _reconcile_med(r, ref, acts) -> int:
        if not ref:
            return 0
        n = 0
        if ref["status"] and r.get("status") != ref["status"]:
            acts.append(("set_status", f"{r.get('id')} {r.get('status')}->{ref['status']}"))
            r["status"] = ref["status"]
            n += 1
        dos = (r.get("dosageInstruction") or [{}])
        if not isinstance(dos, list) or not dos or not isinstance(dos[0], dict):
            dos = [{}]
        d = dos[0]
        cur = None
        if isinstance(d.get("doseAndRate"), list) and d["doseAndRate"] and isinstance(d["doseAndRate"][0], dict):
            from .metrics import norm_dose
            cur = norm_dose(d["doseAndRate"][0].get("doseQuantity"))
        if ref["dose"] and cur != ref["dose"]:
            v, u = ref["dose"]
            d["doseAndRate"] = [{"doseQuantity": {"value": v, "unit": u, "system": fhir.UCUM, "code": u}}]
            acts.append(("set_dose", f"{r.get('id')} -> {v} {u}"))
            n += 1
        from .metrics import per_day, route_key
        if ref["freq"] and per_day(d) != ref["freq"]:
            for fk, fv in FREQS.items():
                f, p, u = fv["repeat"]
                if round(f * {"h": 24.0, "d": 1.0, "wk": 1 / 7.0}[u] / p, 4) == ref["freq"]:
                    d["timing"] = {"repeat": {"frequency": f, "period": p, "periodUnit": u}}
                    n += 1
                    break
        if ref["route"] and route_key(d) != ref["route"]:
            d["route"] = {"coding": [{"system": SNOMED,
                                      "code": ROUTES[ref["route"]]["snomed"], "display": ROUTES[ref["route"]]["display"]}]}
            n += 1
        if ref["prn"] and not d.get("asNeededBoolean"):
            d["asNeededBoolean"] = True
        r["dosageInstruction"] = [d]
        return n


# ---------------------------------------------------------------------------
# Temporal Agent: source-grounded timestamps and axiom checking
# ---------------------------------------------------------------------------
class TemporalAgent:
    name = "temporal"

    def run(self, b, F: Parse, text: str, policy: str = "provenance", pid: str = "p1") -> Tuple[Dict, AgentReport]:
        out = copy.deepcopy(b)
        ref_bundle, _ = build_bundle(F, text, "full")
        R = bundle_facts(ref_bundle)
        D = bundle_facts(out)
        stats = dict(kept=0, corrected_explicit=0, corrected_derived=0, inserted_explicit=0, inserted_derived=0,
                     reduced=0, removed=0, flagged=0, shifted=0, slots=0, vague=0)
        acts = []
        src_kind = {}
        for mt in F.mentions:
            if mt.texpr is not None:
                src_kind.setdefault((mt.kind, mt.key), mt.texpr.kind)
        for e in out["entry"]:
            r = e["resource"]
            rt = r["resourceType"]
            key = D["ids"].get(f"{rt}/{r.get('id')}")
            if rt == "Condition" and key:
                self._slot(r, "onsetDateTime", "onsetString", R["condition"].get(key, {}).get("onset"),
                           src_kind.get(("condition", key)), policy, stats, acts, rt + " " + key, key in R["condition"])
                if F.note_date and policy == "provenance":
                    r["recordedDate"] = F.note_date.isoformat()
            elif rt == "MedicationRequest" and key:
                self._slot(r, "authoredOn", None, R["drug"].get(key, {}).get("authored"),
                           src_kind.get(("drug", key)), policy, stats, acts, rt + " " + key, key in R["drug"])
            elif rt == "AllergyIntolerance" and key:
                self._slot(r, "onsetDateTime", "onsetString", R["allergen"].get(key, {}).get("onset"),
                           src_kind.get(("allergen", key)), policy, stats, acts, rt + " " + key, key in R["allergen"])
            elif rt == "Observation":
                lab = bundle_facts({"resourceType": "Bundle", "entry": [e]})["lab"]
                if lab:
                    cand = [eff for k, v, eff in R["lab"] if k == lab[0][0] and v == lab[0][1]]
                    self._slot(r, "effectiveDateTime", None, cand[0] if cand else None, "absolute", policy, stats, acts,
                               "Observation " + lab[0][0], bool(cand))
            elif rt == "Encounter" and policy == "provenance":
                per = r.get("period") if isinstance(r.get("period"), dict) else {}
                want = {k: v for k, v in (("start", F.admit), ("end", F.discharge)) if v}
                for k in ("start", "end"):
                    if k in want and per.get(k) != want[k].isoformat():
                        per[k] = want[k].isoformat()
                        stats["corrected_explicit"] += 1
                    elif k not in want and k in per:
                        per.pop(k)
                        stats["removed"] += 1
                if per:
                    r["period"] = per
        # axiom check on the agent's own axioms
        res = axioms.check(out, axioms.AGENT_AXIOMS)
        viol = sum(1 for v in res.values() for x in v if not x)
        if viol:
            if policy == "closest_valid":
                stats["shifted"] += self._shift(out)
            else:
                stats["flagged"] += viol
                implicated = [f"{e['resource']['resourceType']}/{e['resource']['id']}" for e in out["entry"]
                              if e["resource"]["resourceType"] in ("Condition", "MedicationRequest")][:2]
                di = fhir.detected_issue(f"t{len(out['entry'])}", pid, "TIME", "moderate",
                                         f"{viol} temporal axiom violation(s) present in the source; values preserved",
                                         implicated)
                out["entry"].append({"fullUrl": f"{fhir.BASE}/DetectedIssue/{di['id']}", "resource": di})
                acts.append(("flag", f"{viol} violations preserved"))
        n = max(stats["slots"], 1)
        feats = dict(slots=float(stats["slots"]), vague=stats["vague"] / n,
                     corrected=(stats["corrected_explicit"] + stats["corrected_derived"]) / n,
                     reduced=stats["reduced"] / n, removed=stats["removed"] / n, flagged=float(stats["flagged"]),
                     derived=(stats["corrected_derived"] + stats["inserted_derived"]) / n)
        raw = 1 - (stats["vague"] + stats["removed"] + 2 * stats["flagged"]) / max(n, 1)
        rep = AgentReport("temporal", feats, max(0.0, raw), acts)
        rep.stats = stats
        return out, rep

    @staticmethod
    def _slot(r, dt_field, txt_field, src, src_kind, policy, stats, acts, label, known):
        cur_dt = r.get(dt_field)
        cur_txt = r.get(txt_field) if txt_field else None
        stats["slots"] += 1
        if isinstance(src, tuple):
            stats["vague"] += 1
        derived = src_kind in ("relative", "hd")
        if policy == "closest_valid":
            # original policy: keep whatever the draft has; only fill gaps from the source
            if cur_dt is None and cur_txt is None and src is not None:
                if isinstance(src, str):
                    r[dt_field] = src
                    stats["inserted_derived" if derived else "inserted_explicit"] += 1
                elif txt_field:
                    r[txt_field] = src[1]
            else:
                stats["kept"] += 1
            return
        if not known:
            return
        if isinstance(src, str):
            if cur_dt == src:
                stats["kept"] += 1
            else:
                kind = "corrected" if (cur_dt is not None or cur_txt is not None) else "inserted"
                stats[f"{kind}_{'derived' if derived else 'explicit'}"] += 1
                acts.append((f"{kind}_timestamp", f"{label} {cur_dt}->{src} ({src_kind})"))
                r[dt_field] = src
                if txt_field and txt_field in r:
                    r.pop(txt_field)
        elif isinstance(src, tuple):
            if cur_dt is not None:
                r.pop(dt_field)
                stats["reduced"] += 1
                acts.append(("reduce_precision", f"{label} {cur_dt} -> '{src[1]}'"))
            if txt_field:
                r[txt_field] = src[1]
        else:
            if cur_dt is not None:
                r.pop(dt_field)
                stats["removed"] += 1
                acts.append(("remove_unsupported", f"{label} {cur_dt}"))
            if txt_field and cur_txt is not None:
                r.pop(txt_field)
                stats["removed"] += 1

    @staticmethod
    def _shift(b) -> int:
        """Original closest-valid policy: move timestamps until the axioms hold."""
        from .axioms import SPECIFIC_INDICATION, interval
        n = 0
        enc = next(iter(fhir.resources_of(b, "Encounter")), {})
        per = enc.get("period", {}) if isinstance(enc, dict) else {}
        if per.get("start") and per.get("end") and axioms.le(per["start"], per["end"]) is False:
            per["end"] = per["start"]
            n += 1
        conds = fhir.resources_of(b, "Condition")
        recorded = [c.get("recordedDate") for c in conds if c.get("recordedDate")]
        ref = per.get("end") or (max(recorded) if recorded else None)
        onset_by_key = {}
        from .metrics import _code
        from .terminology import COND_BY_CODE, DRUG_BY_CODE, RXNORM, SNOMED
        for c in conds:
            o = c.get("onsetDateTime")
            if o and c.get("recordedDate") and axioms.le(o, c["recordedDate"]) is False:
                c["onsetDateTime"] = c["recordedDate"]
                n += 1
            if c.get("onsetDateTime") and ref and axioms.le(c["onsetDateTime"], ref) is False:
                c["onsetDateTime"] = ref
                n += 1
            k = COND_BY_CODE.get(_code(c.get("code"), SNOMED) or "", {}).get("key")
            if k and c.get("onsetDateTime"):
                onset_by_key[k] = c["onsetDateTime"]
        for m in fhir.resources_of(b, "MedicationRequest"):
            a = m.get("authoredOn")
            if not a:
                continue
            if ref and axioms.le(a, ref) is False:
                m["authoredOn"] = ref
                n += 1
            if len(a) >= 10 and per.get("start") and axioms.le(per["start"], m["authoredOn"]) is False:
                m["authoredOn"] = per["start"]
                n += 1
            k = DRUG_BY_CODE.get(_code(m.get("medicationCodeableConcept"), RXNORM) or "", {}).get("key")
            ck = SPECIFIC_INDICATION.get(k)
            if ck in onset_by_key and axioms.le(onset_by_key[ck], m["authoredOn"]) is False:
                m["authoredOn"] = onset_by_key[ck]          # closest valid value: no source support
                n += 1
        for al in fhir.resources_of(b, "AllergyIntolerance"):
            if al.get("onsetDateTime") and ref and axioms.le(al["onsetDateTime"], ref) is False:
                al["onsetDateTime"] = ref
                n += 1
        return n


# ---------------------------------------------------------------------------
# Pharmacovigilance Agent
# ---------------------------------------------------------------------------
class PharmacovigilanceAgent:
    name = "pharmacovigilance"

    def run(self, b, F: Optional[Parse], kb, add_issues: bool = True) -> Tuple[Dict, List[frozenset], AgentReport]:
        D = bundle_facts(b)
        state = state_from_facts(D)
        sig = fire_rules(state, kb)
        alerts = [frozenset(s["components"]) for s in sig]
        out = copy.deepcopy(b)
        out["entry"] = [e for e in out["entry"] if isinstance(e, dict) and isinstance(e.get("resource"), dict)]
        out["entry"] = [e for e in out["entry"] if e["resource"].get("resourceType") != "DetectedIssue"
                        or e["resource"].get("code", {}).get("coding", [{}])[0].get("code") == "TIME"]
        if add_issues:
            inv = {}
            for ref, k in D["ids"].items():
                inv.setdefault(k, ref)
            for i, s in enumerate(sig, 1):
                imp = [inv[c] for c in s["components"] if c in inv]
                di = fhir.detected_issue(f"pv{i}", "p1", "ALGY" if s["type"] == "allergy" else "DRG",
                                         "high" if s["severity"] == "major" else "moderate",
                                         f"{s['type']}: {' + '.join(s['components'])} ({s['rule_id']})", imp)
                out["entry"].append({"fullUrl": f"{fhir.BASE}/DetectedIssue/{di['id']}", "resource": di})
        n_med = len(state["meds"])
        unk = F.unknown_med_tokens if F is not None else 0
        unver = 0.0
        feats = dict(n_meds=float(n_med), unknown_med_tokens=float(unk), n_alerts=float(len(alerts)),
                     unk_ratio=unk / max(1, unk + n_med))
        raw = 1 - unk / max(1, unk + n_med)
        return out, alerts, AgentReport("pharmacovigilance", feats, raw, [])


# ---------------------------------------------------------------------------
# Governance Agent: population stability monitoring
# ---------------------------------------------------------------------------
def psi(ref_counts, cur_counts, floor: float = 1e-6) -> float:
    """Population Stability Index, sum_i (p_i - q_i) ln(p_i / q_i). It is a
    non-negative magnitude of distribution change and carries no direction."""
    R, C = sum(ref_counts), sum(cur_counts)
    s = 0.0
    for r, c in zip(ref_counts, cur_counts):
        p = max(c / C if C else 0.0, floor)
        q = max(r / R if R else 0.0, floor)
        s += (p - q) * math.log(p / q)
    return s


class GovernanceAgent:
    """Keeps reference histograms of monitored features and a rolling window."""
    name = "governance"

    def __init__(self, bins: int = 10, window: int = 500, threshold: float = 0.25):
        self.bins, self.window, self.threshold = bins, window, threshold
        self.ref = {}
        self.buf = []

    def fit(self, rows: List[Dict[str, float]]):
        for f in rows[0]:
            vals = sorted(r[f] for r in rows)
            qs = [vals[int(len(vals) * k / self.bins)] for k in range(1, self.bins)]
            edges = sorted(set(qs))
            self.ref[f] = (edges, self._hist(edges, [r[f] for r in rows]))

    @staticmethod
    def _hist(edges, vals):
        import bisect
        h = [0] * (len(edges) + 1)
        for v in vals:
            h[bisect.bisect_right(edges, v)] += 1
        return h

    def update(self, row: Dict[str, float]) -> Dict[str, float]:
        self.buf.append(row)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        return self.current()

    def current(self) -> Dict[str, float]:
        if len(self.buf) < self.window // 2:
            return {f: 0.0 for f in self.ref}
        return {f: psi(h, self._hist(edges, [r[f] for r in self.buf])) for f, (edges, h) in self.ref.items()}


GOV_FEATURES = ["oov", "sup_frac", "frac_learned", "unknown_med_tokens", "vague"]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class MeshConfig:
    order: Tuple[str, ...] = ("S", "R", "T", "P")
    use_schema: bool = True
    use_reasoning: bool = True
    use_temporal: bool = True
    use_pv: bool = True
    repair: bool = True
    temporal_policy: str = "provenance"
    parallel: bool = False
    grounding: bool = True


class Mesh:
    def __init__(self, lex_schema: Lexicon, lex_full: Lexicon, headers, kb):
        self.lex_schema, self.lex_full, self.headers, self.kb = lex_schema, lex_full, headers, kb
        self.S, self.R, self.T, self.P = SchemaAgent(), ReasoningAgent(), TemporalAgent(), PharmacovigilanceAgent()

    def parse_full(self, text):
        return parse_note(text, self.lex_full, "full", self.headers)

    def run(self, text: str, cfg: MeshConfig, draft=None, F: Optional[Parse] = None, Ps: Optional[Parse] = None) -> Dict:
        timing = {}
        t0 = time.perf_counter()
        F = F or self.parse_full(text)
        timing["parse"] = time.perf_counter() - t0
        reports = {}
        t0 = time.perf_counter()
        # Schema Agent ---------------------------------------------------------
        if draft is None:
            if Ps is None:
                Ps = parse_note(text, self.lex_schema, "schema", self.headers)
            b, _ = build_bundle(Ps, text, "schema", naive=not cfg.use_schema)
        else:
            b = draft
        pre = fhir.validate_bundle(b) if b is not None else {"valid": False, "errors": [("bundle_structure", "none")]}
        actions = []
        if cfg.use_schema and not pre["valid"]:
            b, actions = schema_repair(b, F.gender)
        reports["S"] = self.S.check(b, pre, actions)
        timing["S"] = time.perf_counter() - t0
        if b is None:
            return dict(bundle=None, alerts=[], reports=reports, pre_valid=False, post_valid=False, schema_abstain=True)
        alerts: List[frozenset] = []
        if cfg.parallel:
            # every agent audits the same draft; proposals are merged afterwards
            bR, rR = self.R.audit(b, F, text, cfg.repair, self.lex_full, cfg.grounding) if cfg.use_reasoning else (b, None)
            _, rT = self.T.run(b, F, text, cfg.temporal_policy) if cfg.use_temporal else (b, None)
            _, alerts, rP = self.P.run(b, F, self.kb) if cfg.use_pv else (b, [], None)
            if cfg.use_temporal and cfg.repair:
                bR, rT = self.T.run(bR, F, text, cfg.temporal_policy)
            b = bR
            for k, r in (("R", rR), ("T", rT), ("P", rP)):
                if r is not None:
                    reports[k] = r
            if cfg.use_pv:
                b, _, _ = self.P.run(b, F, self.kb)
        else:
            for step in cfg.order[1:]:
                t0 = time.perf_counter()
                if step == "R" and cfg.use_reasoning:
                    nb, reports["R"] = self.R.audit(b, F, text, cfg.repair, self.lex_full, cfg.grounding)
                    b = nb if cfg.repair else b
                elif step == "T" and cfg.use_temporal:
                    nb, reports["T"] = self.T.run(b, F, text, cfg.temporal_policy)
                    b = nb if cfg.repair else b
                elif step == "P" and cfg.use_pv:
                    b, alerts, reports["P"] = self.P.run(b, F, self.kb)
                timing[step] = time.perf_counter() - t0
        post = fhir.validate_bundle(b)
        reports["S"].features["post_valid"] = float(post["valid"])
        return dict(bundle=b, alerts=alerts, reports=reports, pre_valid=pre["valid"], post_valid=post["valid"],
                    pre_errors=pre.get("by_kind", {}), schema_abstain=not post["valid"], F=F, timing=timing)


def feature_vector(reports: Dict[str, AgentReport], agent: str) -> List[float]:
    r = reports.get(agent)
    if r is None:
        return []
    return [r.raw_confidence] + [r.features[k] for k in sorted(r.features)]


# ---------------------------------------------------------------------------
# Deterministic baselines
# ---------------------------------------------------------------------------
def run_rule(text: str, lex_rule: Lexicon, kb) -> Dict:
    P = parse_note(text, lex_rule, "rule")
    b, _ = build_bundle(P, text, "rule")
    b, alerts, _ = PharmacovigilanceAgent().run(b, None, kb)
    conf = sum(m.score for m in P.mentions) / max(len(P.mentions), 1) * (1 - P.oov_tokens / max(P.clinical_tokens, 1))
    return dict(bundle=b, alerts=alerts, conf_raw=conf)


def run_single(text: str, lex: Lexicon, headers, kb) -> Dict:
    P = parse_note(text, lex, "single", headers)
    b, _ = build_bundle(P, text, "single")
    b, alerts, _ = PharmacovigilanceAgent().run(b, None, kb)
    conf = sum(m.score for m in P.mentions) / max(len(P.mentions), 1) * (1 - P.oov_tokens / max(P.clinical_tokens, 1))
    return dict(bundle=b, alerts=alerts, conf_raw=conf)
