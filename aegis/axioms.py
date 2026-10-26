"""Temporal axioms used for the Temporal Consistency Ratio (TCR).

Eleven axioms are defined. The Temporal Agent checks only T1-T7
(``AGENT_AXIOMS``); T8-T11 are held out and used only for evaluation, so that
TCR can be reported on constraints the system never optimises.

Partial dates (YYYY, YYYY-MM) are treated as intervals; an ordering constraint
a <= b is satisfied when it is possible, i.e. earliest(a) <= latest(b).
"""
from __future__ import annotations

import calendar
import datetime as dt
from typing import Dict, List, Optional, Tuple

from .fhir import resources_of, valid_datetime
from .terminology import COND_BY_CODE, DRUG_BY_CODE, RXNORM, SNOMED

SPECIFIC_INDICATION = {"metformin": "t2dm", "insulin_glargine": "t2dm", "levothyroxine": "hypothyroid",
                       "allopurinol": "gout", "tamsulosin": "bph", "alendronate": "osteoporosis"}
ALL_AXIOMS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"]
AGENT_AXIOMS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
HELDOUT_AXIOMS = ["T8", "T9", "T10", "T11"]
DESCRIPTIONS = {
    "T1": "Encounter.period.start <= Encounter.period.end",
    "T2": "Condition.onset <= Condition.recordedDate",
    "T3": "Condition.onset <= reference date (encounter end or note date)",
    "T4": "MedicationRequest.authoredOn <= reference date",
    "T5": "Day-precision MedicationRequest.authoredOn >= Encounter.period.start (inpatient orders)",
    "T6": "Indication onset <= start of an indication-specific medication",
    "T7": "AllergyIntolerance.onset <= reference date",
    "T8": "Observation.effective <= reference date",
    "T9": "Observation.effective >= Encounter.period.start",
    "T10": "Encounter.period.start <= reference date",
    "T11": "Condition.recordedDate >= Encounter.period.start",
}


def interval(s) -> Optional[Tuple[dt.date, dt.date]]:
    if not isinstance(s, str) or not valid_datetime(s):
        return None
    s = s[:10]
    try:
        if len(s) == 4:
            y = int(s)
            return dt.date(y, 1, 1), dt.date(y, 12, 31)
        if len(s) == 7:
            y, m = int(s[:4]), int(s[5:7])
            return dt.date(y, m, 1), dt.date(y, m, calendar.monthrange(y, m)[1])
        d = dt.date.fromisoformat(s)
        return d, d
    except ValueError:
        return None


def le(a, b) -> Optional[bool]:
    ia, ib = interval(a), interval(b)
    if ia is None or ib is None:
        return None
    return ia[0] <= ib[1]


def _code(cc, system):
    if not isinstance(cc, dict):
        return None
    for c in cc.get("coding", []) or []:
        if isinstance(c, dict) and c.get("system") == system:
            return c.get("code")
    return None


def check(bundle, axioms: List[str] = ALL_AXIOMS) -> Dict[str, List[bool]]:
    """Return {axiom: [result per applicable instance]}."""
    out = {a: [] for a in axioms}
    enc = next(iter(resources_of(bundle, "Encounter")), None)
    start = end = None
    if isinstance(enc, dict) and isinstance(enc.get("period"), dict):
        start, end = enc["period"].get("start"), enc["period"].get("end")
    conds = resources_of(bundle, "Condition")
    recorded = [c.get("recordedDate") for c in conds if isinstance(c.get("recordedDate"), str)]
    ref = end or (max(recorded) if recorded else None)

    def add(a, v):
        if a in out and v is not None:
            out[a].append(v)

    add("T1", le(start, end) if start and end else None)
    add("T10", le(start, ref) if start and ref else None)
    onset_by_key = {}
    for c in conds:
        onset = c.get("onsetDateTime")
        if onset:
            add("T2", le(onset, c.get("recordedDate")) if c.get("recordedDate") else None)
            add("T3", le(onset, ref) if ref else None)
            k = COND_BY_CODE.get(_code(c.get("code"), SNOMED) or "", {}).get("key")
            if k:
                onset_by_key[k] = onset
        if c.get("recordedDate") and start:
            add("T11", le(start, c["recordedDate"]))
    for m in resources_of(bundle, "MedicationRequest"):
        a = m.get("authoredOn")
        if not a:
            continue
        add("T4", le(a, ref) if ref else None)
        if isinstance(a, str) and len(a) >= 10 and start:
            add("T5", le(start, a))
        k = DRUG_BY_CODE.get(_code(m.get("medicationCodeableConcept"), RXNORM) or "", {}).get("key")
        if k in SPECIFIC_INDICATION and SPECIFIC_INDICATION[k] in onset_by_key:
            add("T6", le(onset_by_key[SPECIFIC_INDICATION[k]], a))
    for al in resources_of(bundle, "AllergyIntolerance"):
        if al.get("onsetDateTime") and ref:
            add("T7", le(al["onsetDateTime"], ref))
    for o in resources_of(bundle, "Observation"):
        e = o.get("effectiveDateTime")
        if e:
            add("T8", le(e, ref) if ref else None)
            add("T9", le(start, e) if start else None)
    return out


def tcr(bundle, axioms: List[str] = ALL_AXIOMS) -> Tuple[int, int]:
    r = check(bundle, axioms)
    sat = sum(sum(v) for v in r.values())
    n = sum(len(v) for v in r.values())
    return sat, n
