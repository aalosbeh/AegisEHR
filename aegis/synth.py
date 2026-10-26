"""Synthetic benchmark generator.

Generation order (this order is what keeps the labels independent of the
systems under test):

1. Sample a latent clinical state for one encounter: demographics, encounter
   dates, conditions with onset, medications with dose/route/frequency and
   reconciliation actions, allergies, laboratory results.
2. Choose, for every fact, how it will be expressed (surface form tier,
   temporal precision, vague wording, negated or family-history mentions).
3. Build the gold FHIR R4 bundle from (1) and (2). The gold is
   *source-faithful*: a date is only present at the precision the text
   supports, and vague temporal wording is kept as ``onsetString``.
4. Compute gold pharmacovigilance signals by applying the **reference** KB to
   the reconciled state (active medications, verified allergies, active
   conditions, latest laboratory values).
5. Render the note text for the chosen documentation profile.

No system code is imported here, and the reference KB is never read by any
system.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from . import fhir
from .knowledge import fire_rules, reference_kb
from .terminology import (ALLERGEN_BY_KEY, ALLERGENS, COND_BY_KEY, CONDITIONS, DRUG_BY_KEY, DRUGS,
                          FAMILY_KEYS, FREQS, LAB_BY_KEY, LABS, REACTIONS, ROS_NEGATIVE_KEYS, ROUTES,
                          UNIT_TEXT)

# drug -> condition pairs used for the "medication requires indication" axiom
SPECIFIC_INDICATION = {"metformin": "t2dm", "insulin_glargine": "t2dm", "levothyroxine": "hypothyroid",
                       "allopurinol": "gout", "tamsulosin": "bph", "alendronate": "osteoporosis"}
RESOLVABLE_EVENTS = {"mi", "stroke", "dvt", "pe", "pna", "uti", "gi_bleed", "cellulitis", "sepsis"}
ACUTE_CC = ["pna", "uti", "cellulitis", "sepsis", "chest_pain", "dyspnea", "gi_bleed", "hypokalemia",
            "hf", "copd", "afib", "dvt", "pe", "hyperkalemia", "bradycardia"]

# at most one drug per therapeutic group unless a signal is injected on purpose
EXCLUSIVE_GROUPS = [{"warfarin", "apixaban", "enoxaparin", "heparin"}, {"atorvastatin", "simvastatin"},
                    {"omeprazole", "pantoprazole"}, {"lisinopril", "losartan"},
                    {"metoprolol", "carvedilol", "propranolol"}, {"sertraline", "citalopram", "fluoxetine"},
                    {"ibuprofen", "naproxen"}, {"ciprofloxacin", "levofloxacin"}, {"diltiazem", "verapamil"},
                    {"oxycodone", "morphine", "tramadol"}, {"amoxicillin", "ampicillin"},
                    {"azithromycin", "clarithromycin"}]

FILLER = {
    "hpi": ["Reports decreased oral intake and generalized weakness.", "Was seen in the emergency department and admitted for further management.",
            "Symptoms have been progressively worsening.", "Lives at home with family and is independent in daily activities.",
            "Denies recent travel.", "Came in after family noticed confusion."],
    "course": ["Hospital course was uncomplicated.", "Patient improved with supportive care and was ambulating at discharge.",
               "Remained hemodynamically stable throughout the stay.", "Tolerated diet and was cleared by physical therapy."],
}

SECTIONS = ["cc", "hpi", "pmh", "meds", "allergies", "fhx", "ros", "exam", "labs", "plan", "dcmeds"]


@dataclass
class Profile:
    key: str
    name: str
    label: str
    share: float
    headers: Dict[str, str]
    titles: Dict[str, str]
    date_style: str
    abbrev_p: float
    site_p: float
    rare_p: float
    typo_p: float
    vague_p: float
    conflict_p: float
    narrative_p: float
    age: Tuple[float, float]
    pv_inject_p: float
    allergy_p: float
    inconsistency_p: float = 0.03
    near_miss_p: float = 0.07
    novel_p: float = 0.0          # drift only: share of mentions using unseen forms
    extra_meds: float = 0.0       # drift only: population shift toward polypharmacy


PROFILES = {
    "A": Profile("A", "Site-A", "academic", 0.395,
                 headers=dict(cc="CHIEF COMPLAINT:", hpi="HISTORY OF PRESENT ILLNESS:", pmh="PAST MEDICAL HISTORY:",
                              meds="MEDICATIONS ON ADMISSION:", allergies="ALLERGIES:", fhx="FAMILY HISTORY:",
                              ros="REVIEW OF SYSTEMS:", exam="PHYSICAL EXAMINATION:", labs="PERTINENT RESULTS:",
                              plan="ASSESSMENT AND PLAN:", dcmeds="DISCHARGE MEDICATIONS:"),
                 titles=dict(hp="ADMISSION HISTORY AND PHYSICAL", progress="PROGRESS NOTE", dc="DISCHARGE SUMMARY"),
                 date_style="iso", abbrev_p=0.30, site_p=0.0, rare_p=0.03, typo_p=0.004, vague_p=0.221,
                 conflict_p=0.074, narrative_p=0.0, age=(58, 17), pv_inject_p=0.12, allergy_p=0.14),
    "B": Profile("B", "Site-B", "community", 0.333,
                 headers=dict(cc="CC:", hpi="HPI:", pmh="PMHx:", meds="Home Meds:", allergies="ALL:", fhx="FHx:",
                              ros="ROS:", exam="PE:", labs="Labs:", plan="A/P:", dcmeds="D/C Meds:"),
                 titles=dict(hp="H&P", progress="Progress Note", dc="D/C Summary"),
                 date_style="us", abbrev_p=0.58, site_p=0.30, rare_p=0.04, typo_p=0.010, vague_p=0.284,
                 conflict_p=0.112, narrative_p=0.0, age=(61, 18), pv_inject_p=0.15, allergy_p=0.13),
    "C": Profile("C", "Site-C", "rural", 0.272,
                 headers=dict(cc="Reason:", hpi="Hx:", pmh="Prior hx:", meds="Rx:", allergies="Allergy:", fhx="Family:",
                              ros="Sx review:", exam="Exam:", labs="Lab:", plan="Plan:", dcmeds="Going home on:"),
                 titles=dict(hp="Admission note", progress="Daily note", dc="Discharge note"),
                 date_style="dmy", abbrev_p=0.22, site_p=0.35, rare_p=0.04, typo_p=0.015, vague_p=0.153,
                 conflict_p=0.049, narrative_p=0.25, age=(64, 17), pv_inject_p=0.10, allergy_p=0.12),
}
SITE_FORM_KEY = {"A": "A", "B": "B", "C": "C"}

RACES = [("White", 0.59), ("Black", 0.19), ("Hispanic", 0.13), ("Asian", 0.05), ("Other/Unknown", 0.04)]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_FULL = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
               "October", "November", "December"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:16], 16)


def fmt_date(d: dt.date, style: str, precision: str = "day", rng: Optional[random.Random] = None) -> str:
    """Render a date as a clinician of the given profile would write it."""
    if precision == "year":
        return str(d.year)
    if precision == "month":
        if style == "iso":
            return f"{MONTHS_FULL[d.month - 1]} {d.year}"
        if style == "us":
            return f"{d.month}/{d.year}"
        if style == "eu":
            return f"{d.month:02d}.{d.year}"
        return f"{MONTHS[d.month - 1]} {d.year}"
    if style == "iso":
        return d.isoformat()
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    if style == "eu":
        return f"{d.day:02d}.{d.month:02d}.{d.year}"
    return f"{d.day}-{MONTHS[d.month - 1]}-{str(d.year)[2:]}"


def iso(d: dt.date, precision: str = "day") -> str:
    if precision == "year":
        return f"{d.year:04d}"
    if precision == "month":
        return f"{d.year:04d}-{d.month:02d}"
    return d.isoformat()


def _is_abbrev(s: str) -> bool:
    return s.isupper() or (len(s) <= 5 and any(c.isupper() for c in s)) or s in ("a-fib", "glargine")


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 1)
    op = rng.random()
    if op < 0.3:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op < 0.6:
        return word[:i] + word[i + 1:]
    if op < 0.8:
        return word[:i] + word[i] + word[i:]
    near = "qwertyuiopasdfghjklzxcvbnm"
    return word[:i] + rng.choice(near) + word[i + 1:]


def novel_form(canonical: str, rng: random.Random) -> str:
    """Unseen surface form used only by drift profiles (vowel-stripped or dotted)."""
    words = canonical.split()
    if len(words) > 1 and rng.random() < 0.5:
        return ".".join(w[0].upper() for w in words) + "."
    w = max(words, key=len)
    stripped = w[0] + "".join(c for c in w[1:] if c.lower() not in "aeiou")
    return canonical.replace(w, stripped)


@dataclass
class Mention:
    kind: str
    key: str
    text: str
    tier: str
    section: str
    assertion: str = "present"     # present | negated | family | refuted


@dataclass
class Rec:
    rid: str
    site: str
    note_type: str
    age: int
    sex: str
    race: str
    admit: dt.date
    discharge: Optional[dt.date]
    note_date: dt.date
    hospital_day: int
    conditions: List[Dict] = field(default_factory=list)
    meds: List[Dict] = field(default_factory=list)
    allergies: List[Dict] = field(default_factory=list)
    labs: List[Dict] = field(default_factory=list)
    ros_neg: List[str] = field(default_factory=list)
    family: List[Tuple[str, str]] = field(default_factory=list)
    delabeled: List[str] = field(default_factory=list)
    mentions: List[Mention] = field(default_factory=list)
    flags: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# latent state
# ---------------------------------------------------------------------------
class Generator:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.ref_kb = reference_kb()
        self.alerting = [r for r in self.ref_kb if r["severity"] in ("major", "moderate")]

    # -- surface forms ------------------------------------------------------
    def surface(self, concept: Dict, kind: str, prof: Profile, rng: random.Random, section: str) -> Tuple[str, str]:
        site_forms = concept.get("site", {}).get(SITE_FORM_KEY.get(prof.key, ""), [])
        r = rng.random()
        if prof.novel_p and r < prof.novel_p:
            return novel_form(concept["base"][0], rng), "novel"
        r = rng.random()
        if site_forms and r < prof.site_p:
            s, tier = rng.choice(site_forms), "site"
        elif concept.get("rare") and r < prof.site_p + prof.rare_p:
            s, tier = rng.choice(concept["rare"]), "rare"
        else:
            base = concept["base"]
            abbr = [x for x in base[1:] if _is_abbrev(x)]
            syn = [x for x in base[1:] if not _is_abbrev(x)]
            r2 = rng.random()
            if abbr and r2 < prof.abbrev_p:
                s, tier = rng.choice(abbr), "abbr"
            elif syn and r2 < prof.abbrev_p + 0.25:
                s, tier = rng.choice(syn), "syn"
            else:
                s, tier = base[0], "canonical"
            if tier in ("canonical", "syn") and rng.random() < 0.35:
                s = s[0].upper() + s[1:]
        if kind == "drug" and concept["key"] == "morphine" and prof.key == "B" and rng.random() < 0.3:
            s, tier = "MS", "ambiguous"
        words = s.split()
        if prof.typo_p and rng.random() < prof.typo_p * 6:
            j = max(range(len(words)), key=lambda k: len(words[k]))
            if len(words[j]) >= 5:
                words[j] = _typo(words[j], rng)
                s = " ".join(words)
                tier += "+typo"
        return s, tier

    # -- one record ---------------------------------------------------------
    def sample_state(self, i: int, prof: Profile) -> Rec:
        rng = random.Random(_seed(self.seed, prof.key, i))
        age = int(min(95, max(18, rng.gauss(*prof.age))))
        sex = "male" if rng.random() < 0.49 else "female"
        race = rng.choices([r for r, _ in RACES], weights=[w for _, w in RACES])[0]
        note_type = rng.choices(["hp", "progress", "dc"], weights=[0.45, 0.20, 0.35])[0]
        admit = dt.date(2019, 1, 1) + dt.timedelta(days=rng.randrange(0, 7 * 365))
        los = rng.randint(1, 9)
        discharge = admit + dt.timedelta(days=los) if note_type == "dc" else None
        hd = 1 if note_type == "hp" else (rng.randint(2, 6) if note_type == "progress" else los + 1)
        note_date = admit if note_type == "hp" else (admit + dt.timedelta(days=hd - 1) if note_type == "progress" else discharge)
        rec = Rec(rid=f"{prof.key}-{i:06d}", site=prof.key, note_type=note_type, age=age, sex=sex, race=race,
                  admit=admit, discharge=discharge, note_date=note_date, hospital_day=hd)

        # conditions ---------------------------------------------------------
        n_cond = min(9, 1 + _poisson(rng, 1.4 + 0.035 * max(0, age - 40)))
        pool = [c for c in CONDITIONS if c["key"] not in ("chest_pain", "dyspnea")]
        pool = [c for c in pool if not (c["key"] == "bph" and sex == "female")]
        weights = [c["prev"] * (2.718 ** (c["age_slope"] * (age - 55))) * (0.15 if c["key"] == "breast_ca" and sex == "male" else 1)
                   for c in pool]
        chosen = []
        cc_key = rng.choice(ACUTE_CC)
        chosen.append(cc_key)
        while len(chosen) < n_cond:
            c = rng.choices(pool, weights=weights)[0]["key"]
            if c not in chosen:
                chosen.append(c)
        for k in chosen:
            self._add_condition(rec, k, rng, prof, chief=(k == cc_key))

        # medications --------------------------------------------------------
        for c in list(rec.conditions):
            for j, dk in enumerate(COND_BY_KEY[c["key"]]["meds"]):
                if rng.random() < (0.55 if j == 0 else 0.28) and not self._has_med(rec, dk):
                    inpatient = not COND_BY_KEY[c["key"]]["chronic"] and c["chief"]
                    self._add_med(rec, dk, rng, prof, inpatient=inpatient)
        for dk, p in (("acetaminophen", 0.18), ("omeprazole", 0.08), ("aspirin", 0.10), ("ondansetron", 0.06),
                      ("enoxaparin", 0.10), ("heparin", 0.06), ("gabapentin", 0.05), ("oxycodone", 0.04)):
            if rng.random() < p + prof.extra_meds and not self._has_med(rec, dk):
                self._add_med(rec, dk, rng, prof, inpatient=dk in ("ondansetron", "enoxaparin", "heparin", "oxycodone"))
        if prof.extra_meds:
            for _ in range(int(rng.random() * 6 * prof.extra_meds)):
                dk = rng.choice(DRUGS)["key"]
                if not self._has_med(rec, dk):
                    self._add_med(rec, dk, rng, prof, inpatient=False)

        # allergies ------------------------------------------------------------
        if rng.random() < prof.allergy_p:
            for ak in rng.sample([a["key"] for a in ALLERGENS], k=1 if rng.random() < 0.8 else 2):
                self._add_allergy(rec, ak, rng, prof)

        # labs -----------------------------------------------------------------
        for lk in rng.sample([l["key"] for l in LABS], k=rng.randint(2, 5)):
            self._add_lab(rec, lk, rng)
        cond_keys = {c["key"] for c in rec.conditions}
        if "ckd" in cond_keys:
            self._set_lab(rec, "egfr", rng.choice([22, 26, 28, 34, 41, 48]), rng)
            self._set_lab(rec, "creatinine", round(rng.uniform(1.6, 3.2), 1), rng)
        if "hyperkalemia" in cond_keys:
            self._set_lab(rec, "potassium", round(rng.uniform(5.7, 6.5), 1), rng)
        if "hypokalemia" in cond_keys:
            self._set_lab(rec, "potassium", round(rng.uniform(2.8, 3.3), 1), rng)
        if self._has_med(rec, "warfarin") and rng.random() < 0.6:
            self._set_lab(rec, "inr", round(rng.uniform(1.8, 3.2), 1), rng)

        # pharmacovigilance injection and near misses ----------------------------
        if rng.random() < prof.pv_inject_p:
            self._inject_signal(rec, rng, prof)
        if rng.random() < prof.near_miss_p:
            self._inject_near_miss(rec, rng, prof)

        # medication-list conflicts (reconciliation actions in the plan) --------
        if rng.random() < prof.conflict_p * 1.6:
            self._inject_conflict(rec, rng)

        # documentation inconsistency (kept in gold, must be flagged, not fixed)
        if rng.random() < prof.inconsistency_p:
            self._inject_inconsistency(rec, rng, prof)

        # pertinent negatives and family history ---------------------------------
        negs = [k for k in ROS_NEGATIVE_KEYS if k not in cond_keys]
        rec.ros_neg = rng.sample(negs, k=min(len(negs), rng.randint(1, 4)))
        if rng.random() < 0.55:
            fam = [k for k in FAMILY_KEYS if k not in cond_keys]
            rel = ["mother", "father", "sister", "brother"]
            rec.family = [(rng.choice(rel), k) for k in rng.sample(fam, k=rng.randint(1, 2))]
        return rec

    # ---- condition --------------------------------------------------------
    def _add_condition(self, rec: Rec, key: str, rng, prof: Profile, chief: bool = False):
        c = COND_BY_KEY[key]
        status = "active"
        if not chief and key in RESOLVABLE_EVENTS:
            status = "resolved"
        if chief or not c["chronic"] and status == "active":
            days = rng.randint(1, 7)
            onset = rec.admit - dt.timedelta(days=days)
            r = rng.random()
            if r < prof.vague_p:
                expr = dict(kind="vague", text=rng.choice(["a few days ago", "recently", "over the past week", "several days ago"]))
            elif r < prof.vague_p + 0.45:
                expr = dict(kind="relative", days=days)
            elif r < prof.vague_p + 0.75:
                expr = dict(kind="absolute", precision="day")
            else:
                expr = dict(kind="none")
        else:
            onset = dt.date(rec.admit.year - rng.randint(1, 20), rng.randint(1, 12), rng.randint(1, 28))
            r = rng.random()
            if r < 0.40:
                expr = dict(kind="none")
            elif r < 0.40 + 0.5 * prof.vague_p:
                expr = dict(kind="vague", text=rng.choice(["long-standing", "for many years", "remote", "years ago",
                                                          "for a long time"]))
            elif r < 0.40 + 0.5 * prof.vague_p + 0.25:
                expr = dict(kind="absolute", precision="year")
            elif r < 0.40 + 0.5 * prof.vague_p + 0.33:
                expr = dict(kind="absolute", precision="month")
            else:
                expr = dict(kind="absolute", precision="day" if key in RESOLVABLE_EVENTS else "year")
            if status == "resolved" and expr["kind"] == "none":
                expr = dict(kind="absolute", precision="year")
        rec.conditions.append(dict(key=key, onset=onset, expr=expr, status=status, chief=chief))

    def _has_med(self, rec, dk):
        return any(m["key"] == dk for m in rec.meds)

    def _add_med(self, rec: Rec, dk: str, rng, prof: Profile, inpatient: bool, force: bool = False):
        if not force:
            for g in EXCLUSIVE_GROUPS:
                if dk in g and any(m["key"] in g for m in rec.meds):
                    return None
        d = DRUG_BY_KEY[dk]
        dose = rng.choice(d["doses"])
        unit = d["unit"]
        oral = [r for r in d["routes"] if r != "IV"]
        route = rng.choice(d["routes"]) if inpatient or not oral else oral[0]
        freq = rng.choice(d["freqs"])
        m = dict(key=dk, dose=dose, unit=unit, route=route, freq=freq, prn=d["prn"] and rng.random() < 0.8,
                 status="active", category="inpatient" if inpatient else "community", start=None, start_expr=None,
                 action=None, route_stated=rng.random() > (0.35 if prof.key == "C" else 0.08))
        if inpatient:
            day = rng.randint(0, max(0, rec.hospital_day - 1))
            m["start"] = rec.admit + dt.timedelta(days=day)
            m["start_expr"] = rng.choice(["absolute", "hd"]) if rng.random() < 0.85 else None
        elif rng.random() < 0.22:
            m["start"] = dt.date(rec.admit.year - rng.randint(0, 8), rng.randint(1, 12), 15)
            m["start_expr"] = "year" if rng.random() < 0.75 else "vague"
        rec.meds.append(m)
        return m

    def _add_allergy(self, rec: Rec, ak: str, rng, prof: Profile, reaction: bool = True):
        if any(a["key"] == ak for a in rec.allergies):
            return
        rx = rng.choice(REACTIONS)["key"] if reaction and rng.random() < 0.8 else None
        sev = rng.choice(["mild", "moderate", "severe"]) if rx and rng.random() < 0.4 else None
        if rx == "anaphylaxis":
            sev = "severe"
        onset = None
        r = rng.random()
        if r < 0.25:
            onset = ("year", dt.date(rec.admit.year - rng.randint(1, 30), 6, 1))
        elif r < 0.33:
            onset = ("vague", "as a child")
        rec.allergies.append(dict(key=ak, reaction=rx, severity=sev, onset=onset))

    def _add_lab(self, rec: Rec, lk: str, rng, value=None):
        lab = LAB_BY_KEY[lk]
        lo, hi = lab["normal"]
        if value is None:
            value = rng.uniform(lo, hi) if rng.random() < 0.88 else rng.uniform(lo * 0.9, hi * 1.1)
        value = round(value, lab["decimals"]) if lab["decimals"] else int(round(value))
        rec.labs.append(dict(key=lk, value=value, day=0, repeat=False))

    def _set_lab(self, rec: Rec, lk: str, value, rng):
        rec.labs = [l for l in rec.labs if l["key"] != lk]
        self._add_lab(rec, lk, rng, value=value)

    def _drug_for_target(self, target, is_class, rng):
        if not is_class:
            return target
        opts = [d["key"] for d in DRUGS if d["cls"] == target]
        return rng.choice(opts)

    def _inject_signal(self, rec: Rec, rng, prof: Profile):
        weights = [0.35 if r["rare"] else 1.0 for r in self.alerting]
        r = rng.choices(self.alerting, weights=weights)[0]
        rec.flags["injected_rule"] = r["id"]
        if r["type"] == "ddi":
            for dk in (r["a"], r["b"]):
                if not self._has_med(rec, dk):
                    self._add_med(rec, dk, rng, prof, inpatient=DRUG_BY_KEY[dk]["cls"] in
                                  ("fluoroquinolone", "macrolide", "sulfonamide_abx", "azole", "oxazolidinone", "antiemetic"),
                                  force=True)
        elif r["type"] == "allergy":
            self._add_allergy(rec, r["allergen"], rng, prof)
            dk = self._drug_for_target(r["target"], r["is_class"], rng)
            if not self._has_med(rec, dk):
                self._add_med(rec, dk, rng, prof, inpatient=True, force=True)
        elif r["type"] == "condition":
            if r["condition"] not in {c["key"] for c in rec.conditions}:
                self._add_condition(rec, r["condition"], rng, prof)
                if rec.conditions[-1]["status"] == "resolved":
                    rec.conditions[-1]["status"] = "active"
            dk = self._drug_for_target(r["target"], r["is_class"], rng)
            if not self._has_med(rec, dk):
                self._add_med(rec, dk, rng, prof, inpatient=False, force=True)
        elif r["type"] == "lab":
            tgt = rng.choice(r["targets"])
            dk = self._drug_for_target(tgt, r["is_class"], rng)
            if not self._has_med(rec, dk):
                self._add_med(rec, dk, rng, prof, inpatient=False, force=True)
            lab = LAB_BY_KEY[r["lab"]]
            v = r["threshold"] + (rng.uniform(0.3, 1.0) if r["op"] == ">" else -rng.uniform(4 if r["lab"] == "egfr" else 0.2,
                                                                                            12 if r["lab"] == "egfr" else 0.6))
            self._set_lab(rec, r["lab"], v, rng)
        for m in rec.meds:  # injected pairs must be active at the end of the encounter
            m.setdefault("protected", False)
        for dk in self._components(r):
            for m in rec.meds:
                if m["key"] == dk:
                    m["protected"] = True

    @staticmethod
    def _components(rule):
        if rule["type"] == "ddi":
            return [rule["a"], rule["b"]]
        if rule["type"] in ("allergy", "condition"):
            return [] if rule["is_class"] else [rule["target"]]
        return [] if rule["is_class"] else list(rule["targets"])

    def _inject_near_miss(self, rec: Rec, rng, prof: Profile):
        kind = rng.choice(["stopped", "delabeled", "lab_normalized", "negated_condition"])
        rec.flags["near_miss"] = kind
        if kind == "stopped":
            pair = rng.choice([("warfarin", "ibuprofen"), ("lisinopril", "potassium_chloride"),
                               ("sertraline", "tramadol"), ("simvastatin", "clarithromycin"), ("apixaban", "aspirin")])
            for dk in pair:
                if not self._has_med(rec, dk):
                    self._add_med(rec, dk, rng, prof, inpatient=False, force=True)
            for m in rec.meds:
                if m["key"] == pair[1]:
                    m["status"], m["action"], m["protected"] = "stopped", "stop", True
        elif kind == "delabeled":
            rec.delabeled.append("penicillin")
            rec.allergies = [a for a in rec.allergies if a["key"] != "penicillin"]
            if not self._has_med(rec, "amoxicillin"):
                self._add_med(rec, "amoxicillin", rng, prof, inpatient=True, force=True)
        elif kind == "lab_normalized":
            if not self._has_med(rec, "spironolactone"):
                self._add_med(rec, "spironolactone", rng, prof, inpatient=False, force=True)
            self._set_lab(rec, "potassium", round(rng.uniform(5.7, 6.2), 1), rng)
            rec.labs.append(dict(key="potassium", value=round(rng.uniform(4.2, 4.9), 1), day=1, repeat=True))
        else:
            if any(c["key"] == "gi_bleed" and c["chief"] for c in rec.conditions):
                rec.flags["near_miss"] = None
                return
            if not self._has_med(rec, "warfarin"):
                self._add_med(rec, "warfarin", rng, prof, inpatient=False, force=True)
            rec.flags["force_ros_neg"] = "gi_bleed"
            rec.conditions = [c for c in rec.conditions if c["key"] != "gi_bleed"]

    def _inject_conflict(self, rec: Rec, rng):
        home = [m for m in rec.meds if m["category"] == "community" and m["action"] is None and not m.get("protected")]
        if not home:
            return
        m = rng.choice(home)
        d = DRUG_BY_KEY[m["key"]]
        r = rng.random()
        if r < 0.40 and len(d["doses"]) > 1:
            new = rng.choice([x for x in d["doses"] if x != m["dose"]])
            m["old_dose"], m["dose"], m["action"] = m["dose"], new, "dose_change"
        elif r < 0.80:
            m["status"], m["action"] = "stopped", "stop"
        else:
            m["status"], m["action"] = "on-hold", "hold"
        rec.flags["conflict"] = m["action"]

    def _inject_inconsistency(self, rec: Rec, rng, prof: Profile):
        dk = rng.choice(list(SPECIFIC_INDICATION))
        ck = SPECIFIC_INDICATION[dk]
        if ck not in {c["key"] for c in rec.conditions}:
            self._add_condition(rec, ck, rng, prof)
        cond = next(c for c in rec.conditions if c["key"] == ck)
        if cond["status"] == "resolved":
            cond["status"] = "active"
        cond["onset"] = dt.date(rec.admit.year - rng.randint(2, 6), rng.randint(1, 12), 1)
        cond["expr"] = dict(kind="absolute", precision="year")
        med = next((m for m in rec.meds if m["key"] == dk), None) or self._add_med(rec, dk, rng, prof, inpatient=False,
                                                                                    force=True)
        med["start"] = dt.date(cond["onset"].year - rng.randint(1, 4), 6, 15)
        med["start_expr"] = "year"
        med["category"] = "community"
        rec.flags["source_inconsistent"] = f"{dk}<{ck}"

    # ---- reconciled state and gold ------------------------------------------
    @staticmethod
    def final_state(rec: Rec) -> Dict:
        latest = {}
        for l in sorted(rec.labs, key=lambda x: x["day"]):
            latest[l["key"]] = l["value"]
        return dict(meds=[m["key"] for m in rec.meds if m["status"] == "active"],
                    allergies=[a["key"] for a in rec.allergies],
                    conditions=[c["key"] for c in rec.conditions if c["status"] == "active"],
                    labs=latest)

    def gold_bundle(self, rec: Rec) -> Dict:
        res = [fhir.patient("p1", rec.sex)]
        status = "finished" if rec.note_type == "dc" else "in-progress"
        res.append(fhir.encounter("e1", "p1", status, "IMP", iso(rec.admit),
                                  iso(rec.discharge) if rec.discharge else None))
        for i, c in enumerate(rec.conditions, 1):
            cd = COND_BY_KEY[c["key"]]
            onset, onset_text = None, None
            e = c["expr"]
            if e["kind"] == "relative":
                onset = iso(c["onset"])
            elif e["kind"] == "absolute":
                onset = iso(c["onset"], e["precision"])
            elif e["kind"] == "vague":
                onset_text = e["text"]
            res.append(fhir.condition(f"c{i}", "p1", cd["snomed"], cd["display"], c["status"], onset, onset_text,
                                      iso(rec.note_date), cd["icd10"]))
        for i, m in enumerate(rec.meds, 1):
            d = DRUG_BY_KEY[m["key"]]
            authored = None
            if m["start"] is not None and m["start_expr"] in ("absolute", "hd"):
                authored = iso(m["start"])
            elif m["start"] is not None and m["start_expr"] == "year":
                authored = iso(m["start"], "year")
            dose, unit = _dose_as_written(m)
            res.append(fhir.medication_request(
                f"m{i}", "p1", d["rxcui"], d["display"], m["status"], m["category"], dose, unit,
                ROUTES[m["route"]]["snomed"] if m["route_stated"] else None, FREQS[m["freq"]]["repeat"], m["prn"],
                authored, ROUTES[m["route"]]["display"] if m["route_stated"] else None))
        for i, a in enumerate(rec.allergies, 1):
            ad = ALLERGEN_BY_KEY[a["key"]]
            rx = [(next(r for r in REACTIONS if r["key"] == a["reaction"])["snomed"],
                   next(r for r in REACTIONS if r["key"] == a["reaction"])["display"])] if a["reaction"] else None
            onset = iso(a["onset"][1], "year") if a["onset"] and a["onset"][0] == "year" else None
            al = fhir.allergy(f"a{i}", "p1", ad["snomed"], ad["display"], rx, a["severity"], onset)
            if a["onset"] and a["onset"][0] == "vague":
                al["onsetString"] = a["onset"][1]
            res.append(al)
        for i, l in enumerate(rec.labs, 1):
            lab = LAB_BY_KEY[l["key"]]
            res.append(fhir.observation(f"o{i}", "p1", lab["loinc"], lab["display"], l["value"], lab["unit"],
                                        iso(rec.admit + dt.timedelta(days=l["day"]))))
        return fhir.bundle(res)

    # ---- rendering ----------------------------------------------------------
    def render(self, rec: Rec, prof: Profile, rng: random.Random) -> str:
        H = prof.headers
        ds = prof.date_style
        narrative = rng.random() < prof.narrative_p
        lines: List[str] = []
        title = prof.titles[rec.note_type]
        lines.append(f"[{prof.name}] {title}")
        if rec.note_type == "dc":
            lines.append(f"Admission date: {fmt_date(rec.admit, ds)}   Discharge date: {fmt_date(rec.discharge, ds)}")
        elif rec.note_type == "progress":
            lines.append(f"Date: {fmt_date(rec.note_date, ds)}   Hospital day #{rec.hospital_day}")
        else:
            lines.append(f"Date of admission: {fmt_date(rec.admit, ds)}")

        def ment(kind, key, concept, section, assertion="present"):
            s, tier = self.surface(concept, kind, prof, rng, section)
            rec.mentions.append(Mention(kind, key, s, tier, section, assertion))
            return s

        conds = rec.conditions
        chief = next(c for c in conds if c["chief"])
        others = [c for c in conds if not c["chief"]]
        sex_word = {"A": ("man", "woman"), "B": ("M", "F"), "C": ("male", "female")}[prof.key][rec.sex == "female"]

        # chief complaint / HPI ----------------------------------------------
        cc_txt = ment("condition", chief["key"], COND_BY_KEY[chief["key"]], "hpi")
        onset_txt = self._onset_phrase(chief, rec, prof, rng, acute=True)
        hx = ", ".join(ment("condition", c["key"], COND_BY_KEY[c["key"]], "hpi") for c in others[:3]) if others else ""
        if prof.key == "A":
            hpi = f"The patient is a {rec.age}-year-old {sex_word}" + (f" with a history of {hx}" if hx else "") + \
                  f" who presents with {cc_txt}{onset_txt}."
        elif prof.key == "B":
            hpi = f"{rec.age} y/o {sex_word}" + (f" w/ PMH {hx}" if hx else "") + f" p/w {cc_txt}{onset_txt}."
        else:
            hpi = f"{rec.age}yo {sex_word} came in with {cc_txt}{onset_txt}." + (f" Has {hx}." if hx else "")
        hpi += " " + " ".join(rng.sample(FILLER["hpi"], k=rng.randint(1, 2)))
        if not narrative:
            lines.append(f"{H['cc']} {cc_txt}")
            lines.append(f"{H['hpi']} {hpi}")
        else:
            lines.append(hpi)

        # PMH ------------------------------------------------------------------
        pmh_items = []
        for c in others:
            name = ment("condition", c["key"], COND_BY_KEY[c["key"]], "pmh")
            prefix = {"A": "history of ", "B": "h/o ", "C": "past "}[prof.key] if c["status"] == "resolved" else ""
            pmh_items.append(prefix + name + self._onset_phrase(c, rec, prof, rng, acute=False))
        if rng.random() < 0.12:
            neg_pool = [k for k in ("t2dm", "copd", "ckd", "cirrhosis", "epilepsy") if k not in {c["key"] for c in conds}]
            if neg_pool:
                k = rng.choice(neg_pool)
                nm = ment("condition", k, COND_BY_KEY[k], "pmh", "negated")
                pmh_items.append({"A": f"no history of {nm}", "B": f"no h/o {nm}", "C": f"never had {nm}"}[prof.key])
        if pmh_items:
            sep = "; " if prof.key == "A" else ", "
            if narrative:
                lines.append("Past history: " + sep.join(pmh_items) + ".")
            else:
                lines.append(f"{H['pmh']} " + sep.join(pmh_items) + ".")

        # medications ----------------------------------------------------------
        home = [m for m in rec.meds if m["category"] == "community"]
        if home:
            items = [self._med_text(m, prof, rng, ment, use_old=True, section="meds") for m in home]
            if narrative:
                lines.append("Takes " + ", ".join(items) + ".")
            elif prof.key == "A":
                lines.append(H["meds"])
                lines += [f"  {k + 1}. {t}" for k, t in enumerate(items)]
            else:
                lines.append(f"{H['meds']} " + "; ".join(items))

        # allergies ------------------------------------------------------------
        al_items = []
        for a in rec.allergies:
            nm = ment("allergen", a["key"], ALLERGEN_BY_KEY[a["key"]], "allergies")
            extra = []
            if a["reaction"]:
                rxd = next(r for r in REACTIONS if r["key"] == a["reaction"])
                extra.append(((a["severity"] + " ") if a["severity"] else "") + rng.choice(rxd["forms"]))
            if a["onset"]:
                extra.append(str(a["onset"][1].year) if a["onset"][0] == "year" else a["onset"][1])
            sep = {"A": " (", "B": " -> ", "C": " - "}[prof.key]
            al_items.append(nm + ((sep + ", ".join(extra) + (")" if prof.key == "A" else "")) if extra else ""))
        for k in rec.delabeled:
            nm = ment("allergen", k, ALLERGEN_BY_KEY[k], "allergies", "refuted")
            al_items.append(f"previously labeled {nm} allergic; tolerated amoxicillin challenge, label removed")
        if not al_items:
            al_txt = rng.choice({"A": ["No known drug allergies."], "B": ["NKDA"], "C": ["none known", "NKDA"]}[prof.key])
        else:
            al_txt = "; ".join(al_items) + "."
        lines.append(("Allergic to " if narrative and al_items else ("Allergies: " if narrative else f"{H['allergies']} ")) + al_txt)

        # family history ---------------------------------------------------------
        if rec.family:
            fh = []
            for rel, k in rec.family:
                nm = ment("condition", k, COND_BY_KEY[k], "fhx", "family")
                fh.append(f"{rel} with {nm}" if prof.key != "B" else f"{rel[0].upper()}: {nm}")
            lines.append(("Family history: " if narrative else f"{H['fhx']} ") + "; ".join(fh) + ".")

        # review of systems (pertinent negatives) ---------------------------------
        negs = list(rec.ros_neg)
        if rec.flags.get("force_ros_neg") and rec.flags["force_ros_neg"] not in negs:
            negs.append(rec.flags["force_ros_neg"])
        names = [ment("condition", k, COND_BY_KEY[k], "ros", "negated") for k in negs]
        if names:
            lead = {"A": "Denies ", "B": "neg for ", "C": "no "}[prof.key]
            joiner = ", " if prof.key != "C" else ", no "
            lines.append(("" if narrative else f"{H['ros']} ") + lead + joiner.join(names) + ".")

        # exam --------------------------------------------------------------------
        if not narrative:
            sat = rng.randint(90, 99)
            lines.append(f"{H['exam']} BP {rng.randint(100, 170)}/{rng.randint(55, 100)}, HR {rng.randint(55, 115)}, "
                         f"SpO2 {sat}% on {'RA' if rng.random() < 0.6 else '2L NC'}. "
                         + rng.choice(["Lungs clear.", "Mild bibasilar crackles.", "No edema.", "Abdomen soft."]))

        # labs --------------------------------------------------------------------
        if rec.labs:
            first = [l for l in rec.labs if l["day"] == 0]
            rep = [l for l in rec.labs if l["day"] > 0]

            def labtxt(l):
                nm = ment("lab", l["key"], LAB_BY_KEY[l["key"]], "labs")
                u = UNIT_TEXT.get(LAB_BY_KEY[l["key"]]["unit"], "")
                return f"{nm} {l['value']}{(' ' + u) if u and u != '%' else u}"

            if prof.key == "A":
                lab_line = f"{fmt_date(rec.admit, ds)}: " + ", ".join(labtxt(l) for l in first) + "."
            elif prof.key == "B":
                lab_line = f"({fmt_date(rec.admit, ds, rng=rng)}) " + " ".join(labtxt(l) for l in first)
            else:
                lab_line = ", ".join(labtxt(l) for l in first) + " on admission."
            for l in rep:
                lab_line += f" Repeat {labtxt(l)} on hospital day #{l['day'] + 1}."
            lines.append(("Labs: " if narrative else f"{H['labs']} ") + lab_line)

        # plan ----------------------------------------------------------------------
        plan = []
        for m in rec.meds:
            if m["category"] == "inpatient":
                t = self._med_text(m, prof, rng, ment, use_old=False, section="plan")
                when = ""
                if m["start_expr"] == "absolute":
                    when = f" ({fmt_date(m['start'], ds)})"
                elif m["start_expr"] == "hd":
                    when = f" on hospital day #{(m['start'] - rec.admit).days + 1}"
                plan.append({"A": "Start ", "B": "start ", "C": "Started "}[prof.key] + t + when)
            elif m["action"] == "dose_change":
                nm = ment("drug", m["key"], DRUG_BY_KEY[m["key"]], "plan")
                plan.append(f"{'Increase' if m['dose'] > m['old_dose'] else 'Decrease'} {nm} to {_num(m['dose'])} {UNIT_TEXT[m['unit']]}")
            elif m["action"] == "stop":
                nm = ment("drug", m["key"], DRUG_BY_KEY[m["key"]], "plan")
                plan.append(rng.choice({"A": [f"Discontinue {nm}", f"Stop {nm}"], "B": [f"d/c {nm}", f"D/C {nm}"],
                                        "C": [f"stopped {nm}", f"took off {nm}"]}[prof.key]))
            elif m["action"] == "hold":
                nm = ment("drug", m["key"], DRUG_BY_KEY[m["key"]], "plan")
                plan.append({"A": f"Hold {nm}", "B": f"hold {nm}", "C": f"holding {nm} for now"}[prof.key])
        if plan:
            lines.append(("Plan: " if narrative else f"{H['plan']} ") + ". ".join(plan) + ".")

        if rec.note_type == "dc":
            lines.append(rng.choice(FILLER["course"]))
            active = [m for m in rec.meds if m["status"] == "active"]
            if active:
                items = [self._med_text(m, prof, rng, ment, use_old=False, section="dcmeds") for m in active]
                lines.append(("Discharge medications: " if narrative else f"{H['dcmeds']} ") + "; ".join(items) + ".")
        return "\n".join(lines)

    def _onset_phrase(self, c: Dict, rec: Rec, prof: Profile, rng, acute: bool) -> str:
        e = c["expr"]
        ds = prof.date_style
        if e["kind"] == "none":
            return ""
        if e["kind"] == "vague":
            return f" ({e['text']})" if not acute else f" starting {e['text']}"
        if e["kind"] == "relative":
            n = e["days"]
            return {"A": f" beginning {n} days prior to admission", "B": f" x{n}d PTA",
                    "C": f" for {n} days before coming in"}[prof.key]
        prec = e["precision"]
        d = fmt_date(c["onset"], ds, prec)
        if acute:
            return {"A": f" since {d}", "B": f" since {d}", "C": f" since {d}"}[prof.key]
        return {"A": f" (diagnosed {d})", "B": f" dx {d}", "C": f" since {d}"}[prof.key]

    def _med_text(self, m: Dict, prof: Profile, rng, ment, use_old: bool, section: str) -> str:
        d = DRUG_BY_KEY[m["key"]]
        nm = ment("drug", m["key"], d, section)
        dose = m.get("old_dose", m["dose"]) if use_old else m["dose"]
        unit_txt = UNIT_TEXT[m["unit"]]
        val, u = dose, unit_txt
        if m["unit"] == "mg" and dose >= 1000 and rng.random() < 0.5:
            val, u = dose / 1000, "g"
            m.setdefault("written_g", set()).add(section)
        if prof.key == "B":
            num = _num(val)
            if num.startswith("0.") and rng.random() < 0.3:
                num = num[1:]
            dose_txt = f"{num}{u}" if u in ("mg", "mcg", "g") else f"{num} {u}"
        else:
            dose_txt = f"{_num(val)} {u}"
        route = ""
        if m["route_stated"]:
            rf = ROUTES[m["route"]]["forms"]
            route = " " + (rf[0].lower() if prof.key == "B" else (rf[1] if prof.key == "C" and len(rf) > 1 else rf[0]))
        ff = FREQS[m["freq"]]["forms"]
        freq = " " + ({"A": ff[0] if not ff[0].isupper() else ff[1], "B": ff[0], "C": ff[1] if len(ff) > 1 else ff[0]}[prof.key])
        prn = (" prn" if prof.key == "B" else " as needed") if m["prn"] else ""
        start = ""
        if use_old and m["category"] == "community" and m["start"] is not None:
            if m["start_expr"] == "year":
                start = {"A": f" (since {m['start'].year})", "B": f" (started {m['start'].year})",
                         "C": f", on it since {m['start'].year}"}[prof.key]
            elif m["start_expr"] == "vague":
                start = " (for years)"
        return f"{nm} {dose_txt}{route}{freq}{prn}{start}"


def _num(x) -> str:
    if float(x).is_integer():
        return str(int(x))
    return f"{x:g}"


def _dose_as_written(m: Dict):
    """Gold dose follows the latest statement; grams are kept as grams when written so."""
    return float(m["dose"]), m["unit"]


def _poisson(rng: random.Random, lam: float) -> int:
    L, k, p = 2.718281828 ** (-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def make_record(gen: Generator, i: int, prof: Profile) -> Dict:
    rec = gen.sample_state(i, prof)
    rng = random.Random(_seed(gen.seed, "render", prof.key, i))
    text = gen.render(rec, prof, rng)
    gold = gen.gold_bundle(rec)
    state = Generator.final_state(rec)
    signals = fire_rules(state, gen.ref_kb)
    tiers: Dict[str, int] = {}
    for mt in rec.mentions:
        tiers[mt.tier] = tiers.get(mt.tier, 0) + 1
    meta = dict(
        age=rec.age, sex=rec.sex, race=rec.race, note_type=rec.note_type, n_meds=len(rec.meds),
        n_conditions=len(rec.conditions), n_allergies=len(rec.allergies),
        temporal_vague=sum(1 for c in rec.conditions if c["expr"]["kind"] == "vague") +
        sum(1 for m in rec.meds if m["start_expr"] == "vague") +
        sum(1 for a in rec.allergies if a["onset"] and a["onset"][0] == "vague"),
        conflict=rec.flags.get("conflict"), near_miss=rec.flags.get("near_miss"),
        injected_rule=rec.flags.get("injected_rule"), source_inconsistent=rec.flags.get("source_inconsistent"),
        tiers=tiers,
        mentions=[dict(kind=m.kind, key=m.key, text=m.text, tier=m.tier, section=m.section, assertion=m.assertion)
                  for m in rec.mentions],
        n_tokens=len(text.split()),
    )
    return dict(id=rec.rid, site=prof.name, profile=prof.key, text=text, gold=gold,
                gold_signals=[dict(type=s["type"], components=list(s["components"]), severity=s["severity"],
                                   rule_id=s["rule_id"]) for s in signals],
                gold_state=state, note_date=iso(rec.note_date), admit=iso(rec.admit), meta=meta)


def generate(n_per_site: Dict[str, int], seed: int = 42, profiles: Optional[Dict[str, Profile]] = None):
    gen = Generator(seed)
    profiles = profiles or PROFILES
    for key, n in n_per_site.items():
        prof = profiles[key]
        for i in range(n):
            yield make_record(gen, i, prof)


def drift_profile(base: str, kind: str, magnitude: float) -> Profile:
    """Documentation profile after a shift of the given kind and magnitude in [0, 1]."""
    p = PROFILES[base]
    if kind == "lexical":      # new vendor vocabulary: unseen surface forms
        return replace(p, key=base, novel_p=0.6 * magnitude, typo_p=p.typo_p + 0.02 * magnitude)
    if kind == "template":     # new note template and date format
        heads = dict(p.headers)
        if magnitude > 0:
            heads.update(pmh="Problems:", meds="Current outpatient prescriptions:", allergies="Intolerances:",
                         labs="Results review:", plan="Orders:")
        return replace(p, key=base, headers=heads, date_style="eu" if magnitude >= 0.5 else p.date_style,
                       novel_p=0.25 * magnitude)
    if kind == "population":   # older, polypharmacy population
        return replace(p, key=base, age=(p.age[0] + 15 * magnitude, p.age[1]), extra_meds=0.6 * magnitude,
                       pv_inject_p=p.pv_inject_p * (1 + magnitude))
    raise ValueError(kind)


def write_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
