"""Shared text processing used by the agents and the deterministic baselines.

``parse_note`` runs in one of four modes so that systems differ only in the
capabilities they are given:

* ``rule``   - restricted dictionary (canonical names and multi-word synonyms),
               no sectioning beyond allergy lists, no assertion handling,
               ISO and US dates only;
* ``schema`` - base + learned lexicon, section-aware typing, exact matching,
               no assertion handling, absolute dates in the mention window
               (the Schema Agent's structural mapping);
* ``single`` - ``schema`` plus NegEx-style negation and relative day
               expressions (the single-pass pipeline baseline);
* ``full``   - everything: typo-tolerant matching, ambiguity resolution,
               negation / family history / refuted allergies / history cues,
               relative, hospital-day and vague temporal expressions, plan
               actions and repeat laboratory values (used by the Reasoning and
               Temporal agents).
"""
from __future__ import annotations

import datetime as dt
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .terminology import (ALLERGEN_BY_KEY, COND_BY_KEY, DRUG_BY_KEY, FREQS, LAB_BY_KEY, ROUTES,
                          base_lexicon)

# events for which "history of" means the condition is no longer active
EVENT_CONCEPTS = {"mi", "stroke", "dvt", "pe", "pna", "uti", "gi_bleed", "cellulitis", "sepsis"}
# domain knowledge about ambiguous abbreviations: (surface, section family) -> (kind, key)
AMBIG_RULES = {("ms", "med"): ("drug", "morphine"), ("ms", "problem"): ("condition", "ms"),
               ("pe", "problem"): ("condition", "pe"), ("ra", "problem"): ("condition", "rheumatoid")}

TOKEN_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}\.\d{1,2}\.\d{4}|\d{1,2}\.\d{4}|\d{1,2}/\d{4}|"
    r"\d{1,2}-[A-Za-z]{3}-\d{2,4}|#\d+|\d+(?:\.\d+)?(?=[A-Za-z%])|\.\d+|\d+(?:\.\d+)?|"
    r"[A-Za-z][A-Za-z0-9]*(?:[\-/'][A-Za-z0-9]+)*\+?|[^\sA-Za-z0-9]")

BASE_HEADERS = {
    "chief complaint": "cc", "cc": "cc",
    "history of present illness": "hpi", "hpi": "hpi", "hx": "hpi",
    "past medical history": "pmh", "pmh": "pmh", "pmhx": "pmh", "past history": "pmh",
    "medications on admission": "meds", "medications": "meds", "home meds": "meds", "meds": "meds",
    "home medications": "meds",
    "allergies": "allergies", "all": "allergies", "allergy": "allergies",
    "family history": "fhx", "fhx": "fhx", "family": "fhx",
    "review of systems": "ros", "ros": "ros",
    "physical examination": "exam", "pe": "exam", "exam": "exam", "physical exam": "exam",
    "pertinent results": "labs", "labs": "labs", "lab": "labs", "results": "labs",
    "assessment and plan": "plan", "a/p": "plan", "plan": "plan", "assessment": "plan",
    "discharge medications": "dcmeds", "d/c meds": "dcmeds", "discharge meds": "dcmeds",
}
RULE_HEADERS = {"allergies": "allergies", "all": "allergies", "allergy": "allergies"}
HEADER_RE = re.compile(r"^\s*([A-Za-z][A-Za-z/&\- ]{0,40}):\s*(.*)$")
MED_SECTIONS = {"meds", "dcmeds", "plan"}
PROBLEM_SECTIONS = {"cc", "hpi", "pmh", "narrative", "unknown"}

NEG_TRIGGERS = [("no", "history", "of"), ("no", "h/o"), ("neg", "for"), ("negative", "for"), ("never", "had"),
                ("no", "evidence", "of"), ("free", "of"), ("absence", "of"), ("ruled", "out"),
                ("denies",), ("denied",), ("deny",), ("no",), ("without",), ("not",)]
NEG_STOP = {".", ";", ":", "but", "however", "except", "presents", "p/w", "came", "who", "reports", "with"}
FAMILY_CUES = {"mother", "father", "sister", "brother", "family", "m", "f"}
REFUTE_CUES = [("label", "removed"), ("previously", "labeled"), ("delabeled",), ("tolerated",)]
HIST_CUES = [("history", "of"), ("h/o",), ("past",), ("hx", "of")]
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                          "nov", "dec"])}
FULL_MONTHS = {m: i + 1 for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                               "august", "september", "october", "november", "december"])}
VAGUE = ["a few days ago", "recently", "over the past week", "several days ago", "long-standing", "for many years",
         "remote", "years ago", "for a long time", "as a child", "for years", "last week", "some time ago"]
UNIT_MAP = {"mg": "mg", "milligram": "mg", "milligrams": "mg", "mcg": "ug", "microgram": "ug", "micrograms": "ug",
            "ug": "ug", "g": "g", "gram": "g", "grams": "g", "units": "[iU]", "unit": "[iU]", "u": "[iU]",
            "meq": "meq", "puff": "{puff}", "puffs": "{puff}"}
ROUTE_MAP = {}
for _rk, _rv in ROUTES.items():
    for _f in _rv["forms"]:
        ROUTE_MAP[_f.lower()] = _rk
FREQ_MAP = {}
for _fk, _fv in FREQS.items():
    for _f in _fv["forms"]:
        FREQ_MAP[_f.lower()] = _fk
STOPWORDS = set("""a an the and or of with for to in on at by from is was were be been has have had his her
their pt patient who presents presented p/w came come coming reports noted denies admitted admission
since started start stop stopped hold increase decrease daily twice once times as needed prn po iv sc
mg mcg g units puffs meq year years day days ago old y/o yo male female man woman m f history h/o
diagnosed dx past prior before after then x also new home meds plan labs on admission repeat hospital
note date discharge lungs clear soft abdomen edema crackles mild bibasilar no neg family mother father
sister brother""".split())


def norm(tok: str) -> str:
    t = tok.lower()
    return t[:-1] if len(t) > 1 and t.endswith(".") else t


def tokenize(text: str):
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------
class Lexicon:
    """Longest-match dictionary over normalized token sequences, plus a
    symmetric-delete index for edit-distance-1 lookups (typo tolerance)."""

    def __init__(self, entries: List[Tuple[str, str, str, str, float]]):
        # entries: (surface, kind, key, tier, weight)
        self.trie: Dict = {}
        self.entries = []
        for surface, kind, key, tier, w in entries:
            self.add(surface, kind, key, tier, w)
        self._fuzzy = None

    def add(self, surface, kind, key, tier, w=1.0):
        toks = tuple(norm(t) for t, _, _ in tokenize(surface))
        if not toks:
            return
        node = self.trie
        for t in toks:
            node = node.setdefault(t, {})
        lst = node.setdefault("$", [])
        if not any(x[0] == kind and x[1] == key for x in lst):
            lst.append((kind, key, tier, w))
        self.entries.append((toks, kind, key, tier, w))
        self._fuzzy = None

    def match(self, ntoks: List[str], i: int):
        node, best, j = self.trie, None, i
        while j < len(ntoks) and ntoks[j] in node:
            node = node[ntoks[j]]
            j += 1
            if "$" in node:
                best = (j, node["$"])
        return best

    # -- typo tolerance --------------------------------------------------------
    def _build_fuzzy(self):
        idx = defaultdict(set)
        for toks, kind, key, tier, w in self.entries:
            s = " ".join(toks)
            if len(s.replace(" ", "")) < 5:
                continue
            idx[s].add((kind, key, s))
            for k in range(len(s)):
                idx[s[:k] + s[k + 1:]].add((kind, key, s))
        self._fuzzy = idx

    def fuzzy(self, s: str):
        if self._fuzzy is None:
            self._build_fuzzy()
        if len(s.replace(" ", "")) < 5:
            return []
        cands = set(self._fuzzy.get(s, set()))
        for k in range(len(s)):
            cands |= self._fuzzy.get(s[:k] + s[k + 1:], set())
        return sorted(cands)


def build_lexicon(mode: str, learned: Optional[List[Tuple[str, str, str, float]]] = None) -> Lexicon:
    entries = []
    for s, kind, key in base_lexicon(include_abbrev=(mode != "rule")):
        entries.append((s, kind, key, "base", 1.0))
    if mode != "rule" and learned:
        for s, kind, key, p in learned:
            entries.append((s, kind, key, "learned", p))
    return Lexicon(entries)


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------
def _mk(y, m=1, d=1):
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


def parse_date_token(tok: str, style_hint: Optional[str] = None):
    """Return (date, precision) for a single date-like token, or None."""
    t = tok.strip()
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m:
        d = _mk(int(m[1]), int(m[2]), int(m[3]))
        return (d, "day") if d else None
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", t)
    if m:
        y = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
        d = _mk(y, int(m[1]), int(m[2]))
        return (d, "day") if d else None
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{4})", t)
    if m:
        d = _mk(int(m[3]), int(m[2]), int(m[1]))
        return (d, "day") if d else None
    m = re.fullmatch(r"(\d{1,2})-([A-Za-z]{3})-(\d{2,4})", t)
    if m and m[2].lower() in MONTHS:
        y = int(m[3]) + (2000 if len(m[3]) == 2 else 0)
        d = _mk(y, MONTHS[m[2].lower()], int(m[1]))
        return (d, "day") if d else None
    m = re.fullmatch(r"(\d{1,2})[/\.](\d{4})", t)
    if m and 1 <= int(m[1]) <= 12:
        return (_mk(int(m[2]), int(m[1])), "month")
    m = re.fullmatch(r"(19|20)\d{2}", t)
    if m:
        return (_mk(int(t)), "year")
    return None


def date_iso(d: dt.date, precision: str) -> str:
    if precision == "year":
        return f"{d.year:04d}"
    if precision == "month":
        return f"{d.year:04d}-{d.month:02d}"
    return d.isoformat()


@dataclass
class TExpr:
    kind: str                 # absolute | relative | hd | vague
    value: Optional[str]      # ISO string at precision (None for vague)
    text: str
    precision: Optional[str] = None


def find_temporal(window: str, admit: Optional[dt.date], mode: str) -> Optional[TExpr]:
    """Find the first temporal expression in a mention window."""
    low = window.lower()
    if mode == "full":
        for v in VAGUE:
            if v in low:
                return TExpr("vague", None, v)
    toks = [t for t, _, _ in tokenize(window)]
    ltoks = [t.lower() for t in toks]
    # relative day expressions anchored to admission
    if mode in ("single", "full") and admit is not None:
        m = re.search(r"(\d+)\s+days?\s+(prior to admission|before coming in|before admission|ago)", low) or \
            re.search(r"\bx(\d+)d\b", low) or re.search(r"\bfor (\d+) days before", low)
        if m:
            n = int(m.group(1))
            d = admit - dt.timedelta(days=n)
            return TExpr("relative", d.isoformat(), m.group(0), "day")
    if mode == "full" and admit is not None:
        m = re.search(r"hospital day #?(\d+)|\bhd ?#?(\d+)", low)
        if m:
            n = int(m.group(1) or m.group(2))
            d = admit + dt.timedelta(days=n - 1)
            return TExpr("hd", d.isoformat(), m.group(0), "day")
    for k, t in enumerate(toks):
        if mode == "rule" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}", t):
            continue
        p = parse_date_token(t)
        if p and p[0]:
            if p[1] == "year":
                prev = ltoks[k - 1] if k else ""
                nxt = ltoks[k + 1] if k + 1 < len(ltoks) else ""
                if nxt in UNIT_MAP or nxt in ("mg", "units", "unit"):
                    continue
                if prev in MONTHS or prev in FULL_MONTHS:
                    mm = MONTHS.get(prev) or FULL_MONTHS.get(prev)
                    d = _mk(p[0].year, mm)
                    return TExpr("absolute", date_iso(d, "month"), f"{toks[k - 1]} {t}", "month")
                cues = ("dx", "diagnosed", "since", "(", "started", "in", "on")
                if prev not in cues and not (mode == "full" and prev == ","):
                    continue
            return TExpr("absolute", date_iso(p[0], p[1]), t, p[1])
    return None


# ---------------------------------------------------------------------------
# Parse result structures
# ---------------------------------------------------------------------------
@dataclass
class Mention:
    kind: str
    key: str
    surface: str
    start: int
    end: int
    section: str
    match: str                # base | learned | fuzzy | ambiguous
    score: float
    assertion: str = "present"
    historical: bool = False
    texpr: Optional[TExpr] = None
    line: int = 0


@dataclass
class MedInfo:
    key: str
    dose: Optional[float] = None
    unit: Optional[str] = None
    route: Optional[str] = None
    freq: Optional[str] = None
    prn: bool = False
    texpr: Optional[TExpr] = None
    section: str = "meds"


@dataclass
class Parse:
    admit: Optional[dt.date]
    note_date: Optional[dt.date]
    discharge: Optional[dt.date]
    note_type: str
    gender: Optional[str]
    mentions: List[Mention] = field(default_factory=list)
    meds: List[MedInfo] = field(default_factory=list)          # every med mention with attributes, in order
    actions: List[Tuple[str, str, Optional[float], Optional[str]]] = field(default_factory=list)
    labs: List[Tuple[str, float, Optional[str]]] = field(default_factory=list)
    sections: Dict[str, int] = field(default_factory=dict)
    oov_tokens: int = 0
    clinical_tokens: int = 0
    unknown_headers: int = 0
    unknown_med_tokens: int = 0


# ---------------------------------------------------------------------------
# Sectioning
# ---------------------------------------------------------------------------
def content_section(line: str, lex: Lexicon) -> str:
    low = line.strip().lower()
    if low.startswith("takes ") or low.startswith("on "):
        return "meds"
    if low.startswith("allergic to") or low.startswith("allergies"):
        return "allergies"
    if low.startswith("family history"):
        return "fhx"
    if low.startswith("past history"):
        return "pmh"
    if low.startswith("labs") or low.startswith("lab"):
        return "labs"
    if low.startswith("plan"):
        return "plan"
    if low.startswith("discharge medications"):
        return "dcmeds"
    if re.match(r"^(no |denies|neg for)", low):
        return "ros"
    if re.search(r"\d+(\.\d+)?\s*(mg|mcg|units|meq|puffs)\b", low):
        return "meds"
    return "narrative"


def split_sections(text: str, headers: Dict[str, str], lex: Lexicon, mode: str):
    """Yield (line_no, section, body_text, body_offset)."""
    out = []
    cur = "narrative"
    offset = 0
    unknown = 0
    for ln, line in enumerate(text.split("\n")):
        m = HEADER_RE.match(line)
        body, boff = line, offset
        if m and len(m.group(1).split()) <= 5:
            h = m.group(1).strip().lower()
            if h in headers:
                cur = headers[h]
                body, boff = m.group(2), offset + m.start(2)
            elif h.startswith("date") or h.startswith("admission"):
                cur = "header"
            elif mode == "rule":
                cur = "narrative"
            else:
                unknown += 1
                cur = content_section(m.group(2), lex) if m.group(2).strip() else "unknown"
                body, boff = m.group(2), offset + m.start(2)
        elif ln == 0 or line.startswith("["):
            cur = "header"
        elif mode != "rule" and not line.startswith("  "):
            cs = content_section(line, lex)
            if cs != "narrative" or cur in ("header", "narrative"):
                cur = cs
        out.append((ln, cur, body, boff))
        offset += len(line) + 1
        if cur == "header":
            cur = "narrative"
    return out, unknown


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
def _header_dates(text: str):
    admit = note = dis = None
    ntype = "hp"
    m = re.search(r"Admission date:\s*(\S+)\s+Discharge date:\s*(\S+)", text)
    if m:
        a, d = parse_date_token(m[1]), parse_date_token(m[2])
        admit, dis = (a[0] if a else None), (d[0] if d else None)
        note, ntype = dis, "dc"
    else:
        m = re.search(r"Date:\s*(\S+)\s+Hospital day #(\d+)", text)
        if m:
            d = parse_date_token(m[1])
            if d:
                note = d[0]
                admit = note - dt.timedelta(days=int(m[2]) - 1)
            ntype = "progress"
        else:
            m = re.search(r"Date of admission:\s*(\S+)", text)
            if m:
                d = parse_date_token(m[1])
                admit = note = d[0] if d else None
    return admit, note, dis, ntype


def _gender(text: str) -> Optional[str]:
    m = re.search(r"\b\d+\s*(?:-year-old|y/o|yo)\s+(man|woman|M|F|male|female)\b", text)
    if not m:
        return None
    return "male" if m[1].lower() in ("man", "m", "male") else "female"


def _pick_candidate(cands, section: str, surface_norm: str, mode: str):
    fam = "med" if section in MED_SECTIONS else ("allergy" if section == "allergies" else
                                                 ("lab" if section == "labs" else "problem"))
    if mode == "full":
        rule = AMBIG_RULES.get((surface_norm, fam))
        if rule:
            return rule[0], rule[1], "ambiguous"
    pref = {"med": ["drug"], "allergy": ["allergen"], "lab": ["lab"], "problem": ["condition"]}[fam]
    for k in pref:
        for c in cands:
            if c[0] == k:
                return c[0], c[1], c[2]
    if mode == "rule":
        c = cands[0]
        return c[0], c[1], c[2]
    return None


def _assertion(ntoks: List[str], i: int, section: str, mode: str, line_ntoks: List[str], line_start: int):
    """NegEx-style scope check looking back from token i within the line."""
    if section == "fhx":
        return "family"
    if mode == "full" and section == "allergies":
        a = i
        while a > line_start and line_ntoks[a - 1] != ";":
            a -= 1
        b = i
        while b < len(line_ntoks) and line_ntoks[b] != ";":
            b += 1
        seg = " ".join(line_ntoks[a:b])
        if any(" ".join(c) in seg for c in REFUTE_CUES):
            return "refuted"
    j = i - 1
    while j >= line_start:
        t = ntoks[j]
        if t in NEG_STOP and t != "with":
            break
        for trig in NEG_TRIGGERS:
            L = len(trig)
            if tuple(ntoks[j - L + 1:j + 1]) == trig and j - L + 1 >= line_start:
                return "negated"
        if t == "with" and mode == "full":
            break
        j -= 1
    if mode == "full" and section in ("narrative", "unknown"):
        seg = ntoks[max(line_start, i - 4):i]
        if any(c in seg for c in ("mother", "father", "sister", "brother")):
            return "family"
    return "present"


def _historical(ntoks, i, line_start):
    for cue in HIST_CUES:
        L = len(cue)
        for back in (L, L + 1):
            a = i - back
            if a >= line_start and tuple(ntoks[a:a + L]) == cue:
                return True
    return False


def parse_note(text: str, lex: Lexicon, mode: str = "full", headers: Optional[Dict[str, str]] = None) -> Parse:
    headers = headers if headers is not None else (RULE_HEADERS if mode == "rule" else BASE_HEADERS)
    admit, note, dis, ntype = _header_dates(text)
    P = Parse(admit=admit, note_date=note, discharge=dis, note_type=ntype, gender=_gender(text))
    lines, P.unknown_headers = split_sections(text, headers, lex, mode)
    for ln, section, body, boff in lines:
        if section == "header" or not body.strip():
            continue
        P.sections[section] = P.sections.get(section, 0) + 1
        toks = tokenize(body)
        ntoks = [norm(t) for t, _, _ in toks]
        line_mentions: List[Mention] = []
        i = 0
        covered = [False] * len(toks)
        while i < len(toks):
            m = lex.match(ntoks, i)
            if m:
                j, cands = m
                surf_norm = " ".join(ntoks[i:j])
                pick = _pick_candidate(cands, section, surf_norm, mode)
                if pick and not (section == "exam" and mode != "rule"):
                    kind, key, tier = pick
                    w = next((c[3] for c in cands if c[1] == key), 1.0)
                    match = "ambiguous" if tier == "ambiguous" else ("learned" if tier == "learned" else "base")
                    score = {"base": 1.0, "learned": 0.75 * w + 0.2, "ambiguous": 0.7}[match]
                    mt = Mention(kind, key, body[toks[i][1]:toks[j - 1][2]], boff + toks[i][1], boff + toks[j - 1][2],
                                 section, match, score, line=ln)
                    mt._ti = i  # token index within the line
                    line_mentions.append(mt)
                    for k in range(i, j):
                        covered[k] = True
                    i = j
                    continue
            i += 1
        # typo-tolerant pass over uncovered alphabetic spans (full mode only)
        if mode == "full" and section != "exam":
            i = 0
            while i < len(toks):
                if covered[i] or not ntoks[i][0].isalpha() or ntoks[i] in STOPWORDS:
                    i += 1
                    continue
                hit = None
                for n in (3, 2, 1):
                    if i + n > len(toks) or any(covered[i:i + n]):
                        continue
                    s = " ".join(ntoks[i:i + n])
                    cands = lex.fuzzy(s)
                    if cands:
                        cc = [(c[0], c[1], "fuzzy", 1.0) for c in cands]
                        pick = _pick_candidate(cc, section, s, "single")
                        if pick:
                            hit = (n, pick)
                            break
                if hit:
                    n, (kind, key, _) = hit
                    mt = Mention(kind, key, body[toks[i][1]:toks[i + n - 1][2]], boff + toks[i][1],
                                 boff + toks[i + n - 1][2], section, "fuzzy", 0.6, line=ln)
                    mt._ti = i
                    line_mentions.append(mt)
                    for k in range(i, i + n):
                        covered[k] = True
                    i += n
                else:
                    i += 1
        line_mentions.sort(key=lambda x: x._ti)
        # OOV statistics on clinical sections
        if section in PROBLEM_SECTIONS | MED_SECTIONS | {"pmh", "allergies", "ros", "fhx"}:
            for k, t in enumerate(ntoks):
                if t[0].isalpha() and len(t) > 2 and t not in STOPWORDS:
                    P.clinical_tokens += 1
                    if not covered[k] and t not in FREQ_MAP and t not in ROUTE_MAP and t not in UNIT_MAP:
                        P.oov_tokens += 1
                        if section in MED_SECTIONS:
                            P.unknown_med_tokens += 1
        # assertion and history
        for mt in line_mentions:
            if mode in ("single", "full"):
                mt.assertion = _assertion(ntoks, mt._ti, section, mode, ntoks, 0)
            if mode == "full" and mt.kind == "condition":
                mt.historical = _historical(ntoks, mt._ti, 0)
        # mention windows: temporal expressions and medication attributes
        for idx, mt in enumerate(line_mentions):
            nxt = line_mentions[idx + 1].start - boff if idx + 1 < len(line_mentions) else len(body)
            wstart = mt.end - boff
            window = body[wstart:nxt]
            cut = len(window)
            for d in (";",) + (("\n",)):
                p = window.find(d)
                if p >= 0:
                    cut = min(cut, p)
            if mt.kind == "condition":
                for d in (",", ". ", " who ", " p/w ", " came "):
                    p = window.find(d)
                    if p >= 0:
                        cut = min(cut, p)
            elif mt.kind == "drug":
                p = window.find(",")
                if p >= 0 and not window[p:].lower().startswith(", on it since"):
                    cut = min(cut, p)
            win = window[:cut]
            if mt.kind in ("condition", "allergen"):
                mt.texpr = find_temporal(win, P.admit, mode)
            if mt.kind == "drug":
                info = parse_med_window(mt.key, win, mode)
                info.section = section
                if section in MED_SECTIONS:
                    info.texpr = find_temporal(win, P.admit, mode)
                mt.texpr = info.texpr
                if section in MED_SECTIONS or mode == "rule":
                    P.meds.append(info)
            if mt.kind == "lab":
                mv = re.match(r"\s*(-?\d*\.?\d+)", window)
                if mv:
                    eff = None
                    if mode == "full":
                        rep = re.match(r"\s*-?\d*\.?\d+\s*\S*\s+on hospital day #(\d+)", window)
                        if rep and P.admit:
                            eff = (P.admit + dt.timedelta(days=int(rep[1]) - 1)).isoformat()
                    if eff is None:
                        eff = _line_date(body, P.admit, mode)
                    P.labs.append((mt.key, float(mv[1]), eff))
        # plan actions
        if section == "plan" and mode == "full":
            for mt in line_mentions:
                if mt.kind != "drug":
                    continue
                pre = body[max(0, mt.start - boff - 18):mt.start - boff].lower()
                post = body[mt.end - boff:mt.end - boff + 30]
                if re.search(r"(discontinue|stop|d/c|stopped|took off)\s*$", pre):
                    P.actions.append(("stop", mt.key, None, None))
                elif re.search(r"(hold|holding)\s*$", pre):
                    P.actions.append(("hold", mt.key, None, None))
                elif re.search(r"(increase|decrease)\s*$", pre):
                    mm = re.match(r"\s*to\s+(\d*\.?\d+)\s*([A-Za-z]+)", post)
                    if mm:
                        P.actions.append(("dose", mt.key, float(mm[1]), UNIT_MAP.get(mm[2].lower())))
        P.mentions.extend(line_mentions)
    return P


def _line_date(body: str, admit, mode):
    low = body.lower()
    for t, _, _ in tokenize(body):
        if mode == "rule" and not re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4}", t):
            continue
        p = parse_date_token(t)
        if p and p[0] and p[1] == "day":
            return p[0].isoformat()
    if mode in ("single", "full") and "on admission" in low and admit:
        return admit.isoformat()
    return None


def parse_med_window(key: str, win: str, mode: str) -> MedInfo:
    info = MedInfo(key=key)
    low = win.lower()
    m = re.search(r"(?<![\d.])(\d*\.?\d+)\s*(mg|mcg|micrograms?|milligrams?|grams?|g|units?|u|meq|puffs?)\b", low)
    if m:
        info.dose = float(m[1])
        info.unit = UNIT_MAP.get(m[2], None)
    for f in sorted(ROUTE_MAP, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", low):
            info.route = ROUTE_MAP[f]
            break
    for f in sorted(FREQ_MAP, key=len, reverse=True):
        if re.search(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", low):
            info.freq = FREQ_MAP[f]
            break
    info.prn = bool(re.search(r"\bprn\b|as needed", low))
    return info


# ---------------------------------------------------------------------------
# Learning site vocabulary and headers from training notes (self-supervised)
# ---------------------------------------------------------------------------
def gold_concepts(gold: Dict) -> Dict[str, set]:
    from .terminology import ALLERGEN_BY_CODE, COND_BY_CODE, DRUG_BY_CODE, LAB_BY_CODE
    out = {"condition": set(), "drug": set(), "allergen": set(), "lab": set()}
    for e in gold["entry"]:
        r = e["resource"]
        rt = r["resourceType"]
        try:
            if rt == "Condition":
                out["condition"].add(COND_BY_CODE[r["code"]["coding"][0]["code"]]["key"])
            elif rt == "MedicationRequest":
                out["drug"].add(DRUG_BY_CODE[r["medicationCodeableConcept"]["coding"][0]["code"]]["key"])
            elif rt == "AllergyIntolerance":
                out["allergen"].add(ALLERGEN_BY_CODE[r["code"]["coding"][0]["code"]]["key"])
            elif rt == "Observation":
                out["lab"].add(LAB_BY_CODE[r["code"]["coding"][0]["code"]]["key"])
        except KeyError:
            pass
    return out


ANY_BAD = re.compile(r"^(x\d+d|\d.*|pta|starting|beginning|since|recently|several|few|over|past|week|weeks|ago|dx|"
                     r"diagnosed|h/o|pmh|pmhx|/|long-standing|remote|years?|days?|coming|before|prior|admission|"
                     r"hospital|repeat|on|it|as|needed|prn)$")


def learn_vocabulary(records, min_count: int = 8, min_precision: float = 0.7, max_records: int = 30000,
                     single_precision: float = 0.85, single_count: int = 15):
    """Mine surface forms that the base lexicon misses.

    For each training note, n-grams (1-4 tokens) in uncovered, non-stopword
    regions are paired with the gold concepts of the matching kind that the
    base lexicon did not find in that note. A form is accepted when it
    co-occurs with one concept in at least ``min_precision`` of its
    occurrences and at least ``min_count`` times. Only the note text and its
    gold bundle are used.
    """
    lex = build_lexicon("schema")
    pair = Counter()
    occ = Counter()
    kind_of_section = {"meds": "drug", "dcmeds": "drug", "plan": "drug", "allergies": "allergen", "labs": "lab",
                       "pmh": "condition", "hpi": "condition", "cc": "condition", "narrative": None,
                       "unknown": None, "ros": None, "fhx": None}
    for n, r in enumerate(records):
        if n >= max_records:
            break
        gc = gold_concepts(r["gold"])
        lines, _ = split_sections(r["text"], BASE_HEADERS, lex, "schema")
        seen = set()
        for ln, section, body, boff in lines:
            kinds = [kind_of_section.get(section)] if kind_of_section.get(section) else ["condition", "drug"]
            toks = tokenize(body)
            nt = [norm(t) for t, _, _ in toks]
            cov = [False] * len(nt)
            in_line = defaultdict(set)
            i = 0
            while i < len(nt):
                m = lex.match(nt, i)
                if m:
                    for k in range(i, m[0]):
                        cov[k] = True
                    for c in m[1]:
                        in_line[c[0]].add(c[1])
                    i = m[0]
                else:
                    i += 1
            missing = {k: gc[k] - in_line[k] for k in gc}
            kinds = [k for k in kinds if missing.get(k)]
            if not kinds:
                continue
            for i in range(len(nt)):
                for L in (1, 2, 3, 4):
                    seg = nt[i:i + L]
                    if len(seg) < L or any(cov[i:i + L]):
                        break
                    if not all(re.match(r"[a-z]", t) or t in ("+", "/", "-") for t in seg):
                        break
                    if any(ANY_BAD.match(t) for t in seg):
                        break
                    if seg[0] in STOPWORDS or seg[-1] in STOPWORDS:
                        continue
                    if any(t in (",", ";", ".", ":", "(", ")") for t in seg):
                        break
                    s = " ".join(seg)
                    if (s, section) in seen:
                        continue
                    seen.add((s, section))
                    occ[s] += 1
                    for kind in kinds:
                        for key in missing[kind]:
                            pair[(s, kind, key)] += 1
    learned = []
    best = {}
    for (s, kind, key), c in pair.items():
        single = " " not in s
        if single and len(s) < 3:
            continue
        if c >= (single_count if single else min_count) and c / occ[s] >= (single_precision if single else min_precision):
            if s not in best or c > best[s][3]:
                best[s] = (s, kind, key, c)
    for s, kind, key, c in best.values():
        learned.append((s, kind, key, round(c / occ[s], 3)))
    return sorted(learned)


def learn_headers(records, min_count: int = 30, max_records: int = 20000):
    """Map unknown header strings to sections by the content found under them."""
    lex = build_lexicon("schema")
    votes = defaultdict(Counter)
    for n, r in enumerate(records):
        if n >= max_records:
            break
        for line in r["text"].split("\n"):
            m = HEADER_RE.match(line)
            if not m or len(m.group(1).split()) > 5:
                continue
            h = m.group(1).strip().lower()
            if h in BASE_HEADERS or h.startswith("date") or h.startswith("admission"):
                continue
            body = m.group(2)
            sec = content_section(body, lex)
            toks = [norm(t) for t, _, _ in tokenize(body)]
            kinds = Counter()
            i = 0
            while i < len(toks):
                mm = lex.match(toks, i)
                if mm:
                    for c in mm[1][:1]:
                        kinds[c[0]] += 1
                    i = mm[0]
                else:
                    i += 1
            if re.match(r"^\s*(no |denies|neg for)", body.lower()):
                sec = "ros"
            elif kinds.get("allergen") and not kinds.get("drug") or body.strip().lower() in ("nkda", "none known"):
                sec = "allergies"
            elif kinds.get("drug") and re.search(r"\d", body):
                sec = "meds" if not re.match(r"^\s*(start|stop|d/c|hold)", body.lower()) else "plan"
            elif kinds.get("lab"):
                sec = "labs"
            elif kinds.get("condition"):
                sec = "pmh"
            votes[h][sec] += 1
    out = {}
    for h, c in votes.items():
        if sum(c.values()) >= min_count:
            out[h] = c.most_common(1)[0][0]
    return out
