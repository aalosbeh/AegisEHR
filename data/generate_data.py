"""Generate a small synthetic clinical-note corpus for Aegis-EHR.

Mimics the three-site distribution from the paper:
    Site-A (academic)   290K notes -> 40% of synthetic mix
    Site-B (community)  245K notes -> 33% of synthetic mix
    Site-C (rural)      200K notes -> 27% of synthetic mix
Per-site abbreviation density drawn from N(14.3, 6.2), N(18.7, 8.9), N(9.8, 4.1).
Injects 22% temporal ambiguity, 8% medication conflicts, 13% allergy mentions.

[SIMULATED] -- this corpus is procedurally generated, NOT real PHI.

Author: Anas AlSobeh, Utah Valley University, 2026
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = PROJECT_ROOT / "data" / "synthetic_corpus.jsonl"
STATS_PATH = PROJECT_ROOT / "data" / "site_stats.csv"

# ---------------------------------------------------------------------------
# Fixed clinical vocabulary (50 SNOMED-like codes)
# ---------------------------------------------------------------------------
SNOMED_VOCAB = [
    ("38341003",  "Hypertension"),
    ("44054006",  "Type 2 diabetes"),
    ("13645005",  "COPD"),
    ("195967001", "Asthma"),
    ("84114007",  "Heart failure"),
    ("22298006",  "Myocardial infarction"),
    ("230690007", "Stroke"),
    ("90708001",  "Renal disease"),
    ("363346000", "Malignant neoplasm"),
    ("271737000", "Anemia"),
    ("40930008",  "Hypothyroidism"),
    ("73211009",  "Diabetes mellitus"),
    ("49436004",  "Atrial fibrillation"),
    ("66071002",  "Hepatitis B"),
    ("128302006", "Chronic hepatitis C"),
    ("709044004", "Chronic kidney disease"),
    ("254837009", "Breast cancer"),
    ("363406005", "Colon cancer"),
    ("363418001", "Pancreatic cancer"),
    ("32398004",  "Bronchitis"),
    ("233604007", "Pneumonia"),
    ("56717001",  "Tuberculosis"),
    ("431855005", "Chronic kidney disease stage 4"),
    ("55822004",  "Hyperlipidemia"),
    ("35489007",  "Depression"),
    ("197480006", "Anxiety disorder"),
    ("47505003",  "PTSD"),
    ("65363002",  "Otitis media"),
    ("78275009",  "Eustachian salpingitis"),
    ("232717009", "Coronary artery bypass grafting"),
    ("3738000",   "Viral hepatitis"),
    ("271807003", "Skin rash"),
    ("422587007", "Nausea"),
    ("422400008", "Vomiting"),
    ("21522001",  "Abdominal pain"),
    ("29857009",  "Chest pain"),
    ("267036007", "Dyspnea"),
    ("386661006", "Fever"),
    ("84229001",  "Fatigue"),
    ("25064002",  "Headache"),
    ("162397003", "Pain in throat"),
    ("49727002",  "Cough"),
    ("64531003",  "Nasal discharge"),
    ("271594007", "Syncope"),
    ("16001004",  "Otalgia"),
    ("3199001",   "Auricular hematoma"),
    ("267102003", "Sore throat"),
    ("162076009", "Excessive flatus"),
    ("249366005", "Joint stiffness"),
    ("309253009", "Loss of appetite"),
]
assert len(SNOMED_VOCAB) >= 50

ABBREVIATIONS = [
    "BP", "HR", "RR", "T", "SpO2", "Hgb", "Hct", "WBC", "Plt", "Na",
    "K", "Cl", "BUN", "Cr", "Gluc", "AST", "ALT", "Tbili", "INR", "PTT",
    "EKG", "CXR", "CT", "MRI", "U/A", "PRN", "BID", "TID", "QID", "QD",
    "PO", "IV", "IM", "SC", "NPO", "DNR", "ICU", "ER", "OR", "PACU",
    "CHF", "DM", "HTN", "CAD", "COPD", "CKD", "ESRD", "MI", "CVA", "TIA",
]

MEDS = [
    "lisinopril 10 mg", "metformin 500 mg", "atorvastatin 40 mg",
    "warfarin 5 mg", "aspirin 81 mg", "amlodipine 5 mg",
    "levothyroxine 50 mcg", "albuterol inhaler", "omeprazole 20 mg",
    "insulin glargine 20 units", "furosemide 40 mg", "carvedilol 12.5 mg",
    "clopidogrel 75 mg", "rosuvastatin 20 mg", "spironolactone 25 mg",
    "amoxicillin 500 mg", "azithromycin 250 mg", "ciprofloxacin 500 mg",
    "ibuprofen 600 mg", "acetaminophen 500 mg",
]

ALLERGIES = ["penicillin", "sulfa", "latex", "iodine", "peanuts", "shellfish",
             "codeine", "morphine", "ibuprofen"]

NOTE_KINDS = ["admission", "progress", "discharge"]

NUM_RESOURCES = 16
NUM_AXIOMS = 27
NUM_PHARMA = 3


# ---------------------------------------------------------------------------
def _site_choice(r: random.Random) -> str:
    p = r.random()
    if p < 0.40:
        return "Site-A"
    if p < 0.73:
        return "Site-B"
    return "Site-C"


def _abbrev_density(site: str, r: random.Random) -> float:
    if site == "Site-A":
        mu, sd = 14.3, 6.2
    elif site == "Site-B":
        mu, sd = 18.7, 8.9
    else:
        mu, sd = 9.8, 4.1
    return max(0.0, r.gauss(mu, sd))


def _generate_note(site: str, kind: str, r: random.Random) -> Dict[str, Any]:
    density = _abbrev_density(site, r)  # abbreviations per 1000 tokens
    target_words = r.randint(80, 220)
    n_abbrev = max(0, int(round(density * target_words / 1000.0)))

    # pick 2-5 diagnoses
    dxs = r.sample(SNOMED_VOCAB, k=r.randint(2, 5))
    meds = r.sample(MEDS, k=r.randint(1, 4))
    allergies: List[str] = []
    if r.random() < 0.13:
        allergies = r.sample(ALLERGIES, k=r.randint(1, 2))

    parts: List[str] = []
    parts.append(
        f"[{kind.upper()} NOTE - {site}] "
        f"Pt is a {r.randint(22, 92)}-yo {r.choice(['M','F'])} "
        f"presenting with {dxs[0][1].lower()}."
    )
    parts.append(
        f"PMH: {', '.join(d[1] for d in dxs[1:])}."
    )
    parts.append(
        f"Meds: {', '.join(meds)}."
    )
    if allergies:
        parts.append(f"Allergies: {', '.join(allergies)}.")
    else:
        parts.append("Allergies: NKDA.")

    # vitals with abbreviations
    abv_pool = r.sample(ABBREVIATIONS, k=min(n_abbrev, len(ABBREVIATIONS)))
    if abv_pool:
        vit = ", ".join(
            f"{a} {r.randint(40, 180)}" for a in abv_pool[:8]
        )
        parts.append(f"Vitals: {vit}.")

    # temporal ambiguity (22% of records)
    temporal_violated: List[int] = []
    if r.random() < 0.22:
        parts.append(
            "Pt reports symptoms started 'a few days ago', possibly prior to medication start."
        )
        # mark 1-3 axiom indices violated
        temporal_violated = r.sample(range(NUM_AXIOMS), k=r.randint(1, 3))

    # medication conflicts (8%)
    pharma_alerts: List[int] = []
    if r.random() < 0.08:
        parts.append(
            "Note: concomitant warfarin and ibuprofen prescribed -- DDI risk."
        )
        pharma_alerts.append(0)  # DDI

    # allergy mention with prescribed med (subset of pharma alerts)
    if allergies and r.random() < 0.30:
        parts.append(
            f"Caution: documented {allergies[0]} allergy with newly ordered antibiotic."
        )
        pharma_alerts.append(1)  # allergy collision

    note_text = " ".join(parts)

    # FHIR bundle (synthetic; resource indices map to NUM_RESOURCES)
    entries = [
        {"resourceType": "Patient",          "resourceType_idx": 0},
        {"resourceType": "Encounter",        "resourceType_idx": 1},
    ]
    for d in dxs:
        entries.append({
            "resourceType": "Condition",
            "resourceType_idx": 2,
            "code": d[0],
            "display": d[1],
        })
    for m in meds:
        entries.append({
            "resourceType": "MedicationRequest",
            "resourceType_idx": 3,
            "medication": m,
        })
    for a in allergies:
        entries.append({
            "resourceType": "AllergyIntolerance",
            "resourceType_idx": 4,
            "substance": a,
        })

    spi_target = round(0.85 + r.random() * 0.13, 4)  # 0.85-0.98

    return {
        "note_text": note_text,
        "site": site,
        "kind": kind,
        "abbrev_density": round(density, 3),
        "fhir_bundle": {"resourceType": "Bundle", "entry": entries},
        "entities": [d[1] for d in dxs] + meds + allergies,
        "temporal_axioms_violated": temporal_violated,
        "pharma_alerts": pharma_alerts,
        "spi_target": spi_target,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="Number of notes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=str(OUT_PATH))
    args = ap.parse_args()

    r = random.Random(args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"Site-A": 0, "Site-B": 0, "Site-C": 0}
    abbrev_sums = {"Site-A": 0.0, "Site-B": 0.0, "Site-C": 0.0}

    with out_path.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            site = _site_choice(r)
            kind = r.choice(NOTE_KINDS)
            rec = _generate_note(site, kind, r)
            counts[site] += 1
            abbrev_sums[site] += rec["abbrev_density"]
            f.write(json.dumps(rec) + "\n")

    # site stats csv
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with STATS_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["site", "n_notes", "share_pct", "mean_abbrev_density",
                    "paper_n_notes [SIMULATED]"])
        paper_n = {"Site-A": 290_000, "Site-B": 245_000, "Site-C": 200_000}
        total = max(args.n, 1)
        for s in ("Site-A", "Site-B", "Site-C"):
            mean_ab = abbrev_sums[s] / max(counts[s], 1)
            w.writerow([s, counts[s], round(100 * counts[s] / total, 2),
                        round(mean_ab, 3), paper_n[s]])

    print(f"Wrote {args.n} synthetic records to {out_path}")
    print(f"Site distribution: {counts}")
    print(f"Stats: {STATS_PATH}")


if __name__ == "__main__":
    main()
