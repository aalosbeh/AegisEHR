"""Terminology inventory used by the synthetic benchmark and by every system.

Each concept carries its standard code and three tiers of surface forms:

* ``base``  - canonical name, common synonyms, standard abbreviations and brand
  names. This tier plays the role of a terminology server (UMLS-like) and is
  available to all systems, including the LLM prompts (canonical names only).
* ``site``  - forms preferred by one simulated documentation profile and absent
  from the base tier. They can only be learned from training notes of that
  profile, which is what makes leave-one-profile-out evaluation informative.
* ``rare``  - forms used in a small share of mentions and absent from the base
  tier. Few training examples exist, so lexical systems usually miss them.

Codes were taken from public terminology browsers (SNOMED CT International,
ICD-10-CM, RxNorm ingredient level, LOINC). They are used here as identifiers
for a synthetic benchmark; verify against current releases before reuse.
"""
from __future__ import annotations

SNOMED = "http://snomed.info/sct"
ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
LOINC = "http://loinc.org"
UCUM = "http://unitsofmeasure.org"


def _c(key, snomed, icd10, display, base, site=None, rare=None, chronic=True,
       prev=1.0, meds=(), age_slope=0.0):
    return dict(key=key, snomed=snomed, icd10=icd10, display=display,
                base=list(base), site=site or {}, rare=list(rare or []),
                chronic=chronic, prev=prev, meds=list(meds), age_slope=age_slope)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------
CONDITIONS = [
    _c("htn", "38341003", "I10", "Hypertension",
       ["hypertension", "high blood pressure", "HTN"],
       site={"B": ["HTN-ess"], "C": ["high BP"]}, rare=["hi blood pressure"],
       prev=5.0, meds=["lisinopril", "amlodipine", "losartan", "hydrochlorothiazide", "metoprolol"],
       age_slope=0.03),
    _c("t2dm", "44054006", "E11.9", "Type 2 diabetes mellitus",
       ["type 2 diabetes mellitus", "type 2 diabetes", "diabetes mellitus type 2", "T2DM", "DM2"],
       site={"B": ["DM-II", "NIDDM"], "C": ["sugar diabetes"]}, rare=["adult onset diabetes"],
       prev=3.5, meds=["metformin", "insulin_glargine"], age_slope=0.02),
    _c("copd", "13645005", "J44.9", "Chronic obstructive pulmonary disease",
       ["chronic obstructive pulmonary disease", "chronic obstructive lung disease", "COPD"],
       site={"C": ["bad lungs from smoking"]}, rare=["chronic airflow obstruction"],
       prev=1.5, meds=["albuterol", "prednisone"], age_slope=0.02),
    _c("asthma", "195967001", "J45.909", "Asthma",
       ["asthma", "reactive airway disease"], site={"B": ["RAD"]},
       prev=1.4, meds=["albuterol"], age_slope=-0.01),
    _c("hf", "84114007", "I50.9", "Heart failure",
       ["heart failure", "congestive heart failure", "CHF"],
       site={"B": ["HFrEF"], "C": ["weak heart"]}, rare=["cardiac decompensation"],
       prev=1.6, meds=["furosemide", "carvedilol", "spironolactone", "lisinopril", "digoxin"],
       age_slope=0.03),
    _c("mi", "22298006", "I21.9", "Myocardial infarction",
       ["myocardial infarction", "heart attack", "MI"], site={"B": ["NSTEMI"]},
       rare=["coronary occlusion event"], prev=0.9, meds=["aspirin", "atorvastatin", "metoprolol", "clopidogrel"],
       age_slope=0.02),
    _c("afib", "49436004", "I48.91", "Atrial fibrillation",
       ["atrial fibrillation", "AFib", "a-fib"], site={"B": ["AF w RVR"], "C": ["irregular heartbeat"]},
       rare=["auricular fibrillation"], prev=1.6, meds=["warfarin", "apixaban", "metoprolol", "diltiazem", "amiodarone", "digoxin"],
       age_slope=0.03),
    _c("ckd", "709044004", "N18.9", "Chronic kidney disease",
       ["chronic kidney disease", "chronic renal insufficiency", "CKD"],
       site={"B": ["CRI"], "C": ["kidney trouble"]}, rare=["chronic nephropathy"],
       prev=1.6, meds=["furosemide"], age_slope=0.03),
    _c("hld", "55822004", "E78.5", "Hyperlipidemia",
       ["hyperlipidemia", "high cholesterol", "dyslipidemia", "HLD"],
       site={"C": ["cholesterol problem"]}, prev=3.0, meds=["atorvastatin", "simvastatin"], age_slope=0.02),
    _c("hypothyroid", "40930008", "E03.9", "Hypothyroidism",
       ["hypothyroidism", "underactive thyroid"], site={"B": ["hypothyroid"], "C": ["low thyroid"]},
       prev=1.3, meds=["levothyroxine"]),
    _c("pna", "233604007", "J18.9", "Pneumonia",
       ["pneumonia", "PNA"], site={"B": ["CAP"]}, rare=["lung infection"], chronic=False,
       prev=1.4, meds=["ceftriaxone", "azithromycin", "levofloxacin", "amoxicillin"]),
    _c("depression", "35489007", "F32.9", "Depressive disorder",
       ["depressive disorder", "depression", "major depression", "MDD"],
       site={"C": ["low mood problem"]}, prev=1.6, meds=["sertraline", "citalopram", "fluoxetine"],
       age_slope=-0.005),
    _c("anxiety", "197480006", "F41.9", "Anxiety disorder",
       ["anxiety disorder", "anxiety", "generalized anxiety disorder", "GAD"],
       site={"C": ["nerves"]}, prev=1.2, meds=["sertraline"], age_slope=-0.01),
    _c("gerd", "235595009", "K21.9", "Gastroesophageal reflux disease",
       ["gastroesophageal reflux disease", "acid reflux", "GERD"], site={"C": ["heartburn trouble"]},
       prev=1.8, meds=["omeprazole", "pantoprazole"]),
    _c("oa", "396275006", "M19.90", "Osteoarthritis",
       ["osteoarthritis", "degenerative joint disease", "OA", "DJD"], site={"C": ["arthritis in knees"]},
       prev=1.8, meds=["acetaminophen", "ibuprofen", "naproxen", "tramadol"], age_slope=0.02),
    _c("cad", "53741008", "I25.10", "Coronary artery disease",
       ["coronary artery disease", "coronary arteriosclerosis", "CAD"], site={"B": ["ASCVD"]},
       rare=["ischemic heart disease"], prev=1.8, meds=["aspirin", "atorvastatin", "metoprolol", "clopidogrel"],
       age_slope=0.03),
    _c("stroke", "230690007", "I63.9", "Cerebrovascular accident",
       ["cerebrovascular accident", "stroke", "CVA"], site={"C": ["brain attack"]},
       prev=0.8, meds=["aspirin", "clopidogrel", "atorvastatin"], age_slope=0.02),
    _c("uti", "68566005", "N39.0", "Urinary tract infection",
       ["urinary tract infection", "UTI"], site={"C": ["bladder infection"]}, chronic=False,
       prev=1.2, meds=["ciprofloxacin", "tmp_smx", "ceftriaxone"]),
    _c("cellulitis", "128045006", "L03.90", "Cellulitis",
       ["cellulitis"], site={"B": ["SSTI"]}, rare=["skin infection of leg"], chronic=False,
       prev=0.8, meds=["tmp_smx", "amoxicillin", "vancomycin"]),
    _c("sepsis", "91302008", "A41.9", "Sepsis",
       ["sepsis"], site={"B": ["SIRS w source"]}, chronic=False, prev=0.5, meds=["vancomycin", "ceftriaxone"]),
    _c("anemia", "271737000", "D64.9", "Anemia",
       ["anemia", "anaemia"], site={"C": ["low blood"]}, prev=1.0),
    _c("pud", "13200003", "K27.9", "Peptic ulcer",
       ["peptic ulcer", "peptic ulcer disease", "PUD"], site={"C": ["stomach ulcer"]},
       prev=0.6, meds=["pantoprazole", "omeprazole"]),
    _c("gout", "90560007", "M10.9", "Gout", ["gout"], site={"C": ["gouty toe"]},
       prev=0.8, meds=["allopurinol"]),
    _c("obesity", "414916001", "E66.9", "Obesity", ["obesity"], site={"B": ["morbid obesity"]}, prev=1.5),
    _c("dvt", "128053003", "I82.409", "Deep venous thrombosis",
       ["deep venous thrombosis", "deep vein thrombosis", "DVT"], site={"C": ["blood clot in leg"]},
       prev=0.6, meds=["apixaban", "enoxaparin", "warfarin"]),
    _c("pe", "59282003", "I26.99", "Pulmonary embolism",
       ["pulmonary embolism", "PE"], site={"C": ["clot in lung"]}, prev=0.4,
       meds=["apixaban", "enoxaparin", "heparin"]),
    _c("epilepsy", "84757009", "G40.909", "Epilepsy",
       ["epilepsy", "seizure disorder"], site={"B": ["Sz d/o"], "C": ["fits"]}, prev=0.5, meds=["gabapentin"]),
    _c("migraine", "37796009", "G43.909", "Migraine", ["migraine"], site={"C": ["bad headaches"]},
       prev=0.7, age_slope=-0.01),
    _c("cirrhosis", "19943007", "K74.60", "Cirrhosis of liver",
       ["cirrhosis of liver", "liver cirrhosis", "cirrhosis"], site={"B": ["ESLD"]}, prev=0.4),
    _c("ms", "24700007", "G35", "Multiple sclerosis", ["multiple sclerosis", "MS"], prev=0.25, age_slope=-0.01),
    _c("osteoporosis", "64859006", "M81.0", "Osteoporosis", ["osteoporosis"], site={"C": ["thin bones"]},
       prev=0.9, meds=["alendronate"], age_slope=0.03),
    _c("bph", "266569009", "N40.0", "Benign prostatic hyperplasia",
       ["benign prostatic hyperplasia", "enlarged prostate", "BPH"], prev=0.7, meds=["tamsulosin"], age_slope=0.02),
    _c("hepc", "128302006", "B18.2", "Chronic hepatitis C",
       ["chronic hepatitis C", "hepatitis C", "HCV"], site={"C": ["hep C"]}, prev=0.4),
    _c("breast_ca", "254837009", "C50.919", "Malignant neoplasm of breast",
       ["malignant neoplasm of breast", "breast cancer", "breast CA"], prev=0.4, meds=["methotrexate"]),
    _c("hypokalemia", "43339004", "E87.6", "Hypokalemia", ["hypokalemia", "low potassium"],
       chronic=False, prev=0.5, meds=["potassium_chloride"]),
    _c("hyperkalemia", "14140009", "E87.5", "Hyperkalemia", ["hyperkalemia", "high potassium"],
       chronic=False, prev=0.3),
    _c("gi_bleed", "74474003", "K92.2", "Gastrointestinal hemorrhage",
       ["gastrointestinal hemorrhage", "GI bleed", "GIB"], site={"C": ["bleeding from the gut"]},
       chronic=False, prev=0.4, meds=["pantoprazole"]),
    _c("bradycardia", "48867003", "R00.1", "Bradycardia", ["bradycardia"], site={"B": ["brady"]},
       chronic=False, prev=0.3),
    _c("chest_pain", "29857009", "R07.9", "Chest pain", ["chest pain", "CP"], chronic=False, prev=1.0),
    _c("dyspnea", "267036007", "R06.00", "Dyspnea", ["dyspnea", "shortness of breath", "SOB"],
       chronic=False, prev=1.0),
    _c("rheumatoid", "69896004", "M06.9", "Rheumatoid arthritis", ["rheumatoid arthritis", "RA"],
       prev=0.4, meds=["methotrexate", "prednisone"]),
]

# Symptoms commonly listed as negatives in review of systems (pertinent negatives).
ROS_NEGATIVE_KEYS = ["chest_pain", "dyspnea", "pna", "gi_bleed", "stroke", "mi", "dvt", "pe", "uti"]
# Conditions plausibly reported in family history.
FAMILY_KEYS = ["mi", "t2dm", "breast_ca", "stroke", "htn", "cad", "depression", "ms", "hld"]


def _d(key, rxcui, display, base, doses, unit, routes, freqs, cls, site=None, rare=None,
       high_alert=False, prn=False):
    return dict(key=key, rxcui=rxcui, display=display, base=list(base), doses=list(doses),
                unit=unit, routes=list(routes), freqs=list(freqs), cls=cls, site=site or {},
                rare=list(rare or []), high_alert=high_alert, prn=prn)


# ---------------------------------------------------------------------------
# Medications (RxNorm ingredient level). Doses in the unit given.
# ---------------------------------------------------------------------------
DRUGS = [
    _d("lisinopril", "29046", "lisinopril", ["lisinopril", "Zestril", "Prinivil"], [5, 10, 20, 40], "mg", ["PO"], ["daily"], "acei"),
    _d("losartan", "52175", "losartan", ["losartan", "Cozaar"], [25, 50, 100], "mg", ["PO"], ["daily"], "arb"),
    _d("amlodipine", "17767", "amlodipine", ["amlodipine", "Norvasc"], [2.5, 5, 10], "mg", ["PO"], ["daily"], "ccb"),
    _d("hydrochlorothiazide", "5487", "hydrochlorothiazide", ["hydrochlorothiazide", "HCTZ"], [12.5, 25], "mg", ["PO"], ["daily"], "thiazide",
       site={"C": ["water pill"]}),
    _d("metoprolol", "6918", "metoprolol", ["metoprolol", "Lopressor", "Toprol"], [25, 50, 100], "mg", ["PO"], ["BID", "daily"], "bb_selective",
       site={"B": ["metop"]}),
    _d("carvedilol", "20352", "carvedilol", ["carvedilol", "Coreg"], [3.125, 6.25, 12.5, 25], "mg", ["PO"], ["BID"], "bb_nonselective"),
    _d("propranolol", "8787", "propranolol", ["propranolol", "Inderal"], [10, 20, 40], "mg", ["PO"], ["BID", "TID"], "bb_nonselective"),
    _d("diltiazem", "3443", "diltiazem", ["diltiazem", "Cardizem"], [120, 180, 240], "mg", ["PO"], ["daily"], "ccb_nd",
       site={"B": ["dilt"]}),
    _d("verapamil", "11170", "verapamil", ["verapamil", "Calan"], [80, 120], "mg", ["PO"], ["TID"], "ccb_nd"),
    _d("furosemide", "4603", "furosemide", ["furosemide", "Lasix"], [20, 40, 80], "mg", ["PO", "IV"], ["daily", "BID"], "loop",
       site={"C": ["fluid pill"]}),
    _d("spironolactone", "9997", "spironolactone", ["spironolactone", "Aldactone"], [25, 50], "mg", ["PO"], ["daily"], "k_sparing",
       site={"B": ["spiro"]}),
    _d("potassium_chloride", "8591", "potassium chloride", ["potassium chloride", "KCl", "K-Dur"], [10, 20, 40], "meq", ["PO"], ["daily", "BID"], "k_supp",
       high_alert=True),
    _d("digoxin", "3407", "digoxin", ["digoxin", "Lanoxin"], [0.125, 0.25], "mg", ["PO"], ["daily"], "glycoside", high_alert=True,
       site={"B": ["dig"]}),
    _d("amiodarone", "703", "amiodarone", ["amiodarone", "Cordarone", "Pacerone"], [200, 400], "mg", ["PO"], ["daily"], "antiarrhythmic",
       site={"B": ["amio"]}),
    _d("warfarin", "11289", "warfarin", ["warfarin", "Coumadin"], [2, 2.5, 5, 7.5], "mg", ["PO"], ["daily", "QHS"], "vka", high_alert=True,
       site={"C": ["blood thinner pill"]}),
    _d("apixaban", "1364430", "apixaban", ["apixaban", "Eliquis"], [2.5, 5], "mg", ["PO"], ["BID"], "doac", high_alert=True),
    _d("enoxaparin", "67108", "enoxaparin", ["enoxaparin", "Lovenox"], [40, 60, 80], "mg", ["SC"], ["daily", "BID"], "lmwh", high_alert=True),
    _d("heparin", "5224", "heparin", ["heparin", "UFH"], [5000], "[iU]", ["SC"], ["TID", "BID"], "ufh", high_alert=True),
    _d("aspirin", "1191", "aspirin", ["aspirin", "ASA"], [81, 325], "mg", ["PO"], ["daily"], "nsaid_asa",
       site={"C": ["baby aspirin"]}),
    _d("clopidogrel", "32968", "clopidogrel", ["clopidogrel", "Plavix"], [75], "mg", ["PO"], ["daily"], "p2y12"),
    _d("atorvastatin", "83367", "atorvastatin", ["atorvastatin", "Lipitor"], [10, 20, 40, 80], "mg", ["PO"], ["daily", "QHS"], "statin"),
    _d("simvastatin", "36567", "simvastatin", ["simvastatin", "Zocor"], [10, 20, 40], "mg", ["PO"], ["QHS"], "statin"),
    _d("metformin", "6809", "metformin", ["metformin", "Glucophage"], [500, 850, 1000], "mg", ["PO"], ["BID", "daily"], "biguanide",
       site={"C": ["sugar pill"]}),
    _d("insulin_glargine", "274783", "insulin glargine", ["insulin glargine", "Lantus", "glargine"], [10, 15, 20, 30], "[iU]", ["SC"], ["QHS"], "insulin",
       high_alert=True, site={"B": ["basal insulin"]}),
    _d("levothyroxine", "10582", "levothyroxine", ["levothyroxine", "Synthroid"], [25, 50, 75, 100], "ug", ["PO"], ["daily"], "thyroid",
       site={"C": ["thyroid pill"]}, rare=["T4 replacement"]),
    _d("omeprazole", "7646", "omeprazole", ["omeprazole", "Prilosec"], [20, 40], "mg", ["PO"], ["daily"], "ppi"),
    _d("pantoprazole", "40790", "pantoprazole", ["pantoprazole", "Protonix"], [40], "mg", ["PO", "IV"], ["daily", "BID"], "ppi"),
    _d("albuterol", "435", "albuterol", ["albuterol", "ProAir", "Ventolin"], [2], "{puff}", ["INH"], ["Q4H"], "saba", prn=True),
    _d("prednisone", "8640", "prednisone", ["prednisone"], [5, 10, 20, 40], "mg", ["PO"], ["daily"], "steroid"),
    _d("ibuprofen", "5640", "ibuprofen", ["ibuprofen", "Motrin", "Advil"], [400, 600, 800], "mg", ["PO"], ["TID", "Q6H"], "nsaid", prn=True),
    _d("naproxen", "7258", "naproxen", ["naproxen", "Naprosyn", "Aleve"], [250, 500], "mg", ["PO"], ["BID"], "nsaid"),
    _d("acetaminophen", "161", "acetaminophen", ["acetaminophen", "Tylenol", "APAP"], [500, 650, 1000], "mg", ["PO"], ["Q6H"], "analgesic", prn=True),
    _d("tramadol", "10689", "tramadol", ["tramadol", "Ultram"], [50], "mg", ["PO"], ["Q6H"], "opioid", prn=True),
    _d("oxycodone", "7804", "oxycodone", ["oxycodone", "Roxicodone"], [5, 10], "mg", ["PO"], ["Q4H"], "opioid", high_alert=True, prn=True),
    _d("morphine", "7052", "morphine", ["morphine", "MSO4", "MS Contin"], [2, 4], "mg", ["IV"], ["Q4H"], "opioid", high_alert=True, prn=True),
    _d("gabapentin", "25480", "gabapentin", ["gabapentin", "Neurontin"], [100, 300, 600], "mg", ["PO"], ["TID"], "gabapentinoid"),
    _d("sertraline", "36437", "sertraline", ["sertraline", "Zoloft"], [50, 100], "mg", ["PO"], ["daily"], "ssri"),
    _d("citalopram", "2556", "citalopram", ["citalopram", "Celexa"], [10, 20], "mg", ["PO"], ["daily"], "ssri"),
    _d("fluoxetine", "4493", "fluoxetine", ["fluoxetine", "Prozac"], [20, 40], "mg", ["PO"], ["daily"], "ssri"),
    _d("haloperidol", "5093", "haloperidol", ["haloperidol", "Haldol"], [0.5, 1, 2], "mg", ["PO", "IV"], ["Q6H"], "antipsychotic", prn=True),
    _d("ondansetron", "26225", "ondansetron", ["ondansetron", "Zofran"], [4, 8], "mg", ["PO", "IV"], ["Q8H"], "antiemetic", prn=True),
    _d("amoxicillin", "723", "amoxicillin", ["amoxicillin", "Amoxil"], [500, 875], "mg", ["PO"], ["TID", "BID"], "penicillin",
       site={"B": ["amox"]}),
    _d("ampicillin", "733", "ampicillin", ["ampicillin"], [1000, 2000], "mg", ["IV"], ["Q6H"], "penicillin"),
    _d("ceftriaxone", "2193", "ceftriaxone", ["ceftriaxone", "Rocephin"], [1000, 2000], "mg", ["IV"], ["daily"], "cephalosporin",
       site={"B": ["CTX"]}),
    _d("azithromycin", "18631", "azithromycin", ["azithromycin", "Zithromax", "Z-pak"], [250, 500], "mg", ["PO"], ["daily"], "macrolide",
       site={"B": ["azithro"]}),
    _d("clarithromycin", "21212", "clarithromycin", ["clarithromycin", "Biaxin"], [500], "mg", ["PO"], ["BID"], "macrolide"),
    _d("ciprofloxacin", "2551", "ciprofloxacin", ["ciprofloxacin", "Cipro"], [250, 500], "mg", ["PO"], ["BID"], "fluoroquinolone"),
    _d("levofloxacin", "82122", "levofloxacin", ["levofloxacin", "Levaquin"], [500, 750], "mg", ["PO", "IV"], ["daily"], "fluoroquinolone",
       site={"B": ["levo"]}),
    _d("tmp_smx", "10180", "sulfamethoxazole / trimethoprim", ["sulfamethoxazole / trimethoprim", "trimethoprim-sulfamethoxazole",
       "TMP-SMX", "Bactrim"], [800], "mg", ["PO"], ["BID"], "sulfonamide_abx", site={"C": ["Septra"]}),
    _d("vancomycin", "11124", "vancomycin", ["vancomycin"], [1000, 1250], "mg", ["IV"], ["Q12H"], "glycopeptide",
       site={"B": ["vanc"]}),
    _d("fluconazole", "4450", "fluconazole", ["fluconazole", "Diflucan"], [150, 200], "mg", ["PO"], ["daily"], "azole"),
    _d("linezolid", "190376", "linezolid", ["linezolid", "Zyvox"], [600], "mg", ["PO", "IV"], ["BID"], "oxazolidinone"),
    _d("methotrexate", "6851", "methotrexate", ["methotrexate", "MTX"], [7.5, 15], "mg", ["PO"], ["weekly"], "antimetabolite",
       high_alert=True),
    _d("allopurinol", "519", "allopurinol", ["allopurinol", "Zyloprim"], [100, 300], "mg", ["PO"], ["daily"], "xo_inhibitor"),
    _d("tamsulosin", "77492", "tamsulosin", ["tamsulosin", "Flomax"], [0.4], "mg", ["PO"], ["daily"], "alpha_blocker"),
    _d("alendronate", "46041", "alendronate", ["alendronate", "Fosamax"], [70], "mg", ["PO"], ["weekly"], "bisphosphonate"),
]

# Ambiguous surface forms: the same string can denote different concepts.
# The generator uses them deliberately; systems must resolve them from context.
AMBIGUOUS = {
    "MS": [("condition", "ms"), ("drug", "morphine")],
    "PE": [("condition", "pe"), ("section", "exam")],
    "RA": [("condition", "rheumatoid"), ("other", "room air")],
}


# ---------------------------------------------------------------------------
# Allergens (SNOMED CT substance concepts) and reactions
# ---------------------------------------------------------------------------
ALLERGENS = [
    dict(key="penicillin", snomed="764146007", display="Penicillin", base=["penicillin", "PCN", "penicillins"],
         site={"B": ["PCNs"], "C": ["pen"]}, drug_classes=["penicillin"], drugs=[]),
    dict(key="sulfonamide", snomed="387406002", display="Sulfonamide", base=["sulfonamide", "sulfa", "sulfa drugs"],
         site={}, drug_classes=["sulfonamide_abx"], drugs=[]),
    dict(key="codeine", snomed="387494007", display="Codeine", base=["codeine"], site={}, drug_classes=[], drugs=[]),
    dict(key="morphine_allergen", snomed="373529000", display="Morphine", base=["morphine"], site={},
         drug_classes=[], drugs=["morphine"]),
    dict(key="aspirin_allergen", snomed="387458008", display="Aspirin", base=["aspirin", "ASA"], site={},
         drug_classes=["nsaid", "nsaid_asa"], drugs=["aspirin"]),
    dict(key="ibuprofen_allergen", snomed="387207008", display="Ibuprofen", base=["ibuprofen", "NSAIDs", "Motrin"],
         site={}, drug_classes=["nsaid", "nsaid_asa"], drugs=["ibuprofen"]),
    dict(key="latex", snomed="111088007", display="Latex", base=["latex"], site={}, drug_classes=[], drugs=[]),
    dict(key="peanut", snomed="762952008", display="Peanut", base=["peanut", "peanuts"], site={}, drug_classes=[], drugs=[]),
]

REACTIONS = [
    dict(key="hives", snomed="126485001", display="Urticaria", forms=["hives", "urticaria"]),
    dict(key="rash", snomed="271807003", display="Eruption of skin", forms=["rash", "skin rash"]),
    dict(key="anaphylaxis", snomed="39579001", display="Anaphylaxis", forms=["anaphylaxis", "anaphylactic reaction"]),
    dict(key="angioedema", snomed="41291007", display="Angioedema", forms=["angioedema", "lip swelling"]),
    dict(key="nausea", snomed="422587007", display="Nausea", forms=["nausea", "GI upset"]),
    dict(key="wheeze", snomed="56018004", display="Wheezing", forms=["wheezing"]),
]


# ---------------------------------------------------------------------------
# Laboratory observations (LOINC). Values are sampled in the unit given.
# ---------------------------------------------------------------------------
LABS = [
    dict(key="potassium", loinc="2823-3", display="Potassium [Moles/volume] in Serum or Plasma",
         base=["potassium", "K"], site={"C": ["K+"]}, unit="mmol/L", normal=(3.5, 5.0), decimals=1),
    dict(key="creatinine", loinc="2160-0", display="Creatinine [Mass/volume] in Serum or Plasma",
         base=["creatinine", "Cr"], site={"B": ["Creat"]}, unit="mg/dL", normal=(0.6, 1.2), decimals=1),
    dict(key="egfr", loinc="33914-3", display="Glomerular filtration rate/1.73 sq M.predicted",
         base=["eGFR", "GFR"], site={}, unit="mL/min/{1.73_m2}", normal=(60, 110), decimals=0),
    dict(key="inr", loinc="6301-6", display="INR in Platelet poor plasma by Coagulation assay",
         base=["INR"], site={}, unit="{INR}", normal=(0.9, 1.2), decimals=1),
    dict(key="sodium", loinc="2951-2", display="Sodium [Moles/volume] in Serum or Plasma",
         base=["sodium", "Na"], site={}, unit="mmol/L", normal=(135, 145), decimals=0),
    dict(key="glucose", loinc="2345-7", display="Glucose [Mass/volume] in Serum or Plasma",
         base=["glucose", "Gluc"], site={"C": ["sugar level"]}, unit="mg/dL", normal=(70, 140), decimals=0),
    dict(key="hemoglobin", loinc="718-7", display="Hemoglobin [Mass/volume] in Blood",
         base=["hemoglobin", "Hgb"], site={"B": ["Hb"]}, unit="g/dL", normal=(12.0, 16.0), decimals=1),
    dict(key="a1c", loinc="4548-4", display="Hemoglobin A1c/Hemoglobin.total in Blood",
         base=["hemoglobin A1c", "A1c", "HbA1c"], site={}, unit="%", normal=(4.5, 5.6), decimals=1),
    dict(key="tsh", loinc="3016-3", display="Thyrotropin [Units/volume] in Serum or Plasma",
         base=["TSH"], site={}, unit="m[IU]/L", normal=(0.4, 4.0), decimals=2),
]

# Display units for rendering (UCUM code -> how clinicians write it).
UNIT_TEXT = {"mg": "mg", "ug": "mcg", "meq": "mEq", "[iU]": "units", "{puff}": "puffs",
             "mmol/L": "", "mg/dL": "", "mL/min/{1.73_m2}": "", "{INR}": "", "g/dL": "", "%": "%",
             "m[IU]/L": ""}
UCUM_DOSE_UNITS = {"mg", "ug", "meq", "[iU]", "{puff}", "g"}

ROUTES = {
    "PO": dict(snomed="26643006", display="Oral route", forms=["PO", "by mouth", "orally"]),
    "IV": dict(snomed="47625008", display="Intravenous route", forms=["IV", "intravenously"]),
    "SC": dict(snomed="34206005", display="Subcutaneous route", forms=["SC", "subcut", "SQ"]),
    "INH": dict(snomed="447694001", display="Respiratory tract route", forms=["inhaled", "INH"]),
}

# Frequency codes -> FHIR Timing.repeat (frequency, period, periodUnit) and text forms.
FREQS = {
    "daily": dict(repeat=(1, 1, "d"), forms=["daily", "QD", "once daily", "qday"]),
    "BID": dict(repeat=(2, 1, "d"), forms=["BID", "twice daily", "b.i.d."]),
    "TID": dict(repeat=(3, 1, "d"), forms=["TID", "three times daily", "t.i.d."]),
    "QHS": dict(repeat=(1, 1, "d"), forms=["QHS", "at bedtime", "nightly"]),
    "Q4H": dict(repeat=(1, 4, "h"), forms=["q4h", "every 4 hours"]),
    "Q6H": dict(repeat=(1, 6, "h"), forms=["q6h", "every 6 hours"]),
    "Q8H": dict(repeat=(1, 8, "h"), forms=["q8h", "every 8 hours"]),
    "Q12H": dict(repeat=(1, 12, "h"), forms=["q12h", "every 12 hours"]),
    "weekly": dict(repeat=(1, 1, "wk"), forms=["weekly", "once a week", "qweek"]),
}

COND_BY_KEY = {c["key"]: c for c in CONDITIONS}
DRUG_BY_KEY = {d["key"]: d for d in DRUGS}
ALLERGEN_BY_KEY = {a["key"]: a for a in ALLERGENS}
LAB_BY_KEY = {l["key"]: l for l in LABS}
REACTION_BY_KEY = {r["key"]: r for r in REACTIONS}
COND_BY_CODE = {c["snomed"]: c for c in CONDITIONS}
DRUG_BY_CODE = {d["rxcui"]: d for d in DRUGS}
ALLERGEN_BY_CODE = {a["snomed"]: a for a in ALLERGENS}
LAB_BY_CODE = {l["loinc"]: l for l in LABS}
HIGH_ALERT = {d["key"] for d in DRUGS if d["high_alert"]}


def base_lexicon(include_abbrev: bool = True):
    """Return list of (surface, kind, key) from the base tier.

    ``include_abbrev=False`` keeps only canonical names and multi-word synonyms;
    the rule-based converter uses this restricted dictionary.
    """
    out = []

    def short(s):
        return s.isupper() or len(s) <= 4 or any(ch in s for ch in "-/") and len(s) <= 7

    for c in CONDITIONS:
        for s in c["base"]:
            if include_abbrev or not short(s):
                out.append((s, "condition", c["key"]))
    for d in DRUGS:
        for s in d["base"]:
            if include_abbrev or not short(s):
                out.append((s, "drug", d["key"]))
    for a in ALLERGENS:
        for s in a["base"]:
            out.append((s, "allergen", a["key"]))
    for lab in LABS:
        for s in lab["base"]:
            out.append((s, "lab", lab["key"]))
    return out


def terminology_table_for_prompt() -> str:
    """Compact code list given to LLM systems (canonical names only)."""
    lines = ["CONDITIONS (SNOMED CT):"]
    lines += [f"  {c['snomed']} | {c['display']}" for c in CONDITIONS]
    lines.append("MEDICATIONS (RxNorm ingredient):")
    lines += [f"  {d['rxcui']} | {d['display']}" for d in DRUGS]
    lines.append("ALLERGENS (SNOMED CT substance):")
    lines += [f"  {a['snomed']} | {a['display']}" for a in ALLERGENS]
    lines.append("LABS (LOINC, UCUM unit):")
    lines += [f"  {l['loinc']} | {l['display']} | {l['unit']}" for l in LABS]
    return "\n".join(lines)
