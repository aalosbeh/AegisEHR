"""FHIR R4 resource builders and a strict structural validator.

The validator implements the subset of HL7 FHIR R4 rules that matter for the
resources produced by reconciliation (Patient, Encounter, Condition,
MedicationRequest, AllergyIntolerance, Observation, DetectedIssue,
Provenance): element whitelists for every resource and datatype, required
elements, required value-set bindings, the R4 ``dateTime`` grammar with
calendar checks, choice-type exclusivity, ele-1 (no empty elements),
reference resolution inside the bundle, UCUM units, and a profile rule that
clinical codes must come from the benchmark value sets.

It is not a replacement for the official HL7 validator; it is a transparent,
dependency-free rule set whose every check is listed in ``CHECKS`` so that
compliance can be reported per rule (pre-gate and post-gate).
"""
from __future__ import annotations

import calendar
import re
from typing import Dict, List, Optional, Tuple

from .terminology import (ALLERGEN_BY_CODE, COND_BY_CODE, DRUG_BY_CODE, LAB_BY_CODE, LOINC,
                          RXNORM, SNOMED, UCUM, ICD10)

BASE = "http://aegis.example/fhir"
CS_CLINICAL = "http://terminology.hl7.org/CodeSystem/condition-clinical"
CS_VERIF = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
CS_AI_CLINICAL = "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical"
CS_AI_VERIF = "http://terminology.hl7.org/CodeSystem/allergyintolerance-verification"
CS_ACT = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
CS_MR_CAT = "http://terminology.hl7.org/CodeSystem/medicationrequest-category"
CS_OBS_CAT = "http://terminology.hl7.org/CodeSystem/observation-category"
CS_DI = "http://terminology.hl7.org/CodeSystem/v3-ActCode"

# ---------------------------------------------------------------------------
# Builders (used for gold bundles and by the deterministic agents)
# ---------------------------------------------------------------------------

def ref(rt: str, rid: str) -> Dict:
    return {"reference": f"{rt}/{rid}"}


def cc(system: str, code: str, display: str) -> Dict:
    return {"coding": [{"system": system, "code": code, "display": display}]}


def patient(rid: str, gender: str) -> Dict:
    return {"resourceType": "Patient", "id": rid, "gender": gender}


def encounter(rid: str, pid: str, status: str, cls: str, start: Optional[str], end: Optional[str] = None) -> Dict:
    r = {"resourceType": "Encounter", "id": rid, "status": status,
         "class": {"system": CS_ACT, "code": cls}, "subject": ref("Patient", pid)}
    period = {}
    if start:
        period["start"] = start
    if end:
        period["end"] = end
    if period:
        r["period"] = period
    return r


def condition(rid: str, pid: str, snomed: str, display: str, clinical: str = "active",
              onset: Optional[str] = None, onset_text: Optional[str] = None,
              recorded: Optional[str] = None, icd10: Optional[str] = None) -> Dict:
    code = {"coding": [{"system": SNOMED, "code": snomed, "display": display}]}
    if icd10:
        code["coding"].append({"system": ICD10, "code": icd10})
    r = {"resourceType": "Condition", "id": rid,
         "clinicalStatus": cc(CS_CLINICAL, clinical, clinical.capitalize()),
         "verificationStatus": cc(CS_VERIF, "confirmed", "Confirmed"),
         "code": code, "subject": ref("Patient", pid)}
    if onset:
        r["onsetDateTime"] = onset
    elif onset_text:
        r["onsetString"] = onset_text
    if recorded:
        r["recordedDate"] = recorded
    return r


def medication_request(rid: str, pid: str, rxcui: str, display: str, status: str, category: str,
                       dose: Optional[float], unit: Optional[str], route: Optional[str],
                       freq: Optional[Tuple[int, float, str]], prn: bool = False,
                       authored: Optional[str] = None, route_display: Optional[str] = None,
                       freq_text: Optional[str] = None) -> Dict:
    r = {"resourceType": "MedicationRequest", "id": rid, "status": status, "intent": "order",
         "category": [cc(CS_MR_CAT, category, category.capitalize())],
         "medicationCodeableConcept": cc(RXNORM, rxcui, display),
         "subject": ref("Patient", pid)}
    if authored:
        r["authoredOn"] = authored
    dosage: Dict = {}
    if freq_text:
        dosage["text"] = freq_text
    if freq:
        f, p, u = freq
        dosage["timing"] = {"repeat": {"frequency": int(f), "period": p, "periodUnit": u}}
    if prn:
        dosage["asNeededBoolean"] = True
    if route:
        dosage["route"] = {"coding": [{"system": SNOMED, "code": route, "display": route_display or route}]}
    if dose is not None and unit:
        dosage["doseAndRate"] = [{"doseQuantity": {"value": dose, "unit": unit, "system": UCUM, "code": unit}}]
    if dosage:
        r["dosageInstruction"] = [dosage]
    return r


def allergy(rid: str, pid: str, snomed: str, display: str, reactions=None, severity: Optional[str] = None,
            onset: Optional[str] = None, verification: str = "confirmed") -> Dict:
    r = {"resourceType": "AllergyIntolerance", "id": rid,
         "clinicalStatus": cc(CS_AI_CLINICAL, "active", "Active"),
         "verificationStatus": cc(CS_AI_VERIF, verification, verification.capitalize()),
         "code": cc(SNOMED, snomed, display), "patient": ref("Patient", pid)}
    if onset:
        r["onsetDateTime"] = onset
    if reactions:
        rx = {"manifestation": [cc(SNOMED, c, d) for c, d in reactions]}
        if severity:
            rx["severity"] = severity
        r["reaction"] = [rx]
    return r


def observation(rid: str, pid: str, loinc: str, display: str, value: float, unit: str,
                effective: Optional[str]) -> Dict:
    r = {"resourceType": "Observation", "id": rid, "status": "final",
         "category": [cc(CS_OBS_CAT, "laboratory", "Laboratory")],
         "code": cc(LOINC, loinc, display), "subject": ref("Patient", pid),
         "valueQuantity": {"value": value, "unit": unit, "system": UCUM, "code": unit}}
    if effective:
        r["effectiveDateTime"] = effective
    return r


def detected_issue(rid: str, pid: str, code: str, severity: str, detail: str, implicated: List[str]) -> Dict:
    return {"resourceType": "DetectedIssue", "id": rid, "status": "preliminary",
            "code": cc(CS_DI, code, {"DRG": "Drug interaction", "ALGY": "Allergy alert",
                                     "DUPTHPY": "Duplicate therapy", "TIME": "Timing detected issue"}.get(code, code)),
            "severity": severity, "patient": ref("Patient", pid), "detail": detail,
            "implicated": [{"reference": x} for x in implicated]}


def bundle(resources: List[Dict]) -> Dict:
    return {"resourceType": "Bundle", "type": "collection",
            "entry": [{"fullUrl": f"{BASE}/{r['resourceType']}/{r['id']}", "resource": r} for r in resources]}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
_DT_RE = re.compile(
    r"^([0-9]([0-9]([0-9][1-9]|[1-9]0)|[1-9]00)|[1-9]000)(-(0[1-9]|1[0-2])(-(0[1-9]|[1-2][0-9]|3[0-1])"
    r"(T([01][0-9]|2[0-3]):[0-5][0-9]:([0-5][0-9]|60)(\.[0-9]+)?(Z|(\+|-)((0[0-9]|1[0-3]):[0-5][0-9]|14:00)))?)?)?$")
_ID_RE = re.compile(r"^[A-Za-z0-9\-\.]{1,64}$")

DOMAIN = {
    "common": {"resourceType", "id", "meta", "implicitRules", "language", "text", "contained", "extension",
               "modifierExtension"},
    "Patient": {"identifier", "active", "name", "telecom", "gender", "birthDate", "deceasedBoolean",
                "deceasedDateTime", "address", "maritalStatus", "multipleBirthBoolean", "multipleBirthInteger",
                "photo", "contact", "communication", "generalPractitioner", "managingOrganization", "link"},
    "Encounter": {"identifier", "status", "statusHistory", "class", "classHistory", "type", "serviceType",
                  "priority", "subject", "episodeOfCare", "basedOn", "participant", "appointment", "period",
                  "length", "reasonCode", "reasonReference", "diagnosis", "account", "hospitalization",
                  "location", "serviceProvider", "partOf"},
    "Condition": {"identifier", "clinicalStatus", "verificationStatus", "category", "severity", "code",
                  "bodySite", "subject", "encounter", "onsetDateTime", "onsetAge", "onsetPeriod", "onsetRange",
                  "onsetString", "abatementDateTime", "abatementAge", "abatementPeriod", "abatementRange",
                  "abatementString", "recordedDate", "recorder", "asserter", "stage", "evidence", "note"},
    "MedicationRequest": {"identifier", "status", "statusReason", "intent", "category", "priority",
                          "doNotPerform", "reportedBoolean", "reportedReference", "medicationCodeableConcept",
                          "medicationReference", "subject", "encounter", "supportingInformation", "authoredOn",
                          "requester", "performer", "performerType", "recorder", "reasonCode", "reasonReference",
                          "instantiatesCanonical", "instantiatesUri", "basedOn", "groupIdentifier",
                          "courseOfTherapyType", "insurance", "note", "dosageInstruction", "dispenseRequest",
                          "substitution", "priorPrescription", "detectedIssue", "eventHistory"},
    "AllergyIntolerance": {"identifier", "clinicalStatus", "verificationStatus", "type", "category",
                           "criticality", "code", "patient", "encounter", "onsetDateTime", "onsetAge",
                           "onsetPeriod", "onsetRange", "onsetString", "recordedDate", "recorder", "asserter",
                           "lastOccurrence", "note", "reaction"},
    "Observation": {"identifier", "basedOn", "partOf", "status", "category", "code", "subject", "focus",
                    "encounter", "effectiveDateTime", "effectivePeriod", "effectiveTiming", "effectiveInstant",
                    "issued", "performer", "valueQuantity", "valueCodeableConcept", "valueString",
                    "valueBoolean", "valueInteger", "valueRange", "valueRatio", "valueSampledData", "valueTime",
                    "valueDateTime", "valuePeriod", "dataAbsentReason", "interpretation", "note", "bodySite",
                    "method", "specimen", "device", "referenceRange", "hasMember", "derivedFrom", "component"},
    "DetectedIssue": {"identifier", "status", "code", "severity", "patient", "identifiedDateTime",
                      "identifiedPeriod", "author", "implicated", "evidence", "detail", "reference", "mitigation"},
    "Provenance": {"target", "occurredPeriod", "occurredDateTime", "recorded", "policy", "location", "reason",
                   "activity", "agent", "entity", "signature"},
}
DT_KEYS = {
    "CodeableConcept": {"coding", "text", "id", "extension"},
    "Coding": {"system", "version", "code", "display", "userSelected", "id", "extension"},
    "Reference": {"reference", "type", "identifier", "display", "id", "extension"},
    "Quantity": {"value", "comparator", "unit", "system", "code", "id", "extension"},
    "Period": {"start", "end", "id", "extension"},
    "Dosage": {"sequence", "text", "additionalInstruction", "patientInstruction", "timing", "asNeededBoolean",
               "asNeededCodeableConcept", "site", "route", "method", "doseAndRate", "maxDosePerPeriod",
               "maxDosePerAdministration", "maxDosePerLifetime", "id", "extension", "modifierExtension"},
    "Timing": {"event", "repeat", "code", "id", "extension", "modifierExtension"},
    "Repeat": {"boundsDuration", "boundsRange", "boundsPeriod", "count", "countMax", "duration", "durationMax",
               "durationUnit", "frequency", "frequencyMax", "period", "periodMax", "periodUnit", "dayOfWeek",
               "timeOfDay", "when", "offset", "id", "extension"},
    "DoseAndRate": {"type", "doseRange", "doseQuantity", "rateRatio", "rateRange", "rateQuantity", "id", "extension"},
    "Reaction": {"substance", "manifestation", "description", "onset", "severity", "exposureRoute", "note", "id",
                 "extension", "modifierExtension"},
    "Entry": {"fullUrl", "resource", "link", "search", "request", "response", "id", "extension", "modifierExtension"},
    "Bundle": {"resourceType", "id", "meta", "implicitRules", "language", "identifier", "type", "timestamp",
               "total", "link", "entry", "signature"},
}
CHOICES = {
    "Condition": [("onset", ["onsetDateTime", "onsetAge", "onsetPeriod", "onsetRange", "onsetString"]),
                  ("abatement", ["abatementDateTime", "abatementAge", "abatementPeriod", "abatementRange", "abatementString"])],
    "AllergyIntolerance": [("onset", ["onsetDateTime", "onsetAge", "onsetPeriod", "onsetRange", "onsetString"])],
    "MedicationRequest": [("medication", ["medicationCodeableConcept", "medicationReference"])],
    "Observation": [("effective", ["effectiveDateTime", "effectivePeriod", "effectiveTiming", "effectiveInstant"]),
                    ("value", [k for k in DOMAIN["Observation"] if k.startswith("value")])],
}
VS = {
    "encounter_status": {"planned", "arrived", "triaged", "in-progress", "onleave", "finished", "cancelled",
                         "entered-in-error", "unknown"},
    "condition_clinical": {"active", "recurrence", "relapse", "inactive", "remission", "resolved"},
    "condition_verif": {"unconfirmed", "provisional", "differential", "confirmed", "refuted", "entered-in-error"},
    "mr_status": {"active", "on-hold", "cancelled", "completed", "entered-in-error", "stopped", "draft", "unknown"},
    "mr_intent": {"proposal", "plan", "order", "original-order", "reflex-order", "filler-order", "instance-order", "option"},
    "mr_category": {"inpatient", "outpatient", "community", "discharge"},
    "ai_clinical": {"active", "inactive", "resolved"},
    "ai_verif": {"unconfirmed", "confirmed", "refuted", "entered-in-error"},
    "severity": {"mild", "moderate", "severe"},
    "obs_status": {"registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"},
    "gender": {"male", "female", "other", "unknown"},
    "period_unit": {"s", "min", "h", "d", "wk", "mo", "a"},
    "di_status": {"registered", "preliminary", "final", "amended", "corrected", "cancelled", "entered-in-error", "unknown"},
    "di_severity": {"high", "moderate", "low"},
    "bundle_type": {"document", "message", "transaction", "transaction-response", "batch", "batch-response",
                    "history", "searchset", "collection"},
}
UCUM_OK = {"mg", "ug", "g", "meq", "[iU]", "{puff}", "mmol/L", "mg/dL", "mL/min/{1.73_m2}", "{INR}", "g/dL", "%",
           "m[IU]/L", "mL", "mg/kg", "U"}

CHECKS = ["bundle_structure", "unknown_element", "missing_required", "value_set_binding", "invalid_datetime",
          "terminology_valueset", "dangling_reference", "choice_conflict", "empty_element", "datatype",
          "duplicate_id", "ucum_unit"]


def valid_datetime(s) -> bool:
    if not isinstance(s, str) or not _DT_RE.match(s):
        return False
    parts = s[:10].split("-")
    if len(parts) == 3:
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if d > calendar.monthrange(y, m)[1]:
            return False
    return True


class _V:
    def __init__(self):
        self.errors: List[Tuple[str, str]] = []

    def err(self, kind: str, msg: str):
        self.errors.append((kind, msg))


def _empty(x) -> bool:
    return x is None or x == "" or x == [] or x == {}


def _check_keys(v: _V, obj, allowed, where):
    if not isinstance(obj, dict):
        v.err("datatype", f"{where}: expected object")
        return False
    for k, val in obj.items():
        if k not in allowed:
            v.err("unknown_element", f"{where}.{k}")
        if _empty(val):
            v.err("empty_element", f"{where}.{k}")
    return True


def _coding(v, c, where):
    if _check_keys(v, c, DT_KEYS["Coding"], where):
        if "system" in c and not isinstance(c["system"], str):
            v.err("datatype", f"{where}.system")
        if "code" in c and not isinstance(c["code"], str):
            v.err("datatype", f"{where}.code")


def _cc(v, x, where, required_system=None, valueset=None, vs_kind="value_set_binding"):
    """Validate a CodeableConcept. Returns the first matching code or None."""
    if not _check_keys(v, x, DT_KEYS["CodeableConcept"], where):
        return None
    codings = x.get("coding", [])
    if not isinstance(codings, list):
        v.err("datatype", f"{where}.coding")
        return None
    found = None
    for i, c in enumerate(codings):
        _coding(v, c, f"{where}.coding[{i}]")
        if isinstance(c, dict) and (required_system is None or c.get("system") == required_system):
            if found is None and isinstance(c.get("code"), str):
                found = c.get("code")
    if valueset is not None:
        if found is None:
            v.err("missing_required" if required_system else vs_kind, f"{where}: no coding from {required_system}")
        elif found not in valueset:
            v.err(vs_kind, f"{where}: code {found!r} not in value set")
    return found


def _reference(v, r, where, targets, ids):
    if not _check_keys(v, r, DT_KEYS["Reference"], where):
        return
    s = r.get("reference")
    if not isinstance(s, str):
        v.err("missing_required", f"{where}.reference")
        return
    if s not in ids:
        v.err("dangling_reference", f"{where} -> {s}")
    elif targets and s.split("/")[0] not in targets:
        v.err("dangling_reference", f"{where} wrong target type {s}")


def _quantity(v, q, where, need_ucum=True):
    if not _check_keys(v, q, DT_KEYS["Quantity"], where):
        return
    if "value" not in q:
        v.err("missing_required", f"{where}.value")
    elif isinstance(q["value"], bool) or not isinstance(q["value"], (int, float)):
        v.err("datatype", f"{where}.value must be a number")
    if "code" in q:
        if q.get("system") != UCUM:
            v.err("ucum_unit", f"{where}: coded unit without UCUM system")
        elif not isinstance(q["code"], str) or q["code"] not in UCUM_OK:
            v.err("ucum_unit", f"{where}: unit {q['code']!r} is not a valid UCUM code here")
    elif need_ucum:
        v.err("ucum_unit", f"{where}: missing UCUM code")


def _dt(v, val, where):
    if not valid_datetime(val):
        v.err("invalid_datetime", f"{where}={val!r}")


def _in_vs(val, vs):
    # FHIR codes are strings; any other JSON type (list, object, number) fails the binding instead of raising
    return isinstance(val, str) and val in VS[vs]


def _status(v, r, key, vs, where):
    if key not in r:
        v.err("missing_required", f"{where}.{key}")
    elif not _in_vs(r[key], vs):
        v.err("value_set_binding", f"{where}.{key}={r[key]!r}")


def _resource(v: _V, r: Dict, ids: set, where: str):
    rt = r.get("resourceType")
    allowed = DOMAIN["common"] | DOMAIN.get(rt, set())
    _check_keys(v, r, allowed, where)
    rid = r.get("id")
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        v.err("datatype", f"{where}.id={rid!r}")
    for _, opts in CHOICES.get(rt, []):
        if sum(1 for o in opts if o in r) > 1:
            v.err("choice_conflict", f"{where}: more than one of {opts}")

    if rt == "Patient":
        if "gender" in r and not _in_vs(r["gender"], "gender"):
            v.err("value_set_binding", f"{where}.gender")
        if "birthDate" in r and (not valid_datetime(r["birthDate"]) or "T" in str(r["birthDate"])):
            v.err("invalid_datetime", f"{where}.birthDate")
    elif rt == "Encounter":
        _status(v, r, "status", "encounter_status", where)
        if "class" not in r:
            v.err("missing_required", f"{where}.class")
        else:
            _check_keys(v, r["class"], DT_KEYS["Coding"], f"{where}.class")
        if "subject" in r:
            _reference(v, r["subject"], f"{where}.subject", {"Patient"}, ids)
        if "period" in r and _check_keys(v, r["period"], DT_KEYS["Period"], f"{where}.period"):
            for k in ("start", "end"):
                if k in r["period"]:
                    _dt(v, r["period"][k], f"{where}.period.{k}")
    elif rt == "Condition":
        if "subject" not in r:
            v.err("missing_required", f"{where}.subject")
        else:
            _reference(v, r["subject"], f"{where}.subject", {"Patient"}, ids)
        if "code" not in r:
            v.err("missing_required", f"{where}.code")
        else:
            _cc(v, r["code"], f"{where}.code", SNOMED, COND_BY_CODE, "terminology_valueset")
        if "clinicalStatus" in r:
            _cc(v, r["clinicalStatus"], f"{where}.clinicalStatus", CS_CLINICAL, VS["condition_clinical"])
        if "verificationStatus" in r:
            _cc(v, r["verificationStatus"], f"{where}.verificationStatus", CS_VERIF, VS["condition_verif"])
        for k in ("onsetDateTime", "recordedDate", "abatementDateTime"):
            if k in r:
                _dt(v, r[k], f"{where}.{k}")
        if "onsetString" in r and not isinstance(r["onsetString"], str):
            v.err("datatype", f"{where}.onsetString")
    elif rt == "MedicationRequest":
        _status(v, r, "status", "mr_status", where)
        _status(v, r, "intent", "mr_intent", where)
        if "medicationCodeableConcept" in r:
            _cc(v, r["medicationCodeableConcept"], f"{where}.medicationCodeableConcept", RXNORM, DRUG_BY_CODE,
                "terminology_valueset")
        elif "medicationReference" in r:
            _reference(v, r["medicationReference"], f"{where}.medicationReference", {"Medication"}, ids)
        else:
            v.err("missing_required", f"{where}.medication[x]")
        if "subject" not in r:
            v.err("missing_required", f"{where}.subject")
        else:
            _reference(v, r["subject"], f"{where}.subject", {"Patient"}, ids)
        if "authoredOn" in r:
            _dt(v, r["authoredOn"], f"{where}.authoredOn")
        if "category" in r:
            if not isinstance(r["category"], list):
                v.err("datatype", f"{where}.category must be an array")
            else:
                for i, c in enumerate(r["category"]):
                    _cc(v, c, f"{where}.category[{i}]", CS_MR_CAT, VS["mr_category"])
        if "dosageInstruction" in r:
            di = r["dosageInstruction"]
            if not isinstance(di, list):
                v.err("datatype", f"{where}.dosageInstruction must be an array")
                di = []
            for i, d in enumerate(di):
                w = f"{where}.dosageInstruction[{i}]"
                if not _check_keys(v, d, DT_KEYS["Dosage"], w):
                    continue
                if "asNeededBoolean" in d and not isinstance(d["asNeededBoolean"], bool):
                    v.err("datatype", f"{w}.asNeededBoolean")
                if "route" in d:
                    _cc(v, d["route"], f"{w}.route")
                if "timing" in d and _check_keys(v, d["timing"], DT_KEYS["Timing"], f"{w}.timing"):
                    rep = d["timing"].get("repeat")
                    if rep is not None and _check_keys(v, rep, DT_KEYS["Repeat"], f"{w}.timing.repeat"):
                        if "frequency" in rep and (isinstance(rep["frequency"], bool) or not isinstance(rep["frequency"], int) or rep["frequency"] < 1):
                            v.err("datatype", f"{w}.timing.repeat.frequency")
                        if "period" in rep and (isinstance(rep["period"], bool) or not isinstance(rep["period"], (int, float)) or rep["period"] <= 0):
                            v.err("datatype", f"{w}.timing.repeat.period")
                        if "periodUnit" in rep and not _in_vs(rep["periodUnit"], "period_unit"):
                            v.err("value_set_binding", f"{w}.timing.repeat.periodUnit={rep['periodUnit']!r}")
                        if ("period" in rep) != ("periodUnit" in rep):
                            v.err("missing_required", f"{w}.timing.repeat period/periodUnit (tim-1)")
                    if "code" in d["timing"]:
                        _cc(v, d["timing"]["code"], f"{w}.timing.code")
                if "doseAndRate" in d:
                    dr = d["doseAndRate"]
                    if not isinstance(dr, list):
                        v.err("datatype", f"{w}.doseAndRate must be an array")
                        dr = []
                    for j, x in enumerate(dr):
                        if _check_keys(v, x, DT_KEYS["DoseAndRate"], f"{w}.doseAndRate[{j}]") and "doseQuantity" in x:
                            _quantity(v, x["doseQuantity"], f"{w}.doseAndRate[{j}].doseQuantity")
    elif rt == "AllergyIntolerance":
        if "patient" not in r:
            v.err("missing_required", f"{where}.patient")
        else:
            _reference(v, r["patient"], f"{where}.patient", {"Patient"}, ids)
        if "code" not in r:
            v.err("missing_required", f"{where}.code")
        else:
            _cc(v, r["code"], f"{where}.code", SNOMED, ALLERGEN_BY_CODE, "terminology_valueset")
        if "clinicalStatus" in r:
            _cc(v, r["clinicalStatus"], f"{where}.clinicalStatus", CS_AI_CLINICAL, VS["ai_clinical"])
        if "verificationStatus" in r:
            _cc(v, r["verificationStatus"], f"{where}.verificationStatus", CS_AI_VERIF, VS["ai_verif"])
        if "onsetDateTime" in r:
            _dt(v, r["onsetDateTime"], f"{where}.onsetDateTime")
        if "reaction" in r:
            rx = r["reaction"] if isinstance(r["reaction"], list) else []
            if not isinstance(r["reaction"], list):
                v.err("datatype", f"{where}.reaction must be an array")
            for i, x in enumerate(rx):
                w = f"{where}.reaction[{i}]"
                if not _check_keys(v, x, DT_KEYS["Reaction"], w):
                    continue
                man = x.get("manifestation")
                if not isinstance(man, list) or not man:
                    v.err("missing_required", f"{w}.manifestation")
                else:
                    for j, m in enumerate(man):
                        _cc(v, m, f"{w}.manifestation[{j}]")
                if "severity" in x and not _in_vs(x["severity"], "severity"):
                    v.err("value_set_binding", f"{w}.severity={x['severity']!r}")
    elif rt == "Observation":
        _status(v, r, "status", "obs_status", where)
        if "code" not in r:
            v.err("missing_required", f"{where}.code")
        else:
            _cc(v, r["code"], f"{where}.code", LOINC, LAB_BY_CODE, "terminology_valueset")
        if "subject" in r:
            _reference(v, r["subject"], f"{where}.subject", {"Patient"}, ids)
        if "effectiveDateTime" in r:
            _dt(v, r["effectiveDateTime"], f"{where}.effectiveDateTime")
        if "valueQuantity" in r:
            _quantity(v, r["valueQuantity"], f"{where}.valueQuantity")
    elif rt == "DetectedIssue":
        _status(v, r, "status", "di_status", where)
        if "severity" in r and not _in_vs(r["severity"], "di_severity"):
            v.err("value_set_binding", f"{where}.severity")
        if "patient" in r:
            _reference(v, r["patient"], f"{where}.patient", {"Patient"}, ids)
        for i, x in enumerate(r.get("implicated", []) if isinstance(r.get("implicated"), list) else []):
            _reference(v, x, f"{where}.implicated[{i}]", None, ids)
    elif rt == "Provenance":
        if "target" not in r or "recorded" not in r or "agent" not in r:
            v.err("missing_required", f"{where}: Provenance.target/recorded/agent")
        for i, x in enumerate(r.get("target", []) if isinstance(r.get("target"), list) else []):
            _reference(v, x, f"{where}.target[{i}]", None, ids)


SUPPORTED = set(DOMAIN) - {"common"}


def validate_bundle(b) -> Dict:
    """Validate a bundle. Returns {'valid': bool, 'errors': [(kind, msg)], 'by_kind': {...}}."""
    v = _V()
    if not isinstance(b, dict) or b.get("resourceType") != "Bundle":
        v.err("bundle_structure", "top-level object is not a Bundle")
        return _result(v, 0)
    _check_keys(v, b, DT_KEYS["Bundle"], "Bundle")
    if not _in_vs(b.get("type"), "bundle_type"):
        v.err("bundle_structure", f"Bundle.type={b.get('type')!r}")
    entries = b.get("entry")
    if not isinstance(entries, list) or not entries:
        v.err("bundle_structure", "Bundle.entry missing or empty")
        return _result(v, 0)
    ids, seen = set(), set()
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("resource"), dict):
            r = e["resource"]
            key = f"{r.get('resourceType')}/{r.get('id')}"
            if key in seen:
                v.err("duplicate_id", key)
            seen.add(key)
            ids.add(key)
    n_pat = 0
    for i, e in enumerate(entries):
        w = f"entry[{i}]"
        if not _check_keys(v, e, DT_KEYS["Entry"], w):
            continue
        r = e.get("resource")
        if not isinstance(r, dict):
            v.err("bundle_structure", f"{w}.resource missing")
            continue
        rt = r.get("resourceType")
        if not isinstance(rt, str) or rt not in SUPPORTED:
            v.err("bundle_structure", f"{w}: unsupported resourceType {rt!r}")
            continue
        n_pat += rt == "Patient"
        fu = e.get("fullUrl")
        if not isinstance(fu, str) or not (fu.startswith("http://") or fu.startswith("https://") or fu.startswith("urn:uuid:")):
            v.err("bundle_structure", f"{w}.fullUrl not absolute: {fu!r}")
        elif fu.startswith("http") and not fu.endswith(f"/{rt}/{r.get('id')}"):
            v.err("bundle_structure", f"{w}.fullUrl disagrees with resource id")
        _resource(v, r, ids, f"{w}:{rt}")
    if n_pat != 1:
        v.err("bundle_structure", f"expected exactly one Patient, found {n_pat}")
    return _result(v, len(entries))


def _result(v: _V, n_entries: int) -> Dict:
    by_kind: Dict[str, int] = {}
    for k, _ in v.errors:
        by_kind[k] = by_kind.get(k, 0) + 1
    return {"valid": not v.errors, "errors": v.errors, "by_kind": by_kind, "n_entries": n_entries}


def resources_of(b, rt: Optional[str] = None) -> List[Dict]:
    if not isinstance(b, dict):
        return []
    out = []
    for e in b.get("entry", []) or []:
        if isinstance(e, dict) and isinstance(e.get("resource"), dict):
            r = e["resource"]
            if rt is None or r.get("resourceType") == rt:
                out.append(r)
    return out
