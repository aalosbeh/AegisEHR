"""LLM systems served locally through Ollama.

Three single-model baselines share one prompt:

* ``direct``    - one call, JSON mode (the monolithic LLM hypothesis);
* ``validator`` - ReAct-style tool use: the FHIR validator is called on the
  output and its error list is returned to the model for up to two retries;
* ``reflect``   - Reflexion-style: the model writes a critique of its own bundle
  against the note, then produces a revised bundle (one round).

The Aegis-EHR LLM variant uses the ``direct`` output as the Schema Agent's draft
and adds the verification agents; it makes no additional LLM calls.

All calls use temperature 0, a fixed seed, and record the model digest, the
exact prompt, token counts, latency and per-token log-probabilities.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import requests

from .fhir import validate_bundle
from .terminology import terminology_table_for_prompt

OLLAMA = "http://localhost:11434"

SYSTEM_PROMPT = """You are a clinical data engineer. Convert one clinical note into a single FHIR R4 Bundle (JSON).

Rules
1. Output only one JSON object: {"resourceType":"Bundle","type":"collection","entry":[...]}. Each entry is {"fullUrl":"http://aegis.example/fhir/<Type>/<id>","resource":{...}}.
2. Use exactly one Patient (id "p1") and one Encounter (id "e1", class code "IMP", status "finished" for discharge summaries, otherwise "in-progress"). Reference the patient as {"reference":"Patient/p1"}.
3. Allowed resource types: Patient, Encounter, Condition, MedicationRequest, AllergyIntolerance, Observation, DetectedIssue. Keep the output compact: omit optional elements not listed here; give a display string only for clinical codes, written before the code.
4. Codes must come from the terminology list below (SNOMED CT for conditions and allergens, RxNorm for medications, LOINC for labs, UCUM for units). Do not invent codes.
5. Include only facts asserted about this patient. Do not create resources for negated findings ("denies", "no history of"), for family history, or for allergies that the note says were removed.
6. Medications: one MedicationRequest per drug with the final reconciled state: status "active", "stopped" (discontinued) or "on-hold" (held), intent "order"; apply dose changes from the plan; authoredOn only when the note gives a start date. Put dose in dosageInstruction[0].doseAndRate[0].doseQuantity (value, unit, system "http://unitsofmeasure.org", code), frequency in dosageInstruction[0].timing.repeat (frequency, period, periodUnit), route in dosageInstruction[0].route, and asNeededBoolean when PRN.
7. Dates: ISO 8601 only, at the precision the note supports (YYYY, YYYY-MM or YYYY-MM-DD). Resolve relative expressions (e.g. "3 days prior to admission", "hospital day #2") from the admission date. If the timing is vague ("a few days ago", "long-standing", "as a child"), use onsetString with the original words. Never invent a date that the note does not support.
8. Conditions: clinicalStatus "resolved" for past events written as "history of", "h/o" or "past"; otherwise "active". recordedDate is the note date.
9. Labs: one Observation per reported value, status "final", valueQuantity with the UCUM unit from the list, effectiveDateTime is the date the value was obtained.
10. Safety: add a DetectedIssue (status "preliminary", severity "high" or "moderate", code system "http://terminology.hl7.org/CodeSystem/v3-ActCode", implicated = references to the involved resources) for each clinically significant drug-drug interaction, drug-allergy conflict, drug-condition contraindication or drug-lab problem among the active medications.

Terminology
{TERMINOLOGY}
"""

REFLECT_PROMPT = """Review the FHIR bundle you produced against the clinical note. List concrete problems only: facts that are missing, facts that are not supported by the note (negated, family history, removed allergies), wrong codes, wrong medication status or dose after the plan, dates that are wrong or not supported by the note, invalid FHIR structure, and missing or unjustified DetectedIssue resources. Be brief (at most 12 bullet points)."""

REVISE_PROMPT = """Now output the corrected FHIR R4 Bundle as a single JSON object, applying your review. Follow all the original rules."""


def system_prompt() -> str:
    return SYSTEM_PROMPT.replace("{TERMINOLOGY}", terminology_table_for_prompt())


def model_info(model: str) -> Dict:
    r = requests.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=60).json()
    tags = requests.get(f"{OLLAMA}/api/tags", timeout=60).json()
    digest = next((m["digest"] for m in tags["models"] if m["name"] == model), None)
    return dict(model=model, digest=digest, details=r.get("details", {}), modified_at=r.get("modified_at"))


def chat(model: str, messages: List[Dict], json_mode: bool = True, num_predict: int = 3000,
         num_ctx: int = 10240, seed: int = 42, timeout: int = 900) -> Dict:
    body = {"model": model, "messages": messages, "stream": False, "logprobs": True,
            "options": {"temperature": 0, "seed": seed, "num_ctx": num_ctx, "num_predict": num_predict, "top_p": 1.0},
            "keep_alive": "30m"}
    if json_mode:
        body["format"] = "json"
    t0 = time.time()
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    lp = [x["logprob"] for x in (d.get("logprobs") or [])]
    return dict(content=d["message"]["content"], logprobs=lp, prompt_tokens=d.get("prompt_eval_count", 0),
                output_tokens=d.get("eval_count", 0), wall=time.time() - t0, done_reason=d.get("done_reason"))


def parse_json(s: str) -> Optional[Dict]:
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None


def few_shot_messages(examples: List[Dict]) -> List[Dict]:
    msgs = []
    for ex in examples:
        msgs.append({"role": "user", "content": f"Clinical note:\n{ex['text']}"})
        msgs.append({"role": "assistant", "content": json.dumps(ex["target"], separators=(",", ":"))})
    return msgs


def _summ(calls: List[Dict]) -> Dict:
    return dict(n_calls=len(calls), prompt_tokens=sum(c["prompt_tokens"] for c in calls),
                output_tokens=sum(c["output_tokens"] for c in calls), wall=sum(c["wall"] for c in calls))


def run_direct(model: str, note: str, shots: List[Dict]) -> Dict:
    msgs = [{"role": "system", "content": system_prompt()}] + few_shot_messages(shots) + \
           [{"role": "user", "content": f"Clinical note:\n{note}"}]
    c = chat(model, msgs)
    return dict(bundle=parse_json(c["content"]), raw=c["content"], logprobs=c["logprobs"], done_reason=c["done_reason"],
                **_summ([c]))


def run_validator(model: str, note: str, shots: List[Dict], first: Dict, max_retries: int = 2) -> Dict:
    msgs = [{"role": "system", "content": system_prompt()}] + few_shot_messages(shots) + \
           [{"role": "user", "content": f"Clinical note:\n{note}"}]
    out, calls = first, []
    raw = first["raw"]
    for k in range(max_retries):
        b = parse_json(raw)
        v = validate_bundle(b) if b is not None else {"valid": False, "errors": [("bundle_structure", "output is not valid JSON")]}
        if v["valid"]:
            break
        err = "\n".join(f"- {kind}: {msg}" for kind, msg in v["errors"][:15])
        msgs = msgs + [{"role": "assistant", "content": raw},
                       {"role": "user", "content": "The FHIR validator rejected this bundle:\n" + err +
                        "\nReturn the complete corrected Bundle as one JSON object."}]
        c = chat(model, msgs)
        calls.append(c)
        raw = c["content"]
        out = dict(bundle=parse_json(raw), raw=raw, logprobs=c["logprobs"], done_reason=c["done_reason"])
    s = _summ(calls)
    s["n_calls"] += 1
    s["output_tokens"] += first["output_tokens"]
    s["prompt_tokens"] += first["prompt_tokens"]
    s["wall"] += first["wall"]
    return dict(bundle=parse_json(raw) if out is first else out["bundle"], raw=raw, logprobs=out["logprobs"],
                done_reason=out.get("done_reason"), retries=len(calls), **s)


def run_reflect(model: str, note: str, shots: List[Dict], first: Dict) -> Dict:
    base = [{"role": "system", "content": system_prompt()}] + few_shot_messages(shots) + \
           [{"role": "user", "content": f"Clinical note:\n{note}"}, {"role": "assistant", "content": first["raw"]}]
    c1 = chat(model, base + [{"role": "user", "content": REFLECT_PROMPT}], json_mode=False, num_predict=500)
    c2 = chat(model, base + [{"role": "user", "content": REFLECT_PROMPT}, {"role": "assistant", "content": c1["content"]},
                             {"role": "user", "content": REVISE_PROMPT}])
    s = _summ([c1, c2])
    s["n_calls"] += 1
    s["output_tokens"] += first["output_tokens"]
    s["prompt_tokens"] += first["prompt_tokens"]
    s["wall"] += first["wall"]
    return dict(bundle=parse_json(c2["content"]), raw=c2["content"], reflection=c1["content"], logprobs=c2["logprobs"],
                done_reason=c2["done_reason"], **s)
