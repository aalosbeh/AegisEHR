"""Run the local LLM systems on the stratified LLM evaluation subset.

Usage:
  python scripts/03_run_llm.py --model qwen2.5:7b --variant direct
  python scripts/03_run_llm.py --model qwen2.5:7b --variant validator   # needs the direct run
  python scripts/03_run_llm.py --model qwen2.5:7b --variant reflect     # needs the direct run

Outputs are appended to results/llm/<model>_<variant>.jsonl, so an interrupted
run resumes where it stopped. Run metadata (model digest, prompt hash, decoding
options, dates) goes to results/llm/<model>_<variant>.meta.json.
"""
import argparse
import datetime as dt
import hashlib
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis import fhir  # noqa: E402
from aegis.llm import model_info, run_direct, run_reflect, run_validator, system_prompt  # noqa: E402
from aegis.synth import read_jsonl  # noqa: E402
from aegis.terminology import ALLERGEN_BY_KEY, COND_BY_KEY, DRUG_BY_KEY, LAB_BY_KEY  # noqa: E402

DATA = ROOT / "data" / "corpus"
OUT = ROOT / "results" / "llm"


def subset_ids(n_per_site: int, seed: int = 11):
    path = ROOT / "data" / "llm_subset.json"
    if path.exists():
        return json.loads(path.read_text())
    by_site = {}
    for r in read_jsonl(DATA / "test.jsonl"):
        by_site.setdefault(r["profile"], []).append(r["id"])
    rng = random.Random(seed)
    ids = []
    for k in sorted(by_site):
        ids += rng.sample(sorted(by_site[k]), n_per_site)
    path.write_text(json.dumps(ids))
    return ids


def _target_with_issues(r):
    """Gold bundle plus one DetectedIssue per gold signal (few-shot targets only)."""
    b = json.loads(json.dumps(r["gold"]))
    res = [e["resource"] for e in b["entry"]]
    index = {}
    for x in res:
        rt = x["resourceType"]
        code = None
        if rt == "Condition":
            code = next((k for k, c in COND_BY_KEY.items() if c["snomed"] == x["code"]["coding"][0]["code"]), None)
        elif rt == "MedicationRequest":
            code = next((k for k, d in DRUG_BY_KEY.items() if d["rxcui"] == x["medicationCodeableConcept"]["coding"][0]["code"]), None)
        elif rt == "AllergyIntolerance":
            code = next((k for k, a in ALLERGEN_BY_KEY.items() if a["snomed"] == x["code"]["coding"][0]["code"]), None)
        elif rt == "Observation":
            code = next((k for k, l in LAB_BY_KEY.items() if l["loinc"] == x["code"]["coding"][0]["code"]), None)
        if code and code not in index:
            index[code] = f"{rt}/{x['id']}"
    for i, s in enumerate(r["gold_signals"], 1):
        code = {"ddi": "DRG", "allergy": "ALGY", "condition": "DRG", "lab": "DRG"}[s["type"]]
        imp = [index[c] for c in s["components"] if c in index]
        sev = "high" if s["severity"] == "major" else "moderate"
        detail = f"{s['type']} signal: " + " + ".join(s["components"])
        res.append(fhir.detected_issue(f"d{i}", "p1", code, sev, detail, imp))
    return compact(fhir.bundle(res))


CLINICAL_SYSTEMS = ("http://snomed.info/sct", "http://www.nlm.nih.gov/research/umls/rxnorm", "http://loinc.org")


def compact(b):
    """Compact R4-valid target: keep display strings only on clinical codings
    (SNOMED CT, RxNorm, LOINC), written before the code so the model names the
    concept before choosing its code; drop secondary ICD-10 codings,
    Condition.verificationStatus and category elements (optional in R4)."""
    def walk(x):
        if isinstance(x, dict):
            if "system" in x and "code" in x and set(x) <= {"system", "code", "display"}:
                if x["system"] in CLINICAL_SYSTEMS and "display" in x:
                    return {"system": x["system"], "display": x["display"], "code": x["code"]}
                return {"system": x["system"], "code": x["code"]}
            out = {}
            for k, v in x.items():
                if k in ("verificationStatus", "category") and x.get("resourceType") not in ("DetectedIssue",):
                    if k == "verificationStatus" and x.get("resourceType") == "AllergyIntolerance":
                        out[k] = walk(v)
                    continue
                if k == "display" and x.get("resourceType") is None and "system" not in x:
                    continue
                out[k] = walk(v)
            if "coding" in out and isinstance(out["coding"], list):
                out["coding"] = [c for c in out["coding"] if c.get("system") != "http://hl7.org/fhir/sid/icd-10-cm"]
            return out
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(b)


def few_shots():
    path = ROOT / "data" / "llm_fewshot_ids.json"
    want = json.loads(path.read_text()) if path.exists() else None
    chosen = {}
    for r in read_jsonl(DATA / "train.jsonl"):
        if want:
            if r["id"] in want:
                chosen[r["id"]] = r
            if len(chosen) == len(want):
                break
            continue
        m, n = r["meta"], len(r["gold"]["entry"])
        if "A" not in chosen and r["profile"] == "A" and m["note_type"] == "hp" and m["n_allergies"] and \
                m["conflict"] and 9 <= n <= 13 and r["gold_signals"]:
            chosen["A"] = r
        if "B" not in chosen and r["profile"] == "B" and m["note_type"] == "dc" and m["temporal_vague"] and \
                m["conflict"] and 9 <= n <= 13 and any(x["assertion"] == "negated" for x in m["mentions"]):
            chosen["B"] = r
        if len(chosen) == 2:
            break
    shots = list(chosen.values())
    if not want:
        path.write_text(json.dumps([s["id"] for s in shots]))
    return [dict(id=s["id"], text=s["text"], target=_target_with_issues(s)) for s in shots]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--variant", choices=["direct", "validator", "reflect"], default="direct")
    ap.add_argument("--n_per_site", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.model.replace(":", "-").replace("/", "_")
    out_path = OUT / f"{tag}_{args.variant}.jsonl"
    ids = subset_ids(args.n_per_site)
    if args.limit:
        ids = ids[: args.limit]
    want = set(ids)
    notes = {r["id"]: r["text"] for r in read_jsonl(DATA / "test.jsonl") if r["id"] in want}
    shots = few_shots()
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.open() if l.strip()}
    first = {}
    if args.variant != "direct":
        src = OUT / f"{tag}_direct.jsonl"
        first = {d["id"]: d for d in map(json.loads, src.open())}
    meta = dict(model=model_info(args.model), variant=args.variant, started=dt.datetime.now().isoformat(),
                prompt_sha256=hashlib.sha256(system_prompt().encode()).hexdigest(),
                few_shot_ids=[s["id"] for s in shots], options=dict(temperature=0, seed=42, num_ctx=10240,
                                                                     num_predict=3000, top_p=1.0, format="json"),
                retry_policy="validator: up to 2 retries with the first 15 validator errors; reflect: 1 critique + 1 revision",
                n_records=len(ids))
    (OUT / f"{tag}_{args.variant}.meta.json").write_text(json.dumps(meta, indent=2))
    todo = [i for i in ids if i not in done]
    print(f"{args.model} {args.variant}: {len(todo)} to run ({len(done)} done)", flush=True)

    def work(rid):
        note = notes[rid]
        if args.variant == "direct":
            o = run_direct(args.model, note, shots)
        elif args.variant == "validator":
            o = run_validator(args.model, note, shots, first[rid])
        else:
            o = run_reflect(args.model, note, shots, first[rid])
        lp = o.pop("logprobs") or []
        o["mean_logprob"] = sum(lp) / len(lp) if lp else None
        o["min_logprob"] = min(lp) if lp else None
        o["logprobs"] = [round(x, 4) for x in lp]
        o["id"] = rid
        return o

    with out_path.open("a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, rid): rid for rid in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                o = fu.result()
            except Exception as e:  # keep going; the record is retried on the next run
                print("error", futs[fu], repr(e), flush=True)
                continue
            f.write(json.dumps(o) + "\n")
            f.flush()
            if k % 10 == 0:
                print(f"{k}/{len(todo)} last wall {o['wall']:.1f}s out_tokens {o['output_tokens']}", flush=True)
    meta["finished"] = dt.datetime.now().isoformat()
    (OUT / f"{tag}_{args.variant}.meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
