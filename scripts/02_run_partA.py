"""Part A: architecture-controlled evaluation on the full validation and test splits.

1. Learn site vocabulary and section headers from training notes only.
2. Run the rule-based converter, the single-pass pipeline, Aegis-EHR and its
   ablations on every validation and test record (multiprocessing).
3. Store one JSON line per record with compact evaluations, agent features,
   temporal-repair statistics, KB-coverage variants and per-agent timing.

Usage: python scripts/02_run_partA.py [--workers 14] [--limit N] [--holdout Site-C]
With --holdout the vocabulary/headers are learned without that profile and only
its test records are processed (leave-one-profile-out).
"""
import argparse
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.agents import Mesh, MeshConfig, PharmacovigilanceAgent, feature_vector, run_rule, run_single  # noqa: E402
from aegis.knowledge import kb_overlap, operational_kb  # noqa: E402
from aegis.metrics import evaluate_record  # noqa: E402
from aegis.nlp import BASE_HEADERS, build_lexicon, learn_headers, learn_vocabulary, parse_note  # noqa: E402
from aegis.synth import read_jsonl  # noqa: E402

DATA = ROOT / "data" / "corpus"
RES = ROOT / "results"
KEEP = ["valid", "cf1", "precision", "recall", "spi", "halluc", "tf", "n_slots", "fabricated", "precise_pred",
        "tcr_sat", "tcr_n", "tcr_agent_sat", "tcr_agent_n", "tcr_held_sat", "tcr_held_n", "pv_tp", "pv_fn", "pv_fp",
        "pv_alerts", "pv_gold_pos", "pv_pred_pos", "acceptable", "n_critical", "swes", "medbleu", "critical", "major",
        "minor", "n_gold", "n_pred", "tp", "fp", "fn", "slot_counts"]
SEMANTIC_CRIT = {"hallucinated_condition", "hallucinated_drug", "hallucinated_allergen", "missed_drug",
                 "missed_allergen", "drug_status", "dose_error", "unknown_code"}

CONFIGS = {
    "aegis": MeshConfig(),
    "no_schema": MeshConfig(use_schema=False),
    "no_reasoning": MeshConfig(use_reasoning=False),
    "no_temporal": MeshConfig(use_temporal=False),
    "no_pv": MeshConfig(use_pv=False),
    "no_repair": MeshConfig(repair=False),
    "closest_valid": MeshConfig(temporal_policy="closest_valid"),
    "order_STRP": MeshConfig(order=("S", "T", "R", "P")),
    "order_SPRT": MeshConfig(order=("S", "P", "R", "T")),
    "parallel": MeshConfig(parallel=True),
}
KB_VARIANTS = {"cov100": dict(coverage=1.0), "cov70": dict(coverage=0.7), "cov50": dict(coverage=0.5)}

G = {}


def compact(e):
    return {k: e[k] for k in KEEP}


def labels(e):
    return dict(schema_ok=bool(e["valid"]),
                semantic_ok=not any(k in SEMANTIC_CRIT for k in e["critical"]),
                temporal_ok=not e["critical"].get("fabricated_timestamp") and not e["major"].get("wrong_timestamp"),
                pv_ok=e["pv_fn"] == 0, acceptable=bool(e["acceptable"]))


def init(learned_path):
    L = json.loads(Path(learned_path).read_text())
    vocab = [tuple(v) for v in L["vocab"]]
    headers = dict(BASE_HEADERS)
    headers.update(L["headers"])
    G["headers"] = headers
    G["lex_rule"] = build_lexicon("rule")
    G["lex"] = build_lexicon("schema", vocab)
    G["kb"] = operational_kb()
    G["kbs"] = {k: operational_kb(**v) for k, v in KB_VARIANTS.items()}
    G["mesh"] = Mesh(G["lex"], G["lex"], headers, G["kb"])


def work(rec):
    text = rec["text"]
    out = dict(id=rec["id"], split=rec["split"], profile=rec["profile"],
               meta={k: rec["meta"][k] for k in ("age", "sex", "race", "note_type", "n_meds", "n_conditions",
                                                 "n_allergies", "temporal_vague", "conflict", "near_miss",
                                                 "source_inconsistent", "tiers", "n_tokens")},
               n_gold_signals=len(rec["gold_signals"]), sys={}, conf={}, timing={})
    t = time.perf_counter()
    o = run_rule(text, G["lex_rule"], G["kb"])
    out["timing"]["rule"] = time.perf_counter() - t
    out["sys"]["rule"] = compact(evaluate_record(rec, o["bundle"], o["alerts"]))
    out["conf"]["rule"] = o["conf_raw"]
    t = time.perf_counter()
    o = run_single(text, G["lex"], G["headers"], G["kb"])
    out["timing"]["single"] = time.perf_counter() - t
    out["sys"]["single"] = compact(evaluate_record(rec, o["bundle"], o["alerts"]))
    out["conf"]["single"] = o["conf_raw"]
    mesh = G["mesh"]
    F = parse_note(text, G["lex"], "full", G["headers"])
    Ps = parse_note(text, G["lex"], "schema", G["headers"])
    for name, cfg in CONFIGS.items():
        t = time.perf_counter()
        o = mesh.run(text, cfg, F=F, Ps=Ps)
        el = time.perf_counter() - t
        e = evaluate_record(rec, o["bundle"], o["alerts"])
        out["sys"][name] = compact(e)
        out["sys"][name]["pre_valid"] = bool(o["pre_valid"])
        out["sys"][name]["pre_errors"] = o.get("pre_errors", {})
        if name in ("aegis", "closest_valid", "no_repair", "no_temporal", "no_reasoning", "order_STRP", "parallel"):
            out.setdefault("features", {})[name] = {a: feature_vector(o["reports"], a) for a in ("S", "R", "T", "P")}
            out.setdefault("labels", {})[name] = labels(e)
        if name == "aegis":
            out["timing"]["aegis"] = el
            out["timing"]["aegis_steps"] = o["timing"]
            out["gov"] = dict(oov=o["reports"]["R"].features["oov"], sup_frac=o["reports"]["R"].features["sup_frac"],
                              frac_learned=o["reports"]["R"].features["frac_learned"],
                              unknown_med_tokens=o["reports"]["P"].features["unknown_med_tokens"],
                              vague=o["reports"]["T"].features["vague"])
            out["temporal_stats"] = {"aegis": o["reports"]["T"].stats}
            out["repairs"] = dict(R=[a[0] for a in o["reports"]["R"].actions], S=len(o["reports"]["S"].actions))
            for kname, kb in G["kbs"].items():
                _, alerts, _ = PharmacovigilanceAgent().run(o["bundle"], F, kb)
                ek = evaluate_record(rec, o["bundle"], alerts)
                out.setdefault("kb", {})[kname] = {k: ek[k] for k in ("pv_tp", "pv_fn", "pv_fp", "pv_alerts",
                                                                       "pv_pred_pos", "acceptable")}
        if name == "closest_valid":
            out["temporal_stats"]["closest_valid"] = o["reports"]["T"].stats
    return out


def learn(holdout=None, n=40000, seed=0):
    recs = [r for r in read_jsonl(DATA / "train.jsonl") if holdout is None or r["site"] != holdout]
    random.Random(seed).shuffle(recs)
    recs = recs[:n]
    t = time.time()
    vocab = learn_vocabulary(recs, max_records=n)
    headers = learn_headers(recs, max_records=min(n, 20000))
    return dict(vocab=vocab, headers=headers, n_train=len(recs), seconds=time.time() - t, holdout=holdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--holdout", default=None)
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    tag = f"loso_{args.holdout}" if args.holdout else "partA"
    learned_path = RES / f"learned_{tag}.json"
    if not learned_path.exists():
        L = learn(args.holdout)
        learned_path.write_text(json.dumps(L, indent=1))
        print(f"learned {len(L['vocab'])} forms, headers {L['headers']} in {L['seconds']:.0f}s", flush=True)
    (RES / "kb_overlap.json").write_text(json.dumps({"default": kb_overlap(operational_kb()),
                                                     **{k: kb_overlap(operational_kb(**v)) for k, v in KB_VARIANTS.items()}},
                                                    indent=1))
    recs = []
    for sp in ("val", "test"):
        if args.holdout is None:
            rr = list(read_jsonl(DATA / f"{sp}.jsonl"))
        elif sp == "val":      # calibration data from the other profiles only
            rr = [r for r in read_jsonl(DATA / f"{sp}.jsonl") if r["site"] != args.holdout]
        else:
            rr = [r for r in read_jsonl(DATA / f"{sp}.jsonl") if r["site"] == args.holdout]
        recs += rr[: args.limit] if args.limit else rr
    out_path = RES / f"{tag}_records.jsonl"
    t = time.time()
    with Pool(args.workers, initializer=init, initargs=(str(learned_path),)) as pool, out_path.open("w") as f:
        for k, o in enumerate(pool.imap(work, recs, chunksize=32), 1):
            f.write(json.dumps(o) + "\n")
            if k % 2000 == 0:
                print(f"{k}/{len(recs)} {time.time() - t:.0f}s", flush=True)
    print(f"done {len(recs)} records in {time.time() - t:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
