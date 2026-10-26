"""Three smaller experiments on the test split.

1. Robustness to token corruption (0-50% of alphabetic tokens receive one
   random edit); calibrators come from the clean validation split.
2. Pairwise agent ablations (two agents removed at once).
3. End-to-end latency in a single process, including all parsing.

Usage: python scripts/07_noise_pairwise_latency.py [--workers 6] [--n 2400]
"""
import argparse
import json
import pickle
import random
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis import calibration as cal  # noqa: E402
from aegis import stats  # noqa: E402
from aegis.agents import Mesh, MeshConfig, feature_vector, run_rule, run_single  # noqa: E402
from aegis.knowledge import operational_kb  # noqa: E402
from aegis.metrics import evaluate_record  # noqa: E402
from aegis.nlp import BASE_HEADERS, build_lexicon  # noqa: E402
from aegis.synth import _typo, read_jsonl  # noqa: E402

RES = ROOT / "results"
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
PAIRS = {"no_S_R": dict(use_schema=False, use_reasoning=False), "no_S_T": dict(use_schema=False, use_temporal=False),
         "no_S_P": dict(use_schema=False, use_pv=False), "no_R_T": dict(use_reasoning=False, use_temporal=False),
         "no_R_P": dict(use_reasoning=False, use_pv=False), "no_T_P": dict(use_temporal=False, use_pv=False)}
G = {}


def init():
    L = json.loads((RES / "learned_partA.json").read_text())
    headers = dict(BASE_HEADERS)
    headers.update(L["headers"])
    G["headers"] = headers
    G["lex"] = build_lexicon("schema", [tuple(v) for v in L["vocab"]])
    G["kb"] = operational_kb()
    G["mesh"] = Mesh(G["lex"], G["lex"], headers, G["kb"])


def corrupt(text, p, seed):
    rng = random.Random(seed)
    lines = text.split("\n")
    out = lines[:2]
    for line in lines[2:]:
        m = re.match(r"^(\s*[A-Za-z][A-Za-z/&\- ]{0,40}:\s*)(.*)$", line)
        head, body = (m.group(1), m.group(2)) if m else ("", line)
        body = re.sub(r"[A-Za-z]{4,}", lambda w: _typo(w.group(0), rng) if rng.random() < p else w.group(0), body)
        out.append(head + body)
    return "\n".join(out)


def work_noise(job):
    rec, p = job
    text = corrupt(rec["text"], p, hash((rec["id"], p)) % (2 ** 31)) if p > 0 else rec["text"]
    o = G["mesh"].run(text, MeshConfig())
    ea = evaluate_record(rec, o["bundle"], o["alerts"])
    s = run_single(text, G["lex"], G["headers"], G["kb"])
    es = evaluate_record(rec, s["bundle"], s["alerts"])
    return dict(id=rec["id"], p=p, aegis=dict(acceptable=ea["acceptable"], cf1=ea["cf1"], valid=ea["valid"],
                                              spi=ea["spi"], halluc=ea["halluc"]),
                single=dict(acceptable=es["acceptable"], cf1=es["cf1"], spi=es["spi"], halluc=es["halluc"]),
                feats={a: feature_vector(o["reports"], a) for a in ("S", "R", "T", "P")}, single_conf=s["conf_raw"])


def work_pair(rec):
    out = {"id": rec["id"]}
    for name, kw in PAIRS.items():
        o = G["mesh"].run(rec["text"], MeshConfig(**kw))
        e = evaluate_record(rec, o["bundle"], o["alerts"])
        out[name] = dict(acceptable=e["acceptable"], cf1=e["cf1"], tf=e["tf"], valid=e["valid"],
                         pv_tp=e["pv_tp"], pv_fn=e["pv_fn"], pv_alerts=e["pv_alerts"], pv_fp=e["pv_fp"])
    return out


def latency(recs):
    """Single-process wall clock, including all parsing; per-agent steps from the orchestrator."""
    init()
    lat = {"rule": [], "single": [], "aegis": []}
    steps = {}
    lex_rule = build_lexicon("rule")
    for r in recs:
        t = time.perf_counter()
        run_rule(r["text"], lex_rule, G["kb"])
        lat["rule"].append(time.perf_counter() - t)
        t = time.perf_counter()
        run_single(r["text"], G["lex"], G["headers"], G["kb"])
        lat["single"].append(time.perf_counter() - t)
        t = time.perf_counter()
        o = G["mesh"].run(r["text"], MeshConfig())
        lat["aegis"].append(time.perf_counter() - t)
        for k, v in o["timing"].items():
            steps.setdefault(k, []).append(v)
    summary = {k: dict(mean_ms=1000 * float(np.mean(v)), median_ms=1000 * float(np.median(v)),
                       p95_ms=1000 * float(np.quantile(v, 0.95)), records_per_s=float(1 / np.mean(v)))
               for k, v in lat.items()}
    return summary, {k: 1000 * float(np.mean(v)) for k, v in steps.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n", type=int, default=2400)
    ap.add_argument("--only", choices=["all", "latency"], default="all")
    args = ap.parse_args()
    if args.only == "latency":
        test = list(read_jsonl(ROOT / "data" / "corpus" / "test.jsonl"))
        random.Random(5).shuffle(test)
        path = RES / "tables" / "noise_pairwise_latency.json"
        T = json.loads(path.read_text())
        T["latency_single_process"], T["latency_steps_ms"] = latency(test[:2000])
        path.write_text(json.dumps(T, indent=1))
        print(T["latency_single_process"], T["latency_steps_ms"])
        return
    test = list(read_jsonl(ROOT / "data" / "corpus" / "test.jsonl"))
    random.Random(5).shuffle(test)
    sub = test[: args.n]
    C = pickle.load(open(RES / "calibrators_partA.pkl", "rb"))
    T = {}
    with Pool(args.workers, initializer=init) as pool:
        noise = pool.map(work_noise, [(r, p) for p in RATES for r in sub], chunksize=40)
        pairs = pool.map(work_pair, sub, chunksize=40)
    # noise summary
    ns = {}
    for p in RATES:
        rr = [x for x in noise if x["p"] == p]
        Fx = {a: [x["feats"][a] for x in rr] for a in cal.AGENTS}
        conf = C["aegis"]["aggs"]["noisy_and"](cal.predict_agents(C["aegis"]["models"], Fx))
        acc = np.array([x["aegis"]["acceptable"] for x in rr], float)
        valid = np.array([x["aegis"]["valid"] for x in rr], bool)
        auto = (conf >= 0.70) & valid
        sconf = cal.predict_platt(C["single"], [[x["single_conf"]] for x in rr])
        sacc = np.array([x["single"]["acceptable"] for x in rr], float)
        sauto = sconf >= 0.70
        ns[str(p)] = dict(
            aegis=dict(acceptable=float(acc.mean()), cf1=float(np.mean([x["aegis"]["cf1"] for x in rr])),
                       spi=float(np.mean([x["aegis"]["spi"] for x in rr])),
                       halluc=float(np.mean([x["aegis"]["halluc"] for x in rr])),
                       coverage=float(auto.mean()), selective_acc=float(acc[auto].mean()) if auto.any() else None,
                       silent_error_rate=float(((1 - acc) * auto).mean()), ece=stats.ece(conf, acc)["ece"]),
            single=dict(acceptable=float(sacc.mean()), cf1=float(np.mean([x["single"]["cf1"] for x in rr])),
                        spi=float(np.mean([x["single"]["spi"] for x in rr])),
                        halluc=float(np.mean([x["single"]["halluc"] for x in rr])),
                        coverage=float(sauto.mean()), selective_acc=float(sacc[sauto].mean()) if sauto.any() else None,
                        silent_error_rate=float(((1 - sacc) * sauto).mean()), ece=stats.ece(sconf, sacc)["ece"]))
    T["noise"] = ns
    # pairwise
    base = {r["id"]: r for r in map(json.loads, open(RES / "partA_records.jsonl")) if r["split"] == "test"}
    pw = {}
    full_acc = float(np.mean([base[x["id"]]["sys"]["aegis"]["acceptable"] for x in pairs]))
    full_cf1 = float(np.mean([base[x["id"]]["sys"]["aegis"]["cf1"] for x in pairs]))
    single = {}
    for one in ("no_schema", "no_reasoning", "no_temporal", "no_pv"):
        single[one] = dict(acceptable=float(np.mean([base[x["id"]]["sys"][one]["acceptable"] for x in pairs])) - full_acc,
                           cf1=float(np.mean([base[x["id"]]["sys"][one]["cf1"] for x in pairs])) - full_cf1)
    for name in PAIRS:
        acc = float(np.mean([x[name]["acceptable"] for x in pairs]))
        cf1 = float(np.mean([x[name]["cf1"] for x in pairs]))
        tp = sum(x[name]["pv_tp"] for x in pairs)
        fn = sum(x[name]["pv_fn"] for x in pairs)
        pw[name] = dict(acceptable=acc, d_acceptable=acc - full_acc, cf1=cf1, d_cf1=cf1 - full_cf1,
                        tf=float(np.mean([x[name]["tf"] for x in pairs])), valid=float(np.mean([x[name]["valid"] for x in pairs])),
                        pv_sensitivity=tp / max(tp + fn, 1))
    key = {"S": "no_schema", "R": "no_reasoning", "T": "no_temporal", "P": "no_pv"}
    for name in PAIRS:
        a, b = name.split("_")[1], name.split("_")[2]
        pw[name]["additive_expected_d_acceptable"] = single[key[a]]["acceptable"] + single[key[b]]["acceptable"]
    T["pairwise"] = dict(full=dict(acceptable=full_acc, cf1=full_cf1), single=single, pairs=pw, n=len(pairs))
    T["latency_single_process"], T["latency_steps_ms"] = latency(sub[:2000])
    (RES / "tables" / "noise_pairwise_latency.json").write_text(json.dumps(T, indent=1))
    print(json.dumps(T, indent=1))


if __name__ == "__main__":
    main()
