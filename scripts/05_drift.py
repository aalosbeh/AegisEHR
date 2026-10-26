"""Governance Agent drift experiment.

Pools of records are generated under three kinds of documentation shift
(lexical: unseen surface forms; template: new section headers and date format;
population: older, polypharmacy population) at three magnitudes, and processed
by Aegis-EHR. Daily record streams mix in-distribution test records with a
drift pool according to an abrupt or gradual (30-day ramp) schedule starting
on day 40, 60 or 80; 20 replicate streams per scenario and 100 no-drift control
streams. The Governance Agent computes PSI over a 600-record rolling window for
five monitored features against the validation reference distribution.

Reported: detection rate, false alarms, detection delay (alarm - onset),
lead time (material degradation - alarm), and a signed standardized mean
difference per feature, which gives the direction PSI cannot.

Usage: python scripts/05_drift.py [--workers 6]
"""
import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis import calibration as cal  # noqa: E402
from aegis.agents import Mesh, MeshConfig, feature_vector  # noqa: E402
from aegis.knowledge import operational_kb  # noqa: E402
from aegis.metrics import evaluate_record  # noqa: E402
from aegis.nlp import BASE_HEADERS, build_lexicon  # noqa: E402
from aegis.synth import PROFILES, Generator, drift_profile, make_record  # noqa: E402

RES = ROOT / "results"
FEATS = ["oov", "sup_frac", "frac_learned", "unknown_med_tokens", "vague"]
KINDS = ["lexical", "template", "population"]
MAGS = [0.25, 0.5, 1.0]
G = {}


def init():
    L = json.loads((RES / "learned_partA.json").read_text())
    headers = dict(BASE_HEADERS)
    headers.update(L["headers"])
    lex = build_lexicon("schema", [tuple(v) for v in L["vocab"]])
    G["mesh"] = Mesh(lex, lex, headers, operational_kb())
    G["gen"] = Generator(seed=777)


def process(job):
    kind, mag, site, i = job
    prof = drift_profile(site, kind, mag)
    r = make_record(G["gen"], 500000 + i, prof)
    o = G["mesh"].run(r["text"], MeshConfig())
    e = evaluate_record(r, o["bundle"], o["alerts"])
    rep = o["reports"]
    gov = dict(oov=rep["R"].features["oov"], sup_frac=rep["R"].features["sup_frac"],
               frac_learned=rep["R"].features["frac_learned"], unknown_med_tokens=rep["P"].features["unknown_med_tokens"],
               vague=rep["T"].features["vague"])
    return dict(kind=kind, mag=mag, gov=gov, acceptable=bool(e["acceptable"]), cf1=e["cf1"], valid=bool(e["valid"]),
                features={a: feature_vector(rep, a) for a in ("S", "R", "T", "P")})


def edges_from(ref_vals, bins=10):
    v = np.sort(ref_vals)
    return np.unique(v[(np.arange(1, bins) * len(v) / bins).astype(int)])


def psi_counts(ref, cur, floor=1e-6):
    p = np.maximum(cur / cur.sum(-1, keepdims=True), floor)
    q = np.maximum(ref / ref.sum(), floor)
    return ((p - q) * np.log(p / q)).sum(-1)


def simulate(id_pool, dr_pool, ref, rng, onset=None, shape="abrupt", days=120, per_day=200, window=600,
             thresholds=(0.1, 0.25), ramp=30, delta=0.02, base_acc=None):
    """One stream. Returns alarm days per threshold, degradation day, KS alarm day, signed SMD at alarm."""
    n = days * per_day
    t = np.repeat(np.arange(days), per_day)
    if onset is None:
        m = np.zeros(days)
    elif shape == "abrupt":
        m = (np.arange(days) >= onset).astype(float)
    else:
        m = np.clip((np.arange(days) - onset + 1) / ramp, 0, 1)
    from_drift = rng.random(n) < m[t]
    idx_id = rng.integers(0, len(id_pool["acc"]), n)
    idx_dr = rng.integers(0, len(dr_pool["acc"]), n) if dr_pool is not None else idx_id
    X = np.where(from_drift[:, None], dr_pool["X"][idx_dr] if dr_pool is not None else 0, id_pool["X"][idx_id])
    acc = np.where(from_drift, dr_pool["acc"][idx_dr] if dr_pool is not None else 0, id_pool["acc"][idx_id])
    conf = np.where(from_drift, dr_pool["conf"][idx_dr] if dr_pool is not None else 0, id_pool["conf"][idx_id])
    psi_daily = np.zeros((days, len(FEATS)))
    smd_daily = np.zeros((days, len(FEATS)))
    ks_alarm = None
    from scipy.stats import ks_2samp
    for d in range(days):
        hi = (d + 1) * per_day
        lo = max(0, hi - window)
        if hi - lo < window // 2:
            continue
        W = X[lo:hi]
        for f in range(len(FEATS)):
            e = ref["edges"][f]
            cnt = np.bincount(np.searchsorted(e, W[:, f], side="right"), minlength=len(e) + 1)
            psi_daily[d, f] = psi_counts(ref["counts"][f], cnt[None, :])[0]
            smd_daily[d, f] = (W[:, f].mean() - ref["mean"][f]) / max(ref["sd"][f], 1e-9)
        if ks_alarm is None and d % 2 == 0:
            ps = [ks_2samp(W[:, f], ref["sample"][:, f]).pvalue for f in range(len(FEATS))]
            if min(ps) < 0.01 / len(FEATS):
                ks_alarm = d
    peak = psi_daily.max(1)
    alarms = {}
    for thr in thresholds:
        hit = np.where(peak > thr)[0]
        alarms[str(thr)] = int(hit[0]) if len(hit) else None
    acc_day = acc.reshape(days, per_day).mean(1)
    roll = np.convolve(acc_day, np.ones(7) / 7, mode="full")[:days]
    roll[:6] = acc_day[:6].mean()
    degr = None
    if onset is not None:
        bad = np.where((np.arange(days) >= onset) & (roll < base_acc - delta))[0]
        degr = int(bad[0]) if len(bad) else None
    a = alarms["0.25"]
    smd = smd_daily[a].tolist() if a is not None else None
    auto_day = ((conf >= 0.70).reshape(days, per_day)).mean(1)
    silent_day = (((conf >= 0.70) & (acc < 0.5)).reshape(days, per_day)).mean(1)
    return dict(alarms=alarms, degradation=degr, ks_alarm=ks_alarm, smd_at_alarm=smd, peak_psi=peak.tolist(),
                roll_acc=roll.tolist(), auto_rate=auto_day.tolist(), silent=silent_day.tolist(),
                psi_by_feature=psi_daily.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--per_pool", type=int, default=1200)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    C = pickle.load(open(RES / "calibrators_partA.pkl", "rb"))["aegis"]
    rows = [json.loads(l) for l in open(RES / "partA_records.jsonl")]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]

    def joint(feature_rows):
        Fx = {a: [f[a] for f in feature_rows] for a in cal.AGENTS}
        return C["aggs"]["noisy_and"](cal.predict_agents(C["models"], Fx))

    ref_X = np.array([[r["gov"][f] for f in FEATS] for r in val])
    ref = dict(edges=[edges_from(ref_X[:, f]) for f in range(len(FEATS))], mean=ref_X.mean(0), sd=ref_X.std(0),
               sample=ref_X[np.random.default_rng(0).choice(len(ref_X), 3000, replace=False)])
    ref["counts"] = [np.bincount(np.searchsorted(ref["edges"][f], ref_X[:, f], side="right"),
                                 minlength=len(ref["edges"][f]) + 1) for f in range(len(FEATS))]
    id_pool = dict(X=np.array([[r["gov"][f] for f in FEATS] for r in test]),
                   acc=np.array([r["sys"]["aegis"]["acceptable"] for r in test], float),
                   conf=joint([r["features"]["aegis"] for r in test]))
    base_acc = float(id_pool["acc"].mean())
    jobs = []
    for kind in KINDS:
        for mag in MAGS:
            for site, prof in PROFILES.items():
                for i in range(int(round(args.per_pool * prof.share))):
                    jobs.append((kind, mag, site, i))
    with Pool(args.workers, initializer=init) as pool:
        out = pool.map(process, jobs, chunksize=50)
    pools = {}
    pool_stats = {}
    for kind in KINDS:
        for mag in MAGS:
            sub = [o for o in out if o["kind"] == kind and o["mag"] == mag]
            X = np.array([[o["gov"][f] for f in FEATS] for o in sub])
            pools[(kind, mag)] = dict(X=X, acc=np.array([o["acceptable"] for o in sub], float),
                                      conf=joint([o["features"] for o in sub]))
            pool_stats[f"{kind}@{mag}"] = dict(n=len(sub), acceptable=float(np.mean([o["acceptable"] for o in sub])),
                                               cf1=float(np.mean([o["cf1"] for o in sub])),
                                               auto_rate=float((pools[(kind, mag)]["conf"] >= 0.7).mean()),
                                               mean_features={f: float(X[:, k].mean()) for k, f in enumerate(FEATS)})
    pool_stats["in_distribution"] = dict(n=len(test), acceptable=base_acc, auto_rate=float((id_pool["conf"] >= 0.7).mean()),
                                         mean_features={f: float(id_pool["X"][:, k].mean()) for k, f in enumerate(FEATS)})
    rng = np.random.default_rng(2026)
    runs = []
    examples = {}
    for kind in KINDS:
        for mag in MAGS:
            for shape in ("abrupt", "gradual"):
                for onset in (40, 60, 80):
                    for rep in range(args.reps):
                        s = simulate(id_pool, pools[(kind, mag)], ref, rng, onset, shape, base_acc=base_acc)
                        runs.append(dict(kind=kind, mag=mag, shape=shape, onset=onset, rep=rep,
                                         alarm_010=s["alarms"]["0.1"], alarm_025=s["alarms"]["0.25"],
                                         degradation=s["degradation"], ks_alarm=s["ks_alarm"], smd=s["smd_at_alarm"]))
                        if rep == 0 and onset == 60 and mag == 1.0:
                            examples[f"{kind}_{shape}"] = dict(peak_psi=s["peak_psi"], roll_acc=s["roll_acc"],
                                                               auto_rate=s["auto_rate"], silent=s["silent"],
                                                               psi_by_feature=s["psi_by_feature"],
                                                               alarm=s["alarms"]["0.25"], degradation=s["degradation"],
                                                               onset=onset)
    controls = []
    for rep in range(100):
        s = simulate(id_pool, None, ref, rng, None, base_acc=base_acc)
        controls.append(dict(alarm_010=s["alarms"]["0.1"], alarm_025=s["alarms"]["0.25"], ks_alarm=s["ks_alarm"],
                             max_psi=float(max(s["peak_psi"]))))
    # summaries ------------------------------------------------------------------------------
    summ = {}
    for kind in KINDS:
        for mag in MAGS:
            for shape in ("abrupt", "gradual"):
                rr = [r for r in runs if r["kind"] == kind and r["mag"] == mag and r["shape"] == shape]
                ent = {"n": len(rr)}
                for det in ("alarm_025", "alarm_010", "ks_alarm"):
                    false = [r for r in rr if r[det] is not None and r[det] < r["onset"]]
                    det_ok = [r for r in rr if r[det] is not None and r[det] >= r["onset"]]
                    delays = [r[det] - r["onset"] for r in det_ok]
                    leads = [r["degradation"] - r[det] for r in det_ok if r["degradation"] is not None]
                    ent[det] = dict(detection_rate=len(det_ok) / len(rr), false_alarm_rate=len(false) / len(rr),
                                    delay_median=float(np.median(delays)) if delays else None,
                                    delay_iqr=[float(np.quantile(delays, 0.25)), float(np.quantile(delays, 0.75))] if delays else None,
                                    lead_median=float(np.median(leads)) if leads else None,
                                    lead_iqr=[float(np.quantile(leads, 0.25)), float(np.quantile(leads, 0.75))] if leads else None,
                                    early_warning_share=float(np.mean([l > 0 for l in leads])) if leads else None)
                ent["degradation_rate"] = float(np.mean([r["degradation"] is not None for r in rr]))
                smds = [r["smd"] for r in rr if r["smd"] is not None]
                ent["signed_smd_at_alarm"] = {f: float(np.mean([s[k] for s in smds])) for k, f in enumerate(FEATS)} if smds else None
                summ[f"{kind}@{mag}@{shape}"] = ent
    ctrl = dict(n=len(controls), streams_with_false_alarm_025=float(np.mean([c["alarm_025"] is not None for c in controls])),
                streams_with_false_alarm_010=float(np.mean([c["alarm_010"] is not None for c in controls])),
                streams_with_ks_alarm=float(np.mean([c["ks_alarm"] is not None for c in controls])),
                max_psi_median=float(np.median([c["max_psi"] for c in controls])),
                max_psi_p95=float(np.quantile([c["max_psi"] for c in controls], 0.95)))
    T = dict(pools=pool_stats, scenarios=summ, controls=ctrl, base_acceptability=base_acc,
             design=dict(days=120, per_day=200, window=600, ramp_days=30, onsets=[40, 60, 80], reps=args.reps,
                         thresholds=[0.1, 0.25], degradation_rule="7-day rolling acceptability below baseline minus 2 pp",
                         features=FEATS, ks="two-sample KS per feature, Bonferroni alpha 0.01, checked every 2 days"))
    (RES / "tables" / "drift.json").write_text(json.dumps(T, indent=1))
    (RES / "figdata" / "drift_examples.json").write_text(json.dumps(examples))
    (RES / "drift_runs.json").write_text(json.dumps(runs))
    print(json.dumps(pool_stats, indent=1))
    print(json.dumps(ctrl, indent=1))
    for k, v in summ.items():
        a = v["alarm_025"]
        print(k, "det %.2f FA %.2f delay %s lead %s | degr %.2f | KS det %.2f FA %.2f" % (
            a["detection_rate"], a["false_alarm_rate"], a["delay_median"], a["lead_median"], v["degradation_rate"],
            v["ks_alarm"]["detection_rate"], v["ks_alarm"]["false_alarm_rate"]))


if __name__ == "__main__":
    main()
