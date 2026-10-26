"""Analyse Part A records: calibration, abstention-aware comparison, ablations,
aggregators, temporal repair provenance, KB sensitivity, subgroups, failures,
MedBLEU validity, latency, statistical tests and leave-one-profile-out.

Usage: python scripts/04_analyze_partA.py
Writes results/tables/*.json and results/figdata/*.json; saves the fitted
calibrators to results/calibrators_partA.pkl.
"""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sst

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis import calibration as cal  # noqa: E402
from aegis import stats  # noqa: E402
from aegis.knowledge import operational_kb, reference_kb  # noqa: E402

RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figdata"
TAB.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
TAU_AUTO, TAU_REVIEW = 0.70, 0.50


def load(path):
    return [json.loads(l) for l in open(path)]


def pooled(E, num, den):
    return sum(e[num] for e in E) / max(sum(e[den] for e in E), 1)


def system_summary(E, n_boot=2000):
    """Emitted-bundle metrics with 95% bootstrap intervals."""
    out = {}
    for k in ("cf1", "spi", "halluc", "tf", "medbleu", "acceptable", "valid", "swes", "precision", "recall"):
        m, lo, hi = stats.boot_ci([float(e[k]) for e in E], n_boot)
        out[k] = [m, lo, hi]
    out["pre_valid"] = float(np.mean([e.get("pre_valid", e["valid"]) for e in E]))
    out["utr"] = list(stats.boot_ratio_ci([e["fabricated"] for e in E], [e["precise_pred"] for e in E], n_boot))
    out["tcr"] = list(stats.boot_ratio_ci([e["tcr_sat"] for e in E], [e["tcr_n"] for e in E], n_boot))
    out["tcr_agent"] = list(stats.boot_ratio_ci([e["tcr_agent_sat"] for e in E], [e["tcr_agent_n"] for e in E], n_boot))
    out["tcr_held"] = list(stats.boot_ratio_ci([e["tcr_held_sat"] for e in E], [e["tcr_held_n"] for e in E], n_boot))
    out.update(pv_metrics(E))
    return out


def pv_metrics(E):
    tp = sum(e["pv_tp"] for e in E)
    fn = sum(e["pv_fn"] for e in E)
    alerts = sum(e["pv_alerts"] for e in E)
    fp = sum(e["pv_fp"] for e in E)
    rtp = sum(1 for e in E if e["pv_gold_pos"] and e["pv_pred_pos"])
    rfp = sum(1 for e in E if not e["pv_gold_pos"] and e["pv_pred_pos"])
    rtn = sum(1 for e in E if not e["pv_gold_pos"] and not e["pv_pred_pos"])
    rfn = sum(1 for e in E if e["pv_gold_pos"] and not e["pv_pred_pos"])
    sens = tp / max(tp + fn, 1)
    ppv = (alerts - fp) / max(alerts, 1)
    spec = rtn / max(rtn + rfp, 1)
    return dict(pv_sensitivity=sens, pv_ppv=ppv, pv_f1=2 * sens * ppv / max(sens + ppv, 1e-12),
                pv_specificity=spec, pv_fpr=1 - spec, pv_record_sensitivity=rtp / max(rtp + rfn, 1),
                pv_mcc=stats.mcc(rtp, rfp, rtn, rfn), alerts_per_1000=1000 * alerts / len(E),
                false_alerts_per_1000=1000 * fp / len(E), pv_signals=tp + fn)


def feats_of(rows, cfg):
    F = {a: [r["features"][cfg][a] for r in rows] for a in cal.AGENTS if rows[0]["features"][cfg][a]}
    L = {k: [r["labels"][cfg][k] for r in rows] for k in ("schema_ok", "semantic_ok", "temporal_ok", "pv_ok", "acceptable")}
    return F, L


def selective(conf, correct, valid=None, tau=TAU_AUTO):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    valid = np.ones(len(conf), bool) if valid is None else np.asarray(valid, bool)
    auto = (conf >= tau) & valid
    n = len(conf)
    cov = auto.mean()
    rc = stats.risk_coverage(conf * valid, correct)
    return dict(coverage=float(cov), selective_acc=float(correct[auto].mean()) if auto.any() else float("nan"),
                silent_error_rate=float(((1 - correct) * auto).sum() / n), automation_yield=float((correct * auto).sum() / n),
                full_set_acc_abstain_as_error=float((correct * auto).sum() / n), aurc=rc["aurc"], eaurc=rc["eaurc"],
                review_rate=float(((conf >= TAU_REVIEW) & (conf < tau) & valid).mean()),
                abstain_rate=float(((conf < TAU_REVIEW) | ~valid).mean()))


def main():
    rows = load(RES / "partA_records.jsonl")
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    systems = list(test[0]["sys"])
    T = {}

    # ---------------- calibration and joint confidence --------------------------
    calib = {}
    joint = {}
    for cfg in ("aegis", "closest_valid", "no_repair", "no_temporal", "no_reasoning", "order_STRP", "parallel"):
        Fv, Lv = feats_of(val, cfg)
        Ft, Lt = feats_of(test, cfg)
        models = cal.fit_agents(Fv, Lv)
        sv, stt = cal.predict_agents(models, Fv), cal.predict_agents(models, Ft)
        aggs = {}
        for kind in cal.AGGREGATORS:
            ag = cal.Aggregator(kind).fit(sv, Lv["acceptable"])
            aggs[kind] = (ag, ag(stt))
        raw_t = cal.raw_agents(Ft)
        calib[cfg] = dict(models=models, aggs={k: v[0] for k, v in aggs.items()})
        joint[cfg] = dict(agent_sigma=stt, agg={k: v[1] for k, v in aggs.items()},
                          uncal=cal.noisy_and(raw_t, np.ones(len(raw_t))), labels=Lt)
    for name in ("single", "rule"):
        cv = [[r["conf"][name]] for r in val]
        m = cal.fit_platt(cv, [r["sys"][name]["acceptable"] for r in val])
        calib[name] = m
        joint[name] = dict(conf=cal.predict_platt(m, [[r["conf"][name]] for r in test]),
                           raw=np.array([r["conf"][name] for r in test]))
    with open(RES / "calibrators_partA.pkl", "wb") as f:
        pickle.dump(calib, f)

    # ---------------- main table (full coverage) --------------------------------
    main_tab = {}
    for name in ("rule", "single", "aegis"):
        main_tab[name] = system_summary([r["sys"][name] for r in test])
    T["main"] = main_tab

    # ---------------- abstention-aware comparison ---------------------------------
    acc = {n: np.array([r["sys"][n]["acceptable"] for r in test], float) for n in systems}
    valid = {n: np.array([r["sys"][n]["valid"] for r in test], bool) for n in systems}
    sel = {}
    a_conf = joint["aegis"]["agg"]["noisy_and"]
    sel["aegis"] = selective(a_conf, acc["aegis"], valid["aegis"])
    sel["aegis_uncalibrated"] = selective(joint["aegis"]["uncal"], acc["aegis"], valid["aegis"])
    for name in ("single", "rule"):
        sel[name] = selective(joint[name]["conf"], acc[name], valid[name])
        sel[name + "_no_abstention"] = selective(np.ones(len(test)), acc[name], valid[name])
    sel["aegis_no_abstention"] = selective(np.ones(len(test)), acc["aegis"], valid["aegis"])
    cov_a = sel["aegis"]["coverage"]
    for name in ("single", "rule"):
        sel[name + "_matched"] = stats.at_coverage(joint[name]["conf"], acc[name], cov_a)
    sel["aegis_matched"] = stats.at_coverage(a_conf, acc["aegis"], cov_a)
    for target in (0.8, 0.9, 0.95):
        for name, conf in (("aegis", a_conf * valid["aegis"]), ("single", joint["single"]["conf"]),
                           ("rule", joint["rule"]["conf"])):
            sel.setdefault(f"at_{int(target * 100)}", {})[name] = stats.at_coverage(conf, acc[name], target)
    for lam in (0.1, 0.25, 0.5):
        for name, conf in (("aegis", a_conf), ("single", joint["single"]["conf"]), ("rule", joint["rule"]["conf"])):
            auto = (np.asarray(conf) >= TAU_AUTO) & valid[name]
            cost = ((1 - acc[name]) * auto).sum() + lam * (~auto).sum()
            sel.setdefault("cost_per_1000", {}).setdefault(name, {})[str(lam)] = float(1000 * cost / len(test))
        cost_noabs = ((1 - acc["single"])).sum()
        sel["cost_per_1000"].setdefault("single_no_abstention", {})[str(lam)] = float(1000 * cost_noabs / len(test))
    T["selective"] = sel
    curves = {}
    for name, conf, corr in (("Aegis-EHR", a_conf * valid["aegis"], acc["aegis"]),
                             ("Single-pass", joint["single"]["conf"], acc["single"]),
                             ("Rule-based", joint["rule"]["conf"], acc["rule"])):
        rc = stats.risk_coverage(conf, corr)
        idx = np.linspace(0, len(rc["coverage"]) - 1, 200).astype(int)
        curves[name] = dict(coverage=rc["coverage"][idx].tolist(), risk=rc["risk"][idx].tolist(), aurc=rc["aurc"])
    (FIG / "partA_risk_coverage.json").write_text(json.dumps(curves))

    # ---------------- calibration metrics --------------------------------------------
    calt = {}
    for kind in cal.AGGREGATORS:
        calt[f"aegis_{kind}"] = stats.ece(joint["aegis"]["agg"][kind], acc["aegis"])
    calt["aegis_uncalibrated"] = stats.ece(joint["aegis"]["uncal"], acc["aegis"])
    for name in ("single", "rule"):
        calt[name] = stats.ece(joint[name]["conf"], acc[name])
        calt[name + "_raw"] = stats.ece(np.clip(joint[name]["raw"], 0, 1), acc[name])
    for a in cal.AGENTS:
        calt[f"agent_{a}"] = stats.ece(joint["aegis"]["agent_sigma"][a], joint["aegis"]["labels"][cal.AXIS[a]])
    T["calibration"] = {k: {kk: vv for kk, vv in v.items() if kk != "bins"} for k, v in calt.items()}
    (FIG / "partA_reliability.json").write_text(json.dumps({k: v["bins"] for k, v in calt.items()}))

    # ---------------- aggregator comparison --------------------------------------------
    agg_tab = {}
    for kind in cal.AGGREGATORS:
        conf = joint["aegis"]["agg"][kind]
        s = selective(conf, acc["aegis"], valid["aegis"])
        e = calt[f"aegis_{kind}"]
        agg_tab[kind] = dict(ece=e["ece"], brier=e["brier"], mce=e["mce"], aurc=s["aurc"], coverage=s["coverage"],
                             selective_acc=s["selective_acc"], silent_error_rate=s["silent_error_rate"],
                             weights=(calib["aegis"]["aggs"][kind].w.tolist() if calib["aegis"]["aggs"][kind].w is not None else None))
    # weakest-link probe: one agent forced to low confidence, others high
    probe = {}
    hi = {a: np.array([0.95]) for a in cal.AGENTS}
    for agent in cal.AGENTS:
        for low in (0.05, 0.5):
            sig = dict(hi)
            sig[agent] = np.array([low])
            probe[f"{agent}={low}"] = {k: float(calib["aegis"]["aggs"][k](sig)[0]) for k in cal.AGGREGATORS}
            probe[f"{agent}={low}"]["noisy_and_unit_w"] = float(cal.noisy_and(sig, np.ones(4))[0])
            probe[f"{agent}={low}"]["noisy_or_unit_w"] = float(cal.noisy_or(sig, np.ones(4))[0])
    agg_tab["_weakest_link_probe"] = probe
    T["aggregators"] = agg_tab

    # ---------------- ablations (full coverage) ----------------------------------------
    ab = {}
    for name in systems:
        E = [r["sys"][name] for r in test]
        s = system_summary(E, n_boot=500)
        ab[name] = {k: (v[0] if isinstance(v, list) else v) for k, v in s.items()}
    T["ablation"] = ab

    # ---------------- temporal repair provenance -------------------------------------------
    tr = {}
    for pol in ("aegis", "closest_valid"):
        c = Counter()
        for r in test:
            c.update(r["temporal_stats"][pol])
        E = [r["sys"][pol] for r in test]
        sub = [r["sys"][pol] for r in test if r["meta"]["source_inconsistent"]]
        tr[pol] = dict(counts=dict(c), tf=float(np.mean([e["tf"] for e in E])),
                       tcr=pooled(E, "tcr_sat", "tcr_n"), tcr_agent=pooled(E, "tcr_agent_sat", "tcr_agent_n"),
                       utr=pooled(E, "fabricated", "precise_pred"),
                       wrong_timestamps=sum(e["major"].get("wrong_timestamp", 0) for e in E),
                       fabricated=sum(e["critical"].get("fabricated_timestamp", 0) for e in E),
                       inconsistent_records=len(sub),
                       tf_inconsistent=float(np.mean([e["tf"] for e in sub])) if sub else None,
                       tcr_inconsistent=pooled(sub, "tcr_sat", "tcr_n") if sub else None,
                       acceptable=float(np.mean([e["acceptable"] for e in E])))
    T["temporal_repair"] = tr

    # ---------------- KB coverage sensitivity --------------------------------------------
    kbt = {"default": pv_metrics([r["sys"]["aegis"] for r in test])}
    kbt["default"]["acceptable"] = float(np.mean([r["sys"]["aegis"]["acceptable"] for r in test]))
    for k in test[0]["kb"]:
        E = [dict(r["kb"][k], pv_gold_pos=r["sys"]["aegis"]["pv_gold_pos"]) for r in test]
        m = pv_metrics(E)
        m["acceptable"] = float(np.mean([e["acceptable"] for e in E]))
        kbt[k] = m
    kbt["overlap"] = json.loads((RES / "kb_overlap.json").read_text())
    T["kb_sensitivity"] = kbt

    # ---------------- subgroup performance -----------------------------------------------
    def band(a):
        return "18-39" if a < 40 else "40-64" if a < 65 else "65-79" if a < 80 else ">=80"
    sg = defaultdict(dict)
    groups = {"age": lambda r: band(r["meta"]["age"]), "sex": lambda r: r["meta"]["sex"],
              "race": lambda r: r["meta"]["race"], "profile": lambda r: r["profile"],
              "note_type": lambda r: r["meta"]["note_type"]}
    for gname, fn in groups.items():
        buckets = defaultdict(list)
        for r in test:
            buckets[fn(r)].append(r)
        for g, rr in sorted(buckets.items()):
            ent = {"n": len(rr), "mean_meds": float(np.mean([r["meta"]["n_meds"] for r in rr]))}
            for name in ("single", "aegis"):
                E = [r["sys"][name] for r in rr]
                pm = pv_metrics(E)
                ent[name] = dict(acceptable=stats.wilson(sum(e["acceptable"] for e in E), len(E)),
                                 cf1=float(np.mean([e["cf1"] for e in E])), pv_sensitivity=pm["pv_sensitivity"],
                                 pv_ppv=pm["pv_ppv"])
            sg[gname][g] = ent
        for name in ("single", "aegis"):
            vals = [v[name]["acceptable"][0] for g, v in sg[gname].items() if not g.startswith("_")]
            sg[gname].setdefault("_max_gap", {})[name] = float(max(vals) - min(vals))
    T["subgroups"] = sg

    # ---------------- failure taxonomy (Aegis, test) ----------------------------------------
    op_ids = {r["id"] for r in operational_kb()}
    dec = cal.decide(a_conf, valid["aegis"])
    fail = Counter()
    review_reasons = Counter()
    for i, r in enumerate(test):
        e = r["sys"]["aegis"]
        if dec[i] != "AUTO":
            sig = {a: joint["aegis"]["agent_sigma"][a][i] for a in cal.AGENTS}
            review_reasons[min(sig, key=sig.get)] += 1
        if e["acceptable"]:
            continue
        for k, v in e["critical"].items():
            fail[k] += v
    T["failures"] = dict(critical_counts=dict(fail), n_unacceptable=int(sum(1 for r in test if not r["sys"]["aegis"]["acceptable"])),
                         decisions=dict(Counter(dec.tolist())), lowest_agent_in_non_auto=dict(review_reasons),
                         silent_errors=int(sum(1 for i, r in enumerate(test) if dec[i] == "AUTO" and not r["sys"]["aegis"]["acceptable"])))
    # missed signals attributable to KB coverage
    ref = {r["id"]: r for r in reference_kb()}
    missing_rules = sorted(rid for rid, r in ref.items() if r["severity"] in ("major", "moderate") and rid not in op_ids)
    T["failures"]["operational_kb_missing_alerting_rules"] = missing_rules
    gold_by_id = {}
    from aegis.synth import read_jsonl
    for g in read_jsonl(ROOT / "data" / "corpus" / "test.jsonl"):
        gold_by_id[g["id"]] = g["gold_signals"]
    ms = set(missing_rules)
    unavoidable = sum(1 for r in test for sgl in gold_by_id[r["id"]] if sgl["severity"] == "major" and sgl["rule_id"] in ms)
    all_major = sum(1 for r in test for sgl in gold_by_id[r["id"]] if sgl["severity"] == "major")
    T["failures"]["major_signals_total"] = all_major
    T["failures"]["major_signals_with_rule_absent_from_operational_kb"] = unavoidable
    T["failures"]["missed_major_signals_observed"] = int(sum(r["sys"]["aegis"]["critical"].get("missed_major_signal", 0) for r in test))

    # ---------------- MedBLEU validity -------------------------------------------------------
    mb, sw, ac, cf = [], [], [], []
    for r in test:
        for name in ("rule", "single", "aegis", "no_reasoning", "no_temporal", "no_repair", "no_schema"):
            e = r["sys"][name]
            mb.append(e["medbleu"]); sw.append(e["swes"]); ac.append(e["acceptable"]); cf.append(e["cf1"])
    mb, sw, ac, cf = map(np.asarray, (mb, sw, ac, cf))
    from sklearn.metrics import roc_auc_score
    T["medbleu_validity"] = dict(
        n_pairs=int(len(mb)), spearman_swes=float(sst.spearmanr(mb, sw)[0]), spearman_cf1=float(sst.spearmanr(mb, cf)[0]),
        auroc_acceptable=float(roc_auc_score(ac, mb)), kendall_swes=float(sst.kendalltau(mb[:20000], sw[:20000])[0]),
        high_bleu_unacceptable=float(np.mean(~ac[mb >= 0.9].astype(bool))) if (mb >= 0.9).any() else None,
        n_high_bleu=int((mb >= 0.9).sum()),
        unacceptable_share_by_bleu_decile=[float(1 - ac[(mb >= lo) & (mb < lo + 0.1)].mean()) if ((mb >= lo) & (mb < lo + 0.1)).any() else None
                                           for lo in np.arange(0, 1.0, 0.1)])

    # ---------------- latency ---------------------------------------------------------------------
    lat = {}
    for name in ("rule", "single", "aegis"):
        v = np.array([r["timing"][name] for r in test]) * 1000
        lat[name] = dict(mean_ms=float(v.mean()), median_ms=float(np.median(v)), p95_ms=float(np.quantile(v, 0.95)))
    steps = defaultdict(list)
    for r in test:
        for k, v in r["timing"]["aegis_steps"].items():
            steps[k].append(v * 1000)
    lat["aegis_steps_ms"] = {k: float(np.mean(v)) for k, v in steps.items()}
    T["latency"] = lat

    # ---------------- statistical tests (family: Aegis vs 2 baselines x 6 endpoints) ----------------
    pvals, eff = {}, {}
    for base in ("single", "rule"):
        for k in ("cf1", "spi", "tf", "medbleu", "swes"):
            a = [r["sys"]["aegis"][k] for r in test]
            b = [r["sys"][base][k] for r in test]
            key = f"aegis_vs_{base}:{k}"
            pvals[key] = stats.paired_permutation(a, b, 10000)
            eff[key] = dict(mean_diff=float(np.mean(a) - np.mean(b)), cohens_dz=stats.cohens_dz(a, b),
                            cliffs_delta=stats.cliffs_delta(a, b))
        a = [r["sys"]["aegis"]["acceptable"] for r in test]
        b = [r["sys"][base]["acceptable"] for r in test]
        p, n10, n01 = stats.mcnemar(a, b)
        key = f"aegis_vs_{base}:acceptable"
        pvals[key] = p
        eff[key] = dict(risk_difference=float(np.mean(a) - np.mean(b)), n_aegis_only=n10, n_baseline_only=n01,
                        odds_ratio=(n10 + 0.5) / (n01 + 0.5))
    adj = stats.holm(pvals)
    T["tests"] = {k: dict(p=pvals[k], p_holm=adj[k], **eff[k]) for k in pvals}
    T["tests"]["_family"] = f"{len(pvals)} comparisons: Aegis-EHR vs single-pass and rule-based on CF-F1, SPI, TF, MedBLEU, SWES (paired permutation, 10,000 sign flips) and acceptability (exact McNemar); Holm-Bonferroni."

    # ---------------- leave-one-profile-out --------------------------------------------------------
    loso = {}
    for site in ("Site-A", "Site-B", "Site-C"):
        p = RES / f"loso_{site}_records.jsonl"
        if not p.exists():
            continue
        lr_all = load(p)
        lval = [r for r in lr_all if r["split"] == "val"]
        lr = [r for r in lr_all if r["split"] == "test"]
        inprof = [r for r in test if r["profile"] == site[-1]]
        ent = {}
        for name in ("rule", "single", "aegis"):
            lo = system_summary([r["sys"][name] for r in lr], n_boot=500)
            ip = system_summary([r["sys"][name] for r in inprof], n_boot=500)
            ent[name] = dict(loso={k: (v[0] if isinstance(v, list) else v) for k, v in lo.items()},
                             in_profile={k: (v[0] if isinstance(v, list) else v) for k, v in ip.items()})
        if lval:
            Fv, Lv = feats_of(lval, "aegis")
            Ft, Lt = feats_of(lr, "aegis")
            models = cal.fit_agents(Fv, Lv)
            ag = cal.Aggregator("noisy_and").fit(cal.predict_agents(models, Fv), Lv["acceptable"])
            cf = ag(cal.predict_agents(models, Ft))
            accl = np.array([r["sys"]["aegis"]["acceptable"] for r in lr], float)
            vl = np.array([r["sys"]["aegis"]["valid"] for r in lr], bool)
            ent["aegis"]["selective_loso"] = selective(cf, accl, vl)
            ent["aegis"]["ece_loso"] = stats.ece(cf, accl)["ece"]
            ms = cal.fit_platt([[r["conf"]["single"]] for r in lval], [r["sys"]["single"]["acceptable"] for r in lval])
            cs = cal.predict_platt(ms, [[r["conf"]["single"]] for r in lr])
            ent["single"]["selective_loso"] = selective(cs, np.array([r["sys"]["single"]["acceptable"] for r in lr], float))
            ent["single"]["at_aegis_coverage"] = stats.at_coverage(cs, np.array([r["sys"]["single"]["acceptable"] for r in lr], float),
                                                                    max(ent["aegis"]["selective_loso"]["coverage"], 0.01))
        if lval:
            from aegis.agents import psi as psi_fn
            feats_g = ["oov", "sup_frac", "frac_learned", "unknown_med_tokens", "vague"]
            Xv = np.array([[r["gov"][f] for f in feats_g] for r in lval])
            Xt = np.array([[r["gov"][f] for f in feats_g] for r in lr])
            gov = {}
            win_alarm = []
            for k, f in enumerate(feats_g):
                vals = np.sort(Xv[:, k])
                edges = np.unique(vals[(np.arange(1, 10) * len(vals) / 10).astype(int)])
                ref = np.bincount(np.searchsorted(edges, Xv[:, k], side="right"), minlength=len(edges) + 1)
                cur = np.bincount(np.searchsorted(edges, Xt[:, k], side="right"), minlength=len(edges) + 1)
                gov[f] = dict(psi=psi_fn(ref.tolist(), cur.tolist()),
                              smd=float((Xt[:, k].mean() - Xv[:, k].mean()) / max(Xv[:, k].std(), 1e-9)))
                w = []
                for a in range(0, len(Xt) - 600 + 1, 300):
                    c = np.bincount(np.searchsorted(edges, Xt[a:a + 600, k], side="right"), minlength=len(edges) + 1)
                    w.append(psi_fn(ref.tolist(), c.tolist()))
                win_alarm.append(w)
            peak = np.max(np.array(win_alarm), axis=0) if win_alarm and len(win_alarm[0]) else np.array([])
            alarm_share = float(np.mean(peak > 0.25)) if len(peak) else None
            ent["governance"] = dict(features=gov, window_alarm_share=alarm_share,
                                     max_psi=float(max(v["psi"] for v in gov.values())),
                                     auto_coverage_with_governance=0.0 if alarm_share and alarm_share > 0.5 else ent["aegis"]["selective_loso"]["coverage"],
                                     silent_error_with_governance=0.0 if alarm_share and alarm_share > 0.5 else ent["aegis"]["selective_loso"]["silent_error_rate"])
        ent["n"] = len(lr)
        ent["learned_forms"] = len(json.loads((RES / f"learned_loso_{site}.json").read_text())["vocab"])
        ent["learned_headers"] = json.loads((RES / f"learned_loso_{site}.json").read_text())["headers"]
        loso[site] = ent
    T["loso"] = loso
    T["learned"] = dict(n_forms=len(json.loads((RES / "learned_partA.json").read_text())["vocab"]),
                        headers=json.loads((RES / "learned_partA.json").read_text())["headers"])
    T["n"] = dict(val=len(val), test=len(test))
    (TAB / "partA.json").write_text(json.dumps(T, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: T[k] for k in ("selective", "aggregators", "temporal_repair")}, indent=1,
                     default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))[:6000])


if __name__ == "__main__":
    main()
