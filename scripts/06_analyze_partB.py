"""Part B: LLM baselines and Aegis-EHR on LLM drafts (stratified test subset).

For every model with a ``direct`` run, Aegis-EHR receives the model's own
bundle as the Schema Agent's draft (no additional LLM calls). Calibration of
all confidences (LLM log-probability scores and Aegis agent confidences) uses
2-fold cross-fitting stratified by profile.

Usage: python scripts/06_analyze_partB.py
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis import calibration as cal  # noqa: E402
from aegis import stats  # noqa: E402
from aegis.agents import Mesh, MeshConfig, PharmacovigilanceAgent, feature_vector  # noqa: E402
from aegis.fhir import validate_bundle  # noqa: E402
from aegis.knowledge import operational_kb  # noqa: E402
from aegis.knowledge import fire_rules  # noqa: E402
from aegis.metrics import alerts_from_detected_issues, bundle_facts, evaluate_record, state_from_facts  # noqa: E402
from aegis.nlp import BASE_HEADERS, build_lexicon, parse_note  # noqa: E402
from aegis.synth import read_jsonl  # noqa: E402

RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figdata"
LLM = RES / "llm"
SEM = {"hallucinated_condition", "hallucinated_drug", "hallucinated_allergen", "missed_drug", "missed_allergen",
       "drug_status", "dose_error", "unknown_code"}
ABL = {"aegis": MeshConfig(), "schema_only": MeshConfig(use_reasoning=False, use_temporal=False),
       "schema_reasoning": MeshConfig(use_temporal=False), "closest_valid": MeshConfig(temporal_policy="closest_valid")}
TAU = 0.70


def labels(e):
    return dict(schema_ok=bool(e["valid"]), semantic_ok=not any(k in SEM for k in e["critical"]),
                temporal_ok=not e["critical"].get("fabricated_timestamp") and not e["major"].get("wrong_timestamp"),
                pv_ok=e["pv_fn"] == 0, acceptable=bool(e["acceptable"]))


def crossfit_platt(x, y, folds):
    out = np.zeros(len(x))
    for f in set(folds):
        tr = [i for i in range(len(x)) if folds[i] != f]
        te = [i for i in range(len(x)) if folds[i] == f]
        m = cal.fit_platt([[x[i]] for i in tr], [y[i] for i in tr])
        out[te] = cal.predict_platt(m, [[x[i]] for i in te])
    return out


def crossfit_aegis(F, L, folds, valid, kind="noisy_and"):
    n = len(valid)
    sig_all = {a: np.zeros(n) for a in F}
    joint = {k: np.zeros(n) for k in cal.AGGREGATORS}
    for f in set(folds):
        tr = [i for i in range(n) if folds[i] != f]
        te = [i for i in range(n) if folds[i] == f]
        Ftr = {a: [F[a][i] for i in tr] for a in F}
        Fte = {a: [F[a][i] for i in te] for a in F}
        Ltr = {k: [L[k][i] for i in tr] for k in L}
        models = cal.fit_agents(Ftr, Ltr)
        str_, ste = cal.predict_agents(models, Ftr), cal.predict_agents(models, Fte)
        for a in ste:
            sig_all[a][te] = ste[a]
        for k in cal.AGGREGATORS:
            ag = cal.Aggregator(k).fit(str_, Ltr["acceptable"])
            joint[k][te] = ag(ste)
    return sig_all, joint


def summarize(E, conf=None, valid=None, n_boot=2000):
    out = {}
    for k in ("cf1", "spi", "halluc", "tf", "medbleu", "acceptable", "valid", "swes", "precision", "recall"):
        out[k] = list(stats.boot_ci([float(e[k]) for e in E], n_boot))
    out["utr"] = list(stats.boot_ratio_ci([e["fabricated"] for e in E], [e["precise_pred"] for e in E], n_boot))
    out["tcr"] = list(stats.boot_ratio_ci([e["tcr_sat"] for e in E], [e["tcr_n"] for e in E], n_boot))
    out["tcr_held"] = list(stats.boot_ratio_ci([e["tcr_held_sat"] for e in E], [e["tcr_held_n"] for e in E], n_boot))
    tp = sum(e["pv_tp"] for e in E)
    fn = sum(e["pv_fn"] for e in E)
    al = sum(e["pv_alerts"] for e in E)
    fp = sum(e["pv_fp"] for e in E)
    rfp = sum(1 for e in E if not e["pv_gold_pos"] and e["pv_pred_pos"])
    rtn = sum(1 for e in E if not e["pv_gold_pos"] and not e["pv_pred_pos"])
    rtp = sum(1 for e in E if e["pv_gold_pos"] and e["pv_pred_pos"])
    rfn = sum(1 for e in E if e["pv_gold_pos"] and not e["pv_pred_pos"])
    out.update(pv_sensitivity=tp / max(tp + fn, 1), pv_ppv=(al - fp) / max(al, 1),
               pv_specificity=rtn / max(rtn + rfp, 1), pv_fpr=rfp / max(rtn + rfp, 1),
               alerts_per_1000=1000 * al / len(E), pv_mcc=stats.mcc(rtp, rfp, rtn, rfn))
    if conf is not None:
        conf = np.asarray(conf, float)
        acc = np.array([e["acceptable"] for e in E], float)
        v = np.ones(len(E), bool) if valid is None else np.asarray(valid, bool)
        auto = (conf >= TAU) & v
        rc = stats.risk_coverage(conf * v, acc)
        c = stats.ece(conf, acc)
        out.update(coverage=float(auto.mean()), selective_acc=float(acc[auto].mean()) if auto.any() else None,
                   silent_error_rate=float(((1 - acc) * auto).mean()), automation_yield=float((acc * auto).mean()),
                   aurc=rc["aurc"], ece=c["ece"], brier=c["brier"], mce=c["mce"],
                   at80=stats.at_coverage(conf * v, acc, 0.8), at90=stats.at_coverage(conf * v, acc, 0.9))
    return out


def main():
    ids = json.loads((ROOT / "data" / "llm_subset.json").read_text())
    want = set(ids)
    recs = {r["id"]: r for r in read_jsonl(ROOT / "data" / "corpus" / "test.jsonl") if r["id"] in want}
    L = json.loads((RES / "learned_partA.json").read_text())
    headers = dict(BASE_HEADERS)
    headers.update(L["headers"])
    lex = build_lexicon("schema", [tuple(v) for v in L["vocab"]])
    kb = operational_kb()
    mesh = Mesh(lex, lex, headers, kb)
    partA = {}
    for line in open(RES / "partA_records.jsonl"):
        r = json.loads(line)
        if r["id"] in want:
            partA[r["id"]] = r
    runs = {}
    for p in sorted(LLM.glob("*.jsonl")):
        tag = p.stem
        runs[tag] = {d["id"]: d for d in map(json.loads, p.open())}
    # a model enters only once its direct run covers the whole subset; partial runs are reported, not analysed
    models = sorted({t.rsplit("_", 1)[0] for t in runs if t.endswith("_direct")
                     and len(set(runs[t]) & set(ids)) >= len(ids) * 0.95})
    for t in sorted(runs):
        if t.endswith("_direct") and t.rsplit("_", 1)[0] not in models:
            print(f"skipping incomplete run {t}: {len(runs[t])}/{len(ids)}")
    common = [i for i in ids if all(i in runs[f"{m}_direct"] for m in models)]
    for v in ("validator", "reflect"):
        for m in models:
            if f"{m}_{v}" in runs and len(runs[f"{m}_{v}"]) >= len(common) * 0.95:
                common = [i for i in common if i in runs[f"{m}_{v}"]]
    print(f"models {models}; records with all runs: {len(common)}")
    folds = [0 if (k % 2 == 0) else 1 for k, _ in enumerate(common)]  # ids are sorted within site blocks
    T = dict(n=len(common), models=models, systems={}, fhir={}, temporal={}, tests={}, ablation={}, meta={})
    per = defaultdict(dict)   # system -> id -> eval
    conf = defaultdict(dict)
    valid_final = defaultdict(dict)
    feats = defaultdict(lambda: defaultdict(list))
    labs = defaultdict(lambda: defaultdict(list))
    fhir_tab = {}
    temporal = defaultdict(Counter)
    actions = defaultdict(Counter)
    for name in ("rule", "single", "aegis"):
        for i in common:
            per[f"det_{name}"][i] = partA[i]["sys"][name]
    for m in models:
        for variant in ("direct", "validator", "reflect"):
            tag = f"{m}_{variant}"
            if tag not in runs or not all(i in runs[tag] for i in common):
                continue
            errs = Counter()
            n_valid = n_parsed = 0
            for i in common:
                d = runs[tag][i]
                b = d.get("bundle")
                rec = recs[i]
                v = validate_bundle(b) if b is not None else {"valid": False, "errors": [], "by_kind": {"unparseable_json": 1}}
                n_valid += v["valid"]
                n_parsed += b is not None
                errs.update(v["by_kind"])
                F = bundle_facts(b)
                alerts = alerts_from_detected_issues(b, F) if b is not None else []
                e = evaluate_record(rec, b, alerts, v)
                per[f"{tag}"][i] = e
                lp = d.get("mean_logprob")
                conf[tag][i] = math.exp(lp) if lp is not None and b is not None else 0.0
                valid_final[tag][i] = True
                if variant == "direct":
                    kb_alerts = [frozenset(x["components"]) for x in fire_rules(state_from_facts(F), kb)] if b is not None else []
                    per[f"{tag}+kb"][i] = evaluate_record(rec, b, kb_alerts, v)
            fhir_tab[tag] = dict(parsed=n_parsed / len(common), valid=n_valid / len(common), errors=dict(errs),
                                 mean_output_tokens=float(np.mean([runs[tag][i]["output_tokens"] for i in common])),
                                 mean_calls=float(np.mean([runs[tag][i].get("n_calls", 1) for i in common])),
                                 mean_wall_s=float(np.mean([runs[tag][i]["wall"] for i in common])),
                                 median_wall_s=float(np.median([runs[tag][i]["wall"] for i in common])),
                                 tokens_per_s=float(np.sum([runs[tag][i]["output_tokens"] for i in common]) /
                                                    np.sum([runs[tag][i]["wall"] for i in common])),
                                 unparsed_length_cutoff=sum(runs[tag][i].get("bundle") is None and
                                                            runs[tag][i].get("done_reason") == "length" for i in common),
                                 unparsed_whitespace_runaway=sum(runs[tag][i].get("bundle") is None and
                                                                 len(runs[tag][i].get("raw") or "") -
                                                                 len((runs[tag][i].get("raw") or "").rstrip()) >= 20
                                                                 for i in common))
        # Aegis-EHR on this model's drafts --------------------------------------------
        for an, cfg in ABL.items():
            sname = f"{m}_aegis" if an == "aegis" else f"{m}_aegis_{an}"
            pre_valid = repaired_valid = abstain_schema = dropped = 0
            for i in common:
                d = runs[f"{m}_direct"][i]
                rec = recs[i]
                F = parse_note(rec["text"], lex, "full", headers)
                if d.get("bundle") is None:
                    # unparsable LLM output: the gate withholds it; no silent fallback to Aegis-EHR's own parser
                    o = dict(bundle=None, alerts=[], reports={}, pre_valid=False, post_valid=False)
                else:
                    o = mesh.run(rec["text"], cfg, draft=d["bundle"], F=F)
                if o["bundle"] is None:
                    e = evaluate_record(rec, {"resourceType": "Bundle", "type": "collection", "entry": []}, [],
                                        {"valid": False, "errors": [], "by_kind": {}})
                    abstain_schema += 1
                    fv = None
                else:
                    e = evaluate_record(rec, o["bundle"], o["alerts"])
                    pre_valid += bool(o["pre_valid"])
                    repaired_valid += bool(o["post_valid"])
                    abstain_schema += not o["post_valid"]
                    dropped += sum(1 for a in o["reports"]["S"].actions if a[0] == "drop_resource")
                    fv = {a: feature_vector(o["reports"], a) for a in ("S", "R", "T", "P")}
                per[sname][i] = e
                valid_final[sname][i] = o["bundle"] is not None and o["post_valid"]
                if an in ("aegis", "closest_valid") and o["bundle"] is not None:
                    temporal[sname].update(o["reports"]["T"].stats)
                if an == "aegis" and o["bundle"] is not None:
                    gold = bundle_facts(rec["gold"])
                    for act, detail in o["reports"]["R"].actions:
                        actions[sname][act] += 1
                        if act == "drop_ungrounded":
                            rt, key = detail.split(" ", 1)
                            kind = {"Condition": "condition", "MedicationRequest": "drug",
                                    "AllergyIntolerance": "allergen"}.get(rt)
                            actions[sname]["drop_ungrounded_correct"] += int(kind is not None and key in gold[kind])
                if an == "aegis":
                    for a in ("S", "R", "T", "P"):
                        width = {"S": 6, "R": 14, "T": 8, "P": 5}[a]
                        feats[sname][a].append(fv[a] if fv and fv[a] else [0.0] * width)
                    for k, val in labels(e).items():
                        labs[sname][k].append(val)
            if an == "aegis":
                fhir_tab[sname] = dict(pre_gate_valid=pre_valid / len(common), post_repair_valid=repaired_valid / len(common),
                                       schema_abstentions=abstain_schema / len(common),
                                       dropped_resources_per_record=dropped / len(common))
    # confidences ---------------------------------------------------------------------------
    acc_of = lambda s: [per[s][i]["acceptable"] for i in common]
    final_conf = {}
    agg_tab = {}
    for s in list(per):
        if s.startswith("det_"):
            base = s[4:]
            if base == "aegis":
                pc = partA
                import pickle
                C = pickle.load(open(RES / "calibrators_partA.pkl", "rb"))["aegis"]
                Fv = {a: [pc[i]["features"]["aegis"][a] for i in common] for a in cal.AGENTS}
                sig = cal.predict_agents(C["models"], Fv)
                final_conf[s] = C["aggs"]["noisy_and"](sig)
            else:
                import pickle
                m_ = pickle.load(open(RES / "calibrators_partA.pkl", "rb"))[base]
                final_conf[s] = cal.predict_platt(m_, [[partA[i]["conf"][base]] for i in common])
        elif s.endswith("_aegis"):
            sig, joint = crossfit_aegis(feats[s], labs[s], folds, [valid_final[s][i] for i in common])
            final_conf[s] = joint["noisy_and"]
            agg_tab[s] = {}
            acc = np.array(acc_of(s), float)
            v = np.array([valid_final[s][i] for i in common], bool)
            for k in cal.AGGREGATORS:
                c = stats.ece(joint[k], acc)
                rc = stats.risk_coverage(joint[k] * v, acc)
                auto = (joint[k] >= TAU) & v
                agg_tab[s][k] = dict(ece=c["ece"], brier=c["brier"], aurc=rc["aurc"], coverage=float(auto.mean()),
                                     selective_acc=float(acc[auto].mean()) if auto.any() else None,
                                     silent_error_rate=float(((1 - acc) * auto).mean()))
            final_conf[s + "_uncalibrated"] = None
        elif s.split("_")[-1] in ("direct", "validator", "reflect"):
            x = [conf[s][i] for i in common]
            final_conf[s] = crossfit_platt(x, acc_of(s), folds)
            final_conf[s + "_raw"] = np.array(x)
    for s in per:
        c = final_conf.get(s)
        v = [valid_final[s].get(i, per[s][i]["valid"]) for i in common] if s.endswith("_aegis") else None
        T["systems"][s] = summarize([per[s][i] for i in common], c, v)
        if final_conf.get(s + "_raw") is not None:
            cr = stats.ece(final_conf[s + "_raw"], np.array(acc_of(s), float))
            T["systems"][s]["ece_raw"] = cr["ece"]
    T["fhir"] = fhir_tab
    T["temporal"] = {k: dict(v) for k, v in temporal.items()}
    T["reasoning_actions"] = {k: dict(v) for k, v in actions.items()}
    T["critical_counts"] = {sname: dict(sum((Counter(per[sname][i]["critical"]) for i in common), Counter()))
                            for sname in per}
    T["aggregators"] = agg_tab
    # curves for figures
    curves = {}
    for s in per:
        if final_conf.get(s) is None:
            continue
        acc = np.array(acc_of(s), float)
        rc = stats.risk_coverage(np.asarray(final_conf[s]), acc)
        curves[s] = dict(coverage=rc["coverage"].tolist(), risk=rc["risk"].tolist(), aurc=rc["aurc"],
                         bins=stats.ece(final_conf[s], acc)["bins"])
    (FIG / "partB_curves.json").write_text(json.dumps(curves))
    # tests: Aegis-LLM vs each single-model variant of the same model ---------------------------
    for m in models:
        pv, ef = {}, {}
        target = f"{m}_aegis"
        for base in (f"{m}_direct", f"{m}_validator", f"{m}_reflect"):
            if base not in per:
                continue
            for k in ("cf1", "spi", "tf", "medbleu"):
                a = [per[target][i][k] for i in common]
                b = [per[base][i][k] for i in common]
                key = f"{target}_vs_{base}:{k}"
                pv[key] = stats.paired_permutation(a, b, 10000)
                ef[key] = dict(mean_diff=float(np.mean(a) - np.mean(b)), cohens_dz=stats.cohens_dz(a, b),
                               cliffs_delta=stats.cliffs_delta(a, b))
            for k in ("acceptable", "valid"):
                a = [per[target][i][k] for i in common]
                b = [per[base][i][k] for i in common]
                key = f"{target}_vs_{base}:{k}"
                p, n10, n01 = stats.mcnemar(a, b)
                pv[key] = p
                ef[key] = dict(risk_difference=float(np.mean(a) - np.mean(b)), n_aegis_only=n10, n_baseline_only=n01)
        adj = stats.holm(pv)
        T["tests"][m] = {k: dict(p=pv[k], p_holm=adj[k], **ef[k]) for k in pv}
        T["tests"][m]["_family"] = f"{len(pv)} comparisons for {m}: Holm-Bonferroni within model"
    T["meta"] = {p.stem: json.loads(p.read_text()) for p in LLM.glob("*.meta.json")
                 if p.stem.replace(".meta", "").rsplit("_", 1)[0] in models}
    T["records"] = common
    (TAB / "partB.json").write_text(json.dumps(T, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    for s, d in T["systems"].items():
        print(f"{s:34s} valid {d['valid'][0]:.3f} CF1 {d['cf1'][0]:.3f} SPI {d['spi'][0]:.3f} hall {d['halluc'][0]:.3f} TF {d['tf'][0]:.3f} "
              f"UTR {d['utr'][0]:.3f} acc {d['acceptable'][0]:.3f} BLEU {d['medbleu'][0]:.3f} PVsens {d['pv_sensitivity']:.3f} "
              f"PPV {d['pv_ppv']:.3f} cov {d.get('coverage', float('nan')):.3f} sel {d.get('selective_acc') or float('nan'):.3f} "
              f"silent {d.get('silent_error_rate', float('nan')):.3f} ECE {d.get('ece', float('nan')):.3f}")
    print(json.dumps(T["fhir"], indent=1))
    print(json.dumps(T["temporal"], indent=1))


if __name__ == "__main__":
    main()
