"""Generate the Aegis-EHR synthetic benchmark (train/val/test JSONL + corpus statistics).

Usage:  python scripts/01_generate_corpus.py --total 120000 --seed 42
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aegis.synth import PROFILES, Generator, make_record  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=120000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "corpus"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gen = Generator(args.seed)
    files = {s: open(out / f"{s}.jsonl", "w", encoding="utf-8") for s in ("train", "val", "test")}
    stats = {}
    for key, prof in PROFILES.items():
        n = int(round(args.total * prof.share))
        n_train, n_val = int(0.8 * n), int(0.1 * n)
        st = dict(records=n, abbr_mentions=0, mentions=0, tokens=0, temporal_vague=0, conflict=0,
                  signal_records=0, allergy=0, meds=0, high_complexity=0, typo=0, site_forms=0)
        for i in range(n):
            r = make_record(gen, i, prof)
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            r["split"] = split
            files[split].write(json.dumps(r) + "\n")
            m = r["meta"]
            st["mentions"] += len(m["mentions"])
            st["abbr_mentions"] += sum(v for k, v in m["tiers"].items() if k.startswith("abbr") or k.startswith("ambiguous"))
            st["typo"] += sum(v for k, v in m["tiers"].items() if "typo" in k)
            st["site_forms"] += sum(v for k, v in m["tiers"].items() if k.startswith("site") or k.startswith("rare"))
            st["tokens"] += m["n_tokens"]
            st["temporal_vague"] += bool(m["temporal_vague"])
            st["conflict"] += bool(m["conflict"])
            st["signal_records"] += bool(r["gold_signals"])
            st["allergy"] += m["n_allergies"] > 0
            st["meds"] += m["n_meds"] > 0
            st["high_complexity"] += (m["n_conditions"] >= 5 or m["n_meds"] >= 8 or m["temporal_vague"] >= 2)
        stats[prof.name] = st
        print(prof.name, n)
    for f in files.values():
        f.close()
    (out / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
