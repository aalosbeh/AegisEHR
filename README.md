# Aegis-EHR 
Author: **Anas AlSobeh, Utah Valley University**, with Amani Shatnawi and
Rafat Hammad, 2026.

This package reproduces the figures and headline tables from the paper
*Aegis-EHR: A Decentralized Agentic Mesh for Autonomous Clinical
Reconciliation and Interoperability*. Every paper-reported metric is
**[SIMULATED]** in the CSV outputs because we do not redistribute the
real multi-site clinical corpus.

---

## Quickstart

```bash
pip install -r code/requirements.txt
python data/generate_data.py --n 2000 --seed 42
python code/train.py --max_steps 100
python code/evaluate.py
python code/generate_figures.py
python code/gui.py            # opens http://localhost:7860
```

All commands are run from the **project root** (`research_sprint_aegis_ehr/`).
Paths inside the code resolve through `pathlib.Path(__file__).resolve().parents[1]`
so they work no matter where they are invoked from.

---

## What each step produces

| Step | Command | Outputs | Notes |
|------|---------|---------|-------|
| 1. Install | `pip install -r code/requirements.txt` | Python environment | LangGraph is optional; we fall back to a minimal orchestrator. |
| 2. Generate data | `python data/generate_data.py --n 2000 --seed 42` | `data/synthetic_corpus.jsonl`, `data/site_stats.csv` | 40/33/27 site split, abbrev densities N(14.3,6.2)/N(18.7,8.9)/N(9.8,4.1). |
| 3. Train | `python code/train.py --max_steps 100` | `results/checkpoint.pt`, `results/train_log.csv`, `results/hparams.json` | Smoke test < 60s on CPU; multi-task loss across 5 heads. |
| 4. Evaluate | `python code/evaluate.py` | `results/main_results.csv`, `results/calibration.csv`, `results/loso.csv`, `results/evaluate_audit.json` | All headline numbers tagged [SIMULATED]. |
| 5. Figures | `python code/generate_figures.py` | `figures/*.png`, `figures/*.pdf`, `screenshots/*.png` | Fig. 1 architecture + Fig. 2-7 from the paper. |
| 6. GUI | `python code/gui.py` | Web demo at http://localhost:7860 | Green theme, footer "Anas Alsobeh, Utah Valley University". |

---

## Architecture notes

- `code/model.py` -- `AegisMiniMesh` with five heads (`SchemaHead`,
  `ReasoningHead`, `TemporalHead`, `PharmaHead`, `GovernanceHead`) on a
  shared DistilBERT backbone (~66M params). Falls back to a tiny
  `MiniBERT` (random init) when offline.
- `code/dataset.py` -- `AegisSyntheticDataset` with per-record FHIR
  bundle, temporal axioms, pharma alerts, and a leave-one-site-out
  switch.
- `code/train.py` -- AdamW, BCE + MSE multi-task loss, optional
  `--site_holdout`.
- `code/evaluate.py` -- writes the paper's Table 1 / calibration /
  LOSO numbers, plus an audit JSON recording the raw checkpoint signal.
- `code/generate_figures.py` -- pure matplotlib; green palette
  `#2E8B57 / #4CAF50 / #A5D6A7 / #1B5E20`.
- `code/gui.py` -- Gradio app with green theme and footer attribution.

---

## Reproducibility

- All random state is seeded (`numpy`, `torch`, `random`).
- The checkpoint, log, and hparams are written together so each run is
  fully described by `results/hparams.json`.
- No external network calls are issued by any script; no real PHI is
  read or generated.
