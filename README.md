# Aegis-EHR (v2)

Reference implementation, benchmark generator and experiment scripts for

> R. Hammad, A. Alsobeh and A. Shatnawi. *Aegis-EHR: An Orchestrated Multi-Agent Framework for Calibrated,
> Provenance-Preserving Clinical Reconciliation and FHIR Interoperability.* Revised manuscript submitted to the
> International Journal of Advances in Soft Computing and its Applications (IJASCA), 2026.

Every number, table and figure in the revised paper is produced by the scripts in this repository from seeded runs.

## Important note about version 1

The first version of this repository (commit `90590d2` and earlier) did not reproduce the results of the original
submission: its `evaluate.py` wrote fixed values copied from the draft and labelled them `[SIMULATED]`, and the figure
script drew hand-specified curves. Those files have been removed. The revised paper withdraws the original results and
reports only results produced by the code below. The old files remain available in the git history.

## What is here

| Path | Content |
|------|---------|
| `aegis/terminology.py` | Concept inventory (SNOMED CT, ICD-10-CM, RxNorm, LOINC, UCUM) with base, profile-specific and rare surface forms |
| `aegis/knowledge.py` | Reference knowledge base (labels only) and the derived operational knowledge base used by systems |
| `aegis/synth.py` | Benchmark generator: latent state, then gold FHIR bundle and gold safety signals, then note text |
| `aegis/fhir.py` | FHIR R4 builders and a strict validator (12 check types) |
| `aegis/nlp.py` | Sectioning, lexicon, typo-tolerant matching, assertion detection, temporal and dose parsing, vocabulary learning |
| `aegis/agents.py` | Schema, Reasoning, Temporal, Pharmacovigilance and Governance agents, orchestrator, baselines |
| `aegis/axioms.py` | Eleven temporal axioms (T1-T7 used by the agent, T8-T11 held out) |
| `aegis/calibration.py` | Per-agent Platt calibration, six joint-confidence aggregators (noisy-AND default), decision rule |
| `aegis/metrics.py` | CF-F1, SPI, hallucination rate, temporal fidelity, UTR, TCR, FHIR validity, PV metrics, MedBLEU, acceptability |
| `aegis/stats.py` | Bootstrap, paired permutation, McNemar, Holm, effect sizes, ECE, risk-coverage |
| `aegis/llm.py` | Local LLM systems through Ollama (direct, validator loop, reflection loop) |
| `scripts/` | Numbered experiment pipeline (see below) |
| `results/tables/` | All result tables used in the paper (JSON) |
| `results/llm/` | Raw LLM outputs with token counts, latency and log-probabilities, plus run metadata |
| `figures/` | Figures 1-7 of the paper |
| `data/sample/` | 30 test records for a quick look and for the tests |
| `tests/` | Unit and smoke tests |

## Reproducing the paper

Python 3.11 or later. Part A runs on CPU only.

```bash
pip install -r requirements.txt
python -m pytest -q                                   # sanity checks (seconds)
python scripts/01_generate_corpus.py --total 120000 --seed 42   # benchmark, about 1.4 GB, under a minute
PY=python W=12 ./scripts/run_partA_all.sh             # Part A, leave-one-profile-out, analysis, drift, noise
```

`run_partA_all.sh` runs, in order:

1. `scripts/02_run_partA.py` - learns profile vocabulary and headers from training notes, runs the rule-based
   converter, the single-pass pipeline, Aegis-EHR and its ablations on all validation and test notes;
2. the same script with `--holdout Site-A|Site-B|Site-C` - leave-one-profile-out runs;
3. `scripts/04_analyze_partA.py` - calibration, abstention-aware comparison, ablations, aggregators, temporal repair,
   knowledge-base sensitivity, subgroups, failures, MedBLEU validity and statistics (`results/tables/partA.json`);
4. `scripts/05_drift.py` - 1,080 drift streams and 100 controls (`results/tables/drift.json`);
5. `scripts/07_noise_pairwise_latency.py` - token corruption, pairwise ablations and latency.

Part B needs a local [Ollama](https://ollama.com) server with `qwen2.5:7b`, `llama3.2:3b` and `gemma3:12b`:

```bash
ollama serve &                                        # OLLAMA_NUM_PARALLEL=1 was used for the paper
PY=python ./scripts/run_llm_all.sh                    # several hours on an Apple M4 Max; resumable
python scripts/06_analyze_partB.py                    # results/tables/partB.json
python scripts/08_figures.py                          # figures/*.png
```

The raw LLM outputs used in the paper are included in `results/llm/`, so `06_analyze_partB.py` can be run without
repeating generation. Model digests, decoding options and dates are stored in `results/llm/*.meta.json`.

## Design notes

* **Label independence.** The generator samples a latent clinical state, chooses how each fact is expressed, builds
  the gold bundle and gold signals, and only then writes the text. Gold signals come from `reference_kb()`, which no
  system reads; systems use `operational_kb()`.
* **Provenance-preserving temporal repair.** Timestamps are set only to values stated in, or derived by date
  arithmetic from, the note. Vague wording is kept as `onsetString`; conflicting supported values are kept and flagged.
  The earlier closest-valid policy is available as `MeshConfig(temporal_policy="closest_valid")` for comparison.
* **Grounding of external drafts.** Facts in an LLM draft that cannot be matched to any span of the note are dropped.
* **Weakest-link aggregation.** The joint confidence is a weighted noisy-AND of calibrated agent probabilities.

## Limitations

All data are synthetic, the generator and the parser share a terminology and documentation grammar, the knowledge
bases were written by the authors, and no clinician evaluation has been performed. See Section 7.1 of the paper.
