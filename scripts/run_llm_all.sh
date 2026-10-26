#!/bin/bash
# Runs every LLM system used in the paper, in order. Each step resumes if interrupted.
set -u
PY=${PY:-python}
cd "$(dirname "$0")/.."
$PY scripts/03_run_llm.py --model qwen2.5:7b --variant direct
$PY scripts/03_run_llm.py --model llama3.2:3b --variant direct
$PY scripts/03_run_llm.py --model qwen2.5:7b --variant validator
$PY scripts/03_run_llm.py --model qwen2.5:7b --variant reflect
$PY scripts/03_run_llm.py --model gemma3:12b --variant direct
