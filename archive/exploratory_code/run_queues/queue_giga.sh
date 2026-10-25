#!/bin/bash
# Sparse exploratory runs (1 question per patient x category). Resumable.
cd "$(dirname "$0")/.."
R=".venv/bin/python -m ehr_agent.run --sample 1 --models GigaChat-2-Max GigaChat-2"
$R --conditions domain_v2 --split test --out runs/giga_v2.jsonl
$R --conditions domain fullctx hybrid --budgets 8 --split summary --out runs/giga_summary.jsonl
$R --conditions domain generic fullctx --split para --out runs/giga_para.jsonl
$R --conditions hybrid --split test --out runs/giga_hybrid.jsonl
$R --conditions domain domain_v2 generic fullctx --split probe --out runs/giga_probe.jsonl
$R --conditions domain --budgets 1 2 8 --split test --out runs/giga_budget.jsonl
echo "giga queue done"
