#!/bin/bash
# Starts after queue_giga.sh: age/sex-aware intervals (domain_ri), sparse.
cd "$(dirname "$0")/.."
until grep -q "giga queue done" runs/queue_giga.log 2>/dev/null; do sleep 30; done
R=".venv/bin/python -m ehr_agent.run --sample 1 --models GigaChat-2-Max GigaChat-2"
$R --conditions domain_ri --split test --out runs/giga_ri.jsonl
$R --conditions domain_ri --budgets 8 --split summary --out runs/giga_summary.jsonl
echo "giga queue 2 done"
