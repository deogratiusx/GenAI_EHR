#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run --models GigaChat-2"
$R --conditions domain_v3r --split probe3 --out runs/v3_giga.jsonl
$R --conditions domain_v3r --split pdf --out runs/v3_giga.jsonl
$R --sample 1 --conditions domain_v3r --split test --out runs/v3_giga.jsonl
$R --conditions domain_v3r --split pdf --out runs/v3_giga.jsonl
echo "v3r giga done"
