#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run --models GigaChat-2"
$R --conditions domain_v3 domain_v3p --split probe3 --out runs/v3_giga.jsonl
$R --sample 1 --conditions domain_v3 domain_v3p --split test --out runs/v3_giga.jsonl
$R --conditions domain_v3 domain_v3p --split pdf --out runs/v3_giga.jsonl
echo "v3p giga done"
