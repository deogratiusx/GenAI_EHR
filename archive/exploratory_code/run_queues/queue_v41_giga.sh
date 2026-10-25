#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run --models GigaChat-2"
$R --conditions domain_v41a domain_v41 --split probe4 --out runs/v4_giga.jsonl
$R --conditions domain_v41a domain_v41 --split pdf --out runs/v4_giga.jsonl
$R --conditions domain_v41a domain_v41 --split probe3 --out runs/v4_giga.jsonl
$R --sample 1 --conditions domain_v41a domain_v41 --split test --out runs/v4_giga.jsonl
$R --conditions domain_v41a domain_v41 --split pdf --out runs/v4_giga.jsonl
echo "v41 giga done"
