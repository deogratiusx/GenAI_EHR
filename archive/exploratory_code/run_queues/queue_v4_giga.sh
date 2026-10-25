#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run --models GigaChat-2"
$R --conditions domain domain_v3 domain_v4a domain_v4 --split probe4 --out runs/v4_giga.jsonl
$R --conditions domain_v4a domain_v4 --split pdf --out runs/v4_giga.jsonl
$R --conditions domain_v4a domain_v4 --split probe3 --out runs/v4_giga.jsonl
$R --sample 1 --conditions domain_v4a domain_v4 --split test --out runs/v4_giga.jsonl
$R --conditions domain domain_v3 domain_v4a domain_v4 --split probe4 --out runs/v4_giga.jsonl
$R --conditions domain_v4a domain_v4 --split pdf --out runs/v4_giga.jsonl
echo "v4 giga done"
