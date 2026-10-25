#!/bin/bash
# Tools v3 (T1+T2+T8) vs v1/v2 on GigaChat Lite. Sparse.
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run --models GigaChat-2"
$R --conditions domain domain_v2 domain_v3 --split probe3 --out runs/v3_giga.jsonl
$R --sample 1 --conditions domain_v3 --split test --out runs/v3_giga.jsonl
$R --conditions domain_v3 --split pdf --out runs/v3_giga.jsonl
$R --conditions domain domain_v2 domain_v3 --split probe3 --out runs/v3_giga.jsonl
$R --conditions domain_v3 --split pdf --out runs/v3_giga.jsonl
echo "v3 giga done"
