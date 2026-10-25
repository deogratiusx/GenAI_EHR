#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run --models qwen3.5-4b med-qwen3.5-4b"
$R --conditions domain domain_v2 domain_v3 --split probe3 --out runs/v3_local.jsonl
$R --conditions domain_v3 --split pdf --out runs/v3_local.jsonl
$R --sample 1 --conditions domain_v3 --split test --out runs/v3_local.jsonl
echo "v3 local done"
