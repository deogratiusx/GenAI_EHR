#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
until grep -q "local queue done" runs/queue_local.log 2>/dev/null; do sleep 30; done
R=".venv/bin/python -m ehr_agent.run"
$R --models qwen3.5-4b --conditions domain domain_v2 --split pdf --out runs/qwen_pdf.jsonl
$R --models med-qwen3.5-4b --conditions domain --split pdf --out runs/qwen_pdf.jsonl
echo "local queue 2 done"
