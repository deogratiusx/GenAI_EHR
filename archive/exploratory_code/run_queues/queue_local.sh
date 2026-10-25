#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run --sample 1"
$R --models qwen3.5-4b --conditions domain generic fullctx --split test --out runs/qwen.jsonl
$R --models qwen3.5-4b --conditions domain fullctx --budgets 8 --split summary --out runs/qwen_summary.jsonl
$R --models qwen3.5-4b --conditions domain --split para --out runs/qwen_para.jsonl
while [ ! -f models/Med-Qwen-3.5-4B-it-4bit/model.safetensors.index.json ] && [ ! -f models/Med-Qwen-3.5-4B-it-4bit/model.safetensors ]; do sleep 30; done
sleep 30
$R --models med-qwen3.5-4b --conditions domain fullctx --split test --out runs/medqwen.jsonl
$R --models med-qwen3.5-4b --conditions domain fullctx --budgets 8 --split summary --out runs/qwen_summary.jsonl
$R --models qwen3.5-4b --conditions domain_v2 --split test --out runs/qwen_v2.jsonl
echo "local queue done"
