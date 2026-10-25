#!/bin/bash
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.run"
$R --sample 1 --models med-qwen3.5-4b --conditions domain fullctx --split test --out runs/medqwen.jsonl
$R --models qwen3.5-4b med-qwen3.5-4b --conditions domain --split pdf --out runs/qwen_pdf.jsonl
$R --sample 1 --models qwen3.5-4b med-qwen3.5-4b --conditions generic --split test --out runs/medqwen.jsonl
$R --sample 1 --models qwen3.5-4b --conditions domain --split para --out runs/qwen_para.jsonl
$R --models qwen3.5-4b med-qwen3.5-4b --conditions domain --budgets 8 --split pdf_summary --out runs/qwen_pdf.jsonl
$R --sample 1 --models med-qwen3.5-4b qwen3.5-4b --conditions domain fullctx --budgets 8 --split summary --out runs/qwen_summary.jsonl
$R --models qwen3.5-4b --conditions domain_v2 --split pdf --out runs/qwen_pdf.jsonl
echo "local queue v2 done"
