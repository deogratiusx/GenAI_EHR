#!/bin/bash
# Priority: contrasts by source (table vs PDF) for Max and Lite; ablations after.
cd "$(dirname "$0")/.."
R=".venv/bin/python -m ehr_agent.run --models GigaChat-2-Max GigaChat-2"
$R --conditions domain domain_v2 generic fullctx --split pdf --out runs/giga_pdf.jsonl
$R --conditions domain domain_v2 fullctx --budgets 8 --split pdf_summary --out runs/giga_pdf.jsonl
$R --conditions domain domain_v2 generic fullctx --split pdf --out runs/giga_pdf.jsonl
$R --sample 1 --conditions domain_ri --split test --out runs/giga_ri.jsonl
$R --sample 1 --conditions domain --budgets 1 2 8 --split test --out runs/giga_budget.jsonl
echo "giga queue v2 done"
