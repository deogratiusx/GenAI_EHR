#!/bin/bash
# Second source (PDF case histories). Starts after queue_giga2.sh.
cd "$(dirname "$0")/.."
until grep -q "giga queue 2 done" runs/queue_giga2.log 2>/dev/null; do sleep 30; done
R=".venv/bin/python -m ehr_agent.run --models GigaChat-2-Max GigaChat-2"
$R --conditions domain domain_v2 generic fullctx --split pdf --out runs/giga_pdf.jsonl
$R --conditions domain domain_v2 fullctx --budgets 8 --split pdf_summary --out runs/giga_pdf.jsonl
$R --conditions domain domain_v2 generic fullctx --split pdf --out runs/giga_pdf.jsonl
echo "giga queue 3 done"
