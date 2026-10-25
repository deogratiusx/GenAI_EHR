#!/bin/bash
# Final run, local models on Apple Silicon (MLX). Long-context baselines on a reduced subset:
# a whole check-up takes ~1.5 min per answer, a map-reduce pass over a PDF ~4 min.
cd "$(dirname "$0")/.."
export EHR_REF_POLICY=layered HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.eval.run --sample 1"
POINT="test para probe probe3 probe4 pdf trend"
for M in qwen3.5-4b med-qwen3.5-4b; do
  OUT=runs/final_${M}.jsonl
  for S in $POINT summary pdf_summary report; do $R --models $M --systems tools --split $S --out $OUT; done
  for S in summary pdf_summary report; do $R --models $M --systems report --split $S --out $OUT; done
done
OUT=runs/final_qwen3.5-4b.jsonl
for S in $POINT; do $R --models qwen3.5-4b --systems code --split $S --out $OUT; done
$R --models qwen3.5-4b --systems record --split test --out $OUT
$R --models qwen3.5-4b --systems record_chunked --split pdf --ids K01-01 K02-01 K03-01 K04-01 K05-01 K06-01 --out $OUT
echo FINAL_LOCAL_DONE
