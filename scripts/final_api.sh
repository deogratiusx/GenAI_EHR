#!/bin/bash
# Final run, cloud models (GigaChat allows one concurrent request: Max first, then Lite).
# Sparse stratified sample (1 question per split × patient × category), layered reference policy.
cd "$(dirname "$0")/.."
export EHR_REF_POLICY=layered HF_HUB_OFFLINE=1
R=".venv/bin/python -m ehr_agent.eval.run --sample 1"
POINT="test para probe probe3 probe4 pdf trend"
SUMMARY="summary pdf_summary report"
for M in GigaChat-2-Max GigaChat-2; do
  OUT=runs/final_${M}.jsonl
  SYS="tools tools_routed tools_plain code record"
  [ "$M" = GigaChat-2 ] && SYS="$SYS tools_compact tools_prefetch"  # tool-set ablations on the weak model
  for S in $POINT; do $R --models $M --systems $SYS --split $S --out $OUT; done
  for S in pdf trend; do $R --models $M --systems record_chunked --split $S --out $OUT; done
  $R --models $M --systems tools report report_nocheck record --split summary --out $OUT
  for S in pdf_summary report; do $R --models $M --systems tools report report_nocheck record_chunked --split $S --out $OUT; done
done
# Run-to-run variance: three runs at temperature 0.5 of the main system on the table test set.
for M in GigaChat-2-Max GigaChat-2; do
  $R --models $M --systems tools --split test --reps 3 --temperature 0.5 --out runs/final_variance.jsonl
done
echo FINAL_API_DONE
