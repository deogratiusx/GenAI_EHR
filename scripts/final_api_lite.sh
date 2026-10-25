#!/bin/bash
# Reduced final run for GigaChat Lite (time-boxed before the defense): the sets and systems the
# pre-registered hypotheses H1–H2 need; tool-set ablations and the variance runs are left out.
cd "$(dirname "$0")/.."
export EHR_REF_POLICY=layered HF_HUB_OFFLINE=1
while pgrep -f "eval.run --sample 1 --models GigaChat-2-Max" >/dev/null; do sleep 5; done
R=".venv/bin/python -m ehr_agent.eval.run --sample 1"
OUT=runs/final_GigaChat-2.jsonl
for S in test pdf trend; do $R --models GigaChat-2 --systems tools tools_plain code record --split $S --out $OUT; done
for S in pdf trend; do $R --models GigaChat-2 --systems record_chunked --split $S --out $OUT; done
$R --models GigaChat-2 --systems tools report report_nocheck --split summary --out $OUT
for S in pdf_summary report; do $R --models GigaChat-2 --systems tools report report_nocheck --split $S --out $OUT; done
# retry pass for backend errors of both cloud models (resumable: only missing episodes run)
for S in test para probe probe3 probe4 pdf trend; do $R --models GigaChat-2-Max --systems tools tools_routed tools_plain code record --split $S --out runs/final_GigaChat-2-Max.jsonl; done
for S in test pdf trend; do $R --models GigaChat-2 --systems tools tools_plain code record --split $S --out $OUT; done
echo FINAL_LITE_DONE
