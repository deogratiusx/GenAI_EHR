# Set of tools for EHR LLM-assisted analytics

*Course project — Modern Methods and Algorithms of Generative AI. Data: anonymized check-up records from a partner private clinic (AI-mnesis).*

## 1. Problem and hypothesis

A private clinic asked for a tool that lets doctors quickly summarize a check-up record, query specific
parameters and reconcile conclusions from many specialists ("simulate a case conference"). Frontier
assistants already do this with code execution and long context, but three constraints rule them out in
Russian healthcare: patient data should not leave the RF (strictly: the clinic), clinical answers need
traceable accuracy, and self-hosting a frontier-size model costs more than the product earns.

The deployable options are an RF-hosted API (GigaChat-2 Lite / Pro / Max) or a small open model on the
clinic's own hardware (Qwen3.5-4B, 4-bit, run here on a laptop with MLX). The research question is whether
**tool design** can close the gap that model size opens.

**H1.** Narrow, typed, medically-aware tools let a small or RF-hosted LLM answer doctors' questions about an
EHR more accurately and with better grounding than the two approaches general-purpose assistants use:
(a) a code interpreter over the raw table and (b) the whole record pasted into the context.

## 2. Related work

- **Tool use and agent loops.** ReAct (Yao et al., 2023) interleaves reasoning with actions; Toolformer
  (Schick et al., 2023) and Gorilla (Patil et al., 2023) teach models to call APIs; ToolLLM/ToolBench and the
  Berkeley Function Calling Leaderboard evaluate argument correctness. We use native function calling
  (GigaChat `functions`, Qwen chat-template tools) rather than prompt-parsed ReAct.
- **Interface design matters as much as the model.** SWE-agent (Yang et al., 2024) showed that an
  *agent–computer interface* — compact outputs, guard-rails, informative errors — changes success rates more
  than prompt tweaks. Our v1→v2 iteration (§6) is the same lesson in a medical setting.
- **LLM agents over EHR.** EHRAgent (Shi et al., 2024) generates code over relational EHR tables;
  MedAgentBench (Jiang et al., 2025) evaluates agents on FHIR tool calls. Both use English data and
  frontier models; we target Russian free text, RF-hosted or local models and a check-up (multi-specialist)
  record rather than an inpatient database.
- **Long context vs retrieval.** "Lost in the Middle" (Liu et al., 2024) shows degraded use of evidence in
  long inputs; our full-context baseline (~40–45k tokens per record) tests this on clinical data.

## 3. Data

Six anonymized check-up records exported from the clinic MIS (one row per protocol field: visit, date,
doctor, protocol, field group, field, value). 650–750 rows per patient, 15–28 protocols (labs, ultrasound,
echocardiography, CT/MRI, endoscopy, 8–12 specialists), 2–3 days per check-up.

Real-world defects we kept rather than cleaned, because the tools must survive them:

- two export layouts (with/without patient-code column), decimal commas, section-header rows in ultrasound
  protocols, the doctor type hidden inside a "protocol name" field;
- **no reference intervals** in lab rows — the values arrive without the lab blank;
- **conflicting anthropometry** across documents (e.g. P002 weight 75 vs 95 kg; P003 54 / 64 / 79 kg);
- **contradictory medication history** (P002: "takes no medication" vs "Noliprel");
- **P005: the export is shifted by one row** in the blood count, urinalysis and anthropometry (Hb "33.3 g/L",
  RBC "102", the phrase "Жалоб нет." in the thrombocrit field). A correct assistant must refuse to interpret it.

Split: P001–P002 = development (tools were debugged on them), P003, P004, P006 = test, P005 = test
(data-quality case). A probe set (13 questions on new topics) was written after the v1 error analysis and
before tools v2 were implemented.

## 4. Method

Sections 4–8 describe the study as it ran: the exploratory iterations v1–v4, under the condition names
stored in `runs/`. Section 9 describes the final, restructured system and maps the old names onto it.

### 4.1 Conditions (same model, same loop, same final-answer contract)

| Condition | What the model gets |
|---|---|
| **domain** | 9 typed tools (below) + `final_answer` |
| **generic** | `run_python(code)` over a pandas DataFrame of the record + `final_answer` (code-interpreter baseline) |
| **fullctx** | the whole record serialized into the prompt (~130–140k chars) + `final_answer` |
| **domain_v2** | the same 9 tools after the error-analysis fixes in §6 |

Domain tools: `get_overview`, `get_lab` (synonym resolution ru/en/abbreviations, unit, indicative adult
interval by sex, H/L/N flag, explicit `found=false`), `list_abnormal_labs(panel)`, `search_notes` (BM25 with
Russian prefix stemming and a medical abbreviation lexicon), `get_conclusions(specialty)`, `get_diagnoses`
(ICD-10), `get_document`, `check_consistency` (anthropometry conflicts, lab identities — MCHC≈Hb/Hct,
MCH≈Hb/RBC, WBC differential ≈100%, bilirubin total = direct + indirect — and text in numeric fields),
`calculate` (BMI, non-HDL, HOMA-IR, De Ritis). All arithmetic and reference intervals are deterministic.

`final_answer(answer, citations[], status ∈ {answered, not_found, uncertain})`: every claim must cite row ids
(`P003:r527`), `not_found` for absent data, `uncertain` for contradictory or corrupted data.

### 4.2 Agent loop

Native function calling; up to *B* tool calls per question (B = 4 by default; ablation 1/2/4/8). The harness
validates arguments against the JSON schema (required keys, types, enums), returns informative errors,
answers repeated identical calls from cache with a note, marks empty results with a hint, and forces
`final_answer` when the budget is spent. Every call is logged (arguments, validity, emptiness, result size,
latency, tokens). Temperature ≈ 0.

### 4.3 Questions and gold

49 test + 13 probe questions (plus 22 dev), gold built before the runs:

| Category | Gold | Example |
|---|---|---|
| lookup | value from the data | «Какой уровень ЛПНП?» |
| abnormal | set of out-of-range analytes (only analytes with lab-independent intervals count; the rest are neutral) | «Какие показатели липидограммы вне нормы?» |
| not_found | the analyte was never measured | «Какой уровень Lp(a)?» |
| icd | all ICD-10 codes in the record | |
| anthropo | do height/weight agree across documents | |
| freetext | key facts (regex) from one text protocol | «Что показала колоноскопия?» |
| integrate | key facts across several specialists | «Сведите все данные о печени» |
| quality | corrupted values must be flagged, not reported | «Какой гемоглобин?» (P005) |

### 4.4 Metrics

Correctness per category (exact number; set F1; ICD F1; key-fact recall; correct refusal / flag).
Grounding: share of cited ids that exist; share of gold rows cited; episodes with a number that appears
nowhere in the record or in tool outputs ("fabricated"). Process: tool calls, invalid calls, tokens, latency.
95% bootstrap CIs over questions.

### 4.5 Second source and reference intervals (added during the study)

- **Source 2 — inpatient case histories (PDF, form 003/у, another clinic, 6 cardiology cases, 90–139 pages).**
  An adapter maps PDFs onto the same record model (`sources/inpatient_pdf.py`): rule-based PII masking (patient
  and staff names, DOB, addresses, phones, card number; a later systematic audit found one miss, see §9), age at admission
  from the title page, LIS lab tables parsed with the lab's own interval and flag, the rest split into
  page/heading blocks. 17 questions are built from the facts behind the insurers' objections (max AST,
  enoxaparin dose, weight 100 vs 120 kg, discharge antithrombotics, LDL 6.31 and absent ezetimibe, smoking).
- **Reference intervals.** Layered policy (`EHR_REF_POLICY=layered`): lab blank → guideline decision limit
  (lipids, glucose, HbA1c, vitamin D) → NHANES-derived intervals by sex and age (direct method,
  survey-weighted 2.5–97.5 percentiles, simplified LAVE; 2302 reference adults; thyroid, PSA and
  testosterone from other cycles) → indicative catalog. LAVE matters: the ALT upper limit for men 40–59
  drops from 88 to 48 U/L. Coverage: PDFs 100 % lab blank; check-ups 62 % NHANES, 11 % guideline, 27 %
  catalog. Ages were removed by anonymization and are estimated from record clues (`data/assumed_ages.json`).
  Exploratory runs below use the indicative policy; the final run will use the layered one.

## 5. Results (exploratory, sparse)

Same questions for every model: 22 check-up questions (one per patient × category) and all 17 PDF
questions; 95 % bootstrap CIs are wide (±0.15–0.25), so differences below ~0.15 are trends, not effects.
Models: GigaChat-2 Lite and Max (RF API), Qwen3.5-4B and Med-Qwen-3.5-4B (same base, general vs medical
post-training; 4-bit MLX on a MacBook Air M5).

| Model | Domain tools v1: table / PDF | Tools v2: table / PDF | Whole record in context: table / PDF* | Code (pandas): table / PDF |
|---|---|---|---|---|
| GigaChat-2 Lite | 0.70 / 0.46 | 0.75 / 0.60 | 0.72 / 0.79* | 0.18 / 0.11 |
| GigaChat-2 Max | 0.85 / 0.72 | **0.93** / 0.66 | 0.84 / 0.71* | 0.43 / 0.48 |
| Qwen3.5-4B (local) | 0.68 / 0.68 | — / 0.68 | 0.83 / n.a. | 0.46 / — |
| Med-Qwen-4B (local) | **0.80 / 0.77** | — | 0.76 / n.a. | 0.22 / — |

\* Only 7 of 17 PDF questions: three of six case histories (≈ 100–230k tokens) exceed the GigaChat context
window, so "paste the record" is not applicable to half of the inpatient cases.

**H1 (tools vs generic access).** Supported against the code-interpreter baseline for every model
(+0.3–0.5). Against the whole-record baseline the picture depends on the task: tools win on
interval-dependent questions (out-of-range panels) and on data quality, the whole record wins on
cross-specialist synthesis; overall they are on par for Max, and tools are the only option for long
inpatient records. Phrasing robustness (L1 → L3): accuracy drops by 0.19 (Max) and 0.33 (Lite) with tools,
0.23–0.25 with the whole record — tools do not yet buy robustness to vague questions.

**Medical specialization (same base).** With tools, Med-Qwen beats the general Qwen3.5 (0.80 vs 0.68
table, 0.77 vs 0.68 PDF) and matches GigaChat-2 Max on PDF. Without tools it is not better (0.76 vs 0.83)
and it has lost general agentic skills: in the code condition it repeats the same failing call or reads a
non-existent CSV (0.22). Its answers are 2–3× shorter (72–123 chars), which hurts summaries (0.37–0.54 vs
0.76–0.77 for Qwen3.5) — consistent with an RL reward for short MCQ answers.

**Whole-check-up summaries.** With tools (8-call budget) both 4B models beat their own whole-record
summaries: Qwen3.5 0.76 vs 0.55, Med-Qwen 0.54 vs 0.29. The whole-record summaries lose the lab section
(0.08–0.17: abnormal values need intervals the model does not have) and never flag the corrupted P005
export; tool summaries get both (labs 1.0, quality 1.0) but miss recommendations (plan 0.11–0.19).

**Source shift (table → PDF).** GigaChat loses 0.13–0.27, the 4B models ≈0. The table → PDF gap is mostly
retrieval (below), not reasoning.

### 5.1 Where the agent fails (failure locus, as in RAG evaluation)

Errors with domain tools v1, all four models pooled:

| | Gold evidence never reached the model | Part of it reached the model | All of it reached the model, answer still wrong |
|---|---|---|---|
| Check-up tables (24 errors) | 46 % | 38 % | 17 % |
| PDF (28 errors) | 61 % | 4 % | 36 % |
| Whole record in context | — | — | 100 % (by construction) |

*Correction.* An earlier version of this section reported "100 % retrieval failures on tables". That was an
analysis bug: the row-id pattern used to collect what the model had seen did not match the check-up id
format (`P003:r367`), so tables always looked unseen. The corrected split above still makes retrieval
(full or partial) the dominant failure — 83 % of table errors, 65 % of PDF errors.

Offline retrieval benchmark (no LLM; query = question text): on tables the right **document** is in the
top-6 for 96 % of questions but the right **row** only for 60 % — the search lands in the document and
returns the description instead of the conclusion. On PDF even the document is found for 43 %: the
lexical index cannot connect "antithrombotic therapy" with "ticagrelor … rivaroxaban".

### 5.2 Model errors, by mechanism

1. **Retrieval miss → confident wrong or "not found" answer.** Max on PDF takes the first plausible hit:
   enoxaparin "1 mg/kg" from the plan instead of "0.8" from the prescription sheet; "aspirin" as the whole
   discharge regimen. Prescription sheets and discharge recommendations have no typed tool.
2. **Oversized first call.** Unfiltered `get_conclusions` / `get_overview` are truncated on almost every
   PDF episode (Max: 15 truncations in 15 episodes); Qwen spends its budget on them and exhausts the
   4-call limit in 32–40 % of episodes.
3. **Tool selection and query formulation by the weak model.** Lite picks an inapplicable first tool in
   23 % of check-up episodes, gives up after an empty result in 57 %, and turns vague questions into
   narrow queries ("what about fats?" → only total cholesterol).
4. **Citation format leaks from the prompt.** The system prompt shows the id format `P00X:rN`; on PDF
   Lite writes `P001:K01:p5:l8` or `P001:r99` — 30–36 % valid citations.
5. **Corrupted data is reported as fact unless a deterministic check pushes a warning.** For P005
   (shifted export) every model with the whole record reports a shifted value as the patient's (Hb "102",
   platelets "274"); with tools v1 too (Hb "33.3"). Tools v2, which embed the warning in each value,
   move Lite and Max to 1.0 on these questions.
6. **Summaries miss recommendations and inconsistencies** (plan section 0–0.5 for all models): no tool
   exposes "recommendations/plan", and inconsistencies appear only if the model happens to call
   `check_consistency`.
7. **Format slips.** Lite answers in plain text in 9–27 % of episodes, once in English; Med-Qwen answers
   in plain text after a tool call until nudged.

## 6. Proposed iteration (tools v3) — hypotheses, interventions, measurements

The data say the main bottleneck is evidence access (65–83 % of tool-condition errors), not model reasoning. The iteration therefore moves
intelligence from the model into the tools and the harness, which also helps the weak and local models
most. Each intervention has a measurable target and is evaluated as an ablation on a new probe set
written before implementation.

| # | Intervention | Addresses | Theory / precedent | Primary metric (target) |
|---|---|---|---|---|
| T1 | **Document-aware retrieval**: return the hit with its section (description + conclusion) and document header | 5.1, error 1 | parent-document / small-to-big retrieval | offline Row@6 → ≈Doc@6 (0.60 → ≥0.90 tables) |
| T2 | **Hybrid retrieval + ontology expansion**: BM25 + local multilingual dense embeddings (RRF fusion); expansion by ATC drug classes, ICD-10 hierarchy, lay → clinical lexicon («жиры» → липидограмма, «малокровие» → анемия) | 5.1 PDF, error 3, L3 drop | hybrid search; query expansion | PDF Doc@6 0.43 → ≥0.8; L1→L3 drop halved |
| T3 | **Canonical FHIR-like entity layer**: adapters map every source to Observation (labs, vitals), MedicationRequest (prescription sheets, discharge regimen with phase), Condition, DiagnosticReport (ECG, echo, endoscopy with key values); typed tools `get_medications(drug/class, phase)`, `get_studies(type)` | errors 1, 6; source shift | FHIR resources as a source-independent schema | PDF accuracy gap to table ≤0.05; medication questions ≥0.9 |
| T4 | **Budget-aware outputs**: pagination with cursors, "summary first, details on demand", hierarchical table of contents instead of flat overviews | error 2 | progressive disclosure; agent–computer interface design (SWE-agent) | truncations/episode → ~0; budget exhaustion <10 % |
| T5 | **Harness verification**: validate citations at `final_answer` (id exists, cited row contains the numbers) and return a repairable error; before accepting `not_found`, run an automatic broad search ("absence verification") | errors 3, 4 | self-verification / critic loops | valid citations ≥0.95 for every model; give-ups after empty result halved |
| T6 | **Intent routing and macro-tools for weak models**: classify the question (lookup / abnormal / timeline / consistency / summary) and offer a recipe; a `summarize_record` macro that collects every section deterministically and lets the model write only the synthesis | errors 3, 6 | plan-and-execute; map-reduce summarization | Lite bad first choice 23 % → <5 %; summary plan & inconsistency sections ≥0.7 |
| T7 | **Data-quality guardrails everywhere** (v2 mechanism extended): plausibility ranges, unit consistency, date sanity, duplicate rows, for both sources | error 5 | deterministic validators before generation | quality questions 1.0 for all models and access modes |
| T8 | **Prompt generalization**: no source-specific examples (id formats), copy ids verbatim from tool output | error 4 | — | identical citation validity on both sources |

Expected ordering of effects from the failure analysis: T1–T3 (retrieval, 65–83 % of tool-condition errors)
> T5–T6 (weak-model process errors) > T4, T7, T8 (cheap, model-agnostic). A model-side follow-up, outside
this iteration: LoRA fine-tuning of the 4B model on successful agent trajectories (distillation from Max
and Med-Qwen), since medical post-training already helps tool use but costs general skills.

**Evaluation protocol for v3.** (1) Offline retrieval benchmark first (Row@k, Doc@k) — cheap and
model-free. (2) A fresh probe set per category written before implementation. (3) Ablations: v1, v1+each
intervention, v3 on Lite (weak), Qwen3.5 / Med-Qwen (local), Max (strong). (4) More questions per
category (the sparse set gives ±0.2 CIs) and repeated runs at temperature > 0 for variance. (5) Final run
with the layered reference policy, gold rebuilt accordingly.

## 7. Limitations

Small samples (6 check-ups, 6 inpatient cases; sparse runs), wide CIs. Regex fact scoring is strict on
paraphrase and can miss correct answers without the expected number. Gold rows are not exhaustive (a lab
value appears in several documents), so retrieval failures are somewhat over-counted. Tools v1 were
debugged on two dev patients; v2 and the PDF adapter were finished after looking at the structure of test
records (not at model outputs on them). Ages are estimated. NHANES intervals come from another population
and other analysers (free T4 kept on the catalog interval for that reason). Med-Qwen runs with its native
short reasoning, Qwen3.5 without reasoning.

## 8. Iteration v3–v4: results

Implemented: T1 (document-aware search), T2 (hybrid BM25 + local multilingual-e5-small dense search with
RRF; ontology expansion by ATC drug classes and a lay → clinical lexicon), T8 (source-neutral prompt),
T3 (typed entities for every source: `get_medications` with INN normalization and phase, `get_studies`
with key values), T5 (harness: rescue of text-written final answers, citation check, absence
verification). Two probe sets were written before the respective implementation (v3: 9 questions,
v4: 10 questions on medications and studies). Ablations below are sparse (accuracy, n in parentheses).

**Offline retrieval (no LLM).** Gold row within what the tool returns (top-6): tables 0.60 (v1) → 1.00
(v3: hybrid + document context); PDF 0.36 → 0.73 (0.91 at top-12). On PDF the ontology carries most of the
gain (0.82 alone); the small dense model is weak on long noisy text and helps only at depth.

| Model | Version | Tables (22) | PDF (15) | Probe v3 (9) | Probe v4 (10) |
|---|---|---|---|---|---|
| Qwen3.5-4B (local) | v1 | 0.68 | 0.68 | 0.89 | 0.66 |
| | v3 | **0.97** | 0.78 | 0.81 | 0.68 |
| | v4 (v3 + T3 + T5) | 0.88 | **0.93** | **0.96** | **0.82** |
| Med-Qwen-4B (local) | v1 | 0.80 | 0.77 | 0.74 | 0.47 |
| | v3 | 0.84 | 0.81 | 0.78 | 0.55 |
| | v4 | 0.79 | 0.83 | **0.93** | **0.72** |
| GigaChat-2 Lite | v1 | 0.70 | 0.46 | 0.54 | 0.43 |
| | v3 | 0.73 | 0.54 | 0.67 | 0.43 |
| | v3 + T5.1 | 0.75 | **0.74** | **0.89** | **0.65** |
| | v4.1 (v3 + T3 + T5.1) | 0.63 | 0.65 | 0.67 | 0.56 |
| GigaChat-2 Max (reference) | v1 / v2 | 0.85 / 0.93 | 0.72 / 0.66 | — | — |

Findings:

1. **Better tools let a 4B local model reach and pass the RF-hosted frontier model.** Qwen3.5-4B with v4
   reaches 0.93 on inpatient PDFs (GigaChat-2 Max with any earlier tools: 0.66–0.72) and 0.88–0.97 on
   check-ups. This is the strongest evidence for H1: the gap opened by model size is closed by tool design.
2. **Typed entities (T3) help where free text hides structure.** On medication/study questions (probe v4)
   T3 adds +0.14 (Qwen3.5) and +0.17 (Med-Qwen) over v3; on PDFs +0.15 for Qwen3.5. On check-up tables it
   costs a little (0.97 → 0.88): extra tools distract when the data are already structured.
3. **The weak model needs the harness, not more tools.** For Lite the retrieval gains of v3 barely reach
   the answer: it calls `search_notes` in 9 of 44 episodes and writes the final answer as text in most
   episodes. Robust harness handling (T5.1) lifts PDF 0.54 → 0.74 and probe v4 0.43 → 0.65; adding T3 on
   top lowers it again (more tools, harder selection). Tool sets should be matched to model strength.
4. **Harness feedback can backfire.** The first T5 (repairable citation errors) made Lite *worse*
   (PDF 0.39): its reply to the error message ("here are the ids…") replaced the substantive answer, and a
   parser that read only the text after the `final_answer` keyword lost answers written before it
   (GigaChat also leaks a `<|superquote|>` token into JSON). T5.1 keeps the first substantive answer and
   does not run the citation loop on rescued text answers. T5.1 was fixed after seeing v4 results on all
   sets, so its numbers are not a blind estimate.
5. **Prefetch must be routed.** Pre-running text search on the question helped text questions
   (free text 0.33 → 0.83) but hurt structured ones (values, "not measured", anthropometry): the
   prefetched snippet anchors the weak model, which then reports an unrelated number (a vitamin D value
   as CA-125). Intent-routed prefetch removes the damage but did not beat v2 for Lite — consistent with
   distraction by irrelevant context (Shi et al., 2023).
6. **Citation hygiene (T8) is a one-line fix with full effect:** valid PDF citations 0.30–0.36 → 1.00 for
   Lite once the source-specific id example was removed from the prompt.
7. **Analysis bug, corrected.** The first failure-locus table counted all table errors as retrieval
   failures because the id pattern missed the check-up format; see §5.1 for the corrected split.

Remaining from the plan: T4 (paginated, budget-aware outputs), T6 (recipes, `summarize_record`), T7 for
PDFs, a stronger embedding model for PDFs, and the final run with the layered reference policy, more
questions per category and repeated runs.

## 9. Final system and corrections found while restructuring

The exploratory code grew one iteration at a time. For the final run it was rewritten as one package with
layers that do not depend on iterations: `sources → knowledge / retrieval → tools → agent → eval`. The
layers are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The exploratory code is kept
unchanged in `archive/exploratory_code/` so that every episode in `runs/` can be reproduced.

**Iterations become configurations.** An iteration is now a named configuration of one agent
(`agent/systems.py`) rather than a separate code path:

| Final system | Contents | Exploratory equivalent |
|---|---|---|
| `tools` | 11 tools, answer checks, format repair | v4.1 (`domain_v41`) |
| `tools_routed` | only the tools for the question's intent | new (follows finding 3, §8) |
| `tools_prefetch` | the intent's tool is run before the first turn | `domain_v3r` |
| `tools_compact` | without the entity tools | v3 + T5.1 (`domain_v41a`) |
| `tools_plain` | 11 tools, no harness | ablation, new |
| `code` / `record` / `record+tools` | baselines | `generic` / `fullctx` / `hybrid` |
| `record_chunked` | map-reduce over record chunks, no tools | new: closes the gap of item 2 below |

**Equivalence checks of the rewrite.**

- The question set is identical (202 questions).
- Rescoring all logged episodes gives identical scores (2129 episodes).
- Tool outputs on all 12 patients: 244 of 264 calls are identical. The 20 differences are intended:
  - implausible-value marks on P005;
  - output limits and pagination (T4 from §8 is done: lists are cut with `next_offset`, not inside JSON);
  - `get_medications` no longer truncated into a string.
- The offline retrieval benchmark is identical.
- 36 unit tests run on synthetic records only.
- Automatic formatting changed no output: tool results and scores are byte-identical before and after.

**Defects found and fixed. Each affects how §5–8 should be read.**

1. **GigaChat rejected every forced final call (HTTP 422).** The API answers *"undefined functions in
   dialog history: [final_answer]"* to `function_call={"name": "final_answer"}` whenever the full functions
   list is sent. This happens even with an empty history and even when `final_answer` is in the list. A
   forced call is accepted only when the functions list contains `final_answer` alone. The fix sends
   exactly that list on forced turns.
   - **Consequence: selection bias in exploratory GigaChat results.** Episodes that exhausted the budget or
     needed a forced format repair failed and were not saved (by design, backend errors are not logged as
     answers).
   - **Scale.** Lite lost 3–7 of ~50 episodes per condition in the v4 family, and up to 1 in earlier
     iterations (later retries recovered transient errors).
   - **Direction.** The lost episodes are the hard ones, where the model kept calling tools. So the v4-family
     accuracies for Lite in §8 are optimistic by up to ~0.1.
   - After the fix, all 28 smoke episodes on Lite completed under every system, format repair included. The
     final run has no such gap.
2. **Whole-record baselines overflow the context on long PDFs.** A whole case history is up to ~130k
   tokens. That exceeds the GigaChat context limit (130,048 tokens) and the memory budget of a 4B local
   model (limit set to 100k tokens on a 16 GB machine). Such episodes now get the status
   `context_overflow`, are scored N/A and are reported separately; they are not counted as wrong. In the
   exploratory runs they failed and went unsaved (9 Lite and 9 Max `fullctx` episodes). The baseline
   `fullctx` PDF numbers in §5 therefore cover only the cases that fit. For the final run a map-reduce baseline
   (`record_chunked`) reads the whole record in ~60k-character chunks, so the comparison "tools vs. the
   whole record" covers every case. Its excerpts are grounded in the record's own lines (by cited id or
   by word overlap with the quoted text): without that, GigaChat Lite returned bare values ("4.06")
   that the reduce step read as "not found".
3. **PII masking missed a name printed in capitals.** A registry block on one PDF page printed the
   patient's full name in upper case, which the case-sensitive token mask did not catch. Found by the new
   audit (`eval/pii_audit.py`), which re-reads each patient's identity from the raw title page and searches
   every masked record for the name in any case and inflection, the date of birth, the card number, phones,
   addresses and person-like sequences. Masking is now case- and inflection-insensitive, with an extra rule
   for full names in capitals. The audit reports zero residues on all six cases. The one changed block (a
   registry page) is not referenced by any question. The unmasked name reached GigaChat (RF-hosted) only
   through the 8 whole-record (`fullctx`) episodes on that case. No tool-based episode returned that block,
   and no other external service received patient data.
4. **The plausibility rule now applies only without a lab blank.** The rule flags values outside
   physiological limits (e.g. Hb 33 g/L). It now runs only when a row has no lab blank: a blank defines its
   own units (MCHC in g/dL) and its own pathology range (troponin far above the limit is an infarction, not
   an export error).
5. **Medication phase coverage.** Phase rules now read the field label ("Рекомендации на амбулаторный
   этап", "…стационарного лечения"). This lowered the share of mentions with an unknown phase from 0.65 to
   0.44. The precision of the assigned phases was not measured against a labelled sample.
6. **Gold support rows.** The same lab value often appears in several documents. `support_rows` adds every
   row that carries the gold value to the accepted citations (39 questions extended). This reduces the
   over-counting of retrieval failures noted in §7.

**Practicality.** `eval/behavior.py` now reports latency and tokens per answer by model and system (§5a
of `runs/behavior.md`). In the exploratory runs, tool systems needed:

| Model | Median latency | Prompt tokens | `fullctx` on the same questions |
|---|---|---|---|
| GigaChat Lite | ≈2 s | 3–6k | 4 s, 33k tokens |
| GigaChat Max | ≈4.5 s | — | 11 s, 39k tokens |
| Qwen3.5-4B, local MacBook | 12–16 s | 7–13k | 74–108 s, 68–78k tokens |

Tools therefore cut the prompt by 5–10× for every model. For a local model they turn a two-minute answer
into a fifteen-second one.

**Open for the final run.**

- A blind question set, written by someone other than the tool author.
- GigaChat Max on the final systems.
- The layered reference policy.
- Repeated runs at temperature > 0.
- Paired tests between systems with Holm correction.
- A scorer audit on a sample of answers.

## 10. Final run (25.10.2026)

Plan: [docs/FINAL_RUN_PLAN.md](docs/FINAL_RUN_PLAN.md). Full tables: `runs/final_stats.md` (`python -m ehr_agent.eval.stats`).

**Deviations from the plan, forced by time before the defense:**
- **GigaChat Lite** ran only on the test, pdf and trend sets plus summaries; the tool-set ablations
  (routed, compact, prefetch) were dropped.
- **No variance runs** at T = 0.5.
- **Qwen:** whole record on 15 check-up questions; map-reduce on PDFs not run.
- **~10 GigaChat Max episodes** (<2 %) failed with backend errors (429 while the demo shared the key) and were not
  retried.

**Accuracy** (temperature 0, layered references):

| Model | Tools, tables | Tools, PDF | Code, tables / PDF | Whole record, tables | Map-reduce, PDF | Summaries: tools → report |
|---|---|---|---|---|---|---|
| GigaChat Max | 0.86 | 0.93 | 0.38 / 0.44 | 0.78 | 0.80 | 0.62 → 0.83 |
| Qwen3.5-4B | 0.88 | 0.88 | 0.42 / 0.47 | 0.76 (15) | — | 0.61 → 0.92 |
| Med-Qwen-4B | 0.74 | 0.84 | — | — | — | 0.62 → 0.58 |
| GigaChat Lite | 0.63 | 0.71 | 0.05 / 0.10 | 0.80 | 0.55 | 0.30 → 0.62 |

**Pre-registered contrasts** (paired permutation tests, Holm over 29 contrasts):

| # | Result |
|---|---|
| H1a | Tools beat the code interpreter for every model: +0.43…+0.59, p(Holm) = 0.002. |
| H1b | Tools vs the whole record (tables) and vs map-reduce (PDF): no significant difference in either direction (Max +0.08, Lite −0.18 on tables), at 5–16× fewer prompt tokens; the whole record did not fit the context in 15 of 31 PDF questions. |
| H1c | The harness adds +0.04 (Max) and +0.10 (Lite), not significant. |
| H2 | The dossier raises critical recall of summaries for all four models (+0.39…+0.49, all CIs above 0), but with 9 records per model no contrast survives Holm. On the 6 manual checklists the effect is mixed (Qwen +0.16, Max −0.01, Med-Qwen −0.20). |
| H3 | Med-Qwen is 0.11 below its base Qwen (CI −0.18…−0.04, raw p = 0.003, Holm 0.065): borderline. |
| H4 | Qwen3.5-4B with tools is on par with GigaChat Max (+0.01, CI −0.06…+0.07); GigaChat Lite is 0.25 below Max (p(Holm) = 0.002). |

**Scorer audit.** A manual review of 24 stratified answers agrees with the automatic score in 92 % of cases
(±0.25), Cohen's κ = 0.73 on correct-vs-not. Both disagreements are lenient: a keyword matched without the right
fact. The reviewer was the AI assistant used in the project, not a clinician.

**Practicality** (GigaChat Max, inpatient PDFs, medians): tools take 9.8 s and 7k prompt tokens; map-reduce over
the whole record takes 41 s and 115k tokens over ~7 chunks, with no gain in accuracy.
