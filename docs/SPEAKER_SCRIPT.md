# Speaker script — Set of tools for EHR LLM-assisted analytics

Ivan Kutelev · MSc-2, Life Sciences · Modern Methods and Algorithms of Generative AI · 25 Oct 2026

The script follows the 21 slides in order. For each slide:
- **Key message** — the one sentence the audience should remember.
- **Say** — what to say, about 30–60 seconds.
- **Terms** — abbreviations and concepts to explain if asked.

The whole talk is about 12–15 minutes without the demo. Slides 8–11 explain how the system works: spend the most
time there.

---

## 1 · Cover — *Set of tools for EHR LLM-assisted analytics*

**Key message:** We built and evaluated an interface through which an LLM answers a doctor’s questions about one
patient’s record.

**Say:** Good morning. My project comes from the AI-mnesis startup. The question is practical: a doctor asks
something about a patient — a lab value, a drug dose, a summary of a hospital stay — and an LLM has to answer from
the patient’s electronic health record. I compare three ways of giving the model access to the record, on two
kinds of real data and four models.

**Terms:**
- **EHR** — electronic health record, the patient’s digital chart.
- **LLM** — large language model.
- **LLM-assisted analytics** — the model reads and summarizes the record; it does not diagnose.

---

## 2 · Context — AI-mnesis and the constraints

**Key message:** The rules of Russian healthcare decide which models we can use, so the engineering has to make up
for weaker models.

**Say:**
- AI-mnesis (ai-mnesis.ru) works to give doctors’ time back to patients: up to 70 % of it goes to history taking
  and paperwork. The startup builds modules for patient interviews, record analytics and documentation audit, with
  a pilot at a regional hospital.
- Three constraints shape this project:
  - **Data stays in Russia.** Federal law 152 on personal data rules out foreign cloud models. What remains is
    GigaChat from Sber, or an open-weights model run on the clinic’s own hardware.
  - **Verifiability.** Every statement must point to a row of the record. If the data is not there, the system
    must say so.
  - **Cost and speed.** The models we are allowed to use are weaker than the frontier, so the architecture has to
    compensate.

**Terms:**
- **Federal law 152-FZ** — Russian law on personal data; requires storing and processing citizens’ data in Russia.
- **Open-weights model** — a model whose parameters can be downloaded and run locally.
- **Frontier model** — the strongest commercial models (GPT, Claude, Gemini class).

---

## 3 · Three ways to give an LLM the record

**Key message:** We compare the two approaches general assistants use with a third one: typed tools.

**Say:**
- **Whole record in the prompt.** Simple, but a check-up is 33–78 thousand tokens, and a hospital PDF often does
  not fit at all. Long-context research shows two problems: models lose information in the middle of long inputs
  (“Lost in the Middle”, Liu et al. 2023), and irrelevant text distracts them (Shi et al. 2023).
- **Code interpreter.** The model writes Python over a table of the record. Flexible, but the model itself must
  know the column layout, medical synonyms and reference ranges. EHRAgent (Shi et al. 2024) uses this approach.
- **Typed tools.** Narrow functions such as “give me this lab value with its reference range” or “list drugs by
  phase”. Knowledge and arithmetic live in deterministic code. The model only decides which tool to call and how
  to phrase the answer.

The idea comes from SWE-agent’s agent–computer interface: an agent is only as good as the interface it is given.

**Terms:**
- **Token** — the unit of text a model reads, roughly 3–4 characters of Russian.
- **Context window** — the maximum number of tokens a model can read at once.
- **Tool calling / function calling** — the model replies not with text but with a structured request
  “call function X with arguments Y”. The program executes it and returns the result.
- **ReAct** — reason-and-act loop: think, call a tool, observe, repeat.
- **Toolformer** — models that learn when to call tools.
- **Gorilla / BFCL** — benchmarks of tool-calling accuracy.
- **ACI** — agent–computer interface (SWE-agent).

---

## 4 · Models under test

**Key message:** Four models chosen as controlled pairs, all using the same tools and the same loop.

**Say:** The four models form pairs, so that each comparison isolates one factor:
- **GigaChat Lite vs GigaChat Max** isolates model size within one provider.
- **Qwen vs Med-Qwen** isolates medical fine-tuning on the same base model.
- **Qwen vs Max** compares a small local model with a large cloud one.

How they call tools:
- **GigaChat** has native function calling: we send a list of function descriptions in JSON Schema, and the model
  returns a function name and JSON arguments.
- **Qwen** runs locally on Apple Silicon through MLX in 4-bit. Its chat template includes the tools; the model
  writes a `<tool_call>` block that our backend parses.
- **Med-Qwen** does the same but first writes a `<think>` reasoning block.

All four sit behind one backend interface, at temperature 0.

**Terms:**
- **JSON Schema** — a formal description of the arguments a function accepts: names, types, required fields.
- **MLX** — Apple’s machine-learning framework for M-series chips.
- **4-bit quantization** — weights stored in 4 bits instead of 16; about 4× less memory at a small cost in accuracy.
- **Fine-tuning** — further training of a base model on domain data.
- **Temperature 0** — always take the most likely token; makes outputs (nearly) deterministic.
- **Forced call** — making the model call `final_answer` when the budget is spent. GigaChat accepts this only when
  `final_answer` is the only function in the list.

---

## 5 · Research question and hypotheses

**Key message:** The goal is to probe how far small and Russian models can be pushed by the interface around them,
not to ship a clinical product.

**Say:** Four hypotheses, fixed before the final run:
- **H1** — tools beat the code interpreter and are not worse than the whole record at a fraction of the cost; the
  harness (the checks around the model) helps.
- **H2** — for whole-record summaries, a dossier with a report schema raises completeness, and completeness checks
  add on top.
- **H3** — medical fine-tuning does not help when the model works through tools.
- **H4** — a small local model with tools is on par with a large cloud model.

The analysis plan was written down before seeing any results: paired tests and correction for multiple
comparisons. This protects us from fitting the story to the numbers.

**Terms:**
- **Pre-registration** — fixing hypotheses and tests before the data.
- **Paired comparison** — both systems answer the same questions, so we compare per question.
- **Harness** — the deterministic code around the model: validation, checks, repairs.

---

## 6 · Data

**Key message:** Real data with real defects, kept on purpose.

**Say:**
- **Source 1:** six anonymized check-up exports from a partner clinic — tables of 650–750 rows covering labs,
  ultrasound, echocardiography, endoscopy and 8–12 specialists. The export has defects:
  - no lab reference ranges;
  - weight 75 vs 95 kg in different documents;
  - for patient P005 the whole export is shifted by one row, so the haemoglobin field says 33.3 g/L, which is
    physiologically impossible.
- **Source 2:** six cardiology hospital stays from another clinic, PDFs of 90–139 pages stitched from many
  doctors’ documents. Personal data is masked on load, and an audit found no residues.

Both sources become one record model. Every row gets a stable id — `P003:r351`, `K05:p12:l1` — and answers must
cite these ids. The excerpts on the slide are real rows.

**Terms:**
- **Check-up** — a planned multi-specialist examination.
- **Form 003/u** — the Russian inpatient medical record form.
- **PII** — personally identifiable information; masked as [PATIENT], [NAME], [PHONE].
- **Row id** — `P003:r351` is check-up P003, export row 351; `K05:p12:l1` is case K05, page 12, lab row 1;
  `b` instead of `l` means a text block.

---

## 7 · Architecture

**Key message:** Clean layers. The model sees only tool descriptions and JSON results; everything medical is
deterministic code.

**Say:** Read the diagram top-down:
1. The doctor’s question enters the **agent layer**, which has three parts:
   - the **loop** runs the episode;
   - the **harness** routes and checks;
   - the **backends** talk to GigaChat or Qwen.
2. The agent calls the **tools layer**: 12 tools plus `final_answer`.
3. The tools rely on two lower layers:
   - **knowledge** — analyte catalog, reference ranges, synonyms and ontology, drug and study extraction,
     significance rules;
   - **retrieval** — hybrid search.
4. At the bottom, **sources** convert both data types into the same record model.
5. The **evaluation** modules on the right use every layer, but nothing depends on them.

**Terms:**
- **Ontology** — a structured vocabulary. Here: drug classes from the ATC hierarchy and lay-to-clinical terms
  (“fats in blood” → lipid panel).
- **ATC** — Anatomical Therapeutic Chemical classification of drugs (WHO).
- **Episode** — one question answered end to end.

---

## 8 · Anatomy of one episode — *how the system works*

**Key message:** The model chooses and phrases; the code retrieves, judges and checks.

**Say:** Let me walk through one real episode with the question “What is the patient’s LDL?”
1. **Model → call.** The model sees the system prompt and the descriptions of 12 tools. Instead of answering, it
   returns a function call: `get_lab(analyte="ЛПНП")`.
2. **Code → result.** Our code resolves “ЛПНП” to the analyte LDL through a synonym catalog. It finds the row, then
   picks the reference range by a layered policy:
   - the lab’s own range if the record has one;
   - otherwise a clinical-guideline limit;
   - otherwise an age- and sex-specific population range from NHANES;
   - otherwise an indicative catalog value.

   It returns a short JSON: id `P003:r351`, value 4.06, flag H (high), range ≤ 3.0 from guidelines.
3. **Answer → checks.** The model calls `final_answer` with the text, the cited row ids and a status: answered,
   not found or uncertain. Before accepting, the harness checks:
   - the cited ids exist;
   - every number in the answer occurs in the record or in tool outputs;
   - a “not found” answer is double-checked with a broad search.

The table lists four design rules, each preventing a failure we observed:
- **Typed, validated arguments** — a malformed call becomes a repairable error message, not a crash.
- **Output budgets with pagination** — results stay short; long lists are cut with `next_offset` instead of being
  truncated mid-JSON.
- **Explicit emptiness** — `found=false` together with what was checked, so “not measured” is not confused with
  “not retrieved”.
- **A row id on every fact** — every claim can be verified.

The model may make up to 4 tool calls; then the final answer is forced.

**Terms:**
- **LDL / ЛПНП** — low-density lipoprotein, “bad cholesterol”.
- **Flag H/L/N** — above, below or within the reference range.
- **NHANES** — US National Health and Nutrition Examination Survey. We computed population reference intervals
  from its 2017–2018 lab data by the IFCC/CLSI direct method: healthy reference individuals and the 2.5th–97.5th
  percentiles. As a proof of concept it is another population with other analysers, which is stated as a limitation.
- **LAVE** — latent abnormal values exclusion, a method that removes hidden outliers from the reference
  population.
- **Pagination / next_offset** — returning a long list page by page.

---

## 9 · Tools and knowledge

**Key message:** Twelve narrow tools, each backed by an open, citable knowledge source.

**Say:** The tools fall into groups:
- **Labs:** `get_lab`, `list_abnormal_labs` by panel, and `get_trend` for dynamics.
- **Text:** `search_notes`, a hybrid search that returns the found row with its document’s conclusion;
  `get_conclusions`; `get_diagnoses` with ICD-10 codes.
- **Entities:** `get_medications`, with drugs normalized to international names and grouped by phase (before
  admission, in hospital, at discharge); `get_studies` for ECG, echo, CT and similar.
- **Quality:** `check_consistency` finds contradictions and shifted exports.
- **Summaries:** `build_dossier`.

Knowledge sources:
- NHANES and clinical guidelines for reference ranges;
- the EFLM biological variation database for meaningful change;
- ATC for drug classes;
- BM25 plus a multilingual embedding model fused by RRF for search.

**Terms:**
- **ICD-10** — International Classification of Diseases, 10th revision (codes like I21 for myocardial infarction).
- **INN** — international nonproprietary name of a drug (clopidogrel, not its brand Plagril).
- **BM25** — classic keyword-ranking formula.
- **Dense retrieval / embeddings** — search by meaning: the text becomes a vector (multilingual-e5-small), and
  similar texts are close.
- **RRF** — reciprocal rank fusion: merges ranked lists from several searches.
- **RCV** — reference change value: the smallest change between two results that is not measurement noise.
  RCV = 1.96·√2·√(CVa² + CVi²), where CVa is analytical imprecision and CVi is within-person biological variation.
- **EFLM** — European Federation of Clinical Chemistry and Laboratory Medicine.

---

## 10 · The harness

**Key message:** Deterministic checks around a probabilistic model; each check fires at most once.

**Say:**
- **During the episode:**
  - arguments are validated against the schema;
  - repeated calls are answered from a cache;
  - empty results say so explicitly;
  - for weak models, only the tools relevant to the question’s intent are shown.
- **If the model answers in plain text instead of calling `final_answer`,** we ask once for a formatted answer.
  Reasoning and formatting are decoupled, because format constraints are known to hurt reasoning (Tam et al. 2024).
- **Before accepting an answer:** citations exist, numbers are grounded, and “not found” is double-checked.
- **For reports:** every critical item is covered, and no flagged value is called normal.

**Terms:**
- **Format repair** — converting a free-text answer into the structured `final_answer` call.
- **Grounding** — every number traceable to the source.
- **Intent routing** — classifying the question (lab value, medication, study…) to pick relevant tools.

---

## 11 · Use case — a corrupted export

**Key message:** Tools carry data-quality knowledge with every value. Both baselines confidently reported a wrong
number.

**Say:** This is a real final-run episode with GigaChat Max: “What is the patient’s haemoglobin?” on P005, the
shifted export.
1. `get_lab` returns 33.3 g/L, already marked “unreliable — this blood-count panel breaks an internal identity,
   rows likely shifted”.
2. The model calls `check_consistency` and sees weights of 1.86, 7, 70 and 73 kg across documents and four broken
   lab identities.
3. It answers in text. The harness asks for the formatted answer, and the final answer is: “haemoglobin is
   unreliable because of a likely row shift”, status uncertain, citing the row.

With the same question, the code interpreter reported 33.3 g/L, and the whole-record model read a neighbouring
row and said 102 g/L. The model did not need to discover the corruption itself: the code told it.

**Terms:**
- **Lab identity** — an arithmetic relation that must hold, for example MCHC ≈ Hb / Hct × 100, or a leukocyte
  differential summing to 100 %.
- **Status uncertain** — the answer contract has three statuses: answered, not found, uncertain.

---

## 12 · Benchmark

**Key message:** 218 questions with gold answers built from the data, including traps.

**Say:**
- **Ten categories.** They include lab values, out-of-range panels, ICD codes, free-text facts, cross-document
  synthesis, dynamics and summaries, plus two traps:
  - a test that was never done — any number is a hallucination;
  - corrupted data — the right answer is to refuse.
- **Three phrasing levels** of the same question: exact; clinical shorthand (“LDL-C?”); lay (“how is the bad
  cholesterol?”).
- **Gold computed from the record:** numbers, sets of abnormal analytes, codes. Free-text facts are checked with
  regular-expression checklists.
- **Final run:** a stratified sample of 105 questions at temperature 0, with paired tests.
- **Scorer audit:** a manual review of 24 answers agreed with the scorer 92 % of the time, κ = 0.73.

**Terms:**
- **Gold / ground truth** — the reference answer.
- **Stratified sample** — one question per data set × patient × category, so every cell is represented.
- **Regex** — regular expression, a text pattern.
- **Cohen’s κ** — agreement corrected for chance: 0 is chance level, 1 is perfect; 0.6–0.8 is substantial.

---

## 13 · Iterations

**Key message:** Every improvement targeted a failure we measured in the traces.

**Say:**
- **Search.** Hybrid search plus document context raised the share of questions where the needed row was visible
  from 0.60 to 1.00 on tables. Ontology expansion raised it from 0.36 to 0.73 on PDFs.
- **Prompt.** Removing one example id format from the prompt fixed citations: 0.30 → 1.00. The model had been
  copying the wrong id format onto the other source.
- **Typed drug and study entities** added +0.14 on the drug questions.
- **A robust harness** raised Lite from 0.54 to 0.74 on PDFs.
- **The dossier** took summaries from omitting everything to 0.97 critical recall. That figure is in-sample.

**Terms:**
- **Ctx@6** — the needed row is among the top 6 results or in their document context.
- **Probe set** — questions written before an iteration to test it fairly.
- **In-sample** — measured on the same questions the system was tuned on, so optimistic.

---

## 14 · Problems we hit

**Key message:** Most surprises were in the evaluation, not in the model. Agent benchmarks need trace-level
quality control.

**Say:**
- **Selection bias.** GigaChat rejected every forced final call with HTTP 422, so hard episodes silently
  disappeared and early Lite scores were inflated by up to 0.1. The fix: send `final_answer` alone on forced turns.
- **Overflow.** Hospital records of up to 200 thousand tokens made the whole-record baseline not applicable. We
  added a map-reduce baseline instead.
- **Harness backfire.** A harness message once made the weak model replace its answer with “here are the ids”.
  We now keep the first substantive answer.
- **Analysis artefact.** A regular expression missed one id format and produced a false “100 % retrieval failure”.
- **Masking miss.** A patient name printed in capitals escaped masking; an audit found it and it is fixed.

Every one of these was found by reading traces, not aggregate scores.

**Terms:**
- **HTTP 422** — the API rejected the request as invalid.
- **Selection bias** — the measured sample is not representative; here, failures were dropped.
- **Map-reduce** — split the record into chunks, extract relevant lines from each (map), answer from the extracts
  (reduce).

---

## 15 · Whole-record summary and dynamics

**Key message:** For summaries, recall is the code’s job and the verdict is the model’s.

**Say:** A summary fails by omission: the model picks a few tools and misses whole sections. So the work is split
in three:
1. **Dossier.** Code enumerates everything abnormal or changing — with trend, studies, diagnoses, drugs by phase
   and conflicts — and ranks it as critical or significant.
2. **Report schema.** The model writes the report section by section: fact, value, verdict, trend, row id.
3. **Checks.** Every must-mention item is covered or explicitly dismissed. A flagged value cannot be called normal,
   and the trend direction must match the series.

In the final run, critical recall rose for every model: Max 0.40 → 0.89, Lite 0.24 → 0.71, Qwen 0.61 → 1.00,
Med-Qwen 0.37 → 0.82. On the six hand-written checklists the effect is mixed, and with 9 records per model it is
not significant after correction.

**Terms:**
- **Critical recall** — share of must-mention items, such as a raised troponin or a myocardial infarction, that
  the summary covers.
- **Dynamics** — how a lab value changes over the stay; a change counts only above the RCV.
- **Troponin** — a marker of heart-muscle damage.

---

## 16 · Results — accuracy

**Key message:** Tools roughly double the code interpreter’s accuracy, and a 4-billion-parameter local model
matches GigaChat Max.

**Say:**
- **Tools vs code.** With tools, Max reaches 0.86 on check-ups and 0.93 on hospital PDFs; Qwen reaches 0.88 on
  both. The code interpreter gets 0.38–0.47, and Lite only 0.05–0.10.
- **Tools vs the whole record.** On check-ups, which fit into the context, the whole record is close: 0.78–0.80.
  On PDFs it did not fit in half the questions.
- **Weak model.** Lite is clearly behind: 0.63–0.71.

**Terms:**
- **Accuracy** — automatic score from 0 to 1 per question, by gold type, averaged.

---

## 17 · Pre-registered contrasts

**Key message:** One strong result (tools beat code), one clear parity (Qwen = Max), and honest non-results.

**Say:** Each line is a paired difference with its 95 % confidence interval; filled markers survive correction for
29 comparisons.
- **H1a holds for every model,** +0.43 to +0.59.
- **H1b:** tools vs the whole record or map-reduce shows no significant difference, at 5–16 times fewer tokens.
- **H1c:** the harness gives +0.04 to +0.10, not significant.
- **H2:** the dossier raises critical recall for all models, but the samples are too small.
- **H3:** Med-Qwen is 0.11 below its own base, borderline after correction (p = 0.065).
- **H4:** Qwen vs Max is +0.01 — on par. Lite is 0.25 below Max, significant.

**Terms:**
- **Confidence interval (CI)** — the range of plausible true differences.
- **Permutation (sign-flip) test** — randomly swap which system each answer came from to see how often a
  difference this large appears by chance.
- **Bootstrap** — resampling questions to estimate the CI.
- **Holm correction** — adjusts p-values when many hypotheses are tested, so that false positives don’t accumulate.
- **p-value** — the probability of a difference at least this large if there were no real effect.

---

## 18 · Practicality

**Key message:** Tools are cheaper and faster: 16× fewer tokens and 4× faster than reading the whole record.

**Say:**
- **Tools vs map-reduce.** On hospital PDFs with GigaChat Max, tools need about 7 thousand prompt tokens and 10
  seconds. Map-reduce over the whole record needs 115 thousand tokens and 41 seconds, with no gain in accuracy.
- **Local option.** Qwen with tools answers in 10–20 seconds on a MacBook. On a CPU-only server the same model
  would take minutes, because processing the prompt, not generating the answer, is the bottleneck. So the local
  option needs a GPU or Apple Silicon.

**Terms:**
- **Prompt processing (prefill)** — the model reading the input; it scales with input length.
- **Median** — the middle value, robust to outliers.

---

## 19 · Insights

**Key message:** The interface matters as much as the model.

**Say:**
1. **Same tools, same accuracy.** A local 4B model scored 0.89 and GigaChat Max 0.88 with the same tools.
2. **Domain fine-tuning did not transfer.** Through tools, following instructions mattered more than medical
   knowledge.
3. **Weak models need checks fitted to them.** The harness helped Lite more than Max.
4. **When recall matters, enumerate rather than search.** A deterministic dossier fixed omissions in summaries.

---

## 20 · Limitations and next steps

**Key message:** Promising, honest, small.

**Say:**
- **Limitations:**
  - only 12 records, so we can detect only large effects;
  - the questions were written by me, the tool author, not blind;
  - the report system was tuned on the same summaries it is scored on;
  - the regex scorer agrees with manual review at κ = 0.73;
  - the NHANES ranges come from another population.
- **Next steps:**
  - a blind question set written with clinicians;
  - more records from partner clinics;
  - guideline rules for audit verdicts;
  - trends across repeated check-ups;
  - a local model on a GPU server inside the clinic.

---

## 21 · Demo

**Key message:** You can try it yourself: ask a question and see every tool call.

**Say:** The demo runs at ai-mnesis.ru/ehr-demo behind a Keycloak login, with GigaChat Max and only the anonymized
check-ups. Show four things:
1. **A showcase question:** LDL of P003. Point at the tool path and the cited row.
2. **The trap:** haemoglobin of P005, which should be refused.
3. **The tool-use dashboard:** which tools each model calls, and harness interventions.
4. **Recorded runs:** the same question under different systems.

Code, report and the check-up data are in the public repository github.com/deogratiusx/GenAI_EHR.

**Terms:**
- **Keycloak** — open-source login and access-management server.
- **oauth2-proxy** — the gate in front of the demo that checks the Keycloak login and the role.

**Demo tip:** the GigaChat key serves one request at a time. Run one question at a time and wait for the answer.

---

## Likely questions

- **Why not GPT or Claude as a reference?** Data residency rules them out for real patient data; a stronger control
  model was not available within the constraints.
- **Is 0.89 good enough for clinical use?** No: this is a research probe on 12 records with author-written
  questions. It shows the direction, not readiness.
- **Why did the whole record do well on check-ups?** They fit into the context (about 50 thousand tokens); the
  cost is 5–16× more tokens, and it fails on long hospital records.
- **How do you know the scorer is right?** A manual audit of 24 answers gave κ = 0.73; both disagreements were the
  scorer being too lenient.
- **What would you do with more time?** A blind clinician-written test set and more data, then verdict rules from
  clinical guidelines for the audit module.
