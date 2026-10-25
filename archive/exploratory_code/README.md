# Exploratory code (2026-10-24 … 25)

The code that produced every episode in `runs/*.jsonl` before the final restructuring. Kept for
traceability only; the maintained system is `ehr_agent/` at the project root.

Condition names in `runs/` map to this code as follows:

| condition in runs | tool class / harness | later name in the final system |
|---|---|---|
| `domain` | `DomainTools` | — (superseded) |
| `domain_v2` | `DomainToolsV2` (source names in search, compact outputs, quality warnings) | part of `tools` |
| `domain_ri` | `DomainToolsRI` (age/sex intervals) | reference policy `layered` |
| `domain_v3` | `DomainToolsV3` (hybrid search + ontology + document context), source-neutral prompt | part of `tools` |
| `domain_v3p` / `domain_v3r` | v3 + prefetch of search / intent-routed prefetch | system `tools_prefetch` (intent-routed) |
| `domain_v4a` / `domain_v41a` | v3 + answer checks T5 / T5.1 | harness option `checks` |
| `domain_v4` / `domain_v41` | v3 + medications and studies (T3) + T5 / T5.1 | profile `tools` with checks |
| `generic` | `run_python` over a DataFrame | profile `code` |
| `fullctx` / `hybrid` | whole record in the prompt (+ tools) | profiles `record` / `record+tools` |

Run with `PYTHONPATH=archive/exploratory_code` from the project root.

`run_queues/` holds the shell queues that produced the exploratory `runs/*.jsonl` (they call the old module
paths, e.g. `ehr_agent.run --conditions domain_v41`). Their logs are in `runs/logs/`.
