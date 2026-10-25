"""ehr_agent — tools and a tool-use agent for LLM-assisted analysis of electronic health records.

Layers (each depends only on the ones above it):
  sources    xlsx check-up exports and PDF case histories → one Record model (PII masked)
  knowledge  analyte catalog, reference-interval policy, lexicons, medication/study entities
  retrieval  BM25 + dense (e5) hybrid search with document context
  tools      deterministic tools the model calls (labs, text, quality, clinical entities, baselines)
  agent      tool-use loop, answer-checking harness, named system configurations, model backends
  eval       benchmark questions and gold, scoring, experiment runner, analyses
"""
