"""Named system configurations: what the model gets (tools, record text) and what the harness does.

A configuration is data, not code — the final run compares these by name:

  tools          full tool set, answer checks, format repair                     (the system)
  tools_routed   + only the tools relevant to the question's intent              (for weak models)
  tools_prefetch + the intent's tool is run before the first model turn          (one call saved)
  tools_compact  tool set without the entity tools, checks, format repair        (fewer choices)
  tools_plain    full tool set, no harness                                        (ablation: harness effect)
  code           run_python over a DataFrame of the record                        (baseline: code interpreter)
  record         the whole record in the prompt, no tools                         (baseline: long context)
  report         dossier prefetched, sectioned report, coverage and verdict checks (the summary system)
  report_nocheck dossier and sectioned report without coverage/verdict checks   (ablation: checks)
  record_chunked the whole record read in chunks (map), answer from the excerpts   (baseline: map-reduce)
  record+tools   the whole record in the prompt plus the full tool set            (do tools help with equal access?)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemConfig:
    toolset: str = "full"  # key of tools.TOOLSETS
    record_in_prompt: bool = False  # serialize the whole record into the user message
    checks: bool = True  # harness answer checks (citations, numbers, absence)
    format_repair: bool = True  # a text answer is converted by a forced final_answer call
    prefetch: bool = False  # run the intent's tool before the first model turn
    routed_tools: bool = False  # expose only the intent's tools
    report: bool = False  # prefetch build_dossier; the final answer is the sectioned report
    report_checks: bool = False  # harness: every must_mention item covered, verdicts agree with the dossier
    chunked: bool = False  # map-reduce: excerpts from every chunk of the record replace the record (chunked.py)


SYSTEMS: dict[str, SystemConfig] = {
    "tools": SystemConfig(),
    "tools_routed": SystemConfig(routed_tools=True),
    "tools_prefetch": SystemConfig(prefetch=True),
    "tools_compact": SystemConfig(toolset="compact"),
    "tools_plain": SystemConfig(checks=False, format_repair=False),
    "code": SystemConfig(toolset="code", checks=False, format_repair=False),
    "record": SystemConfig(toolset="none", record_in_prompt=True, checks=False, format_repair=False),
    "report": SystemConfig(toolset="report", report=True, report_checks=True),
    "report_nocheck": SystemConfig(toolset="report", report=True),
    "record_chunked": SystemConfig(toolset="none", chunked=True, checks=False, format_repair=False),
    "record+tools": SystemConfig(record_in_prompt=True),
}


def get_system(name: str) -> SystemConfig:
    if name not in SYSTEMS:
        raise KeyError(f"Unknown system '{name}'. Known: {sorted(SYSTEMS)}")
    return SYSTEMS[name]
