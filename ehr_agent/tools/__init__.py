"""The tool layer: what the agent can call over one patient's record.

`EHRTools(patient)` holds the shared state (lab values, quality marks, search index, entities) and
exposes every tool as a method. Tools are grouped by module:

  text.py      get_overview · search_notes · get_conclusions · get_diagnoses · get_document
  labs.py      get_lab · list_abnormal_labs · calculate
  quality.py   check_consistency (+ quality marks attached to lab values)
  clinical.py  get_medications · get_studies
  summary.py   get_trend · build_dossier (+ FINAL_REPORT: the sectioned final answer of report systems)
  baselines.py run_python (code interpreter) · serialize_record (whole record as text)

`TOOLSETS` names the sets offered to a model; `build_toolset(name, patient)` returns the specs (what
the model sees) and the callables (what the harness executes), always including `final_answer`.
"""

from __future__ import annotations

from collections.abc import Callable

from ..knowledge.analytes import lab_values
from ..knowledge.entities import Entities
from ..retrieval import HybridIndex
from ..sources import Patient
from . import baselines, clinical, labs, quality, summary, text
from .base import is_empty, run_tool, spec, validate_args
from .clinical import ClinicalTools
from .labs import LabTools
from .quality import QualityTools
from .summary import FINAL_REPORT, SummaryTools
from .text import TextTools

FINAL_ANSWER = spec(
    "final_answer",
    "Завершить работу и дать ответ врачу. Каждое утверждение подкрепите id строк, скопированными точно в том виде, "
    "в каком они приведены в поле id вывода инструментов.",
    {
        "answer": {"type": "string", "description": "Ответ на русском, кратко и по существу, с числами и единицами"},
        "citations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "id строк-источников, скопированные из вывода инструментов",
        },
        "status": {
            "type": "string",
            "enum": ["answered", "not_found", "uncertain"],
            "description": "not_found — данных в карте нет; uncertain — данные противоречивы или недостоверны",
        },
    },
    ["answer", "citations", "status"],
)

SPECS: dict[str, dict] = {s["name"]: s for mod in (text, labs, quality, clinical, summary) for s in mod.SPECS}

# Named tool sets. "full" is the system; "compact" leaves out the entity tools for weak models (fewer
# choices → fewer selection errors, see the GigaChat-2 Lite results); "code" is the baseline.
TOOLSETS: dict[str, list[str]] = {
    "full": [
        "get_overview",
        "get_lab",
        "list_abnormal_labs",
        "search_notes",
        "get_conclusions",
        "get_diagnoses",
        "get_document",
        "check_consistency",
        "calculate",
        "get_medications",
        "get_studies",
        "get_trend",
    ],
    "compact": [
        "get_overview",
        "get_lab",
        "list_abnormal_labs",
        "search_notes",
        "get_conclusions",
        "get_diagnoses",
        "get_document",
        "check_consistency",
        "calculate",
    ],
    "code": ["run_python"],
    "none": [],
}


# "report": the full set plus the dossier; its final answer is the sectioned report (FINAL_REPORT).
TOOLSETS["report"] = [*TOOLSETS["full"], "build_dossier"]


class EHRTools(TextTools, LabTools, QualityTools, ClinicalTools, SummaryTools):
    """All tools over one patient; heavy state (embeddings, entities) is built once per patient."""

    def __init__(self, p: Patient):
        self.p = p
        self.by_id = p.by_id()
        self.labs = lab_values(p)
        self._init_quality()
        self.search_index = HybridIndex(p)
        self.entities = Entities(p)


def build_toolset(name: str, p: Patient, only: list[str] | None = None) -> tuple[list[dict], dict[str, Callable[..., dict]]]:
    """Specs and callables for a tool set; `only` restricts it further (dynamic tool exposure)."""
    names = TOOLSETS[name]
    if only is not None:
        names = [n for n in names if n in only]
    if name == "code":
        ci = baselines.CodeInterpreter(p)
        return baselines.SPECS + [FINAL_ANSWER], {"run_python": ci.run_python}
    if not names:
        return [FINAL_ANSWER], {}
    tools = EHRTools(p)
    final = FINAL_REPORT if name == "report" else FINAL_ANSWER
    return [SPECS[n] for n in names] + [final], {n: getattr(tools, n) for n in names}


__all__ = ["EHRTools", "FINAL_ANSWER", "FINAL_REPORT", "SPECS", "TOOLSETS", "build_toolset", "is_empty", "run_tool", "validate_args"]
