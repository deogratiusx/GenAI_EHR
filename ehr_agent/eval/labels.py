"""Display names and ordering for reports. Exploratory runs in runs/ use legacy condition names
(see archive/exploratory_code/README.md); the final system uses the names of `agent.systems`."""

# Final system configurations (agent/systems.py)
SYSTEM_LABELS = {
    "tools": "инструменты (система)",
    "tools_routed": "инструменты, отбор по намерению",
    "tools_prefetch": "инструменты + упреждающий вызов",
    "report": "отчёт: досье + проверки полноты",
    "report_nocheck": "отчёт: досье без проверок",
    "tools_compact": "инструменты, компактный набор",
    "tools_plain": "инструменты без обвязки",
    "code": "run_python (код)",
    "record": "вся карта в контексте",
    "record_chunked": "вся карта по фрагментам (map-reduce)",
    "record+tools": "вся карта + инструменты",
}
# Exploratory iterations (runs/*.jsonl before the restructuring)
LEGACY_LABELS = {
    "domain": "инструменты v1",
    "domain_v2": "инструменты v2",
    "domain_v3": "инструменты v3",
    "domain_v3p": "v3 + упреждающий поиск",
    "domain_v3r": "v3 + упреждение по намерению",
    "domain_v4a": "v3 + проверки (T5)",
    "domain_v4": "v4: v3 + сущности + проверки",
    "domain_v41a": "v3 + проверки T5.1",
    "domain_v41": "v4.1: v3 + сущности + T5.1",
    "domain_ri": "v1 + нормы по полу/возрасту",
    "generic": "run_python (код)",
    "fullctx": "вся карта в контексте",
    "hybrid": "вся карта + инструменты v1",
}
COND_RU = {**SYSTEM_LABELS, **LEGACY_LABELS}
CONDS = list(COND_RU)

# Colors: validated categorical slots of the dataviz reference palette (light mode).
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLORS = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(CONDS)}
COLORS.update({"domain": "#2a78d6", "generic": "#eb6834", "fullctx": "#1baf7a", "tools": "#2a78d6", "code": "#eb6834", "record": "#1baf7a"})

# GigaChat is compared on two contrasting tiers only (Lite vs Max); early Pro episodes stay on disk.
EXCLUDED_MODELS = {"GigaChat-2-Pro"}
MODEL_ORDER = ["GigaChat-2", "GigaChat-2-Max", "Qwen3.5-4B-4bit", "Med-Qwen-3.5-4B-it-4bit"]
SHORT = {"GigaChat-2": "Giga Lite", "GigaChat-2-Max": "Giga Max", "Qwen3.5-4B-4bit": "Qwen3.5-4B", "Med-Qwen-3.5-4B-it-4bit": "Med-Qwen-4B"}
SOURCE = {
    "test": "таблица (чекап)",
    "pdf": "PDF (003/у)",
    "para": "таблица, перефраз.",
    "summary": "таблица: сводка",
    "pdf_summary": "PDF: сводка",
    "probe": "таблица: проверочный",
    "probe3": "проверочный v3",
    "probe4": "проверочный v4",
    "final": "финальный слепой набор",
}
