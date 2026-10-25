"""Text tools: navigate documents, search free text, collect conclusions and diagnoses."""

from __future__ import annotations

import re

from ..retrieval import document_context
from ..retrieval.bm25 import expand_synonyms, tokens
from ..sources import BOILERPLATE, CONCLUSION_FIELDS
from .base import OFFSET, page, spec

SPECS = [
    spec(
        "get_overview",
        "Обзор карты: пол, даты, документы по специальностям (document_id · дата · источник), "
        "предупреждения о качестве данных. Вызывайте первым, если не знаете, где искать.",
    ),
    spec(
        "search_notes",
        "Поиск по тексту карты (жалобы, анамнез, протоколы исследований, заключения, назначения, выписка). "
        "Понимает синонимы, бытовые формулировки и классы препаратов (например 'антитромботическая терапия', "
        "'жиры в крови'). Возвращает документы: найденные строки вместе с заключением документа.",
        {
            "query": {"type": "string", "description": "Ключевые слова"},
            "specialty": {"type": "string", "description": "Необязательно: специальность или исследование (УЗИ, кардиолог, офтальмолог…)"},
            "top_k": {"type": "integer", "description": "Сколько результатов (1–12), по умолчанию 6"},
        },
        ["query"],
    ),
    spec(
        "get_conclusions",
        "Заключения, диагнозы (включая МКБ), рекомендации и назначения специалистов. Без specialty — краткий "
        "вид по документам; со specialty — полный текст. Основной инструмент для сводки и сопоставления мнений врачей.",
        {"specialty": {"type": "string", "description": "Необязательно: специальность"}, **OFFSET},
    ),
    spec(
        "get_diagnoses",
        "Компактный список всех диагнозов: коды МКБ-10 с названиями и текстовые диагнозы, с указанием специалиста и id строки.",
    ),
    spec(
        "get_document",
        "Все поля одного документа по document_id из get_overview (постранично).",
        {"document_id": {"type": "string"}, **OFFSET},
        ["document_id"],
    ),
]


class TextTools:
    def _docs_matching(self, query: str) -> list[dict]:
        """Documents whose name or specialty matches the query (used when a text search finds nothing)."""
        terms = [t for t in tokens(expand_synonyms(query)) if len(t) > 2]
        seen, out = set(), []
        for r in self.p.records:
            key = r.visit + r.source
            if key not in seen and any(t in tokens(f"{r.source} {r.specialty}") for t in terms):
                seen.add(key)
                out.append({"document_id": r.visit, "date": r.date, "source": r.source})
        return out[:8]

    # ------------------------------------------------------------------ tools

    def get_overview(self) -> dict:
        by_spec: dict[str, list[str]] = {}
        seen = set()
        for r in self.p.records:
            if (r.visit, r.source) in seen:
                continue
            seen.add((r.visit, r.source))
            by_spec.setdefault(r.specialty, []).append(f"{r.visit} · {r.date} · {r.source[:60]}")
        return {
            "patient": self.p.pid,
            "sex": {"M": "мужской", "F": "женский"}.get(self.p.sex or "", "не указан"),
            "dates": sorted({r.date for r in self.p.records}),
            "lab_results": len(self.labs),
            "documents_by_specialty": by_spec,
            "data_quality_warnings": sorted(set(self.suspicious_groups.values())) or None,
        }

    def search_notes(self, query: str, specialty: str | None = None, top_k: int = 6) -> dict:
        top_k = max(1, min(int(top_k), 12))
        hits = [r for _, r in self.search_index.search(query, top_k, specialty)]
        if not hits and specialty:
            hits = [r for _, r in self.search_index.search(query, top_k)]
        if not hits:
            return {
                "documents": [],
                "note": "Ничего не найдено. Отсутствие совпадений НЕ означает, что исследования не было: "
                "проверьте get_overview или get_conclusions(specialty).",
            }
        return {
            "documents": document_context(self.p, hits),
            "note": "Найденные строки (hits) показаны вместе с заключением документа и соседними строками.",
        }

    def get_conclusions(self, specialty: str | None = None, offset: int = 0) -> dict:
        def full(rows):
            return [
                {"id": r.id, "date": r.date, "specialty": r.specialty, "source": r.source, "field": r.param, "text": r.value[:500]}
                for r in rows
            ]

        concl = [r for r in self.p.records if CONCLUSION_FIELDS.match(r.param.strip()) and r.value and r.value not in {".", "-"}]
        if specialty:
            s = specialty.lower()
            items = full([r for r in concl if s in r.specialty.lower() or s in r.source.lower()])
            if not items:  # fall back to documents whose *name* matches (e.g. "МРТ" → "Магнитно-резонансная томография")
                docs = {d["document_id"] for d in self._docs_matching(specialty)}
                items = [{k: v for k, v in it.items() if k != "specialty"} for it in full([r for r in concl if r.visit in docs])]
            if not items:
                return {
                    "items": [],
                    "note": f"Заключений не найдено. Доступные специальности: {sorted({r.specialty for r in self.p.records})}",
                }
            chunk, meta = page(items, offset, 40)
            return {"items": chunk, **meta}
        by_doc: dict[tuple, dict] = {}
        for r in concl:
            if re.match(r"рекомендации$", r.param.strip(), re.I) and "не является диагнозом" in r.value:
                continue  # standard disclaimer of ultrasound reports
            d = by_doc.setdefault((r.visit, r.source), {"source": r.source[:50], "specialty": r.specialty, "ids": [], "text": ""})
            d["ids"].append(r.id)
            if len(d["text"]) < 160:
                d["text"] = (d["text"] + " | " if d["text"] else "") + r.value[: 160 - len(d["text"])]
        chunk, meta = page(list(by_doc.values()), offset, 60)
        return {
            "items": chunk,
            "mode": "compact",
            **meta,
            "note": "Краткий вид по документам. Для полного текста вызовите get_conclusions(specialty=…).",
        }

    def get_diagnoses(self) -> dict:
        coded: dict[str, dict] = {}
        coded_visits = set()
        for r in self.p.records:
            for code, name in re.findall(r"\{([A-Z]\d{2}(?:\.\d{1,2})?)\}\s*([^{}]*)", r.value):
                e = coded.setdefault(code, {"icd10": code, "name": name.strip()[:80], "specialists": [], "ids": []})
                if r.specialty not in e["specialists"]:
                    e["specialists"].append(r.specialty)
                e["ids"].append(r.id)
                coded_visits.add(r.visit)
        text = [
            {"id": r.id, "specialty": r.specialty, "text": r.value[:160]}
            for r in self.p.records
            if re.match(r"^диагноз", r.param.strip(), re.I)
            and r.value not in {".", "-", ""}
            and r.visit not in coded_visits
            and not re.search(r"\{[A-Z]\d{2}", r.value)
        ]
        return {
            "icd10": list(coded.values()),
            "text_diagnoses": text,
            "note": "icd10 — уникальные коды МКБ-10; text_diagnoses — диагнозы из документов без кода.",
        }

    def get_document(self, document_id: str, offset: int = 0) -> dict:
        recs = [r for r in self.p.records if r.visit == str(document_id).strip()]
        if not recs:
            return {"error": f"Документ {document_id} не найден. Возьмите document_id из get_overview."}
        items = [{"id": r.id, "group": r.group, "param": r.param, "value": r.value[:350]} for r in recs if not BOILERPLATE.match(r.param)]
        chunk, meta = page(items, offset, 40)
        return {"document_id": document_id, "date": recs[0].date, "source": recs[0].source, "items": chunk, **meta}
