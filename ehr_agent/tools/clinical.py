"""Clinical-entity tools: medications and instrumental studies, extracted for every source (`knowledge/entities.py`)."""

from __future__ import annotations

from .base import OFFSET, page, spec

SPECS = [
    spec(
        "get_medications",
        "Назначения препаратов из всех документов (листы назначений, выписка, анамнез, догоспитальный этап, "
        "назначения специалистов): препарат (МНН, торговые названия приводятся к МНН), класс, доза, кратность, путь, фаза, "
        "дата, id. Фильтры: drug (название), drug_class (антиагрегант, антикоагулянт, гиполипидемический, бета-блокатор, "
        "блокатор РАС, диуретик, глифлозин, ИПП…), phase (при выписке, стационар, до госпитализации, догоспитальный, во время вмешательства). "
        "Если препарат не упоминается — found=false.",
        {
            "drug": {"type": "string", "description": "Название препарата"},
            "drug_class": {"type": "string", "description": "Класс препаратов"},
            "phase": {"type": "string", "description": "Фаза: при выписке / стационар / до госпитализации / догоспитальный"},
            **OFFSET,
        },
    ),
    spec(
        "get_studies",
        "Инструментальные исследования: ЭКГ, ЭхоКГ, холтер, УЗИ, КТ, МРТ, эндоскопия, коронарография, "
        "спирография, эластометрия — дата, заключение и ключевые значения (ФВ, ЧСС, ритм).",
        {"kind": {"type": "string", "description": "Тип исследования"}},
    ),
]


class ClinicalTools:
    def get_medications(self, drug: str | None = None, drug_class: str | None = None, phase: str | None = None, offset: int = 0) -> dict:
        res = self.entities.medications(drug, drug_class, phase)
        if res.get("medications"):
            res["medications"], meta = page(res["medications"], offset, 25)
            res.update(meta)
        return res

    def get_studies(self, kind: str | None = None) -> dict:
        return self.entities.study(kind)
