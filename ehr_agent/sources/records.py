"""Source-independent record model: every source is converted into a list of `Record`.

A record is one field of one document (a lab value, a protocol line, a text block of a PDF page) with
a stable citation id: `P003:r527` (check-up export, data row 527) or `K02:p21:b3` (PDF, page 21,
block 3; `lN` for lab-table rows). Tools, retrieval, the agent and the scorer only see `Record`s.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Field names that hold a document's conclusion, diagnosis, recommendations or prescriptions.
CONCLUSION_FIELDS = re.compile(
    r"^(заключение|диагноз|диагноз по мкб|диагноз клинический|рекомендации|"
    r"рекомендации \(немедикаментозное лечение\)|лекарственные назначения|план обследования|"
    r"заключение узи сердца)\b",
    re.I,
)
# Administrative fields repeated in every protocol; never indexed or shown.
BOILERPLATE = re.compile(r"^(защита врача|печать назначений|трудоспособность|повторный при[её]м|название протокола)", re.I)

# Protocol name → specialty. Order matters: first match wins.
SPECIALTY_RULES: list[tuple[str, str]] = [
    (r"лаборант|преаналит", "лаборатория"),
    (r"кардиолог", "кардиолог"),
    (r"офтальмолог|рефракц", "офтальмолог"),
    (r"дерматовенеролог", "дерматовенеролог"),
    (r"уролог", "уролог"),
    (r"отоларинголог", "отоларинголог"),
    (r"хирург", "хирург"),
    (r"гинеколог", "гинеколог"),
    (r"травматолог|ортопед", "травматолог-ортопед"),
    (r"онколог", "онколог"),
    (r"маммограф", "маммография"),
    (r"дневник|осмотр деж", "стационар/документы"),
    (r"эхокардиограф|узи сердца", "УЗИ сердца"),
    (r"узи|ультразвук|доплер|триплекс|магистральн", "УЗИ"),
    (r"эластометр|фиброскан", "эластометрия печени"),
    (r"экг|холтер|спирограф", "функциональная диагностика"),
    (r"компьютерная томограф|магнитно-резонанс|кт|мрт", "КТ/МРТ"),
    (r"гастроскоп|колоноскоп|эгдс|гастролайзер", "эндоскопия"),
    (r"анестез|предоперац|эпикриз|стационар|идс|согласие|опросный|таблица учета", "стационар/документы"),
    (r"протокол приёма|протокол консультации", "терапевт/консультация"),
]


@dataclass
class Record:
    id: str
    patient: str
    row: int
    visit: str
    date: str
    doctor: str
    protocol: str
    group: str
    param: str
    value: str
    specialty: str
    num: float | None = None
    # For "Протокол приёма v1." the real doctor type lives in the "Название протокола" field.
    visit_title: str = ""
    # Filled by sources that carry them (PDF lab tables): the issuing lab's interval, flag and unit.
    ref: str = ""
    lab_flag: str = ""
    unit: str = ""

    @property
    def source(self) -> str:
        """Human-readable document name: the protocol title (or the doctor type for generic protocols)."""
        return self.visit_title or self.protocol

    def short(self, max_value: int = 300) -> dict:
        v = self.value if len(self.value) <= max_value else self.value[:max_value] + "…"
        return {
            "id": self.id,
            "date": self.date,
            "source": self.visit_title or self.protocol,
            "group": self.group,
            "param": self.param,
            "value": v,
        }


@dataclass
class Patient:
    pid: str
    records: list[Record] = field(default_factory=list)
    sex: str | None = None
    age: int | None = None  # assumed (anonymized export), see data/assumed_ages.json
    age_note: str = ""

    def by_id(self) -> dict[str, Record]:
        return {r.id: r for r in self.records}


_NUM_RE = re.compile(r"^\s*([<>]?)\s*(-?\d+(?:[.,]\d+)?)\s*$")


def parse_num(value: str) -> float | None:
    m = _NUM_RE.match(value or "")
    if not m:
        return None
    return float(m.group(2).replace(",", "."))


def specialty_of(protocol: str, title: str = "") -> str:
    text = f"{title} {protocol}".lower()
    if title:
        # "Прием терапевта", "Первичный прием врача- невролога" etc.
        m = re.search(
            r"(терапевт|невролог|гастроэнтеролог|эндокринолог|кардиолог|уролог|гинеколог|хирург|онколог|аллерголог|травматолог|маммолог)",
            title.lower(),
        )
        if m:
            return m.group(1)
    for pat, spec in SPECIALTY_RULES:
        if re.search(pat, text):
            return spec
    return "прочее"
