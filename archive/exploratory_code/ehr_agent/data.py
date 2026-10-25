"""Load clinic EHR exports (xlsx, one row per protocol field) into normalized records.

Two export layouts exist: P001 has no "код пациента" column, P002–P006 do.
Every record gets a stable id ``P002:r123`` (patient + 1-based data row) that the
agent must cite, so answers can be checked against the source row.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import openpyxl

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

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


def _clean(x) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", "").strip()
    return re.sub(r"[ \t]+", " ", s)


def specialty_of(protocol: str, title: str = "") -> str:
    text = f"{title} {protocol}".lower()
    if title:
        # "Прием терапевта", "Первичный прием врача- невролога" etc.
        m = re.search(r"(терапевт|невролог|гастроэнтеролог|эндокринолог|кардиолог|уролог|гинеколог|хирург|онколог|аллерголог|травматолог|маммолог)", title.lower())
        if m:
            return m.group(1)
    for pat, spec in SPECIALTY_RULES:
        if re.search(pat, text):
            return spec
    return "прочее"


def _infer_sex(records: list[Record]) -> str | None:
    text = " ".join(f"{r.protocol} {r.group} {r.param}" for r in records).lower()
    if re.search(r"мошонк|предстательн|пса|простат", text):
        return "M"
    if re.search(r"цервикал|гинеколог|молочн|эндометр|яичник|маммограф", text):
        return "F"
    return None


def load_patient(pid: str, path: Path | None = None) -> Patient:
    path = path or RAW_DIR / f"{pid}.xlsx"
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    header = [(_clean(h)).lower() for h in rows[0]]
    has_patient_col = "код пациента" in header
    off = 1 if has_patient_col else 0
    pat = Patient(pid=pid)
    titles: dict[str, str] = {}
    raw: list[tuple[int, tuple]] = []
    for i, r in enumerate(rows[1:], start=1):
        if not any(c is not None for c in r):
            continue
        raw.append((i, r))
        if _clean(r[5 + off]) == "Название протокола":
            titles.setdefault(_clean(r[0]), _clean(r[6 + off]))
    for i, r in raw:
        visit = _clean(r[0])
        date = _clean(r[1])[:10]
        protocol = _clean(r[3 + off]) or "?"
        title = titles.get(visit, "") if protocol.startswith("Протокол") else ""
        value = _clean(r[6 + off])
        rec = Record(
            id=f"{pid}:r{i}",
            patient=pid,
            row=i,
            visit=visit,
            date=date,
            doctor=_clean(r[2 + off]),
            protocol=protocol,
            group=_clean(r[4 + off]),
            param=_clean(r[5 + off]),
            value=value,
            specialty=specialty_of(protocol, title),
            num=parse_num(value),
            visit_title=title,
        )
        pat.records.append(rec)
    pat.sex = _infer_sex(pat.records)
    ages = RAW_DIR.parent / "assumed_ages.json"
    if ages.exists():
        info = json.loads(ages.read_text(encoding="utf-8")).get(pid)
        if info:
            pat.age = int(info["age"])
            pat.age_note = f"возраст оценён по данным карты ({info['range'][0]}–{info['range'][1]} лет); в выгрузке возраст удалён"
    return pat


@lru_cache(maxsize=None)
def get_patient(pid: str) -> Patient:
    if pid.startswith("K"):  # inpatient PDF case histories (second clinic)
        from .pdf_source import get_pdf_patient
        return get_pdf_patient(pid)
    return load_patient(pid)


def list_patients() -> list[str]:
    return sorted(p.stem for p in RAW_DIR.glob("P*.xlsx"))
