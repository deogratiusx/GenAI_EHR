"""Source 1: check-up exports of a private clinic (xlsx, one row per protocol field).

Two export layouts exist (with and without a "код пациента" column). Decimal commas, section
header rows and the doctor type hidden in a "Название протокола" field are handled here. Ages were
removed by anonymization; an estimate from record clues is read from `data/assumed_ages.json`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import openpyxl

from ..config import DATA_DIR
from .records import Patient, Record, parse_num, specialty_of

RAW_DIR = DATA_DIR / "raw"
AGES_FILE = DATA_DIR / "assumed_ages.json"


def _clean(x) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", "").strip()
    return re.sub(r"[ \t]+", " ", s)


def _infer_sex(records: list[Record]) -> str | None:
    text = " ".join(f"{r.protocol} {r.group} {r.param}" for r in records).lower()
    if re.search(r"мошонк|предстательн|пса|простат", text):
        return "M"
    if re.search(r"цервикал|гинеколог|молочн|эндометр|яичник|маммограф", text):
        return "F"
    return None


def load_checkup_patient(pid: str, path: Path | None = None) -> Patient:
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
    ages = AGES_FILE
    if ages.exists():
        info = json.loads(ages.read_text(encoding="utf-8")).get(pid)
        if info:
            pat.age = int(info["age"])
            pat.age_note = f"возраст оценён по данным карты ({info['range'][0]}–{info['range'][1]} лет); в выгрузке возраст удалён"
    return pat


def list_checkup_patients() -> list[str]:
    return sorted(p.stem for p in RAW_DIR.glob("P*.xlsx"))
