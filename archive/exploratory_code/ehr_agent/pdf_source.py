"""Second source type: inpatient case histories (form 003/у) as PDFs stitched from many documents.

The same `Record` model as the clinic table export, so every tool, the agent loop and the scorer
work unchanged. Differences the tools must survive: free text dominates, labs come as tables with
the issuing lab's own reference interval and flag, and the document carries identifiable data.

PII is masked here, before anything leaves the machine: the patient's name, date of birth, card
number, addresses, phones and staff names. Age is computed from the date of birth at admission and
kept; the date of birth itself is not.
"""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from pathlib import Path

from .data import Patient, Record, parse_num, specialty_of

PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "pdf_raw"

MONTHS = {m: i for i, m in enumerate(["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
                                       "сентября", "октября", "ноября", "декабря"], start=1)}
DATE_RX = re.compile(r"\b(\d{2})\.(\d{2})\.(\d{4})\b")
FIO_FULL = re.compile(r"\b[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:вич|вна|ична|ич|оглы|кызы)\b")
FIO_INIT = re.compile(r"\b[А-ЯЁ][а-яё]{2,}(?:-[А-ЯЁ][а-яё]+)?\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.")
PHONE = re.compile(r"(?:\+7|\b8)\s?\(?\d{3,4}\)?[\s\-]?\d{2,3}[\s\-]?\d{2}[\s\-]?\d{2}\b")
ADDRESS = re.compile(r"\b\d{6},\s*[^\n]{5,120}?(?:кв\.?\s*\d+|д\.?\s*\d+[а-я]?)\b", re.I)

PAGE_SPECIALTY = [
    (r"электрокардиограф|\bэкг\b", "функциональная диагностика"),
    (r"эхокардиограф|эхо-?кг|ультразвук|\bузи\b", "УЗИ"),
    (r"коронарограф|стентир|ангиограф|чкв|рентгенэндоваск", "эндоваскулярная хирургия"),
    (r"выписн|эпикриз", "выписной эпикриз"),
    (r"лист\s+назначен|назначени[яй]\s+лекарств|лекарственн\w+\s+назначен", "лист назначений"),
    (r"дневник|осмотр\s+(лечащ|дежур|заведующ)|первичный осмотр", "кардиолог (дневник/осмотр)"),
    (r"анестез|реанимац", "реанимация/анестезиология"),
    (r"консультац", "консультация"),
]


FIO_TURKIC = re.compile(r"\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+(?:оглы|кызы)\b")
STAFF = re.compile(r"((?:[Вв]рач\w*|[Мм]едицинск\w+ (?:сестра|лаборат\w+ техник)|[Фф]ельдшер|[Зз]ав\.? ?отделением)[\w\-() ]{0,40}?\s)"
                   r"([А-ЯЁ][а-яё]{2,}\s+[А-ЯЁ][а-яё]{2,}(?:\s+[А-ЯЁ][а-яё]+)?)")


def _mask(text: str, patient_tokens: list[str]) -> str:
    for tok in patient_tokens:
        if len(tok) > 2:
            text = re.sub(rf"\b{re.escape(tok)}\b", "[ПАЦИЕНТ]", text)
    text = FIO_TURKIC.sub("[ФИО]", text)
    for _ in range(3):  # overlapping titles ("врач-кардиолог врач …") need more than one pass
        new = STAFF.sub(lambda m: m.group(1) + "[ФИО]", text)
        if new == text:
            break
        text = new
    text = FIO_FULL.sub("[ФИО]", text)
    text = FIO_INIT.sub("[ФИО]", text)
    text = ADDRESS.sub("[АДРЕС]", text)
    text = PHONE.sub("[ТЕЛЕФОН]", text)
    return text


def _identity(first_pages: str) -> dict:
    """Patient name tokens, DOB, sex and admission date from the title page of form 003/у."""
    out: dict = {"tokens": [], "dob": None, "sex": None, "admitted": None, "card": None}
    m = re.search(r"пациента:\s*\n\s*([^\n]+)", first_pages)
    if m:
        out["tokens"] = [t for t in re.split(r"\s+", m.group(1).strip()) if t]
    m = re.search(r"Дата рождения:\s*\n?\s*(\d{1,2})\s+([а-я]+)\s+(\d{4})", first_pages)
    if m and m.group(2) in MONTHS:
        out["dob"] = date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    else:
        m = re.search(r"Дата рождения:\s*\n?\s*(\d{2})\.(\d{2})\.(\d{4})", first_pages)
        if m:
            out["dob"] = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    m = re.search(r"Дата и время поступления\s*\n?\s*(\d{1,2})\s+([а-я]+)\s+(\d{4})", first_pages)
    if m and m.group(2) in MONTHS:
        out["admitted"] = date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
    if re.search(r"Пол:\s*\n?\s*Мужской", first_pages):
        out["sex"] = "M"
    elif re.search(r"Пол:\s*\n?\s*Женский", first_pages):
        out["sex"] = "F"
    m = re.search(r"№\s*(\d{8,})", first_pages)
    if m:
        out["card"] = m.group(1)
    return out


def _page_date(text: str) -> str:
    m = DATE_RX.search(text)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else ""


BOILER = re.compile(r"^(Пациент:|№ карты:|Дата рождения:|Пол:|Местожительство:|Назначил:|Исп\. врач:|Исп\. МО:|Телефон:|"
                    r"Адрес:|Подпись врача:|КОПИЯ сформирована|Время выполнения:|Дата выполнения:|\[ПАЦИЕНТ\]|\[ФИО\]|\[АДРЕС\]|\[ТЕЛЕФОН\])")


def _lab_rows(lines: list[str]) -> list[dict]:
    """Rows of a LIS result table: Наименование | Результат | Ед.изм. | Норма | Флаг | Дата (one cell per line)."""
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == "Наименование"
                     and any(x.strip() == "Результат" for x in lines[i + 1:i + 4]))
    except StopIteration:
        return []
    i = start
    while i < len(lines) and not DATE_RX.search(lines[i]) and lines[i].strip() != "Дата":
        i += 1
    i += 1
    rows, cur = [], []
    for raw in lines[i:]:
        l = raw.strip()
        if l.startswith(("Подпись врача", "КОПИЯ")):
            break
        if not l:
            continue
        cur.append(l)
        if DATE_RX.search(l) and len(cur) >= 4:
            name, value, *rest = cur
            date_cell = rest[-1]
            middle = rest[:-1]
            flag = ""
            if middle and re.match(r"^(Норм|Повы|Пони|Выше|Ниже|Крит)", middle[-1]) is None and len(middle) >= 2 \
                    and re.match(r"^(Норм|Повы|Пони|Выше|Ниже|Крит)", middle[-2] or ""):
                flag = middle[-2] + middle[-1]
                middle = middle[:-2]
            elif middle and re.match(r"^(Норм|Повы|Пони|Выше|Ниже|Крит)", middle[-1]):
                flag = middle[-1]
                middle = middle[:-1]
            if re.match(r"^(Норм|Повы|Пони)", date_cell):  # flag and date on one line
                flag = date_cell.split()[0]
            unit = middle[0] if middle else ""
            norm = middle[1] if len(middle) > 1 else ""
            rows.append({"name": name, "value": value, "unit": "" if unit == "_" else unit, "norm": norm,
                         "flag": flag, "date": date_cell})
            cur = []
    return rows


def _blocks(text: str, limit: int = 600) -> list[tuple[str, str]]:
    """Split free text into (label, chunk): 'Label: value' lines start a new block; long blocks are cut."""
    out: list[tuple[str, str]] = []
    label, buf = "текст", []
    for line in text.splitlines():
        s = line.strip()
        if not s or BOILER.match(s):
            continue
        m = re.match(r"^([А-ЯЁA-Z][^:]{2,60}):\s*(.*)$", s)
        if m and buf:
            out.append((label, " ".join(buf)))
            label, buf = m.group(1).strip(), [m.group(2)] if m.group(2) else []
        elif m:
            label, buf = m.group(1).strip(), [m.group(2)] if m.group(2) else []
        else:
            buf.append(s)
        if sum(len(b) for b in buf) > limit:
            out.append((label, " ".join(buf)))
            buf = []
    if buf:
        out.append((label, " ".join(buf)))
    return [(l, v) for l, v in out if len(v) > 2]


def load_pdf_patient(pid: str) -> Patient:
    import pymupdf

    manifest = json.loads((PDF_DIR / "manifest.json").read_text(encoding="utf-8"))[pid]
    files = [PDF_DIR / d["file"] for d in manifest["docs"]]
    pages: list[tuple[str, int, str]] = []
    for f in files:
        doc = pymupdf.open(str(f))
        pages += [(f.stem.split("_", 2)[-1], i + 1, doc.load_page(i).get_text("text") or "") for i in range(doc.page_count)]
        doc.close()
    ident = _identity("\n".join(t for _, _, t in pages[:3]))
    if ident["sex"] is None:  # the title page only lists the options; LIS blanks state it explicitly
        allt = "\n".join(t for _, _, t in pages)
        ident["sex"] = "M" if re.search(r"Пол:\s*\n?\s*Мужской", allt) else "F" if re.search(r"Пол:\s*\n?\s*Женский", allt) else None
    tokens = ident["tokens"] + ([ident["card"]] if ident["card"] else [])
    pat = Patient(pid=pid, sex=ident["sex"])
    if ident["dob"] and ident["admitted"]:
        a, b = ident["admitted"], ident["dob"]
        pat.age = a.year - b.year - ((a.month, a.day) < (b.month, b.day))
        pat.age_note = "возраст на дату госпитализации (из бланка 003/у; дата рождения скрыта)"
    dob_txt = [ident["dob"].strftime("%d.%m.%Y")] if ident["dob"] else []
    row = 0
    for kind, pno, raw in pages:
        text = _mask(raw, tokens)
        for d in dob_txt:
            text = text.replace(d, "[ДАТА РОЖДЕНИЯ]")
        text = re.sub(r"Дата рождения:\s*\n?[^\n]*", "Дата рождения: [СКРЫТО]", text)
        lines = text.splitlines()
        title = next((l.strip() for l in lines if l.strip() and not BOILER.match(l.strip())), f"страница {pno}")[:90]
        doc_id = f"{pid}-{kind[0]}{pno}"
        pdate = _page_date(text)
        lab = _lab_rows(lines)
        tl = title.lower()
        spec = "лаборатория" if lab else next((s for rx, s in PAGE_SPECIALTY if re.search(rx, tl)), None) \
            or next((s for rx, s in PAGE_SPECIALTY if re.search(rx, text[:400].lower())), None) or specialty_of(title)
        if lab:
            for k, r in enumerate(lab, 1):
                row += 1
                d = DATE_RX.search(r["date"])
                rec = Record(id=f"{pid}:p{pno}:l{k}", patient=pid, row=row, visit=doc_id,
                             date=f"{d.group(3)}-{d.group(2)}-{d.group(1)}" if d else pdate, doctor="",
                             protocol=title, group=title, param=r["name"], value=r["value"], specialty="лаборатория",
                             num=parse_num(r["value"]))
                rec.ref, rec.lab_flag, rec.unit = r["norm"], r["flag"], r["unit"]
                pat.records.append(rec)
            continue
        for k, (label, chunk) in enumerate(_blocks(text), 1):
            chunk = _mask(chunk, tokens)  # names split across lines only match after the join
            row += 1
            pat.records.append(Record(id=f"{pid}:p{pno}:b{k}", patient=pid, row=row, visit=doc_id, date=pdate, doctor="",
                                      protocol=title, group=title, param=label[:80], value=chunk,
                                      specialty=spec, num=parse_num(chunk)))
    return pat


@lru_cache(maxsize=None)
def get_pdf_patient(pid: str) -> Patient:
    return load_pdf_patient(pid)


def list_pdf_patients() -> list[str]:
    mf = PDF_DIR / "manifest.json"
    return sorted(json.loads(mf.read_text(encoding="utf-8"))) if mf.exists() else []
