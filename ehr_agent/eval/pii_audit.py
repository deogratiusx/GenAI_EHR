"""Audit of PII masking on the PDF case histories: what the masked records still contain.

For every PDF patient the identity (name, date of birth, card number) is re-read from the raw title
page, then every masked record is searched for residues:

* the patient's surname, first name or patronymic in any inflected form (stem match, not just the
  exact token the masker replaces);
* the date of birth and the card number;
* phones and postal addresses;
* person-like sequences ("Surname I. O.", three capitalised words ending in a patronymic) that the
  staff-name rules missed.

Only counts are printed. Examples go to data/pii_audit.md, which stays local with the rest of data/.

  python -m ehr_agent.eval.pii_audit
"""

from __future__ import annotations

import re

from ..config import DATA_DIR
from ..sources import get_patient, list_patients
from ..sources.inpatient_pdf import ADDRESS, FIO_CAPS, FIO_FULL, FIO_INIT, PDF_DIR, PHONE, _identity

OUT = DATA_DIR / "pii_audit.md"
NAME_KINDS = {"фамилия", "имя", "отчество"}


def _raw_identity(pid: str) -> dict:
    import json

    import pymupdf

    docs = json.loads((PDF_DIR / "manifest.json").read_text(encoding="utf-8"))[pid]["docs"]
    doc = pymupdf.open(str(PDF_DIR / docs[0]["file"]))
    text = "\n".join(doc.load_page(i).get_text("text") or "" for i in range(min(3, doc.page_count)))
    doc.close()
    return _identity(text)


def _checks(ident: dict) -> dict[str, re.Pattern]:
    checks = {}
    for i, tok in enumerate(ident["tokens"][:3]):
        if len(tok) > 3:  # stem: the token minus a typical case ending, any ending allowed
            stem = re.sub(r"(ов|ев|ин|ий|ая|а|я|ь|й)$", "", tok) if i == 0 else tok[:-1]
            checks[("фамилия", "имя", "отчество")[i]] = re.compile(rf"\b{re.escape(stem)}\w*", re.I)
    if ident["dob"]:
        d = ident["dob"]
        checks["дата рождения"] = re.compile(rf"{d:%d}\.{d:%m}\.{d:%Y}|\b{d.day}\s+\w+\s+{d.year}")
    if ident["card"]:
        checks["номер карты"] = re.compile(re.escape(ident["card"]))
    checks.update({"телефон": PHONE, "адрес": ADDRESS, "ФИО полностью": FIO_FULL, "ФИО заглавными": FIO_CAPS, "ФИО с инициалами": FIO_INIT})
    return checks


def audit() -> None:
    lines = ["# PII audit of masked PDF records", ""]
    print(f"{'patient':8} {'records':>7}  residues by kind")
    for pid in list_patients("pdf"):
        p, checks = get_patient(pid), _checks(_raw_identity(pid))
        hits: dict[str, list[str]] = {k: [] for k in checks}
        for r in p.records:
            text = f"{r.param} {r.value} {r.protocol}"
            for kind, rx in checks.items():
                for m in rx.finditer(text):
                    if kind in NAME_KINDS and m.group().islower():
                        continue  # a lowercase common word sharing the stem, not a name
                    hits[kind].append(f"{r.id}: …{text[max(0, m.start() - 40) : m.end() + 40]}…")
        summary = ", ".join(f"{k}={len(v)}" for k, v in hits.items() if v) or "нет"
        print(f"{pid:8} {len(p.records):>7}  {summary}")
        lines += [f"## {pid}", ""] + [f"- **{k}**: {x}" for k, v in hits.items() for x in v[:10]] + [""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"examples: {OUT}")


if __name__ == "__main__":
    audit()
