"""Data sources. `get_patient(pid)` is the single entry point: `P…` ids are check-up exports (xlsx),
`K…` ids are inpatient case histories (PDF). Both yield the same `Patient`/`Record` model."""

from __future__ import annotations

from functools import cache

from .records import BOILERPLATE, CONCLUSION_FIELDS, Patient, Record, parse_num, specialty_of


@cache
def get_patient(pid: str) -> Patient:
    if pid.startswith("K"):
        from .inpatient_pdf import load_pdf_patient

        return load_pdf_patient(pid)
    from .checkup_xlsx import load_checkup_patient

    return load_checkup_patient(pid)


def list_patients(source: str | None = None) -> list[str]:
    """All patient ids; source='checkup' or 'pdf' to restrict."""
    from .checkup_xlsx import list_checkup_patients
    from .inpatient_pdf import list_pdf_patients

    ids = (list_checkup_patients() if source in (None, "checkup") else []) + (list_pdf_patients() if source in (None, "pdf") else [])
    return ids


__all__ = ["BOILERPLATE", "CONCLUSION_FIELDS", "Patient", "Record", "get_patient", "list_patients", "parse_num", "specialty_of"]
