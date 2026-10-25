"""Synthetic fixtures. No real patient data: every value, name and text here is made up."""

from __future__ import annotations

import os

import pytest

os.environ["EHR_DENSE"] = "0"  # tests run without the local encoder weights
os.environ.setdefault("EHR_REF_POLICY", "indicative")

from ehr_agent.sources.records import Patient, Record, parse_num  # noqa: E402

CBC = "Клинический анализ крови: общий анализ, лейкоцитарная формула"


def rec(
    pid: str,
    i: int,
    param: str,
    value: str,
    *,
    group: str = "",
    protocol: str = "",
    specialty: str = "прочее",
    visit: str = "D1",
    date: str = "2026-01-10",
    **extra,
) -> Record:
    r = Record(
        id=f"{pid}:r{i}",
        patient=pid,
        row=i,
        visit=visit,
        date=date,
        doctor="",
        protocol=protocol or group,
        group=group,
        param=param,
        value=value,
        specialty=specialty,
        num=parse_num(value),
    )
    for k, v in extra.items():
        setattr(r, k, v)
    return r


@pytest.fixture
def checkup() -> Patient:
    """A small synthetic check-up: lipids, a CBC with an internally inconsistent (shifted) block,
    two documents with conflicting weight, an ultrasound conclusion, a specialist's prescription."""
    lab = dict(specialty="лаборатория", protocol="РМ Лаборанта", visit="L1")
    lip = "Липидограмма (Холестерол, Триглицериды, ЛПВП, ЛПНП, ЛПОНП,  Коэффициент атерогенности)"
    rows = [
        rec("P009", 1, "Холестерин сыворотки крови", "6,4", group=lip, **lab),
        rec("P009", 2, "Исследование уровня липопротеидов низкой плотности (ЛПНП)", "4.1", group=lip, **lab),
        rec("P009", 3, "Исследование уровня липопротеидов высокой плотности (ЛПВП)", "1.3", group=lip, **lab),
        rec("P009", 4, "HGB (Концентрация гемоглобина)", "33.3", group=CBC, **lab),
        rec("P009", 5, "HCT (Гематокрит)", "80.8", group=CBC, **lab),
        rec("P009", 6, "MCHС (Средняя концентрация гемоглобина в эритроцитах)", "14.5", group=CBC, **lab),
        rec("P009", 7, "Вес", "75", group="Протокол первичного приёма.", protocol="Протокол приёма v1.", specialty="терапевт", visit="V1"),
        rec("P009", 8, "Вес", "95", group="Протокол первичного приёма.", protocol="Протокол приёма v1.", specialty="терапевт", visit="V2"),
        rec(
            "P009",
            9,
            "Заключение",
            "Эхографические признаки полипа желчного пузыря.",
            protocol="УЗИ органов брюшной полости",
            specialty="УЗИ",
            visit="U1",
        ),
        rec(
            "P009",
            10,
            "Лекарственные назначения",
            "Аторвастатин 20 мг 1 раз в день вечером длительно.",
            protocol="Прием кардиолога.",
            specialty="кардиолог",
            visit="C1",
        ),
    ]
    return Patient(pid="P009", records=rows, sex="M", age=50, age_note="синтетический пациент")
