"""Summary layer: lab dynamics, significance rules, the dossier, the report and its checks."""

import pytest
from conftest import rec
from test_agent import Scripted, call

from ehr_agent.agent import run_episode
from ehr_agent.agent.harness import coverage, verdict_mismatches
from ehr_agent.knowledge.critical import lab_level, text_findings
from ehr_agent.knowledge.dynamics import describe, rcv
from ehr_agent.sources.records import Patient
from ehr_agent.tools import EHRTools


@pytest.fixture
def inpatient() -> Patient:
    """Synthetic inpatient stay: troponin rises and falls, ALT normalizes, an ECG with infarction."""
    lab = dict(specialty="лаборатория", group="Биохимия", protocol="Биохимия")
    rows = [
        rec("K09", 1, "Тропонин I высокочувствительный", "850", date="2026-01-10", ref="0:29", unit="ng/l", **lab),
        rec("K09", 2, "Тропонин I высокочувствительный", "4200", date="2026-01-11", ref="0:29", unit="ng/l", **lab),
        rec("K09", 3, "Тропонин I высокочувствительный", "1200", date="2026-01-13", ref="0:29", unit="ng/l", **lab),
        rec("K09", 4, "АЛТ", "95", date="2026-01-10", ref="0:41", unit="Е/л", **lab),
        rec("K09", 5, "АЛТ", "30", date="2026-01-13", ref="0:41", unit="Е/л", **lab),
        rec("K09", 6, "Заключение", "Острый инфаркт миокарда нижней стенки. Синусовый ритм.", protocol="ЭКГ", specialty="ЭКГ",
            visit="E1"),
    ]  # fmt: skip
    return Patient(pid="K09", records=rows, sex="M", age=60)


def test_rcv_and_trend_description():
    assert 10 < rcv("CREA") < 20 and rcv("CRP") > 100
    pts = [{"date": "d1", "value": 95, "flag": "H", "id": "a"}, {"date": "d2", "value": 30, "flag": "N", "id": "b"}]
    d = describe("ALT", pts)
    assert d["direction"] == "снижение" and d["course"] == "нормализовался к последнему измерению"
    small = describe("NA", [{"date": "d1", "value": 140, "flag": "N", "id": "a"}, {"date": "d2", "value": 141, "flag": "N", "id": "b"}])
    assert small["direction"] == "без значимых изменений"


def test_significance_levels():
    assert lab_level("TROP", 50, "H", 29)[0] == "критично"
    assert lab_level("ALT", 130, "H", 41)[0] == "критично"  # ≥ 3× upper limit
    assert lab_level("ALT", 60, "H", 41)[0] == "значимо"
    assert lab_level("K", 6.4, "H", 5.1)[0] == "критично"
    assert lab_level("GLU", 5.0, "N", 6.1)[0] == "фон"


def test_text_findings_skip_negated_and_benign():
    assert text_findings("Тромбоз глубоких вен голени справа.") == ["тромбоз или эмболия"]
    assert text_findings("Признаков флеботромбоза на момент осмотра нет.") == []
    assert text_findings("Вены проходимы, без признаков тромбоза.") == []
    assert text_findings("Доброкачественное новообразование кожи.") == []


def test_get_trend_and_dossier(inpatient):
    t = EHRTools(inpatient)
    trop = t.get_trend("тропонин")["items"][0]
    assert trop["peak"]["value"] == 4200 and trop["n"] == 3
    d = t.build_dossier()
    labels = [m["label"] for m in d["must_mention"]]
    assert any(lbl.startswith("Тропонин") and "4200" in lbl for lbl in labels)
    assert any("инфаркт" in lbl for lbl in labels)
    alt = next(x for x in d["labs"] if x["analyte"].startswith("АЛТ"))
    assert "нормализовался" in alt["trend"] and alt["flag"] == "N"


def test_mchc_in_g_per_dl_is_not_an_identity_error():
    lab = dict(specialty="лаборатория", group="Общий анализ крови", protocol="Общий анализ крови")
    rows = [rec("K09", 1, "Гемоглобин", "137", unit="г/л", ref="120:160", **lab),
            rec("K09", 2, "Гематокрит", "42.6", unit="%", ref="36:48", **lab),
            rec("K09", 3, "Средняя концентрация гемоглобина в эритроците", "32.2", unit="г/дл", ref="30:38", **lab)]  # fmt: skip
    assert EHRTools(Patient(pid="K09", records=rows, sex="M", age=60))._lab_identity_issues() == []


def test_report_checks(inpatient):
    t = EHRTools(inpatient)
    short = {"answer": "## Ключевые выводы\nАЛТ 30 Е/л — норма [K09:r5]", "citations": ["K09:r5"]}
    _, missing = coverage(short, t.must_mention())
    assert {m["section"] for m in missing} >= {"labs", "studies"}
    wrong = {"answer": "Тропонин 1200 ng/l — в норме, снижение [K09:r3]"}
    assert any("норма" in m for m in verdict_mismatches(wrong, t.dossier()["labs"]))


def test_report_loop_repairs_missing_items(inpatient):
    partial = call("final_answer", key_findings="АЛТ нормализовалась [K09:r5]", citations=["K09:r5"], status="answered")
    full = call(
        "final_answer",
        key_findings="Инфаркт миокарда по ЭКГ — критично [K09:r6]",
        labs="Тропонин I: пик 4200 ng/l — критично, снижение к 1200 [K09:r2, K09:r3]\nАЛТ 95 → 30 Е/л — нормализовалась [K09:r4, K09:r5]",
        citations=["K09:r2", "K09:r3", "K09:r4", "K09:r5", "K09:r6"],
        status="answered",
    )
    ep = run_episode(Scripted([partial, full]), inpatient, "Составьте итоговую сводку", "report")
    assert ep["trace"][0]["name"] == "build_dossier"
    assert any(s.get("action") == "verify" and "missing" in s["message"] for s in ep["trace"] if s["type"] == "harness")
    assert ep["final"]["answer"].startswith("## Ключевые выводы") and "Тропонин" in ep["final"]["answer"]


def test_report_sections_merge_and_debris():
    from ehr_agent.tools.summary import meaningful, merge_reports

    assert not meaningful("\n} ") and not meaningful("не применимо") and not meaningful("error: Отчёт неполон")
    prev = {"key_findings": "Инфаркт миокарда [K09:r6]", "labs": "Тропонин 4200 [K09:r2]", "citations": ["K09:r2"]}
    new = {"key_findings": "Инфаркт миокарда [K09:r6]", "labs": "\n}", "data_quality": "Вес 100 и 120 кг [K09:r7]", "citations": ["K09:r7"]}
    merged = merge_reports(prev, new)
    assert "Тропонин" in merged["answer"] and "Вес" in merged["answer"] and merged["citations"] == ["K09:r2", "K09:r7"]
