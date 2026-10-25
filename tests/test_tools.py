"""Tool layer on a synthetic check-up."""

from ehr_agent.tools import EHRTools, build_toolset, run_tool, validate_args
from ehr_agent.tools.base import fit


def test_get_lab_value_interval_and_flag(checkup):
    t = EHRTools(checkup)
    out = t.get_lab("ЛПНП")
    item = out["items"][0]
    assert out["found"] and item["value"] == 4.1 and item["flag"] == "H" and item["id"] == "P009:r2"


def test_get_lab_not_measured_is_explicit(checkup):
    out = EHRTools(checkup).get_lab("ферритин")
    assert out["found"] is False


def test_shifted_panel_is_flagged_everywhere(checkup):
    t = EHRTools(checkup)
    issues = {i["type"] for i in t.check_consistency()["issues"]}
    assert {"lab_identity", "implausible", "conflict"} <= issues
    hgb = t.get_lab("гемоглобин")
    assert "НЕДОСТОВЕРНО" in hgb["items"][0]["quality"] and "warning" in hgb
    assert t.get_overview()["data_quality_warnings"]


def test_anthropometry_conflict_via_get_lab(checkup):
    out = EHRTools(checkup).get_lab("вес")
    assert {v["value"] for v in out["items"]} == {75.0, 95.0} and "warning" in out


def test_abnormal_panel_filter_and_unknown_panel(checkup):
    t = EHRTools(checkup)
    lip = t.list_abnormal_labs(panel="липидограмма")
    assert {i["analyte"] for i in lip["items"]} == {"Холестерин общий", "ЛПНП"}
    assert "error" in t.list_abnormal_labs(panel="ерунда")


def test_search_returns_document_with_conclusion(checkup):
    out = EHRTools(checkup).search_notes("желчный пузырь")
    rows = [r["id"] for d in out["documents"] for r in d["rows"]]
    assert "P009:r9" in rows


def test_medications_from_specialist(checkup):
    out = EHRTools(checkup).get_medications(drug="аторвастатин")
    m = out["medications"][0]["mentions"][0]
    assert m["dose"] == "20 мг" and m["phase"] == "назначение специалиста (чекап)"


def test_calculators():
    from ehr_agent.sources.records import Patient

    t = EHRTools(Patient(pid="P009", records=[], sex="M", age=50))
    assert t.calculate("bmi", weight_kg=80, height_cm=180)["result"] == 24.69
    assert 90 < t.calculate("egfr", creatinine=80)["result"] < 110
    assert "error" in t.calculate("bmi", weight_kg=80)


def test_toolsets_and_validation(checkup):
    specs, fns = build_toolset("compact", checkup)
    names = [s["name"] for s in specs]
    assert "get_medications" not in names and names[-1] == "final_answer"
    spec = next(s for s in specs if s["name"] == "search_notes")
    assert validate_args(spec, {"query": "x", "top_k": "3"}) == ({"query": "x", "top_k": 3}, None)
    assert validate_args(spec, {})[1].startswith("Не передан")
    assert "error" in run_tool(fns, "get_document", {"document_id": "нет такого"})


def test_fit_truncates_lists_with_next_offset():
    out = fit({"items": [{"x": "y" * 100}] * 100}, 1000)
    assert len(out["items"]) < 100 and out["next_offset"] == len(out["items"])
