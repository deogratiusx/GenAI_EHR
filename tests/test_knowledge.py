"""Analyte catalog, reference policy, entity extraction."""

from conftest import rec

from ehr_agent.knowledge.analytes import BY_CODE, analyte_of, implausible, parse_ref, resolve_analyte
from ehr_agent.knowledge.entities import Entities
from ehr_agent.knowledge.lexicon import expand_query
from ehr_agent.knowledge.reference import resolve_reference
from ehr_agent.sources.records import Patient


def test_resolve_analyte_synonyms():
    assert resolve_analyte("ЛПНП")[0].code == "LDL"
    assert resolve_analyte("ldl")[0].code == "LDL"
    assert resolve_analyte("гемоглобин")[0].code == "HGB"
    assert resolve_analyte("ТТГ")[0].code == "TSH"
    assert resolve_analyte("абракадабра") == []


def test_lis_alias_with_typo_maps_to_catalog():
    r = rec("K09", 1, "Блирубин прямой", "8.1", specialty="лаборатория", group="Биохимия")
    assert analyte_of(r).code == "DBIL"


def test_parse_ref_formats():
    assert parse_ref("9.2:13.9") == (9.2, 13.9)
    assert parse_ref("21,2 - 34,8") == (21.2, 34.8)
    assert parse_ref("<5") == (None, 5.0)
    assert parse_ref("Не обнаружены") == (None, None)


def test_implausible_only_without_lab_blank():
    hgb = BY_CODE["HGB"]
    assert implausible(hgb, 33.3)  # catalog units, impossible haemoglobin
    assert not implausible(hgb, 140)
    # a lab blank defines its own units and a huge deviation is pathology, not an export error
    assert not implausible(BY_CODE["MCHC"], 33.9, lab_ref="32:36")


def test_reference_policy_prefers_lab_blank():
    r = rec("K09", 2, "АЛТ", "117", specialty="лаборатория", group="Биохимия", ref="0:41", lab_flag="Повыш")
    res = resolve_reference(r, BY_CODE["ALT"], "M", 60, policy="layered")
    assert res["source"] == "бланк лаборатории" and res["flag"] == "H"


def test_reference_policy_guideline_limit_for_lipids():
    r = rec("P009", 3, "ЛПНП", "3.4", specialty="лаборатория")
    res = resolve_reference(r, BY_CODE["LDL"], "M", 50, policy="layered")
    assert res["source"] == "порог клинических рекомендаций" and res["flag"] == "H"


def test_reference_policy_indicative_fallback():
    r = rec("P009", 4, "Глюкоза", "5.0", specialty="лаборатория")
    res = resolve_reference(r, BY_CODE["GLU"], "M", 50, policy="indicative")
    assert res["source"] == "ориентировочный справочник" and res["flag"] == "N"


def test_ontology_expansion_links_class_and_drugs():
    q = expand_query("антитромботическая терапия при выписке").lower()
    assert "тикагрелор" in q and "ривароксабан" in q
    assert "лпнп" in expand_query("что с жирами в крови").lower()


def test_medication_phase_from_field_label():
    rows = [
        rec("K09", 1, "Рекомендации на этап стационарного лечения", "Ривароксабан 15 мг 2 раза в день", visit="p71"),
        rec("K09", 2, "Рекомендации на амбулаторный этап", "Ривароксабан 20 мг 1 раз в день", visit="p71"),
        rec("K09", 3, "История настоящего заболевания", "Постоянно принимает лозартан 50 мг", visit="p20"),
        rec("K09", 4, "Описание операции", "После введения гепарина 75 ед. на кг массы тела", visit="p23"),
    ]
    meds = Entities(Patient(pid="K09", records=rows)).meds
    phases = {(m.drug, m.dose): m.phase for m in meds}
    assert phases[("ривароксабан", "15 мг")] == "стационар"
    assert phases[("ривароксабан", "20 мг")] == "при выписке"
    assert phases[("лозартан", "50 мг")] == "до госпитализации (постоянно принимал)"
    assert phases[("гепарин", "75 ед")].startswith("во время")


def test_trade_names_normalized_and_absence_reported():
    rows = [rec("K09", 1, "План лечения", "Брилинта 90 мг 2 раза в день"), rec("K09", 2, "План лечения", "Форсига 10 мг утром")]
    e = Entities(Patient(pid="K09", records=rows))
    assert {m.drug for m in e.meds} == {"тикагрелор", "дапаглифлозин"}
    out = e.medications(drug="варфарин")
    assert out["found"] is False and out["checked_records"] == 2
