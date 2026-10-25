"""PDF parsing and PII masking on made-up text; the scorer's answer readers."""

from ehr_agent.eval.score import CONFLICT_RX, NO_CONFLICT_RX, _mentions, correctness
from ehr_agent.sources.inpatient_pdf import _identity, _lab_rows, _mask
from ehr_agent.sources.records import parse_num, specialty_of


def test_parse_num_and_specialty():
    assert parse_num("4,88") == 4.88 and parse_num("<2.00") == 2.0 and parse_num("отр") is None
    assert specialty_of("УЗИ сердца (эхокардиография)") == "УЗИ сердца"
    assert specialty_of("Протокол приёма v1.", "Первичный прием врача- невролога") == "невролог"


def test_lab_table_rows_with_split_flag():
    lines = [
        "Коагулограмма",
        "Наименование",
        "Результат",
        "Ед.изм.",
        "Норма",
        "Флаг",
        "Дата",
        "МНО",
        "1.05",
        "_",
        "0.8:1.2",
        "Норм",
        "15.12.2025 15:02:56",
        "Концентрация фибриногена",
        "5.68",
        "г/л",
        "2:4",
        "Повы",
        "ш",
        "15.12.2025 15:02:56",
        "Подпись врача:",
    ]
    rows = _lab_rows(lines)
    assert rows[0] == {"name": "МНО", "value": "1.05", "unit": "", "norm": "0.8:1.2", "flag": "Норм", "date": "15.12.2025 15:02:56"}
    assert rows[1]["flag"] == "Повыш" and rows[1]["unit"] == "г/л"


def test_pii_masking_on_synthetic_text():
    text = (
        "Пациент: Иванов Иван Иванович. Врач-кардиолог Петрова Анна Сергеевна. Медсестра Сидоров П. П. "
        "Телефон: +7(3452)560800. Адрес: 625000, Тюменская обл, г Тюмень, ул Ленина, д. 1, кв. 2. Пациент Иванов жалуется."
    )
    out = _mask(text, ["Иванов", "Иван", "Иванович"])
    for leak in ("Иванов", "Петрова", "Сидоров", "560800", "Ленина"):
        assert leak not in out
    assert "[ПАЦИЕНТ]" in out and "[ФИО]" in out and "[ТЕЛЕФОН]" in out
    # registry blocks print the name in capitals; narrative text declines it
    out = _mask("Идентификаторы: ИВАНОВ ИВАН ИВАНОВИЧ. Осмотрен Ивановым. Врач СМИРНОВА ОЛЬГА ПЕТРОВНА.", ["Иванов", "Иван", "Иванович"])
    assert "ИВАНОВ" not in out and "Ивановым" not in out and "СМИРНОВА" not in out


def test_identity_from_title_page():
    page = (
        "Фамилия, имя, отчество (при наличии) пациента:\nИванов Иван Иванович\nДата рождения:\n10 декабря 1964 г.\n"
        "Дата и время поступления\n15 декабря 2025 г. время: 13 час.\nПол:\nМужской"
    )
    ident = _identity(page)
    assert ident["tokens"] == ["Иванов", "Иван", "Иванович"] and ident["dob"].year == 1964 and ident["sex"] == "M"


def test_conflict_readers():
    assert NO_CONFLICT_RX.search("не найдено расхождений в росте и весе")
    assert not NO_CONFLICT_RX.search("рост и вес не согласуются")
    assert CONFLICT_RX.search("вес не согласуется: 75 и 95 кг")


def test_abnormal_mentions_ignore_normal_clauses():
    assert _mentions("ЛПНП 4.1 выше нормы, ЛПВП 1.3 — в норме.") == {"LDL"}


def test_numeric_and_uncertain_gold():
    q = {"gold": {"type": "numeric", "value": 6.31}}
    assert correctness(q, {"answer": "ЛПНП 6,31 ммоль/л"})[0] == 1.0
    u = {"gold": {"type": "uncertain", "trap": r"33[.,]3"}}
    assert correctness(u, {"answer": "Гемоглобин 33,3 г/л", "status": "answered"})[1]["trapped"]
    assert correctness(u, {"answer": "Данные недостоверны: вероятен сдвиг", "status": "uncertain"})[0] == 1.0
