"""Hand-written benchmark content: questions, gold answers and the patterns that read answers.

Gold types: numeric (the number must appear), set (analyte families named as abnormal; F1),
not_found, codes (ICD-10 F1), conflict, facts (share of fact regexes found), uncertain (data must be
flagged as unreliable), checklist (summary sections). `gold_rows` are the record rows that support
the gold; `benchmark.py` extends them automatically (`support_rows`).

Splits: dev (P001–P002, tools were debugged on them) · test (P003–P006; P005 is the corrupted-export
case) · para (L2/L3 rephrasings of test questions) · summary · pdf · pdf_summary · probe / probe3 /
probe4 (each written before the tool iteration it evaluates).
"""

from __future__ import annotations

# Concept families: how an analyte is recognized in a free-text answer.
FAMILY = {
    "CHOL": r"общ\w*\s+холестерин|холестерин\w*\s+общ|холестерин(?!\w*\s*(лп|липопротеид))|холестерол|\bхс\b(?!\s*лп)",
    "LDL": r"лпнп|ldl|низкой плотности",
    "HDL": r"лпвп|hdl|высокой плотности",
    "TG": r"триглицерид|\bтг\b",
    "ALT": r"\bалт\b|аланинаминотрансфераз|\balt\b",
    "AST": r"\bаст\b|аспартатаминотрансфераз|\bast\b",
    "TBIL": r"билирубин\w*\s+общ|общ\w*\s+билирубин",
    "WBC": r"лейкоцит(?!арн)|\bwbc\b",
    "RBC": r"эритроцит(?!арн)|\brbc\b",
    "HGB": r"гемоглобин(?! в эритроц)|\bhgb\b|\bhb\b",
    "HCT": r"гематокрит|\bhct\b",
    "PLT": r"тромбоцит|\bplt\b",
    "ESR": r"\bсоэ\b|\besr\b",
    "NEU": r"нейтрофил|\bneu",
    "LYM": r"лимфоцит|\blym",
    "MONO": r"моноцит|\bmono",
    "EOS": r"эозинофил|\beos",
    "BASO": r"базофил|\bbaso",
    "MCV": r"\bmcv\b|средний объ[её]м эритроц",
    "MCH": r"\bmch\b|среднее содержание гемоглобина",
    "MCHC": r"\bmchc|средняя концентрация гемоглобина",
    "RDW": r"\brdw",
}
# Families recognized only for the layered policy (their intervals are sourced there); the frozen
# indicative gold keeps them neutral.
FAMILY_LAYERED = {
    "ALP": r"щелочн\w*\s+фосфатаз|\bщф\b|\balp\b",
    "GGT": r"\bггт\b|гамма-?\s?глутамил|гамма-?гт|\bggt\b",
}
FAMILY.update(FAMILY_LAYERED)
FAMILY_OF = {"NEUP": "NEU", "LYMP": "LYM", "MONOP": "MONO", "EOSP": "EOS", "BASOP": "BASO"}
PANELS = {
    "липидограммы": ["CHOL", "LDL", "HDL", "TG", "VLDL", "AI", "LPA"],
    "печёночных проб (АЛТ, АСТ, билирубин, ГГТ, ЩФ)": ["ALT", "AST", "TBIL", "DBIL", "IBIL", "GGT", "ALP"],
    "общего анализа крови": [
        "WBC",
        "RBC",
        "HGB",
        "HCT",
        "PLT",
        "ESR",
        "MCV",
        "MCH",
        "MCHC",
        "RDW",
        "MPV",
        "NEUP",
        "LYMP",
        "MONOP",
        "EOSP",
        "BASOP",
        "NEU",
        "LYM",
        "MONO",
        "EOS",
        "BASO",
    ],
}
LOOKUP_ORDER = ["LDL", "HGB", "TSH", "GLU", "VITD", "CRP", "CHOL", "ALT", "HBA1C", "PLT"]
NOT_FOUND_ORDER = ["HBA1C", "LPA", "INS", "PSA", "FERR", "TESTO", "K", "INR", "CER", "CA125", "IGE"]


def F(*patterns: str) -> list[str]:
    return list(patterns)


MANUAL: dict[str, list[dict]] = {
    "P002": [
        {
            "category": "integrate",
            "question": "Принимает ли пациент постоянно какие-либо лекарства? Согласуются ли сведения в разных протоколах?",
            "gold": {"type": "facts", "facts": F(r"нолипрел", r"не принимает|противореч|расхожд|несоглас|в одном|в другом")},
            "gold_rows": ["P002:r127", "P002:r659"],
        },
        {
            "category": "freetext",
            "question": "Что выявлено при УЗИ органов брюшной полости?",
            "gold": {"type": "facts", "facts": F(r"полип", r"кист\w*\s+(в\s+)?печен|печен\w*.{0,30}кист", r"дивертикул")},
            "gold_rows": ["P002:r83"],
        },
    ],
    "P003": [
        {
            "category": "freetext",
            "question": "Что выявлено при колоноскопии и что было сделано?",
            "gold": {"type": "facts", "facts": F(r"новообразован", r"резекц|удален|петл")},
            "gold_rows": ["P003:r527"],
        },
        {
            "category": "freetext",
            "question": "Какие изменения молочных желез выявлены по УЗИ и маммографии? Укажите категории BI-RADS.",
            "gold": {"type": "facts", "facts": F(r"кист", r"фиброз|дисгормональн|диффузн", r"bi-?rads\D{0,15}2")},
            "gold_rows": ["P003:r75", "P003:r392"],
        },
        {
            "category": "freetext",
            "question": "Какова фракция выброса левого желудочка по ЭхоКГ?",
            "gold": {"type": "numeric", "value": 67, "family": r"фв|фракци"},
            "gold_rows": ["P003:r121"],
        },
        {
            "category": "integrate",
            "question": "Сведите данные о состоянии коленных суставов: что показала МРТ и что заключил травматолог-ортопед?",
            "gold": {"type": "facts", "facts": F(r"мениск", r"разрыв", r"капсульно|связочн|повреждени")},
            "gold_rows": ["P003:r413", "P003:r367"],
        },
        {
            "category": "integrate",
            "question": "Какие находки разных исследований требуют консультации онколога, маммолога или динамического контроля?",
            "gold": {
                "type": "facts",
                "facts": F(
                    r"оча\w*.{0,40}л[её]гк|л[её]гк\w*.{0,40}оча|плевр|лимфатическ",
                    r"онколог",
                    r"новообразован\w*.{0,30}(толст|кишк)|колоноскоп",
                    r"маммолог|молочн",
                ),
            },
            "gold_rows": ["P003:r606", "P003:r607", "P003:r527", "P003:r394"],
        },
    ],
    "P004": [
        {
            "category": "freetext",
            "question": "Что выявлено при гастроскопии?",
            "gold": {"type": "facts", "facts": F(r"эритематозн\w*\s+гастропат")},
            "gold_rows": ["P004:r433"],
        },
        {
            "category": "freetext",
            "question": "Каковы результаты эластометрии печени (жёсткость и CAP)?",
            "gold": {"type": "facts", "facts": F(r"9[.,]2", r"233")},
            "gold_rows": ["P004:r617"],
        },
        {
            "category": "freetext",
            "question": "Какова фракция выброса левого желудочка по ЭхоКГ?",
            "gold": {"type": "numeric", "value": 61, "family": r"фв|фракци"},
            "gold_rows": ["P004:r166"],
        },
        {
            "category": "integrate",
            "question": "Сведите все данные о печени: УЗИ, эластометрия, биохимия и заключение гастроэнтеролога.",
            "gold": {
                "type": "facts",
                "facts": F(
                    r"увеличен\w*.{0,40}печен|печен\w*.{0,20}увеличен|гепатомегал",
                    r"9[.,]2",
                    r"нажбп|стеатоз|стеатогепатоз|жиров",
                    r"\bf2\b|фиброз",
                    r"щелочн\w*\s+фосфатаз|\bщф\b",
                ),
            },
            "gold_rows": ["P004:r408", "P004:r617", "P004:r607", "P004:r608"],
        },
        {
            "category": "integrate",
            "question": "Сведите данные, относящиеся к сердечно-сосудистому риску: диагнозы, ЭКГ, ЭхоКГ, липиды, масса тела.",
            "gold": {
                "type": "facts",
                "facts": F(
                    r"i10|гипертенз|гипертони", r"гипертроф|ремоделир", r"триглицерид|\bтг\b", r"лпвп|высокой плотности", r"34[.,]6|ожирен"
                ),
            },
            "gold_rows": ["P004:r587", "P004:r150", "P004:r189", "P004:r520"],
        },
    ],
    "P006": [
        {
            "category": "freetext",
            "question": "Что выявлено на УЗИ щитовидной железы?",
            "gold": {"type": "facts", "facts": F(r"кист", r"прав", r"ti-?rads\D{0,5}2")},
            "gold_rows": ["P006:r383"],
        },
        {
            "category": "freetext",
            "question": "Что показала гастроскопия и какой диагноз поставлен при выписке?",
            "gold": {"type": "facts", "facts": F(r"кардии", r"гэрб|гастроэзофаг", r"негативн|эзофагит")},
            "gold_rows": ["P006:r444", "P006:r511"],
        },
        {
            "category": "freetext",
            "question": "Какова фракция выброса левого желудочка по ЭхоКГ?",
            "gold": {"type": "numeric", "value": 78, "family": r"фв|фракци"},
            "gold_rows": ["P006:r291"],
        },
        {
            "category": "integrate",
            "question": "Подтверждается ли лабораторными данными диагноз, выставленный кардиологом?",
            "gold": {
                "type": "facts",
                "facts": F(
                    r"e78|гиперлипидем|дислипидем", r"лпнп|низкой плотности", r"триглицерид|\bтг\b", r"подтвержда|соответств|да[,.]"
                ),
            },
            "gold_rows": ["P006:r589"],
        },
        {
            "category": "integrate",
            "question": "Какие находки со стороны ЖКТ выявлены разными методами (УЗИ, гастроскопия, колоноскопия, выписной эпикриз)?",
            "gold": {
                "type": "facts",
                "facts": F(
                    r"желчн|полип",
                    r"поджелудочн",
                    r"кардии",
                    r"гэрб",
                    r"колоноскоп\w*.{0,80}(не\s+обнаруж|без|норм)|толст\w*.{0,60}(не\s+обнаруж|без|норм)",
                ),
            },
            "gold_rows": ["P006:r77", "P006:r444", "P006:r449", "P006:r511"],
        },
    ],
    "P005": [
        {
            "category": "quality",
            "question": "Какой уровень гемоглобина у пациентки?",
            "gold": {"type": "uncertain", "trap": r"33[.,]3"},
            "gold_rows": ["P005:r67"],
        },
        {
            "category": "quality",
            "question": "Есть ли у пациентки признаки анемии по общему анализу крови?",
            "gold": {"type": "uncertain", "trap": r"анеми\w*\s+(тяж|выраж)|выраженн\w*\s+анеми|да,\s*есть"},
            "gold_rows": ["P005:r66", "P005:r67"],
        },
        {
            "category": "quality",
            "question": "Можно ли доверять результатам общего анализа мочи? Есть ли в нём патология?",
            "gold": {"type": "uncertain", "trap": r"протеинури|белок\s*\+\+"},
            "gold_rows": ["P005:r202", "P005:r197"],
        },
        {
            "category": "quality",
            "question": "Какой вес пациентки?",
            "gold": {"type": "uncertain", "trap": r"1[.,]86\s*кг|\b7[.,]0\s*кг"},
            "gold_rows": ["P005:r130", "P005:r310"],
        },
    ],
}


# split "probe": written after the first error analysis and before the second tool iteration.
PROBE: dict[str, list[dict]] = {
    "P003": [
        {
            "category": "freetext",
            "question": "Что показала МРТ головного мозга?",
            "gold": {"type": "facts", "facts": F(r"оча", r"сосудист", r"кист")},
            "gold_rows": ["P003:r401"],
        },
        {
            "category": "integrate",
            "question": "Какой диагноз выставил невролог и какие препараты он назначил?",
            "gold": {"type": "facts", "facts": F(r"мигрен", r"люмбалг", r"напроксен|тералив", r"золмитриптан|эксенза")},
            "gold_rows": ["P003:r380", "P003:r383"],
        },
        {
            "category": "freetext",
            "question": "Что выявлено при УЗИ органов брюшной полости?",
            "gold": {"type": "facts", "facts": F(r"нефроптоз", r"висцероптоз")},
            "gold_rows": ["P003:r477"],
        },
    ],
    "P004": [
        {
            "category": "freetext",
            "question": "Что показало УЗИ вен нижних конечностей?",
            "gold": {"type": "facts", "facts": F(r"перфорант", r"несостоятельн", r"тромбоз")},
            "gold_rows": ["P004:r335"],
        },
        {
            "category": "freetext",
            "question": "Что выявлено при УЗИ предстательной железы?",
            "gold": {"type": "facts", "facts": F(r"увеличен", r"гиперплаз", r"диффузн")},
            "gold_rows": ["P004:r278"],
        },
        {
            "category": "integrate",
            "question": "Какое лечение назначил гастроэнтеролог?",
            "gold": {
                "type": "facts",
                "facts": F(r"хеликобакт|эрадикац", r"амоксициллин|флемоксин", r"кларитромицин", r"эзомепразол|тегопразан|нексиум"),
            },
            "gold_rows": ["P004:r611"],
        },
        {
            "category": "freetext",
            "question": "Какова средняя ЧСС днём и ночью по данным холтеровского мониторирования?",
            "gold": {"type": "facts", "facts": F(r"\b75\b", r"\b57\b")},
            "gold_rows": ["P004:r628"],
        },
    ],
    "P006": [
        {
            "category": "freetext",
            "question": "Что показала КТ органов грудной клетки и какие даны рекомендации?",
            "gold": {"type": "facts", "facts": F(r"бронхит", r"нодуляр|4\s*мм|узел|очаг", r"пульмонолог")},
            "gold_rows": ["P006:r624", "P006:r625"],
        },
        {
            "category": "integrate",
            "question": "Какие диагнозы выставил дерматовенеролог?",
            "gold": {"type": "facts", "facts": F(r"крапивниц", r"акрохордон", r"фибром", r"невус")},
            "gold_rows": ["P006:r333"],
        },
        {
            "category": "freetext",
            "question": "Что выявлено при УЗИ органов брюшной полости в отношении селезёнки и почек?",
            "gold": {"type": "facts", "facts": F(r"спленомегал|селез[её]н\w*.{0,20}увелич", r"почк|члк|чашечно")},
            "gold_rows": ["P006:r77"],
        },
    ],
    "P005": [
        {
            "category": "quality",
            "question": "Какое количество тромбоцитов у пациентки?",
            "gold": {"type": "uncertain", "trap": r"10[.,]7"},
            "gold_rows": ["P005:r74"],
        },
        {
            "category": "quality",
            "question": "Есть ли у пациентки лейкоцитоз?",
            "gold": {"type": "uncertain", "trap": r"57[.,]1|да,?\s+(есть|выявлен|имеется)"},
            "gold_rows": ["P005:r55"],
        },
        {
            "category": "quality",
            "question": "Какой гематокрит у пациентки?",
            "gold": {"type": "uncertain", "trap": r"80[.,]8"},
            "gold_rows": ["P005:r68"],
        },
    ],
}


# Phrasing levels for the same intent and the same gold.
#   L1 precise   — the parameter is named as in the record (the test set above);
#   L2 clinical  — abbreviations, Latin/English names, colleague-to-colleague shorthand;
#   L3 vague     — lay or loose wording, the parameter is implied rather than named.
ANALYTE_LEVELS = {
    "ЛПНП": ("Сколько у него LDL-C?", "Как у пациента с «плохим» холестерином?"),
    "Гемоглобин": ("Какой Hb в ОАК?", "Нет ли у пациента малокровия — что с кровью по красным показателям, конкретно с гемоглобином?"),
    "ТТГ": ("TSH какой?", "Как работает щитовидка по анализам гормонов?"),
    "Глюкоза (плазма)": ("Глюкоза натощак (GLU) сколько?", "Сахар в крови в порядке?"),
    "Липопротеин (a)": (
        "Lp(a) определяли? Значение?",
        "Проверяли ли наследственный фактор риска атеросклероза, тот специальный липопротеин?",
    ),
    "Инсулин": ("Инсулин натощак, HOMA?", "Нет ли инсулинорезистентности по гормонам?"),
    "Церулоплазмин": ("Ceruloplasmin какой?", "Обмен меди проверяли — что с белком, который её переносит?"),
    "CA 125": ("Онкомаркер CA-125?", "Сдавал ли пациент онкомаркер, который обычно смотрят по яичникам?"),
    "Гликированный гемоглобин HbA1c": ("A1c сколько?", "Какой у него средний сахар за последние три месяца?"),
}
PANEL_LEVELS = {
    "липидограммы": ("Что в липидном профиле вне референса (ХС, ЛПНП, ЛПВП, ТГ)?", "Что с жирами в крови?"),
    "печёночных проб": ("Трансаминазы и билирубин — есть отклонения?", "Печень по анализам в порядке?"),
    "общего анализа крови": ("Что не так в ОАК?", "Кровь в целом нормальная?"),
}
MANUAL_LEVELS = {
    "P003-11": ("ФКС: находки и манипуляции?", "Что там с кишечником, что нашли и что сделали?"),
    "P003-12": ("МЖ: УЗИ + ММГ, категории BI-RADS?", "Что с грудью по обследованиям?"),
    "P003-13": ("ФВ ЛЖ по ЭхоКГ?", "Как сердце качает?"),
    "P003-14": ("Колено: МРТ и заключение ортопеда?", "Что с ногой, на что жаловалась и что нашли?"),
    "P003-15": ("Какие находки требуют онконастороженности или контроля?", "Есть что-то тревожное, что нельзя оставить без внимания?"),
    "P004-11": ("ЭГДС — заключение?", "Что с желудком?"),
    "P004-12": ("Фиброскан: kPa и CAP?", "Насколько изменена печень по специальному аппаратному исследованию?"),
    "P004-13": ("ФВ ЛЖ?", "Как работает сердце по УЗИ?"),
    "P004-14": ("Печень: УЗИ, фиброскан, биохимия, гастроэнтеролог — сведите.", "Что вообще с печенью у пациента?"),
    "P004-15": ("ССР: МКБ, ЭКГ, ЭхоКГ, липиды, ИМТ — сведите.", "Насколько у него плохо с сердцем и сосудами в целом?"),
    "P006-11": ("УЗИ ЩЖ — заключение, TI-RADS?", "Что с щитовидкой по УЗИ?"),
    "P006-12": ("ЭГДС и диагноз при выписке?", "Что нашли в пищеводе и желудке, с каким диагнозом отпустили?"),
    "P006-13": ("ФВ ЛЖ?", "Как сердце качает?"),
    "P006-14": ("E78 у кардиолога подтверждается липидограммой?", "Правильно ли кардиолог поставил диагноз про холестерин?"),
    "P006-15": ("ЖКТ: УЗИ ОБП, ЭГДС, ФКС, эпикриз — сведите.", "Что у пациента с пищеварением?"),
}


# Whole-check-up summary ("meta-analysis"): gold is a checklist of key findings by section.
SUMMARY_Q = (
    "Составьте итоговую сводку чекапа для врача: ключевые отклонения анализов, находки инструментальных "
    "исследований, диагнозы специалистов, противоречия в данных и что рекомендовано дообследовать или лечить."
)
SUMMARY: dict[str, dict] = {
    "P003": {
        "labs": F(r"лпнп|низкой плотности", r"холестерин", r"гемоглобин|гематокрит"),
        "studies": F(
            r"новообразован\w*.{0,40}(толст|кишк)|колоноскоп",
            r"молочн|bi-?rads",
            r"мениск",
            r"оча\w*.{0,40}л[её]гк|плевр",
            r"нефроптоз|висцероптоз",
            r"грыж",
        ),
        "diagnoses": F(r"мигрен", r"миопи", r"ринит", r"люмбалг|боль внизу спины"),
        "inconsistencies": F(r"вес|масс\w* тела"),
        "plan": F(r"онколог", r"маммолог|узи молочн", r"травматолог|ортопед", r"мскт|оператив"),
    },
    "P004": {
        "labs": F(r"триглицерид|\bтг\b", r"лпвп|высокой плотности", r"витамин d|витамина d|25-oh", r"щелочн|\bщф\b"),
        "studies": F(
            r"увеличен\w*.{0,40}печен|гепатомегал",
            r"спленомегал|селез",
            r"9[.,]2|эластометр",
            r"гастропат|гастрит",
            r"гипертроф|ремоделир",
            r"предстательн|простат",
            r"перфорант|вен",
        ),
        "diagnoses": F(r"i10|гипертенз|гипертони", r"нажбп|k76|стеатоз|жиров", r"ожирен|34[.,]6", r"e55|недостаточност\w* витамин"),
        "inconsistencies": [],
        "plan": F(r"хеликобакт|эрадикац|амоксициллин", r"колекальциферол|витамин d", r"питан|диет|гипокалор"),
    },
    "P006": {
        "labs": F(r"лпнп|низкой плотности", r"триглицерид|\bтг\b", r"лпвп|высокой плотности", r"билирубин"),
        "studies": F(r"полип", r"поджелудочн", r"кист\w*.{0,40}щитовид|щитовид\w*.{0,40}кист|ti-?rads", r"бронхит", r"кардии|гэрб"),
        "diagnoses": F(r"e78|гиперлипидем|дислипидем", r"крапивниц", r"аденоид", r"миопи"),
        "inconsistencies": F(r"рост|вес"),
        "plan": F(r"пульмонолог", r"мскт|кт\b.{0,30}динамик"),
    },
    "P005": {
        "quality": F(r"недостовер|сдвиг|смещ|ошибк|некорректн|несоответств|противореч|сомнит"),
        "trap": r"анеми\w*\s+(тяж|выраж)|тромбоцитопени|протеинури",
    },
}


# Second source: inpatient case histories (PDF, form 003/у, cardiology). Facts behind the insurer's
# objections, asked as questions about the record — not the insurer's legal conclusion.
PDF_Q: dict[str, list[dict]] = {
    "K01": [
        {
            "category": "lookup",
            "question": "Каково максимальное значение АСТ за госпитализацию и в какой день?",
            "gold": {"type": "facts", "facts": F(r"\b400\b", r"05\.11|5 ноября|2025-11-05")},
            "gold_rows": ["K01:p56:l6"],
        },
        {
            "category": "integrate",
            "question": "Как менялись АЛТ и АСТ в динамике за госпитализацию?",
            "gold": {"type": "facts", "facts": F(r"\b132\b", r"\b400\b", r"\b117\b|\b122[.,]6\b|\b134\b")},
            "gold_rows": ["K01:p5:l7", "K01:p56:l6", "K01:p56:l7", "K01:p90:l9"],
        },
        {
            "category": "freetext",
            "question": "Какая фракция выброса левого желудочка указана в диагнозе?",
            "gold": {"type": "facts", "facts": F(r"\b25\s*%|25\s*%\s*по\s*simpson|=\s*25")},
            "gold_rows": ["K01:p99:b6"],
        },
        {
            "category": "integrate",
            "question": "Какой статин и в какой дозе получал пациент, и как это соотносится с уровнем трансаминаз?",
            "gold": {"type": "facts", "facts": F(r"аторвастатин", r"\b80\b", r"аст|трансаминаз|алт")},
            "gold_rows": ["K01:p22:b10", "K01:p100:b9"],
        },
    ],
    "K02": [
        {
            "category": "anthropo",
            "question": "Согласуется ли масса тела пациента в разных документах карты?",
            "gold": {"type": "conflict", "conflict": True, "fields": ["Вес"]},
            "gold_rows": ["K02:p20:b24", "K02:p86:b3"],
        },
        {
            "category": "freetext",
            "question": "Какая разовая доза эноксапарина указана в листе назначений?",
            "gold": {"type": "facts", "facts": F(r"0[.,]8")},
            "gold_rows": ["K02:p67:b8"],
        },
        {
            "category": "lookup",
            "question": "Каков максимальный уровень тропонина I?",
            "gold": {"type": "facts", "facts": F(r"тропонин", r"\d{3,5}")},
            "gold_rows": ["K02:p6:l1"],
        },
    ],
    "K03": [
        {
            "category": "lookup",
            "question": "Какие значения АСТ и АЛТ были 20.10.2025?",
            "gold": {"type": "facts", "facts": F(r"\b237\b", r"\b72\b")},
            "gold_rows": ["K03:p48:l6"],
        },
        {
            "category": "integrate",
            "question": "Курит ли пациент и есть ли в рекомендациях при выписке совет отказаться от курения?",
            "gold": {"type": "facts", "facts": F(r"кур", r"нет|не\s+(указан|содерж|дан|приведен|включ)|отсутств")},
            "gold_rows": ["K03:p20:b7"],
        },
    ],
    "K04": [
        {
            "category": "integrate",
            "question": "Какая антитромботическая терапия рекомендована при выписке?",
            "gold": {"type": "facts", "facts": F(r"тикагрелор|брилинта", r"ривароксабан|ксарелто")},
            "gold_rows": ["K04:p13:b10", "K04:p71:b15"],
        },
        {
            "category": "icd",
            "question": "Какой основной диагноз и код МКБ-10 указаны в карте?",
            "gold": {"type": "facts", "facts": F(r"i22|i21", r"повторн\w*\s+инфаркт|инфаркт")},
            "gold_rows": ["K04:p2:b4"],
        },
    ],
    "K05": [
        {
            "category": "lookup",
            "question": "Какой уровень ЛПНП у пациентки?",
            "gold": {"type": "numeric", "value": 6.31, "family": r"лпнп"},
            "gold_rows": ["K05:p40:l1"],
        },
        {
            "category": "integrate",
            "question": "Какая гиполипидемическая терапия назначена? Назначен ли эзетимиб?",
            "gold": {
                "type": "facts",
                "facts": F(
                    r"аторвастатин",
                    r"эзетимиб\w*\s+(не\s+назнач|не\s+получ|нет|отсутств)|не\s+назначен\w*\s+эзетимиб|эзетимиб[^.]{0,40}(нет|не\s)",
                ),
            },
            "gold_rows": ["K05:p6:b3"],
        },
    ],
    "K06": [
        {
            "category": "freetext",
            "question": "Курит ли пациент? Какой стаж и интенсивность курения?",
            "gold": {"type": "facts", "facts": F(r"кур", r"13\s*лет", r"пачк")},
            "gold_rows": ["K06:p11:b13"],
        },
        {
            "category": "lookup",
            "question": "Каков максимальный уровень АСТ за госпитализацию?",
            "gold": {"type": "facts", "facts": F(r"\b292\b")},
            "gold_rows": ["K06:p44:l6"],
        },
    ],
}
PDF_SUMMARY = {
    "K01": {
        "diagnoses": F(r"i21|инфаркт", r"хобл|обструктивн", r"анеми", r"хсн|сердечн\w* недостаточн"),
        "studies": F(r"стент|тбка|чкв|ангиопласт", r"25\s*%|фв"),
        "labs": F(r"аст|трансаминаз", r"тропонин|кфк"),
        "plan": F(r"аторвастатин|статин", r"ацетилсалицил|аспирин|тикагрелор|клопидогрел"),
    },
    "K02": {
        "diagnoses": F(r"i21|инфаркт", r"гипертон|i11"),
        "studies": F(r"стент|чкв"),
        "inconsistencies": F(r"100", r"120"),
        "plan": F(r"эноксапарин|гепарин|антикоагул"),
    },
}


# split "probe3": written before the retrieval iteration (document-aware hybrid search); new topics,
# several gold rows each, mixed precise and lay wording.
PROBE_RETRIEVAL: dict[str, list[dict]] = {
    "P006": [
        {
            "category": "freetext",
            "question": "Что показала эхокардиография?",
            "gold": {"type": "facts", "facts": F(r"систолическ|диастолическ", r"не\s+(нарушен|утолщ|расширен)")},
            "gold_rows": ["P006:r314"],
        },
        {
            "category": "freetext",
            "question": "Есть ли изменения простаты по УЗИ?",
            "gold": {"type": "facts", "facts": F(r"диффузн", r"простат|предстательн")},
            "gold_rows": ["P006:r155"],
        },
        {
            "category": "freetext",
            "question": "Что посоветовал ЛОР по питанию?",
            "gold": {"type": "facts", "facts": F(r"щадящ", r"сладк|мучн")},
            "gold_rows": ["P006:r134"],
        },
    ],
    "P004": [
        {
            "category": "freetext",
            "question": "Что нашли на КТ грудной клетки?",
            "gold": {"type": "facts", "facts": F(r"объемн|воспалит", r"не\s+выявлен|нет\s+данных|данных\s+за", r"дегенератив")},
            "gold_rows": ["P004:r635"],
        },
    ],
    "P003": [
        {
            "category": "integrate",
            "question": "Чем гастроэнтеролог рекомендовал лечиться и какую диету соблюдать?",
            "gold": {"type": "facts", "facts": F(r"энтеролактис", r"алактозн|без\s+лактоз|лактоз")},
            "gold_rows": ["P003:r596", "P003:r597"],
        },
    ],
    "K01": [
        {
            "category": "freetext",
            "question": "Какие вмешательства на сосудах сердца выполнены и сколько стентов поставлено?",
            "gold": {"type": "facts", "facts": F(r"стент", r"\b3\b|три|трех|трёх", r"пна|передн\w*\s+нисходящ|передн\w*\s+межжелудочк")},
            "gold_rows": ["K01:p2:b5"],
        },
    ],
    "K03": [
        {
            "category": "freetext",
            "question": "Насколько снижена насосная функция сердца?",
            "gold": {"type": "facts", "facts": F(r"\b48\s*%|\b48\b")},
            "gold_rows": ["K03:p40:b9", "K03:p43:b9", "K03:p62:b5"],
        },
    ],
    "K05": [
        {
            "category": "lookup",
            "question": "Какой у пациентки общий холестерин?",
            "gold": {"type": "facts", "facts": F(r"7[.,]57")},
            "gold_rows": ["K05:p41:l1", "K05:p46:b8", "K05:p60:b26", "K05:p81:b6"],
        },
    ],
    "K06": [
        {
            "category": "freetext",
            "question": "Злоупотребляет ли пациент алкоголем?",
            "gold": {"type": "facts", "facts": F(r"алкогол", r"злоупотреб|алкоголизац|да")},
            "gold_rows": ["K06:p11:b13", "K06:p13:b6"],
        },
    ],
}


# split "probe4": written before the medication/study entities and the answer checks were implemented:
# medications (doses, phases, conditional recommendations), studies, honest "was not prescribed".
PROBE_ENTITIES: dict[str, list[dict]] = {
    "K06": [
        {
            "category": "integrate",
            "question": "Какую двойную антиагрегантную терапию получает пациент и на какой срок?",
            "gold": {"type": "facts", "facts": F(r"тикагрелор|брилинта", r"ацетилсалицил|аспирин", r"1\s*год|12\s*мес|год")},
            "gold_rows": ["K06:p12:b3"],
        },
        {
            "category": "freetext",
            "question": "Какая фракция выброса левого желудочка?",
            "gold": {"type": "facts", "facts": F(r"\b54\b")},
            "gold_rows": ["K06:p62:b7", "K06:p82:b3"],
        },
        {
            "category": "integrate",
            "question": "Рекомендован ли пациенту эзетимиб и при каком условии?",
            "gold": {"type": "facts", "facts": F(r"эзетимиб", r"рассмотр|при\s+недостиж|при\s+необходим|добав|цел")},
            "gold_rows": ["K06:p65:b10"],
        },
    ],
    "K01": [
        {
            "category": "freetext",
            "question": "Вводился ли гепарин при ЧКВ и в какой дозе?",
            "gold": {"type": "facts", "facts": F(r"гепарин", r"75\s*(ед|ЕД)")},
            "gold_rows": ["K01:p23:b14", "K01:p99:b26"],
        },
    ],
    "K02": [
        {
            "category": "integrate",
            "question": "Какую антигипертензивную терапию пациент принимал до госпитализации?",
            "gold": {"type": "facts", "facts": F(r"лозартан", r"лерканидипин", r"индапамид")},
            "gold_rows": ["K02:p20:b7", "K02:p23:b5", "K02:p66:b9"],
        },
        {
            "category": "freetext",
            "question": "Что показала ЭКГ при поступлении?",
            "gold": {"type": "facts", "facts": F(r"синусов", r"тахикард|чсс\s*100|100\s*в\s*мин", r"повреждени|субэпикард")},
            "gold_rows": ["K02:p11:b5", "K02:p21:b1"],
        },
    ],
    "K05": [
        {
            "category": "lookup",
            "question": "Получала ли пациентка дапаглифлозин и в какой дозе?",
            "gold": {"type": "facts", "facts": F(r"дапаглифлозин|форсиг", r"10\s*мг")},
            "gold_rows": ["K05:p80:b14", "K05:p83:b4", "K05:p93:b9"],
        },
    ],
    "K04": [
        {
            "category": "not_found",
            "question": "Назначался ли пациенту варфарин?",
            "gold": {"type": "facts", "facts": F(r"варфарин", r"не\s+(назнач|получ|примен|упомина|указ)|нет|отсутств")},
            "gold_rows": [],
        },
    ],
    "P004": [
        {
            "category": "integrate",
            "question": "Какие лекарства назначили пациенту разные специалисты?",
            "gold": {
                "type": "facts",
                "facts": F(r"лорноксикам|ксефокам", r"колекальциферол|вигантол|витамин\s*d", r"амоксициллин|флемоксин", r"кларитромицин"),
            },
            "gold_rows": ["P004:r529", "P004:r549", "P004:r611"],
        },
    ],
    "P006": [
        {
            "category": "freetext",
            "question": "Назначил ли невролог пациенту лекарственную терапию?",
            "gold": {"type": "facts", "facts": F(r"не\s+(нужда|требу|назнач)|нет")},
            "gold_rows": ["P006:r398"],
        },
    ],
}
