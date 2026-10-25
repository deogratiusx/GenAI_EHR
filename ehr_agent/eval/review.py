"""Test plan as one local HTML page: architecture diagram, question sets, categories, systems and models,
every question with its gold and source rows, and the unit tests.

The page shows patient values (gold answers, record rows), so it is written to data/ and stays out of git.

  python -m ehr_agent.eval.review   →  data/test_plan.html
"""

from __future__ import annotations

import ast
import html
import json
import re
from collections import Counter

from ..agent.systems import SYSTEMS
from ..config import PROJECT_DIR, RUNS_DIR
from ..knowledge.analytes import BY_CODE
from ..sources import get_patient
from .benchmark import OUT as QUESTIONS
from .labels import EXCLUDED_MODELS, SYSTEM_LABELS
from .run import sparse

OUT = QUESTIONS.with_name("test_plan.html")
DIAGRAM = PROJECT_DIR / "docs" / "architecture.svg"
TESTS_DIR = PROJECT_DIR / "tests"

CAT_RU = {
    "lookup": "Значение анализа",
    "abnormal": "Отклонения в панели",
    "not_found": "Нет данных (ловушка)",
    "icd": "Коды МКБ-10",
    "anthropo": "Согласованность роста и веса",
    "freetext": "Факт из текстового протокола",
    "integrate": "Сведение данных нескольких документов",
    "quality": "Испорченные данные (ловушка)",
    "summary": "Сводка всей карты",
    "dynamics": "Динамика показателя",
}
CAT_WHAT = {
    "lookup": ("Найти конкретное значение показателя", "число или ключевые факты"),
    "abnormal": ("Перечислить показатели панели вне нормы", "F1 по множеству; штраф за названные нормальные"),
    "not_found": ("Показатель не измерялся: не выдумать значение", "status=not_found, любое число = галлюцинация"),
    "icd": ("Перечислить все коды МКБ-10 из карты", "F1 по кодам"),
    "anthropo": ("Заметить расхождение роста и веса между документами", "найдено ли расхождение и в каких полях"),
    "freetext": ("Извлечь факт из текстового протокола", "доля ключевых фактов в ответе"),
    "integrate": ("Свести данные нескольких документов", "доля ключевых фактов в ответе"),
    "quality": ("Не интерпретировать сдвинутую выгрузку (P005)", "status=uncertain; ловушка — назвать битое число"),
    "summary": (
        "Итоговая сводка всей карты для врача",
        "доля пунктов чек-листа по разделам; critical recall — доля пунктов must_mention досье",
    ),
    "dynamics": ("Описать ход показателя за госпитализацию", "пик, последнее значение, направление с учётом RCV, нормализация"),
}
# (назначение, когда написан)
SPLIT_RU = {
    "dev": ("Отладка инструментов v1 (P001–P002); в оценку не входит", "до прогонов"),
    "test": ("Основная оценка на таблицах чекапа (P003–P006)", "до прогонов, заморожен"),
    "para": ("Перефразировки test: L2 — клинический жаргон и аббревиатуры, L3 — бытовая размытая формулировка", "до прогонов"),
    "summary": ("Сводка всего чекапа", "до прогонов"),
    "pdf": ("Истории болезни другой клиники (K01–K06)", "после добавления PDF-адаптера"),
    "pdf_summary": ("Сводка всей истории болезни", "после добавления PDF-адаптера"),
    "probe": ("Новые темы на таблицах", "после разбора ошибок v1, до v2"),
    "probe3": ("Проверка поиска (v3)", "до реализации v3"),
    "probe4": ("Препараты и исследования (v4)", "до реализации v4"),
    "trend": ("Динамика показателей в PDF (2 на карту)", "из данных, перед финальным прогоном"),
    "report": (
        "Итоговый отчёт для PDF без ручного эталона; черновой эталон — пункты must_mention досье",
        "перед финальным прогоном, требует проверки",
    ),
}
SEC_RU = {
    "labs": "Анализы",
    "studies": "Исследования",
    "diagnoses": "Диагнозы",
    "inconsistencies": "Противоречия",
    "plan": "Рекомендации",
    "quality": "Качество данных",
}
NAMES = {k: a.name for k, a in BY_CODE.items()} | {
    "NEU": "Нейтрофилы",
    "LYM": "Лимфоциты",
    "MONO": "Моноциты",
    "EOS": "Эозинофилы",
    "BASO": "Базофилы",
}
MODELS = [
    ("GigaChat-2 Lite", "API Сбера, данные в РФ", "слабая облачная модель"),
    ("GigaChat-2 Max", "API Сбера, данные в РФ", "сильная облачная модель, ориентир"),
    ("Qwen3.5-4B", "локально, MLX, 4 бит", "лёгкая открытая модель"),
    ("Med-Qwen-3.5-4B", "локально, MLX, 4 бит", "та же база с медицинским дообучением"),
]
SYSTEM_ROLE = {
    "tools": "основная система",
    "tools_routed": "меньше выбора для слабых моделей",
    "tools_prefetch": "экономия одного вызова",
    "report": "сводка: досье + отчёт + проверки полноты и вердиктов",
    "report_nocheck": "абляция проверок отчёта",
    "tools_compact": "влияние размера набора",
    "tools_plain": "абляция обвязки",
    "code": "базовая линия: интерпретатор кода",
    "record": "базовая линия: длинный контекст",
    "record_chunked": "базовая линия: map-reduce по фрагментам карты",
    "record+tools": "помогают ли инструменты при равном доступе",
}


def readable(rx: str) -> str:
    """A gold regex as words: alternatives with ' / ', wildcards as '…'."""
    s = re.sub(r"\(\?[!=<][^)]*\)", "", rx)
    s = re.sub(r"\\d\{\d*,?\d*\}|\\d\+", "число", s)
    s = re.sub(r"(?:\.|\\D|\[\^[^\]]*\])(?:\{\d*,?\d*\}|\*|\+)", "…", s)
    s = (
        s.replace(r"\b", "")
        .replace(r"\w*", "")
        .replace(r"\w+", "")
        .replace(r"\s*", " ")
        .replace(r"\s+", " ")
        .replace(r"\s", " ")
        .replace(r"\d", "#")
    )
    s = s.replace(r"\.", ".").replace("[.,]", ",").replace("[её]", "е").replace("?:", "")
    s = re.sub(r"[()\[\]?^$\\]", "", s)
    return html.escape(re.sub(r"\s*\|\s*", " / ", s).strip())


def gold_html(g: dict) -> str:
    t = g["type"]
    if t == "numeric":
        return f"Число <b>{g['value']}</b>"
    if t == "set":
        pos = ", ".join(NAMES.get(f, f) for f in g["positive"]) or "<i>нет отклонений</i>"
        neg = ", ".join(NAMES.get(f, f) for f in g["negative"]) or "—"
        neu = ", ".join(NAMES.get(f, f) for f in g["neutral"])
        return f"<div><b>Вне нормы:</b> {pos}</div><div class='mut'>Штраф, если названы: {neg}</div>" + (
            f"<div class='mut'>Не учитываются: {neu}</div>" if neu else ""
        )
    if t == "not_found":
        return "<b>Нет данных</b> (status=not_found)"
    if t == "codes":
        return "МКБ-10: <b>" + ", ".join(g["codes"]) + "</b>"
    if t == "conflict":
        return ("Расхождение: <b>" + ", ".join(g["fields"]) + "</b>") if g["conflict"] else "Расхождений <b>нет</b>"
    if t == "facts":
        return "<ol>" + "".join(f"<li title='{html.escape(rx)}'>{readable(rx)}</li>" for rx in g["facts"]) + "</ol>"
    if t == "checklist":
        parts = "".join(
            f"<div><b>{SEC_RU.get(k, k)}:</b> "
            + " · ".join(f"<span title='{html.escape(rx)}'>{readable(rx)}</span>" for rx in v)
            + "</div>"
            for k, v in g["sections"].items()
            if v
        )
        return parts + (f"<div class='mut'>Ловушка: {readable(g['trap'])}</div>" if g.get("trap") else "")
    if t == "must":
        return "<b>Черновой эталон:</b> пункты must_mention досье (critical recall); требует проверки врачом"
    if t == "uncertain":
        return f"<b>Данные недостоверны</b> (status=uncertain)<div class='mut'>Ловушка: назвать {readable(g['trap'])} без оговорки</div>"
    return html.escape(json.dumps(g, ensure_ascii=False))


def rows_html(q: dict) -> str:
    rows = q.get("support_rows") or q["gold_rows"]
    if not rows:
        return "<span class='mut'>—</span>"
    by = get_patient(q["patient"]).by_id()
    out = []
    for rid in rows[:2]:
        r = by.get(rid)
        text = f"{r.param}: {r.value}" if r else "строка не найдена"
        out.append(f"<div class='row'><span class='rid'>{rid}</span> {html.escape(text[:220])}</div>")
    more = f"<div class='mut'>ещё {len(rows) - 2}</div>" if len(rows) > 2 else ""
    return "".join(out) + more


def episode_counts() -> Counter:
    c: Counter = Counter()
    for p in RUNS_DIR.glob("*.jsonl"):
        if "smoke" in p.name:
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e.get("model") not in EXCLUDED_MODELS:
                c[e["qid"]] += 1
    return c


def unit_tests() -> str:
    """Test functions with their module docstring, read from tests/ without importing them."""
    out = []
    for f in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")]
        doc = html.escape(ast.get_docstring(tree) or "")
        items = "".join(f"<li><code>{n[5:]}</code></li>" for n in names)
        out.append(
            f"<div class='card'><h3>{f.name} <span class='mut'>· {len(names)}</span></h3><p class='mut'>{doc}</p><ul class='cols'>{items}</ul></div>"
        )
    return "".join(out)


def table(head: list[str], rows: list[list[str]], cls: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='{cls}'><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


def main() -> None:
    qs = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {q["id"]: q for q in qs}
    in_sample = {q["id"] for q in sparse(qs, 1)}
    eps = episode_counts()

    split_rows = []
    for sp, (what, when) in SPLIT_RU.items():
        g = [q for q in qs if q["split"] == sp]
        pats = sorted({q["patient"] for q in g})
        split_rows.append([f"<code>{sp}</code>", what, when, str(len(g)), ", ".join(pats), f"{sum(eps[q['id']] for q in g):,}"])
    cat_rows = []
    cats = Counter(q["category"] for q in qs if q["split"] != "dev")
    for c, (what, how) in CAT_WHAT.items():
        types = ", ".join(sorted({q["gold"]["type"] for q in qs if q["category"] == c}))
        cat_rows.append([CAT_RU[c], what, f"<code>{types}</code>", how, str(cats[c])])
    sys_rows = [
        [
            f"<code>{name}</code>",
            SYSTEM_LABELS[name],
            {
                "full": "все 12",
                "report": "все 12 + досье",
                "compact": "9, без препаратов и исследований",
                "code": "run_python",
                "none": "—",
            }[cfg.toolset]
            + (" (по намерению)" if cfg.routed_tools else "")
            + (" + упреждающий вызов" if cfg.prefetch else ""),
            "да" if cfg.record_in_prompt else ("по фрагментам" if cfg.chunked else "—"),
            "да" if cfg.checks else "—",
            SYSTEM_ROLE[name],
        ]
        for name, cfg in SYSTEMS.items()
    ]

    trs = []
    for q in qs:
        src = "PDF" if q["patient"].startswith("K") else "таблица"
        base = by_id.get(q.get("base_id") or "")
        orig = f"<div class='mut'>исходный: {html.escape(base['question'])}</div>" if base else ""
        lvl = f"<span class='lv'>L{q['level']}</span>" if q.get("level") else ""
        flag = "<span class='ok' title='в разреженной выборке'>●</span>" if q["id"] in in_sample else ""
        trs.append(
            f"<tr data-split='{q['split']}' data-cat='{q['category']}' data-src='{src}' data-sample='{int(q['id'] in in_sample)}'>"
            f"<td class='id'>{q['id']}<div class='mut'>{q['split']} · {src}</div></td>"
            f"<td><div class='cat'>{CAT_RU[q['category']]}</div><div class='q'>{lvl}{html.escape(q['question'])}</div>{orig}</td>"
            f"<td>{gold_html(q['gold'])}</td><td>{rows_html(q)}</td><td class='num'>{eps[q['id']] or ''} {flag}</td></tr>"
        )

    def opts(values, labels=None):
        return "".join(f"<option value='{v}'>{(labels or {}).get(v, v)}</option>" for v in values)

    svg = DIAGRAM.read_text(encoding="utf-8") if DIAGRAM.exists() else ""
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>План тестов EHR-агента</title><style>
:root{{--bg:#fcfcfb;--fg:#1d1d1b;--mut:#6b6a65;--line:#e4e3de;--card:#f4f3ef;--acc:#2a78d6;--ok:#1baf7a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#efeee9;--mut:#a8a79f;--line:#34332f;--card:#24231f;--acc:#6aa6ef;--ok:#3ccf95}}}}
*{{box-sizing:border-box}} body{{background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,system-ui,sans-serif;margin:0}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:24px;margin:0 0 6px}} h2{{font-size:18px;margin:40px 0 10px}} h3{{font-size:14px;margin:0 0 4px}}
p{{margin:4px 0 10px;max-width:820px}} .mut{{color:var(--mut);font-size:12.5px}}
nav{{display:flex;flex-wrap:wrap;gap:14px;margin:12px 0 0;font-size:13px}} nav a{{color:var(--acc);text-decoration:none}}
.diagram{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;overflow-x:auto}}
.diagram svg{{display:block;width:100%;min-width:900px;height:auto}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%}}
th,td{{border-top:1px solid var(--line);padding:8px 10px;vertical-align:top;text-align:left}}
th{{font-size:12px;color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--bg)}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
td.id{{white-space:nowrap;font-family:ui-monospace,Menlo,monospace;font-size:12px}} td.num{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}
.cat{{font-size:11px;color:var(--acc);text-transform:uppercase;letter-spacing:.03em}} .q{{font-weight:600;max-width:380px}}
.row{{background:var(--card);border-radius:6px;padding:5px 8px;margin-bottom:5px;max-width:460px;font-size:12.5px}}
.rid{{font-family:ui-monospace,Menlo,monospace;font-size:11px;color:var(--mut)}} ol{{margin:0;padding-left:18px}}
.lv{{font-size:10px;border:1px solid var(--line);border-radius:6px;padding:0 4px;margin-right:5px;font-family:ui-monospace,monospace;color:var(--mut)}}
.ok{{color:var(--ok)}} .bar{{position:sticky;top:0;z-index:2;background:var(--bg);padding:10px 0;display:flex;flex-wrap:wrap;gap:8px;align-items:center;border-bottom:1px solid var(--line)}}
select,input{{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:5px 8px;font:inherit;font-size:13px}}
input[type=search]{{min-width:220px}} #count{{margin-left:auto}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}}
ul.cols{{margin:6px 0 0;padding-left:18px;font-size:12.5px}}
#qs th{{top:52px}}
</style></head><body><main>
<h1>План тестов EHR-агента</h1>
<p class="mut">Локальная страница: содержит эталонные значения и строки карт пациентов, не публикуется. Собрана командой
<code>python -m ehr_agent.eval.review</code> из <code>data/questions.jsonl</code> ({len(qs)} вопросов) и журналов <code>runs/</code>.</p>
<nav><a href="#arch">Архитектура</a><a href="#sets">Наборы</a><a href="#cats">Категории</a><a href="#grid">Модели и системы</a><a href="#qs">Все вопросы</a><a href="#unit">Unit-тесты</a></nav>

<h2 id="arch">Архитектура</h2>
<p>Модель не видит карту целиком: она вызывает инструменты, получает компактные результаты с id строк и завершает работу
вызовом <code>final_answer</code>. Зависимости идут сверху вниз, модули оценки справа используют все слои.</p>
<div class="diagram">{svg}</div>

<h2 id="sets">Наборы вопросов</h2>
<p>Эталоны построены до прогонов соответствующей итерации. Последний столбец — сколько эпизодов уже прогнано в
исследовательской фазе (все модели и условия, без GigaChat Pro).</p>
{table(["Набор", "Назначение", "Когда написан", "Вопросов", "Пациенты", "Эпизодов"], split_rows)}

<h2 id="cats">Категории и как они оцениваются</h2>
{table(["Категория", "Что проверяет", "Тип эталона", "Как считается", "Вопросов"], cat_rows)}
<p class="mut">Нормы для категории «Отклонения» в этом файле — по политике <code>indicative</code> (бланк → ориентировочный каталог).
Для политики <code>layered</code> эталоны собраны отдельно в <code>data/questions_layered.jsonl</code>.</p>

<h2 id="grid">Модели и системы</h2>
<div class="cards">{"".join(f"<div class='card'><h3>{m}</h3><div class='mut'>{w}</div><div>{r}</div></div>" for m, w, r in MODELS)}</div>
<p></p>
{table(["Система", "Название", "Инструменты", "Карта в промпте", "Проверки и починка", "Роль в эксперименте"], sys_rows)}

<h2 id="qs">Все вопросы</h2>
<div class="bar">
<select id="fsplit"><option value="">все наборы</option>{opts(SPLIT_RU)}</select>
<select id="fcat"><option value="">все категории</option>{opts(CAT_RU, CAT_RU)}</select>
<select id="fsrc"><option value="">оба источника</option>{opts(["таблица", "PDF"])}</select>
<label class="mut"><input type="checkbox" id="fsample"> только разреженная выборка <span class="ok">●</span></label>
<input type="search" id="fq" placeholder="поиск по тексту">
<span id="count" class="mut"></span></div>
<div class="tw"><table id="qs"><thead><tr><th>id</th><th>Вопрос</th><th>Эталон</th><th>Опорные строки карты</th><th>Эпизодов</th></tr></thead>
<tbody>{"".join(trs)}</tbody></table></div>

<h2 id="unit">Unit-тесты</h2>
<p>Запуск: <code>.venv/bin/python -m pytest</code>. Все тесты работают на синтетических записях, без данных пациентов и без моделей.</p>
<div class="cards">{unit_tests()}</div>
</main>
<script>
const f={{split:fsplit,cat:fcat,src:fsrc}}, rows=[...document.querySelectorAll('#qs tbody tr')];
function apply(){{
  const q=fq.value.toLowerCase(); let n=0;
  rows.forEach(tr=>{{
    const ok=Object.entries(f).every(([k,el])=>!el.value||tr.dataset[k]===el.value)
      &&(!fsample.checked||tr.dataset.sample==='1')&&(!q||tr.textContent.toLowerCase().includes(q));
    tr.style.display=ok?'':'none'; n+=ok;
  }});
  count.textContent=n+' из '+rows.length;
}}
[fsplit,fcat,fsrc,fsample,fq].forEach(el=>el.addEventListener('input',apply)); apply();
</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(OUT, len(qs))


if __name__ == "__main__":
    main()
