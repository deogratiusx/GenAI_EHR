"""Render the question set as a local HTML page for expert review (gold + source rows side by side).

  python -m ehr_agent.review_questions   →  data/questions_review.html  (stays out of git with data/)
"""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path

from .data import get_patient
from .questions import FAMILY, OUT as QUESTIONS
from .refs import BY_CODE

OUT = QUESTIONS.with_name(QUESTIONS.stem.replace("questions", "questions_review") + ".html")
CAT_RU = {
    "lookup": "Значение анализа", "abnormal": "Отклонения в панели", "not_found": "Нет данных (ловушка)",
    "icd": "Коды МКБ-10", "anthropo": "Согласованность роста/веса", "freetext": "Факт из текстового протокола",
    "integrate": "Сведение данных специалистов", "quality": "Испорченные данные (ловушка)",
    "summary": "Сводка всего чекапа",
}
SPLIT_RU = {"test": "тест", "probe": "проверочный (после разбора v1)", "dev": "отладка", "summary": "сводка"}
SEC_RU = {"labs": "Анализы", "studies": "Исследования", "diagnoses": "Диагнозы", "inconsistencies": "Противоречия",
          "plan": "Рекомендации", "quality": "Качество данных"}
FAMILY_RU = {k: (BY_CODE[k].name if k in BY_CODE else k) for k in FAMILY}
FAMILY_RU.update({"NEU": "Нейтрофилы", "LYM": "Лимфоциты", "MONO": "Моноциты", "EOS": "Эозинофилы", "BASO": "Базофилы"})


def gold_html(q: dict) -> str:
    g = q["gold"]
    t = g["type"]
    if t == "numeric":
        return f"Число <b>{g['value']}</b> должно быть в ответе."
    if t == "set":
        pos = ", ".join(FAMILY_RU[f] for f in g["positive"]) or "<i>нет отклонений</i>"
        neg = ", ".join(FAMILY_RU[f] for f in g["negative"]) or "—"
        neu = ", ".join(BY_CODE[f].name if f in BY_CODE else f for f in g["neutral"]) or "—"
        return (f"<div><b>Должны быть названы:</b> {pos}</div><div class='muted'><b>Штраф, если названы:</b> {neg}</div>"
                f"<div class='muted'><b>Не учитываются</b> (норма зависит от лаборатории): {neu}</div>")
    if t == "not_found":
        return "Правильно: <b>status=not_found</b> или «нет данных». Любое число в ответе считается галлюцинацией."
    if t == "codes":
        return "Коды: <b>" + ", ".join(g["codes"]) + "</b> (оценка F1)"
    if t == "conflict":
        return ("Есть расхождения в: <b>" + ", ".join(g["fields"]) + "</b>") if g["conflict"] else "Расхождений <b>нет</b>."
    if t == "facts":
        return "Ключевые факты (регулярные выражения, доля найденных):<ol>" + "".join(
            f"<li><code>{html.escape(rx)}</code></li>" for rx in g["facts"]) + "</ol>"
    if t == "checklist":
        parts = "".join(f"<div><b>{SEC_RU.get(k, k)}:</b> " + ", ".join(f"<code>{html.escape(rx)}</code>" for rx in v) + "</div>"
                        for k, v in g["sections"].items() if v)
        trap = f"<div class='muted'>Ловушка: <code>{html.escape(g['trap'])}</code></div>" if g.get("trap") else ""
        return "Чек-лист ключевых находок (доля найденных):" + parts + trap
    if t == "uncertain":
        return (f"Правильно: пометить данные как недостоверные (status=uncertain или явное указание на сдвиг/ошибку). "
                f"Ловушка: <code>{html.escape(g['trap'])}</code> без оговорки.")
    return html.escape(json.dumps(g, ensure_ascii=False))


def rows_html(q: dict) -> str:
    if not q["gold_rows"]:
        return "<span class='muted'>—</span>"
    p = get_patient(q["patient"])
    by = p.by_id()
    out = []
    for rid in q["gold_rows"]:
        r = by.get(rid)
        if not r:
            out.append(f"<div class='row bad'>{rid}: строка не найдена</div>")
            continue
        src = html.escape((r.visit_title or r.protocol)[:60])
        out.append(f"<div class='row'><span class='rid'>{rid}</span> <span class='muted'>{r.date} · {src} · "
                   f"{html.escape(r.param[:60])}</span><div>{html.escape(r.value[:700])}</div></div>")
    return "".join(out)


def main() -> None:
    qs = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    para: dict[str, list[dict]] = {}
    for q in qs:
        if q["split"] == "para":
            para.setdefault(q["base_id"], []).append(q)
    qs = [q for q in qs if q["split"] in ("test", "probe", "summary")]
    cats = Counter(q["category"] for q in qs)
    sections = []
    for pid in sorted({q["patient"] for q in qs}):
        p = get_patient(pid)
        sex = {"M": "муж.", "F": "жен."}.get(p.sex or "", "?")
        trs = []
        for q in [x for x in qs if x["patient"] == pid]:
            trs.append(
                f"<tr data-cat='{q['category']}' data-split='{q['split']}'><td class='id'>{q['id']}<div class='tag {q['split']}'>"
                f"{SPLIT_RU[q['split']]}</div></td><td><div class='cat'>{CAT_RU[q['category']]}</div>"
                f"<div class='q'>{html.escape(q['question'])}</div>"
                + "".join(f"<div class='para'><span class='lv'>L{v['level']}</span> {html.escape(v['question'])}</div>" for v in para.get(q["id"], []))
                + "</td><td>{gold_html(q)}</td><td>{rows_html(q)}</td>"
                f"<td class='chk'><label><input type='checkbox'> ок</label><textarea placeholder='замечание'></textarea></td></tr>")
        note = " — <b>выгрузка сдвинута, только вопросы о качестве данных</b>" if pid == "P005" else ""
        sections.append(f"<h2>{pid} <span class='muted'>({sex}, {len(p.records)} строк){note}</span></h2>"
                        f"<table><thead><tr><th>id</th><th>Вопрос</th><th>Эталон</th><th>Строки-источники в карте</th>"
                        f"<th>Проверка</th></tr></thead><tbody>{''.join(trs)}</tbody></table>")
    filters = "".join(f"<button data-f='{c}'>{CAT_RU[c]} ({n})</button>" for c, n in cats.items())
    page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Вопросы и эталоны</title><style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#6b6a65;--line:#e4e3de;--card:#f4f3ef;--accent:#2a78d6;--bad:#c62f2f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#f0efec;--muted:#a8a79f;--line:#34332f;--card:#24231f;--accent:#3987e5;--bad:#e66767}}}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:16px 24px}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 8px}} .muted{{color:var(--muted);font-weight:400}}
table{{border-collapse:collapse;width:100%}} th,td{{border-top:1px solid var(--line);padding:8px;vertical-align:top;text-align:left}}
th{{font-size:12px;color:var(--muted);font-weight:600}} td.id{{white-space:nowrap;font-family:ui-monospace,monospace;font-size:12px}}
.cat{{font-size:11px;color:var(--accent);text-transform:uppercase;letter-spacing:.03em}} .q{{font-weight:600;max-width:340px}}
.row{{background:var(--card);border-radius:6px;padding:6px 8px;margin-bottom:6px;max-width:520px}} .row.bad{{color:var(--bad)}}
.rid{{font-family:ui-monospace,monospace;font-size:12px}} code{{font-size:12px;word-break:break-all}} ol{{margin:4px 0;padding-left:18px}}
.tag{{display:inline-block;font-size:10px;padding:1px 6px;border-radius:8px;background:var(--card);color:var(--muted);margin-top:4px;font-family:system-ui}}
.tag.probe{{color:var(--accent)}} .para{{font-size:13px;margin-top:4px;color:var(--muted);max-width:340px}}
.lv{{font-size:10px;border:1px solid var(--line);border-radius:6px;padding:0 4px;margin-right:4px;font-family:ui-monospace,monospace}} .chk textarea{{width:140px;height:44px;margin-top:4px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px}}
.bar{{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:6px}}
.bar button{{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:14px;padding:3px 10px;cursor:pointer;font-size:12px}}
.bar button.on{{border-color:var(--accent);color:var(--accent)}}
</style></head><body>
<h1>Тестовые и проверочные вопросы с эталонами</h1>
<div class="muted">{len(qs)} вопросов + {sum(map(len, para.values()))} перефразировок (L1 — исходная точная формулировка, L2 — клиническая/аббревиатуры, L3 — размытая/бытовая; эталон тот же). Эталон строится из данных (lookup, abnormal, not_found, icd, anthropo) или вручную (freetext, integrate, quality).
Нормы для «abnormal» — ориентировочные взрослые интервалы из ehr_agent/refs.py; в выгрузке клиники бланков нет.</div>
<div class="bar"><button data-f="all" class="on">все</button>{filters}<button data-f="probe">только проверочные</button></div>
{''.join(sections)}
<script>
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{{
  document.querySelectorAll('.bar button').forEach(x=>x.classList.toggle('on',x===b));
  const f=b.dataset.f;
  document.querySelectorAll('tbody tr').forEach(tr=>{{
    tr.style.display=(f==='all'||tr.dataset.cat===f||(f==='probe'&&tr.dataset.split==='probe'))?'':'none';}});
}});
</script></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(OUT, len(qs))


if __name__ == "__main__":
    main()
