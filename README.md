# Инструменты LLM-агента для аналитики медицинских карт

> **Public mirror of the course project.** Patient data are not included: the anonymized check-up exports
> (provided by a partner clinic under agreement) and the inpatient PDF records stay private, as do model weights
> and run logs with record excerpts. The code, tests (synthetic data), report, slides sources and the analysis
> plan are complete. Aggregated results: [REPORT.md](REPORT.md) §8–10. The hosted demo
> (`Dockerfile.demo`, `scripts/build_demo_data.py`) runs inside AI-mnesis behind Keycloak.


Курсовой проект по дисциплине «Современные методы и алгоритмы генеративного ИИ».
Тема: **Set of tools for EHR LLM-assisted analytics**. Выполнен на материалах стартапа AI-mnesis.

**Исследовательский вопрос (H1).** Помогают ли узкие типизированные медицинские инструменты небольшой
локальной модели (Qwen3.5-4B, Med-Qwen-3.5-4B) или модели, размещённой в РФ (GigaChat-2 Lite и Max), отвечать
на вопросы врача по медкарте не хуже или лучше, чем два подхода универсальных ассистентов: интерпретатор кода над
таблицей карты и вся карта в контексте? Сравнение проводится на двух типах источников: почти идеальной таблице
чекапа одной клиники и PDF-историях болезни другой клиники.

Результаты и методология изложены в [REPORT.md](REPORT.md). Устройство системы описано в
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Структура

```
ehr_agent/
  config.py       пути и переключатели (EHR_REF_POLICY, EHR_DENSE)
  sources/        источники → единая модель Record/Patient: таблица чекапа (xlsx), история болезни (PDF, с маскированием ПДн)
  knowledge/      базы знаний: каталог анализов, политика референсов, NHANES, лексикон и онтология, сущности, динамика (RCV), значимость
  retrieval/      поиск по тексту карты: BM25 + dense (e5) + RRF, контекст документа
  tools/          инструменты, которые вызывает модель (12 штук + final_answer), досье и схема отчёта для сводок, и базовые линии (run_python, вся карта)
  agent/          цикл агента, обвязка (маршрутизация, проверки ответа), промпты, бэкенды GigaChat и Qwen (MLX)
  eval/           эталонные вопросы, прогоны, оценка, аналитика, бенчмарк поиска, аудит ПДн
tests/            unit-тесты на синтетических данных (реальных данных пациентов в тестах нет)
app.py            демо на Streamlit: вопрос → цепочка вызовов → ответ → процитированные строки карты
archive/          код исследовательских итераций v1–v4 и скрипты очередей прогонов (для воспроизводимости runs/)
data/, models/, runs/   данные пациентов, веса моделей, журналы прогонов (не в git)
```

## Запуск

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Основные команды (из каталога проекта; для локальных моделей задайте `HF_HUB_OFFLINE=1`):

```bash
.venv/bin/python -m ehr_agent.eval.benchmark          # собрать вопросы и эталоны → data/questions.jsonl
.venv/bin/python -m ehr_agent.eval.run --models GigaChat-2 qwen3.5-4b --systems tools code record --split test --sample 2 --out runs/main.jsonl
.venv/bin/python -m ehr_agent.eval.analyze runs/main.jsonl   # точность с бутстреп-ДИ → runs/summary.md, runs/figs/
.venv/bin/python -m ehr_agent.eval.behavior runs/main.jsonl  # поведение агентов, ошибки, стоимость → runs/behavior.md
.venv/bin/python -m ehr_agent.eval.retrieval_bench    # офлайн-бенчмарк поиска (без LLM)
.venv/bin/python -m ehr_agent.eval.review             # план тестов: схема, наборы, все вопросы → data/test_plan.html
.venv/bin/python -m ehr_agent.eval.pii_audit          # остатки ПДн в замаскированных PDF (только счётчики)
.venv/bin/python -m pytest                            # тесты
.venv/bin/streamlit run app.py                        # демо
```

Модели: `GigaChat-2` (Lite), `GigaChat-2-Max`, `qwen3.5-4b`, `med-qwen3.5-4b`. Системы (`--systems`):
`tools`, `tools_routed`, `tools_prefetch`, `tools_compact`, `tools_plain`, `report`, `report_nocheck`, `code`, `record`,
`record_chunked`, `record+tools`;
их состав описан в [ehr_agent/agent/systems.py](ehr_agent/agent/systems.py).

GigaChat требует `SBER_AUTHORIZATION_KEY` в `.env` корня репозитория и сертификаты НУЦ Минцифры из `certs/`.
Персональный тариф допускает один параллельный запрос, поэтому прогон идёт последовательно. Локальные модели
работают через MLX на Apple Silicon. Med-Qwen сконвертирован в 4 бита из HF-весов (`models/`).

## Данные

Обезличенные выгрузки чекапов (`data/raw/P00*.xlsx`) переданы клиникой-партнёром по соглашению. PDF-истории
болезни (`data/pdf_raw/K0*`) содержат персональные данные. Они маскируются при загрузке, до отправки в любой
API, а `eval.pii_audit` проверяет остатки. Ни данные, ни веса, ни журналы прогонов не публикуются.
