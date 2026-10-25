"""Harness and agent loop with a scripted backend (no model)."""

from ehr_agent.agent import run_episode
from ehr_agent.agent.harness import check_answer, keep_substantive, rescue_text_final, route_intent
from ehr_agent.agent.llm import Reply, parse_qwen_tool_call
from ehr_agent.tools import build_toolset


class Scripted:
    """A backend that replays a list of replies (tool calls or text)."""

    model = "scripted"

    def __init__(self, replies):
        self.replies, self.forced = list(replies), []

    def chat(self, messages, tools, force_final=False):
        self.forced.append(force_final)
        r = self.replies.pop(0) if self.replies else Reply(content="")
        return r


def call(name, **args):
    return Reply(tool_call={"name": name, "arguments": args})


def test_route_intent():
    assert route_intent("Какой уровень гемоглобина у пациентки?")[:2] == ("lab", "get_lab")
    assert route_intent("Какие показатели липидограммы вне нормы?")[1] == "list_abnormal_labs"
    assert route_intent("Получала ли пациентка дапаглифлозин?")[:2] == ("medications", "get_medications")
    assert route_intent("Согласуется ли вес в разных документах?")[1] == "check_consistency"
    assert route_intent("Что показала колоноскопия?")[0] == "studies"


def test_rescue_text_final_variants():
    j = rescue_text_final(
        "final_answer: {<|superquote|>answer<|superquote|>: <|superquote|>Не назначался.<|superquote|>, "
        "<|superquote|>citations<|superquote|>: [], <|superquote|>status<|superquote|>: <|superquote|>not_found<|superquote|>}"
    )
    assert j == {"answer": "Не назначался.", "citations": [], "status": "not_found"}
    before = rescue_text_final("Получала дапаглифлозин 10 мг (K09:p80:b14). final_answer")
    assert before["citations"] == ["K09:p80:b14"] and "10 мг" in before["answer"]


def test_keep_substantive_after_harness_message():
    first = {"answer": "Дапаглифлозин 10 мг", "citations": ["K09:p1:b1"], "status": "answered"}
    meta = {"answer": "Указываю id строк из вывода инструмента", "citations": ["K09:p2:b1"], "status": "answered"}
    assert keep_substantive(first, meta)["answer"] == "Дапаглифлозин 10 мг"


def test_check_answer_citations_numbers_absence(checkup):
    _, fns = build_toolset("full", checkup)
    done = set()
    bad = check_answer({"answer": "ЛПНП 4.1", "citations": ["P009:r999"], "status": "answered"}, checkup, "ЛПНП?", fns, "", done)
    assert bad["invalid_ids"] == ["P009:r999"]
    num = check_answer({"answer": "ЛПНП 7.7", "citations": ["P009:r2"], "status": "answered"}, checkup, "ЛПНП?", fns, "", done)
    assert num["numbers"] == ["7.7"]
    absent = check_answer(
        {"answer": "Нет данных", "citations": [], "status": "not_found"}, checkup, "Что выявлено на УЗИ желчного пузыря?", fns, "", done
    )
    assert absent and "results" in absent
    assert check_answer({"answer": "ЛПНП 4.1", "citations": ["P009:r2"], "status": "answered"}, checkup, "ЛПНП?", fns, "4.1", done) is None


def test_loop_budget_forces_final(checkup):
    replies = [call("get_lab", analyte="ЛПНП")] * 2 + [
        call("final_answer", answer="ЛПНП 4.1 ммоль/л", citations=["P009:r2"], status="answered")
    ]
    b = Scripted(replies)
    ep = run_episode(b, checkup, "Какой уровень ЛПНП?", "tools", budget=2)
    assert ep["final"]["answer"].startswith("ЛПНП 4.1") and ep["n_calls"] == 2 and b.forced[-1] is True


def test_loop_format_repair_converts_text(checkup):
    replies = [
        call("get_lab", analyte="ЛПНП"),
        Reply(content="ЛПНП 4.1 ммоль/л, выше нормы."),
        call("final_answer", answer="ЛПНП 4.1 ммоль/л", citations=["P009:r2"], status="answered"),
    ]
    b = Scripted(replies)
    ep = run_episode(b, checkup, "Какой уровень ЛПНП?", "tools")
    assert ep["final"]["format"] == "final_answer"
    assert any(s.get("action") == "format_repair" and s.get("ok") for s in ep["trace"])


def test_loop_invalid_tool_is_reported(checkup):
    replies = [call("no_such_tool"), call("final_answer", answer="ЛПНП 4.1", citations=["P009:r2"], status="answered")]
    ep = run_episode(Scripted(replies), checkup, "ЛПНП?", "tools_plain")
    assert ep["n_invalid"] == 1 and ep["final"]["answer"] == "ЛПНП 4.1"


def test_qwen_tool_call_parsing():
    xml = "<tool_call>\n<function=get_lab>\n<parameter=analyte>\nЛПНП\n</parameter>\n</function>\n</tool_call>"
    assert parse_qwen_tool_call(xml) == {"name": "get_lab", "arguments": {"analyte": "ЛПНП"}}
    js = '<tool_call>{"name": "get_lab", "arguments": {"analyte": "АЛТ"}}</tool_call>'
    assert parse_qwen_tool_call(js)["arguments"] == {"analyte": "АЛТ"}


def test_loop_prefetch_runs_intent_tool_first(checkup):
    replies = [call("final_answer", answer="ЛПНП 4.1 ммоль/л", citations=["P009:r2"], status="answered")]
    ep = run_episode(Scripted(replies), checkup, "Какой уровень ЛПНП?", "tools_prefetch")
    assert ep["trace"][0]["type"] == "prefetch" and ep["trace"][0]["name"] == "get_lab" and ep["n_calls"] == 0


def test_chunked_map_reduce(checkup, monkeypatch):
    from ehr_agent.agent import chunked

    chunks = chunked.chunk_record(checkup, size=300)
    assert len(chunks) > 2 and all(len(c) <= 300 or "\n" not in c for c in chunks)
    assert "\n".join(chunks).count("P009:r") == sum(c.count("P009:r") for c in chunks)  # no line lost or cut

    monkeypatch.setattr(chunked, "CHUNK_CHARS", 300)
    n = len(chunks)
    # map: the first chunk holds LDL, the others answer НЕТ; reduce: final_answer from the excerpt
    replies = [Reply(content="P009:r2 | ЛПНП = 4.1")] + [Reply(content="НЕТ")] * (n - 1)
    replies.append(call("final_answer", answer="ЛПНП 4.1 ммоль/л", citations=["P009:r2"], status="answered"))
    b = Scripted(replies)
    ep = run_episode(b, checkup, "Какой уровень ЛПНП?", "record_chunked")
    maps = [s for s in ep["trace"] if s["type"] == "map"]
    assert ep["n_chunks"] == n and sum(not s["empty"] for s in maps) == 1
    assert ep["final"]["answer"].startswith("ЛПНП 4.1") and ep["n_calls"] == 0


def test_map_excerpts_are_grounded_in_record_lines(checkup):
    from ehr_agent.agent.chunked import chunk_record, ground_excerpt

    chunk = chunk_record(checkup, size=10_000)[0]
    assert ground_excerpt("P009:r2 | что-то", chunk)[0].startswith("P009:r2 |")  # by id
    assert any(line.startswith("P009:r2 |") for line in ground_excerpt("4.1", chunk))  # bare value
    assert ground_excerpt("ничего похожего", chunk) == []
