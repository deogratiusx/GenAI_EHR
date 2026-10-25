"""Baselines for comparison: how general-purpose assistants access a record, without our tools.

* `CodeInterpreter.run_python` — a pandas DataFrame of the record and a Python sandbox (code interpreter).
* `serialize_record` — the whole record as text, for the "record in context" condition.
The sandbox runs the model's code locally over the patient's own data; it is a research harness, not
a security boundary.
"""

from __future__ import annotations

import contextlib
import io
import re

from ..sources import BOILERPLATE, Patient
from .base import DEFAULT_CHARS, spec

SPECS = [
    spec(
        "run_python",
        "Выполнить Python-код. Доступен pandas DataFrame `df` со строками карты пациента: колонки id, date, "
        "specialty, source, group, param, value (строка), num (число или NaN). Результат — то, что напечатано через print "
        "(до 3500 символов). Названия групп и параметров — длинные русские строки (например 'Липидограмма (Холестерол, "
        "Триглицериды, ЛПВП, ЛПНП…)'), поэтому сначала изучите структуру (df.group.unique(), "
        "df[df.param.str.contains('…', case=False)]) и фильтруйте через str.contains.",
        {"code": {"type": "string"}},
        ["code"],
    ),
]


class CodeInterpreter:
    def __init__(self, p: Patient):
        import pandas as pd

        self.df = pd.DataFrame(
            [
                {
                    "id": r.id,
                    "date": r.date,
                    "specialty": r.specialty,
                    "source": r.source,
                    "group": r.group,
                    "param": r.param,
                    "value": r.value,
                    "num": r.num,
                }
                for r in p.records
            ]
        )

    def run_python(self, code: str) -> dict:
        import numpy as np
        import pandas as pd

        unescaped = False
        if "\n" not in code and "\\n" in code:  # some backends double-escape newlines inside JSON arguments
            code, unescaped = code.replace("\\n", "\n").replace('\\"', '"'), True
        buf = io.StringIO()
        env = {"df": self.df.copy(), "pd": pd, "np": np, "re": re}
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<agent>", "exec"), env)  # noqa: S102 - research sandbox over the patient's own data
        except Exception as exc:  # the model sees its own traceback, as in a code interpreter
            return {"error": f"{type(exc).__name__}: {exc}", "stdout": buf.getvalue()[-1500:]}
        out = buf.getvalue()
        if not out.strip():
            return {"stdout": "", "note": "Код ничего не напечатал. Используйте print()."}
        res = {"stdout": out[:DEFAULT_CHARS]}
        if unescaped:
            res["harness_fix"] = "unescaped_newlines"
        return res


def serialize_record(p: Patient) -> str:
    """One line per field: `id | date | document | group | field = value`."""
    return "\n".join(
        f"{r.id} | {r.date} | {r.source} | {r.group} | {r.param} = {r.value}"
        for r in p.records
        if r.value and not BOILERPLATE.match(r.param)
    )
