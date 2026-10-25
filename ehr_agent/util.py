"""Small text utilities shared by the harness and the evaluation."""

from __future__ import annotations

import re

# Citation ids of both sources: P003:r527 (check-up row) · K02:p21:b3 / K02:p5:l2 (PDF block / lab row)
ID_RX = re.compile(r"(?:P00\d|K0\d):(?:r\d+|p\d+:[bl]\d+)")
_NUM = re.compile(r"(?<![\w:.])\d+(?:[.,]\d+)?")


def norm_number(s: str) -> str:
    """'6,30' → '6.3', '400' → '400' (comparable string form)."""
    s = s.replace(",", ".")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def numbers_in(text: str) -> set[str]:
    """Numbers stated in a text, ignoring citation ids, ICD-10 codes and ISO dates."""
    text = ID_RX.sub(" ", text)
    text = re.sub(r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b", " ", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", " ", text)
    return {norm_number(m) for m in _NUM.findall(text)}
