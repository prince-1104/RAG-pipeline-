from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass
class PageText:
    page_number: int
    text: str


def extract_pages(pdf_bytes: bytes) -> list[PageText]:
    pages: list[PageText] = []
    import io

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages.append(PageText(page_number=i, text=txt))
    return pages


def extract_pages_from_path(path: str | Path) -> list[PageText]:
    pages: list[PageText] = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            pages.append(PageText(page_number=i, text=txt))
    return pages


