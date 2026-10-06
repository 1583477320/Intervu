"""公司 / 部门业务背景检索（可插拔，失败时优雅降级）。

The default implementation queries Wikipedia's open REST/action APIs — no key
needed — in the language configured by ``WIKI_LANG``. Anything else (network
down, page missing, timeout) returns ``None`` and the Builder simply falls
back to a JD-only generic interviewer, per the design doc:

    "公司信息检索通过工具调用，失败时降级为仅基于 JD 的通用面试官"
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

TIMEOUT_SECONDS = 6
MAX_CHARS = 700


@dataclass
class ResearchResult:
    """Best-effort context about the hiring company / business."""
    context: str = ""
    sources: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.context.strip())


class CompanyResearcher:
    """Retrieves a short business background for a company name."""

    def __init__(self, lang: str = "zh", timeout: int = TIMEOUT_SECONDS):
        self._lang = lang or "zh"
        self._timeout = timeout

    def research(self, company: Optional[str]) -> ResearchResult:
        if not company:
            return ResearchResult()
        try:
            return self._research(company.strip())
        except Exception:
            # network errors / timeouts / malformed payload -> degrade
            return ResearchResult()

    # ------------------------------------------------------------------ #
    def _research(self, company: str) -> ResearchResult:
        title = self._search_top_title(company)
        if not title:
            return ResearchResult()
        summary = self._fetch_summary(title)
        if not summary:
            return ResearchResult()
        source = f"https://{self._lang}.wikipedia.org/wiki/{urllib.parse.quote(title)}"
        context = f"{title}：{summary[:MAX_CHARS]}"
        return ResearchResult(context=context, sources=[source])

    def _search_top_title(self, company: str) -> Optional[str]:
        params = urllib.parse.urlencode(
            {
                "action": "query",
                "list": "search",
                "srsearch": company,
                "srlimit": 1,
                "format": "json",
            }
        )
        url = f"https://{self._lang}.wikipedia.org/w/api.php?{params}"
        with urllib.request.urlopen(url, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        hits = data.get("query", {}).get("search") or []
        return hits[0]["title"] if hits else None

    def _fetch_summary(self, title: str) -> str:
        safe = urllib.parse.quote(title)
        url = f"https://{self._lang}.wikipedia.org/api/rest_v1/page/summary/{safe}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        extract = data.get("extract") or ""
        # 只保留前 2-3 句，避免大段无关背景
        sentences = []
        for part in extract.replace("\n", " ").split("。"):
            sentences.append(part)
            if len("。".join(sentences)) >= 180 or len(sentences) >= 3:
                break
        return "。".join(sentences).strip()[:MAX_CHARS]
