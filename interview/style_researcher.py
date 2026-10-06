"""公司 + 岗位「面试风格」检索（面经搜索 → LLM 总结 → 跨会话缓存）。

不同公司 / 岗位的面试风格差异很大（轮次结构、氛围、常考方向都不相同）。
本模块在生成面试官画像时为 (公司, 岗位) 全网检索公开面经（面试经验帖），
交给 LLM 总结成一段“该公司该岗位真实面试风格”描述，并写入 SQLite 缓存
（``interview_styles`` 表）；同一公司同一岗位再次发起新面试时直接复用
缓存风格，不再重复搜索与总结。

与 LLM 强制“真实可用”的原则不同，面经检索属于**增强信息**：未配置
``TAVILY_API_KEY``、网络异常、结果过少或总结失败时一律**优雅降级**返回
空结果 —— 面试照常进行，只是退回“仅基于 JD / Wikipedia 背景”的通用画像。
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .llm import LLMClient, extract_json

TIMEOUT_SECONDS = 8
# 每个搜索词最多取 10 条，配合多个搜索词最终累计 20+ 条去重结果
QUERY_RESULTS = 10
MAX_POSTS = 24
MIN_POSTS = 5  # 少于该数量视为检索失败，不总结也不缓存

TAVILY_ENDPOINT = "https://api.tavily.com/search"

# 岗位名归一化时去掉的前缀（高级后端工程师 == 后端工程师）
_SENIORITY_PREFIXES = ("资深", "高级", "初级", "中级", "中高级", "专家", "主任", "首席")
_COMPANY_SUFFIXES = ("有限公司", "股份有限公司", "有限责任公司")


def _norm_company(name: str) -> str:
    s = re.sub(r"\s+", "", (name or "").strip().lower())
    for suffix in _COMPANY_SUFFIXES:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


def _norm_role(role: str) -> str:
    s = (role or "").strip()
    for prefix in _SENIORITY_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return re.sub(r"\s+", "", s).lower()


@dataclass
class StyleResult:
    """One (company, role) interview-style summary. ``ok=False`` = 检索/总结失败或未配置。"""

    ok: bool = False
    text: str = ""  # 可直接喂给画像 LLM 的风格描述
    sources: list[str] = field(default_factory=list)  # 引用的面经链接
    data: dict[str, Any] = field(default_factory=dict)  # 结构化总结（含 style/rounds/...）
    from_cache: bool = False


class TavilySearcher:
    """Tavily Search API 客户端（urllib 实现，无额外依赖）。"""

    def __init__(self, api_key: str, timeout: float = TIMEOUT_SECONDS):
        self._api_key = api_key
        self._timeout = timeout

    def search(self, query: str, max_results: int = QUERY_RESULTS) -> list[dict[str, Any]]:
        payload = {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "topic": "general",
            "include_answer": False,
            "include_raw_content": False,
        }
        req = urllib.request.Request(
            TAVILY_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return list(data.get("results") or [])


class StyleResearcher:
    """检索并缓存 (company, role) 的面试风格。可插拔 search 后端，失败优雅降级。

    :param llm: 用于把检索到的面经总结成风格描述（真实 LLM，与其它 agent 一致）。
    :param cache: Recorder（缓存表 interview_styles）。None = 不持久化（每次重新检索）。
    :param api_key: Tavily API Key；为空且未注入自定义 ``searcher`` 时直接降级。
    :param searcher: 可插拔搜索后端（默认 TavilySearcher）。测试可注入假搜索器。
    """

    def __init__(
        self,
        llm: LLMClient,
        cache: Any = None,
        api_key: str = "",
        searcher: Optional[Callable[[str], list[dict[str, Any]]]] = None,
        min_posts: int = MIN_POSTS,
    ):
        self._llm = llm
        self._cache = cache
        self._searcher = searcher if searcher is not None else (
            TavilySearcher(api_key).search if (api_key or "").strip() else None
        )
        self._min_posts = min_posts

    # ------------------------------------------------------------- public
    def get_or_create(self, company: str, role_title: str) -> StyleResult:
        """返回 (company, role) 的面试风格：先查缓存，未命中则搜索 + 总结 + 落缓存。"""
        if not (company or "").strip() or not (role_title or "").strip():
            return StyleResult()
        ckey = _norm_company(company)
        rkey = _norm_role(role_title)
        if not ckey or not rkey:
            return StyleResult()
        try:
            if self._cache is not None:
                cached = self._cache.get_interview_style(ckey, rkey)
                if cached is not None:
                    return self._from_data(cached["style"], cached["sources"], from_cache=True)
            if self._searcher is None:  # 未配置搜索后端：不阻塞面试
                return StyleResult()
            posts = self._collect(company, role_title)
            if len(posts) < self._min_posts:
                return StyleResult()
            data = self._summarize(company, role_title, posts)
            if not data:
                return StyleResult()
            sources = [p["url"] for p in posts[:10]]
            if self._cache is not None:
                try:
                    self._cache.save_interview_style(ckey, rkey, data, sources)
                except Exception:
                    pass  # 缓存失败不影响本次使用
            return self._from_data(data, sources, from_cache=False)
        except Exception:
            # 网络错误 / 解析异常 / LLM 异常 → 优雅降级为“无面经风格”
            return StyleResult()

    # ------------------------------------------------------------ private
    def _collect(self, company: str, role_title: str) -> list[dict[str, Any]]:
        seen: set[str] = set()
        posts: list[dict[str, Any]] = []
        for query in self._queries(company, role_title):
            try:
                items = self._searcher(query)  # type: ignore[misc]
            except Exception:
                items = []
            for item in items or []:
                url = str(item.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                title = str(item.get("title") or "").strip()
                content = str(item.get("content") or item.get("raw_content") or "").strip()
                if not title and not content:
                    continue
                posts.append(
                    {
                        "title": title[:200],
                        "url": url,
                        "content": re.sub(r"\s+", " ", content)[:600],
                    }
                )
                if len(posts) >= MAX_POSTS:
                    return posts
        return posts

    @staticmethod
    def _queries(company: str, role_title: str) -> list[str]:
        return [
            f"{company} {role_title} 面经",
            f"{company} {role_title} 面试经验",
            f"{company} {role_title} 面试真题 考察重点",
        ]

    def _summarize(self, company: str, role_title: str, posts: list[dict[str, Any]]) -> dict[str, Any]:
        lines = []
        for i, post in enumerate(posts, 1):
            summary = post["content"] or "（无摘要）"
            lines.append(f"{i}. 标题：{post['title']}\n   链接：{post['url']}\n   内容摘要：{summary}")
        posts_txt = "\n\n".join(lines)

        prompt = f"""你是资深招聘调研专家。下面是全网检索到的「{company} · {role_title}」岗位的 {len(posts)} 篇面经（面试经验分享），内容可能包含噪音、无关信息或少量低质量帖子。

请综合提炼该公司该岗位**真实的面试风格**，只输出一个 JSON 对象，不要输出其它文字：
{{
  "style": "一句话概括面试风格，如：多轮技术面为主、节奏快、重项目深挖与算法手写",
  "rounds": ["典型面试轮次 / 流程，2-5 条"],
  "question_focus": ["常考察的方向 / 题型，3-6 条"],
  "tone": "面试氛围与提问风格，1 句话",
  "summary": "综合总结（150-300 字）：面试流程轮次、考察重点、整体难度与候选人需要注意的地方"
}}
判断依据以**多数面经的共性**为准，剔除明显噪音（如不相关的招聘帖）。如果面经之间相互矛盾，请同时说明。

面经列表：
{posts_txt[:12000]}"""
        raw = self._llm.complete(prompt, task="style_summary", temperature=0.3)
        data = extract_json(raw)
        if not isinstance(data, dict) or not (data.get("style") or data.get("summary")):
            return {}
        return {
            "style": str(data.get("style") or "").strip()[:200],
            "rounds": [str(x)[:120] for x in (data.get("rounds") or [])][:5],
            "question_focus": [str(x)[:150] for x in (data.get("question_focus") or [])][:6],
            "tone": str(data.get("tone") or "").strip()[:200],
            "summary": str(data.get("summary") or "").strip()[:600],
        }

    @staticmethod
    def _from_data(data: dict[str, Any], sources: list[str], from_cache: bool) -> StyleResult:
        parts = []
        if data.get("style"):
            parts.append(f"风格：{data['style']}")
        if data.get("rounds"):
            parts.append("常见轮次：" + "、".join(data["rounds"]))
        if data.get("question_focus"):
            parts.append("考察重点：" + "；".join(data["question_focus"]))
        if data.get("tone"):
            parts.append(f"氛围：{data['tone']}")
        if data.get("summary"):
            parts.append(data["summary"])
        return StyleResult(
            ok=bool(parts),
            text="\n".join(parts)[:1500],
            sources=list(sources),
            data=data,
            from_cache=from_cache,
        )
