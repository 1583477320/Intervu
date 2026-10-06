"""面经风格检索与缓存测试：命中复用、键归一化、优雅降级、画像注入。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_llm import FakeLLM
from interview.builder import InterviewerBuilder
from interview.models import JDInfo
from interview.recorder import Recorder
from interview.researcher import ResearchResult
from interview.style_researcher import StyleResearcher

POSTS = [
    {
        "title": f"字节跳动后端面经 {i}",
        "url": f"https://example.com/mianjing/{i}",
        "content": f"一面项目深挖{i}，二面算法手写，三面系统设计。",
    }
    for i in range(20)
]


class RecordingSearcher:
    """Records queries and returns canned posts (or raises, as configured)."""

    def __init__(self, posts=None, fail: bool = False):
        self.posts = posts or POSTS
        self.fail = fail
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[dict]:
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("search api down")
        return self.posts


def make_service(tmp_path, **kw) -> tuple[StyleResearcher, RecordingSearcher, FakeLLM]:
    rec = Recorder(tmp_path / "styles.db")
    llm = FakeLLM()
    searcher = kw.pop("searcher", None) or RecordingSearcher()
    if searcher is None:
        searcher = RecordingSearcher()
    svc = StyleResearcher(llm, cache=rec, searcher=searcher, **kw)
    return svc, searcher, llm


def test_cache_hit_reuses_style_without_research(tmp_path):
    svc, searcher, llm = make_service(tmp_path)

    first = svc.get_or_create("字节跳动", "高级后端工程师")
    assert first.ok
    assert first.from_cache is False
    assert first.text
    assert first.sources
    assert len(searcher.queries) == 3  # 三种查询词

    # 同公司同岗位再次生成 → 直接命中缓存，不再搜索/总结
    second = svc.get_or_create("字节跳动", "高级后端工程师")
    assert second.ok
    assert second.from_cache is True
    assert len(searcher.queries) == 3  # 未新增查询
    assert second.text == first.text

    # 缓存确实落库
    hits = len([t for t, _ in llm.calls if t == "style_summary"])
    assert hits == 1  # 只总结过一次


def test_same_role_different_company_and_different_role_miss(tmp_path):
    svc, searcher, _ = make_service(tmp_path)
    svc.get_or_create("字节跳动", "后端工程师")
    n1 = len(searcher.queries)

    # 同岗位不同公司 → 重新检索
    svc.get_or_create("阿里巴巴", "后端工程师")
    n2 = len(searcher.queries)
    assert n2 > n1

    # 同公司不同岗位 → 重新检索
    svc.get_or_create("字节跳动", "前端工程师")
    assert len(searcher.queries) > n2


def test_role_normalization_ignores_seniority_prefix(tmp_path):
    """高级后端工程师 与 后端工程师 共享同一缓存键。"""
    svc, searcher, _ = make_service(tmp_path)
    svc.get_or_create("字节跳动", "高级后端工程师")
    n = len(searcher.queries)

    hit = svc.get_or_create("字节跳动", "后端工程师")
    assert hit.ok and hit.from_cache is True
    assert len(searcher.queries) == n


def test_search_failure_degrades_to_empty(tmp_path):
    svc, searcher, _ = make_service(tmp_path, searcher=RecordingSearcher(fail=True))
    result = svc.get_or_create("字节跳动", "后端工程师")
    assert result.ok is False
    assert result.text == ""
    assert searcher.queries  # 尝试过检索


def test_too_few_posts_not_cached(tmp_path):
    svc, searcher, _ = make_service(
        tmp_path, searcher=RecordingSearcher(posts=POSTS[:2]), min_posts=5
    )
    result = svc.get_or_create("字节跳动", "后端工程师")
    assert result.ok is False
    # 因为未满足 min_posts，不缓存 → 再次调用仍会重新检索
    svc.get_or_create("字节跳动", "后端工程师")
    assert len(searcher.queries) == 6  # 3 查询词 × 2 次调用


def test_no_searcher_without_key_returns_empty_quickly(tmp_path):
    """未配置 Tavily key / searcher → 立即返回空，不发起网络请求。"""
    rec = Recorder(tmp_path / "styles2.db")
    svc = StyleResearcher(llm=FakeLLM(), cache=rec, api_key="")
    result = svc.get_or_create("字节跳动", "后端工程师")
    assert result.ok is False


def test_builder_injects_style_into_profile_prompt(tmp_path):
    """生成的面试官画像会参考面经风格（进入 interviewer_profile 的提示词）。"""
    rec = Recorder(tmp_path / "styles3.db")
    llm = FakeLLM()
    searcher = RecordingSearcher()

    class StubResearcher:
        def research(self, company):
            return ResearchResult()

    builder = InterviewerBuilder(
        llm,
        researcher=StubResearcher(),  # type: ignore[arg-type]
        style_researcher=StyleResearcher(llm, cache=rec, searcher=searcher),
    )
    jd = JDInfo(
        role_title="高级后端工程师",
        company="字节跳动",
        seniority="senior",
        hard_skills=["Python", "高并发"],
    )
    profile, research = builder.build(jd)
    assert profile.role_title == "高级后端工程师"

    # 画像生成提示词里带上了面经风格内容（FakeLLM 总结出的风格关键词）
    prompts = [p for t, p in llm.calls if t == "interviewer_profile"]
    assert prompts
    assert "算法手写" in prompts[0]
    assert "字节跳动" in prompts[0]


def test_builder_without_style_researcher_still_works(tmp_path):
    """未注入 style researcher 时行为与之前一致（回归保护）。"""
    llm = FakeLLM()

    class StubResearcher:
        def research(self, company):
            return ResearchResult()

    builder = InterviewerBuilder(llm, researcher=StubResearcher())  # type: ignore[arg-type]
    jd = JDInfo(role_title="后端工程师", company="字节跳动", hard_skills=["Python"])
    profile, research = builder.build(jd)
    assert profile.role_title == "后端工程师"
    # 没有风格服务 → 提示词里不应出现 FakeLLM 总结出的风格内容
    assert not any("算法手写" in p for _, p in llm.calls)
