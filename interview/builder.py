"""JD 解析与面试官构建 Agent（Interviewer Builder）。

Responsibilities (per design doc):
* 解析 JD（delegated to :mod:`interview.jd_parser`）
* 检索公司 / 部门业务背景（:mod:`interview.researcher`，检索失败自动降级为“仅基于 JD”）
* 检索该公司 + 岗位的公开面经并总结成面试风格（:mod:`interview.style_researcher`，
  失败自动降级；同公司同岗位风格跨会话缓存，后续新面试直接复用）
* 生成专属面试官画像：风格、考察重点、开场白（必须来自真实 LLM）
"""
from __future__ import annotations

from typing import Optional

from .jd_parser import JDParser
from .llm import LLMClient, LLMNotConfiguredError, extract_json
from .models import (
    InterviewerProfile,
    InterviewPlan,
    JDInfo,
    default_plan,
    is_ai_role,
)
from .researcher import CompanyResearcher, ResearchResult
from .style_researcher import StyleResearcher, StyleResult

STYLE_BY_SENIORITY = {"junior": "温和", "mid": "引导式", "senior": "严谨"}


class InterviewerBuilder:
    def __init__(
        self,
        llm: LLMClient,
        researcher: Optional[CompanyResearcher] = None,
        style_researcher: Optional[StyleResearcher] = None,
    ):
        if llm is None:
            raise LLMNotConfiguredError("面试官构建器缺少 LLM 客户端。")
        self._llm = llm
        self._parser = JDParser(llm)
        self._researcher = researcher or CompanyResearcher()
        self._styles = style_researcher

    def parse_jd(self, jd_text: str) -> JDInfo:
        return self._parser.parse(jd_text)

    def research(self, jd: JDInfo) -> ResearchResult:
        """Pluggable retrieval; any failure degrades to an empty result."""
        if not jd.company:
            return ResearchResult()
        return self._researcher.research(jd.company)

    def style(self, jd: JDInfo):
        """Retrieve the (company, role) interview style — cached across sessions.

        Never raises: search / summarization failures degrade to an empty result
        and the profile falls back to JD + Wikipedia only.
        """
        if self._styles is None or not jd.company:
            return StyleResult()  # 空结果：无风格参考
        return self._styles.get_or_create(jd.company, jd.role_title or "")

    def build(self, jd: JDInfo) -> tuple[InterviewerProfile, ResearchResult]:
        research = self.research(jd)
        style = self.style(jd)
        context = research.context if research.ok else ""
        return self._build_with_llm(jd, context, style), research

    # ------------------------------------------------------------- llm path
    def _build_with_llm(self, jd: JDInfo, context: str, style=None) -> InterviewerProfile:
        company_line = f"公司/部门：{jd.company}" if jd.company else "公司/部门：未指明"
        context_line = (
            f"已检索到的业务背景：{context}" if context else "没有可用的外部业务背景，仅基于 JD 组织面试。"
        )
        style_line = ""
        if style is not None and getattr(style, "ok", False) and (style.text or "").strip():
            style_line = (
                "\n已检索到该公司 / 岗位的真实面经风格参考：\n"
                f"{style.text}\n"
                "请让画像贴合该风格（语气、考察侧重、轮次节奏），但不要对候选人复述面经内容。"
            )
        prompt = f"""你是资深面试设计专家。请为下面的岗位构建一位专属面试官。

{company_line}
岗位名称：{jd.role_title}
经验要求：{jd.experience_years or '未指明'}
职级：{jd.seniority}
硬技能：{', '.join(jd.hard_skills) or '未指明'}
软素质：{', '.join(jd.soft_skills) or '未指明'}
{context_line}{style_line}

只输出一个 JSON 对象，不要输出其它文字：
{{
  "style": "面试风格，三选一：严谨 | 引导式 | 温和（与职级匹配，初级偏温和引导、资深偏严谨；如面经风格参考与职级默认冲突，以面经风格为准）",
  "tone": "语气描述，如：专业、友好、善于追问",
  "focus_points": ["3-6 个面试考察重点，应覆盖硬技能与软素质（可结合面经中该公司常考的方向）"],
  "opening": "面试官的开场白，200字以内：自我介绍、点明面试岗位与考察方向，如检索到公司背景可自然融入，并预告面试环节安排（如：先聊项目，再看计算机基础，最后手撕算法）。询问候选人是否可以开始。用自然的口语，不要用'开场白：'之类的标签",
  "interview_plan": {{
    "sections": [
      {{"name": "环节名，如：项目深挖 / 大模型基础 / 计算机基础 / 系统设计 / 手撕算法 / 反问环节",
        "question_types": ["本环节题型，如：项目追问 / 概念题 / 应用题 / 设计题 / 算法题 / 开放"],
        "min_turns": 本环节最少轮数整数,
        "max_turns": 本环节最多轮数整数}}
    ],
    "total_turns_hint": 全场总轮数（默认 12 左右）
  }}
}}

环节规划要求：
- AI/算法/Agent 岗必须包含「大模型基础」环节（Transformer/attention、SFT/RLHF、LoRA、RAG、Agent 等考点），且权重不低于经典八股；
- 实习/初级岗包含「手撕算法」环节（难度 easy→medium）；资深岗以系统设计为主，算法可省或降为 1 轮；
- 总轮数与 total_turns_hint 对齐（默认 12），各环节 min_turns 之和不超过总轮数。"""
        raw = self._llm.complete(prompt, task="interviewer_profile", temperature=0.6)
        data = extract_json(raw)

        # 面试剧本：LLM 给了就用，缺/坏则按职级+岗位类型套确定性模板
        plan = InterviewPlan.from_dict(data.get("interview_plan"))
        if not plan.sections:
            plan = default_plan(
                seniority=jd.seniority,
                ai=is_ai_role(jd.role_title, jd.hard_skills),
            )

        profile = InterviewerProfile(
            role_title=jd.role_title,
            company=jd.company,
            style=str(data.get("style") or STYLE_BY_SENIORITY.get(jd.seniority, "严谨")),
            tone=str(data.get("tone") or "专业、友好、有耐心"),
            focus_points=list(data.get("focus_points") or jd.focus_points or jd.hard_skills[:5]),
            company_context=context,
            sources=[],
            opening=str(data.get("opening") or ""),
            plan=plan,
        )
        return profile
