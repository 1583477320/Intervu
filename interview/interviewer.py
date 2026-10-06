"""面试官 Agent（Interviewer Core）— 出题、倾听、追问。

Keeps multi-turn memory (the recent transcript) plus an extracted-facts memory
(「候选人熟悉 X」), so the interviewer can reference what the candidate said
earlier ("你刚才提到……"). Questions follow the difficulty tier chosen by the
Difficulty Controller and react to the latest quality judgment.

Questions always come from the real configured LLM — there is no scripted
fallback question bank, so a missing or failing model surfaces as an error
instead of a fake interview.
"""
from __future__ import annotations

from .difficulty import tier_name
from .llm import LLMClient, LLMNotConfiguredError
from .models import InterviewSection, InterviewerProfile, JDInfo

MAX_ANSWER_CHARS = 1800

# 最后一个环节（反问）：把提问权交给候选人，面试官不再出题。
REVERSE_SECTION_KEYWORDS = ("反问", "候选人提问", "提问环节", "问答环节", "q&a")
HANDOVER_QUESTION = (
    "我这边的技术问题问得差不多了，剩下的时间交给你——"
    "你有什么想问我的吗？团队、业务、成长空间都可以聊。"
)


def is_reverse_section(section: Optional[InterviewSection]) -> bool:
    """True when this section is the closing 「反问」环节。"""
    if section is None:
        return False
    name = (section.name or "").strip().lower()
    return any(k in name for k in REVERSE_SECTION_KEYWORDS)


class Interviewer:
    def __init__(
        self,
        llm: LLMClient,
        profile: InterviewerProfile,
        jd: JDInfo,
    ):
        if llm is None:
            raise LLMNotConfiguredError("面试官缺少 LLM 客户端。")
        self._llm = llm
        self._profile = profile
        self._jd = jd

    # ------------------------------------------------------------ persona
    @property
    def profile(self) -> InterviewerProfile:
        return self._profile

    def opening_message(self) -> str:
        if self._profile.opening.strip():
            return self._profile.opening.strip()
        # deterministic fallback opening
        role = self._jd.role_title or "该岗位"
        company = f"很高兴能代表 {self._profile.company} 与你交流，" if self._profile.company else ""
        return (
            f"你好！{company}我是本次「{role}」面试的面试官。"
            "今天我们会围绕岗位相关的技术基础和实战经验展开交流，"
            "过程中有任何问题都可以随时问我。准备好了我们就开始吧。"
        )

    def closing_message(self, requested: bool) -> str:
        if requested:
            return (
                "好的，既然你想在这里结束，那我们今天的面试就到这里。"
                "感谢你的坦诚分享，我会把整个过程整理成评估报告。"
            )
        return (
            "我们已经交流了足够多的内容，今天的面试就到这里。"
            "感谢你的时间与分享，我会把整个过程整理成评估报告。"
        )

    # --------------------------------------------------------- resume turn
    def _persona(self) -> str:
        who = f"你是{self._profile.role_title or '该公司'}团队的面试官"
        if self._profile.company:
            who += f"（{self._profile.company}）"
        return (
            f"{who}。面试风格：{self._profile.style}；语气：{self._profile.tone}。"
            "说话像真人在聊天：用短句和自然口语衔接，专业术语该用就用，但别堆砌"
            "“请深入剖析 / 请阐述 / 请量化”这类书面词。"
        )

    def first_question(self, *, resume: str) -> str:
        """Produce the first question, anchored on the candidate's resume."""
        focus_txt = "、".join(self._profile.focus_points) or "岗位核心技能"
        context = self._profile.company_context
        resume_txt = (resume or "").strip()[:3000] or "（候选人未提供简历）"

        prompt = f"""{self._persona()}
考察重点：{focus_txt}。
业务背景（可选参考）：{context or '未检索到公司背景，仅基于 JD 面试。'}

候选人简历：
{resume_txt}

这是本次面试的第一个问题：请先简短回应简历中的亮点（可省），然后基于简历内容提出第一个问题。要求：
1. 问题必须紧扣简历中的具体内容（项目经历、技术栈、教育背景、实习/工作经历等），不要泛泛而问；
2. 控制在 150 字以内，用口语化的短句；
3. 一次只问一个问题，不要一句话里连问多个问题；
4. 只输出你对候选人说的一句话，不要输出任何解释、列表或前后缀标签。"""
        raw = self._llm.complete(prompt, task="interviewer_first", temperature=0.7)
        out = (raw or "").strip()[:MAX_ANSWER_CHARS]
        if not out:
            raise LLMNotConfiguredError("面试官模型返回了空内容，请重试。")
        return out

    # ------------------------------------------------------------ next turn
    def ask(
        self,
        *,
        difficulty: int,
        last_judgment_summary: str,
        last_quality: Optional[int],
        transcript_tail: str,
        facts: list[str],
        turn_no: int,
        resume: str = "",
        section: Optional[InterviewSection] = None,
        section_turn: int = 0,
    ) -> str:
        """Produce the next question / follow-up as a natural single turn."""
        # 反问环节：确定性交接提问权，绝不再出技术题（硬约束，不经过 LLM）。
        if is_reverse_section(section):
            return HANDOVER_QUESTION

        focus_txt = "、".join(self._profile.focus_points) or "岗位核心技能"
        facts_txt = "；".join(facts) if facts else "（暂无）"
        context = self._profile.company_context

        quality_hint = "（本回合不需要评分，只需正常推进对话）"
        last_hint = (
            f"候选人对上一题的作答质量评分：{last_quality}/5。评分员点评：{last_judgment_summary or '（无）'}。"
            if last_quality is not None
            else "这是本次面试的第一个问题。"
        )
        section_hint = ""
        if section is not None:
            types = "、".join(section.question_types) or "自由提问"
            section_hint = (
                f"\n【当前环节】{section.name}（本环节第 {section_turn} 问），"
                f"本环节只出这些题型：{types}。"
                "若上一题也属于本环节，请换一个角度或考点，不要重复问。"
            )
        guidance = self._guidance(last_quality)
        resume_txt = (resume or "").strip()[:3000]
        resume_line = (
            f"\n候选人简历（供追问参考，可结合简历深挖或核对前后一致性）：\n{resume_txt}\n"
            if resume_txt
            else ""
        )

        prompt = f"""{self._persona()}
考察重点：{focus_txt}。{quality_hint}
业务背景（可选参考）：{context or '未检索到公司背景，仅基于 JD 面试。'}

当前处于第 {turn_no} 个提问回合。
【当前难度】{tier_name(difficulty)}（难度档位：1 概念基础 → 2 应用场景 → 3 深度设计）
{section_hint}

{last_hint}
{guidance}

关于候选人的记忆：{facts_txt}
{resume_line}
之前的对话：
{transcript_tail or '（暂无）'}

请针对候选人刚才的回答继续面试。要求：
1. 只输出你对候选人说的一句话（可简短肯定/点评后提出下一个问题或追问），控制在 120 字以内，用口语化的短句；
2. 一次只问一个问题，不要一句话里连问多个问题；
3. 优先紧扣考察重点推进；候选人答得好就深挖一层或转向下一考点，答得不好就降一档难度并给提示引导（仅在环节内切换题型，不要跨环节出题）；
4. 若当前是「手撕算法」环节：本轮先给出具体题目（含输入输出示例），让候选人讲思路；
5. 必要时引用候选人之前说过的话（“你刚才提到……”），体现对话记忆；
6. 不要输出任何解释、列表或前后缀标签，不要替候选人作答。"""
        raw = self._llm.complete(prompt, task="interviewer_turn", temperature=0.8)
        out = (raw or "").strip()[:MAX_ANSWER_CHARS]
        if not out:
            raise LLMNotConfiguredError("面试官模型返回了空内容，请重试。")
        return out

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _guidance(last_quality: Optional[int]) -> str:
        if last_quality is None:
            return "请先抛出第一个面试问题（概念或自我介绍+一个考点开场），让候选人热身。"
        if last_quality >= 4:
            return "候选人上轮表现优秀：可以顺着答案深挖一层，或升档进入更难的应用/设计题。"
        if last_quality <= 2:
            return "候选人上轮卡壳：请降低难度，先给提示/引导，或切换到更基础的概念题与场景题，避免让候选人放弃。"
        return "候选人表现平稳：可以在当前难度继续，换一个角度考察同一考点。每个考点不要停留过久。"
