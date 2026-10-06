"""评估报告生成 Agent（Evaluator）。

Reads the recorded conversation (evidence: 答对的题、卡壳点、追问表现),
scores four dimensions and renders a structured Markdown report:
技术深度 / 逻辑表达 / 岗位匹配度 / 沟通风格 + 优点 / 不足 / 建议 / 录用建议.

Scoring always runs against the real configured LLM — there is no offline
deterministic report, so a missing or failing model surfaces as an error
instead of a silently fabricated assessment.
"""
from __future__ import annotations

import re
from typing import Any

from .difficulty import tier_name
from .llm import LLMClient, LLMNotConfiguredError, extract_json
from .models import (
    DIFF_INITIAL,
    InterviewerProfile,
    JDInfo,
    Message,
    SPEAKER_CANDIDATE,
    SPEAKER_INTERVIEWER,
)

HIRE_LABELS = {
    "strong": "强烈推荐",
    "recommend": "推荐",
    "hold": "谨慎 / 再评估",
    "no": "暂不推荐",
}


class Evaluator:
    def __init__(self, llm: LLMClient):
        if llm is None:
            raise LLMNotConfiguredError("评估器缺少 LLM 客户端。")
        self._llm = llm

    def evaluate(
        self,
        jd: JDInfo,
        profile: InterviewerProfile,
        messages: list[Message],
        meta: dict[str, Any],
        session_id: str = "",
    ) -> str:
        """Produce the full Markdown report string."""
        questions = [m for m in messages if m.speaker == SPEAKER_INTERVIEWER]
        answers = [m for m in messages if m.speaker == SPEAKER_CANDIDATE]
        quality_history = meta.get("quality_history") or []
        difficulty_trace = [m.difficulty for m in questions if m.difficulty is not None]
        difficulty_trace = difficulty_trace or [DIFF_INITIAL]

        overview = self._overview(jd, profile, questions, answers, difficulty_trace, quality_history)
        scores, overall, strengths, weaknesses, suggestions, hire, hire_reason = (
            self._score_section(jd, profile, messages, quality_history)
        )
        transcript = self._transcript_block(messages, dialogue=True)

        rows = "".join(
            f"| {d} | {info['score']} / 10 | {info['comment']} |\n"
            for d, info in scores.items()
        )
        bullets = lambda items: "".join(f"- {it}\n" for it in items) or "- （暂无）\n"

        return f"""# 候选人评估报告

## 一、面试概览
{overview}

## 二、维度评分
| 维度 | 得分 | 评语 |
| --- | --- | --- |
{rows}

## 三、总体评价
{overall}

## 四、亮点与证据
{bullets(strengths)}

## 五、待改进之处
{bullets(weaknesses)}

## 六、后续建议
{bullets(suggestions)}

## 七、录用建议
**{hire}** — {hire_reason}

---

*报告由 Interview 系统自动生成。会话 ID：{session_id or '-'}*

## 附：完整对话记录
{transcript}
"""

    # ------------------------------------------------------------ sections
    @staticmethod
    def _overview(
        jd: JDInfo,
        profile: InterviewerProfile,
        questions: list[Message],
        answers: list[Message],
        difficulty_trace: list[int],
        quality_history: list[int],
    ) -> str:
        a_count = len(answers)
        q_count = len(quality_history)  # 每轮候选人在答对应一个提问
        trace_txt = " → ".join(tier_name(d) for d in difficulty_trace[:12])
        return f"""| 项目 | 内容 |
| --- | --- |
| 应聘岗位 | {jd.role_title or '-'} |
| 公司 | {jd.company or '-'} |
| 面试官风格 | {profile.style} |
| 提问轮次 | {q_count} 个提问 + 开场 |
| 候选人作答 | {a_count} 轮 |
| 难度轨迹 | {trace_txt or '-'}"""

    def _score_section(
        self,
        jd: JDInfo,
        profile: InterviewerProfile,
        messages: list[Message],
        quality_history: list[int],
    ) -> tuple[dict, str, list, list, list, str, str]:
        """Return (scores, overall, strengths, weaknesses, suggestions, hire, reason)."""
        return self._score_with_llm(jd, profile, messages, quality_history)

    def _score_with_llm(
        self,
        jd: JDInfo,
        profile: InterviewerProfile,
        messages: list[Message],
        quality_history: list[int],
    ) -> tuple[dict, str, list, list, list, str, str]:
        convo = self._transcript_block(messages, limit=40)
        history_txt = ", ".join(str(q) for q in quality_history) or "（无）"
        prompt = f"""你是一位资深的招聘面试评估专家。请根据完整面试记录，为候选人撰写评估报告内容。

岗位：{jd.role_title}（{jd.company or '公司未指明'}）
职级：{jd.seniority}；硬技能要求：{', '.join(jd.hard_skills) or '-'}
考察重点：{', '.join(profile.focus_points) or '-'}
每题质量评分历史（1-5）：{history_txt}

=== 完整对话 ===
{convo}

只输出一个 JSON 对象（不要输出其它文字）：
{{
  "scores": {{
     "技术深度": {{"score": 1-10 整数, "comment": "一句评语"}},
     "逻辑表达": {{"score": 1-10 整数, "comment": "一句评语"}},
     "岗位匹配度": {{"score": 1-10 整数, "comment": "一句评语"}},
     "沟通风格": {{"score": 1-10 整数, "comment": "一句评语"}}
  }},
  "overall": "总体评价（100-150 字，结合证据）",
  "strengths": ["2-4 条亮点，每条都要引用对话中的具体证据"],
  "weaknesses": ["2-3 条不足，基于对话中的卡壳或薄弱环节"],
  "suggestions": ["2-3 条可执行的提升建议"],
  "hire": "strong | recommend | hold | no 四选一",
  "hire_reason": "录用建议的一句话理由"
}}"""
        raw = self._llm.complete(prompt, task="evaluator", temperature=0.4)
        data = extract_json(raw)
        scores_raw = data.get("scores") or {}
        scores = {}
        for dim in ("技术深度", "逻辑表达", "岗位匹配度", "沟通风格"):
            info = scores_raw.get(dim) or {}
            if isinstance(info, dict):
                score, comment = info.get("score", 5), str(info.get("comment") or "")
            else:  # 模型直接给了数字（{"技术深度": 8}），无评语
                score, comment = info, ""
            score = max(1, min(10, int(score)))
            scores[dim] = {"score": score, "comment": comment}
        hire = str(data.get("hire") or "hold")
        hire = hire if hire in HIRE_LABELS else "hold"
        return (
            scores,
            str(data.get("overall") or "（模型未给出总体评价）"),
            [str(x) for x in data.get("strengths") or []],
            [str(x) for x in data.get("weaknesses") or []],
            [str(x) for x in data.get("suggestions") or []],
            hire,
            str(data.get("hire_reason") or ""),
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _transcript_block(messages: list[Message], limit: int = 0, dialogue: bool = False) -> str:
        """Render the conversation as text.

        ``dialogue=True`` produces one Markdown block per turn (bold speaker +
        blank line between turns) so the appendix renders as a real dialogue
        instead of a single run-on paragraph; the LLM-prompt variant stays as
        compact single lines.
        """
        pool = messages if not limit else messages[-limit:]
        blocks = []
        for m in pool:
            who = "面试官" if m.speaker == SPEAKER_INTERVIEWER else "候选人"
            text = (m.content or "").strip()
            if dialogue:
                # 每轮一段：发言人加粗，内部换行用硬换行保持同段，整轮不被拆散
                body = re.sub(r"\n\s*\n", "\n", text).replace("\n", "  \n")
                blocks.append(f"**{who}**：{body}")
            else:
                blocks.append(f"{who}：{text.replace(chr(10), ' ')[:400]}")
        return ("\n\n" if dialogue else "\n").join(blocks) or "（无记录）"
