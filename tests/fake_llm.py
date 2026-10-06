"""Scripted fake LLM used by tests (replaces the former production MockLLM).

The production system no longer ships any offline model — but unit tests still
need a deterministic, network-free :class:`LLMClient` to exercise the engine.
This test double responds per ``task`` with the same shapes the real model is
asked to produce.
"""
from __future__ import annotations

import json
import re

from interview.llm import LLMClient

ANSWER_MARK = "【候选人回答】"
DIFF_MARK = "【当前难度】"
FOCUS_MARK = "【考察重点】"


def _section(text: str, marker: str) -> str:
    idx = text.rfind(marker)
    if idx == -1:
        return ""
    rest = text[idx + len(marker):]
    for other in (ANSWER_MARK, DIFF_MARK, FOCUS_MARK):
        if other != marker:
            cut = rest.find(other)
            if cut != -1:
                rest = rest[:cut]
    return rest.strip()


class FakeLLM(LLMClient):
    """Deterministic scripted model for tests: no network, no demo mode."""

    name = "fake-test"

    def __init__(self):
        # (task, prompt) pairs of every call, for prompt-level assertions
        self.calls: list[tuple[str, str]] = []

    def complete(self, prompt: str, task: str = "chat", temperature: float = 0.7) -> str:
        del temperature
        self.calls.append((task, prompt))
        if task == "style_summary":
            return json.dumps(
                {
                    "style": "多轮技术面为主、节奏快、重项目深挖与算法手写",
                    "rounds": ["技术一面：项目深挖", "技术二面：算法手写", "HR 面"],
                    "question_focus": ["项目细节", "算法与数据结构", "系统设计"],
                    "tone": "直接、务实，追问深入",
                    "summary": "该公司该岗位通常为 2-3 轮技术面加一轮 HR 面，技术面重视项目深挖与手写算法，整体难度偏高。",
                },
                ensure_ascii=False,
            )
        if task == "jd_parse":
            return json.dumps(
                {
                    "role_title": "后端工程师",
                    "company": None,
                    "experience_years": "3-5 年",
                    "seniority": "mid",
                    "hard_skills": ["Python", "Rust", "MySQL", "Docker"],
                    "soft_skills": ["沟通协作", "责任心"],
                    "responsibilities": ["负责后端服务的设计与开发", "保障系统稳定与性能"],
                    "focus_points": ["Python/Rust 工程能力", "数据库设计", "系统稳定性意识"],
                },
                ensure_ascii=False,
            )
        if task == "interviewer_profile":
            return json.dumps(
                {
                    "style": "严谨",
                    "tone": "专业、友好、有耐心",
                    "focus_points": [
                        "技术基础是否扎实",
                        "是否具备实战项目经验",
                        "问题拆解与表达是否清晰",
                    ],
                    "opening": "你好，我是负责本次面试的面试官。我们开始吧。",
                    "interview_plan": {
                        "sections": [
                            {"name": "项目深挖", "question_types": ["项目追问"], "min_turns": 2, "max_turns": 2},
                            {"name": "计算机基础", "question_types": ["概念题"], "min_turns": 2, "max_turns": 2},
                            {"name": "手撕算法", "question_types": ["算法题"], "min_turns": 1, "max_turns": 1},
                            {"name": "反问环节", "question_types": ["开放"], "min_turns": 1, "max_turns": 1},
                        ],
                        "total_turns_hint": 12,
                    },
                },
                ensure_ascii=False,
            )
        if task == "judge":
            return self._judge(prompt)
        if task == "evaluator":
            return self._evaluator()
        return self._chat(prompt)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _judge(prompt: str) -> str:
        answer = _section(prompt, ANSWER_MARK) or ""
        length = len(answer.strip())
        if length >= 120:
            quality = 5
            summary = "回答详实深入，论证充分，表现优秀。"
        elif length >= 60:
            quality = 4
            summary = "回答较完整，有项目实例支撑，表现良好。"
        elif length >= 20:
            quality = 2
            summary = "回答较简短，深度不足，可能卡壳或缺乏展开。"
        else:
            quality = 1
            summary = "回答过于简单或未正面作答，需要引导。"
        return json.dumps(
            {"quality": quality, "summary": summary, "facts": []},
            ensure_ascii=False,
        )

    @staticmethod
    def _evaluator() -> str:
        return json.dumps(
            {
                "scores": {
                    "技术深度": {"score": 8, "comment": "技术基础扎实，有实际落地经验。"},
                    "逻辑表达": {"score": 8, "comment": "条理清晰，能结合项目阐述。"},
                    "岗位匹配度": {"score": 8, "comment": "技能与岗位要求匹配度较高。"},
                    "沟通风格": {"score": 8, "comment": "沟通顺畅，配合度高。"},
                },
                "overall": "候选人整体表现良好，技术基础扎实，具备相关实战经验，逻辑表达清晰。",
                "strengths": ["技术基础扎实，能深入回答核心问题。"],
                "weaknesses": ["部分深度问题仍可进一步展开。"],
                "suggestions": ["继续加强高并发场景的实战积累。"],
                "hire": "recommend",
                "hire_reason": "整体表现与岗位要求匹配度较好，建议进入下一轮。",
            },
            ensure_ascii=False,
        )

    def _chat(self, prompt: str) -> str:
        difficulty = 2
        m = re.search(rf"{re.escape(DIFF_MARK)}\s*L?(\d)", prompt)
        if m:
            difficulty = int(m.group(1))
        skills = _section(prompt, FOCUS_MARK) or "岗位核心技能"
        focus_pos = prompt.find("考察重点：")
        if focus_pos != -1:
            tail = prompt[focus_pos + len("考察重点："):]
            skills = tail.split("。")[0].strip() or skills
        first = [s.strip() for s in skills.replace("、", ",").split(",") if s.strip()]
        skill = first[0] if first else "岗位核心技能"
        if any(w in skill for w in ("是否", "能力", "扎实", "责任心", "素质", "经验")) and len(skill) > 5:
            skill = "岗位核心技能"

        bank = {
            1: f"好的，我们放慢一点。先回到基础：请用一两句话讲讲「{skill}」最核心的概念或适用场景？",
            2: f"明白。接着想听听实战：能不能结合你实际做过的项目，具体讲讲你是怎么运用「{skill}」解决问题的？遇到了什么难点？",
            3: f"这一块你的基础不错。那我们再往深走一层：如果给你一个高并发、高可用的真实场景，你会如何基于「{skill}」做架构与方案上的取舍？",
        }
        return bank.get(difficulty, bank[2])
