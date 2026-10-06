"""Shared data structures used across the engine modules."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# Difficulty tiers (1 = L1 概念, 2 = L2 应用, 3 = L3 深度设计)
DIFF_MIN = 1
DIFF_MAX = 3
DIFF_INITIAL = 2

SPEAKER_SYSTEM = "system"
SPEAKER_INTERVIEWER = "interviewer"
SPEAKER_CANDIDATE = "candidate"

STATUS_INTERVIEWING = "interviewing"
STATUS_FINISHED = "finished"

# Report dimensions
REPORT_DIMENSIONS = ["技术深度", "逻辑表达", "岗位匹配度", "沟通风格"]


@dataclass
class InterviewSection:
    """One planned interview phase (e.g. 项目深挖 / 八股 / 手撕算法)."""

    name: str
    question_types: list[str] = field(default_factory=list)
    min_turns: int = 1
    max_turns: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "question_types": self.question_types,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
        }

    @classmethod
    def from_dict(cls, d: Any) -> Optional["InterviewSection"]:
        if not isinstance(d, dict) or not str(d.get("name") or "").strip():
            return None
        return cls(
            name=str(d.get("name")),
            question_types=[str(t) for t in (d.get("question_types") or []) if str(t).strip()],
            min_turns=max(1, _safe_int(d.get("min_turns"), 1)),
            max_turns=max(1, _safe_int(d.get("max_turns"), 3)),
        )


@dataclass
class InterviewPlan:
    """Ordered interview sections — the interviewer's "script"."""

    sections: list[InterviewSection] = field(default_factory=list)
    total_turns_hint: int = 12

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [s.to_dict() for s in self.sections],
            "total_turns_hint": self.total_turns_hint,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "InterviewPlan":
        sections = []
        if isinstance(d, dict):
            for s in d.get("sections") or []:
                sec = InterviewSection.from_dict(s)
                if sec:
                    sections.append(sec)
        return cls(sections=sections, total_turns_hint=_safe_int(d.get("total_turns_hint"), 12) if isinstance(d, dict) else 12)


AI_ROLE_KEYWORDS = (
    "llm", "大模型", "agent", "智能体", "算法", "nlp", "机器学习", "深度学习",
    "gpt", "rag", "推荐", "ai ", "aigc", "cv", "多模态",
)


def is_ai_role(role_title: str, hard_skills: list[str] | tuple = ()) -> bool:
    text = f"{role_title} {' '.join(hard_skills)}".lower()
    return any(k in text for k in AI_ROLE_KEYWORDS)


def default_plan(seniority: str = "mid", ai: bool = False) -> InterviewPlan:
    """Deterministic section template used when the LLM plan is missing/invalid."""
    if ai:
        sections = [
            InterviewSection("项目深挖", ["项目追问", "场景题"], 3, 4),
            InterviewSection("大模型基础", ["概念题", "应用题"], 3, 4),
            InterviewSection("计算机基础", ["概念题"], 2, 3),
            InterviewSection("手撕算法", ["算法题"], 1, 2),
            InterviewSection("反问环节", ["开放"], 1, 1),
        ]
    elif seniority == "senior":
        sections = [
            InterviewSection("项目深挖", ["项目追问", "场景题"], 3, 4),
            InterviewSection("系统设计", ["设计题"], 2, 3),
            InterviewSection("计算机基础", ["概念题"], 2, 3),
            InterviewSection("手撕算法", ["算法题"], 1, 2),
            InterviewSection("反问环节", ["开放"], 1, 1),
        ]
    else:
        sections = [
            InterviewSection("项目深挖", ["项目追问", "场景题"], 3, 4),
            InterviewSection("计算机基础", ["概念题"], 3, 4),
            InterviewSection("手撕算法", ["算法题"], 1, 2),
            InterviewSection("反问环节", ["开放"], 1, 1),
        ]
    return InterviewPlan(sections=sections, total_turns_hint=12)


@dataclass
class JDInfo:
    role_title: str = ""
    company: Optional[str] = None
    experience_years: Optional[str] = None
    seniority: str = "mid"  # junior / mid / senior
    hard_skills: list[str] = field(default_factory=list)
    soft_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    focus_points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_title": self.role_title,
            "company": self.company,
            "experience_years": self.experience_years,
            "seniority": self.seniority,
            "hard_skills": self.hard_skills,
            "soft_skills": self.soft_skills,
            "responsibilities": self.responsibilities,
            "focus_points": self.focus_points,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JDInfo":
        return cls(
            role_title=str(d.get("role_title") or ""),
            company=d.get("company") or None,
            experience_years=d.get("experience_years") or None,
            seniority=str(d.get("seniority") or "mid"),
            hard_skills=list(d.get("hard_skills") or []),
            soft_skills=list(d.get("soft_skills") or []),
            responsibilities=list(d.get("responsibilities") or []),
            focus_points=list(d.get("focus_points") or []),
        )


@dataclass
class InterviewerProfile:
    role_title: str = ""
    company: Optional[str] = None
    style: str = "严谨"  # e.g. 严谨 / 引导式 / 温和
    tone: str = "专业、友好、有耐心"
    focus_points: list[str] = field(default_factory=list)
    company_context: str = ""  # research notes ("" -> JD-only interviewer)
    sources: list[str] = field(default_factory=list)
    opening: str = ""  # 开场白 (fallback when LLM unavailable)
    plan: InterviewPlan = field(default_factory=InterviewPlan)  # 面试环节剧本

    def to_dict(self) -> dict[str, Any]:
        return {
            "role_title": self.role_title,
            "company": self.company,
            "style": self.style,
            "tone": self.tone,
            "focus_points": self.focus_points,
            "company_context": self.company_context,
            "sources": self.sources,
            "opening": self.opening,
            "plan": self.plan.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InterviewerProfile":
        return cls(
            role_title=str(d.get("role_title") or ""),
            company=d.get("company") or None,
            style=str(d.get("style") or "严谨"),
            tone=str(d.get("tone") or "专业、友好、有耐心"),
            focus_points=list(d.get("focus_points") or []),
            company_context=str(d.get("company_context") or ""),
            sources=list(d.get("sources") or []),
            opening=str(d.get("opening") or ""),
            plan=_profile_plan(d),
        )


def _profile_plan(d: dict[str, Any]) -> "InterviewPlan":
    """Plan with fallback: profiles saved before the plan feature have none."""
    plan = InterviewPlan.from_dict(d.get("plan"))
    if not plan.sections:
        plan = default_plan(ai=is_ai_role(str(d.get("role_title") or "")))
    return plan


@dataclass
class Message:
    """One recorded chat message."""
    speaker: str
    content: str
    difficulty: Optional[int] = None
    quality: Optional[int] = None
    id: Optional[int] = None
    session_id: Optional[str] = None
    created_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "speaker": self.speaker,
            "content": self.content,
            "difficulty": self.difficulty,
            "quality": self.quality,
            "created_at": self.created_at,
        }


@dataclass
class QualityJudgment:
    """Output of the Difficulty Controller for one candidate answer."""
    quality: int = 3  # 1..5
    summary: str = ""
    facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"quality": self.quality, "summary": self.summary, "facts": self.facts}


def _safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def clamp_difficulty(value: int) -> int:
    return max(DIFF_MIN, min(DIFF_MAX, value))


def jdump(obj: Any) -> str:
    """Compact, non-ASCII-safe JSON dumps used for DB storage."""
    return json.dumps(obj, ensure_ascii=False)
