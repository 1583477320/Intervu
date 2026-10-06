"""Orchestrator 编排层 — session lifecycle + routing.

Lifecycle: 初始化（JD→画像）→ 面试（多轮）→ 收尾 → 报告.

Each candidate answer runs the pipeline:
    Recorder 落库 → DifficultyController 评分调档 → Interviewer 追问/出题 → Recorder 落库
and every interview ends either when the candidate asks to stop, when the
target question count is reached, or when the session is explicitly ended.
"""
from __future__ import annotations

import re
import threading
import uuid
from typing import Any, Optional

from .builder import InterviewerBuilder
from .difficulty import DifficultyController, adapt_difficulty, tier_name
from .evaluator import Evaluator
from .interviewer import Interviewer
from .llm import LLMClient, LLMNotConfiguredError
from .models import (
    DIFF_INITIAL,
    InterviewSection,
    JDInfo,
    Message,
    SPEAKER_CANDIDATE,
    SPEAKER_INTERVIEWER,
    STATUS_FINISHED,
    STATUS_INTERVIEWING,
)
from .recorder import Recorder
from .style_researcher import StyleResearcher

STOP_PATTERNS = [
    r"结束面试", r"今天就到(?:这里|这)?", r"不面了", r"到此为止", r"到这里吧",
    r"没有(?:其他|别的)?问题", r"没什么(?:要)?问", r"停止面试", r"我想(?:要)?结束",
    r"^(?:好的?)?结束$", r"^(?:好的?)?再见$",
    r"\b(?:thanks?|thx)\b.*\b(?:interview|today|结束)\b",
]


def wants_to_stop(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    for pat in STOP_PATTERNS:
        if re.search(pat, t):
            return True
    # “结束”单独出现在长句中间不算（例如“我结束了上一份工作”）
    if re.search(r"\b结束\b|结束", t) and len(t) <= 24 and "工作" not in t and "项目" not in t:
        return True
    return False


class Orchestrator:
    """One orchestrator per interview session."""

    def __init__(
        self,
        llm: LLMClient,
        recorder: Optional[Recorder] = None,
        *,
        max_turns: int = 12,
        transcript_window: int = 12,
        style_researcher: Optional[StyleResearcher] = None,
    ):
        if llm is None:
            raise LLMNotConfiguredError("编排器缺少 LLM 客户端。")
        self._llm = llm
        self._recorder = recorder or Recorder("data/interview.db")
        self._builder = InterviewerBuilder(llm, style_researcher=style_researcher)
        self._difficulty = DifficultyController(llm)
        self._evaluator = Evaluator(llm)
        self._max_turns = max_turns
        self._window = transcript_window
        self._locks: dict[str, threading.Lock] = {}
        self._lock_guard = threading.Lock()

    @property
    def llm(self) -> LLMClient:
        return self._llm

    # ================================================================ start
    def start_interview(self, jd_text: str) -> dict[str, Any]:
        jd = self._builder.parse_jd(jd_text)
        profile, research = self._builder.build(jd)

        session_id = uuid.uuid4().hex[:12]
        facts: list[str] = []
        meta: dict[str, Any] = {"quality_history": [], "difficulty_history": []}
        with self._lock(session_id):
            self._recorder.create_session(
                session_id,
                profile=profile.to_dict(),
                jd=jd.to_dict(),
                difficulty=DIFF_INITIAL,
                facts=facts,
            )
            self._recorder.update_session_turn(session_id, DIFF_INITIAL, facts, meta)

            interviewer = Interviewer(self._llm, profile, jd)
            opening = interviewer.opening_message()
            self._recorder.save_message(
                Message(
                    session_id=session_id,
                    speaker=SPEAKER_INTERVIEWER,
                    content=opening,
                    difficulty=DIFF_INITIAL,
                )
            )

        return {
            "session_id": session_id,
            "role_title": profile.role_title,
            "company": profile.company,
            "style": profile.style,
            "tone": profile.tone,
            "focus_points": profile.focus_points,
            "company_context": profile.company_context,
            "sources": research.sources,
            "difficulty": DIFF_INITIAL,
            "tier": tier_name(DIFF_INITIAL),
            "opening": opening,
            "status": STATUS_INTERVIEWING,
            "max_turns": self._max_turns,
        }

    # =============================================================== resume
    def submit_resume(self, session_id: str, resume_text: str) -> dict[str, Any]:
        """Save the candidate's resume, then have the interviewer ask the first
        question anchored on the resume. Only allowed once, before any answer."""
        session = self._require_session(session_id)
        if session["status"] == STATUS_FINISHED:
            raise ValueError("该面试已结束，无法提交简历")
        resume_text = (resume_text or "").strip()
        if not resume_text:
            raise ValueError("简历内容不能为空")

        with self._lock(session_id):
            session = self._require_session(session_id)
            if session.get("resume"):
                raise ValueError("该会话已提交过简历，请直接继续作答")

            profile = self._profile(session)
            jd = self._jd(session)
            interviewer = Interviewer(self._llm, profile, jd)
            question = interviewer.first_question(resume=resume_text)

            self._recorder.save_resume(session_id, resume_text)
            self._recorder.save_message(
                Message(
                    session_id=session_id,
                    speaker=SPEAKER_INTERVIEWER,
                    content=question,
                    difficulty=DIFF_INITIAL,
                )
            )

        return {
            "reply": question,
            "difficulty": DIFF_INITIAL,
            "tier": tier_name(DIFF_INITIAL),
            "signal": 0,
            "turn": 0,
        }

    # ================================================================ turns
    def answer(self, session_id: str, text: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        if session["status"] == STATUS_FINISHED:
            raise ValueError("该面试已结束，无法继续作答")
        text = (text or "").strip()
        if not text:
            raise ValueError("回答内容不能为空")

        with self._lock(session_id):
            profile = self._profile(session)
            jd = self._jd(session)
            interviewer = Interviewer(self._llm, profile, jd)

            # 1) record candidate turn
            self._recorder.save_message(
                Message(session_id=session_id, speaker=SPEAKER_CANDIDATE, content=text)
            )

            # 2) difficulty controller: judge this answer
            messages = self._recorder.list_messages(session_id)
            tail = self._format_transcript(messages, window=self._window)
            focus = profile.focus_points or jd.focus_points or jd.hard_skills[:5]
            judgment = self._difficulty.judge(focus, tail, text)

            # 3) adapt difficulty with deterministic rules
            meta = dict(session["meta"] or {})
            history = list(meta.get("quality_history") or [])
            history.append(judgment.quality)
            meta["quality_history"] = history

            current_diff = int(session["difficulty"] or DIFF_INITIAL)
            diff_trace = list(meta.get("difficulty_history") or [DIFF_INITIAL])
            new_diff, signal = adapt_difficulty(current_diff, history)
            diff_trace.append(new_diff)
            meta["difficulty_history"] = diff_trace
            meta["last_judgment"] = {"quality": judgment.quality, "summary": judgment.summary}

            # 4) interviewer asks next / follows up
            turn_no = len(history)  # 每轮候选人作答对应一个新提问
            section: Optional[InterviewSection] = None
            section_state: dict[str, int] = {}
            if wants_to_stop(text):
                reply = interviewer.closing_message(requested=True)
                reason = "candidate_requested"
            elif turn_no > self._max_turns:
                reply = interviewer.closing_message(requested=False)
                reason = "max_turns_reached"
            else:
                section, section_state = self._advance_section(session, meta, turn_no)
                reply = interviewer.ask(
                    difficulty=new_diff,
                    last_judgment_summary=judgment.summary,
                    last_quality=judgment.quality,
                    transcript_tail=tail,
                    facts=list(session["facts"] or []) + judgment.facts,
                    turn_no=turn_no,
                    resume=session.get("resume") or "",
                    section=section,
                    section_turn=section_state.get("turns_in_section", 0),
                )
                reason = None

            # 5) persist interviewer turn + updated state
            self._recorder.save_message(
                Message(
                    session_id=session_id,
                    speaker=SPEAKER_INTERVIEWER,
                    content=reply,
                    difficulty=new_diff,
                    quality=judgment.quality,
                )
            )
            merged_facts = self._merge_facts(session["facts"], judgment.facts)
            self._recorder.update_session_turn(session_id, new_diff, merged_facts, meta)

            payload = {
                "reply": reply,
                "difficulty": new_diff,
                "tier": tier_name(new_diff),
                "signal": signal,
                "quality": judgment.quality,
                "judgment": judgment.summary,
                "turn": turn_no,
                "section": section.name if section else None,
                "section_turn": section_state.get("turns_in_section") if section else None,
                "finished": reason is not None,
            }
            if reason is not None:
                payload["reason"] = reason
                payload["report_md"] = self._finalize(session_id)
            return payload

    # ================================================================ end
    def end_interview(self, session_id: str, reason: str = "user_requested") -> dict[str, Any]:
        session = self._require_session(session_id)
        if session["status"] != STATUS_FINISHED:
            with self._lock(session_id):
                profile = self._profile(session)
                jd = self._jd(session)
                interviewer = Interviewer(self._llm, profile, jd)
                if reason in ("candidate_requested", "max_turns_reached"):
                    closing = interviewer.closing_message(requested=reason == "candidate_requested")
                else:
                    closing = interviewer.closing_message(requested=False)
                self._recorder.save_message(
                    Message(
                        session_id=session_id,
                        speaker=SPEAKER_INTERVIEWER,
                        content=closing,
                        difficulty=int(session["difficulty"] or DIFF_INITIAL),
                    )
                )
                report_md = self._finalize(session_id)
        else:
            report_md = session["report_md"] or self._finalize(session_id)
        return {"session_id": session_id, "finished": True, "report_md": report_md}

    def _finalize(self, session_id: str) -> str:
        session = self._require_session(session_id)
        messages = self._recorder.list_messages(session_id)
        profile = self._profile(session)
        jd = self._jd(session)
        report_md = self._evaluator.evaluate(
            jd, profile, messages, session["meta"] or {}, session_id=session_id
        )
        self._recorder.finish_session(session_id, report_md)
        return report_md

    # ================================================================ reads
    def get_history(self, session_id: str) -> dict[str, Any]:
        session = self._require_session(session_id)
        messages = self._recorder.list_messages(session_id)
        return {
            **session,
            "messages": [m.to_dict() for m in messages],
            "difficulty_trace": [
                m.difficulty for m in messages if m.speaker == SPEAKER_INTERVIEWER and m.difficulty
            ],
            "meta": session["meta"],
        }

    def list_sessions(self, limit: int = 20, q: str = "") -> list[dict[str, Any]]:
        return self._recorder.list_sessions(limit, q)

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        self._require_session(session_id)
        self._recorder.rename_session(session_id, title)
        return self._require_session(session_id)

    def delete_session(self, session_id: str) -> None:
        self._require_session(session_id)
        self._recorder.delete_session(session_id)

    # ================================================================ misc
    def _advance_section(
        self, session: dict[str, Any], meta: dict[str, Any], turn_no: int
    ) -> tuple[Optional[InterviewSection], dict[str, int]]:
        """Deterministic section state machine: advance when the current
        section hits its max_turns (or its min_turns with a good answer).

        Returns (current_section, plan_state); plan_state is persisted in meta.
        """
        sections = (self._profile(session).plan.sections or [])
        if not sections:
            return None, {}
        meta_plan = dict((session["meta"] or {}).get("plan_state") or {})
        idx = int(meta_plan.get("current_section") or 0)
        idx = max(0, min(idx, len(sections) - 1))
        turns_in = int(meta_plan.get("turns_in_section") or 0)
        quality = None
        try:
            quality = int((meta.get("last_judgment") or {}).get("quality"))
        except (TypeError, ValueError):
            quality = None

        sec = sections[idx]
        done = turns_in >= sec.max_turns or (
            turns_in >= sec.min_turns and quality is not None and quality >= 4
        )
        if done and idx < len(sections) - 1:
            idx += 1
            turns_in = 1
        else:
            turns_in += 1

        plan_state = {"current_section": idx, "turns_in_section": turns_in}
        meta["plan_state"] = plan_state
        return sections[idx], plan_state

    def _require_session(self, session_id: str) -> dict[str, Any]:
        session = self._recorder.get_session(session_id)
        if session is None:
            raise KeyError(f"会话不存在: {session_id}")
        return session

    def _lock(self, session_id: str) -> threading.Lock:
        with self._lock_guard:
            lock = self._locks.setdefault(session_id, threading.Lock())
        return lock

    @staticmethod
    def _profile(session: dict[str, Any]):
        from .models import InterviewerProfile

        return InterviewerProfile.from_dict(session["profile"])

    @staticmethod
    def _jd(session: dict[str, Any]) -> JDInfo:
        return JDInfo.from_dict(session["jd"])

    @staticmethod
    def _format_transcript(messages: list[Message], window: int) -> str:
        pool = messages[-window:] if window else messages
        lines = []
        for m in pool:
            who = "面试官" if m.speaker == SPEAKER_INTERVIEWER else "候选人"
            lines.append(f"{who}：{(m.content or '').replace(chr(10), ' ')[:600]}")
        return "\n".join(lines)

    @staticmethod
    def _merge_facts(old: list[str], new: list[str]) -> list[str]:
        seen, out = set(), []
        for fact in list(old) + list(new):
            f = (fact or "").strip()
            if f and f not in seen:
                seen.add(f)
                out.append(f)
        return out[:30]
