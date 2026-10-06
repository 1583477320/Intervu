"""难度控制器（Difficulty Controller）。

* Input : recent candidate answers + focus points
* Output: per-answer quality judgment, plus a difficulty signal (-1 / 0 / +1)

Judging is done by a *separate* model call from the one that asks questions —
「面试官与评分解耦」—— so the same agent is not both player and referee.
Difficulty adaptation is deterministic, mirroring the design doc:

    "连续 2 题答对 → 升档；连续 2 题卡壳 → 降档并切换题型"

There is no offline scoring proxy: a judge always runs against the real
configured LLM, and failures surface as errors instead of being silently
replaced by length-based heuristics.
"""
from __future__ import annotations

from .llm import LLMClient, LLMNotConfiguredError, extract_json
from .models import DIFF_INITIAL, DIFF_MAX, DIFF_MIN, QualityJudgment

GOOD_THRESHOLD = 4  # quality >= 4 counts as a solid answer
WEAK_THRESHOLD = 2  # quality <= 2 counts as stuck / weak
STREAK = 2          # consecutive answers needed to move a tier

TIER_NAMES = {
    DIFF_MIN: "L1 · 概念基础",
    2: "L2 · 应用场景",
    DIFF_MAX: "L3 · 深度设计",
}

def tier_name(difficulty: int) -> str:
    return TIER_NAMES.get(difficulty, TIER_NAMES[DIFF_INITIAL])


def adapt_difficulty(current: int, quality_history: list[int]) -> tuple[int, int]:
    """Return (next_difficulty, signal) from the last two judgments."""
    recent = [q for q in quality_history if q is not None]
    signal = 0
    if len(recent) >= STREAK:
        last = recent[-STREAK:]
        if all(q >= GOOD_THRESHOLD for q in last):
            signal = 1
        elif all(q <= WEAK_THRESHOLD for q in last):
            signal = -1
    nxt = max(DIFF_MIN, min(DIFF_MAX, current + signal))
    return nxt, signal


class DifficultyController:
    """Judges candidate answers and applies the tier-adjustment policy."""

    def __init__(self, llm: LLMClient):
        if llm is None:
            raise LLMNotConfiguredError("难度控制器缺少 LLM 客户端。")
        self._llm = llm

    def judge(self, focus_points: list[str], transcript_tail: str, answer: str) -> QualityJudgment:
        """Score the latest candidate answer (1..5) and extract key facts."""
        focus_txt = "、".join(focus_points) if focus_points else "岗位核心技能"
        prompt = f"""你是一位独立的面试评分员，与提问的面试官完全分离，只负责客观评价候选人最新一次回答的质量。

岗位考察重点：{focus_txt}

之前的对话（供参考）：
{transcript_tail or '（无）'}

请从正确性、深度、逻辑表达三个角度，评价候选人最后一段【候选人回答】的质量。
只输出一个 JSON 对象：
{{
  "quality": 1-5 的整数（1 完全没答上/答错；2 有明显错误或很浅；3 中规中矩；4 正确且有细节；5 正确、深入且有项目佐证或独特见解），
  "summary": "一句话点评（中文，20-40 字）",
  "facts": ["从回答中抽取出可确认的能力事实，如 '候选人熟悉 Redis 缓存淘汰策略'；没有可确认的则空数组 []"]
}}
不要输出其它文字。

【候选人回答】
{answer}"""
        raw = self._llm.complete(prompt, task="judge", temperature=0.2)
        data = extract_json(raw)
        quality = int(data.get("quality", 3))
        facts = data.get("facts") or []
        if not isinstance(facts, list):
            facts = []
        return QualityJudgment(
            quality=max(1, min(5, quality)),
            summary=str(data.get("summary") or ""),
            facts=[str(f) for f in facts if str(f).strip()],
        )
