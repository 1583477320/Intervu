import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from interview.difficulty import adapt_difficulty, tier_name
from interview.orchestrator import wants_to_stop


def test_two_good_answers_raise():
    assert adapt_difficulty(1, [5, 5]) == (2, 1)
    assert adapt_difficulty(2, [4, 5]) == (3, 1)


def test_two_weak_answers_lower():
    assert adapt_difficulty(3, [1, 2]) == (2, -1)
    assert adapt_difficulty(2, [2, 1]) == (1, -1)


def test_mixed_answers_keep_level():
    assert adapt_difficulty(2, [5, 2]) == (2, 0)
    assert adapt_difficulty(2, [5]) == (2, 0)  # 样本不足不调档


def test_levels_are_clamped():
    # 已是最高档时即使连续优秀也不再上调
    assert adapt_difficulty(3, [5, 5]) == (3, 1)
    # 已是最低档时即使连续卡壳也不再下调
    assert adapt_difficulty(1, [1, 1]) == (1, -1)


def test_tier_names():
    assert tier_name(1) == "L1 · 概念基础"
    assert "L3" in tier_name(3)


def test_stop_intent_detection():
    assert wants_to_stop("我想结束面试")
    assert wants_to_stop("今天就到这里吧，谢谢")
    assert wants_to_stop("结束")
    assert wants_to_stop("Thank you for the interview")
    assert not wants_to_stop("我结束了上一份工作，目前正在看新机会")  # 长句中的“结束”
    assert not wants_to_stop("我负责的支付项目刚刚成功上线")
