import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_llm import FakeLLM
from interview.difficulty import DIFF_INITIAL
from interview.orchestrator import Orchestrator
from interview.recorder import Recorder

JD = """高级后端工程师

岗位职责：
- 负责后端核心服务的设计、开发与优化；
- 参与高并发系统的架构演进，保障稳定与性能。

任职要求：
1. 3-5 年 Python / Rust 后端开发经验；
2. 熟悉 MySQL、Redis 与消息队列；
3. 熟悉分布式、微服务架构；
4. 良好的沟通协作能力与责任心。
"""

LONG_ANSWER = (
    "我之前负责的订单服务每天有千万级请求。当时我们用的是 Python FastAPI 加 MySQL，"
    "后来读热点上来之后我们在前面加了一层 Redis 缓存，把订单快照按用户维度做了缓存，"
    "并配合消息队列削峰，把核心链路的 P99 延迟从 800ms 降到了 120ms 左右。"
    "同时我们把写操作拆成了多个独立领域服务，用幂等键保证最终一致，上线时做了全链路压测，"
    "整个方案平稳度过了双十一大促。"
)
SHORT_ANSWER = "不太确定，我理解得比较浅。"


def make_orchestrator(tmp_path):
    rec = Recorder(tmp_path / "test.db")
    return Orchestrator(llm=FakeLLM(), recorder=rec)


def test_full_lifecycle_with_report(tmp_path):
    orch = make_orchestrator(tmp_path)

    # --- 初始化
    start = orch.start_interview(JD)
    sid = start["session_id"]
    assert sid
    assert start["status"] == "interviewing"
    assert start["difficulty"] == DIFF_INITIAL
    assert start["opening"]
    assert start["max_turns"] == 12

    # --- 多轮作答：长回答 -> 高质量 -> 难度应上调
    last_tier = DIFF_INITIAL
    for _ in range(4):
        r = orch.answer(sid, LONG_ANSWER)
        assert r["finished"] is False
        assert r["quality"] == 5
        assert r["reply"]
        last_tier = r["difficulty"]
    assert last_tier == 3  # 连续高质量应答后应升到 L3

    # --- 卡壳两轮 -> 难度下调
    for _ in range(2):
        r = orch.answer(sid, SHORT_ANSWER)
        assert r["quality"] <= 2
    assert r["difficulty"] == 2

    # --- 记录与状态
    history = orch.get_history(sid)
    assert len(history["messages"]) >= 8
    speakers = [m["speaker"] for m in history["messages"]]
    assert "candidate" in speakers and "interviewer" in speakers
    assert len(history["meta"]["quality_history"]) == 6
    assert history["meta"]["difficulty_history"][-1] == 2

    # --- 结束并产出报告
    end = orch.end_interview(sid)
    assert end["finished"] is True
    report = end["report_md"]
    for section in ["候选人评估报告", "面试概览", "维度评分", "总体评价", "录用建议", "完整对话记录"]:
        assert section in report, f"报告缺少段落: {section}"
    # 完整对话记录以对话形式展示：发言人加粗、逐轮分段
    assert "**面试官**：" in report
    assert "**候选人**：" in report

    # --- 持久化：重新打开会话可回溯
    again = orch.get_history(sid)
    assert again["status"] == "finished"
    assert again["report_md"] == report
    assert orch._recorder.count_messages(sid) == len(again["messages"])


def test_candidate_can_request_end(tmp_path):
    orch = make_orchestrator(tmp_path)
    sid = orch.start_interview(JD)["session_id"]
    r = orch.answer(sid, "结束面试")
    assert r["finished"] is True
    assert r["reason"] == "candidate_requested"
    assert "候选人评估报告" in r["report_md"]


def test_answer_after_finish_raises(tmp_path):
    orch = make_orchestrator(tmp_path)
    sid = orch.start_interview(JD)["session_id"]
    orch.answer(sid, "结束面试")
    try:
        orch.answer(sid, "继续")
        assert False, "已结束的会话不应允许继续作答"
    except ValueError:
        pass


def test_interviewer_prompts_are_colloquial_and_ask_one_question(tmp_path):
    """首问与追问都要口语化、且一次只问一个问题。"""
    llm = FakeLLM()
    rec = Recorder(tmp_path / "test.db")
    orch = Orchestrator(llm=llm, recorder=rec)
    sid = orch.start_interview(JD)["session_id"]
    orch.submit_resume(sid, "本科计算机，实习做过订单服务的缓存与削峰优化。")
    orch.answer(sid, LONG_ANSWER)

    for task in ("interviewer_first", "interviewer_turn"):
        prompts = [p for t, p in llm.calls if t == task]
        assert prompts, f"未调用 {task}"
        assert "口语" in prompts[0]
        assert "一次只问一个问题" in prompts[0]


def test_reverse_section_hands_over_questioning(tmp_path):
    """进入最后一环「反问」后，面试官不再出技术题，而是把提问权交给候选人。"""
    llm = FakeLLM()
    rec = Recorder(tmp_path / "test.db")
    orch = Orchestrator(llm=llm, recorder=rec)
    sid = orch.start_interview(JD)["session_id"]

    reply, section = None, None
    for _ in range(10):
        before = sum(1 for t, _ in llm.calls if t == "interviewer_turn")
        r = orch.answer(sid, LONG_ANSWER)
        reply, section = r["reply"], r["section"]
        if section == "反问环节":
            # 本轮不再调用出题 LLM（交接句是确定性输出）
            after = sum(1 for t, _ in llm.calls if t == "interviewer_turn")
            assert after == before
            break

    assert section == "反问环节", "应推进到反问环节"
    assert "你有什么想问我的吗" in reply


def test_unknown_session_raises(tmp_path):
    orch = make_orchestrator(tmp_path)
    try:
        orch.answer("nope", "你好")
        assert False, "未知会话应报错"
    except KeyError:
        pass
