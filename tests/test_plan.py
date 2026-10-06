import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_llm import FakeLLM
from interview.builder import InterviewerBuilder
from interview.models import InterviewPlan, default_plan, is_ai_role
from interview.orchestrator import Orchestrator
from interview.recorder import Recorder

JD_BACKEND = """后端工程师

岗位职责：
- 负责后端服务开发。
任职要求：
1. 3 年 Python 经验；2. 熟悉 MySQL。
"""

JD_AI = """大模型算法工程师

岗位职责：
- 负责 LLM/Agent 方向的算法研发与落地。
任职要求：
1. 熟悉 Transformer、RAG；2. 熟悉 Python。
"""


def test_default_plan_templates():
    # 通用岗：项目深挖 + 八股 + 算法 + 反问
    p = default_plan("mid", ai=False)
    assert [s.name for s in p.sections] == ["项目深挖", "计算机基础", "手撕算法", "反问环节"]

    # AI 岗：多一个「大模型基础」，排在八股前
    p = default_plan("junior", ai=True)
    names = [s.name for s in p.sections]
    assert "大模型基础" in names
    assert names.index("大模型基础") < names.index("计算机基础")

    # 资深岗：系统设计替代大模型环节
    p = default_plan("senior", ai=False)
    assert "系统设计" in [s.name for s in p.sections]


def test_is_ai_role():
    assert is_ai_role("大模型算法工程师")
    assert is_ai_role("后端工程师", ["LLM", "RAG"])
    assert not is_ai_role("前端工程师", ["React", "CSS"])


def test_builder_parses_llm_plan_with_fallback(tmp_path):
    llm = FakeLLM()
    builder = InterviewerBuilder(llm)

    # FakeLLM 返回了完整 plan -> 原样采用
    jd = builder.parse_jd(JD_BACKEND)
    profile, _ = builder.build(jd)
    assert [s.name for s in profile.plan.sections] == [
        "项目深挖", "计算机基础", "手撕算法", "反问环节",
    ]

    # LLM 没给 plan -> 套 default_plan 兜底
    class NoPlanLLM(FakeLLM):
        def complete(self, prompt, task="chat", temperature=0.7):
            raw = super().complete(prompt, task, temperature)
            if task == "interviewer_profile":
                import json
                data = json.loads(raw)
                data.pop("interview_plan", None)
                return json.dumps(data, ensure_ascii=False)
            return raw

    profile2, _ = InterviewerBuilder(NoPlanLLM()).build(jd)
    assert profile2.plan.sections  # 兜底模板生效
    assert isinstance(profile2.plan, InterviewPlan)


def test_section_advancement(tmp_path):
    rec = Recorder(tmp_path / "t.db")
    orch = Orchestrator(llm=FakeLLM(), recorder=rec)
    sid = orch.start_interview(JD_BACKEND)["session_id"]

    # FakeLLM plan：项目深挖 max_turns=2 -> 两轮后应进入「计算机基础」
    seen = []
    for _ in range(3):
        r = orch.answer(sid, "这一块我做了详细的方案设计和压测验证。" * 5)
        seen.append(r["section"])
    assert seen[0] == "项目深挖"
    assert seen[1] == "项目深挖"
    assert seen[2] == "计算机基础"

    # plan_state 已持久化
    history = orch.get_history(sid)
    assert history["meta"]["plan_state"]["current_section"] == 1
