import importlib
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fake_llm import FakeLLM  # noqa: E402

JD = (
    "高级后端工程师\n\n岗位职责：负责后端服务的设计与开发，保障系统稳定。\n\n"
    "任职要求：1. 3-5 年 Python 开发经验；2. 熟悉 MySQL、Redis；"
    "3. 良好的沟通协作能力。"
)

LONG_ANSWER = (
    "我做过一个千万级请求的订单服务，最初直接查 MySQL，热点数据上来之后加了 Redis 缓存，"
    "并引入消息队列削峰，把 P99 延迟从 800ms 降到了 120ms，整个链路做了全链路压测，"
    "在双十一期间平稳运行，我还写了详细的方案文档并组织了复盘。"
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Import web.server with an isolated DB path and a scripted fake LLM.

    The production service never runs a mock model; tests inject the fake as
    the ManagedLLM's underlying adapter after import.
    """
    tmp = tmp_path_factory.mktemp("api")
    os.environ["OPENAI_API_KEY"] = "sk-test-placeholder"
    os.environ["OPENAI_BASE_URL"] = ""
    os.environ["OPENAI_MODEL"] = ""
    os.environ["DB_PATH"] = str(tmp / "api.db")

    from fastapi.testclient import TestClient

    srv = importlib.import_module("web.server")
    srv.llm._client = FakeLLM()  # 无网络：用脚本化客户端替换真实适配器
    with TestClient(srv.app) as c:
        yield srv, c


def test_health(server):
    srv, client = server
    h = client.get("/api/health").json()
    assert h["ok"] is True
    assert h["configured"] is True
    assert h["model"] == "fake-test"


def test_start_answer_end_flow(server):
    srv, client = server
    # 创建会话
    r = client.post("/api/sessions", json={"jd": JD})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["session_id"]
    assert s["opening"]
    assert s["max_turns"] == 12

    sid = s["session_id"]

    # 高质量回答 → 难度升到 L3
    for _ in range(2):
        ans = client.post(f"/api/sessions/{sid}/answer", json={"answer": LONG_ANSWER})
        assert ans.status_code == 200, ans.text
        body = ans.json()
        assert body["finished"] is False
        assert body["reply"]
        assert body["quality"] == 5
    assert body["difficulty"] == 3

    # 主动结束 → 返回报告（md + html）
    end = client.post(f"/api/sessions/{sid}/end", json={"reason": "user_requested"})
    assert end.status_code == 200, end.text
    eb = end.json()
    assert eb["finished"] is True
    assert "候选人评估报告" in eb["report_md"]
    assert "<h1>" in eb["report_html"]

    # 会话可回溯
    detail = client.get(f"/api/sessions/{sid}")
    assert detail.status_code == 200
    d = detail.json()
    assert d["status"] == "finished"
    assert d["report_html"]
    assert len(d["messages"]) >= 5


RESUME = (
    "张三 · 3 年后端开发经验\n教育背景：某大学计算机科学与技术本科\n"
    "项目经历：负责订单服务开发，使用 Python/MySQL/Redis，参与高并发改造；\n"
    "技能：Python、MySQL、Redis、Docker。"
)


def _docx_bytes(text: str) -> bytes:
    import io

    from docx import Document

    buf = io.BytesIO()
    d = Document()
    d.add_paragraph(text)
    d.save(buf)
    return buf.getvalue()


def test_resume_flow(server):
    """生成面试官后先上传简历（文件形式），面试官基于简历提出第一个问题。"""
    srv, client = server
    sid = client.post("/api/sessions", json={"jd": JD}).json()["session_id"]

    # 不支持的格式 → 400
    r = client.post(f"/api/sessions/{sid}/resume", files={"file": ("resume.exe", b"x", "application/octet-stream")})
    assert r.status_code == 400
    assert "不支持" in r.json()["detail"]

    # 上传 txt 简历 → 返回基于简历的第一个问题
    r = client.post(
        f"/api/sessions/{sid}/resume",
        files={"file": ("resume.txt", RESUME.encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["reply"]
    assert body["difficulty"] == 2
    assert body["turn"] == 0

    # 简历已落库，会话详情可见
    d = client.get(f"/api/sessions/{sid}").json()
    assert d["resume"] == RESUME

    # 重复提交 → 400
    r = client.post(
        f"/api/sessions/{sid}/resume",
        files={"file": ("resume.txt", RESUME.encode("utf-8"), "text/plain")},
    )
    assert r.status_code == 400
    assert "已提交过简历" in r.json()["detail"]

    # 上传 docx 简历同样可用
    sid2 = client.post("/api/sessions", json={"jd": JD}).json()["session_id"]
    r = client.post(
        f"/api/sessions/{sid2}/resume",
        files={"file": ("resume.docx", _docx_bytes(RESUME), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )
    assert r.status_code == 200, r.text
    assert client.get(f"/api/sessions/{sid2}").json()["resume"] == RESUME

    # 文件里提取不到文字 → 400
    sid3 = client.post("/api/sessions", json={"jd": JD}).json()["session_id"]
    r = client.post(f"/api/sessions/{sid3}/resume", files={"file": ("empty.txt", "   \n  ".encode(), "text/plain")})
    assert r.status_code == 400
    assert "未能" in r.json()["detail"]

    # 提交简历后正常作答并结束
    ans = client.post(f"/api/sessions/{sid}/answer", json={"answer": LONG_ANSWER})
    assert ans.status_code == 200, ans.text
    assert ans.json()["finished"] is False
    end = client.post(f"/api/sessions/{sid}/end", json={"reason": "user_requested"})
    assert end.status_code == 200 and end.json()["finished"] is True


def test_endpoint_validation(server):
    srv, client = server
    # JD 太短
    r = client.post("/api/sessions", json={"jd": "hi"})
    assert r.status_code == 422
    # 空回答
    r = client.post("/api/sessions", json={"jd": JD})
    sid = r.json()["session_id"]
    r = client.post(f"/api/sessions/{sid}/answer", json={"answer": "   "})
    assert r.status_code == 400
    # 未知会话
    r = client.get("/api/sessions/does-not-exist")
    assert r.status_code == 404


def test_llm_runtime_config(server):
    """LLM can be reconfigured at runtime; clearing the key blocks operations."""
    srv, client = server
    # initial: configured with the placeholder key; the key itself is never echoed back
    st = client.get("/api/llm/config").json()
    assert st["configured"] is True
    assert st["api_key_set"] is True
    assert "api_key" not in st
    assert "default_base_url" in st and "default_model" in st

    # switch to a (fake, never called) real-mode config
    r = client.post(
        "/api/llm/config",
        json={"api_key": "sk-test-not-real", "base_url": "", "model": "", "persist": False},
    )
    assert r.status_code == 200, r.text
    st = r.json()
    assert st["configured"] is True
    assert st["api_key_set"] is True
    assert "sk-test-not-real" not in r.text  # never echoed back

    # connection test with empty key no longer reports an offline demo path
    t = client.post("/api/llm/test", json={"api_key": "", "base_url": "", "model": ""}).json()
    assert t["ok"] is False
    assert "未填写 API Key" in t["message"]

    # clearing the key -> unconfigured; model operations must fail with guidance
    r = client.post("/api/llm/config", json={"api_key": ""})
    assert r.status_code == 200
    assert r.json()["configured"] is False
    h = client.get("/api/health").json()
    assert h["configured"] is False

    block = client.post("/api/sessions", json={"jd": JD})
    assert block.status_code == 400
    assert "尚未配置 LLM" in block.json()["detail"]

    # restore configured state + fake adapter so other tests stay green
    r = client.post("/api/llm/config", json={"api_key": "sk-test-placeholder"})
    assert r.json()["configured"] is True
    srv.llm._client = FakeLLM()
    h = client.get("/api/health").json()
    assert h["configured"] is True


def test_sessions_manage_rename_search_delete(server):
    """Sidebar management endpoints: list / search / rename / delete."""
    srv, client = server

    # 创建两个会话
    sid1 = client.post("/api/sessions", json={"jd": JD}).json()["session_id"]
    sid2 = client.post("/api/sessions", json={"jd": JD}).json()["session_id"]

    # 列表含两者，且带消息数
    lst = client.get("/api/sessions?limit=50").json()["sessions"]
    ids = {s["id"] for s in lst}
    assert sid1 in ids and sid2 in ids
    by_id = {s["id"]: s for s in lst}
    assert by_id[sid1]["message_count"] >= 1  # 开场白已落库
    assert by_id[sid1]["title"] is None

    # 重命名
    r = client.post(f"/api/sessions/{sid1}/rename", json={"title": "张三-高级后端 一面"})
    assert r.status_code == 200, r.text
    assert r.json()["title"] == "张三-高级后端 一面"

    # 搜索：命中 title
    hit = client.get("/api/sessions?q=" + "张三").json()["sessions"]
    assert [s["id"] for s in hit] == [sid1]
    # 搜索：命中 role_title（FakeLLM 解析出的默认岗位名，无自定义 title 的会话也能搜到）
    hit2 = client.get("/api/sessions?q=" + "后端").json()["sessions"]
    assert sid2 in {s["id"] for s in hit2}
    # 搜索无结果
    empty = client.get("/api/sessions?q=" + "不存在岗位xyz").json()["sessions"]
    assert empty == []

    # 详情可回看（含消息）
    detail = client.get(f"/api/sessions/{sid1}")
    assert detail.status_code == 200
    d = detail.json()
    assert d["status"] == "interviewing"
    assert len(d["messages"]) >= 1

    # 删除：先删 sid1，消息一并删除；再删 sid2
    r = client.delete(f"/api/sessions/{sid1}")
    assert r.status_code == 200 and r.json()["ok"] is True
    assert client.get(f"/api/sessions/{sid1}").status_code == 404
    assert client.delete(f"/api/sessions/{sid1}").status_code == 404  # 重复删除

    client.delete(f"/api/sessions/{sid2}")
    remain = client.get("/api/sessions").json()["sessions"]
    assert sid1 not in {s["id"] for s in remain}


def test_llm_models_list(server, monkeypatch):
    """POST /api/llm/models fetches the model ids for the account."""
    srv, client = server

    captured = {}

    def fake_list_models(api_key, base_url):
        captured["api_key"] = api_key
        captured["base_url"] = base_url
        return True, ["gpt-4o-mini", "deepseek-chat", "moonshot-v1-8k"]

    monkeypatch.setattr("web.server.ManagedLLM.list_models", staticmethod(fake_list_models))

    # 表单里给 Key → 用表单凭据拉取
    r = client.post(
        "/api/llm/models",
        json={"api_key": "sk-typed-here", "base_url": "https://api.deepseek.com/v1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["models"] == ["gpt-4o-mini", "deepseek-chat", "moonshot-v1-8k"]
    assert captured["api_key"] == "sk-typed-here"
    assert captured["base_url"] == "https://api.deepseek.com/v1"

    # 表单为空 → 回退到当前已配置凭据
    r = client.post("/api/llm/models", json={"api_key": "", "base_url": ""})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert captured["api_key"] == "sk-test-placeholder"

    # 拉取失败 → ok=False + message
    def failing(api_key, base_url):
        return False, "拉取模型列表失败: 401 auth"

    monkeypatch.setattr("web.server.ManagedLLM.list_models", staticmethod(failing))
    r = client.post("/api/llm/models", json={"api_key": "", "base_url": ""})
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "401" in r.json()["message"]
