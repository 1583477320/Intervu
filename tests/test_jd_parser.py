import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fake_llm import FakeLLM
from interview.jd_parser import JDParser

SAMPLE_JD = """字节跳动 · 高级后端工程师（交易中台方向）

【关于我们】
字节跳动是全球领先的互联网科技公司，旗下拥有抖音、TikTok、飞书等产品。

【岗位职责】
1. 负责交易中台核心服务的设计、开发与优化，支撑大规模交易场景；
2. 参与高并发、高可用分布式系统的架构设计与演进。

【任职要求】
1. 本科及以上学历，计算机相关专业，3-5 年及以上后端开发经验；
2. 精通 Python 或 Go/Rust，具备扎实的计算机基础；
3. 熟悉 MySQL、Redis、消息队列（Kafka），理解其原理并有实战经验；
4. 熟悉分布式系统、微服务架构，对高并发、高可用场景有深入理解；
5. 具备良好的沟通协作能力与责任心。
"""


def test_fallback_parse_extracts_core_fields():
    info = JDParser(llm=None).parse(SAMPLE_JD)
    assert "高级后端" in info.role_title
    assert info.seniority == "senior"
    assert "Python" in info.hard_skills
    assert "MySQL" in info.hard_skills
    assert "Redis" in info.hard_skills
    assert info.responsibilities and any("交易" in x or "架构" in x for x in info.responsibilities)
    assert info.focus_points  # 非空


def test_fallback_parse_handles_mid_level_jd():
    jd = """中级前端工程师

岗位职责：
- 负责 Web 端 React 应用的开发与维护；
- 与后端协作完成接口联调。

任职要求：
1. 2-4 年前端开发经验；
2. 熟悉 JavaScript、TypeScript 与 React；
3. 熟悉 HTTP 协议，了解浏览器渲染原理。
"""
    info = JDParser(llm=None).parse(jd)
    assert info.seniority in ("mid", "senior")
    assert "React" in info.hard_skills
    assert info.experience_years and "年" in info.experience_years


def test_fake_llm_parse_path():
    info = JDParser(llm=FakeLLM()).parse(SAMPLE_JD)
    assert info.role_title == "后端工程师"
    assert info.seniority == "mid"
    assert "Python" in info.hard_skills


def test_empty_jd_raises():
    try:
        JDParser(llm=None).parse("   ")
        assert False, "空 JD 应当抛错"
    except ValueError:
        pass
