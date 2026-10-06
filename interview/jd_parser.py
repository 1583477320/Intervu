"""JD 解析 — extract skills, soft qualities, seniority and focus points.

Two paths:
* LLM path — the model returns a structured JSON description of the JD.
* Fallback path — deterministic regex / keyword extraction, used when the
  model reply is unusable so the pipeline never dead-ends on parsing.

The fallback is intentionally coarse. A missing API key is *not* silently
absorbed: :class:`LLMNotConfiguredError` always propagates.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .llm import LLMClient, LLMNotConfiguredError, extract_json
from .models import JDInfo

SECTION_HEADERS = [
    "任职要求", "职位要求", "岗位要求", "任职资格", "岗位职责", "工作职责",
    "职责描述", "工作内容", "职位描述", "职位诱惑", "福利待遇", "公司简介",
    "关于我们", "我们提供", "加分项",
]

# (keyword, seniority)
SENIORITY_RULES = [
    ("资深", "senior"), ("高级", "senior"), ("senior", "senior"),
    ("专家", "senior"), ("架构", "senior"), ("主管", "senior"), ("leader", "senior"),
    ("初级", "junior"), ("应届", "junior"), ("实习", "junior"), ("junior", "junior"),
    ("中级", "mid"), ("中高级", "mid"),
]

KNOWN_SKILLS = [
    "Python", "Rust", "Go", "Golang", "Java", "C++", "C#", "C语言", "PHP",
    "Ruby", "Swift", "Kotlin", "Objective-C", "JavaScript", "TypeScript",
    "Node.js", "NodeJS", "React", "Vue", "Angular", "小程序", "Flutter",
    "Android", "iOS", "HTML", "CSS", "SQL", "MySQL", "PostgreSQL", "Oracle",
    "MongoDB", "Redis", "Elasticsearch", "Kafka", "Spark", "Flink", "Hadoop",
    "ClickHouse", "TiDB", "SQLite", "Docker", "Kubernetes", "K8s", "Nginx",
    "AWS", "阿里云", "腾讯云", "Linux", "Git", "CI/CD", "微服务", "分布式",
    "高并发", "高可用", "RESTful", "gRPC", "消息队列", "RPC", "Django",
    "Flask", "FastAPI", "Spring", "SpringBoot", "MyBatis", "PyTorch",
    "TensorFlow", "机器学习", "深度学习", "大模型", "LLM", "NLP", "推荐系统",
    "搜索", "广告", "数据仓库", "数据挖掘", "BI", "A/B测试", "Serverless",
    "SRE", "DevOps", "性能优化", "单元测试", "TDD",
]

KNOWN_SOFT = [
    "沟通", "协作", "团队", "责任心", "主动性", "学习能力", "抗压",
    "逻辑思维", "自驱", "ownership", "owner", "解决问题", "英语",
]

YEARS_RE = re.compile(r"(\d+\s*[-~到至]\s*\d+\s*年|\d+\s*年(?:以上|及以上|左右)?|3年以下|1-2年)")
HEADER_ONLY = {"职责", "要求", "职位描述", "任职要求", "职位要求"}


class JDParser:
    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm = llm

    def parse(self, jd_text: str) -> JDInfo:
        text = jd_text.strip()
        if not text:
            raise ValueError("JD 内容为空")

        if self._llm is not None:
            try:
                return self._parse_with_llm(text)
            except LLMNotConfiguredError:
                raise  # 未配置 API Key：必须显式报错
            except Exception:
                # 模型调用/输出异常时降级到确定性解析路径
                pass
        return self._parse_fallback(text)

    # ------------------------------------------------------------- llm path
    def _parse_with_llm(self, text: str) -> JDInfo:
        prompt = f"""你是一名资深的 HR 技术招聘专家。请阅读下面的职位描述（JD），提取结构化信息。
只输出一个 JSON 对象，不要输出任何其他文字，格式如下：
{{
  "role_title": "岗位名称（中文，含职级关键词如 高级/资深/初级）",
  "company": "若 JD 中能明确识别出公司/团队名称则填公司名，否则填 null",
  "experience_years": "经验要求原文，如 '3-5 年'；无则填 null",
  "seniority": "junior | mid | senior 三选一",
  "hard_skills": ["技术/硬技能关键词列表，如 Python、高并发、Docker"],
  "soft_skills": ["软素质列表，如 沟通协作、责任心"],
  "responsibilities": ["岗位职责要点，2-4 条"],
  "focus_points": ["面试时应重点考察的 3-6 个考点"]
}}

=== 职位描述开始 ===
{text[:6000]}
=== 职位描述结束 ===
"""
        raw = self._llm.complete(prompt, task="jd_parse", temperature=0.2)
        data = extract_json(raw)
        return self._normalize(JDInfo.from_dict(data), text)

    # ------------------------------------------------------ fallback path
    def _parse_fallback(self, text: str) -> JDInfo:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        info = JDInfo()
        info.role_title = self._guess_role_title(lines)
        info.experience_years = self._extract_years(text)
        info.seniority = self._extract_seniority(text)
        info.company = self._extract_company(text)
        info.hard_skills = self._extract_skills(text)
        info.soft_skills = self._extract_soft(text)
        info.responsibilities = self._extract_section(lines, "职责")
        info.focus_points = self._extract_section(lines, "要求") or list(info.hard_skills[:5])
        return info

    def _normalize(self, info: JDInfo, text: str) -> JDInfo:
        """Fill gaps left by the LLM with fallback extractions."""
        if not info.role_title:
            info.role_title = self._guess_role_title([ln.strip() for ln in text.splitlines() if ln.strip()]) or "该职位"
        if not info.hard_skills:
            info.hard_skills = self._extract_skills(text)
        if not info.responsibilities:
            info.responsibilities = self._extract_section(
                [ln.strip() for ln in text.splitlines() if ln.strip()], "职责"
            )
        if not info.focus_points:
            info.focus_points = list(info.hard_skills[:5]) or ["岗位核心技能"]
        if info.seniority not in ("junior", "mid", "senior"):
            info.seniority = self._extract_seniority(text)
        return info

    # -------------------------------------------------------- heuristics
    @staticmethod
    def _guess_role_title(lines: list[str]) -> str:
        for ln in lines:
            clean = re.sub(r"^[#\-\*\d\.、\s]+", "", ln).strip()
            if not clean or len(clean) > 40:
                continue
            if any(h in clean for h in HEADER_ONLY):
                continue
            if any(c in clean for c in "，。：；") or "负责" in clean or "熟悉" in clean:
                continue  # 描述句而非标题
            return clean
        return "该职位"

    @staticmethod
    def _extract_years(text: str) -> Optional[str]:
        m = YEARS_RE.search(text)
        return m.group(1).replace(" ", "") if m else None

    @staticmethod
    def _extract_seniority(text: str) -> str:
        lower = text.lower()
        for kw, level in SENIORITY_RULES:
            if kw.lower() in lower:
                return level
        return "mid"

    @staticmethod
    def _extract_company(text: str) -> Optional[str]:
        patterns = [
            r"公司(?:名称|名字)?[：:]\s*([^\s，,。；;]+)",
            r"关于\s*([\u4e00-\u9fa5A-Za-z0-9]{2,20})\s*(?:公司|集团)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        # a leading line like "XX科技 招聘" 
        for ln in text.splitlines()[:3]:
            m = re.match(r"^([\u4e00-\u9fa5A-Za-z0-9]{2,20}(?:科技|技术|网络|信息|软件|集团|公司))\s*$", ln.strip())
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _extract_skills(text: str) -> list[str]:
        found: list[str] = []
        for kw in KNOWN_SKILLS:
            if re.search(rf"(?<![A-Za-z]){re.escape(kw)}(?![A-Za-z])", text, re.IGNORECASE):
                found.append(kw)
        # de-dup case variants (Go vs GO) via title casing for ASCII words
        seen, out = set(), []
        for kw in found:
            key = kw.lower()
            if key not in seen:
                seen.add(key)
                out.append(kw)
        return out

    @staticmethod
    def _extract_soft(text: str) -> list[str]:
        found: list[str] = []
        for kw in KNOWN_SOFT:
            if re.search(rf"(?<![A-Za-z]){re.escape(kw)}(?![A-Za-z])", text, re.IGNORECASE):
                found.append(kw)
        return found

    @staticmethod
    def _extract_section(lines: list[str], kind: str) -> list[str]:
        """Grab bullet lines under a 职责 / 要求 header until the next section."""
        wanted = "职责" if kind == "职责" else "要求"
        started = False
        out: list[str] = []
        for ln in lines:
            is_header = any(ln.startswith(h) or h in ln[:6] for h in SECTION_HEADERS)
            if is_header:
                header_hit = any(h in ln for h in SECTION_HEADERS)
                if started:
                    break
                if header_hit and (("职责" in ln and wanted == "职责") or (("要求" in ln or "资格" in ln) and wanted == "要求")):
                    started = True
                continue
            if started:
                clean = re.sub(r"^[#\-\*\d\.、\s]+", "", ln).strip()
                if clean and len(clean) > 4:
                    out.append(clean)
        # dedupe but keep order
        seen, dedup = set(), []
        for item in out:
            if item not in seen:
                seen.add(item)
                dedup.append(item)
        return dedup[:6]
