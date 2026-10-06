"""LLM adapters.

Every consumer talks to a common :class:`LLMClient` so the interviewer
engine can run against any OpenAI-compatible endpoint (OpenAI, DeepSeek,
Moonshot, local gateways...).

There is deliberately **no offline demo mode / mock model**: interviews only
run against a real configured model. When no API key is configured the system
is in an "unconfigured" state and every model call raises
:class:`LLMNotConfiguredError` with guidance to the settings UI / .env file.

The web layer drives everything through :class:`ManagedLLM` — a facade whose
underlying adapter can be swapped at runtime (key / base URL / model from the
frontend) without rebuilding the engine agents.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Optional

from openai import OpenAI

from .config import Settings


class LLMError(RuntimeError):
    """Raised when the underlying model call fails."""


class LLMNotConfiguredError(LLMError):
    """Raised when a model call is attempted but no API key is configured."""


def extract_json(text: str) -> dict:
    """Best-effort extraction of the first JSON object from a model reply."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"模型输出中未找到 JSON: {text[:200]}")
    raw = text[start : end + 1]
    # strip markdown fences if the find above started inside one
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


class LLMClient:
    """Minimal interface implemented by all adapters."""

    name: str = "base"

    def complete(self, prompt: str, task: str = "chat", temperature: float = 0.7) -> str:
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    """Thin wrapper around an OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: Settings, timeout: float = 60.0):
        self.name = settings.effective_model
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.effective_base_url,
            timeout=timeout,
            max_retries=1,
        )
        self._model = settings.effective_model

    def complete(self, prompt: str, task: str = "chat", temperature: float = 0.7) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = resp.choices[0].message.content or ""
            return content.strip()
        except Exception as exc:  # network / auth / rate-limit errors
            raise LLMError(f"LLM 调用失败（{self._model}）: {exc}") from exc

    def list_models(self) -> list[str]:
        """List the model ids available to this client (``GET /models``)."""
        try:
            resp = self._client.models.list()
            return sorted({m.id for m in resp.data})
        except Exception as exc:  # network / auth errors
            raise LLMError(f"拉取模型列表失败: {exc}") from exc


class ManagedLLM(LLMClient):
    """Runtime-configurable LLM facade used by the web layer.

    The engine's agents (builder / difficulty / interviewer / evaluator) all
    hold one reference to this object and call ``complete()`` per request, so
    :meth:`configure` can swap the underlying adapter — between different real
    OpenAI-compatible endpoints — live without rebuilding agents or in-flight
    sessions. ``name`` proxies the currently active adapter.

    When no API key is configured the facade holds no adapter (``configured``
    is ``False``) and every ``complete()`` raises :class:`LLMNotConfiguredError`.
    """

    def __init__(self, settings: Settings):
        self._lock = threading.RLock()
        self._api_key = (settings.api_key or "").strip()
        self._base_url = settings.base_url
        self._model = settings.model
        self._client: Optional[LLMClient] = self._build_client()

    # ------------------------------------------------------------ live config
    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def name(self) -> str:
        with self._lock:
            if self._client is None:
                return "未配置"
            return self._client.name

    def _build_client(self) -> Optional[LLMClient]:
        if not self._api_key:
            return None
        return OpenAICompatClient(
            Settings(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._model,
            )
        )

    def configure(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Swap the active adapter. ``None`` keeps the current value."""
        with self._lock:
            if api_key is not None:
                self._api_key = (api_key or "").strip()
            if base_url is not None:
                self._base_url = (base_url or "").strip()
            if model is not None:
                self._model = (model or "").strip()
            self._client = self._build_client()

    def complete(self, prompt: str, task: str = "chat", temperature: float = 0.7) -> str:
        with self._lock:
            client = self._client
        if client is None:
            raise LLMNotConfiguredError(
                "尚未配置 LLM：缺少 API Key。请在页面右上角「⚙ 设置」中填写，"
                "或在项目 .env 中配置 OPENAI_API_KEY 后重启。"
            )
        return client.complete(prompt, task=task, temperature=temperature)

    # ------------------------------------------------------------ connection
    @staticmethod
    def ping(api_key: str = "", base_url: str = "", model: str = "") -> tuple[bool, str]:
        """Try the given (unsaved) credentials with a tiny call.

        Returns ``(ok, message)``. Real endpoints are given a short timeout so
        a wrong host / key fails fast instead of hanging.
        """
        api_key = (api_key or "").strip()
        if not api_key:
            return False, "未填写 API Key，无法测试连接。"
        settings = Settings(api_key=api_key, base_url=base_url, model=model)
        try:
            client = OpenAICompatClient(settings, timeout=15.0)
            raw = client.complete("请只回复四个字：连接成功", task="test", temperature=0)
        except LLMError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"连接失败: {exc}"
        return True, (raw or "").strip()[:100] or "连接成功"

    @staticmethod
    def list_models(api_key: str = "", base_url: str = "") -> tuple[bool, list[str] | str]:
        """Fetch the model ids available to ``api_key`` via ``GET /models``.

        Returns ``(True, [ids])`` on success or ``(False, error_message)``.
        A short timeout is used so a wrong host / key fails fast.
        """
        api_key = (api_key or "").strip()
        if not api_key:
            return False, "未填写 API Key，无法拉取模型列表。"
        settings = Settings(api_key=api_key, base_url=base_url)
        try:
            client = OpenAICompatClient(settings, timeout=15.0)
            return True, client.list_models()
        except LLMError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"拉取模型列表失败: {exc}"
