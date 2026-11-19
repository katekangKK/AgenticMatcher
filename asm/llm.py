"""OpenAI-compatible chat transport. Credentials never enter saved run records."""
import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError


@dataclass
class ChatResult:
    content: str
    finish_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def is_truncated(self):
        return self.finish_reason == "length"

    def __str__(self):
        return self.content


class ConfigurationError(ValueError):
    """Actionable configuration error whose message contains no user values."""


class LLMClient:
    _MAX_RETRIES = 3
    _BASE_DELAY = 2.0

    def __init__(self, model: Optional[str] = None,
                 api_key=None, base_url=None, api_key_env=None):
        self.model = (model if model is not None else os.getenv("ASM_MODEL", "")).strip()
        if not self.model:
            raise ConfigurationError("Specify a model with --model or ASM_MODEL.")
        endpoint = (base_url if base_url is not None else os.getenv("ASM_BASE_URL", "")).strip()
        if not endpoint:
            raise ConfigurationError("Specify a base URL with --base-url or ASM_BASE_URL.")
        try:
            url = urlsplit(endpoint)
            port = url.port
        except ValueError:
            raise ConfigurationError("Base URL must be a valid URL with a hostname and optional port.") from None
        if not url.hostname or any(c.isspace() for c in endpoint) or port == 0:
            raise ConfigurationError("Base URL must be a valid URL with a hostname and optional port.")
        if url.username or url.password or url.query or url.fragment:
            raise ConfigurationError("Base URL must not contain credentials, query parameters, or fragments.")
        if url.scheme != "https" and not (url.scheme == "http" and url.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise ConfigurationError("Use HTTPS, or HTTP on localhost for a local server.")
        env_name = api_key_env if api_key_env is not None else "ASM_API_KEY"
        key = (api_key if api_key is not None else os.getenv(env_name, "")).strip()
        if not key:
            raise ConfigurationError("Set ASM_API_KEY, or the environment variable selected by --api-key-env.")
        self.client = OpenAI(base_url=endpoint, api_key=key, timeout=120.0, max_retries=0)

    def chat(self, messages, max_tokens=8192, temperature=0.7, top_p=0.9, stop=None):
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p, stop=stop, stream=False)
                usage = getattr(response, "usage", None)
                choice = response.choices[0] if response.choices else None
                return ChatResult(
                    content=(choice.message.content or "").strip() if choice else "",
                    finish_reason=(choice.finish_reason or "stop") if choice else "stop",
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0)
            except Exception as exc:
                retryable = isinstance(exc, (APIConnectionError, APITimeoutError)) or (
                    isinstance(exc, APIStatusError) and exc.status_code in {429, 500, 502, 503, 504, 529})
                if retryable and attempt < self._MAX_RETRIES:
                    warnings.warn("Provider request failed; retrying without logging response details.")
                    time.sleep(self._BASE_DELAY * 2 ** attempt)
                    continue
                # Provider bodies and chained exceptions can contain request headers,
                # credentials or prompts. Do not include them in public exceptions.
                raise RuntimeError("Provider request failed. Check credentials, endpoint, model and quota.") from None

    def batch_chat(self, messages_list, **kwargs):
        return [self.chat(messages, **kwargs) for messages in messages_list]
