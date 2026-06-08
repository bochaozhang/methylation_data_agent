"""
LLM backend factory for MethyAgent.
Supports OpenAI GPT-4o (and compatible APIs), Anthropic Claude, ZhipuAI GLM, and Ollama.
Switch backends via config/settings.yaml without changing agent code.

Environment variables (all optional, override settings.yaml values):
    OPENAI_API_KEY      API key for OpenAI or compatible endpoint
    OPENAI_BASE_URL     Custom base URL (Azure, relay, vLLM, Ollama-compat, etc.)
                        Default: https://api.openai.com/v1
    OPENAI_MODEL        Model name override (e.g. gpt-4o-mini, deepseek-chat)
    ZHIPU_API_KEY       API key for ZhipuAI (https://open.bigmodel.cn)
    ZHIPU_MODEL         ZhipuAI model name (default: glm-4-flash)
    ANTHROPIC_API_KEY   API key for Anthropic
    ANTHROPIC_MODEL     Model name override for Anthropic
"""
import os
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel

# ZhipuAI base URL constant
_ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


def _is_zhipu_url(url: str) -> bool:
    """Return True if the URL points to ZhipuAI's endpoint."""
    return "bigmodel.cn" in url


def get_llm(config: Dict[str, Any]) -> BaseChatModel:
    """
    Instantiate and return a LangChain chat model based on config.

    Priority for each parameter:
        1. Environment variable (OPENAI_MODEL, OPENAI_BASE_URL, ...)
        2. config dict (from settings.yaml)
        3. Hard-coded default

    Supported backends:
        openai    — OpenAI or any OpenAI-compatible API (including ZhipuAI relay)
        zhipu     — ZhipuAI GLM series (explicit backend, handles auth header correctly)
        anthropic — Anthropic Claude
        ollama    — Local Ollama

    Args:
        config: The 'llm' section of settings.yaml, e.g.:
            {"backend": "zhipu", "model": "glm-4-flash", "temperature": 0}

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the backend is unsupported or the API key is missing.
    """
    backend = config.get("backend", "openai").lower()
    temperature = config.get("temperature", 0)
    max_tokens = config.get("max_tokens", 4096)
    api_key_env = config.get("api_key_env", "OPENAI_API_KEY")

    # ------------------------------------------------------------------ #
    #  Auto-detect ZhipuAI from OPENAI_BASE_URL                          #
    #  If user set backend=openai but pointed BASE_URL at bigmodel.cn,   #
    #  silently upgrade to the zhipu path so auth works correctly.        #
    # ------------------------------------------------------------------ #
    base_url_env = os.environ.get("OPENAI_BASE_URL") or config.get("base_url") or ""
    if backend == "openai" and _is_zhipu_url(base_url_env):
        backend = "zhipu"

    # ------------------------------------------------------------------ #
    #  ZhipuAI                                                            #
    # ------------------------------------------------------------------ #
    if backend == "zhipu":
        api_key = (
            os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        if not api_key:
            raise ValueError(
                "ZhipuAI API key not found. "
                "Set ZHIPU_API_KEY (or OPENAI_API_KEY) in your .env file. "
                "Get a key at: https://open.bigmodel.cn"
            )

        model = (
            os.environ.get("ZHIPU_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or config.get("model")
            or "glm-4-flash"
        )

        # Try langchain-zhipuai first; fall back to ChatOpenAI with explicit header
        try:
            from langchain_community.chat_models import ChatZhipuAI  # type: ignore
            return ChatZhipuAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                zhipuai_api_key=api_key,
            )
        except ImportError:
            pass

        # Fallback: ChatOpenAI with default_headers to force Authorization: Bearer
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=_ZHIPU_BASE_URL,
            default_headers={"Authorization": f"Bearer {api_key}"},
        )

    # ------------------------------------------------------------------ #
    #  OpenAI (or generic compatible endpoint)                            #
    # ------------------------------------------------------------------ #
    elif backend == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get(api_key_env) or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found. "
                f"Set the environment variable: {api_key_env} (or OPENAI_API_KEY)"
            )

        model = (
            os.environ.get("OPENAI_MODEL")
            or config.get("model")
            or "gpt-4o-mini"
        )

        base_url = base_url_env or None

        kwargs: Dict[str, Any] = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
        if base_url:
            kwargs["base_url"] = base_url

        return ChatOpenAI(**kwargs)

    # ------------------------------------------------------------------ #
    #  Anthropic                                                          #
    # ------------------------------------------------------------------ #
    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get(api_key_env) or os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError(
                f"Anthropic API key not found. "
                f"Set the environment variable: {api_key_env} (or ANTHROPIC_API_KEY)"
            )

        model = (
            os.environ.get("ANTHROPIC_MODEL")
            or config.get("model")
            or "claude-3-5-haiku-20241022"
        )

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    # ------------------------------------------------------------------ #
    #  Ollama (local)                                                     #
    # ------------------------------------------------------------------ #
    elif backend == "ollama":
        from langchain_ollama import ChatOllama

        model = (
            os.environ.get("OPENAI_MODEL")
            or config.get("model")
            or "llama3"
        )
        base_url = (
            base_url_env
            or "http://localhost:11434"
        )
        # Strip /v1 suffix — Ollama native API doesn't use it
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM backend: '{backend}'. "
            "Choose from: openai, zhipu, anthropic, ollama"
        )
