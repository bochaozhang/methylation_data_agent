"""
LLM backend factory for MethyAgent.
Supports OpenAI GPT-4o, Anthropic Claude, and local Ollama models.
Switch backends via config/settings.yaml without changing agent code.
"""
import os
from typing import Any, Dict

from langchain_core.language_models import BaseChatModel


def get_llm(config: Dict[str, Any]) -> BaseChatModel:
    """
    Instantiate and return a LangChain chat model based on config.

    Args:
        config: The 'llm' section of settings.yaml, e.g.:
            {
                "backend": "openai",
                "model": "gpt-4o",
                "api_key_env": "OPENAI_API_KEY",
                "temperature": 0,
                "max_tokens": 4096
            }

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the backend is unsupported or the API key is missing.
    """
    backend = config.get("backend", "openai").lower()
    model = config.get("model", "gpt-4o")
    temperature = config.get("temperature", 0)
    max_tokens = config.get("max_tokens", 4096)
    api_key_env = config.get("api_key_env", "")

    if backend == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"OpenAI API key not found. Set the environment variable: {api_key_env}"
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    elif backend == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise ValueError(
                f"Anthropic API key not found. Set the environment variable: {api_key_env}"
            )
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    elif backend == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            temperature=temperature,
        )

    else:
        raise ValueError(
            f"Unsupported LLM backend: '{backend}'. "
            "Choose from: openai, anthropic, ollama"
        )
