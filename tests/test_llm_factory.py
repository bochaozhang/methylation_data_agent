"""
Tests for utils.llm_factory.get_llm(json_mode=...) — the response_format
(JSON mode) switch for glm-5.2 / OpenAI-compatible backends.

JSON mode routes response_format={"type":"json_object"} through model_kwargs
(the installed langchain_openai has no direct response_format field, so a
top-level response_format kwarg is forwarded to model_kwargs with a warning;
setting model_kwargs directly is warning-free). Backends without native
json_object mode (anthropic, ollama) ignore json_mode.

No network: get_llm only constructs the model; only .invoke would hit the API.
Dummy keys satisfy construction-time key checks.
"""
import os
import unittest

# Dummy keys so construction doesn't require real credentials.
os.environ.setdefault("ZHIPU_API_KEY", "dummy")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
os.environ.setdefault("ANTHROPIC_API_KEY", "dummy")
os.environ.setdefault("DEEPSEEK_API_KEY", "dummy")
os.environ.setdefault("QWEN_API_KEY", "dummy")
os.environ.setdefault("KIMI_API_KEY", "dummy")

from utils.llm_factory import get_llm  # noqa: E402


def _has_json_mode(llm) -> bool:
    """True if the model is configured to emit JSON (response_format=json_object)."""
    mk = getattr(llm, "model_kwargs", None) or {}
    return mk.get("response_format") == {"type": "json_object"}


class TestJsonMode(unittest.TestCase):
    def test_zhipu_json_mode_on(self):
        # The active backend (glm-5.2). This is the user-facing case.
        llm = get_llm({"backend": "zhipu", "model": "glm-5.2"}, json_mode=True)
        self.assertTrue(_has_json_mode(llm), f"mk={llm.model_kwargs}")

    def test_zhipu_json_mode_off_is_default(self):
        llm = get_llm({"backend": "zhipu", "model": "glm-5.2"})  # default json_mode=False
        self.assertFalse(_has_json_mode(llm))

    def test_openai_json_mode_toggle(self):
        self.assertTrue(_has_json_mode(
            get_llm({"backend": "openai", "model": "gpt-4o-mini"}, json_mode=True)))
        self.assertFalse(_has_json_mode(
            get_llm({"backend": "openai", "model": "gpt-4o-mini"}, json_mode=False)))

    def test_deepseek_json_mode(self):
        llm = get_llm({"backend": "deepseek", "model": "deepseek-chat"}, json_mode=True)
        self.assertTrue(_has_json_mode(llm), f"mk={llm.model_kwargs}")

    def test_deepseek_reasoner_keeps_token_config_with_json_mode(self):
        # reasoner configures tokens via model_kwargs; json_mode must ADD
        # response_format WITHOUT disturbing that config. Compare against the
        # json_mode=False baseline for the same model so the assertion is robust
        # to langchain_openai reclassifying max_completion_tokens → max_tokens.
        base = get_llm({"backend": "deepseek", "model": "deepseek-reasoner"}, json_mode=False)
        jsonm = get_llm({"backend": "deepseek", "model": "deepseek-reasoner"}, json_mode=True)
        self.assertEqual(jsonm.model_kwargs.get("response_format"),
                         {"type": "json_object"})          # json_mode added it
        self.assertEqual(getattr(jsonm, "max_tokens", None),
                         getattr(base, "max_tokens", None))  # token config unchanged

    def test_anthropic_ignores_json_mode(self):
        # No native json_object mode → json_mode is a silent no-op (kept on the
        # prompt-contract + _safe_json fallback path).
        llm = get_llm({"backend": "anthropic", "model": "claude-3-5-haiku-20241022"},
                      json_mode=True)
        self.assertFalse(_has_json_mode(llm))

    def test_ollama_ignores_json_mode(self):
        llm = get_llm({"backend": "ollama", "model": "llama3"}, json_mode=True)
        self.assertFalse(_has_json_mode(llm))


if __name__ == "__main__":
    unittest.main()
