# core/llm_manager.py
import os
from typing import Any

from langchain_openai import ChatOpenAI

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiLLM:
    """Custom wrapper for Google Gemini using google-generativeai SDK."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai not installed. Run: pip install google-generativeai")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not provided")

        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=self.api_key)

    def __call__(self, prompt: str, **kwargs) -> str:
        try:
            response = genai.GenerativeModel(self.model_name).generate_content(prompt)
            if hasattr(response, "text"):
                return response.text
            elif response.candidates:
                return response.candidates[0].content.parts[0].text
            return str(response)
        except Exception as e:
            return f"❌ Gemini call failed: {e}"

    def predict(self, text: str, **kwargs) -> str:
        return self.__call__(text, **kwargs)

    def invoke(self, input_data: Any, **kwargs) -> str:
        if hasattr(input_data, "messages"):
            content = "\n".join([m.content for m in input_data.messages if hasattr(m, "content")])
            return self.__call__(content, **kwargs)
        return self.__call__(input_data, **kwargs)


class LLMManager:
    """Factory to get the correct LLM based on backend."""

    @staticmethod
    def get_llm(backend: str, api_key: str, model_name: str):
        backend = backend.lower()

        if backend == "google gemini":
            # ✅ Force Gemini via API key (no Vertex AI / ADC)
            api_key = api_key or os.getenv("GEMINI_API_KEY")
            model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
            return GeminiLLM(api_key=api_key, model_name=model_name)

        elif backend == "ollama":
            # ✅ Use local Ollama via OpenAI-compatible wrapper
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key or "ollama",
                openai_api_base="http://localhost:11434/v1",
                temperature=0.7,
            )

        else:
            # ✅ Default: OpenAI-compatible (OpenAI, Groq, Anthropic, etc.)
            return ChatOpenAI(
                model=model_name,
                openai_api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
                openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
                temperature=0.7,
            )
