try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class GeminiLLM:
    """Custom LLM wrapper for Google Gemini using genai SDK."""
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model_name = model_name.replace("gemini/", "")
        self.temperature = 0.7
        self.max_tokens = 2048
        self.client = genai.Client(api_key=api_key) if GENAI_AVAILABLE else None

    def __call__(self, prompt: str, **kwargs) -> str:
        if not self.client:
            return "Error: Gemini client not available"
        resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
        if hasattr(resp, 'text'):
            return resp.text
        if hasattr(resp, 'candidates') and resp.candidates:
            return resp.candidates[0].content.parts[0].text
        return str(resp)

    def invoke(self, input_data, **kwargs):
        return self.__call__(input_data, **kwargs)

    def predict(self, text: str, **kwargs) -> str:
        return self.__call__(text, **kwargs)

    def generate(self, prompts, **kwargs):
        if isinstance(prompts, list):
            return [self.__call__(p, **kwargs) for p in prompts]
        return self.__call__(prompts, **kwargs)
