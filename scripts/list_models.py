import google.generativeai as genai
from src.app.settings import get_settings

s = get_settings()
genai.configure(api_key=s.GOOGLE_API_KEY)

print("Available models (generateContent supported):")
for m in genai.list_models():
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print("-", m.name)
