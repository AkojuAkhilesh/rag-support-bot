from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # === API Keys / Providers ===
    GOOGLE_API_KEY: str = "AIzaSyDtoNMQE8YGyZG55U7a1E_-OL_LJ7Foscs"
    LLM_PROVIDER: str = "gemini"
    EMBEDDINGS_PROVIDER: str = "gemini"     # or "gemini"
    VECTOR_DB: str = "chroma"

    # === RAG Configuration ===
    INDEX_NAME: str = "support-knowledge"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    TOP_K: int = 6

    # === Model Names ===
    GEMINI_MODEL: str = "gemini-1.5-flash"
    GEMINI_EMBED_MODEL: str = "text-embedding-004"

    # === Load from .env ===
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached settings object so you can use:
        settings = get_settings()
    anywhere in your code.
    """
    return Settings()
