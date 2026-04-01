from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # Core settings
    APP_NAME: str = "Production-Level RAG System"
    DEBUG: bool = False

    # OpenRouter configuration
    OPENROUTER_API_KEY: str

    # Chunking configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Retrieval configuration
    VECTOR_TOP_K: int = 5
    KEYWORD_TOP_K: int = 5
    FINAL_TOP_K: int = 5

    # Storage paths
    DATA_DIR: str = "./data"
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"
    METADATA_STORE_PATH: str = "./data/metadata.json"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
