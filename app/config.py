"""
Configuration module for the Drug Repurposing Chat API.

All settings are loaded from environment variables (or a .env file).
Model names and API keys must never be hardcoded — use the variables
defined here and set values in your .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required:
        GEMINI_API_KEY: Google Gemini API key.

    Optional (shown with defaults):
        GEMINI_EMBEDDING_MODEL: Gemini embedding model ID.
        GEMINI_EMBEDDING_DIMENSION: Expected output dimension of the embedding model.
        GEMINI_CHAT_MODEL: Gemini generative model ID used for chat responses.
        CHROMA_DB_DIR: Filesystem path where ChromaDB persists its data.
        CHROMA_COLLECTION_NAME: Name of the ChromaDB collection.
        DOCS_DIR: Root directory containing per-drug PDF subfolders.
        DRUGS_CACHE_TTL_SECONDS: How long (seconds) the in-memory drug list is considered fresh.
        MAX_DRUGS_TO_LOAD: Cap on unique drugs scanned at startup (0 = unlimited).
        S3_BUCKET: Optional S3 bucket name for PDF storage/ingestion.
        S3_PREFIX: Key prefix inside the S3 bucket.
        S3_REGION: AWS region for the S3 bucket.
    """

    gemini_api_key: str
    gemini_embedding_model: str
    gemini_embedding_dimension: int
    gemini_chat_model: str
    chroma_db_dir: str = "./data/chroma"
    chroma_collection_name: str = "drug_docs"
    docs_dir: str = "./data/docs"
    drugs_cache_ttl_seconds: int = 300
    # 0 = unlimited; set >0 to cap startup scan and reduce cold-start latency
    max_drugs_to_load: int = 0
    s3_bucket: Optional[str] = None
    s3_prefix: str = ""
    s3_region: Optional[str] = None

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = None  # Let python-dotenv auto-detect file encoding


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
