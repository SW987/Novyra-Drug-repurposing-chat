from pydantic import BaseModel
from pydantic_settings import BaseSettings
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    gemini_api_key: str
    gemini_embedding_model: str = "models/embedding-001"
    gemini_chat_model: str = "models/gemini-2.0-flash-exp"
    chroma_db_dir: str = "./data/chroma"
    chroma_collection_name: str = "drug_docs"
    docs_dir: str = r"C:\Users\saadw\Downloads\repurposing research papers for 3 drugs"

    class Config:
        env_file = ".env"
        case_sensitive = False
        env_file_encoding = None # Allow python-dotenv to auto-detect encoding


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
