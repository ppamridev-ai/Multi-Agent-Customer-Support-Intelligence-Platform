"""Central application configuration."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

TICKETS_RAW_PATH = RAW_DATA_DIR / "support_tickets_10k.csv"
FAQS_RAW_PATH = KNOWLEDGE_BASE_DIR / "faq_knowledge_base_150.csv"
TICKETS_PROCESSED_PATH = PROCESSED_DATA_DIR / "support_tickets_processed.csv"
FAQS_PROCESSED_PATH = PROCESSED_DATA_DIR / "faqs_processed.csv"

MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_PATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

CATEGORY_MODEL_PATH = MODELS_DIR / "category_classifier.joblib"
PRIORITY_MODEL_PATH = MODELS_DIR / "priority_classifier.joblib"
SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_classifier.joblib"

VALIDATION_METRICS_PATH = ARTIFACTS_DIR / "validation_metrics.json"

CHROMA_PERSIST_DIR = VECTOR_STORE_DIR / "chroma"
FAQ_COLLECTION_NAME = "customer_support_faqs"


class Settings(BaseSettings):
    app_name: str = "Multi-Agent Customer Support Platform"
    environment: str = "development"

    openai_api_key: str | None = None
    langsmith_api_key: str | None = None
    langsmith_project: str = "multi-agent-customer-support"
    langsmith_tracing: bool = False

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 3

    category_confidence_threshold: float = 0.70
    priority_confidence_threshold: float = 0.70
    sentiment_confidence_threshold: float = 0.70

    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()