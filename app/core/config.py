from pydantic_settings import BaseSettings
from pydantic import Field, computed_field
from typing import List
import json
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 먼저 로드
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    newsapi_key: str = Field(..., env="NEWSAPI_KEY")
    
    # Email Settings
    smtp_server: str = Field(default="smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    email_user: str = Field(..., env="EMAIL_USER")
    email_password: str = Field(..., env="EMAIL_PASSWORD")
    # 이메일 수신자를 computed_field로 처리
    @computed_field
    @property
    def recipients(self) -> List[str]:
        recipients_env = os.getenv('EMAIL_RECIPIENTS', '')
        
        if not recipients_env.strip():
            return []
        
        # [email1, email2] 형식 처리
        if recipients_env.strip().startswith('[') and recipients_env.strip().endswith(']'):
            try:
                # JSON 형식 시도
                return json.loads(recipients_env)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 수동 파싱
                cleaned = recipients_env.strip().strip('[]')
                emails = [email.strip().strip('"').strip("'") for email in cleaned.split(',') if email.strip()]
                return [email for email in emails if email]
        else:
            # 콤마로 구분된 문자열
            return [email.strip() for email in recipients_env.split(',') if email.strip()]
    
    # News Settings
    max_articles_per_source: int = Field(default=50)
    relevance_threshold: float = Field(default=0.5)
    
    # Keywords for classification
    urgent_keywords: List[str] = Field(default=[
        "ban", "suspended", "terminated", "denied", "rejected", "cancelled",
        "emergency", "immediate", "deadline", "expires", "closing", "final"
    ])
    
    important_keywords: List[str] = Field(default=[
        "new policy", "change", "update", "requirement", "process", "application",
        "eligibility", "rule", "regulation", "law", "bill", "announcement"
    ])
    
    # Vector Store / RAG Settings
    # embedding_model: Which sentence-transformer model to use for creating embeddings.
    #   Must match the model used during ingestion — if you change this, re-ingest everything.
    #   Benchmarked against MiniLM, mpnet, and bge-large — bge-base had the best
    #   balance of score magnitude (0.80 avg) and search speed (30ms).
    embedding_model: str = Field(default="BAAI/bge-base-en-v1.5")
    # embedding_dimension: The output vector size of the embedding model.
    #   BAAI/bge-base-en-v1.5 = 768, all-MiniLM-L6-v2 = 384, bge-large = 1024.
    embedding_dimension: int = Field(default=768)
    # retrieval_top_k: How many chunks to retrieve from the vector store per query.
    #   More = more context for the LLM but also more noise and higher token cost.
    retrieval_top_k: int = Field(default=5)

    # LLM Settings
    # anthropic_api_key: Get from console.anthropic.com. Empty string = LLM disabled.
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    # llm_model: Which Claude model to use. Haiku = cheapest + fastest.
    llm_model: str = Field(default="claude-3-5-haiku-20241022", env="LLM_MODEL")
    # llm_temperature: 0.0 = deterministic output (best for classification).
    llm_temperature: float = Field(default=0.0)
    # llm_max_tokens: Max output tokens. 1024 is plenty for structured JSON.
    llm_max_tokens: int = Field(default=1024)
    # use_llm: Master kill switch. False = skip LLM, use keyword+semantic only.
    use_llm: bool = Field(default=True, env="USE_LLM")

    # File paths
    @property
    def base_dir(self) -> Path:
        return Path(__file__).parent.parent.parent

    @property
    def config_dir(self) -> Path:
        return self.base_dir / "config"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def knowledge_base_dir(self) -> Path:
        return self.base_dir / "knowledge_base"

    @property
    def index_dir(self) -> Path:
        return self.knowledge_base_dir / "index"

    @property
    def prompts_dir(self) -> Path:
        return self.config_dir / "prompts"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 추가 필드 무시

# Global settings instance
settings = Settings()

if __name__ == "__main__":
    print(settings.base_dir)
    print(settings.config_dir)
    print(settings.logs_dir)
    print(settings.recipients)