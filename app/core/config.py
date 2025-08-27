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