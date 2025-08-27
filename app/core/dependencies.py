from functools import lru_cache
from fastapi import Depends
from typing import List

from app.core.config import settings, Settings
from app.services.news_collector import NewsAPICollector, WhiteHouseCollector, NewsCollectorInterface
from app.services.news_analyzer import NewsAnalyzer
from app.services.email_service import EmailService

@lru_cache()
def get_settings() -> Settings:
    return settings

def get_news_collectors() -> List[NewsCollectorInterface]:
    """뉴스 수집기들 의존성 주입"""
    collectors = []
    
    # NewsAPI collector (API 키가 있는 경우에만)
    if settings.newsapi_key:
        collectors.append(
            NewsAPICollector(
                api_key=settings.newsapi_key,
                max_articles=settings.max_articles_per_source
            )
        )
    
    # WhiteHouse collector (나중에 구현)
    # collectors.append(WhiteHouseCollector())
    
    return collectors

def get_news_analyzer() -> NewsAnalyzer:
    """뉴스 분석기 의존성 주입"""
    return NewsAnalyzer()

def get_email_service() -> EmailService:
    """이메일 서비스 의존성 주입"""
    return EmailService()