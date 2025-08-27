from pydantic import BaseModel, HttpUrl
from datetime import datetime
from enum import Enum
from typing import List, Optional

class ImportanceLevel(Enum):
    NEEDS_ATTENTION = "needs_attention"
    GOOD_TO_KNOW = "good_to_know"
    NO_ATTENTION_REQUIRED = "no_attention_required"
    
class NewsSource(Enum):
    NEWS_API = "news_api"
    WHITE_HOUSE = "white_house"

class NewsArticle(BaseModel):
    title: str
    description: Optional[str]
    content: Optional[str]
    url: HttpUrl
    published_at: datetime
    source: NewsSource
    keywords: List[str]
    importance_level: Optional[ImportanceLevel]
    relevance_score: Optional[float] = None
    summary: Optional[str] = None

class AnalyzedNews(BaseModel):
    article: NewsArticle
    total_count: int
    importance_breakdown: dict[ImportanceLevel, int]
    analysis_date: datetime