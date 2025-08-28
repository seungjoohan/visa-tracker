from abc import ABC, abstractmethod
from typing import List
import aiohttp
from datetime import datetime, timedelta
from urllib.parse import quote
from app.models.news import NewsArticle, NewsSource
import logging

logger = logging.getLogger(__name__)

class NewsCollectorInterface(ABC):
    @abstractmethod
    async def collect_news(self, keywords: List[str]) -> List[NewsArticle]:
        pass

class NewsAPICollector(NewsCollectorInterface):
    def __init__(self, api_key: str, max_articles: int = 50):
        self.api_key = api_key
        self.max_articles = max_articles
        self.base_url = "https://newsapi.org/v2/everything"
    
    async def collect_news(self, keywords: List[str]) -> List[NewsArticle]:
        """NewsAPI에서 비자 관련 뉴스 수집"""
        try:
            # 키워드를 OR 조건으로 결합
            query = " OR ".join([f'"{keyword}"' for keyword in keywords])
            
            # 최근 2일간의 뉴스만 수집 (API 효율성 향상)
            from_date = datetime.now() - timedelta(days=2)
            
            params = {
                "q": query,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(self.max_articles, 100),  # API 제한
                "from": from_date.strftime("%Y-%m-%d"),
                "apiKey": self.api_key,
                "domains": "reuters.com,bloomberg.com,cnn.com,bbc.com,apnews.com,wsj.com,ft.com,politico.com,axios.com,thehill.com"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"NewsAPI request failed: {response.status}")
                        return []
                    
                    data = await response.json()
                    
                    if data.get("status") != "ok":
                        logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                        return []
                    
                    articles = []
                    for article_data in data.get("articles", []):
                        try:
                            # 필수 필드 검증
                            if not all([article_data.get("title"), article_data.get("url")]):
                                continue
                            
                            # 제거된 기사 필터링
                            if article_data.get("title") == "[Removed]":
                                continue
                            
                            article = NewsArticle(
                                title=article_data["title"],
                                description=article_data.get("description"),
                                content=article_data.get("content"),
                                url=article_data["url"],
                                published_at=datetime.fromisoformat(
                                    article_data["publishedAt"].replace("Z", "+00:00")
                                ),
                                source=NewsSource.NEWS_API,
                                keywords=self._extract_relevant_keywords(article_data, keywords),
                                importance_level=None  # 나중에 분석 단계에서 설정
                            )
                            articles.append(article)
                            
                        except Exception as e:
                            logger.warning(f"Failed to parse article: {e}")
                            continue
                    
                    logger.info(f"Collected {len(articles)} articles from NewsAPI")
                    return articles
                    
        except Exception as e:
            logger.error(f"Error collecting news from NewsAPI: {e}")
            return []
    
    def _extract_relevant_keywords(self, article_data: dict, search_keywords: List[str]) -> List[str]:
        """기사에서 관련 키워드 추출"""
        text = f"{article_data.get('title', '')} {article_data.get('description', '')} {article_data.get('content', '')}".lower()
        
        found_keywords = []
        for keyword in search_keywords:
            if keyword.lower() in text:
                found_keywords.append(keyword)
        
        return found_keywords

# WhiteHouse Collector는 일단 placeholder로 남겨둠 (나중에 구현)
class WhiteHouseCollector(NewsCollectorInterface):
    async def collect_news(self, keywords: List[str]) -> List[NewsArticle]:
        """WhiteHouse.gov RSS 수집 (미구현)"""
        logger.info("WhiteHouse collector not implemented yet")
        return []