from typing import List, Dict
import re
import logging
from app.models.news import NewsArticle, ImportanceLevel
from app.core.config import settings

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self):
        self.urgent_keywords = settings.urgent_keywords
        self.important_keywords = settings.important_keywords
        
        # 비자 관련 키워드들
        self.visa_keywords = [
            "visa", "h1b", "h-1b", "f1", "f-1", "green card", "citizenship", 
            "immigration", "uscis", "dhs", "cbp", "ice", "work permit", 
            "student visa", "tourist visa", "l1", "l-1", "o1", "o-1",
            "eb1", "eb2", "eb3", "eb5", "diversity visa", "naturalization", "permanent resident"
        ]
    
    async def classify_importance(self, article: NewsArticle) -> ImportanceLevel:
        """기사의 중요도 분류"""
        text = f"{article.title} {article.description or ''} {article.content or ''}".lower()
        
        # 긴급 키워드 체크
        urgent_score = self._count_keyword_matches(text, self.urgent_keywords)
        if urgent_score > 0:
            logger.info(f"Article classified as NEEDS_ATTENTION: {article.title[:50]}...")
            return ImportanceLevel.NEEDS_ATTENTION
        
        # 중요 키워드 체크
        important_score = self._count_keyword_matches(text, self.important_keywords)
        if important_score > 1:  # 2개 이상의 중요 키워드
            logger.info(f"Article classified as GOOD_TO_KNOW: {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW
        
        # 제목에 비자 키워드가 있으면 좋은 정보
        title_visa_score = self._count_keyword_matches(article.title.lower(), self.visa_keywords)
        if title_visa_score > 0 and important_score > 0:
            logger.info(f"Article classified as GOOD_TO_KNOW: {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW
        
        logger.info(f"Article classified as NO_ATTENTION_REQUIRED: {article.title[:50]}...")
        return ImportanceLevel.NO_ATTENTION_REQUIRED
    
    async def calculate_relevance_score(self, article: NewsArticle) -> float:
        """기사의 비자 관련 점수 계산 (0.0 ~ 1.0)"""
        text = f"{article.title} {article.description or ''}".lower()
        
        # 비자 키워드 매칭 점수
        visa_matches = self._count_keyword_matches(text, self.visa_keywords)
        visa_score = min(visa_matches * 0.3, 1.0)
        
        # 제목에서의 매칭에 가중치
        title_matches = self._count_keyword_matches(article.title.lower(), self.visa_keywords)
        title_score = min(title_matches * 0.5, 1.0)
        
        # 최종 점수 계산
        final_score = min(visa_score + title_score * 0.5, 1.0)
        
        logger.debug(f"Relevance score for '{article.title[:30]}...': {final_score:.2f}")
        return final_score
    
    async def generate_summary(self, article: NewsArticle) -> str:
        """기사 요약 생성 (간단한 버전)"""
        # 현재는 간단한 요약 (첫 2문장)
        content = article.description or article.content or ""
        if not content:
            return article.title
        
        # 문장 분리 (간단한 버전)
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # 첫 2문장 또는 150자 제한
        if len(sentences) >= 2:
            summary = f"{sentences[0]}. {sentences[1]}."
        elif sentences:
            summary = f"{sentences[0]}."
        else:
            summary = article.title
        
        # 길이 제한
        if len(summary) > 150:
            summary = summary[:147] + "..."
        
        return summary
    
    def _count_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """텍스트에서 키워드 매칭 개수 계산"""
        count = 0
        for keyword in keywords:
            # 단어 경계를 고려한 매칭
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                count += 1
        return count
    
    async def filter_relevant_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """관련성이 높은 기사만 필터링"""
        relevant_articles = []
        
        for article in articles:
            relevance_score = await self.calculate_relevance_score(article)
            article.relevance_score = relevance_score
            
            if relevance_score >= settings.relevance_threshold:
                relevant_articles.append(article)
            else:
                logger.debug(f"Article filtered out (low relevance): {article.title[:50]}...")
        
        logger.info(f"Filtered {len(relevant_articles)} relevant articles from {len(articles)} total")
        return relevant_articles