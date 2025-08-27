import logging
from typing import List, Set
from datetime import datetime

from app.core.dependencies import get_news_collectors, get_news_analyzer, get_email_service
from app.models.news import NewsArticle
from app.core.config import settings

logger = logging.getLogger(__name__)

async def run_daily_process():
    """일일 뉴스 처리 파이프라인"""
    try:
        logger.info("Starting daily news processing...")
        
        # 의존성 가져오기
        collectors = get_news_collectors()
        analyzer = get_news_analyzer()
        email_service = get_email_service()
        
        if not collectors:
            logger.error("No news collectors available - check API keys")
            return False
        
        keywords = ["visa", "immigration", "h1b", "green card", "citizenship", "f1", "f-1", "work permit"]
        
        # 1. 뉴스 수집
        logger.info(f"Collecting news from {len(collectors)} sources...")
        all_articles = []
        for collector in collectors:
            try:
                articles = await collector.collect_news(keywords)
                all_articles.extend(articles)
                logger.info(f"Collected {len(articles)} articles from {collector.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error collecting from {collector.__class__.__name__}: {e}")
                continue
        
        if not all_articles:
            logger.warning("No articles collected")
            return False
        
        # 2. 중복 제거
        logger.info("Removing duplicates...")
        unique_articles = remove_duplicates(all_articles)
        logger.info(f"After duplicate removal: {len(unique_articles)} articles")
        
        # 3. 관련성 필터링
        logger.info("Filtering relevant articles...")
        relevant_articles = await analyzer.filter_relevant_articles(unique_articles)
        
        if not relevant_articles:
            logger.warning("No relevant articles found")
            return False
        
        # 4. 분석 및 분류
        logger.info("Analyzing and classifying articles...")
        analyzed_articles = []
        for article in relevant_articles:
            try:
                # 중요도 분류
                article.importance_level = await analyzer.classify_importance(article)
                
                # 요약 생성
                article.summary = await analyzer.generate_summary(article)
                
                analyzed_articles.append(article)
                
            except Exception as e:
                logger.error(f"Error analyzing article '{article.title[:50]}...': {e}")
                continue
        
        logger.info(f"Successfully analyzed {len(analyzed_articles)} articles")
        
        # 5. 이메일 전송
        logger.info("Sending daily summary email...")
        email_success = await email_service.send_daily_summary(analyzed_articles)
        
        if email_success:
            logger.info("Daily processing completed successfully")
            return True
        else:
            logger.error("Failed to send email")
            return False
            
    except Exception as e:
        logger.error(f"Error in daily processing: {e}")
        return False

def remove_duplicates(articles: List[NewsArticle]) -> List[NewsArticle]:
    """URL 기반으로 중복 기사 제거"""
    seen_urls: Set[str] = set()
    unique_articles = []
    
    for article in articles:
        url_str = str(article.url)
        if url_str not in seen_urls:
            seen_urls.add(url_str)
            unique_articles.append(article)
        else:
            logger.debug(f"Duplicate article removed: {article.title[:50]}...")
    
    return unique_articles