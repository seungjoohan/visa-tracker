import logging
from typing import List, Set
from datetime import datetime

from app.core.dependencies import get_news_collectors, get_news_analyzer, get_email_service
from app.models.news import NewsArticle, ImportanceLevel
from app.core.config import settings
from app.services.pipeline_logger import PipelineStageLogger

logger = logging.getLogger(__name__)

async def run_daily_process():
    """일일 뉴스 처리 파이프라인"""
    try:
        logger.info("Starting daily news processing...")

        pipeline_logger = PipelineStageLogger(settings.logs_dir)

        # 의존성 가져오기
        collectors = get_news_collectors()
        analyzer = get_news_analyzer()
        email_service = get_email_service()

        if not collectors:
            logger.error("No news collectors available - check API keys")
            return False

        keywords = ["visa", "immigration", "h1b", "green card", "citizenship", "f1", "f-1", "work permit"]

        # 1. 뉴스 수집
        with pipeline_logger.log_stage("collect", {"sources": len(collectors)}):
            all_articles = []
            for collector in collectors:
                try:
                    articles = await collector.collect_news(keywords)
                    all_articles.extend(articles)
                    logger.info(f"Collected {len(articles)} articles from {collector.__class__.__name__}")
                except Exception as e:
                    logger.error(f"Error collecting from {collector.__class__.__name__}: {e}")
                    continue
            pipeline_logger.update_stage({"article_count": len(all_articles)})

        if not all_articles:
            logger.warning("No articles collected")
            return False

        # 2. 중복 제거
        with pipeline_logger.log_stage("deduplicate", {"before_count": len(all_articles)}):
            unique_articles = remove_duplicates(all_articles)
            pipeline_logger.update_stage({"after_count": len(unique_articles)})

        # 3. 관련성 필터링
        with pipeline_logger.log_stage("filter", {"before_count": len(unique_articles)}):
            relevant_articles = await analyzer.filter_relevant_articles(unique_articles)
            pipeline_logger.update_stage({"after_count": len(relevant_articles)})

        if not relevant_articles:
            logger.warning("No relevant articles found")
            return False

        # 4. 분석 및 분류
        with pipeline_logger.log_stage("analyze", {"article_count": len(relevant_articles)}):
            analyzed_articles = []
            llm_count = 0
            fallback_count = 0

            for article in relevant_articles:
                try:
                    # Try full LLM+RAG analysis first (result is cached internally)
                    analysis = await analyzer.analyze_article(article)

                    if analysis:
                        # Use LLM-derived results
                        importance_map = {
                            "needs_attention": ImportanceLevel.NEEDS_ATTENTION,
                            "good_to_know": ImportanceLevel.GOOD_TO_KNOW,
                            "no_attention_required": ImportanceLevel.NO_ATTENTION_REQUIRED,
                        }
                        article.importance_level = importance_map.get(
                            analysis.importance, ImportanceLevel.NO_ATTENTION_REQUIRED
                        )
                        article.summary = analysis.impact_summary
                        article.analysis_result = analysis
                        llm_count += 1
                    else:
                        # Fallback to keyword+semantic classification
                        article.importance_level = await analyzer.classify_importance(article)
                        article.summary = await analyzer.generate_summary(article)
                        fallback_count += 1

                    analyzed_articles.append(article)

                except Exception as e:
                    logger.error(f"Error analyzing article '{article.title[:50]}...': {e}")
                    continue

            pipeline_logger.update_stage({
                "analyzed_count": len(analyzed_articles),
                "llm_count": llm_count,
                "fallback_count": fallback_count,
            })

        logger.info(f"Successfully analyzed {len(analyzed_articles)} articles")

        # 5. 이메일 전송
        with pipeline_logger.log_stage("email", {"article_count": len(analyzed_articles)}):
            email_success = await email_service.send_daily_summary(analyzed_articles)
            pipeline_logger.update_stage({"success": email_success})

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
