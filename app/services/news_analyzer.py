from typing import List, Dict
import re
import logging
from app.models.news import NewsArticle, ImportanceLevel
from app.core.config import settings
from app.services.semantic_classifier import SemanticClassifier

logger = logging.getLogger(__name__)

class NewsAnalyzer:
    def __init__(self, use_semantic: bool = True):
        self.urgent_keywords = settings.urgent_keywords
        self.important_keywords = settings.important_keywords
        self.use_semantic = use_semantic

        # Initialize semantic classifier if enabled
        self.semantic_classifier = None
        if self.use_semantic:
            try:
                self.semantic_classifier = SemanticClassifier()
                self.semantic_classifier.load_model()
                logger.info("Semantic classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic classifier: {e}. Falling back to keyword-only mode.")
                self.use_semantic = False

        # 비자 관련 키워드들
        self.visa_keywords = [
            "visa", "h1b", "h-1b", "f1", "f-1", "green card", "citizenship",
            "immigration", "uscis", "dhs", "cbp", "ice", "work permit",
            "student visa", "tourist visa", "l1", "l-1", "o1", "o-1",
            "eb1", "eb2", "eb3", "eb5", "diversity visa", "naturalization", "permanent resident"
        ]
    
    async def classify_importance(self, article: NewsArticle) -> ImportanceLevel:
        """기사의 중요도 분류 (keyword + semantic)"""
        text = f"{article.title} {article.description or ''} {article.content or ''}".lower()

        # Method 1: Keyword-based classification (fast, precise)
        urgent_score = self._count_keyword_matches(text, self.urgent_keywords)
        if urgent_score > 0:
            logger.info(f"Article classified as NEEDS_ATTENTION (keyword): {article.title[:50]}...")
            return ImportanceLevel.NEEDS_ATTENTION

        important_score = self._count_keyword_matches(text, self.important_keywords)
        title_visa_score = self._count_keyword_matches(article.title.lower(), self.visa_keywords)

        # Method 2: Semantic classification (catches nuanced language)
        if self.use_semantic and self.semantic_classifier:
            predicted_level, scores = self.semantic_classifier.classify_urgency(text)

            # Log semantic scores for debugging
            logger.debug(f"Semantic scores: {scores}")

            # Use semantic classification if confidence is high
            max_score = scores[predicted_level]

            # High confidence threshold (>0.7 similarity)
            if max_score > 0.7:
                logger.info(f"Article classified as {predicted_level.value} (semantic, confidence={max_score:.2f}): {article.title[:50]}...")
                return predicted_level

            # Medium confidence: combine with keywords
            if max_score > 0.6:
                # If semantic suggests urgent/important AND we have some keyword matches
                if predicted_level == ImportanceLevel.NEEDS_ATTENTION and (urgent_score > 0 or important_score > 0):
                    logger.info(f"Article classified as NEEDS_ATTENTION (semantic+keyword): {article.title[:50]}...")
                    return ImportanceLevel.NEEDS_ATTENTION

                if predicted_level == ImportanceLevel.GOOD_TO_KNOW and important_score > 0:
                    logger.info(f"Article classified as GOOD_TO_KNOW (semantic+keyword): {article.title[:50]}...")
                    return ImportanceLevel.GOOD_TO_KNOW

        # Fallback to original keyword logic
        if important_score > 1:
            logger.info(f"Article classified as GOOD_TO_KNOW (keyword): {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW

        if title_visa_score > 0 and important_score > 0:
            logger.info(f"Article classified as GOOD_TO_KNOW (keyword): {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW

        logger.info(f"Article classified as NO_ATTENTION_REQUIRED: {article.title[:50]}...")
        return ImportanceLevel.NO_ATTENTION_REQUIRED
    
    async def calculate_relevance_score(self, article: NewsArticle) -> float:
        """기사의 비자 관련 점수 계산 (0.0 ~ 1.0) - keyword + semantic"""
        text = f"{article.title} {article.description or ''}".lower()

        # Method 1: Keyword-based score
        visa_matches = self._count_keyword_matches(text, self.visa_keywords)
        visa_score = min(visa_matches * 0.3, 1.0)

        title_matches = self._count_keyword_matches(article.title.lower(), self.visa_keywords)
        title_score = min(title_matches * 0.5, 1.0)

        keyword_score = min(visa_score + title_score * 0.5, 1.0)

        # Method 2: Semantic relevance score
        if self.use_semantic and self.semantic_classifier:
            semantic_score = self.semantic_classifier.compute_relevance_score(text)

            # Combine both scores (weighted average)
            # Give more weight to semantic if it's confident
            final_score = (keyword_score * 0.4) + (semantic_score * 0.6)

            logger.debug(f"Relevance for '{article.title[:30]}...': keyword={keyword_score:.2f}, semantic={semantic_score:.2f}, final={final_score:.2f}")
            return final_score
        else:
            # Fallback to keyword-only
            logger.debug(f"Relevance score for '{article.title[:30]}...': {keyword_score:.2f} (keyword-only)")
            return keyword_score
    
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
        """관련성이 높은 기사만 필터링 (batch processing for efficiency)"""
        if not articles:
            return []

        # Batch processing for semantic classifier
        if self.use_semantic and self.semantic_classifier:
            texts = [f"{a.title} {a.description or ''}".lower() for a in articles]
            semantic_scores = self.semantic_classifier.batch_compute_relevance(texts)

            relevant_articles = []
            for article, semantic_score in zip(articles, semantic_scores):
                # Calculate keyword score
                text = f"{article.title} {article.description or ''}".lower()
                visa_matches = self._count_keyword_matches(text, self.visa_keywords)
                visa_score = min(visa_matches * 0.3, 1.0)
                title_matches = self._count_keyword_matches(article.title.lower(), self.visa_keywords)
                title_score = min(title_matches * 0.5, 1.0)
                keyword_score = min(visa_score + title_score * 0.5, 1.0)

                # Combine scores
                final_score = (keyword_score * 0.4) + (semantic_score * 0.6)
                article.relevance_score = final_score

                if final_score >= settings.relevance_threshold:
                    relevant_articles.append(article)
                    logger.debug(f"Article kept (relevance={final_score:.2f}): {article.title[:50]}...")
                else:
                    logger.debug(f"Article filtered out (relevance={final_score:.2f}): {article.title[:50]}...")
        else:
            # Fallback to sequential keyword-only processing
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

    async def batch_classify_importance(self, articles: List[NewsArticle]) -> List[ImportanceLevel]:
        """
        Batch classify importance for multiple articles (more efficient)

        Args:
            articles: List of articles to classify

        Returns:
            List of ImportanceLevel predictions
        """
        if not articles:
            return []

        results = []

        # Use batch semantic classification if available
        if self.use_semantic and self.semantic_classifier:
            texts = [f"{a.title} {a.description or ''} {a.content or ''}".lower() for a in articles]
            semantic_results = self.semantic_classifier.batch_classify_urgency(texts)

            for article, (predicted_level, scores) in zip(articles, semantic_results):
                text = f"{article.title} {article.description or ''} {article.content or ''}".lower()

                # Check keywords first (highest priority)
                urgent_score = self._count_keyword_matches(text, self.urgent_keywords)
                if urgent_score > 0:
                    results.append(ImportanceLevel.NEEDS_ATTENTION)
                    logger.info(f"Article classified as NEEDS_ATTENTION (keyword): {article.title[:50]}...")
                    continue

                important_score = self._count_keyword_matches(text, self.important_keywords)
                title_visa_score = self._count_keyword_matches(article.title.lower(), self.visa_keywords)

                # Use semantic classification with confidence thresholds
                max_score = scores[predicted_level]

                if max_score > 0.7:
                    results.append(predicted_level)
                    logger.info(f"Article classified as {predicted_level.value} (semantic, confidence={max_score:.2f}): {article.title[:50]}...")
                elif max_score > 0.6 and important_score > 0:
                    if predicted_level in [ImportanceLevel.NEEDS_ATTENTION, ImportanceLevel.GOOD_TO_KNOW]:
                        results.append(predicted_level)
                        logger.info(f"Article classified as {predicted_level.value} (semantic+keyword): {article.title[:50]}...")
                    else:
                        # Fallback to keyword logic
                        results.append(self._keyword_classify(important_score, title_visa_score, article))
                else:
                    # Fallback to keyword logic
                    results.append(self._keyword_classify(important_score, title_visa_score, article))
        else:
            # Fallback to sequential processing
            for article in articles:
                level = await self.classify_importance(article)
                results.append(level)

        return results

    def _keyword_classify(self, important_score: int, title_visa_score: int, article: NewsArticle) -> ImportanceLevel:
        """Helper method for keyword-based classification"""
        if important_score > 1:
            logger.info(f"Article classified as GOOD_TO_KNOW (keyword): {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW

        if title_visa_score > 0 and important_score > 0:
            logger.info(f"Article classified as GOOD_TO_KNOW (keyword): {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW

        logger.info(f"Article classified as NO_ATTENTION_REQUIRED: {article.title[:50]}...")
        return ImportanceLevel.NO_ATTENTION_REQUIRED