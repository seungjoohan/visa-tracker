"""
News analysis service with RAG-augmented LLM classification.

=== THE RAG ANALYSIS PIPELINE ===

For each news article, we:
1. RETRIEVE: Search the vector store for policy chunks relevant to this article
2. CONSTRUCT: Build a prompt with the article + retrieved policy context
3. GENERATE: Send to LLM, get structured JSON response
4. PARSE: Validate JSON into an AnalysisResult (Pydantic model)
5. FALLBACK: If any step fails, use keyword+semantic classification instead

The fallback chain ensures the pipeline never breaks:
    LLM + RAG → LLM without RAG → Semantic + Keyword → Keyword only

=== CACHING ===

analyze_article() is called by both classify_importance() and generate_summary().
To avoid making two LLM calls per article, we cache results by URL. The cache
lives on the analyzer instance, which is created fresh per pipeline run.
"""

from typing import List, Dict, Optional
import re
import logging
from app.models.news import NewsArticle, ImportanceLevel
from app.models.llm import AnalysisResult
from app.core.config import settings
from app.services.semantic_classifier import SemanticClassifier
from app.services.llm_provider import LLMProvider
from app.services.retriever import Retriever
from app.services.llm_usage_logger import LLMUsageLogger

logger = logging.getLogger(__name__)


class NewsAnalyzer:
    def __init__(
        self,
        use_semantic: bool = True,
        llm_provider: Optional[LLMProvider] = None,
        retriever: Optional[Retriever] = None,
        usage_logger: Optional[LLMUsageLogger] = None,
    ):
        self.urgent_keywords = settings.urgent_keywords
        self.important_keywords = settings.important_keywords
        self.use_semantic = use_semantic

        # Existing semantic classifier (sentence-transformers)
        self.semantic_classifier = None
        if self.use_semantic:
            try:
                self.semantic_classifier = SemanticClassifier()
                self.semantic_classifier.load_model()
                logger.info("Semantic classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load semantic classifier: {e}. Falling back to keyword-only mode.")
                self.use_semantic = False

        # NEW: LLM + RAG components (injected via dependency injection)
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.usage_logger = usage_logger

        # Cache: URL → AnalysisResult to avoid double LLM calls per article
        # (classify_importance and generate_summary both call analyze_article)
        self._analysis_cache: dict[str, Optional[AnalysisResult]] = {}

        # Load prompt template from file
        self._classify_prompt = self._load_prompt("classify.txt")

        # 비자 관련 키워드들
        self.visa_keywords = [
            "visa", "h1b", "h-1b", "f1", "f-1", "green card", "citizenship",
            "immigration", "uscis", "dhs", "cbp", "ice", "work permit",
            "student visa", "tourist visa", "l1", "l-1", "o1", "o-1",
            "eb1", "eb2", "eb3", "eb5", "diversity visa", "naturalization", "permanent resident"
        ]

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template from config/prompts/."""
        prompt_path = settings.prompts_dir / filename
        if prompt_path.exists():
            return prompt_path.read_text()
        logger.warning(f"Prompt template not found: {prompt_path}")
        return ""

    # -------------------------------------------------------------------------
    # LLM-powered analysis (NEW)
    # -------------------------------------------------------------------------

    async def analyze_article(self, article: NewsArticle) -> Optional[AnalysisResult]:
        """
        Full RAG analysis pipeline for one article.

        Returns None if LLM is unavailable or any step fails.
        The caller should fall back to keyword+semantic classification.
        """
        if not self.llm_provider or not self._classify_prompt:
            return None

        # Check cache first (keyed by URL to avoid double calls)
        cache_key = str(article.url)
        if cache_key in self._analysis_cache:
            return self._analysis_cache[cache_key]

        llm_response = None
        try:
            # Step 1: RETRIEVE policy context from vector store
            policy_context = ""
            if self.retriever:
                query = f"{article.title} {article.description or ''}"
                policy_context = self.retriever.retrieve_and_format(query)

            # Step 2: CONSTRUCT prompt from template
            prompt = self._classify_prompt.format(
                policy_context=policy_context or "No policy context available.",
                title=article.title,
                source=article.source.value,
                published_at=article.published_at.isoformat(),
                content=article.content or article.description or "",
            )

            # Step 3: GENERATE — call the LLM
            system_msg = (
                "You are an immigration policy analyst. "
                "Always respond with valid JSON only, no markdown fences or commentary."
            )
            llm_response = await self.llm_provider.generate(
                prompt=prompt,
                system=system_msg,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )

            # Step 4: PARSE — extract JSON from LLM response
            text = llm_response.text.strip()
            # Strip markdown code fences if the LLM adds them despite instructions
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0].strip()

            result = AnalysisResult.model_validate_json(text)

            # Log success
            if self.usage_logger:
                self.usage_logger.log_call(llm_response, operation="classify")

            self._analysis_cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(
                f"LLM analysis failed for '{article.title[:50]}...': {e}. "
                f"Falling back to keyword+semantic."
            )
            if self.usage_logger and llm_response:
                self.usage_logger.log_call(
                    llm_response, operation="classify",
                    success=False, error=str(e),
                )
            self._analysis_cache[cache_key] = None
            return None

    # -------------------------------------------------------------------------
    # Classification (updated to try LLM first)
    # -------------------------------------------------------------------------

    async def classify_importance(self, article: NewsArticle) -> ImportanceLevel:
        """기사의 중요도 분류 — LLM first, then keyword + semantic fallback."""

        # Try LLM-based classification first
        if self.llm_provider:
            analysis = await self.analyze_article(article)
            if analysis:
                importance_map = {
                    "needs_attention": ImportanceLevel.NEEDS_ATTENTION,
                    "good_to_know": ImportanceLevel.GOOD_TO_KNOW,
                    "no_attention_required": ImportanceLevel.NO_ATTENTION_REQUIRED,
                }
                level = importance_map.get(analysis.importance)
                if level:
                    logger.info(
                        f"Article classified as {level.value} "
                        f"(LLM, confidence={analysis.confidence:.2f}): "
                        f"{article.title[:50]}..."
                    )
                    return level

        # Fallback: existing keyword + semantic logic
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
            logger.debug(f"Semantic scores: {scores}")
            max_score = scores[predicted_level]

            if max_score > 0.7:
                logger.info(f"Article classified as {predicted_level.value} (semantic, confidence={max_score:.2f}): {article.title[:50]}...")
                return predicted_level

            if max_score > 0.6:
                if predicted_level == ImportanceLevel.NEEDS_ATTENTION and (urgent_score > 0 or important_score > 0):
                    logger.info(f"Article classified as NEEDS_ATTENTION (semantic+keyword): {article.title[:50]}...")
                    return ImportanceLevel.NEEDS_ATTENTION
                if predicted_level == ImportanceLevel.GOOD_TO_KNOW and important_score > 0:
                    logger.info(f"Article classified as GOOD_TO_KNOW (semantic+keyword): {article.title[:50]}...")
                    return ImportanceLevel.GOOD_TO_KNOW

        # Fallback to keyword logic
        if important_score > 1:
            logger.info(f"Article classified as GOOD_TO_KNOW (keyword): {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW

        if title_visa_score > 0 and important_score > 0:
            logger.info(f"Article classified as GOOD_TO_KNOW (keyword): {article.title[:50]}...")
            return ImportanceLevel.GOOD_TO_KNOW

        logger.info(f"Article classified as NO_ATTENTION_REQUIRED: {article.title[:50]}...")
        return ImportanceLevel.NO_ATTENTION_REQUIRED

    # -------------------------------------------------------------------------
    # Summary generation (updated to try LLM first)
    # -------------------------------------------------------------------------

    async def generate_summary(self, article: NewsArticle) -> str:
        """기사 요약 생성 — LLM impact_summary first, then stub fallback."""

        # Try LLM-based summary (uses cached result from classify_importance)
        if self.llm_provider:
            analysis = await self.analyze_article(article)
            if analysis:
                return analysis.impact_summary

        # Fallback: existing stub (first 2 sentences, 150 char limit)
        content = article.description or article.content or ""
        if not content:
            return article.title

        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) >= 2:
            summary = f"{sentences[0]}. {sentences[1]}."
        elif sentences:
            summary = f"{sentences[0]}."
        else:
            summary = article.title

        if len(summary) > 150:
            summary = summary[:147] + "..."

        return summary

    # -------------------------------------------------------------------------
    # Relevance scoring (unchanged)
    # -------------------------------------------------------------------------

    async def calculate_relevance_score(self, article: NewsArticle) -> float:
        """기사의 비자 관련 점수 계산 (0.0 ~ 1.0) - keyword + semantic"""
        text = f"{article.title} {article.description or ''}".lower()

        visa_matches = self._count_keyword_matches(text, self.visa_keywords)
        visa_score = min(visa_matches * 0.3, 1.0)

        title_matches = self._count_keyword_matches(article.title.lower(), self.visa_keywords)
        title_score = min(title_matches * 0.5, 1.0)

        keyword_score = min(visa_score + title_score * 0.5, 1.0)

        if self.use_semantic and self.semantic_classifier:
            semantic_score = self.semantic_classifier.compute_relevance_score(text)
            final_score = (keyword_score * 0.4) + (semantic_score * 0.6)
            logger.debug(f"Relevance for '{article.title[:30]}...': keyword={keyword_score:.2f}, semantic={semantic_score:.2f}, final={final_score:.2f}")
            return final_score
        else:
            logger.debug(f"Relevance score for '{article.title[:30]}...': {keyword_score:.2f} (keyword-only)")
            return keyword_score

    # -------------------------------------------------------------------------
    # Helpers (unchanged)
    # -------------------------------------------------------------------------

    def _count_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """텍스트에서 키워드 매칭 개수 계산"""
        count = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, text):
                count += 1
        return count

    async def filter_relevant_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """관련성이 높은 기사만 필터링 (batch processing for efficiency)"""
        if not articles:
            return []

        if self.use_semantic and self.semantic_classifier:
            texts = [f"{a.title} {a.description or ''}".lower() for a in articles]
            semantic_scores = self.semantic_classifier.batch_compute_relevance(texts)

            relevant_articles = []
            for article, semantic_score in zip(articles, semantic_scores):
                text = f"{article.title} {article.description or ''}".lower()
                visa_matches = self._count_keyword_matches(text, self.visa_keywords)
                visa_score = min(visa_matches * 0.3, 1.0)
                title_matches = self._count_keyword_matches(article.title.lower(), self.visa_keywords)
                title_score = min(title_matches * 0.5, 1.0)
                keyword_score = min(visa_score + title_score * 0.5, 1.0)

                final_score = (keyword_score * 0.4) + (semantic_score * 0.6)
                article.relevance_score = final_score

                if final_score >= settings.relevance_threshold:
                    relevant_articles.append(article)
                    logger.debug(f"Article kept (relevance={final_score:.2f}): {article.title[:50]}...")
                else:
                    logger.debug(f"Article filtered out (relevance={final_score:.2f}): {article.title[:50]}...")
        else:
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
        """Batch classify importance for multiple articles."""
        if not articles:
            return []

        results = []

        if self.use_semantic and self.semantic_classifier:
            texts = [f"{a.title} {a.description or ''} {a.content or ''}".lower() for a in articles]
            semantic_results = self.semantic_classifier.batch_classify_urgency(texts)

            for article, (predicted_level, scores) in zip(articles, semantic_results):
                text = f"{article.title} {article.description or ''} {article.content or ''}".lower()

                urgent_score = self._count_keyword_matches(text, self.urgent_keywords)
                if urgent_score > 0:
                    results.append(ImportanceLevel.NEEDS_ATTENTION)
                    logger.info(f"Article classified as NEEDS_ATTENTION (keyword): {article.title[:50]}...")
                    continue

                important_score = self._count_keyword_matches(text, self.important_keywords)
                title_visa_score = self._count_keyword_matches(article.title.lower(), self.visa_keywords)

                max_score = scores[predicted_level]

                if max_score > 0.7:
                    results.append(predicted_level)
                    logger.info(f"Article classified as {predicted_level.value} (semantic, confidence={max_score:.2f}): {article.title[:50]}...")
                elif max_score > 0.6 and important_score > 0:
                    if predicted_level in [ImportanceLevel.NEEDS_ATTENTION, ImportanceLevel.GOOD_TO_KNOW]:
                        results.append(predicted_level)
                        logger.info(f"Article classified as {predicted_level.value} (semantic+keyword): {article.title[:50]}...")
                    else:
                        results.append(self._keyword_classify(important_score, title_visa_score, article))
                else:
                    results.append(self._keyword_classify(important_score, title_visa_score, article))
        else:
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
