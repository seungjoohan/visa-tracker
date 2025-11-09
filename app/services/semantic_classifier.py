from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Tuple
import logging
from app.models.news import ImportanceLevel

logger = logging.getLogger(__name__)

class SemanticClassifier:
    """
    Semantic classifier using sentence-transformers for:
    1. Relevance filtering (is this truly about visas?)
    2. Urgency classification (how important is this?)
    """

    # Example sentences for each importance level
    URGENCY_EXAMPLES = {
        ImportanceLevel.NEEDS_ATTENTION: [
            "Immigration authorities suspend all H1B visa processing immediately due to policy changes",
            "Federal government announces immediate ban on certain visa categories",
            "USCIS deadline approaching for visa renewal applications this week",
            "Emergency executive order terminates work permit program",
            "Visa denials surge as new restrictions take effect today",
            "Courts block immigration policy creating urgent situation for visa holders",
            "Breaking: All student visa interviews cancelled effective immediately",
            "Government announces visa application deadline expires this month"
        ],
        ImportanceLevel.GOOD_TO_KNOW: [
            "New H1B lottery system to be implemented next year",
            "USCIS updates visa application requirements and procedures",
            "Immigration officials announce changes to green card processing times",
            "Congress proposes bill to reform visa allocation system",
            "Policy changes may affect future visa applicants",
            "Updated guidelines released for visa interview preparation",
            "New visa fee structure announced for upcoming fiscal year",
            "USCIS publishes annual immigration statistics and trends"
        ],
        ImportanceLevel.NO_ATTENTION_REQUIRED: [
            "Historical overview of immigration policy in the United States",
            "General statistics about immigration patterns published",
            "Opinion piece discussing immigration system challenges",
            "Profile of immigration official appointed to new position",
            "Academic study examines long-term immigration trends",
            "Immigration office announces routine schedule changes",
            "General news about immigration debate in Congress",
            "Background article on visa system history"
        ]
    }

    # Example sentences for relevance checking
    RELEVANCE_EXAMPLES = {
        "highly_relevant": [
            "H1B visa lottery results announced for this year",
            "Green card processing times updated by USCIS",
            "F1 student visa interview requirements changed",
            "Work permit applications face new restrictions",
            "L1 visa holders affected by policy update",
            "Citizenship application process streamlined",
            "EB2 visa category sees processing improvements",
            "Tourist visa appointments availability increased"
        ],
        "not_relevant": [
            "Stock market reaches new high today",
            "Sports team wins championship game",
            "Weather forecast predicts rain this weekend",
            "Technology company launches new product",
            "Movie breaks box office records",
            "Restaurant opens new location downtown",
            "Celebrity announces engagement news",
            "Local school receives academic award"
        ]
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic classifier

        Args:
            model_name: Sentence-transformers model
                - 'all-MiniLM-L6-v2': Fast, lightweight (80MB), good quality
                - 'all-mpnet-base-v2': Best quality (420MB), slower
        """
        self.model_name = model_name
        self.model = None
        self.urgency_embeddings = {}
        self.relevance_embeddings = {}

    def load_model(self):
        """Load model and pre-compute example embeddings"""
        if self.model is not None:
            return  # Already loaded

        logger.info(f"Loading sentence-transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        # Pre-compute urgency example embeddings
        for level, examples in self.URGENCY_EXAMPLES.items():
            embeddings = self.model.encode(examples, convert_to_numpy=True)
            # Store average embedding for each importance level
            self.urgency_embeddings[level] = np.mean(embeddings, axis=0)

        # Pre-compute relevance example embeddings
        for category, examples in self.RELEVANCE_EXAMPLES.items():
            embeddings = self.model.encode(examples, convert_to_numpy=True)
            self.relevance_embeddings[category] = np.mean(embeddings, axis=0)

        logger.info("Semantic classifier initialized successfully")

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return float(np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        ))

    def compute_relevance_score(self, text: str) -> float:
        """
        Compute semantic relevance score (0.0 to 1.0)

        Args:
            text: Article text to evaluate

        Returns:
            Relevance score (higher = more relevant to visa/immigration)
        """
        if self.model is None:
            self.load_model()

        # Encode the text
        text_embedding = self.model.encode([text], convert_to_numpy=True)[0]

        # Compare to relevant vs not relevant examples
        relevant_similarity = self._cosine_similarity(
            text_embedding,
            self.relevance_embeddings["highly_relevant"]
        )

        not_relevant_similarity = self._cosine_similarity(
            text_embedding,
            self.relevance_embeddings["not_relevant"]
        )

        # Normalize to 0-1 scale
        # If more similar to relevant examples, score is high
        score = (relevant_similarity - not_relevant_similarity + 1) / 2
        return max(0.0, min(1.0, score))

    def classify_urgency(self, text: str) -> Tuple[ImportanceLevel, Dict[ImportanceLevel, float]]:
        """
        Classify urgency level using semantic similarity

        Args:
            text: Article text to classify

        Returns:
            Tuple of (predicted_level, similarity_scores_dict)
        """
        if self.model is None:
            self.load_model()

        # Encode the text
        text_embedding = self.model.encode([text], convert_to_numpy=True)[0]

        # Compute similarity to each importance level
        scores = {}
        for level, level_embedding in self.urgency_embeddings.items():
            similarity = self._cosine_similarity(text_embedding, level_embedding)
            scores[level] = similarity

        # Get the level with highest similarity
        predicted_level = max(scores, key=scores.get)

        return predicted_level, scores

    def batch_classify_urgency(self, texts: List[str]) -> List[Tuple[ImportanceLevel, Dict[ImportanceLevel, float]]]:
        """
        Classify urgency for multiple texts efficiently

        Args:
            texts: List of article texts

        Returns:
            List of (predicted_level, scores) tuples
        """
        if self.model is None:
            self.load_model()

        if not texts:
            return []

        logger.info(f"Batch classifying {len(texts)} articles")

        # Batch encode all texts (much faster than one-by-one)
        text_embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)

        results = []
        for text_embedding in text_embeddings:
            # Compute similarity to each importance level
            scores = {}
            for level, level_embedding in self.urgency_embeddings.items():
                similarity = self._cosine_similarity(text_embedding, level_embedding)
                scores[level] = similarity

            # Get the level with highest similarity
            predicted_level = max(scores, key=scores.get)
            results.append((predicted_level, scores))

        return results

    def batch_compute_relevance(self, texts: List[str]) -> List[float]:
        """
        Compute relevance scores for multiple texts efficiently

        Args:
            texts: List of article texts

        Returns:
            List of relevance scores
        """
        if self.model is None:
            self.load_model()

        if not texts:
            return []

        logger.info(f"Batch computing relevance for {len(texts)} articles")

        # Batch encode all texts
        text_embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)

        scores = []
        for text_embedding in text_embeddings:
            relevant_similarity = self._cosine_similarity(
                text_embedding,
                self.relevance_embeddings["highly_relevant"]
            )

            not_relevant_similarity = self._cosine_similarity(
                text_embedding,
                self.relevance_embeddings["not_relevant"]
            )

            # Normalize to 0-1 scale
            score = (relevant_similarity - not_relevant_similarity + 1) / 2
            scores.append(max(0.0, min(1.0, score)))

        return scores
