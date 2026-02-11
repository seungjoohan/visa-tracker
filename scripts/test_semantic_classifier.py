"""
Test script for semantic classifier
Run with: python scripts/test_semantic_classifier.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.semantic_classifier import SemanticClassifier
from app.models.news import ImportanceLevel

def test_semantic_classifier():
    print("=" * 80)
    print("Testing Semantic Classifier")
    print("=" * 80)

    classifier = SemanticClassifier()
    print("\n[1/3] Loading model...")
    classifier.load_model()
    print("✓ Model loaded successfully\n")

    # Test cases
    test_articles = [
        {
            "title": "USCIS Suspends All H1B Processing Immediately",
            "description": "Immigration authorities announced emergency suspension of all H1B visa processing effective today",
            "expected": ImportanceLevel.NEEDS_ATTENTION
        },
        {
            "title": "New H1B Lottery System Announced for 2026",
            "description": "USCIS releases updated guidelines for H1B lottery process starting next fiscal year",
            "expected": ImportanceLevel.GOOD_TO_KNOW
        },
        {
            "title": "Immigration Official Appointed to Advisory Board",
            "description": "Government announces new appointment to immigration policy advisory committee",
            "expected": ImportanceLevel.NO_ATTENTION_REQUIRED
        },
        {
            "title": "Visa Application Deadline This Week for Current Holders",
            "description": "Renewal applications must be submitted by Friday to avoid expiration",
            "expected": ImportanceLevel.NEEDS_ATTENTION
        },
        {
            "title": "Study Shows Immigration Trends Over Past Decade",
            "description": "Academic research examines patterns in visa applications and approvals from 2015-2025",
            "expected": ImportanceLevel.NO_ATTENTION_REQUIRED
        }
    ]

    print("=" * 80)
    print("[2/3] Testing Urgency Classification")
    print("=" * 80)

    correct = 0
    for i, test in enumerate(test_articles, 1):
        text = f"{test['title']} {test['description']}"
        predicted_level, scores = classifier.classify_urgency(text)

        match = "✓" if predicted_level == test['expected'] else "✗"
        correct += (predicted_level == test['expected'])

        print(f"\n{match} Test {i}:")
        print(f"   Title: {test['title']}")
        print(f"   Expected: {test['expected'].value}")
        print(f"   Predicted: {predicted_level.value}")
        print(f"   Confidence scores:")
        for level, score in scores.items():
            print(f"     - {level.value}: {score:.3f}")

    accuracy = (correct / len(test_articles)) * 100
    print(f"\n{'=' * 80}")
    print(f"Accuracy: {correct}/{len(test_articles)} ({accuracy:.1f}%)")
    print(f"{'=' * 80}\n")

    # Test relevance scoring
    print("=" * 80)
    print("[3/3] Testing Relevance Scoring")
    print("=" * 80)

    relevance_tests = [
        {
            "text": "H1B visa holders face new application requirements",
            "expected_high": True
        },
        {
            "text": "Stock market reaches new all-time high today",
            "expected_high": False
        },
        {
            "text": "Green card processing times updated by immigration officials",
            "expected_high": True
        },
        {
            "text": "Local restaurant wins best pizza award",
            "expected_high": False
        }
    ]

    for i, test in enumerate(relevance_tests, 1):
        score = classifier.compute_relevance_score(test['text'])
        is_high = score > 0.5
        match = "✓" if is_high == test['expected_high'] else "✗"

        print(f"\n{match} Test {i}:")
        print(f"   Text: {test['text']}")
        print(f"   Relevance Score: {score:.3f}")
        print(f"   Expected: {'High' if test['expected_high'] else 'Low'}")
        print(f"   Got: {'High' if is_high else 'Low'}")

    print("\n" + "=" * 80)
    print("Testing Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_semantic_classifier()
