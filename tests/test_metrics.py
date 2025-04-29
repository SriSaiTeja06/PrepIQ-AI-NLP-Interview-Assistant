import pytest
from src.metrics import InterviewMetrics
import numpy as np

def test_technical_accuracy():
    """Test technical accuracy calculation."""
    evaluator = InterviewMetrics()
    
    # Test with relevant answer
    question = "Explain how a decision tree works."
    answer = "A decision tree is a supervised learning algorithm that splits data based on feature values..."
    expected_skills = ["machine learning", "algorithms", "data structures"]
    
    score = evaluator.calculate_technical_accuracy(question, answer, expected_skills)
    assert 0.0 <= score <= 1.0

def test_completeness():
    """Test completeness calculation."""
    evaluator = InterviewMetrics()
    
    # Test with complete answer
    question = "Explain how a decision tree works."
    answer = "A decision tree is a supervised learning algorithm that splits data based on feature values..."
    
    score = evaluator.calculate_completeness(question, answer)
    assert 0.0 <= score <= 1.0

def test_clarity():
    """Test clarity calculation."""
    evaluator = InterviewMetrics()
    
    # Test with clear answer
    answer = "A decision tree is a supervised learning algorithm that splits data based on feature values..."
    
    score = evaluator.calculate_clarity(answer)
    assert 0.0 <= score <= 1.0

def test_relevance():
    """Test relevance calculation."""
    evaluator = InterviewMetrics()
    
    # Test with relevant answer
    question = "Explain how a decision tree works."
    answer = "A decision tree is a supervised learning algorithm that splits data based on feature values..."
    
    score = evaluator.calculate_relevance(question, answer)
    assert 0.0 <= score <= 1.0

def test_evaluate_answer():
    """Test complete answer evaluation."""
    evaluator = InterviewMetrics()
    
    question = "Explain how a decision tree works."
    answer = "A decision tree is a supervised learning algorithm that splits data based on feature values..."
    expected_skills = ["machine learning", "algorithms", "data structures"]
    
    metrics = evaluator.evaluate_answer(question, answer, expected_skills)
    
    assert isinstance(metrics, dict)
    assert "technical_accuracy" in metrics
    assert "completeness" in metrics
    assert "clarity" in metrics
    assert "relevance" in metrics
    assert "overall_score" in metrics
    
    for score in metrics.values():
        assert 0.0 <= score <= 1.0

if __name__ == "__main__":
    pytest.main(["-v", __file__])
