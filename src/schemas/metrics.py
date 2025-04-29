"""
Evaluation metrics schema for the AI-NLP Interview Assistant.
"""

from dataclasses import dataclass

@dataclass
class EvaluationMetrics:
    """
    Schema for evaluation metrics of interview answers.
    
    Attributes:
        technical_accuracy: Score for technical correctness (0-1)
        completeness: Score for completeness of the answer (0-1)
        clarity: Score for clarity and structure (0-1)
        relevance: Score for relevance to the question (0-1)
        overall_score: Overall quality score (0-1)
    """
    technical_accuracy: float
    completeness: float
    clarity: float
    relevance: float
    overall_score: float
