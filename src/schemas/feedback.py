"""
Feedback schema for the AI-NLP Interview Assistant.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from .metrics import EvaluationMetrics

@dataclass
class Feedback:
    """
    Schema for feedback on interview answers.
    
    Attributes:
        id: Unique identifier
        answer_id: ID of the associated answer
        content: Feedback content text
        metrics: Evaluation metrics used to generate the feedback
        created_at: Timestamp when the feedback was created
    """
    id: str
    answer_id: str
    content: str
    metrics: EvaluationMetrics
    created_at: datetime = field(default_factory=datetime.utcnow)
