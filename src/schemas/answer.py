"""
Answer schema for the AI-NLP Interview Assistant.
"""

from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class Answer:
    """
    Schema for interview answers.
    
    Attributes:
        id: Unique identifier
        question_id: ID of the associated question
        content: Actual answer text
        audio_source: Path to audio file if answer was recorded
        confidence: Confidence score for the answer (e.g., from STT)
        created_at: Timestamp when the answer was created
    """
    id: str
    question_id: str
    content: str
    audio_source: Optional[str] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
