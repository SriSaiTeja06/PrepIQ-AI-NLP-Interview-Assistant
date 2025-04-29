"""
Question schema for the AI-NLP Interview Assistant.
"""

from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class Question:
    """
    Schema for interview questions.
    
    Attributes:
        id: Unique identifier
        role: Job role the question is for
        type: Type of question (technical, behavioral)
        difficulty: Difficulty level (easy, medium, hard)
        content: Actual question text
        expected_skills: List of skills expected to be demonstrated
        created_at: Timestamp when the question was created
    """
    id: str
    role: str
    type: str
    difficulty: str
    content: str
    expected_skills: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
