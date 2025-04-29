from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class Question(BaseModel):
    """
    Schema for interview questions.
    """
    id: str = Field(..., description="Unique identifier for the question")
    role: str = Field(..., description="Job role associated with the question")
    type: str = Field(..., description="Type of question (technical/behavioral)")
    difficulty: str = Field(..., description="Difficulty level (easy/medium/hard)")
    content: str = Field(..., description="The actual question text")
    expected_skills: List[str] = Field(..., description="Skills being tested")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Answer(BaseModel):
    """
    Schema for candidate answers.
    """
    id: str = Field(..., description="Unique identifier for the answer")
    question_id: str = Field(..., description="ID of the associated question")
    content: str = Field(..., description="The actual answer text")
    audio_source: Optional[str] = Field(None, description="Path to audio recording if applicable")
    confidence: Optional[float] = Field(None, description="Confidence score from STT")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class EvaluationMetrics(BaseModel):
    """
    Schema for answer evaluation metrics.
    """
    technical_accuracy: float = Field(..., description="Score for technical correctness")
    completeness: float = Field(..., description="Score for completeness of answer")
    clarity: float = Field(..., description="Score for clarity of explanation")
    relevance: float = Field(..., description="Score for relevance to question")
    overall_score: float = Field(..., description="Overall evaluation score")

class Feedback(BaseModel):
    """
    Schema for feedback generation.
    """
    id: str = Field(..., description="Unique identifier for the feedback")
    answer_id: str = Field(..., description="ID of the associated answer")
    strengths: List[str] = Field(..., description="Positive aspects of the answer")
    areas_for_improvement: List[str] = Field(..., description="Areas needing improvement")
    specific_recommendations: List[str] = Field(..., description="Specific suggestions for improvement")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class InterviewSession(BaseModel):
    """
    Schema for an interview session.
    """
    id: str = Field(..., description="Unique identifier for the session")
    role: str = Field(..., description="Target job role")
    questions: List[Question] = Field(..., description="List of questions asked")
    answers: List[Answer] = Field(..., description="List of candidate answers")
    feedback: List[Feedback] = Field(..., description="List of feedback items")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="When the session was completed")

class Role(BaseModel):
    """
    Schema for job roles.
    """
    id: str = Field(..., description="Unique identifier for the role")
    name: str = Field(..., description="Role name")
    description: str = Field(..., description="Role description")
    required_skills: List[str] = Field(..., description="Key skills for this role")
    technical_domains: List[str] = Field(..., description="Technical domains covered")
    created_at: datetime = Field(default_factory=datetime.utcnow)

# Example usage
def main():
    # Create a sample question
    question = Question(
        id="q1",
        role="Data Scientist",
        type="technical",
        difficulty="medium",
        content="Explain how a decision tree works.",
        expected_skills=["machine learning", "algorithms", "data structures"]
    )

    # Create a sample answer
    answer = Answer(
        id="a1",
        question_id="q1",
        content="A decision tree is a supervised learning algorithm that splits data based on feature values...",
        confidence=0.95
    )

    # Create evaluation metrics
    metrics = EvaluationMetrics(
        technical_accuracy=0.85,
        completeness=0.9,
        clarity=0.92,
        relevance=0.95,
        overall_score=0.9
    )

    # Create feedback
    feedback = Feedback(
        id="f1",
        answer_id="a1",
        strengths=["Clear explanation of decision tree mechanics", "Good use of examples"],
        areas_for_improvement=["Could explain pruning techniques", "Could add more about overfitting"],
        specific_recommendations=["Add discussion about tree depth", "Include common hyperparameters"]
    )

    print("Question:", question.json(indent=2))
    print("\nAnswer:", answer.json(indent=2))
    print("\nMetrics:", metrics.json(indent=2))
    print("\nFeedback:", feedback.json(indent=2))

if __name__ == "__main__":
    main()
