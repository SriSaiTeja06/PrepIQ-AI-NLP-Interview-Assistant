from src.models.base import BaseModel
from src.schemas import Question, Answer, Feedback
from typing import Dict, List, Optional
import random
from datetime import datetime

class BaselineModel(BaseModel):
    """Simple baseline model for the interview assistant system."""
    
    def __init__(self, model_name: str = "baseline", device: str = "cpu"):
        """
        Initialize the baseline model.
        
        Args:
            model_name (str): Name of the model
            device (str): Device to run the model on ("cpu" or "cuda")
        """
        super().__init__(model_name, device)
        
    def load_model(self) -> None:
        """Load the baseline model."""
        pass
        
    def generate_answer(self, question: Question, context: Optional[str] = None) -> Answer:
        """
        Generate a simple baseline answer to a given question.
        
        Args:
            question (Question): The question to answer
            context (Optional[str]): Additional context for the answer
            
        Returns:
            Answer: The generated answer
        """
        answer_text = f"""
        Here's a simple answer to your question about {question.role}:
        
        The key aspects to consider are:
        - {' '.join(question.expected_skills[:3])}
        - {' '.join(question.expected_skills[3:])}
        
        This is a {question.difficulty} level question focusing on {question.type}.
        """
        
        return Answer(
            id=f"a_{question.id}",
            question_id=question.id,
            content=answer_text,
            confidence=0.7, 
            created_at=datetime.utcnow()
        )
        
    def evaluate_answer(self, question: Question, answer: Answer) -> Dict[str, float]:
        """
        Evaluate the quality of an answer using simple heuristics.
        
        Args:
            question (Question): The original question
            answer (Answer): The answer to evaluate
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        metrics = {
            "technical_accuracy": 0.6,
            "completeness": 0.7,
            "clarity": 0.8,
            "relevance": 0.7
        }
        
        if question.type == "technical":
            metrics["technical_accuracy"] *= 1.2
        elif question.type == "behavioral":
            metrics["clarity"] *= 1.2
            
        for key in metrics:
            metrics[key] = min(1.0, max(0.0, metrics[key]))
            
        return metrics
        
    def generate_feedback(self, question: Question, answer: Answer) -> Feedback:
        """
        Generate simple feedback for an answer.
        
        Args:
            question (Question): The original question
            answer (Answer): The answer to provide feedback on
            
        Returns:
            Feedback: Generated feedback
        """
        if question.type == "technical":
            feedback = {
                "strengths": [
                    "Basic technical understanding demonstrated",
                    "Mentioned key concepts"
                ],
                "areas_for_improvement": [
                    "Could provide more specific examples",
                    "Could explain implementation details"
                ],
                "specific_recommendations": [
                    "Review the expected skills in more depth",
                    "Practice explaining technical concepts clearly"
                ]
            }
        else:  # behavioral
            feedback = {
                "strengths": [
                    "Basic understanding of the situation",
                    "Mentioned relevant skills"
                ],
                "areas_for_improvement": [
                    "Could provide more specific examples",
                    "Could explain actions in more detail"
                ],
                "specific_recommendations": [
                    "Use the STAR method for better structure",
                    "Practice providing concrete examples"
                ]
            }
            
        return Feedback(
            id=f"f_{answer.id}",
            answer_id=answer.id,
            strengths=feedback["strengths"],
            areas_for_improvement=feedback["areas_for_improvement"],
            specific_recommendations=feedback["specific_recommendations"],
            created_at=datetime.utcnow()
        )
