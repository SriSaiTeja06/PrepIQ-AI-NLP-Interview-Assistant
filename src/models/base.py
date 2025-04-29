from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import torch
from transformers import PreTrainedModel
from src.schemas import Question, Answer, Feedback

class BaseModel(ABC):
    """Base class for all models in the interview assistant system."""
    
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the base model.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (str): Device to run the model on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model and tokenizer."""
        pass
        
    @abstractmethod
    def generate_answer(self, question: Question, context: Optional[str] = None) -> Answer:
        """
        Generate an answer to a given question.
        
        Args:
            question (Question): The question to answer
            context (Optional[str]): Additional context for the answer
            
        Returns:
            Answer: The generated answer
        """
        pass
        
    @abstractmethod
    def evaluate_answer(self, question: Question, answer: Answer) -> Dict[str, float]:
        """
        Evaluate the quality of an answer.
        
        Args:
            question (Question): The original question
            answer (Answer): The answer to evaluate
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        pass
        
    @abstractmethod
    def generate_feedback(self, question: Question, answer: Answer) -> Feedback:
        """
        Generate feedback for an answer.
        
        Args:
            question (Question): The original question
            answer (Answer): The answer to provide feedback on
            
        Returns:
            Feedback: Generated feedback
        """
        pass
        
    def to(self, device: str) -> None:
        """
        Move the model to the specified device.
        
        Args:
            device (str): Device to move the model to ("cpu" or "cuda")
        """
        self.device = device
        if self.model is not None:
            self.model.to(device)
