import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.models.base import BaseModel
from src.schemas import Question, Answer, Feedback
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

class TransformerModel(BaseModel):
    """Transformer-based model for the interview assistant system."""
    
    def __init__(self, model_name: str = "google/flan-t5-large", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the transformer model.
        
        Args:
            model_name (str): Name of the pretrained transformer model
            device (str): Device to run the model on ("cpu" or "cuda")
        """
        super().__init__(model_name, device)
        self.load_model()
        
    def load_model(self) -> None:
        """Load the pretrained transformer model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
    def _prepare_input(self, question: Question, context: Optional[str] = None) -> str:
        """
        Prepare the input prompt for the model.
        
        Args:
            question (Question): The question to answer
            context (Optional[str]): Additional context for the answer
            
        Returns:
            str: Prepared input prompt
        """
        prompt = f"""
        Role: {question.role}
        Question Type: {question.type}
        Difficulty: {question.difficulty}
        Expected Skills: {', '.join(question.expected_skills)}
        
        Question: {question.content}
        """
        
        if context:
            prompt += f"\nContext: {context}"
            
        return prompt
        
    def generate_answer(self, question: Question, context: Optional[str] = None) -> Answer:
        """
        Generate an answer to a given question using the transformer model.
        
        Args:
            question (Question): The question to answer
            context (Optional[str]): Additional context for the answer
            
        Returns:
            Answer: The generated answer
        """
        prompt = self._prepare_input(question, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
        answer_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return Answer(
            id=f"a_{question.id}",
            question_id=question.id,
            content=answer_text,
            confidence=0.9,  
            created_at=datetime.utcnow()
        )
        
    def evaluate_answer(self, question: Question, answer: Answer) -> Dict[str, float]:
        """
        Evaluate the quality of an answer using the transformer model.
        
        Args:
            question (Question): The original question
            answer (Answer): The answer to evaluate
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        prompt = f"""
        Evaluate the following answer based on the question and expected skills:
        
        Question: {question.content}
        Expected Skills: {', '.join(question.expected_skills)}
        
        Answer: {answer.content}
        
        Provide scores for:
        - Technical Accuracy (0-1)
        - Completeness (0-1)
        - Clarity (0-1)
        - Relevance (0-1)
        
        Format: {{"technical_accuracy": X, "completeness": Y, "clarity": Z, "relevance": W}}
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=1,
                temperature=0.2,
                top_p=0.9,
                do_sample=False
            )
            
        metrics_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            metrics = json.loads(metrics_text)
            return metrics
        except json.JSONDecodeError:
            return {
                "technical_accuracy": 0.5,
                "completeness": 0.5,
                "clarity": 0.5,
                "relevance": 0.5
            }
            
    def generate_feedback(self, question: Question, answer: Answer) -> Feedback:
        """
        Generate feedback for an answer using the transformer model.
        
        Args:
            question (Question): The original question
            answer (Answer): The answer to provide feedback on
            
        Returns:
            Feedback: Generated feedback
        """
        prompt = f"""
        Generate feedback for the following answer:
        
        Question: {question.content}
        Expected Skills: {', '.join(question.expected_skills)}
        
        Answer: {answer.content}
        
        Provide:
        - Strengths (list of points)
        - Areas for improvement (list of points)
        - Specific recommendations (list of points)
        
        Format:
        {{
            "strengths": ["point1", "point2"],
            "areas_for_improvement": ["point1", "point2"],
            "specific_recommendations": ["point1", "point2"]
        }}
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=1024,
                num_beams=1,
                temperature=0.2,
                top_p=0.9,
                do_sample=False
            )
            
        feedback_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        try:
            feedback_dict = json.loads(feedback_text)
            return Feedback(
                id=f"f_{answer.id}",
                answer_id=answer.id,
                strengths=feedback_dict.get("strengths", []),
                areas_for_improvement=feedback_dict.get("areas_for_improvement", []),
                specific_recommendations=feedback_dict.get("specific_recommendations", []),
                created_at=datetime.utcnow()
            )
        except json.JSONDecodeError:
            return Feedback(
                id=f"f_{answer.id}",
                answer_id=answer.id,
                strengths=["Good attempt at answering"],
                areas_for_improvement=["Could provide more technical details"],
                specific_recommendations=["Review the expected skills"],
                created_at=datetime.utcnow()
            )
