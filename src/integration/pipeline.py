"""
Interview Pipeline for connecting all components of the AI-NLP Interview Assistant.
This module implements the unified system pipeline.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from src.models.transformer import TransformerModel
from src.models.baseline import BaselineModel
from src.models.custom_evaluator import CustomEvaluatorModel
from src.feedback.generator import FeedbackGenerator
from src.schemas import Question, Answer, Feedback, EvaluationMetrics
from src.speech import SpeechToText

# Configure logging
logger = logging.getLogger(__name__)

class InterviewPipeline:
    """
    Unified pipeline connecting all components of the AI-NLP Interview Assistant.
    This class integrates the question generator, answer evaluator, and feedback generator.
    """
    
    def __init__(self, 
                 model_type: str = "transformer",
                 evaluator_type: str = "custom",
                 model_path: Optional[str] = None,
                 evaluator_path: Optional[str] = None,
                 use_speech_to_text: bool = False,
                 stt_api_key: Optional[str] = None):
        """
        Initialize the interview pipeline.
        
        Args:
            model_type: Type of model to use ("transformer" or "baseline")
            evaluator_type: Type of evaluator to use ("custom" or "baseline")
            model_path: Path to the pretrained model
            evaluator_path: Path to the pretrained evaluator model
            use_speech_to_text: Whether to use speech-to-text conversion
            stt_api_key: API key for speech-to-text service
        """
        # Initialize model
        if model_type == "transformer":
            self.model = TransformerModel()
        elif model_type == "baseline":
            self.model = BaselineModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Initialize evaluator
        if evaluator_type == "custom":
            self.evaluator = CustomEvaluatorModel(model_path=evaluator_path)
        elif evaluator_type == "baseline":
            self.evaluator = self.model
        else:
            raise ValueError(f"Unknown evaluator type: {evaluator_type}")
        
        # Initialize feedback generator
        self.feedback_generator = FeedbackGenerator()
        
        # Initialize speech-to-text converter if requested
        self.use_speech_to_text = use_speech_to_text
        if use_speech_to_text:
            self.speech_to_text = SpeechToText(api_key=stt_api_key)
        else:
            self.speech_to_text = None
        
        logger.info(f"Interview pipeline initialized with {model_type} model and {evaluator_type} evaluator")
    
    def process_audio_input(self, audio_path: str) -> str:
        """
        Process audio input and convert to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        if not self.use_speech_to_text or not self.speech_to_text:
            raise ValueError("Speech-to-text functionality is not enabled")
        
        logger.info(f"Processing audio input from {audio_path}")
        
        text = self.speech_to_text.transcribe(audio_path)
        
        logger.info(f"Audio transcribed to text: {text[:50]}...")
        
        return text
    
    def generate_question(self, role: str, difficulty: str = "medium", question_type: str = "technical") -> Question:
        """
        Generate an interview question for a specific role.
        
        Args:
            role: Job role for the question
            difficulty: Difficulty level (easy, medium, hard)
            question_type: Type of question (technical, behavioral)
            
        Returns:
            Generated question
        """
        logger.info(f"Generating {difficulty} {question_type} question for {role}")
        
        
        # Generate a unique ID
        question_id = f"q_{role.replace(' ', '_')}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Generate expected skills based on role
        expected_skills = []
        if role.lower() == "software engineer":
            expected_skills = ["Algorithms", "Data Structures", "Problem Solving"]
        elif role.lower() == "data scientist":
            expected_skills = ["Machine Learning", "Statistics", "Data Analysis"]
        
        # Generate question content
        if question_type.lower() == "technical":
            if role.lower() == "software engineer":
                content = "Explain the difference between depth-first search and breadth-first search algorithms and when you would use each."
            elif role.lower() == "data scientist":
                content = "Describe how you would handle imbalanced datasets in a classification problem."
            else:
                content = f"What are the key technical skills required for a {role} position?"
        else:  # behavioral
            content = f"Describe a challenging project you worked on as a {role} and how you overcame the obstacles."
        
        # Create question object
        question = Question(
            id=question_id,
            role=role,
            type=question_type,
            difficulty=difficulty,
            content=content,
            expected_skills=expected_skills,
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Generated question: {content}")
        
        return question
    
    def generate_answer(self, question: Question, context: Optional[str] = None) -> Answer:
        """
        Generate an answer to the given question.
        
        Args:
            question: Question to answer
            context: Optional additional context
            
        Returns:
            Generated answer
        """
        logger.info(f"Generating answer for question: {question.content}")
        
        # Use the model to generate an answer
        answer = self.model.generate_answer(question, context)
        
        logger.info(f"Generated answer: {answer.content[:50]}...")
        
        return answer
    
    def evaluate_answer(self, question: Question, answer: Answer) -> EvaluationMetrics:
        """
        Evaluate an answer using the evaluation model.
        
        Args:
            question: Original question
            answer: Answer to evaluate
            
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating answer for question: {question.content}")
        
        # Use the evaluator to evaluate the answer
        metrics = self.evaluator.evaluate_answer(question, answer)
        
        logger.info(f"Evaluation metrics: {metrics.__dict__}")
        
        return metrics
    
    def generate_feedback(self, question: Question, answer: Answer, metrics: Optional[EvaluationMetrics] = None) -> Feedback:
        """
        Generate feedback for an answer.
        
        Args:
            question: Original question
            answer: Answer to provide feedback on
            metrics: Optional evaluation metrics (will be calculated if not provided)
            
        Returns:
            Generated feedback
        """
        logger.info(f"Generating feedback for answer to question: {question.content}")
        
        # Evaluate the answer if metrics not provided
        if metrics is None:
            metrics = self.evaluate_answer(question, answer)
        
        # Generate feedback using the feedback generator
        feedback = self.feedback_generator.generate_feedback(question, answer, metrics)
        
        logger.info(f"Generated feedback: {feedback.content[:50]}...")
        
        return feedback
    
    def process_interview_question(self, 
                                  question: Union[Question, str],
                                  answer_text: Optional[str] = None,
                                  audio_path: Optional[str] = None,
                                  generate_model_answer: bool = False) -> Dict[str, Any]:
        """
        Process a complete interview question workflow.
        
        Args:
            question: Question object or question text
            answer_text: Text of the answer (if provided)
            audio_path: Path to audio file (if using speech input)
            generate_model_answer: Whether to generate a model answer
            
        Returns:
            Dictionary with question, answer, metrics, and feedback
        """
        # Convert question text to Question object if needed
        if isinstance(question, str):
            question = Question(
                id=f"q_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                role="General",
                type="general",
                difficulty="medium",
                content=question,
                expected_skills=[],
                created_at=datetime.utcnow()
            )
        
        # Get answer text from audio if provided
        if answer_text is None and audio_path:
            answer_text = self.process_audio_input(audio_path)
        
        # Create Answer object from text
        if answer_text:
            answer = Answer(
                id=f"a_{question.id}",
                question_id=question.id,
                content=answer_text,
                audio_source=audio_path,
                confidence=1.0,
                created_at=datetime.utcnow()
            )
        elif generate_model_answer:
            # Generate model answer if requested
            answer = self.generate_answer(question)
        else:
            raise ValueError("Either answer_text, audio_path, or generate_model_answer must be provided")
        
        # Evaluate the answer
        metrics = self.evaluate_answer(question, answer)
        
        # Generate feedback
        feedback = self.generate_feedback(question, answer, metrics)
        
        # Return all components
        return {
            "question": question,
            "answer": answer,
            "metrics": metrics,
            "feedback": feedback
        }
    
    def get_available_roles(self) -> List[str]:
        """
        Get list of available roles for interview questions.
        
        Returns:
            List of available roles
        """
        # This would typically be loaded from a database or configuration
        return [
            "Software Engineer",
            "Data Scientist",
            "Machine Learning Engineer",
            "Frontend Developer",
            "Backend Developer",
            "Full Stack Developer",
            "DevOps Engineer",
            "Cloud Engineer",
            "Data Engineer",
            "Security Engineer",
            "QA Engineer",
            "Mobile Developer",
            "Site Reliability Engineer"
        ]
