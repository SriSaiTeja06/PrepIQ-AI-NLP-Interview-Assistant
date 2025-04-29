import json
from typing import List, Dict, Optional
import os
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from src.schemas import Question, Answer, Feedback
import torch # Added torch import

class InterviewDataset(Dataset):
    """Dataset class for interview assistant training."""
    
    @dataclass
    class TrainingExample:
        """Data structure for a single training example."""
        input_text: str
        target_text: str
        metadata: Dict[str, str]
        
    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        task: str = "all"
    ):
        """
        Initialize the interview dataset.
        
        Args:
            data_dir (str): Directory containing the training data
            tokenizer (PreTrainedTokenizer): Tokenizer to use
            max_length (int): Maximum sequence length
            task (str): Task type ("answer", "evaluate", "feedback", or "all")
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        self.examples = []
        
        # Load all data files
        self._load_data(data_dir)
        
    def _load_data(self, data_dir: str) -> None:
        """Load data from the specified directory."""
        for filename in os.listdir(data_dir):
            if filename.endswith("_questions.json"):
                role = filename.replace("_questions.json", "")
                
                # Load questions
                with open(os.path.join(data_dir, filename), "r") as f:
                    questions = [Question(**q) for q in json.load(f)]
                
                # Load answers
                with open(os.path.join(data_dir, f"{role}_answers.json"), "r") as f:
                    answers = [Answer(**a) for a in json.load(f)]
                
                # Load feedback
                with open(os.path.join(data_dir, f"{role}_feedback.json"), "r") as f:
                    feedbacks = [Feedback(**f) for f in json.load(f)]
                
                # Create training examples
                self._create_examples(questions, answers, feedbacks, role)
                
    def _create_examples(
        self,
        questions: List[Question],
        answers: List[Answer],
        feedbacks: List[Feedback],
        role: str
    ) -> None:
        """Create training examples from the loaded data."""
        
        # Create answer generation examples
        if self.task in ["answer", "all"]:
            for question in questions:
                self.examples.append(
                    self.TrainingExample(
                        input_text=self._prepare_answer_input(question),
                        target_text="",  # Will be filled during training
                        metadata={
                            "role": role,
                            "question_id": question.id,
                            "task": "answer"
                        }
                    )
                )
                
        # Create evaluation examples
        if self.task in ["evaluate", "all"]:
            for answer in answers:
                question = next(q for q in questions if q.id == answer.question_id)
                self.examples.append(
                    self.TrainingExample(
                        input_text=self._prepare_evaluation_input(question, answer),
                        target_text="",  # Will be filled during training
                        metadata={
                            "role": role,
                            "answer_id": answer.id,
                            "task": "evaluate"
                        }
                    )
                )
                
        # Create feedback examples
        if self.task in ["feedback", "all"]:
            for feedback in feedbacks:
                answer = next(a for a in answers if a.id == feedback.answer_id)
                question = next(q for q in questions if q.id == answer.question_id)
                self.examples.append(
                    self.TrainingExample(
                        input_text=self._prepare_feedback_input(question, answer),
                        target_text="",  # Will be filled during training
                        metadata={
                            "role": role,
                            "feedback_id": feedback.id,
                            "task": "feedback"
                        }
                    )
                )
                
    def _prepare_answer_input(self, question: Question) -> str:
        """Prepare input for answer generation."""
        return f"""
        Role: {question.role}
        Question Type: {question.type}
        Difficulty: {question.difficulty}
        Expected Skills: {', '.join(question.expected_skills)}
        
        Question: {question.content}
        """
        
    def _prepare_evaluation_input(self, question: Question, answer: Answer) -> str:
        """Prepare input for answer evaluation."""
        return f"""
        Question: {question.content}
        Expected Skills: {', '.join(question.expected_skills)}
        
        Answer: {answer.content}
        
        Evaluate the answer based on:
        - Technical Accuracy (0-1)
        - Completeness (0-1)
        - Clarity (0-1)
        - Relevance (0-1)
        """
        
    def _prepare_feedback_input(self, question: Question, answer: Answer) -> str:
        """Prepare input for feedback generation."""
        return f"""
        Question: {question.content}
        Expected Skills: {', '.join(question.expected_skills)}
        
        Answer: {answer.content}
        
        Provide feedback including:
        - Strengths
        - Areas for improvement
        - Specific recommendations
        """
        
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            example.input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize target
        targets = self.tokenizer(
            example.target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze()
        }
