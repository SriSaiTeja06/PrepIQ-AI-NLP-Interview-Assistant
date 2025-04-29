"""
Custom Evaluator Model using Sentence Transformers and a custom head.
"""
import os
import glob
import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from src.models.base import BaseModel
from src.schemas import Question, Answer, EvaluationMetrics, Feedback
from datetime import datetime 

logger = logging.getLogger(__name__)

# --- Custom Model Components ---

class FeatureExtractor(nn.Module):
    """
    Extracts features using a pretrained transformer model.
    """
    def __init__(self, pretrained_model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

class MetricPredictor(nn.Module):
    """
    Predicts a single evaluation metric score from features.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.dense2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x.squeeze(-1) 

class CustomEvaluator(nn.Module):
    """
    Combines feature extraction and metric prediction heads.
    """
    def __init__(self, pretrained_model_name: str, hidden_dim: int = 256):
        """
        Args:
            pretrained_model_name: Name of the transformer model from Hugging Face
            hidden_dim: Dimension of the hidden layer in predictor heads
        """
        super().__init__()
        self.feature_extractor = FeatureExtractor(pretrained_model_name)
        
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(pretrained_model_name)
        encoder_output_dim = config.hidden_size

        self.technical_accuracy_head = MetricPredictor(encoder_output_dim, hidden_dim)
        self.completeness_head = MetricPredictor(encoder_output_dim, hidden_dim)
        self.clarity_head = MetricPredictor(encoder_output_dim, hidden_dim)
        self.relevance_head = MetricPredictor(encoder_output_dim, hidden_dim)

    def forward(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Dictionary containing predicted scores for each metric.
        """
        features = self.feature_extractor(input_ids, attention_mask)
        
        technical_accuracy = self.technical_accuracy_head(features)
        completeness = self.completeness_head(features)
        clarity = self.clarity_head(features)
        relevance = self.relevance_head(features)
        
        return {
            "technical_accuracy": technical_accuracy,
            "completeness": completeness,
            "clarity": clarity,
            "relevance": relevance
        }

# --- Wrapper Model Class ---

class CustomEvaluatorModel(BaseModel):
    """
    Wrapper class for the CustomEvaluator, handling loading, preparation, and evaluation.
    Inherits from BaseModel for consistency.
    """
    
    def __init__(self, model_path: Optional[str] = None, models_dir: str = "models"):
        """
        Initialize the custom evaluator model. Automatically loads the latest
        trained model from `models_dir` if `model_path` is not provided.

        Args:
            model_path (Optional[str]): Specific path to model weights. If None,
                                         loads the latest model from `models_dir`.
            models_dir (str): Directory containing saved model checkpoints.
                              Defaults to "models".
        """
        super().__init__(model_name="sentence-transformers/all-mpnet-base-v2")
        
        determined_path = model_path

        # Always check for 'custom_evaluator_best.pt' first
        best_model_path = os.path.join(models_dir, "custom_evaluator_best.pt")
        if not determined_path and os.path.exists(best_model_path):
            logger.info(f"Found 'custom_evaluator_best.pt'. Loading this model: {best_model_path}")
            determined_path = best_model_path
        elif not determined_path:
            logger.warning(f"No 'custom_evaluator_best.pt' found in '{models_dir}/'. Please train the model first.")
            determined_path = None

        self.model_path = determined_path
        self.load_model()
        
        # Load role-specific criteria for evaluation (remains the same)
        self.role_criteria = self._load_role_criteria()
    
    def load_model(self) -> None:
        """Load the pretrained tokenizer, base model, and trained weights if available."""
        logger.info(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        logger.info("Initializing CustomEvaluator structure...")
        self.model = CustomEvaluator(pretrained_model_name=self.model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Setting model device to: {self.device}")

        if self.model_path and os.path.exists(self.model_path):
            try:
                logger.info(f"Loading trained weights from: {self.model_path}")
                if self.device == torch.device("cuda"):
                    self.model.load_state_dict(torch.load(self.model_path))
                else:
                    self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device("cpu")))
                logger.info("Trained weights loaded successfully.")
            except Exception as e:
                 logger.error(f"Failed to load weights from {self.model_path}: {e}. Using base model weights only.")
                 
        else:
            if self.model_path: 
                 logger.warning(f"Model path specified but not found: {self.model_path}. Using base model weights only.")
            else: 
                 logger.info("No trained model path specified or found. Using base model weights only.")
            
        self.model.to(self.device)
        self.model.eval() 

    def _load_role_criteria(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load role-specific criteria for evaluation.
        (Placeholder implementation)
        """
        return {
            "Software Engineer": {
                "technical_keywords": ["algorithm", "data structure", "complexity", "efficiency", "testing", "debugging"],
                "expected_concepts": ["time complexity", "space complexity", "edge cases", "optimization"]
            },
            "Data Scientist": {
                "technical_keywords": ["machine learning", "statistics", "data", "model", "algorithm", "prediction"],
                "expected_concepts": ["feature engineering", "model evaluation", "statistical significance", "bias-variance tradeoff"]
            },
            
        }
        
    def _prepare_input(self, question: Question, answer: Answer) -> str:
        """
        Prepare input text for the model by combining question and answer details.
        """
        q_content = getattr(question, 'content', 'Unknown Question')
        q_role = getattr(question, 'role', 'unknown')
        q_difficulty = getattr(question, 'difficulty', 'unknown')
        a_content = getattr(answer, 'content', '')
        return f"Question: {q_content}\nRole: {q_role}\nDifficulty: {q_difficulty}\nAnswer: {a_content}"
        
    def evaluate_answer(self, question: Question, answer: Answer) -> EvaluationMetrics:
        """
        Evaluate an answer using the custom neural model.
        """
        if not self.model or not self.tokenizer:
             logger.error("Model or tokenizer not loaded properly for evaluation.")
             return EvaluationMetrics(technical_accuracy=0.0, completeness=0.0, clarity=0.0, relevance=0.0, overall_score=0.0)

        # Prepare input
        input_text = self._prepare_input(question, answer)
        
        # Tokenize
        try:
            inputs = self.tokenizer(input_text, 
                                   return_tensors="pt", 
                                   truncation=True, 
                                   max_length=512, 
                                   padding="max_length")
        except Exception as e:
             logger.error(f"Tokenization failed: {e}")
             return EvaluationMetrics(technical_accuracy=0.0, completeness=0.0, clarity=0.0, relevance=0.0, overall_score=0.0)

        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            if not isinstance(outputs, dict) or not all(k in outputs for k in ["technical_accuracy", "completeness", "clarity", "relevance"]):
                 logger.error(f"Model output is not a dict or missing expected keys: {outputs}")
                 raise ValueError("Invalid output format from model.")

            # Extract scores, move to CPU before numpy conversion or float casting
            tech_acc = float(outputs['technical_accuracy'].cpu())
            completeness = float(outputs['completeness'].cpu())
            clarity = float(outputs['clarity'].cpu())
            relevance = float(outputs['relevance'].cpu())

            # Calculate overall score using CPU values
            overall_score = float(np.mean([
                tech_acc,
                completeness,
                clarity,
                relevance
            ]))

            metrics = EvaluationMetrics(
                technical_accuracy=tech_acc,
                completeness=completeness,
                clarity=clarity,
                relevance=relevance,
                overall_score=overall_score
            )
            return metrics

        except Exception as e:
             logger.exception(f"Error during model inference or metric calculation: {e}")
             return EvaluationMetrics(technical_accuracy=0.0, completeness=0.0, clarity=0.0, relevance=0.0, overall_score=0.0)

        
    def generate_answer(self, question: Question, context: Optional[str] = None) -> Answer:
        """
        Generate an answer to the question. (Placeholder)
        """
        logger.warning("generate_answer called on CustomEvaluatorModel, which is not designed for generation.")
        answer_text = "This model focuses on evaluation, not answer generation."
        
        q_id = getattr(question, 'id', 'unknown_question')

        return Answer(
            id=f"a_{q_id}",
            question_id=q_id,
            content=answer_text,
            confidence=0.1, 
            created_at=datetime.utcnow()
        )
        
    def generate_feedback(self, question: Question, answer: Answer) -> Feedback:
        """
        Generate feedback for an answer. (Placeholder - uses simple rule-based text)
        """
        logger.warning("generate_feedback called on CustomEvaluatorModel. Using internal basic feedback generation.")
        # First evaluate the answer
        metrics = self.evaluate_answer(question, answer)
        
        # Generate feedback based on metrics
        feedback_text = self._generate_feedback_text(question, answer, metrics)
        
        ans_id = getattr(answer, 'id', 'unknown_answer')

        return Feedback(
            id=f"f_{ans_id}",
            answer_id=ans_id,
            content=feedback_text,
            metrics=metrics,
            created_at=datetime.utcnow()
        )
        
    def _generate_feedback_text(self, question: Question, answer: Answer, metrics: EvaluationMetrics) -> str:
        """
        Generate basic feedback text based on evaluation metrics.
        (This is a simplified version, the main FeedbackGenerator class is more sophisticated)
        """
        strengths = []
        areas_for_improvement = []
        
        # Analyze technical accuracy
        if metrics.technical_accuracy >= 0.8:
            strengths.append("Strong technical understanding.")
        elif metrics.technical_accuracy >= 0.5:
            areas_for_improvement.append("Technical accuracy could be improved.")
        else:
            areas_for_improvement.append("Significant technical inaccuracies noted.")
        
        # Analyze completeness
        if metrics.completeness >= 0.8:
            strengths.append("Comprehensive answer.")
        elif metrics.completeness >= 0.5:
            areas_for_improvement.append("Answer could be more complete.")
        else:
            areas_for_improvement.append("Answer is incomplete.")
        
        # Analyze clarity
        if metrics.clarity >= 0.8:
            strengths.append("Clear and well-structured explanation.")
        elif metrics.clarity >= 0.5:
            areas_for_improvement.append("Clarity could be improved with better structure.")
        else:
            areas_for_improvement.append("Answer lacks clarity.")
        
        # Analyze relevance
        if metrics.relevance >= 0.8:
            strengths.append("Highly relevant to the question.")
        elif metrics.relevance >= 0.5:
            areas_for_improvement.append("Answer could be more focused on the question.")
        else:
            areas_for_improvement.append("Answer includes irrelevant information or is off-topic.")
        
        # Compile the feedback
        q_content = getattr(question, 'content', 'the question')
        feedback = f"## Feedback for your answer to: '{q_content}'\n\n"
        feedback += f"### Overall Score: {metrics.overall_score:.2f}\n\n"
        
        if strengths:
             feedback += "### Strengths:\n"
             for strength in strengths:
                 feedback += f"- {strength}\n"
             feedback += "\n"
        
        if areas_for_improvement:
             feedback += "### Areas for Improvement:\n"
             for area in areas_for_improvement:
                 feedback += f"- {area}\n"
             feedback += "\n"
        
        feedback += "### Detailed Metrics:\n"
        feedback += f"- Technical Accuracy: {metrics.technical_accuracy:.2f}\n"
        feedback += f"- Completeness: {metrics.completeness:.2f}\n"
        feedback += f"- Clarity: {metrics.clarity:.2f}\n"
        feedback += f"- Relevance: {metrics.relevance:.2f}\n"
        
        return feedback
