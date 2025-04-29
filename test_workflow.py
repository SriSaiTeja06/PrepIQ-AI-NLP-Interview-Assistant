"""
End-to-End workflow test for the AI-NLP Interview Assistant.
This script tests the complete workflow from question generation to feedback.
"""

import os
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from src.schemas import Question, Answer, EvaluationMetrics
from src.models.transformer import TransformerModel
from src.models.custom_evaluator import CustomEvaluatorModel
from src.feedback.generator import FeedbackGenerator
from src.integration.pipeline import InterviewPipeline

def test_end_to_end():
    """Test the complete end-to-end workflow."""
    try:
        logger.info("Starting end-to-end workflow test")
        
        logger.info("Initializing interview pipeline")
        pipeline = InterviewPipeline(
            model_type="transformer",
            evaluator_type="custom",
            use_speech_to_text=False
        )
        
        # 1. Test role retrieval
        logger.info("Testing role retrieval")
        roles = pipeline.get_available_roles()
        logger.info(f"Available roles: {roles}")
        assert len(roles) > 0, "No roles returned"
        
        # 2. Test question generation
        role = roles[0]  
        logger.info(f"Testing question generation for role: {role}")
        question = pipeline.generate_question(
            role=role,
            difficulty="medium",
            question_type="technical"
        )
        logger.info(f"Generated question: {question.content}")
        assert question.role == role, "Question role doesn't match"
        
        # 3. Test answer generation
        logger.info("Testing answer generation")
        answer = pipeline.generate_answer(question)
        logger.info(f"Generated answer: {answer.content[:100]}...")
        assert answer.question_id == question.id, "Answer question ID doesn't match"
        
        # 4. Test answer evaluation
        logger.info("Testing answer evaluation")
        metrics = pipeline.evaluate_answer(question, answer)
        logger.info(f"Evaluation metrics: {metrics.__dict__}")
        assert 0 <= metrics.technical_accuracy <= 1, "Technical accuracy out of range"
        assert 0 <= metrics.overall_score <= 1, "Overall score out of range"
        
        # 5. Test feedback generation
        logger.info("Testing feedback generation")
        feedback = pipeline.generate_feedback(question, answer, metrics)
        logger.info(f"Generated feedback: {feedback.content[:100]}...")
        assert feedback.answer_id == answer.id, "Feedback answer ID doesn't match"
        
        # 6. Test complete pipeline
        logger.info("Testing complete pipeline processing")
        test_answer_text = "This is a test answer for the complete pipeline test."
        result = pipeline.process_interview_question(
            question=question,
            answer_text=test_answer_text
        )
        logger.info(f"Complete pipeline result: {result['metrics'].__dict__}")
        assert result['question'].id == question.id, "Pipeline question ID doesn't match"
        assert result['answer'].content == test_answer_text, "Pipeline answer content doesn't match"
        
        logger.info("End-to-end workflow test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during end-to-end test: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_end_to_end()
    sys.exit(0 if success else 1)
