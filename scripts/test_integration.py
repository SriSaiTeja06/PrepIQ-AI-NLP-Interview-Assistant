"""
Test script for the integrated interview pipeline.
This script tests the complete integration of the question generator, answer evaluator, and feedback generator.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.integration.pipeline import InterviewPipeline
from src.schemas import Question, Answer, EvaluationMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_question_generation(pipeline: InterviewPipeline):
    """Test question generation for different roles."""
    roles = pipeline.get_available_roles()
    logger.info(f"Testing question generation for {len(roles)} roles")
    
    for role in roles[:3]:  
        for difficulty in ["easy", "medium", "hard"]:
            for question_type in ["technical", "behavioral"]:
                logger.info(f"Generating {difficulty} {question_type} question for {role}")
                question = pipeline.generate_question(
                    role=role,
                    difficulty=difficulty,
                    question_type=question_type
                )
                logger.info(f"Generated question: {question.content}")
                assert question.role == role
                assert question.difficulty == difficulty
                assert question.type == question_type
    
    logger.info("✓ Question generation tests passed")
    return True

def test_answer_generation(pipeline: InterviewPipeline):
    """Test answer generation for sample questions."""
    question = Question(
        id="q_test_1",
        role="Software Engineer",
        type="technical",
        difficulty="medium",
        content="What is the difference between a list and a tuple in Python?",
        expected_skills=["Python", "Data Structures"],
        created_at=datetime.utcnow()
    )
    
    logger.info(f"Testing answer generation for question: {question.content}")
    answer = pipeline.generate_answer(question)
    
    logger.info(f"Generated answer: {answer.content[:100]}...")
    assert answer.question_id == question.id
    assert len(answer.content) > 0
    
    logger.info("✓ Answer generation tests passed")
    return answer

def test_answer_evaluation(pipeline: InterviewPipeline, question: Question, answer: Answer):
    """Test answer evaluation."""
    logger.info(f"Testing answer evaluation for question: {question.content}")
    metrics = pipeline.evaluate_answer(question, answer)
    
    logger.info(f"Evaluation metrics: {metrics.__dict__}")
    assert 0 <= metrics.technical_accuracy <= 1
    assert 0 <= metrics.completeness <= 1
    assert 0 <= metrics.clarity <= 1
    assert 0 <= metrics.relevance <= 1
    assert 0 <= metrics.overall_score <= 1
    
    logger.info("✓ Answer evaluation tests passed")
    return metrics

def test_feedback_generation(pipeline: InterviewPipeline, question: Question, answer: Answer, metrics: EvaluationMetrics):
    """Test feedback generation."""
    logger.info(f"Testing feedback generation for question: {question.content}")
    feedback = pipeline.generate_feedback(question, answer, metrics)
    
    logger.info(f"Generated feedback: {feedback.content[:100]}...")
    assert feedback.answer_id == answer.id
    assert len(feedback.content) > 0
    
    logger.info("✓ Feedback generation tests passed")
    return feedback

def test_complete_pipeline(pipeline: InterviewPipeline):
    """Test the complete interview pipeline workflow."""
    logger.info("Testing complete interview pipeline")
    
    question = Question(
        id="q_test_complete",
        role="Data Scientist",
        type="technical",
        difficulty="medium",
        content="Explain how you would handle missing data in a dataset.",
        expected_skills=["Data Preprocessing", "Statistics"],
        created_at=datetime.utcnow()
    )
    
    answer_text = (
        "When handling missing data in a dataset, I typically follow a systematic approach. "
        "First, I assess the extent and pattern of missingness - whether it's missing completely at random (MCAR), "
        "missing at random (MAR), or missing not at random (MNAR). "
        "For small amounts of missing data, removal strategies like listwise or pairwise deletion might be appropriate. "
        "For larger amounts, I prefer imputation methods such as mean/median/mode imputation for simple cases, "
        "or more sophisticated approaches like KNN, regression, or multiple imputation for complex datasets. "
        "I also consider using algorithms that handle missing values inherently, such as certain tree-based methods. "
        "The choice depends on the specific context, the amount of missing data, and the impact on subsequent analyses."
    )
    
    logger.info(f"Processing question: {question.content}")
    result = pipeline.process_interview_question(
        question=question,
        answer_text=answer_text
    )
    
    logger.info(f"Complete pipeline result:")
    logger.info(f"Question: {result['question'].content}")
    logger.info(f"Answer: {result['answer'].content[:100]}...")
    logger.info(f"Metrics: {result['metrics'].__dict__}")
    logger.info(f"Feedback: {result['feedback'].content[:100]}...")
    
    assert result['question'].id == question.id
    assert result['answer'].question_id == question.id
    assert result['feedback'].answer_id == result['answer'].id
    
    logger.info("✓ Complete pipeline tests passed")
    return result

def save_test_results(result, output_dir):
    """Save test results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    serialized_result = {
        "question": {
            "id": result["question"].id,
            "role": result["question"].role,
            "type": result["question"].type,
            "difficulty": result["question"].difficulty,
            "content": result["question"].content,
            "expected_skills": result["question"].expected_skills,
            "created_at": result["question"].created_at.isoformat()
        },
        "answer": {
            "id": result["answer"].id,
            "question_id": result["answer"].question_id,
            "content": result["answer"].content,
            "created_at": result["answer"].created_at.isoformat()
        },
        "metrics": {
            "technical_accuracy": result["metrics"].technical_accuracy,
            "completeness": result["metrics"].completeness,
            "clarity": result["metrics"].clarity,
            "relevance": result["metrics"].relevance,
            "overall_score": result["metrics"].overall_score
        },
        "feedback": {
            "id": result["feedback"].id,
            "answer_id": result["feedback"].answer_id,
            "content": result["feedback"].content,
            "created_at": result["feedback"].created_at.isoformat()
        }
    }
    
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(serialized_result, f, indent=2)
    
    with open(output_dir / "question.txt", "w", encoding="utf-8") as f:
        f.write(result["question"].content)
    
    with open(output_dir / "answer.txt", "w", encoding="utf-8") as f:
        f.write(result["answer"].content)
    
    with open(output_dir / "metrics.txt", "w", encoding="utf-8") as f:
        for key, value in result["metrics"].__dict__.items():
            f.write(f"{key}: {value}\n")
    
    with open(output_dir / "feedback.txt", "w", encoding="utf-8") as f:
        f.write(result["feedback"].content)
    
    logger.info(f"Test results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test the integrated interview pipeline")
    parser.add_argument("--model-type", type=str, default="transformer", choices=["transformer", "baseline"],
                        help="Type of model to use")
    parser.add_argument("--evaluator-type", type=str, default="custom", choices=["custom", "baseline"],
                        help="Type of evaluator to use")
    parser.add_argument("--output-dir", type=str, default="test_results",
                        help="Directory to save test results")
    args = parser.parse_args()
    
    try:
        logger.info("Initializing interview pipeline")
        pipeline = InterviewPipeline(
            model_type=args.model_type,
            evaluator_type=args.evaluator_type
        )
        
        test_question_generation(pipeline)
        
        question = Question(
            id="q_test_2",
            role="Software Engineer",
            type="technical",
            difficulty="medium",
            content="Explain the concept of object-oriented programming and its main principles.",
            expected_skills=["OOP", "Software Design"],
            created_at=datetime.utcnow()
        )
        
        answer = test_answer_generation(pipeline)
        metrics = test_answer_evaluation(pipeline, question, answer)
        feedback = test_feedback_generation(pipeline, question, answer, metrics)
        
        result = test_complete_pipeline(pipeline)
        
        save_test_results(result, args.output_dir)
        
        logger.info("All tests passed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
