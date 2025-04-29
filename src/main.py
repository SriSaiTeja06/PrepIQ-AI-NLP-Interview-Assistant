from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
import logging
import base64  # Import base64 module
import os
import json
from src.speech_to_text import SpeechToText
from src.schemas import Question, Answer, EvaluationMetrics, Feedback 
from config.config import settings
from src.models.custom_evaluator import CustomEvaluatorModel
from src.feedback.generator import FeedbackGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
stt = SpeechToText() 
custom_model = CustomEvaluatorModel()
feedback_generator = FeedbackGenerator()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-NLP Interview Assistant API"
)

QUESTIONS_DIR = os.path.join("organized_data", "questions")

class TranscriptionRequest(BaseModel):
    audio_content: str
    domain: str = "general"

class EvaluationRequest(BaseModel):
    question: str
    answer: str
    expected_skills: List[str]

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    metrics: Dict[str, float]

# --- Request Schemas ---
class QuestionRequest(BaseModel):
    role: str
    difficulty: str = 'medium'
    type: str = 'technical'

@app.post("/api/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    """
    Transcribe audio content to text.
    """
    try:
        try:
            if "," in request.audio_content:
                base64_audio = request.audio_content.split(',')[1]
            else:
                base64_audio = request.audio_content
            audio_bytes = base64.b64decode(base64_audio)
        except Exception as decode_error:
            logger.error(f"Base64 decoding error: {decode_error}")
            raise HTTPException(status_code=400, detail="Invalid base64 audio data")
        
        transcript = stt.transcribe_audio(audio_bytes)
        
        if transcript is None: # Check for None explicitly
            raise HTTPException(status_code=400, detail="Transcription failed")
            
        return {"transcript": transcript}
        
    except HTTPException as http_exc: # Keep specific HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.exception(f"Transcription failed in main handler: {type(e).__name__}: {e}")
        detail_msg = f"Transcription failed: {e}" if str(e) else "Transcription failed due to an internal error."
        status_code = 400 if isinstance(e, (ValueError, TypeError)) else 500
        raise HTTPException(status_code=status_code, detail=detail_msg)

@app.post("/api/evaluate")
async def evaluate_answer(request: EvaluationRequest):
    """
    Evaluate an answer and return metrics.
    """
    logger.info(f"Received evaluation request for question: {request.question[:50]}...") 
    try:
        logger.info("Calling custom_model.evaluate_answer...")
        temp_question_id = "temp_q_id" 
        temp_answer_id = "temp_a_id"   
        
        question_obj = Question(
            id=temp_question_id,
            role="unknown", 
            type="unknown", 
            difficulty="unknown", 
            content=request.question
        )
        answer_obj = Answer(
            id=temp_answer_id,
            question_id=temp_question_id,
            content=request.answer
        )
        
        metrics_obj = custom_model.evaluate_answer(question_obj, answer_obj)
        logger.info(f"Evaluation successful. Metrics object type: {type(metrics_obj)}")

        # Convert EvaluationMetrics object to dict if needed
        if isinstance(metrics_obj, dict):
            metrics = metrics_obj
        elif hasattr(metrics_obj, 'dict'):
             metrics = metrics_obj.dict()
        elif hasattr(metrics_obj, '__dict__'):
             metrics = metrics_obj.__dict__ 
        else:
             logger.warning("Metrics object is not a dict and has no dict() method. Trying direct conversion.")
             metrics = dict(metrics_obj) 

        logger.info(f"Returning metrics: {metrics}")
        return {"metrics": metrics}

    except Exception as e:
        logger.exception(f"Evaluation error occurred: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"Failed to evaluate answer: {str(e)}")

import random
import datetime

@app.post("/api/questions", response_model=Question)  
async def generate_question_endpoint(request: QuestionRequest):
    """
    Generate an interview question based on role, difficulty, and type.
    Loads questions from JSON files in organized_data/questions/.
    """
    logger.info(f"Generating question for role: {request.role}, type: {request.type}, difficulty: {request.difficulty}")
    try:
        role_dir_name = request.role.replace(" ", "_")
        role_dir_path = os.path.join(QUESTIONS_DIR, role_dir_name)
        default_dir_path = os.path.join(QUESTIONS_DIR, "Default")

        target_dir_path = role_dir_path
        if not os.path.isdir(target_dir_path):
            logger.warning(f"Role directory not found: {role_dir_path}. Trying default directory.")
            target_dir_path = default_dir_path
            if not os.path.isdir(target_dir_path):
                logger.error(f"Default question directory not found: {target_dir_path}")
                raise HTTPException(status_code=404, detail="Question directory not found for role or default.")

        all_questions_in_dir = []
        json_files = [f for f in os.listdir(target_dir_path) if f.endswith('.json')]

        if not json_files:
             logger.error(f"No JSON question files found in directory: {target_dir_path}")
             raise HTTPException(status_code=404, detail="No question files found in the directory.")

        for filename in json_files:
            filepath = os.path.join(target_dir_path, filename)
            try:
                with open(filepath, "r") as f:
                    question_data = json.load(f)
                    if isinstance(question_data, dict):
                         all_questions_in_dir.append(question_data)
                    else:
                         logger.warning(f"Unexpected data format in {filepath}, expected a single JSON object.")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {filepath}")
                continue 
            except Exception as file_error:
                 logger.error(f"Error reading file {filepath}: {file_error}")
                 continue 

        if not all_questions_in_dir:
             logger.error(f"No valid questions loaded from directory: {target_dir_path}")
             raise HTTPException(status_code=404, detail="No valid questions could be loaded.")

        filtered_questions = [
            q for q in all_questions_in_dir
            if q.get("type", "technical").lower() == request.type.lower() and \
               q.get("difficulty", "medium").lower() == request.difficulty.lower()
        ]

        selected_question = None
        if filtered_questions:
            selected_question = random.choice(filtered_questions)
            logger.info(f"Found {len(filtered_questions)} matching questions. Selected one.")
        else:
            logger.warning(f"No questions found matching criteria in {target_dir_path}. Selecting a random question from the directory.")
            selected_question = random.choice(all_questions_in_dir)

        if not selected_question:
             logger.error(f"Could not select any question from {target_dir_path}")
             raise HTTPException(status_code=500, detail="Failed to select a question.")

        content = selected_question.get("content")
        skills = selected_question.get("expected_skills", [])

        question_obj = Question(
            id=selected_question.get("id", f"q_{random.randint(1000, 9999)}"),
            role=request.role,
            type=selected_question.get("type", request.type),
            difficulty=selected_question.get("difficulty", request.difficulty),
            content=content,
            expected_skills=skills,
            created_at=selected_question.get("created_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
        )
        return question_obj

    except FileNotFoundError:
        logger.error(f"Question directory check failed unexpectedly.")
        raise HTTPException(status_code=404, detail="Question directory path error.")
    except Exception as e:
        logger.exception(f"Question generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate question: {str(e)}")


@app.post("/api/generate-feedback")
async def generate_feedback(request: FeedbackRequest):
    """
    Generate feedback based on evaluation metrics.
    """
    logger.info(f"Received feedback request for question: {request.question[:50]}...")
    logger.info(f"Metrics received: {request.metrics}")
    try:
        logger.info("Calling feedback_generator.generate_feedback...")
        try:
            metrics_for_feedback = EvaluationMetrics(**request.metrics)
        except Exception as pydantic_error:
             logger.error(f"Error creating EvaluationMetrics from request data: {pydantic_error}")
             raise HTTPException(status_code=400, detail="Invalid metrics format provided.")

        temp_question_id = "temp_q_id"
        temp_answer_id = "temp_a_id"

        question_obj = Question(
            id=temp_question_id,
            role="unknown",
            type="unknown",
            difficulty="unknown",
            content=request.question
        )
        answer_obj = Answer(
            id=temp_answer_id,
            question_id=temp_question_id,
            content=request.answer
        )

        feedback_obj = feedback_generator.generate_feedback(
            question_obj,
            answer_obj,
            metrics_for_feedback
        )
        logger.info(f"Feedback generation successful. Feedback object type: {type(feedback_obj)}")

        if isinstance(feedback_obj, dict):
            feedback = feedback_obj
        elif hasattr(feedback_obj, 'dict'):
             feedback = feedback_obj.dict()
        elif hasattr(feedback_obj, '__dict__'):
             feedback = feedback_obj.__dict__
        else:
             logger.warning("Feedback object is not a dict and has no dict() method. Trying direct conversion.")
             feedback = dict(feedback_obj) 

        logger.info(f"Returning feedback: {feedback}")
        return {"feedback": feedback}

    except Exception as e:
        logger.exception(f"Feedback generation error occurred: {str(e)}") 
        raise HTTPException(status_code=500, detail=f"Failed to generate feedback: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy", "version": settings.VERSION}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
