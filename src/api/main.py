from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import sys
import os
from datetime import datetime

from src.integration.pipeline import InterviewPipeline
from src.schemas import Question, Answer, Feedback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-NLP Interview Assistant API",
    description="API for the AI-NLP Interview Assistant system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize the interview pipeline
pipeline = InterviewPipeline(
    model_type="transformer",
    evaluator_type="custom",
    use_speech_to_text=False  
)

# API Input/Output Models
class QuestionRequest(BaseModel):
    role: str
    difficulty: str = "medium"
    type: str = "technical"

class AnswerRequest(BaseModel):
    question_id: str
    content: str

class EvaluationRequest(BaseModel):
    question_id: str
    answer_id: str

class FeedbackRequest(BaseModel):
    question_id: str
    answer_id: str

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to the AI-NLP Interview Assistant API",
        "version": "1.0.0"
    }

@app.get("/roles")
async def get_roles():
    """
    Get list of available roles.
    
    Returns:
        List[str]: List of available roles
    """
    try:
        roles = pipeline.get_available_roles()
        return roles
    except Exception as e:
        logger.error(f"Error getting roles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/questions")
async def generate_question(request: QuestionRequest):
    """
    Generate a question for a specific role.
    
    Args:
        request (QuestionRequest): The question request
        
    Returns:
        Question: The generated question
    """
    try:
        question = pipeline.generate_question(
            role=request.role,
            difficulty=request.difficulty,
            question_type=request.type
        )
        return question
    except Exception as e:
        logger.error(f"Error generating question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answers")
async def generate_answer(request: AnswerRequest):
    """
    Generate an answer to a question.
    
    Args:
        request (AnswerRequest): The answer request
        
    Returns:
        Answer: The generated answer
    """
    try:
        question = Question(
            id=request.question_id,
            role="General",
            type="general",
            difficulty="medium",
            content=request.content,
            expected_skills=[],
            created_at=datetime.utcnow()
        )
        
        answer = pipeline.generate_answer(question)
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate")
async def evaluate_answer(question: Question, answer: Answer):
    """
    Evaluate an answer.
    
    Args:
        question (Question): The original question
        answer (Answer): The answer to evaluate
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    try:
        metrics = pipeline.evaluate_answer(question, answer)
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def generate_feedback(question: Question, answer: Answer):
    """
    Generate feedback for an answer.
    
    Args:
        question (Question): The original question
        answer (Answer): The answer to provide feedback on
        
    Returns:
        Feedback: Generated feedback
    """
    try:
        feedback = pipeline.generate_feedback(question, answer)
        return feedback
    except Exception as e:
        logger.error(f"Error generating feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_interview_question(
    question: Question,
    answer_text: Optional[str] = None,
    generate_model_answer: bool = False
):
    """
    Process a complete interview question workflow.
    
    Args:
        question (Question): The question
        answer_text (str, optional): The answer text
        generate_model_answer (bool): Whether to generate a model answer
        
    Returns:
        Dict[str, Any]: Question, answer, metrics, and feedback
    """
    try:
        result = pipeline.process_interview_question(
            question=question,
            answer_text=answer_text,
            generate_model_answer=generate_model_answer
        )
        return result
    except Exception as e:
        logger.error(f"Error processing interview question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-audio")
async def upload_audio(
    file: UploadFile = File(...),
    question_id: str = Form(...)
):
    """
    Upload audio for speech-to-text processing.
    
    Args:
        file (UploadFile): The audio file
        question_id (str): ID of the associated question
        
    Returns:
        Dict[str, Any]: Question, answer, metrics, and feedback
    """
    if not pipeline.use_speech_to_text:
        raise HTTPException(status_code=400, detail="Speech-to-text functionality is not enabled")
    
    try:
        # Save the uploaded file
        file_location = f"uploads/{file.filename}"
        os.makedirs("uploads", exist_ok=True)
        
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        # Create a dummy question
        question = Question(
            id=question_id,
            role="General",
            type="general",
            difficulty="medium",
            content="Placeholder question",
            expected_skills=[],
            created_at=datetime.utcnow()
        )
        
        # Process the audio
        result = pipeline.process_interview_question(
            question=question,
            audio_path=file_location
        )
        
        return result
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
