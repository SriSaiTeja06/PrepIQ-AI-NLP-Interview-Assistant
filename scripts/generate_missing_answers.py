import json
import os
from datetime import datetime
from src.schemas import Question, Answer, Feedback # Assuming Feedback schema exists
from src.metrics import InterviewMetrics
from src.data_generation import DataGenerator
import logging
from dataclasses import asdict, field # Import field as well

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define general answers (adjust content as needed)
GENERAL_ANSWERS = {
    "IDK": [
        "I don't know the answer to that question.",
        "I'm not sure about that specific topic.",
        "I haven't encountered that before, so I don't have an answer.",
        "That's outside my current knowledge base.",
        "I have no idea."
    ],
    "Irrelevant": [
        "That reminds me of a funny story about my cat.",
        "The weather has been really nice lately, hasn't it?",
        "Did you see the game last night?",
        "I think pineapple belongs on pizza.",
        "My favorite color is blue."
    ],
    "Gibberish": [
        "Asdf jkl; qwer poiuy.",
        "Blob blorf snicklefritz.",
        "Glibber glop snorf blarg.",
        "Wobble wibble flumph.",
        "csjaoicmjaiojakmcopamascmap,[acq[apd[qpdwkl[pqwkdo[pqkcopkokopk-0i3e28e201e9i09ed]]]]]",
        "i92i19831823",
        "Zzzzzzzzzzzzzzzzzzzzz."
    ],
    "Relevant but incorrect": [
        "To optimize Python, you should always use global variables.", 
        "The best way to handle that situation is to ignore the stakeholder.",
        "React uses a real DOM, which makes it faster.",
        "You prioritize the backlog based on alphabetical order.",
        "CloudFormation is primarily used for managing databases."
    ]
}

def generate_general_answers(answers_base_dir: str):
    """Generates and saves general category answers."""
    general_answers_dir = os.path.join(answers_base_dir, "General")
    os.makedirs(general_answers_dir, exist_ok=True)
    logger.info("Generating general category answers...")

    for category, answer_list in GENERAL_ANSWERS.items():
        category_path = os.path.join(general_answers_dir, category)
        os.makedirs(category_path, exist_ok=True)

        for i, answer_content in enumerate(answer_list):
            answer_id = f"a_General_{category}_{i}"
            answer_filepath = os.path.join(category_path, f"{answer_id}.json")

            if not os.path.exists(answer_filepath):
                logger.info(f"Generating general answer: {category} #{i}")
                answer = Answer(
                    id=answer_id,
                    question_id="general_placeholder", 
                    content=answer_content,
                    confidence=0.0, 
                    created_at=datetime.utcnow()
                )
                answer_data = {**asdict(answer), "created_at": answer.created_at.isoformat()}
                try:
                    with open(answer_filepath, 'w', encoding='utf-8') as af:
                        json.dump(answer_data, af, indent=2)
                except Exception as e_gen:
                    logger.error(f"Error saving general answer {answer_filepath}: {str(e_gen)}")
            else:
                 logger.debug(f"General answer file already exists, skipping: {answer_filepath}")


def main():
    """Generate missing answers and metrics for existing questions, including general categories."""
    logger.info("Starting answer and metrics generation...")

    data_generator = DataGenerator()
    metrics_generator = InterviewMetrics()
    questions_base_dir = "organized_data/questions"
    answers_base_dir = "organized_data/answers"

    # Generate General Answers first
    generate_general_answers(answers_base_dir)

    # Now process role-specific questions
    for role_dir in os.listdir(questions_base_dir):
        role_questions_path = os.path.join(questions_base_dir, role_dir)
        role_answers_path = os.path.join(answers_base_dir, role_dir) 

        if not os.path.isdir(role_questions_path):
            continue

        role = role_dir.replace("_", " ")
        logger.info(f"Processing role: {role}")

        # Ensure the base answer directory for the role exists
        os.makedirs(role_answers_path, exist_ok=True)

        # Iterate through question files in the role directory
        for filename in os.listdir(role_questions_path):
            if filename.startswith("q_") and filename.endswith(".json"):
                question_file_path = os.path.join(role_questions_path, filename)
                try:
                    with open(question_file_path, 'r', encoding='utf-8') as f: 
                        question_data = json.load(f)
                    if not all(k in question_data for k in ["id", "role", "type", "difficulty", "content", "expected_skills"]):
                         logger.error(f"Skipping invalid question file {question_file_path}: Missing required fields.")
                         continue
                    question = Question(**question_data)

                    question_answer_dir = os.path.join(role_answers_path, question.id) # Dir for this specific question's answers
                    os.makedirs(question_answer_dir, exist_ok=True)

                    for quality in ["high", "medium", "low"]:
                        answer_id = f"a_{question.id}_{quality}"
                        answer_filename = f"{answer_id}.json"
                        answer_filepath = os.path.join(question_answer_dir, answer_filename)

                        # Check if answer file already exists
                        if not os.path.exists(answer_filepath):
                            logger.info(f"Generating answer for question {question.id} with quality {quality}")
                            try:
                                answer = data_generator.generate_sample_answer(question, quality)

                                # Save the individual answer file
                                answer_data = {**asdict(answer), "created_at": answer.created_at.isoformat()}
                                with open(answer_filepath, 'w', encoding='utf-8') as af: # Added encoding
                                    json.dump(answer_data, af, indent=2)

                            except Exception as e_gen:
                                logger.error(f"Error generating/saving answer/feedback for {question.id} ({quality}): {str(e_gen)}")


                except FileNotFoundError:
                    logger.warning(f"Question file not found: {question_file_path}")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from file: {question_file_path}")
                except TypeError as te:
                     logger.error(f"TypeError initializing Question from file {question_file_path}: {str(te)}. Data: {question_data}")
                except Exception as e:
                    logger.error(f"Error processing question file {question_file_path}: {str(e)}")

    logger.info("Answer and metrics generation completed!")

if __name__ == "__main__":  
    main()