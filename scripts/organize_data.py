"""
Script to reorganize data files into the expected structure for training.
"""
import os
import json
import shutil
from datetime import datetime

def organize_data(source_dir, target_dir):
    """
    Organize data files into the expected directory structure.
    
    Args:
        source_dir: Directory containing the original data files
        target_dir: Directory where the organized data will be placed
    """
    print(f"Organizing data from {source_dir} to {target_dir}")
    
    # Create target directory structure
    os.makedirs(os.path.join(target_dir, "roles"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "questions"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "answers"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "feedback"), exist_ok=True)
    
    # Get all files in source directory
    files = os.listdir(source_dir)
    
    # Process roles and job descriptions
    roles = set()
    for file in files:
        if file.endswith("_job_desc.txt"):
            role = file.replace("_job_desc.txt", "")
            role = role.replace("_", " ")  
            roles.add(role)
            
            # Read job description
            with open(os.path.join(source_dir, file), 'r', encoding='utf-8') as f:
                job_desc = f.read()
            
            # Create role JSON file
            role_data = {
                "name": role,
                "description": job_desc,
                "required_skills": [],  
                "created_at": datetime.now().isoformat()
            }
            
            # Write role file
            role_file = os.path.join(target_dir, "roles", f"{role.replace(' ', '_')}.json")
            with open(role_file, 'w', encoding='utf-8') as f:
                json.dump(role_data, f, indent=2)
    
    # Process questions
    for role in roles:
        question_files = [f for f in files if f.startswith(role.replace(" ", "_")) and f.endswith("_questions.json")]
        questions_dir = os.path.join(target_dir, "questions", role.replace(" ", "_"))
        os.makedirs(questions_dir, exist_ok=True)
        
        for question_file in question_files:
            with open(os.path.join(source_dir, question_file), 'r', encoding='utf-8') as f:
                questions = json.load(f)
            
            for i, question in enumerate(questions):
                if "id" not in question:
                    question["id"] = f"q_{role.replace(' ', '_')}_{i+1}"
                
                question_path = os.path.join(questions_dir, f"{question['id']}.json")
                with open(question_path, 'w', encoding='utf-8') as f:
                    json.dump(question, f, indent=2)
    
    # Process answers
    for role in roles:
        answer_files = [f for f in files if f.startswith(role.replace(" ", "_")) and f.endswith("_answers.json")]
        
        for answer_file in answer_files:
            with open(os.path.join(source_dir, answer_file), 'r', encoding='utf-8') as f:
                answers = json.load(f)
            
            for answer in answers:
                if "id" not in answer:
                    answer["id"] = f"a_{answer.get('question_id', 'unknown')}"
                
                if "question_id" in answer:
                    question_id = answer["question_id"]
                    answer_dir = os.path.join(target_dir, "answers", role.replace(" ", "_"), question_id)
                    os.makedirs(answer_dir, exist_ok=True)
                    
                    answer_path = os.path.join(answer_dir, f"{answer['id']}.json")
                    with open(answer_path, 'w', encoding='utf-8') as f:
                        json.dump(answer, f, indent=2)
    
    # Process feedback
    for role in roles:
        feedback_files = [f for f in files if f.startswith(role.replace(" ", "_")) and f.endswith("_feedback.json")]
        
        for feedback_file in feedback_files:
            with open(os.path.join(source_dir, feedback_file), 'r', encoding='utf-8') as f:
                feedbacks = json.load(f)
            
            for feedback in feedbacks:
                if "id" not in feedback:
                    feedback["id"] = f"f_{feedback.get('answer_id', 'unknown')}"
                
                if "answer_id" in feedback and "metrics" in feedback:
                    answer_id = feedback["answer_id"]
                    parts = answer_id.split("_")
                    if len(parts) >= 3:
                        question_id = "_".join(parts[1:])
                        
                        answer_dir = os.path.join(target_dir, "answers", role.replace(" ", "_"), question_id)
                        if os.path.exists(answer_dir):
                            metrics_path = os.path.join(answer_dir, f"{answer_id}_metrics.json")
                            with open(metrics_path, 'w', encoding='utf-8') as f:
                                json.dump(feedback["metrics"], f, indent=2)
                            
                            feedback_dir = os.path.join(target_dir, "feedback", role.replace(" ", "_"), question_id)
                            os.makedirs(feedback_dir, exist_ok=True)
                            
                            feedback_path = os.path.join(feedback_dir, f"{feedback['id']}.json")
                            with open(feedback_path, 'w', encoding='utf-8') as f:
                                json.dump(feedback, f, indent=2)
    
    print(f"Data organization complete. Organized data is in {target_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        source_dir = sys.argv[1]
        target_dir = sys.argv[2]
    else:
        source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        target_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "organized_data")
    
    organize_data(source_dir, target_dir)
