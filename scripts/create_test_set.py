"""
Script to create a test set by splitting the organized data.
"""
import os
import json
import random
import shutil
from datetime import datetime

def create_test_set(source_dir, test_dir, test_ratio=0.2):
    """
    Create a test set by splitting the organized data.
    
    Args:
        source_dir: Directory containing the organized data
        test_dir: Directory where the test set will be placed
        test_ratio: Ratio of data to use for testing
    """
    print(f"Creating test set from {source_dir} with test ratio {test_ratio}")
    
    # Create test directory structure
    os.makedirs(os.path.join(test_dir, "roles"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "questions"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "answers"), exist_ok=True)
    
    # Get all roles
    roles_dir = os.path.join(source_dir, "roles")
    roles = []
    
    for role_file in os.listdir(roles_dir):
        if role_file.endswith(".json"):
            role = role_file.replace(".json", "")
            roles.append(role)
            
            src_role_file = os.path.join(roles_dir, role_file)
            dst_role_file = os.path.join(test_dir, "roles", role_file)
            shutil.copy2(src_role_file, dst_role_file)
    
    print(f"Found {len(roles)} roles")
    
    # Process each role
    for role in roles:
        os.makedirs(os.path.join(test_dir, "questions", role), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "answers", role), exist_ok=True)
        
        questions_dir = os.path.join(source_dir, "questions", role)
        if not os.path.exists(questions_dir):
            print(f"No questions directory found for role: {role}")
            continue
            
        question_files = [f for f in os.listdir(questions_dir) if f.endswith(".json")]
        
        num_test_questions = max(1, int(len(question_files) * test_ratio))
        num_test_questions = min(num_test_questions, len(question_files))
        test_question_files = random.sample(question_files, num_test_questions)
        
        print(f"Selected {len(test_question_files)} test questions for role: {role}")
        
        # Process each test question
        for question_file in test_question_files:
            question_id = question_file.replace(".json", "")
            
            src_question_file = os.path.join(questions_dir, question_file)
            dst_question_file = os.path.join(test_dir, "questions", role, question_file)
            shutil.copy2(src_question_file, dst_question_file)
            
            answers_dir = os.path.join(source_dir, "answers", role, question_id)
            if not os.path.exists(answers_dir):
                print(f"No answers found for question: {question_id}")
                continue
                
            os.makedirs(os.path.join(test_dir, "answers", role, question_id), exist_ok=True)
            
            for answer_file in os.listdir(answers_dir):
                src_answer_file = os.path.join(answers_dir, answer_file)
                dst_answer_file = os.path.join(test_dir, "answers", role, question_id, answer_file)
                shutil.copy2(src_answer_file, dst_answer_file)
                
                if not answer_file.endswith("_metrics.json"):
                    with open(src_answer_file, 'r', encoding='utf-8') as f:
                        answer = json.load(f)
                    
                    answer["in_test_set"] = True
                    
                    with open(src_answer_file, 'w', encoding='utf-8') as f:
                        json.dump(answer, f, indent=2)
    
    print(f"Test set creation complete. Test set is in {test_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        source_dir = sys.argv[1]
        test_dir = sys.argv[2]
    else:
        # Default paths
        source_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "organized_data")
        test_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "test_data")
    
    create_test_set(source_dir, test_dir)
