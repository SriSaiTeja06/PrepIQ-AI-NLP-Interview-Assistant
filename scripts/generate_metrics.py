"""
Script to generate synthetic metrics for answer evaluation.
"""
import os
import json
import random
from datetime import datetime

def generate_metrics(data_dir):
    """
    Generate synthetic metrics for answers in the organized data.
    
    Args:
        data_dir: Directory containing the organized data
    """
    print(f"Generating metrics for answers in {data_dir}")
    
    # Get all roles
    roles_dir = os.path.join(data_dir, "roles")
    roles = []
    
    for role_file in os.listdir(roles_dir):
        if role_file.endswith(".json"):
            role = role_file.replace(".json", "")
            roles.append(role)
    
    print(f"Found {len(roles)} roles")
    
    # Process each role
    for role in roles:
        role_answers_dir = os.path.join(data_dir, "answers", role)
        if not os.path.exists(role_answers_dir):
            print(f"No answers directory found for role: {role}")
            continue
        
        # Process each question
        for question_id in os.listdir(role_answers_dir):
            question_answers_dir = os.path.join(role_answers_dir, question_id)
            if not os.path.isdir(question_answers_dir):
                continue
            
            # Get all answer files
            answer_files = [f for f in os.listdir(question_answers_dir) if f.endswith(".json") and not f.endswith("_metrics.json")]
            
            for answer_file in answer_files:
                answer_path = os.path.join(question_answers_dir, answer_file)
                metrics_path = os.path.join(question_answers_dir, answer_file.replace(".json", "_metrics.json"))
                
                # Skip if metrics already exist
                if os.path.exists(metrics_path):
                    continue
                
                # Load answer
                with open(answer_path, 'r', encoding='utf-8') as f:
                    answer = json.load(f)
                
                # Generate synthetic metrics based on answer quality
                quality_hint = None
                if "_high" in answer_file:
                    quality_hint = "high"
                elif "_medium" in answer_file:
                    quality_hint = "medium"
                elif "_low" in answer_file:
                    quality_hint = "low"
                
                metrics = generate_quality_metrics(quality_hint)
                
                # Save metrics file
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2)
    
    print(f"Metrics generation complete")

def generate_quality_metrics(quality_hint=None):
    """
    Generate synthetic metrics based on quality hint.
    
    Args:
        quality_hint: Hint about expected quality (high, medium, low)
        
    Returns:
        Dictionary of metrics
    """
    if quality_hint == "high":
        # High quality answers get good scores
        technical_accuracy = random.uniform(0.8, 0.95)
        completeness = random.uniform(0.8, 0.95)
        clarity = random.uniform(0.8, 0.95)
        relevance = random.uniform(0.8, 0.95)
    elif quality_hint == "medium":
        # Medium quality answers get moderate scores
        technical_accuracy = random.uniform(0.6, 0.8)
        completeness = random.uniform(0.6, 0.8)
        clarity = random.uniform(0.6, 0.8)
        relevance = random.uniform(0.6, 0.8)
    elif quality_hint == "low":
        # Low quality answers get poor scores
        technical_accuracy = random.uniform(0.3, 0.6)
        completeness = random.uniform(0.3, 0.6)
        clarity = random.uniform(0.3, 0.6)
        relevance = random.uniform(0.3, 0.6)
    else:
        # Default random metrics
        technical_accuracy = random.uniform(0.3, 0.95)
        completeness = random.uniform(0.3, 0.95)
        clarity = random.uniform(0.3, 0.95)
        relevance = random.uniform(0.3, 0.95)
    
    # Calculate overall score as average
    overall_score = (technical_accuracy + completeness + clarity + relevance) / 4
    
    return {
        "technical_accuracy": round(technical_accuracy, 2),
        "completeness": round(completeness, 2),
        "clarity": round(clarity, 2),
        "relevance": round(relevance, 2),
        "overall_score": round(overall_score, 2)
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        # Default path
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "organized_data")
    
    generate_metrics(data_dir)
