"""
Evaluation script for the Custom Answer Evaluator models.
This script evaluates both role-specific and cross-role models on test data.
"""

import os
import sys
import json
import torch
import argparse
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.custom_evaluator import CustomEvaluatorModel
from src.schemas import Question, Answer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_test_data(test_dir, role=None):
    """
    Load test data for evaluation.
    
    Args:
        test_dir: Directory containing test data
        role: Optional role to filter by
        
    Returns:
        List of test examples
    """
    logger.info(f"Loading test data from {test_dir}")
    
    test_examples = []
    roles_dir = os.path.join(test_dir, "roles")
    
    # Get roles
    roles = []
    for role_file in os.listdir(roles_dir):
        if role_file.endswith(".json"):
            current_role = role_file.replace(".json", "")
            if role is None or current_role == role:
                roles.append(current_role)
    
    # Load data for each role
    for current_role in roles:
        questions_dir = os.path.join(test_dir, "questions", current_role)
        answers_dir = os.path.join(test_dir, "answers", current_role)
        
        # Load questions
        questions = []
        for q_file in os.listdir(questions_dir):
            if q_file.endswith(".json"):
                q_path = os.path.join(questions_dir, q_file)
                with open(q_path, 'r', encoding='utf-8') as f:
                    question_data = json.load(f)
                    questions.append(Question(**question_data))
        
        # Load answers and ground truth evaluations
        for question in questions:
            q_answers_dir = os.path.join(answers_dir, question.id)
            if os.path.exists(q_answers_dir):
                for a_file in os.listdir(q_answers_dir):
                    if a_file.endswith(".json") and not a_file.endswith("_metrics.json"):
                        a_path = os.path.join(q_answers_dir, a_file)
                        with open(a_path, 'r', encoding='utf-8') as f:
                            answer_data = json.load(f)
                            answer = Answer(**answer_data)
                            
                            # Check if metrics exist
                            metrics_file = a_file.replace(".json", "_metrics.json")
                            metrics_path = os.path.join(q_answers_dir, metrics_file)
                            
                            if os.path.exists(metrics_path):
                                with open(metrics_path, 'r', encoding='utf-8') as mf:
                                    metrics_data = json.load(mf)
                                    
                                    test_examples.append({
                                        "question": question,
                                        "answer": answer,
                                        "metrics": metrics_data,
                                        "role": current_role
                                    })
    
    logger.info(f"Loaded {len(test_examples)} test examples")
    return test_examples

def evaluate_model(model, test_examples, threshold=0.5):
    """
    Evaluate a model on test examples.
    
    Args:
        model: The model to evaluate
        test_examples: List of test examples
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info(f"Evaluating model on {len(test_examples)} examples")
    
    results = {
        "predicted_metrics": [],
        "ground_truth_metrics": [],
        "binary_predictions": [],
        "binary_ground_truth": [],
        "by_role": {}
    }
    
    for example in tqdm(test_examples):
        question = example["question"]
        answer = example["answer"]
        ground_truth = example["metrics"]
        role = example["role"]
        
        # Get model predictions
        predicted_metrics = model.evaluate_answer(question, answer)
        
        # Convert to dictionary
        predicted_dict = {
            "technical_accuracy": predicted_metrics.technical_accuracy,
            "completeness": predicted_metrics.completeness,
            "clarity": predicted_metrics.clarity,
            "relevance": predicted_metrics.relevance,
            "overall_score": predicted_metrics.overall_score
        }
        
        # Store predictions and ground truth
        results["predicted_metrics"].append(predicted_dict)
        results["ground_truth_metrics"].append(ground_truth)
        
        # Binary classification (is the answer good or not)
        binary_pred = 1 if predicted_dict["overall_score"] >= threshold else 0
        binary_gt = 1 if ground_truth["overall_score"] >= threshold else 0
        
        results["binary_predictions"].append(binary_pred)
        results["binary_ground_truth"].append(binary_gt)
        
        # Store by role
        if role not in results["by_role"]:
            results["by_role"][role] = {
                "predicted_metrics": [],
                "ground_truth_metrics": [],
                "binary_predictions": [],
                "binary_ground_truth": []
            }
        
        results["by_role"][role]["predicted_metrics"].append(predicted_dict)
        results["by_role"][role]["ground_truth_metrics"].append(ground_truth)
        results["by_role"][role]["binary_predictions"].append(binary_pred)
        results["by_role"][role]["binary_ground_truth"].append(binary_gt)
    
    return results

def compute_metrics(evaluation_results):
    """
    Compute metrics from evaluation results.
    
    Args:
        evaluation_results: Results from model evaluation
        
    Returns:
        Dictionary of computed metrics
    """
    logger.info("Computing metrics from evaluation results")
    
    metrics = {}
    
    # Binary classification metrics
    y_true = evaluation_results["binary_ground_truth"]
    y_pred = evaluation_results["binary_predictions"]
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics["classification_report"] = report
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Regression metrics for each evaluation dimension
    regression_metrics = {}
    
    for metric_name in ["technical_accuracy", "completeness", "clarity", "relevance", "overall_score"]:
        y_true = [example[metric_name] for example in evaluation_results["ground_truth_metrics"]]
        y_pred = [example[metric_name] for example in evaluation_results["predicted_metrics"]]
        
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        rmse = np.sqrt(np.mean(np.square(np.array(y_true) - np.array(y_pred))))
        
        regression_metrics[metric_name] = {
            "mae": mae,
            "rmse": rmse
        }
    
    metrics["regression_metrics"] = regression_metrics
    
    # Metrics by role
    role_metrics = {}
    
    for role, role_results in evaluation_results["by_role"].items():
        role_y_true = role_results["binary_ground_truth"]
        role_y_pred = role_results["binary_predictions"]
        
        if len(role_y_true) > 0:
            report = classification_report(role_y_true, role_y_pred, output_dict=True)
            
            role_metrics[role] = {
                "classification_report": report,
                "sample_count": len(role_y_true)
            }
    
    metrics["by_role"] = role_metrics
    
    return metrics

def generate_plots(evaluation_results, metrics, output_dir):
    """
    Generate evaluation plots.
    
    Args:
        evaluation_results: Results from model evaluation
        metrics: Computed metrics
        output_dir: Directory to save plots
    """
    logger.info("Generating evaluation plots")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    cm = np.array(metrics["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    
    # Metric comparison
    for metric_name in ["technical_accuracy", "completeness", "clarity", "relevance", "overall_score"]:
        plt.figure(figsize=(10, 6))
        
        y_true = [example[metric_name] for example in evaluation_results["ground_truth_metrics"]]
        y_pred = [example[metric_name] for example in evaluation_results["predicted_metrics"]]
        
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.title(f'{metric_name.replace("_", " ").title()} - Predicted vs Ground Truth')
        plt.xlabel('Ground Truth')
        plt.ylabel('Predicted')
        plt.savefig(os.path.join(output_dir, f"{metric_name}_comparison.png"))
        plt.close()
    
    # Performance by role
    role_accuracy = {}
    role_sample_count = {}
    
    for role, role_metrics in metrics["by_role"].items():
        if "classification_report" in role_metrics:
            role_accuracy[role] = role_metrics["classification_report"]["accuracy"]
            role_sample_count[role] = role_metrics["sample_count"]
    
    if role_accuracy:
        plt.figure(figsize=(12, 6))
        
        roles = list(role_accuracy.keys())
        accuracies = list(role_accuracy.values())
        
        # Sort by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        roles = [roles[i] for i in sorted_indices]
        accuracies = [accuracies[i] for i in sorted_indices]
        
        plt.bar(roles, accuracies)
        plt.title('Model Accuracy by Role')
        plt.xlabel('Role')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "accuracy_by_role.png"))
        plt.close()

def evaluate_all_models(args):
    """
    Evaluate all available models.
    
    Args:
        args: Command line arguments
    """
    logger.info("Evaluating all models...")
    
    # Load test data
    test_examples = load_test_data(args.test_dir)
    
    # Find available models
    models_to_evaluate = []
    
    # Add final model if available
    final_model_path = os.path.join(args.models_dir, "final_model", "evaluator_model.pt")
    if os.path.exists(final_model_path):
        models_to_evaluate.append({
            "name": "final_model",
            "path": final_model_path
        })
    
    # Add cross-role models if available
    cross_role_models_path = os.path.join(args.models_dir, "cross_role_models.json")
    if os.path.exists(cross_role_models_path):
        with open(cross_role_models_path, 'r') as f:
            cross_role_models = json.load(f)
            
        for name, path in cross_role_models.items():
            if os.path.exists(path):
                models_to_evaluate.append({
                    "name": f"cross_role_{name}",
                    "path": path
                })
    
    # Add role-specific models if available
    role_models_path = os.path.join(args.models_dir, "role_models.json")
    if os.path.exists(role_models_path):
        with open(role_models_path, 'r') as f:
            role_models = json.load(f)
            
        for role, path in role_models.items():
            if os.path.exists(path):
                models_to_evaluate.append({
                    "name": f"role_specific_{role}",
                    "path": path
                })
    
    logger.info(f"Found {len(models_to_evaluate)} models to evaluate")
    
    # Evaluate each model
    results = {}
    
    for model_info in models_to_evaluate:
        name = model_info["name"]
        path = model_info["path"]
        
        logger.info(f"Evaluating model: {name}")
        
        # Load model
        model = CustomEvaluatorModel(model_path=path)
        
        # Evaluate
        evaluation_results = evaluate_model(model, test_examples)
        
        # Compute metrics
        metrics = compute_metrics(evaluation_results)
        
        # Generate plots
        plots_dir = os.path.join(args.output_dir, name)
        generate_plots(evaluation_results, metrics, plots_dir)
        
        # Store results
        results[name] = {
            "metrics": metrics,
            "model_path": path
        }
    
    # Save all results
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {results_path}")
    
    # Create comparison summary
    create_comparison_summary(results, args.output_dir)

def create_comparison_summary(results, output_dir):
    """
    Create a summary comparing all evaluated models.
    
    Args:
        results: Evaluation results for all models
        output_dir: Directory to save summary
    """
    logger.info("Creating model comparison summary")
    
    # Extract key metrics for comparison
    comparison_data = []
    
    for model_name, model_results in results.items():
        metrics = model_results["metrics"]
        
        # Binary classification performance
        binary_metrics = metrics["classification_report"]["weighted avg"]
        
        # Regression performance
        regression_metrics = metrics["regression_metrics"]["overall_score"]
        
        comparison_data.append({
            "Model": model_name,
            "Accuracy": metrics["classification_report"]["accuracy"],
            "Precision": binary_metrics["precision"],
            "Recall": binary_metrics["recall"],
            "F1 Score": binary_metrics["f1-score"],
            "MAE": regression_metrics["mae"],
            "RMSE": regression_metrics["rmse"]
        })
    
    # Sort by F1 score
    comparison_data = sorted(comparison_data, key=lambda x: x["F1 Score"], reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate comparison plot
    plt.figure(figsize=(12, 8))
    
    models = [d["Model"] for d in comparison_data]
    f1_scores = [d["F1 Score"] for d in comparison_data]
    
    plt.bar(models, f1_scores)
    plt.title('Model Comparison - F1 Score')
    plt.xlabel('Model')
    plt.ylabel('F1 Score')
    plt.ylim([0, 1])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path)
    plt.close()
    
    logger.info(f"Comparison summary saved to: {csv_path}")
    logger.info(f"Comparison plot saved to: {plot_path}")
    
    # Print top 3 models
    logger.info("Top 3 performing models:")
    for i, model in enumerate(comparison_data[:3]):
        logger.info(f"{i+1}. {model['Model']} - Accuracy: {model['Accuracy']:.4f}, F1: {model['F1 Score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate custom evaluator models")
    
    parser.add_argument("--test_dir", type=str, required=True, help="Directory containing test data")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing models")
    parser.add_argument("--output_dir", type=str, default="evaluation", help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Evaluate all models
    evaluate_all_models(args)
