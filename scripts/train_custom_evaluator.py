"""
Training script for the Custom Answer Evaluator model.
Supports loading a checkpoint to continue training.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import logging
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.models.custom_evaluator import CustomEvaluator
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

class EvaluatorDataset(Dataset):
    """
    Dataset class for training the custom evaluator model.
    """
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load and process the data
        self._load_data(data_dir)
        
    def _load_data(self, data_dir):
        """
        Load questions, answers, and metrics from JSON files.
        Handles potential JSON errors gracefully and ignores 'in_test_set' field.
        Includes general category answers.
        """
        logger.info(f"Loading data from directory: {data_dir}")
        roles_dir = os.path.join(data_dir, "roles") 
        questions_base_dir = os.path.join(data_dir, "questions")
        answers_base_dir = os.path.join(data_dir, "answers")

        all_questions = {} 

        processed_roles = 0
        for role_dir in os.listdir(questions_base_dir):
            role_questions_path = os.path.join(questions_base_dir, role_dir)
            role_answers_path = os.path.join(answers_base_dir, role_dir)

            if not os.path.isdir(role_questions_path):
                continue

            role = role_dir.replace("_", " ")
            logger.debug(f"Processing role: {role}")

            # Load questions for the role
            questions_in_role = {}
            for q_file in os.listdir(role_questions_path):
                if q_file.endswith(".json"):
                    q_path = os.path.join(role_questions_path, q_file)
                    try:
                        with open(q_path, 'r', encoding='utf-8') as fq:
                            question_data = json.load(fq)
                            if all(k in question_data for k in ['id', 'role', 'type', 'difficulty', 'content']):
                                question = Question(**question_data)
                                questions_in_role[question.id] = question
                                all_questions[question.id] = question # Add to global dict
                            else:
                                logger.warning(f"Skipping question file due to missing required fields: {q_path}")
                    except Exception as e:
                        logger.warning(f"Skipping question file due to error ({type(e).__name__}): {q_path} - {e}")

            # Load answers and metrics for each question in this role
            if os.path.exists(role_answers_path):
                for question_id, question in questions_in_role.items():
                    q_answers_dir = os.path.join(role_answers_path, question_id)
                    if os.path.exists(q_answers_dir):
                        for a_file in os.listdir(q_answers_dir):
                            if a_file.endswith(".json") and not a_file.endswith("_metrics.json"):
                                a_path = os.path.join(q_answers_dir, a_file)
                                metrics_file = a_file.replace(".json", "_metrics.json")
                                metrics_path = os.path.join(q_answers_dir, metrics_file)

                                if not os.path.exists(metrics_path):
                                    logger.warning(f"Metrics file not found for answer: {a_path}. Skipping.")
                                    continue

                                try:
                                    with open(a_path, 'r', encoding='utf-8') as fa:
                                        answer_data = json.load(fa)
                                        answer_data.pop('in_test_set', None)
                                        if not all(k in answer_data for k in ['id', 'question_id', 'content']):
                                            logger.warning(f"Skipping answer file due to missing required fields: {a_path}")
                                            continue
                                        answer = Answer(**answer_data)

                                    with open(metrics_path, 'r', encoding='utf-8') as fm:
                                        metrics_data = json.load(fm)
                                        if not isinstance(metrics_data, dict):
                                            logger.warning(f"Skipping invalid metrics data (not a dict) in {metrics_path}")
                                            continue
                                        if all(k in metrics_data for k in ['technical_accuracy', 'completeness', 'clarity', 'relevance']):
                                            metrics = EvaluationMetrics(**metrics_data)
                                        else:
                                            logger.warning(f"Skipping metrics file due to missing required fields: {metrics_path}")
                                            continue

                                    self.examples.append({
                                        "question": question,
                                        "answer": answer,
                                        "metrics": metrics
                                    })
                                except Exception as e:
                                    logger.warning(f"Skipping answer/metrics pair due to error ({type(e).__name__}) for files: {a_path}, {metrics_path} - {e}")
            processed_roles += 1

        # --- Start: Load General Answers ---
        general_answers_dir = os.path.join(answers_base_dir, "General")
        if os.path.exists(general_answers_dir) and os.path.isdir(general_answers_dir) and all_questions:
            logger.info("Loading general category answers...")
            for category in os.listdir(general_answers_dir):
                category_path = os.path.join(general_answers_dir, category)
                if os.path.isdir(category_path):
                    for filename in os.listdir(category_path):
                        if filename.endswith(".json"):
                            answer_filepath = os.path.join(category_path, filename)
                            try:
                                with open(answer_filepath, 'r', encoding='utf-8') as f:
                                    answer_data = json.load(f)
                                answer = Answer(**answer_data)

                                # Pair general answers with multiple random questions
                                num_pairings = 5 
                                random_question_ids = random.sample(list(all_questions.keys()), min(num_pairings, len(all_questions)))

                                for q_id in random_question_ids:
                                    question = all_questions[q_id]
                                    # Assign low target metrics for general answers
                                    low_metrics = EvaluationMetrics(
                                        technical_accuracy=0.1,
                                        completeness=0.1,
                                        clarity=0.5, 
                                        relevance=0.0
                                    )
                                    # Adjust metrics based on category if needed
                                    if category == "Relevant but incorrect":
                                         low_metrics.relevance = 0.6 
                                         low_metrics.technical_accuracy = 0.0
                                    elif category == "IDK":
                                         low_metrics.completeness = 0.0

                                    self.examples.append({
                                        "question": question,
                                        "answer": answer,
                                        "metrics": low_metrics 
                                    })
                            except Exception as e:
                                logger.error(f"Error loading or pairing general answer {answer_filepath}: {str(e)}")
        elif not all_questions:
             logger.warning("No role-specific questions loaded, cannot pair general answers.")
        else:
            logger.warning("General answers directory not found or empty.")
        

        if processed_roles == 0 and not self.examples: 
             logger.error("No roles processed and no general answers loaded. Check data directory structure and content.")
        logger.info(f"Loaded {len(self.examples)} total evaluation examples.")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        question = example["question"]
        answer = example["answer"]
        metrics = example["metrics"]
        
        # Prepare input text
        input_text = f"Question: {question.content}\nRole: {question.role}\nDifficulty: {question.difficulty}\nAnswer: {answer.content}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Get target metrics
        target_metrics = torch.tensor([
            metrics.technical_accuracy,
            metrics.completeness,
            metrics.clarity,
            metrics.relevance
        ], dtype=torch.float32)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "target_metrics": target_metrics
        }

def train_evaluator(args):
    """
    Train the custom evaluator model.
    
    Args:
        args: Command line arguments
    """
    logger.info(f"Training custom evaluator model with args: {args}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    
    # Create datasets and dataloaders
    full_dataset = EvaluatorDataset(args.data_dir, tokenizer, args.max_length) 
    
    if len(full_dataset) == 0:
        logger.error("Dataset is empty. Cannot train. Check data loading logs.")
        return None

    # Split dataset into train and validation
    if len(full_dataset) < 2:
         logger.warning("Dataset too small for train/validation split. Using all data for training.")
         train_dataset = full_dataset
         val_dataset = None 
         train_size = len(full_dataset)
         val_size = 0
    else:
        train_size = int(len(full_dataset) * 0.9)
        val_size = len(full_dataset) - train_size
        if train_size == len(full_dataset):
             train_size -= 1
             val_size = 1
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    logger.info(f"Dataset split: Train={train_size}, Validation={val_size}")

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
    
    # Initialize model structure
    model = CustomEvaluator(
        pretrained_model_name=args.pretrained_model,
        hidden_dim=args.hidden_dim
    )

    # --- Load checkpoint if specified ---
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        try:
            logger.info(f"Loading model weights from checkpoint: {args.load_checkpoint}")
            # Load state dict, ensuring it maps to the correct device
            model.load_state_dict(torch.load(args.load_checkpoint, map_location=device))
            logger.info("Successfully loaded weights from checkpoint.")
        except Exception as e:
            logger.error(f"Failed to load checkpoint {args.load_checkpoint}: {e}. Starting training from scratch.")
            # Re-initialize model to ensure clean state if loading failed
            model = CustomEvaluator(
                pretrained_model_name=args.pretrained_model,
                hidden_dim=args.hidden_dim
            )
    elif args.load_checkpoint:
        logger.warning(f"Checkpoint file not found: {args.load_checkpoint}. Starting training from scratch.")
    else:
        logger.info("No checkpoint specified. Starting training from scratch.")
    
    # Move model to device AFTER potentially loading weights
    model.to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = None
    
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_metrics = batch["target_metrics"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            
            loss = 0
            if isinstance(outputs, dict) and all(k in outputs for k in ["technical_accuracy", "completeness", "clarity", "relevance"]):
                 for i, metric_name in enumerate(["technical_accuracy", "completeness", "clarity", "relevance"]):
                     metric_loss = criterion(outputs[metric_name], target_metrics[:, i])
                     loss += metric_loss
            else:
                 logger.error(f"Model output is not a dict or missing keys: {outputs}")
                 continue 

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_dataloader:
            model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validation epoch {epoch+1}"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target_metrics = batch["target_metrics"].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    
                    loss = 0
                    if isinstance(outputs, dict) and all(k in outputs for k in ["technical_accuracy", "completeness", "clarity", "relevance"]):
                         for i, metric_name in enumerate(["technical_accuracy", "completeness", "clarity", "relevance"]):
                             metric_loss = criterion(outputs[metric_name], target_metrics[:, i])
                             loss += metric_loss
                    else:
                         logger.error(f"Model output during validation is not a dict or missing keys: {outputs}")
                         continue 

                    total_val_loss += loss.item()
            
            avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
            logger.info(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save the best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(args.output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_best_model_path = os.path.join(args.output_dir, f"custom_evaluator_best.pt") 
                torch.save(model.state_dict(), current_best_model_path)
                best_model_path = current_best_model_path 
                logger.info(f"Saved new best model to {best_model_path} (Val Loss: {best_val_loss:.4f})")
        else:
             # If no validation, save last model
             os.makedirs(args.output_dir, exist_ok=True)
             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
             current_model_path = os.path.join(args.output_dir, f"custom_evaluator_epoch_{epoch+1}_{timestamp}.pt")
             torch.save(model.state_dict(), current_model_path)
             best_model_path = current_model_path 
             logger.info(f"Saved model after epoch {epoch+1} to {best_model_path}")
        
        # --- Periodic Save every 10 epochs ---
        if (epoch + 1) % 10 == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            periodic_save_path = os.path.join(args.output_dir, "custom_evaluator_best.pt")
            try:
                torch.save(model.state_dict(), periodic_save_path)
                logger.info(f"Periodic save after epoch {epoch+1} to {periodic_save_path}")
                best_model_path = periodic_save_path
            except Exception as e:
                 logger.error(f"Failed periodic save after epoch {epoch+1}: {e}")
        # ------------------------------------

    logger.info(f"Training complete. Final best model saved to {best_model_path}")
    
    return best_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom evaluator model")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing training data (expects roles/, questions/, answers/ subdirs)")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save trained models")
    parser.add_argument("--pretrained_model", type=str, default="sentence-transformers/all-mpnet-base-v2", help="Pretrained transformer model name")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Path to a model checkpoint (.pt file) to load and continue training.") # New argument
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size for custom layers")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Optimizer learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenizer")
    
    args = parser.parse_args()

    # If no checkpoint specified, check for existing best model
    default_best_model = os.path.join(args.output_dir, "custom_evaluator_best.pt")
    if args.load_checkpoint is None and os.path.exists(default_best_model):
        logger.info(f"No checkpoint specified, but found existing best model: {default_best_model}. Will continue training from it.")
        args.load_checkpoint = default_best_model
    elif args.load_checkpoint is None:
        logger.info("No checkpoint specified and no existing best model found. Training from scratch.")

    if not os.path.isdir(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
        
    best_model_path = train_evaluator(args)
    
    # After training, always save the best model as 'custom_evaluator_best.pt'
    if best_model_path:
        try:
            import shutil
            shutil.copyfile(best_model_path, default_best_model)
            logger.info(f"Copied best model to {default_best_model}")
        except Exception as e:
            logger.warning(f"Failed to copy best model to {default_best_model}: {e}")
        logger.info(f"Successfully completed training. Best model saved at: {best_model_path}")
    else:
        logger.error("Training failed or did not produce a model.")
        sys.exit(1)
