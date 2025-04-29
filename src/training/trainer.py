import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from src.training.dataset import InterviewDataset
from src.models.transformer import TransformerModel
from typing import Dict, List, Optional
import os
from datetime import datetime

class Trainer:
    """Trainer class for the interview assistant system."""
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-large",
        output_dir: str = "models",
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name (str): Name of the pretrained model to use
            output_dir (str): Directory to save the trained model
            batch_size (int): Batch size for training
            num_epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            warmup_steps (int): Number of warmup steps
            weight_decay (float): Weight decay
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Set up training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False
        )
        
    def prepare_dataset(
        self,
        data_dir: str,
        task: str = "all",
        validation_split: float = 0.1
    ) -> Dict[str, InterviewDataset]:
        """
        Prepare the training and validation datasets.
        
        Args:
            data_dir (str): Directory containing the training data
            task (str): Task type ("answer", "evaluate", "feedback", or "all")
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            Dict[str, InterviewDataset]: Dictionary containing train and validation datasets
        """
        # Create dataset
        dataset = InterviewDataset(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            task=task
        )
        
        # Split dataset
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size]
        )
        
        return {
            "train": train_dataset,
            "validation": val_dataset
        }
        
    def train(
        self,
        data_dir: str,
        task: str = "all",
        validation_split: float = 0.1
    ) -> None:
        """
        Train the model.
        
        Args:
            data_dir (str): Directory containing the training data
            task (str): Task type ("answer", "evaluate", "feedback", or "all")
            validation_split (float): Fraction of data to use for validation
        """
        # Prepare datasets
        datasets = self.prepare_dataset(
            data_dir=data_dir,
            task=task,
            validation_split=validation_split
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer
        )
        
        # Train model
        trainer.train()
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.output_dir, f"{self.model_name}_{task}_{timestamp}")
        trainer.save_model(model_path)
        
        print(f"Model saved to: {model_path}")
        
    def evaluate(
        self,
        data_dir: str,
        task: str = "all",
        validation_split: float = 0.1
    ) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            data_dir (str): Directory containing the training data
            task (str): Task type ("answer", "evaluate", "feedback", or "all")
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Prepare datasets
        datasets = self.prepare_dataset(
            data_dir=data_dir,
            task=task,
            validation_split=validation_split
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            tokenizer=self.tokenizer
        )
        
        # Evaluate model
        metrics = trainer.evaluate()
        
        return metrics
        
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            model_path (str): Path to save the model
        """
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to load the model from
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
