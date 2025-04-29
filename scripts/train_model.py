from src.training.trainer import Trainer
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Train the interview assistant model."""
    logger.info("Starting model training...")
    
    # Initialize trainer
    trainer = Trainer(
        model_name="google/flan-t5-large",
        output_dir="models",
        batch_size=8,
        num_epochs=3,
        learning_rate=2e-5,
        warmup_steps=1000,
        weight_decay=0.01
    )
    
    # Train for each task
    tasks = ["answer", "evaluate", "feedback"]
    
    for task in tasks:
        logger.info(f"Training for task: {task}")
        
        try:
            # Train model
            trainer.train(
                data_dir="data",
                task=task,
                validation_split=0.1
            )
            
            # Evaluate model
            metrics = trainer.evaluate(
                data_dir="data",
                task=task,
                validation_split=0.1
            )
            
            logger.info(f"Evaluation metrics for {task}: {metrics}")
            
        except Exception as e:
            logger.error(f"Error training task {task}: {str(e)}")
            continue
    
    logger.info("Model training completed!")

if __name__ == "__main__":
    main()
