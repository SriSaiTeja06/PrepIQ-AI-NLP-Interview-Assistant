from src.data_generation import DataGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Generate the complete dataset for the interview assistant."""
    logger.info("Starting dataset generation...")
    
    try:
        generator = DataGenerator()
        generator.generate_dataset()
        
        logger.info("Dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
