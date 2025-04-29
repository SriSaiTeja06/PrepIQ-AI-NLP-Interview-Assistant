import os
from pydantic_settings import BaseSettings
from typing import List, Dict

class Settings(BaseSettings):
    # Project settings
    PROJECT_NAME: str = "AI-NLP Interview Assistant"
    VERSION: str = "1.0.0"
    
    # Speech-to-Text settings
    STT_PROVIDER: str = "google"
    SAMPLE_RATE: int = 16000
    LANGUAGE_CODE: str = "en-US"
    
    # Technical domains and their associated skills
    TECHNICAL_DOMAINS: Dict[str, List[str]] = {
        "data_science": [
            "machine learning", "deep learning", "statistics",
            "data analysis", "python", "pandas", "numpy"
        ],
        "software_engineering": [
            "algorithms", "data structures", "system design",
            "database", "api design", "testing"
        ],
        "devops": [
            "docker", "kubernetes", "ci/cd",
            "aws", "azure", "google cloud"
        ]
    }
    
    # Evaluation metrics weights
    METRICS_WEIGHTS: Dict[str, float] = {
        "technical_accuracy": 0.4,
        "completeness": 0.2,
        "clarity": 0.2,
        "relevance": 0.2
    }
    
    # Supported job roles
    SUPPORTED_ROLES: List[str] = [
        "Data Scientist",
        "Machine Learning Engineer",
        "Software Engineer",
        "DevOps Engineer",
        "Data Engineer",
        "Cloud Engineer",
        "Security Engineer"
    ]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

def main():
    print(f"Project Name: {settings.PROJECT_NAME}")
    print(f"Version: {settings.VERSION}")
    print("\nTechnical Domains:")
    for domain, skills in settings.TECHNICAL_DOMAINS.items():
        print(f"\n{domain}:")
        for skill in skills:
            print(f"- {skill}")

if __name__ == "__main__":
    main()
