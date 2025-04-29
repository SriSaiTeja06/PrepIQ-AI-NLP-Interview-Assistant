# Data Generation Documentation

This document provides information about the data generation process for the AI-NLP Interview Assistant.

## Dataset Structure

The generated dataset is organized as follows:

```
data/
├── {role}_job_desc.txt          # Job description for each role
├── {role}_questions.json        # Interview questions for each role
├── {role}_answers.json         # Sample answers for each question
└── {role}_feedback.json        # Feedback for each answer
```

## Role Coverage

The system supports the following roles:
- Data Scientist
- Machine Learning Engineer
- Software Engineer
- DevOps Engineer
- Data Engineer
- Cloud Engineer
- Security Engineer

## Data Components

### 1. Job Descriptions
- Generated for each role
- Includes key responsibilities and required skills
- Format: Text file

### 2. Interview Questions
- 20-30 technical questions per role
- 10-15 behavioral questions per role
- Each question includes:
  - ID
  - Role
  - Type (technical/behavioral)
  - Difficulty level
  - Expected skills
  - Content

### 3. Sample Answers
- 3 quality levels per question (high, medium, low)
- Technical answers follow a structured format
- Behavioral answers use the STAR method
- Each answer includes:
  - ID
  - Question ID
  - Content
  - Confidence score

### 4. Feedback
- Generated for each answer
- Includes:
  - Strengths
  - Areas for improvement
  - Specific recommendations
  - Quality metrics

## Generation Process

The data generation process involves:
1. Role-specific job descriptions
2. Technical and behavioral questions
3. Quality-controlled sample answers
4. Constructive feedback generation

## Usage

To generate the dataset:
```bash
python scripts/generate_dataset.py
```

The generated data will be saved in the `data` directory.
