# PrepIQ - AI-NLP Interview Assistant

An AI-powered platform to generate interview questions, evaluate candidate answers, and provide detailed feedback for technical and behavioral interviews across multiple software and data roles.

---

## Objectives

- **Automate interview preparation** with role-specific question generation.
- **Evaluate candidate answers** using custom-trained NLP models.
- **Provide actionable feedback** highlighting strengths and areas for improvement.
- **Support multiple roles** including Software Engineer, Data Scientist, DevOps, QA, Mobile Developer, and more.
- **Enable speech-based answers** with integrated speech-to-text (using OpenAI Whisper).
- **Facilitate model training and improvement** with extensible data pipelines.

---

## Expected Outcomes

- A web-based tool for interview practice with realistic, role-specific questions.
- Automated scoring and feedback on candidate answers.
- Support for both text and audio answers.
- Extensible backend for adding new roles, questions, and improving models.

---

## Directory Structure

```
.
├── .gitignore                                               #  Git ignore rules
├── demo.py                                                  #  Script to quickly start backend and frontend
├── error_log.txt                                            #  Log file for errors
├── README.md                                                #  This file
├── requirements.txt                                         #  Python dependencies
├── test_workflow.py                                         #  Workflow for testing
├── test.ipynb                                               #  Jupyter notebook for testing
├── config/                                                  #  Configuration files
│   └── config.py                                            #  Main configuration file
├── docs/                                                    #  Documentation files
│   └── DATA_GENERATION.md                                   #  Documentation for data generation
├── frontend/                                                #  React frontend application
│   ├── package-lock.json                                    #  Dependency lock file for npm
│   ├── package.json                                         #  Frontend dependencies and scripts
│   └── src/                                                 #  Frontend source code
├── organized_data/                                          #  Processed data for training
│   ├── placeholder.txt                                      #  Placeholder file
│   ├── answers/                                             #  Organized answer data
│   ├── questions/                                           #  Organized question data
│   └── roles/                                               #  Organized role data
├── results/                                                 #  Results from evaluations and tests
│   ├── custom_evaluator_input.json                          #  Input for custom evaluator
│   ├── custom_evaluator_output.json                         #  Output from custom evaluator
│   ├── custom_evaluator_test_results.json                   #  Test results for custom evaluator
│   └── placeholder.txt                                      #  Placeholder file
├── scripts/                                              #  Python scripts for data prep, training, evaluation
│   ├── create_test_set.py                                   #  Script to create a test set
│   ├── cross_role_training.py                               #  Script for cross-role training
│   ├── evaluate_models.py                                   #  Script to evaluate models
│   ├── generate_dataset.py                                  #  Script to generate synthetic dataset
│   ├── generate_metrics.py                                  #  Script to generate metrics
│   ├── generate_missing_answers.py                          #  Script to generate missing answers
│   ├── hyperparameter_tuning.py                             #  Script for hyperparameter tuning
│   ├── organize_data.py                                     #  Script to organize data
│   ├── start_server.py                                      #  Script to start the server
│   ├── test_custom_evaluator.py                             #  Script to test custom evaluator
│   ├── test_integration.py                                  #  Script for integration tests
│   ├── train_custom_evaluator.py                            #  Script to train custom evaluator
│   └── train_model.py                                       #  Script to train models
├── src/                                                     #  Backend Python code
│   ├── __init__.py                                          #  Initializes the src directory as a Python package
│   ├── data_generation.py                                   #  Synthetic data generation
│   ├── main.py                                              #  FastAPI app entry point
│   ├── metrics.py                                           #  Metrics calculation
│   ├── schemas.py                                           #  Pydantic schemas
│   └── speech_to_text.py                                    #  OpenAI Whisper integration
├── test_data/                                               #  Test data
│   ├── placeholder.txt                                      #  Placeholder file
│   ├── answers/                                             #  Test answer data
│   ├── questions/                                           #  Test question data
│   └── roles/                                               #  Test role data
└── tests/                                                   #  Unit and integration tests
    ├── test_metrics.py                                      #  Tests for metrics
    ├── test_models.py                                       #  Tests for models
    └── test_speech_to_text.py                               #  Tests for speech-to-text
```

## Setup Instructions

### Prerequisites

Before you begin, ensure you have the following installed on your system:

1.  **Python:** Version 3.9 or higher **AND** 3.11 or lower.
2.  **Git:** For cloning the repository.
3.  **Node.js and npm:** Required for the frontend application. Download from [https://nodejs.org/](https://nodejs.org/).
4.  **FFmpeg:** Required by the Whisper library for audio processing.
    *   **Windows:** Install using Chocolatey (`choco install ffmpeg`) or download from the [FFmpeg website](https://ffmpeg.org/download.html) and add it to your system's PATH.
    *   **macOS:** Install using Homebrew (`brew install ffmpeg`).
    *   **Linux:** Install using your package manager (e.g., `sudo apt update && sudo apt install ffmpeg` on Debian/Ubuntu).

### 1. Clone the Repository

```bash
git clone https://github.com/OneTrickBreach/PrepIQ---AI-NLP-Interview-Assistant.git
cd PrepIQ---AI-NLP-Interview-Assistant
```

### 2. Python Backend Setup

- Navigate to the project root directory if you aren't already there.
- Create and activate a Python virtual environment (recommended):

  ```bash
  # Create the environment (only needs to be done once)
  python -m venv .venv 

  # Activate the environment (do this every time you work on the project)
  # Windows (PowerShell):
  .venv\Scripts\activate
  # macOS/Linux:
  # source .venv/bin/activate 
  ```

- Install Python dependencies:

  ```bash
  # Ensure your virtual environment is active first!
  pip install -r requirements.txt
  ```
  *(Note: This will install PyTorch and Whisper. The first time Whisper runs, it may download the model weights, which can take some time and disk space.)*

### 3. Download the Trained Evaluation Model

The custom-trained evaluation model (`.pt` file) is required for the answer evaluation feature but is **not included** in this repository due to size limits.

- Download the latest trained model checkpoint from this link:
  **[https://drive.google.com/file/d/1puq4Luf4aBJQUyw6FHLV7jU5yAUOgzX_/view?usp=sharing]** *(Link provided by user)*

- After downloading, place the `.pt` file (e.g., `custom_evaluator_best.pt`) inside the `models/` directory in the project root. The backend will automatically load the latest model file from this directory when started.

### 4. Frontend Setup

- Navigate to the frontend directory:

  ```bash
  cd frontend
  ```

- Install Node.js dependencies:

  ```bash
  npm install
  ```

- Go back to the project root directory:

  ```bash
  cd .. 
  ```

### 5. Running the Application

You need to run the backend and frontend servers simultaneously in separate terminals.

- **Terminal 1: Run Backend Server**
  - Make sure you are in the project root directory (`PrepIQ---AI-NLP-Interview-Assistant`).
  - Activate your Python virtual environment (`.\.venv\Scripts\Activate.ps1` or `source .venv/bin/activate`).
  - Start the FastAPI server:

    ```bash
    python -m src.main
    ```

- **Terminal 2: Run Frontend Server**
  - Open a new terminal.
  - Navigate to the `frontend` directory: `cd frontend`
  - Start the React development server:

    ```bash
    npm start dev
    ```

- Once both servers are running, the application should be available in your web browser at `http://localhost:3000`.

---

## Generating Sample Data (Optional)

If you want to generate synthetic sample data for testing or retraining:

```bash
# Ensure your Python virtual environment is active
python scripts/generate_dataset.py
python scripts/organize_data.py
```

> **Note:** This synthetic data is very basic. For real training, replace or augment it with real, annotated examples.

---

## Training the Evaluation Model (Optional)

- To train from scratch:

```bash
# Ensure virtual environment is active
python scripts/train_custom_evaluator.py --data_dir organized_data/ --output_dir models/ --num_epochs 20 --batch_size 8
```

- To continue training from a checkpoint:

```bash
# Ensure virtual environment is active
python scripts/train_custom_evaluator.py --data_dir organized_data/ --output_dir models/ --num_epochs 10 --batch_size 8 --load_checkpoint "models/custom_evaluator_best.pt"
```

---

## Demo

A `demo.py` file is included to quickly start the backend and frontend and open the landing page. To use the demo, ensure that your Python virtual environment is active and that you have installed the Node.js dependencies for the frontend.

To run the demo, execute the following command:

```bash
python demo.py
```

**Important:** Make sure your environment is active before running the demo.

---

## Notes

- The `.gitignore` excludes large data/model directories and environment files.
- Placeholder files are included in excluded directories to preserve structure.
- You can customize roles, questions, and feedback templates by editing the data files or generation scripts.
- Speech-to-text is now handled locally using Whisper. Performance depends on your hardware and the chosen Whisper model size (default is "base").

---
