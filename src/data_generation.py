import json
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass
from datetime import datetime
import os
from src.schemas import Question, Answer, Feedback
from src.metrics import InterviewMetrics
from src.schemas.metrics import EvaluationMetrics # Added this import

class DataGenerator:
    def __init__(self):
        """Initialize the data generator."""
        self.roles = {
            "Data Scientist": {
                "skills": ["Python", "Machine Learning", "Statistics", "Data Analysis", "Pandas", "NumPy"],
                "domains": ["data_science", "machine_learning", "statistics"]
            },
            "Machine Learning Engineer": {
                "skills": ["Python", "TensorFlow", "PyTorch", "Kubernetes", "Docker", "CI/CD"],
                "domains": ["machine_learning", "devops", "cloud"]
            },
            "Software Engineer": {
                "skills": ["Python", "Java", "C++", "System Design", "Algorithms", "Data Structures"],
                "domains": ["software_engineering", "algorithms", "system_design"]
            },
            "DevOps Engineer": {
                "skills": ["Docker", "Kubernetes", "CI/CD", "AWS", "Azure", "Google Cloud"],
                "domains": ["devops", "cloud", "automation"]
            },
            "Data Engineer": {
                "skills": ["Python", "SQL", "ETL", "Data Warehousing", "Apache Airflow", "Apache Kafka"],
                "domains": ["data_engineering", "big_data", "data_warehousing"]
            },
            "Cloud Engineer": {
                "skills": ["AWS", "Azure", "Google Cloud", "Infrastructure as Code", "Terraform", "CloudFormation"],
                "domains": ["cloud", "devops", "infrastructure"]
            },
            "Security Engineer": {
                "skills": ["Python", "Security", "Penetration Testing", "Vulnerability Assessment", "Security Policies"],
                "domains": ["security", "networking", "compliance"]
            },
            "Frontend Developer": {
                "skills": ["JavaScript", "React", "Vue.js", "CSS", "HTML", "Web Performance"],
                "domains": ["frontend", "web_development", "ui_ux"]
            },
            "Backend Developer": {
                "skills": ["Python", "Java", "Node.js", "Database Design", "API Development", "System Architecture"],
                "domains": ["backend", "system_design", "api_development"]
            },
            "Full Stack Developer": {
                "skills": ["JavaScript", "React", "Node.js", "MongoDB", "System Design", "Web Performance"],
                "domains": ["full_stack", "web_development", "system_design"]
            },
            "QA/Test Engineer": {
                "skills": ["Python", "Selenium", "JMeter", "Test Automation", "Performance Testing", "Test Design"],
                "domains": ["testing", "automation", "quality_assurance"]
            },
            "Site Reliability Engineer": {
                "skills": ["Python", "Monitoring", "Alerting", "Incident Response", "Disaster Recovery", "Performance Optimization"],
                "domains": ["sre", "devops", "system_reliability"]
            },
            "Mobile Developer": {
                "skills": ["Swift", "Kotlin", "React Native", "iOS", "Android", "Mobile Architecture"],
                "domains": ["mobile_development", "ios", "android"]
            }
        }

        # Question templates for different types
        self.question_templates = {
            "technical": [
                "Explain how {skill} works.",
                "What are the key concepts in {skill}?",
                "How would you implement {skill} in a production environment?",
                "Discuss the trade-offs of using {skill}.",
                "What are the common challenges with {skill}?",
                "How do you optimize {skill} for performance?"
            ],
            "behavioral": [
                "Tell me about a time when you had to {skill}.",
                "Describe a situation where you had to {skill} under pressure.",
                "How do you approach {skill} in your work?",
                "What's your process for {skill}?",
                "Give an example of when you had to {skill}."
            ]
        }

        # Answer quality levels
        self.answer_qualities = {
            "high": {
                "completeness": 0.9,
                "technical_accuracy": 0.95,
                "clarity": 0.9,
                "relevance": 0.95
            },
            "medium": {
                "completeness": 0.7,
                "technical_accuracy": 0.8,
                "clarity": 0.8,
                "relevance": 0.85
            },
            "low": {
                "completeness": 0.5,
                "technical_accuracy": 0.6,
                "clarity": 0.6,
                "relevance": 0.7
            }
        }

    def generate_job_description(self, role: str) -> str:
        """Generate a job description for a given role."""
        role_info = self.roles[role]
        
        # Create a description combining skills and domains
        description = f"{role} - Senior Level\n\n"
        description += "Key Responsibilities:\n"
        description += "- Lead the development of {}\n".format(", ".join(role_info["skills"][:3]))
        description += "- Design and implement {}\n".format(", ".join(role_info["skills"][3:6]))
        description += "- Collaborate with team on {}\n\n".format(", ".join(role_info["skills"][6:]))
        
        description += "Required Skills:\n"
        for skill in role_info["skills"]:
            description += f"- Proficient in {skill}\n"
        
        return description

    def generate_interview_questions(self, role: str) -> List[Question]:
        """Generate interview questions for a given role."""
        role_info = self.roles[role]
        questions = []
        
        # Generate technical questions
        for _ in range(20):
            skill = random.choice(role_info["skills"])
            template = random.choice(self.question_templates["technical"])
            question_text = template.format(skill=skill)
            
            question = Question(
                id=f"q_{role.replace(' ', '_')}_{len(questions)}",
                role=role,
                type="technical",
                difficulty=random.choice(["easy", "medium", "hard"]),
                content=question_text,
                expected_skills=[skill] + random.sample(role_info["skills"], 2),
                created_at=datetime.utcnow()
            )
            questions.append(question)
        
        # Generate behavioral questions
        for _ in range(15):
            skill = random.choice(role_info["skills"])
            template = random.choice(self.question_templates["behavioral"])
            question_text = template.format(skill=skill)
            
            question = Question(
                id=f"q_{role.replace(' ', '_')}_{len(questions)}",
                role=role,
                type="behavioral",
                difficulty=random.choice(["easy", "medium"]),
                content=question_text,
                expected_skills=[skill] + random.sample(role_info["skills"], 2),
                created_at=datetime.utcnow()
            )
            questions.append(question)
        
        return questions

    def generate_sample_answer(self, question: Question, quality: str = "medium") -> Answer:
        """Generate a sample answer for a given question."""
        quality_metrics = self.answer_qualities[quality]
        
        # Create answer based on question type and quality
        if question.type == "technical":
            answer_text = self._generate_technical_answer(question.content, quality_metrics)
        else:
            answer_text = self._generate_behavioral_answer(question.content, quality_metrics)
        
        return Answer(
            id=f"a_{question.id}_{quality}",
            question_id=question.id,
            content=answer_text,
            confidence=quality_metrics["technical_accuracy"],
            created_at=datetime.utcnow()
        )

    def _generate_technical_answer(self, question: str, metrics: Dict[str, float]) -> str:
        """Generate a technical answer."""
        answer = ""
        
        # Add technical explanation
        if metrics["completeness"] > 0.7:
            answer += "The key concept behind this is...\n\n"
        
        # Add implementation details
        if metrics["technical_accuracy"] > 0.8:
            answer += "To implement this, you would...\n\n"
        
        # Add optimization considerations
        if metrics["relevance"] > 0.8:
            answer += "For better performance, consider...\n\n"
        
        return answer

    def _generate_behavioral_answer(self, question: str, metrics: Dict[str, float]) -> str:
        """Generate a behavioral answer using STAR method."""
        answer = ""
        
        # Situation
        if metrics["completeness"] > 0.7:
            answer += "In my previous role at XYZ company, we faced a situation where...\n\n"
        
        # Task
        if metrics["technical_accuracy"] > 0.8:
            answer += "My task was to...\n\n"
        
        # Action
        if metrics["relevance"] > 0.8:
            answer += "To address this, I...\n\n"
        
        # Result
        if metrics["clarity"] > 0.8:
            answer += "The outcome was...\n\n"
        
        return answer

    def generate_feedback(self, answer: Answer, metrics: Dict[str, float]) -> Feedback:
        """Generate feedback for a given answer."""
        feedback_content = ""
        
        # Generate feedback content based on metrics
        feedback_content += "Strengths:\n"
        if metrics["technical_accuracy"] > 0.8:
            feedback_content += "- Strong technical understanding\n"
        if metrics["completeness"] > 0.8:
            feedback_content += "- Comprehensive explanation\n"
        if metrics["clarity"] > 0.8:
            feedback_content += "- Clear and concise communication\n"
        if not feedback_content.endswith("\n\n"): # Add newline if strengths were added
             feedback_content += "\n"

        feedback_content += "Areas for Improvement:\n"
        if metrics["technical_accuracy"] < 0.8:
            feedback_content += "- Technical accuracy needs improvement\n"
        if metrics["completeness"] < 0.8:
            feedback_content += "- Answer lacks depth\n"
        if metrics["clarity"] < 0.8:
            feedback_content += "- Could be more concise\n"
        if not feedback_content.endswith("\n\n"): # Add newline if improvements were added
             feedback_content += "\n"

        feedback_content += "Specific Recommendations:\n"
        if metrics["technical_accuracy"] < 0.9:
            feedback_content += "- Review fundamental concepts\n"
        if metrics["completeness"] < 0.9:
            feedback_content += "- Provide more detailed explanations\n"
        if metrics["clarity"] < 0.9:
            feedback_content += "- Structure the answer more clearly\n"

        # Create EvaluationMetrics object (assuming it takes the dict directly)
        evaluation_metrics = EvaluationMetrics(**metrics)

        feedback = Feedback(
            id=f"f_{answer.id}",
            answer_id=answer.id,
            content=feedback_content.strip(), # Remove trailing newline
            metrics=evaluation_metrics, # Pass the EvaluationMetrics object
            created_at=datetime.utcnow()
        )
        
        return feedback

    def generate_dataset(self, output_dir: str = "data") -> None:
        """Generate complete dataset for all roles."""
        os.makedirs(output_dir, exist_ok=True)
        
        for role in self.roles:
            # Replace any slashes in role name with underscores
            safe_role_name = role.replace('/', '_')
            
            # Generate job description
            job_desc = self.generate_job_description(role)
            with open(os.path.join(output_dir, f"{safe_role_name}_job_desc.txt"), "w") as f:
                f.write(job_desc)
            
            # Generate questions
            questions = self.generate_interview_questions(role)
            questions_data = []
            questions_data = [{**q.dict(), "created_at": q.created_at.isoformat()} for q in questions] # Reverted to .dict() and list comprehension
            with open(os.path.join(output_dir, f"{safe_role_name}_questions.json"), "w") as f:
                json.dump(questions_data, f, indent=2)

            # Generate answers and feedback
            answers = []
            feedbacks = []

            for question in questions:
                for quality in ["high", "medium", "low"]:
                    answer = self.generate_sample_answer(question, quality)
                    answers.append(answer)

                    # Evaluate answer
                    evaluator = InterviewMetrics()
                    metrics = evaluator.evaluate_answer(
                        question.content,
                        answer.content,
                        question.expected_skills
                    )

                    feedback = self.generate_feedback(answer, metrics)
                    feedbacks.append(feedback)

            # Save answers
            answers_data = [{**a.dict(), "created_at": a.created_at.isoformat()} for a in answers] # Reverted to .dict() and list comprehension
            with open(os.path.join(output_dir, f"{safe_role_name}_answers.json"), "w") as f:
                json.dump(answers_data, f, indent=2)

            # Save feedback
            feedback_data = [{**f.dict(), "created_at": f.created_at.isoformat()} for f in feedbacks] # Reverted to .dict() and list comprehension
            with open(os.path.join(output_dir, f"{safe_role_name}_feedback.json"), "w") as f:
                json.dump(feedback_data, f, indent=2)

# Example usage
def main():
    generator = DataGenerator()
    generator.generate_dataset()

if __name__ == "__main__":
    main()
