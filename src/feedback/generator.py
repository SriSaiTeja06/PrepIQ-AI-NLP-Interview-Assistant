"""
Feedback Generator for the AI-NLP Interview Assistant.
This module implements the feedback generation mechanism.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
from src.schemas import Question, Answer, Feedback, EvaluationMetrics

class FeedbackGenerator:
    """
    Feedback generator for interview answers.
    This class implements a template-based approach with dynamic content selection 
    and personalization elements.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the feedback generator.
        
        Args:
            templates_dir: Directory containing feedback templates
        """
        self.templates_dir = templates_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "feedback_templates"
        )
        
        # Load templates
        self.templates = self._load_templates()
        
        # Load feedback components
        self.feedback_components = self._load_feedback_components()
        
    def _load_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Load feedback templates from files.
        
        Returns:
            Dictionary of templates by category and quality level
        """
        templates = {
            "general": {
                "positive": [],
                "negative": [],
                "neutral": []
            },
            "technical": {
                "positive": [],
                "negative": [],
                "improvement": []
            },
            "communication": {
                "positive": [],
                "negative": [],
                "improvement": []
            }
        }
        
        # Load templates from files if they exist
        template_file = os.path.join(self.templates_dir, "templates.json")
        if os.path.exists(template_file):
            with open(template_file, 'r', encoding='utf-8') as f:
                templates = json.load(f)
        else:
            # Use default templates
            templates["general"]["positive"] = [
                "Your answer demonstrates excellent understanding of the topic.",
                "Your explanation is comprehensive and well-structured.",
                "You've presented a strong response that addresses the question effectively."
            ]
            templates["general"]["negative"] = [
                "Your answer lacks some key components necessary for a complete response.",
                "There are gaps in your explanation that affect its overall quality.",
                "Your response could benefit from more structure and clarity."
            ]
            templates["general"]["neutral"] = [
                "You've provided a straightforward answer to the question.",
                "Your response covers the basics of what was asked.",
                "You've addressed the question in a direct manner."
            ]
            templates["technical"]["positive"] = [
                "You've demonstrated strong technical knowledge in {topic}.",
                "Your explanation of {concept} shows deep understanding.",
                "Your technical accuracy regarding {topic} is impressive."
            ]
            templates["technical"]["negative"] = [
                "Your explanation of {concept} contains some technical inaccuracies.",
                "The implementation details for {topic} need more precision.",
                "Some technical aspects of {concept} are missing from your answer."
            ]
            templates["technical"]["improvement"] = [
                "Consider exploring {topic} in more depth, particularly focusing on {concept}.",
                "To strengthen your answer, include more specific details about {concept}.",
                "Add examples of how {concept} is applied in practical scenarios."
            ]
            templates["communication"]["positive"] = [
                "Your answer is presented in a clear and logical manner.",
                "You've explained complex concepts in an accessible way.",
                "Your communication style is effective and easy to follow."
            ]
            templates["communication"]["negative"] = [
                "Your explanation would benefit from a more structured approach.",
                "Some parts of your answer are difficult to follow due to organization.",
                "The clarity of your response is affected by its organization."
            ]
            templates["communication"]["improvement"] = [
                "Try organizing your answer with a clear introduction, body, and conclusion.",
                "Use specific examples to illustrate your points more effectively.",
                "Consider using technical terminology more precisely in your explanation."
            ]
            
        return templates
    
    def _load_feedback_components(self) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """
        Load feedback components for different metrics and scores.
        
        Returns:
            Dictionary of feedback components
        """
        components = {
            "technical_accuracy": {
                "high": [],
                "medium": [],
                "low": []
            },
            "completeness": {
                "high": [],
                "medium": [],
                "low": []
            },
            "clarity": {
                "high": [],
                "medium": [],
                "low": []
            },
            "relevance": {
                "high": [],
                "medium": [],
                "low": []
            }
        }
        
        components_file = os.path.join(self.templates_dir, "components.json")
        if os.path.exists(components_file):
            with open(components_file, 'r', encoding='utf-8') as f:
                components = json.load(f)
        else:
            components["technical_accuracy"]["high"] = [
                {"text": "Your answer demonstrates excellent technical accuracy.", "suggestion": None},
                {"text": "You've shown a deep understanding of the technical concepts.", "suggestion": None}
            ]
            components["technical_accuracy"]["medium"] = [
                {"text": "Your answer is mostly technically accurate, but could use some refinement.", "suggestion": "Review the core principles of {concept} to ensure complete accuracy."},
                {"text": "The technical aspects of your answer are generally good, with some minor issues.", "suggestion": "Double-check your explanation of {concept} for complete accuracy."}
            ]
            components["technical_accuracy"]["low"] = [
                {"text": "Your answer contains significant technical inaccuracies.", "suggestion": "Study the fundamentals of {concept} to improve your understanding."},
                {"text": "The technical foundation of your answer needs improvement.", "suggestion": "Consider revisiting the basic principles of {topic}."}
            ]
            
            components["completeness"]["high"] = [
                {"text": "Your answer is comprehensive and covers all key aspects.", "suggestion": None},
                {"text": "You've addressed all the important elements of the question.", "suggestion": None}
            ]
            components["completeness"]["medium"] = [
                {"text": "Your answer covers most key points, but misses some important aspects.", "suggestion": "Consider including information about {missing_aspect}."},
                {"text": "Your response is somewhat complete but could benefit from more details.", "suggestion": "Add more details about {missing_aspect} to make your answer more comprehensive."}
            ]
            components["completeness"]["low"] = [
                {"text": "Your answer is incomplete and misses several critical elements.", "suggestion": "Make sure to address {missing_aspect} and {another_missing_aspect}."},
                {"text": "Your response is missing key components necessary for a complete answer.", "suggestion": "Include information about {missing_aspect} to improve completeness."}
            ]
            
            components["clarity"]["high"] = [
                {"text": "Your explanation is very clear and well-structured.", "suggestion": None},
                {"text": "Your answer is presented in a logical and easy-to-follow manner.", "suggestion": None}
            ]
            components["clarity"]["medium"] = [
                {"text": "Your explanation is somewhat clear but could be better organized.", "suggestion": "Try using a more structured approach with clear sections."},
                {"text": "Some parts of your answer could be explained more clearly.", "suggestion": "Clarify your explanation of {concept} for better understanding."}
            ]
            components["clarity"]["low"] = [
                {"text": "Your answer lacks clarity and is difficult to follow.", "suggestion": "Reorganize your response with a clear beginning, middle, and end."},
                {"text": "The structure of your answer makes it hard to understand your points.", "suggestion": "Use a step-by-step approach to explain {concept} more clearly."}
            ]
            
            components["relevance"]["high"] = [
                {"text": "Your answer is highly relevant to the question asked.", "suggestion": None},
                {"text": "You've focused precisely on what was asked in the question.", "suggestion": None}
            ]
            components["relevance"]["medium"] = [
                {"text": "Your answer is somewhat relevant but includes some tangential information.", "suggestion": "Focus more specifically on addressing {main_topic}."},
                {"text": "Parts of your response stray from the main question.", "suggestion": "Keep your answer centered on {main_topic} for better relevance."}
            ]
            components["relevance"]["low"] = [
                {"text": "Your answer isn't directly addressing the question asked.", "suggestion": "Realign your response to focus on {main_topic}."},
                {"text": "Much of your response is off-topic or not relevant to the question.", "suggestion": "Make sure you're answering the specific question about {main_topic}."}
            ]
        
        return components
    
    def generate_feedback(self, question: Question, answer: Answer, metrics: EvaluationMetrics) -> Feedback:
        """
        Generate feedback for an answer based on evaluation metrics.
        
        Args:
            question: The original question
            answer: The answer to provide feedback on
            metrics: Evaluation metrics for the answer
            
        Returns:
            Generated feedback
        """
        # Determine key topics and concepts from the question
        topics = self._extract_topics(question)
        
        # Get strengths and areas for improvement based on metrics
        strengths, improvements = self._analyze_metrics(metrics, topics)
        
        # Generate personalized feedback content
        content = self._generate_content(question, answer, metrics, strengths, improvements)
        
        # Create feedback object
        feedback = Feedback(
            id=f"f_{answer.id}",
            answer_id=answer.id,
            content=content,
            metrics=metrics,
            created_at=datetime.utcnow()
        )
        
        return feedback
    
    def _extract_topics(self, question: Question) -> Dict[str, str]:
        """
        Extract key topics and concepts from the question.
        
        Args:
            question: The question to analyze
            
        Returns:
            Dictionary of topics and concepts
        """
        topics = {
            "main_topic": "",
            "concept": "",
            "missing_aspect": "",
            "another_missing_aspect": ""
        }
        
        # Extract main topic from question content
        if hasattr(question, "content") and question.content:
            words = question.content.split()
            # Simple approach: use nouns after "about", "in", "of" as topics
            for i, word in enumerate(words):
                if word.lower() in ["about", "in", "of", "regarding", "on"]:
                    if i + 1 < len(words):
                        topics["main_topic"] = words[i + 1].strip(".,?!").lower()
                        if i + 2 < len(words):
                            topics["concept"] = words[i + 2].strip(".,?!").lower()
                        break
        
        # Extract from expected skills if main_topic is still empty
        if not topics["main_topic"] and hasattr(question, "expected_skills") and question.expected_skills:
            if len(question.expected_skills) > 0:
                topics["main_topic"] = question.expected_skills[0]
            if len(question.expected_skills) > 1:
                topics["concept"] = question.expected_skills[1]
            if len(question.expected_skills) > 2:
                topics["missing_aspect"] = question.expected_skills[2]
            if len(question.expected_skills) > 3:
                topics["another_missing_aspect"] = question.expected_skills[3]
        
        # Fallbacks for empty fields
        if not topics["main_topic"]:
            topics["main_topic"] = question.role or "this topic"
        if not topics["concept"]:
            topics["concept"] = "key concepts"
        if not topics["missing_aspect"]:
            topics["missing_aspect"] = "important details"
        if not topics["another_missing_aspect"]:
            topics["another_missing_aspect"] = "practical applications"

        # Ensure 'topic' key exists for formatting templates that use it
        topics['topic'] = topics['main_topic']
        
        return topics
    
    def _analyze_metrics(self, metrics: EvaluationMetrics, topics: Dict[str, str]) -> Tuple[List[str], List[str]]:
        """
        Analyze metrics to identify strengths and areas for improvement.
        
        Args:
            metrics: Evaluation metrics
            topics: Key topics and concepts from the question
            
        Returns:
            Tuple of (strengths, improvements) lists
        """
        strengths = []
        improvements = []
        
        # Analyze technical accuracy
        if metrics.technical_accuracy >= 0.8:
            component = np.random.choice(self.feedback_components["technical_accuracy"]["high"])
            strengths.append(component["text"])
        elif metrics.technical_accuracy >= 0.5:
            component = np.random.choice(self.feedback_components["technical_accuracy"]["medium"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        else:
            component = np.random.choice(self.feedback_components["technical_accuracy"]["low"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        
        # Analyze completeness
        if metrics.completeness >= 0.8:
            component = np.random.choice(self.feedback_components["completeness"]["high"])
            strengths.append(component["text"])
        elif metrics.completeness >= 0.5:
            component = np.random.choice(self.feedback_components["completeness"]["medium"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        else:
            component = np.random.choice(self.feedback_components["completeness"]["low"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        
        # Analyze clarity
        if metrics.clarity >= 0.8:
            component = np.random.choice(self.feedback_components["clarity"]["high"])
            strengths.append(component["text"])
        elif metrics.clarity >= 0.5:
            component = np.random.choice(self.feedback_components["clarity"]["medium"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        else:
            component = np.random.choice(self.feedback_components["clarity"]["low"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        
        # Analyze relevance
        if metrics.relevance >= 0.8:
            component = np.random.choice(self.feedback_components["relevance"]["high"])
            strengths.append(component["text"])
        elif metrics.relevance >= 0.5:
            component = np.random.choice(self.feedback_components["relevance"]["medium"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        else:
            component = np.random.choice(self.feedback_components["relevance"]["low"])
            improvements.append(component["text"])
            if component["suggestion"]:
                improvements.append(component["suggestion"].format(**topics))
        
        return strengths, improvements
    
    def _generate_content(self, question: Question, answer: Answer, metrics: EvaluationMetrics,
                          strengths: List[str], improvements: List[str]) -> str:
        """
        Generate detailed feedback content including strengths and areas for improvement.

        Args:
            question: Original question
            answer: Answer being evaluated
            metrics: Evaluation metrics
            strengths: Identified strengths (derived from metrics analysis)
            improvements: Identified areas for improvement (derived from metrics analysis)

        Returns:
            Detailed feedback content string.
        """
        feedback_parts = []

        # 1. Overall Assessment Sentence based on overall score
        if metrics.overall_score >= 0.8:
            overall = "Overall, your answer is strong and effectively addresses the key aspects of the question."
        elif metrics.overall_score >= 0.5:
            overall = "Your answer covers some key aspects but could be more comprehensive and precise in certain areas."
        else:
            overall = "Your answer seems to miss some core requirements of the question or lacks sufficient detail and accuracy."
        feedback_parts.append(overall)
        feedback_parts.append("\n") # Add a newline for better readability

        # 2. Strengths Section
        if strengths:
            feedback_parts.append("**Strengths:**")
            # Use unique strengths to avoid repetition
            unique_strengths = list(dict.fromkeys(strengths))
            for strength in unique_strengths:
                feedback_parts.append(f"- {strength}")
            feedback_parts.append("\n") # Add a newline

        # 3. Areas for Improvement Section (Limited to max 4 points)
        if improvements:
            feedback_parts.append("**Areas for Improvement:**")
            # Use unique improvements to avoid repetition
            unique_improvements = list(dict.fromkeys(improvements))
            
            # Limit to a maximum of 4 improvement points
            improvement_count = 0
            for improvement in unique_improvements:
                if improvement_count < 4:
                    # Combine descriptive and actionable points directly
                    feedback_parts.append(f"- {improvement}")
                    improvement_count += 1
                else:
                    break # Stop adding points once the limit is reached


        # Combine parts into final feedback string
        final_feedback = "\n".join(feedback_parts).strip() # Use newline as separator

        # Fallback if no specific strengths/improvements were generated but score isn't perfect
        if not strengths and not improvements and metrics.overall_score < 0.95:
             final_feedback += "\n\nConsider reviewing the core concepts and requirements related to the question to further refine your response."


        return final_feedback
