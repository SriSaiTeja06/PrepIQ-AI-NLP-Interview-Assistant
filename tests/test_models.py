import unittest
from datetime import datetime
from src.models.transformer import TransformerModel
from src.models.baseline import BaselineModel
from src.schemas import Question, Answer, Feedback

class TestModels(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.transformer_model = TransformerModel()
        self.baseline_model = BaselineModel()
        
        self.test_question = Question(
            id="test_q_1",
            role="Software Engineer",
            type="technical",
            difficulty="medium",
            content="Explain how to implement a binary search algorithm.",
            expected_skills=["Algorithms", "Data Structures", "Python"],
            created_at=datetime.utcnow()
        )
        
    def test_transformer_model(self):
        """Test the transformer model functionality."""
        # Test answer generation
        answer = self.transformer_model.generate_answer(self.test_question)
        self.assertIsInstance(answer, Answer)
        self.assertIsNotNone(answer.content)
        
        # Test answer evaluation
        metrics = self.transformer_model.evaluate_answer(self.test_question, answer)
        self.assertIsInstance(metrics, dict)
        self.assertIn("technical_accuracy", metrics)
        
        # Test feedback generation
        feedback = self.transformer_model.generate_feedback(self.test_question, answer)
        self.assertIsInstance(feedback, Feedback)
        self.assertGreater(len(feedback.strengths), 0)
        
    def test_baseline_model(self):
        """Test the baseline model functionality."""
        # Test answer generation
        answer = self.baseline_model.generate_answer(self.test_question)
        self.assertIsInstance(answer, Answer)
        self.assertIsNotNone(answer.content)
        
        # Test answer evaluation
        metrics = self.baseline_model.evaluate_answer(self.test_question, answer)
        self.assertIsInstance(metrics, dict)
        self.assertIn("technical_accuracy", metrics)
        
        # Test feedback generation
        feedback = self.baseline_model.generate_feedback(self.test_question, answer)
        self.assertIsInstance(feedback, Feedback)
        self.assertGreater(len(feedback.strengths), 0)
        
    def test_model_compatibility(self):
        """Test that both models can handle the same inputs."""
        transformer_answer = self.transformer_model.generate_answer(self.test_question)
        baseline_answer = self.baseline_model.generate_answer(self.test_question)
        
        # Both models should generate valid answers
        self.assertIsInstance(transformer_answer, Answer)
        self.assertIsInstance(baseline_answer, Answer)
        
        # Both models should evaluate answers
        transformer_metrics = self.transformer_model.evaluate_answer(self.test_question, transformer_answer)
        baseline_metrics = self.baseline_model.evaluate_answer(self.test_question, baseline_answer)
        
        self.assertIsInstance(transformer_metrics, dict)
        self.assertIsInstance(baseline_metrics, dict)
        
        # Both models should generate feedback
        transformer_feedback = self.transformer_model.generate_feedback(self.test_question, transformer_answer)
        baseline_feedback = self.baseline_model.generate_feedback(self.test_question, baseline_answer)
        
        self.assertIsInstance(transformer_feedback, Feedback)
        self.assertIsInstance(baseline_feedback, Feedback)

if __name__ == "__main__":
    unittest.main()
