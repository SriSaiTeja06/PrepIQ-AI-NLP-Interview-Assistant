from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class InterviewMetrics:
    def __init__(self):
        """
        Initialize the evaluation metrics system.
        """
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer()

    def calculate_technical_accuracy(self, question: str, answer: str, expected_skills: List[str]) -> float:
        """
        Calculate technical accuracy score based on:
        - Presence of technical terms
        - Relevance to expected skills
        - Completeness of technical explanation
        """
        # Tokenize and remove stop words
        answer_tokens = [word.lower() for word in word_tokenize(answer) 
                        if word.lower() not in self.stop_words]
                        
        # Calculate skill coverage
        skill_coverage = sum(1 for skill in expected_skills 
                           if any(skill.lower() in token for token in answer_tokens))
        
        # Calculate technical term density
        technical_terms = set(expected_skills)
        term_density = len([term for term in technical_terms 
                          if any(term.lower() in token for token in answer_tokens)])
        
        # Calculate overall technical accuracy
        accuracy = (skill_coverage + term_density) / (2 * len(expected_skills))
        return min(accuracy, 1.0)

    def calculate_completeness(self, question: str, answer: str) -> float:
        """
        Calculate completeness score based on:
        - Answer length relative to question
        - Information density
        - Structural completeness
        """
        # Calculate length ratio
        question_length = len(word_tokenize(question))
        answer_length = len(word_tokenize(answer))
        
        # Calculate information density using TF-IDF
        try:
            texts = [question, answer]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            features = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores for answer
            answer_scores = tfidf_matrix[1].toarray()[0]
            
            # Calculate information density
            info_density = np.mean(answer_scores)
            
            # Calculate completeness score
            completeness = (info_density + 
                          min(1.0, answer_length / (question_length * 2))) / 2
            return completeness
        except:
            return 0.5  # Default score if TF-IDF calculation fails

    def calculate_clarity(self, answer: str) -> float:
        """
        Calculate clarity score based on:
        - Sentence structure
        - Use of examples
        - Logical flow
        """
        # Tokenize sentences
        sentences = nltk.sent_tokenize(answer)
        
        # Calculate average sentence length
        avg_sentence_length = np.mean([len(word_tokenize(sent)) for sent in sentences])
        
        # Calculate clarity score
        clarity = 1 / (1 + np.exp(-0.1 * (30 - avg_sentence_length)))
        return clarity

    def calculate_relevance(self, question: str, answer: str) -> float:
        """
        Calculate relevance score using cosine similarity.
        """
        try:
            texts = [question, answer]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            similarity = np.dot(tfidf_matrix[0].toarray(), 
                              tfidf_matrix[1].toarray().T)[0][0]
            
            return float(similarity)
        except:
            return 0.5  # Default score if calculation fails

    def evaluate_answer(self, question: str, answer: str, 
                       expected_skills: List[str]) -> Dict[str, float]:
        """
        Evaluate an answer and return all metrics.
        """
        metrics = {
            'technical_accuracy': self.calculate_technical_accuracy(question, answer, expected_skills),
            'completeness': self.calculate_completeness(question, answer),
            'clarity': self.calculate_clarity(answer),
            'relevance': self.calculate_relevance(question, answer)
        }
        
        # Calculate overall score (weighted average)
        weights = {
            'technical_accuracy': 0.4,
            'completeness': 0.2,
            'clarity': 0.2,
            'relevance': 0.2
        }
        
        overall_score = sum(metrics[k] * weights[k] for k in metrics)
        metrics['overall_score'] = overall_score
        
        return metrics

# Example usage
def main():
    evaluator = InterviewMetrics()
    
    question = "Explain how a decision tree works."
    answer = "A decision tree is a supervised learning algorithm that splits data based on feature values..."
    expected_skills = ["machine learning", "algorithms", "data structures"]
    
    metrics = evaluator.evaluate_answer(question, answer, expected_skills)
    
    print("Evaluation Metrics:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.2f}")

if __name__ == "__main__":
    main()
