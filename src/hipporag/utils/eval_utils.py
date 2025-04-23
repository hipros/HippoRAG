import re
import string

def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

class QAExactMatch:
    """Exact match metric for question answering evaluation"""
    
    def __init__(self, normalize=True):
        """
        Initialize the exact match evaluator
        
        Args:
            normalize: Whether to normalize answers before comparison
        """
        self.normalize = normalize
    
    def calculate(self, prediction, reference):
        """
        Calculate exact match between prediction and reference
        
        Args:
            prediction: The predicted answer
            reference: The reference answer
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        if self.normalize:
            prediction = normalize_answer(prediction)
            reference = normalize_answer(reference)
        
        return float(prediction == reference)

class QAF1Score:
    """F1 score metric for question answering evaluation"""
    
    def __init__(self, normalize=True):
        """
        Initialize the F1 score evaluator
        
        Args:
            normalize: Whether to normalize answers before comparison
        """
        self.normalize = normalize
    
    def calculate(self, prediction, reference):
        """
        Calculate F1 score between prediction and reference
        
        Args:
            prediction: The predicted answer
            reference: The reference answer
            
        Returns:
            F1 score between prediction and reference
        """
        if self.normalize:
            prediction = normalize_answer(prediction)
            reference = normalize_answer(reference)
        
        # Tokenize by splitting on whitespace
        prediction_tokens = prediction.split()
        reference_tokens = reference.split()
        
        # If either is empty, handle edge case
        if len(prediction_tokens) == 0 or len(reference_tokens) == 0:
            return 0.0 if len(prediction_tokens) != len(reference_tokens) else 1.0
        
        # Count common tokens
        common_tokens = [token for token in prediction_tokens if token in reference_tokens]
        
        # Empty prediction or reference
        if len(common_tokens) == 0:
            return 0.0
        
        precision = len(common_tokens) / len(prediction_tokens)
        recall = len(common_tokens) / len(reference_tokens)
        
        # Avoid division by zero
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * precision * recall / (precision + recall)
        return f1

class RetrievalRecall:
    """Recall metric for retrieval evaluation"""
    
    def calculate(self, retrieved_docs, gold_docs):
        """
        Calculate recall of retrieved documents against gold documents
        
        Args:
            retrieved_docs: List of retrieved documents
            gold_docs: List of gold documents
            
        Returns:
            Recall score between 0 and 1
        """
        if not gold_docs:
            return 1.0  # If no gold docs, consider perfect recall
        
        # Count how many gold docs are in retrieved docs
        hit = sum(1 for doc in gold_docs if doc in retrieved_docs)
        
        return hit / len(gold_docs)