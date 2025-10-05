import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
from utils.text_cleaner import text_cleaner

class FactOpinionClassifier:
    def __init__(self):
        self.fact_indicators = [
            'according to', 'reported', 'stated', 'announced', 'confirmed',
            'data shows', 'study found', 'research indicates', 'statistics show',
            'official', 'government', 'percent', 'number', 'amount', 'total',
            'measured', 'recorded', 'documented', 'verified'
        ]
        
        self.opinion_indicators = [
            'i think', 'i believe', 'in my opinion', 'seems like', 'appears to be',
            'probably', 'likely', 'might', 'could', 'should', 'would',
            'amazing', 'terrible', 'wonderful', 'awful', 'best', 'worst',
            'love', 'hate', 'prefer', 'wish', 'hope', 'feel', 'personally'
        ]
        
        # Initialize a simple pipeline
        self.pipeline = None
        self._create_simple_classifier()
    
    def _create_simple_classifier(self):
        """Create a simple TF-IDF based classifier"""

        self.pipeline = Pipeline([        
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('classifier', LogisticRegression())
        ])
    
    def classify_sentence(self, sentence):
        """Classify a single sentence as fact or opinion"""
        if not sentence or not sentence.strip():
            return {"classification": "unknown", "confidence": 0.0}
        
        sentence_lower = sentence.lower()
        
        # Count fact and opinion indicators
        fact_count = sum(1 for indicator in self.fact_indicators if indicator in sentence_lower)
        opinion_count = sum(1 for indicator in self.opinion_indicators if indicator in sentence_lower)
        
        # Simple rule-based classification
        if fact_count > opinion_count:
            classification = "fact"
            confidence = min(0.9, 0.5 + (fact_count * 0.1))
        elif opinion_count > fact_count:
            classification = "opinion"
            confidence = min(0.9, 0.5 + (opinion_count * 0.1))
        else:
            # Check for other patterns
            if self._has_factual_patterns(sentence):
                classification = "fact"
                confidence = 0.6
            elif self._has_opinion_patterns(sentence):
                classification = "opinion"
                confidence = 0.6
            else:
                classification = "neutral"
                confidence = 0.5
        
        return {
            "classification": classification,
            "confidence": confidence,
            "fact_indicators": fact_count,
            "opinion_indicators": opinion_count
        }
    
    def _has_factual_patterns(self, sentence):
        """Check for factual patterns in sentence"""
        factual_patterns = [
            r'\d+\%',  # percentages
            r'\d+\s*(million|billion|thousand)',  # large numbers
            r'(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+',  # dates
            r'\d+\s*(years|months|days|hours)',  # time periods
            r'(increase|decrease|rise|fall|drop)\s+of\s+\d+',  # statistical changes
        ]
        
        for pattern in factual_patterns:
            if re.search(pattern, sentence.lower()):
                return True
        return False
    
    def _has_opinion_patterns(self, sentence):
        """Check for opinion patterns in sentence"""
        opinion_patterns = [
            r'(very|extremely|incredibly|absolutely|totally)\s+\w+',  # intensifiers
            r'(good|bad|great|terrible|amazing|awful|wonderful|horrible)',  # subjective adjectives
            r'(should|must|need to|have to|ought to)',  # modal verbs
        ]
        
        for pattern in opinion_patterns:
            if re.search(pattern, sentence.lower()):
                return True
        return False
    
    def classify_text(self, text):
        """Classify entire text by analyzing sentences"""
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        sentences = text_cleaner.tokenize_sentences(text)
        if not sentences:
            return {"error": "No sentences found"}
        
        results = []
        fact_count = 0
        opinion_count = 0
        neutral_count = 0
        
        for sentence in sentences:
            result = self.classify_sentence(sentence)
            results.append({
                "sentence": sentence,
                "classification": result["classification"],
                "confidence": result["confidence"]
            })
            
            if result["classification"] == "fact":
                fact_count += 1
            elif result["classification"] == "opinion":
                opinion_count += 1
            else:
                neutral_count += 1
        
        total_sentences = len(sentences)
        
        # Overall classification
        if fact_count > opinion_count:
            overall_classification = "mostly_factual"
        elif opinion_count > fact_count:
            overall_classification = "mostly_opinion"
        else:
            overall_classification = "mixed"
        
        return {
            "overall_classification": overall_classification,
            "sentence_breakdown": results,
            "statistics": {
                "total_sentences": total_sentences,
                "fact_sentences": fact_count,
                "opinion_sentences": opinion_count,
                "neutral_sentences": neutral_count,
                "fact_percentage": round((fact_count / total_sentences) * 100, 1),
                "opinion_percentage": round((opinion_count / total_sentences) * 100, 1)
            }
        }

fact_opinion_classifier = FactOpinionClassifier()