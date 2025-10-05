from transformers import pipeline
import logging
from utils.text_cleaner import text_cleaner

class SentimentBiasAnalyzer:
    def __init__(self):
        try:
            # Load sentiment analysis pipeline
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
        except Exception as e:
            logging.error(f"Failed to load sentiment model: {e}")
            self.sentiment_analyzer = None
        
        # Bias keywords for political detection
        self.political_bias_keywords = {
            'left_leaning': [
                'progressive', 'liberal', 'social justice', 'inequality', 'climate change',
                'healthcare for all', 'minimum wage', 'diversity', 'inclusion', 'regulation'
            ],
            'right_leaning': [
                'conservative', 'traditional values', 'free market', 'deregulation', 'tax cuts',
                'border security', 'law and order', 'second amendment', 'small government'
            ],
            'neutral': [
                'according to', 'reported', 'stated', 'official', 'data shows', 'study found'
            ]
        }
        
        # Emotional bias indicators
        self.emotional_bias_keywords = {
            'highly_emotional': [
                'outrageous', 'shocking', 'incredible', 'unbelievable', 'devastating',
                'amazing', 'terrible', 'horrific', 'wonderful', 'spectacular'
            ],
            'moderate_emotional': [
                'concerning', 'interesting', 'notable', 'significant', 'important',
                'positive', 'negative', 'good', 'bad', 'better', 'worse'
            ]
        }
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        if not self.sentiment_analyzer:
            return {"error": "Sentiment analyzer not available"}
        
        try:
            # Truncate text to avoid model limitations
            # Most sentiment models have token limits around 512 tokens
            # Approximate: 1 token ≈ 4 characters for English
            max_chars = 400  # Safe limit for most models
            text_truncated = text[:max_chars] if len(text) > max_chars else text
            
            # Additional validation
            if len(text_truncated.strip()) < 10:
                return {"error": "Text too short for meaningful analysis"}
            
            results = self.sentiment_analyzer(text_truncated)
            
            # Process results
            sentiment_scores = {}
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                sentiment_scores[label] = round(score, 3)
            
            # Determine primary sentiment
            primary_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[primary_sentiment]
            
            return {
                "primary_sentiment": primary_sentiment,
                "confidence": confidence,
                "all_scores": sentiment_scores,
                "text_length": len(text),
                "analyzed_length": len(text_truncated),
                "truncated": len(text) > max_chars
            }
            
        except Exception as e:
            logging.error(f"Error in sentiment analysis: {e}")
            # Fallback to simple rule-based sentiment
            return self._fallback_sentiment_analysis(text)
    
    def detect_political_bias(self, text):
        """Detect political bias in text"""
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        text_lower = text.lower()
        
        # Count keywords for each bias type
        left_count = sum(1 for keyword in self.political_bias_keywords['left_leaning'] 
                        if keyword in text_lower)
        right_count = sum(1 for keyword in self.political_bias_keywords['right_leaning'] 
                         if keyword in text_lower)
        neutral_count = sum(1 for keyword in self.political_bias_keywords['neutral'] 
                           if keyword in text_lower)
        
        # Calculate bias score
        total_political_keywords = left_count + right_count
        
        if total_political_keywords == 0:
            bias_classification = "neutral"
            bias_confidence = 0.5
        elif left_count > right_count:
            bias_classification = "left_leaning"
            bias_confidence = min(0.9, 0.5 + (left_count / (total_political_keywords + 1)) * 0.4)
        elif right_count > left_count:
            bias_classification = "right_leaning"
            bias_confidence = min(0.9, 0.5 + (right_count / (total_political_keywords + 1)) * 0.4)
        else:
            bias_classification = "balanced"
            bias_confidence = 0.6
        
        return {
            "political_bias": bias_classification,
            "confidence": round(bias_confidence, 3),
            "keyword_counts": {
                "left_leaning": left_count,
                "right_leaning": right_count,
                "neutral": neutral_count
            }
        }
    
    def detect_emotional_bias(self, text):
        """Detect emotional bias in text"""
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        text_lower = text.lower()
        
        # Count emotional keywords
        highly_emotional_count = sum(1 for keyword in self.emotional_bias_keywords['highly_emotional'] 
                                   if keyword in text_lower)
        moderate_emotional_count = sum(1 for keyword in self.emotional_bias_keywords['moderate_emotional'] 
                                     if keyword in text_lower)
        
        total_words = len(text.split())
        
        # Calculate emotional bias
        if highly_emotional_count > 0:
            emotional_level = "highly_emotional"
            emotional_score = min(0.9, (highly_emotional_count / total_words) * 10)
        elif moderate_emotional_count > 0:
            emotional_level = "moderately_emotional"
            emotional_score = min(0.7, (moderate_emotional_count / total_words) * 10)
        else:
            emotional_level = "neutral"
            emotional_score = 0.3
        
        return {
            "emotional_bias": emotional_level,
            "emotional_score": round(emotional_score, 3),
            "keyword_counts": {
                "highly_emotional": highly_emotional_count,
                "moderately_emotional": moderate_emotional_count
            }
        }
    
    def comprehensive_bias_analysis(self, text):
        """Perform comprehensive bias analysis"""
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        # Get all analysis results
        sentiment_result = self.analyze_sentiment(text)
        political_bias_result = self.detect_political_bias(text)
        emotional_bias_result = self.detect_emotional_bias(text)
        
        # Combine results
        return {
            "sentiment_analysis": sentiment_result,
            "political_bias": political_bias_result,
            "emotional_bias": emotional_bias_result,
            "text_length": len(text),
            "word_count": len(text.split())
        }

    def _fallback_sentiment_analysis(self, text):
        """Simple rule-based sentiment analysis as fallback"""
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive', 'success', 'growth', 'improve']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'negative', 'failure', 'decline', 'crisis', 'problem']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {
                "primary_sentiment": "positive",
                "confidence": 0.6,
                "all_scores": {"positive": 0.6, "neutral": 0.3, "negative": 0.1},
                "method": "fallback_rule_based"
            }
        elif negative_count > positive_count:
            return {
                "primary_sentiment": "negative", 
                "confidence": 0.6,
                "all_scores": {"positive": 0.1, "neutral": 0.3, "negative": 0.6},
                "method": "fallback_rule_based"
            }
        else:
            return {
                "primary_sentiment": "neutral",
                "confidence": 0.5,
                "all_scores": {"positive": 0.25, "neutral": 0.5, "negative": 0.25},
                "method": "fallback_rule_based"
            }

sentiment_bias_analyzer = SentimentBiasAnalyzer()