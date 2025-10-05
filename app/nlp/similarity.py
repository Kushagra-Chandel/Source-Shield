from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

class TextSimilarityAnalyzer:
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer: {e}")
            self.model = None
    
    def get_embeddings(self, texts):
        """Get embeddings for list of texts"""
        if not self.model:
            return None
        
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logging.error(f"Error getting embeddings: {e}")
            return None
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        embeddings = self.get_embeddings([text1, text2])
        if embeddings is None or len(embeddings) < 2:
            return 0.0
        
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(float(similarity), 3)
    
    def find_similar_sentences(self, target_text, source_texts, threshold=0.5):
        """Find similar sentences from source texts"""
        if not target_text or not source_texts:
            return []
        
        # Get embeddings for target and all source texts
        all_texts = [target_text] + source_texts
        embeddings = self.get_embeddings(all_texts)
        
        if embeddings is None:
            return []
        
        target_embedding = embeddings[0]
        source_embeddings = embeddings[1:]
        
        # Calculate similarities
        similarities = cosine_similarity([target_embedding], source_embeddings)[0]
        
        # Find texts above threshold
        similar_texts = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                similar_texts.append({
                    "text": source_texts[i],
                    "similarity": round(float(similarity), 3),
                    "index": i
                })
        
        # Sort by similarity score
        similar_texts.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_texts
    
    def compare_articles(self, article1, article2):
        """Compare two articles for similarity"""
        if not article1 or not article2:
            return {"error": "Both articles required"}
        
        # Overall similarity
        overall_similarity = self.calculate_similarity(article1, article2)
        
        # Split into sentences and find most similar ones
        from utils.text_cleaner import text_cleaner
        sentences1 = text_cleaner.tokenize_sentences(article1)
        sentences2 = text_cleaner.tokenize_sentences(article2)
        
        # Find similar sentence pairs
        similar_pairs = []
        for i, sent1 in enumerate(sentences1[:10]):  # Limit to first 10 sentences
            for j, sent2 in enumerate(sentences2[:10]):
                similarity = self.calculate_similarity(sent1, sent2)
                if similarity > 0.6:  # High similarity threshold
                    similar_pairs.append({
                        "sentence1": sent1,
                        "sentence2": sent2,
                        "similarity": similarity,
                        "indices": [i, j]
                    })
        
        # Sort similar pairs by similarity
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "overall_similarity": overall_similarity,
            "similar_sentence_pairs": similar_pairs[:5],  # Top 5 pairs
            "total_sentences": [len(sentences1), len(sentences2)]
        }
    
    def detect_content_overlap(self, articles):
        """Detect content overlap among multiple articles"""
        if not articles or len(articles) < 2:
            return {"error": "At least 2 articles required"}
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(articles)):
            for j in range(i + 1, len(articles)):
                similarity = self.calculate_similarity(articles[i], articles[j])
                similarities.append({
                    "article_pair": [i, j],
                    "similarity": similarity
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Calculate average similarity
        avg_similarity = np.mean([s["similarity"] for s in similarities])
        
        return {
            "pairwise_similarities": similarities,
            "average_similarity": round(avg_similarity, 3),
            "highest_similarity": similarities[0] if similarities else None,
            "total_comparisons": len(similarities)
        }

similarity_analyzer = TextSimilarityAnalyzer()