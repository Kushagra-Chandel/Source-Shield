from pymongo import MongoClient
from datetime import datetime
from config import config
import logging

class DatabaseManager:
    def __init__(self):
        try:
            self.client = MongoClient(config.MONGODB_URI)
            self.db = self.client[config.DATABASE_NAME]
            self.collection = self.db[config.COLLECTION_NAME]
            logging.info("Connected to MongoDB successfully")
        except Exception as e:
            logging.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
    
    def save_analysis(self, url, content, analysis_result):
        """Save analysis results to MongoDB"""
        if self.collection is None:
            return False
        
        document = {
            "url": url,
            "content": content[:500],  # Store first 500 chars
            "analysis": analysis_result,
            "timestamp": datetime.now(),
            "created_at": datetime.now()
        }
        
        try:
            result = self.collection.insert_one(document)
            logging.info(f"Analysis saved with ID: {result.inserted_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to save analysis: {e}")
            return False
    
    def get_recent_analyses(self, limit=10):
        """Get recent analysis results"""
        if self.collection is None:
            return []
        
        try:
            results = self.collection.find().sort("timestamp", -1).limit(limit)
            return list(results)
        except Exception as e:
            logging.error(f"Failed to get recent analyses: {e}")
            return []
    
    def search_by_url(self, url):
        """Search for existing analysis by URL"""
        if self.collection is None:
            return None
        
        try:
            result = self.collection.find_one({"url": url})
            return result
        except Exception as e:
            logging.error(f"Failed to search by URL: {e}")
            return None

db_manager = DatabaseManager()