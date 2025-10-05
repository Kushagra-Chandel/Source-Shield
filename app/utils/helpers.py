import re
import logging
from datetime import datetime
from urllib.parse import urlparse

def is_valid_url(url):
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def extract_domain(url):
    """Extract domain from URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return ""

def classify_source_type(url):
    """Classify source type based on URL"""
    domain = extract_domain(url).lower()
    
    news_domains = [
        'bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org',
        'theguardian.com', 'nytimes.com', 'washingtonpost.com',
        'timesofindia.com', 'hindustantimes.com', 'indianexpress.com'
    ]
    
    social_domains = [
        'twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com'
    ]
    
    blog_domains = [
        'medium.com', 'wordpress.com', 'blogspot.com'
    ]
    
    if any(news_domain in domain for news_domain in news_domains):
        return "news"
    elif any(social_domain in domain for social_domain in social_domains):
        return "social"
    elif any(blog_domain in domain for blog_domain in blog_domains):
        return "blog"
    else:
        return "other"

def format_timestamp(timestamp):
    """Format timestamp for display"""
    if isinstance(timestamp, datetime):
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return str(timestamp)

def calculate_confidence_score(scores):
    """Calculate overall confidence score from multiple metrics"""
    if not scores:
        return 0.0
    
    # Simple average of all confidence scores
    total = sum(score for score in scores.values() if isinstance(score, (int, float)))
    count = len([score for score in scores.values() if isinstance(score, (int, float))])
    
    return round(total / count, 2) if count > 0 else 0.0

def truncate_text(text, max_length=100):
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def safe_get(dictionary, key, default=None):
    """Safely get value from dictionary"""
    try:
        return dictionary.get(key, default)
    except:
        return default