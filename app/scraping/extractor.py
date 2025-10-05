import requests
from newspaper import Article
from bs4 import BeautifulSoup
import logging
from config import config
from utils.helpers import is_valid_url, extract_domain

class ContentExtractor:
    def __init__(self):
        self.timeout = config.TIMEOUT
        self.user_agent = config.USER_AGENT
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def extract_from_url(self, url):
        """Extract content from URL using newspaper3k"""
        if not is_valid_url(url):
            return {"error": "Invalid URL"}
        
        # Check for social media URLs that need special handling
        if self._is_social_media_url(url):
            return self._handle_social_media_url(url)
        
        try:
            # Use newspaper3k for article extraction
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract basic information
            content_data = {
                "url": url,
                "title": article.title or "",
                "content": article.text or "",
                "authors": article.authors or [],
                "publish_date": article.publish_date,
                "summary": article.summary or "",
                "domain": extract_domain(url),
                "word_count": len(article.text.split()) if article.text else 0
            }
            
            return content_data
            
        except Exception as e:
            logging.error(f"Error extracting content from {url}: {e}")
            return self.fallback_extraction(url)
    
    def _is_social_media_url(self, url):
        """Check if URL is from social media platform"""
        social_domains = ['twitter.com', 'x.com', 'facebook.com', 'instagram.com', 'linkedin.com']
        domain = extract_domain(url).lower()
        return any(social_domain in domain for social_domain in social_domains)
    
    def _handle_social_media_url(self, url):
        """Handle social media URLs with appropriate message"""
        domain = extract_domain(url)
        
        return {
            "error": f"Social media URLs ({domain}) are not supported for automatic extraction. Please copy and paste the text content directly.",
            "suggestion": "Copy the tweet/post text and use 'Direct Text' input option instead.",
            "url": url,
            "domain": domain
        }
    
    def fallback_extraction(self, url):
        """Fallback method using BeautifulSoup"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title = ""
            if soup.title:
                title = soup.title.string.strip()
            
            # Extract content from common article tags
            content = ""
            article_tags = soup.find_all(['article', 'div'], class_=lambda x: x and any(
                keyword in x.lower() for keyword in ['content', 'article', 'story', 'post']
            ))
            
            if article_tags:
                content = article_tags[0].get_text(strip=True)
            else:
                # Fallback to body text
                content = soup.get_text(strip=True)
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "authors": [],
                "publish_date": None,
                "summary": content[:200] + "..." if len(content) > 200 else content,
                "domain": extract_domain(url),
                "word_count": len(content.split()) if content else 0
            }
            
        except Exception as e:
            logging.error(f"Fallback extraction failed for {url}: {e}")
            return {"error": f"Failed to extract content: {str(e)}"}
    
    def extract_from_text(self, text):
        """Process direct text input"""
        if not text or not text.strip():
            return {"error": "Empty text provided"}
        
        return {
            "url": "direct_input",
            "title": "Direct Text Input",
            "content": text.strip(),
            "authors": [],
            "publish_date": None,
            "summary": text[:200] + "..." if len(text) > 200 else text,
            "domain": "direct_input",
            "word_count": len(text.split())
        }

content_extractor = ContentExtractor()