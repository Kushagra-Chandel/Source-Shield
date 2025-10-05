import os
import yaml
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        # Load configuration from YAML file
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Environment variables
        self.OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
        self.MONGODB_URI = os.getenv('MONGODB_URI') or self.config['database']['mongodb_uri']
        
        # Database settings
        self.DATABASE_NAME = self.config['database']['database_name']
        self.COLLECTION_NAME = self.config['database']['collection_name']
        
        # Model settings
        self.SENTIMENT_MODEL = self.config['models']['sentiment_model']
        self.SIMILARITY_MODEL = self.config['models']['similarity_model']
        
        # OpenAI settings
        self.OPENAI_MODEL = self.config['openai']['model']
        self.MAX_TOKENS = self.config['openai']['max_tokens']
        self.TEMPERATURE = self.config['openai']['temperature']
        
        # Scraping settings
        self.TIMEOUT = self.config['scraping']['timeout']
        self.USER_AGENT = self.config['scraping']['user_agent']

config = Config()