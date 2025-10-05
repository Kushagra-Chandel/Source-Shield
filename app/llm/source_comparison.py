import openai
from openai import OpenAI
import logging
from config import config
import json
import re

class SourceComparisonAnalyzer:
    def __init__(self):
        self.client = None
        self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize OpenAI client with error handling"""
        try:
            if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                logging.info("OpenAI client initialized successfully")
            else:
                logging.warning("OpenAI API key not provided or is placeholder")
                self.client = None
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None
    
    def load_prompt_template(self, prompt_type):
        """Load prompt template from file"""
        try:
            prompt_path = f"prompts/{prompt_type}.txt"
            with open(prompt_path, 'r') as file:
                return file.read().strip()
        except Exception as e:
            logging.error(f"Failed to load prompt template {prompt_type}: {e}")
            return self.get_default_prompt(prompt_type)
    
    def get_default_prompt(self, prompt_type):
        """Get default prompts if files are not available"""
        prompts = {
            "bias_detection": """
            Analyze the following text for bias. Return ONLY a valid JSON object with no extra text:

            Text: {text}

            {{
              "political_bias": "left_leaning",
              "political_confidence": 0.8,
              "emotional_bias": "neutral",
              "emotional_confidence": 0.7,
              "explanation": "Brief explanation here"
            }}
            """,
            
            "fact_opinion": """
            Analyze the following text to classify as facts or opinions. Return ONLY a valid JSON object:

            Text: {text}

            {{
              "overall_classification": "mostly_factual",
              "confidence": 0.8,
              "fact_percentage": 70,
              "opinion_percentage": 30,
              "reasoning": "Brief reasoning here"
            }}
            """,
            
            "source_comparison": """
            Compare the following sources. Return ONLY a valid JSON object:

            Source 1: {source1}
            Source 2: {source2}

            {{
              "factual_consistency": "consistent",
              "similarity_score": 0.8,
              "key_differences": "Main differences here",
              "bias_comparison": "Both sources appear neutral"
            }}
            """
        }
        return prompts.get(prompt_type, "Analyze the following text: {text}")
    
    def _parse_llm_response(self, content):
        """Robust LLM response parsing with multiple fallback strategies"""
        if not content or not content.strip():
            return {"error": "Empty response from LLM"}
        
        # Remove any markdown formatting
        content = content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '')
        if content.startswith('```'):
            content = content.replace('```', '')
        
        # Strategy 1: Try direct JSON parsing
        try:
            parsed = json.loads(content)
            return parsed
        except json.JSONDecodeError as e:
            print(f"Direct JSON parsing failed: {e}")
        
        # Strategy 2: Find and extract JSON block
        try:
            # Look for JSON pattern with curly braces
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, content, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    return parsed
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            print(f"JSON extraction failed: {e}")
        
        # Strategy 3: Try to fix common JSON issues
        try:
            # Fix common issues
            fixed_content = content.strip()
            
            # Add missing closing braces if needed
            open_braces = fixed_content.count('{')
            close_braces = fixed_content.count('}')
            if open_braces > close_braces:
                fixed_content += '}' * (open_braces - close_braces)
            
            # Try parsing fixed content
            parsed = json.loads(fixed_content)
            return parsed
            
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Extract key information with regex
        try:
            result = {}
            
            # Common patterns to extract
            patterns = {
                'political_bias': r'"political_bias":\s*"([^"]+)"',
                'political_confidence': r'"political_confidence":\s*([0-9.]+)',
                'emotional_bias': r'"emotional_bias":\s*"([^"]+)"',
                'emotional_confidence': r'"emotional_confidence":\s*([0-9.]+)',
                'overall_classification': r'"overall_classification":\s*"([^"]+)"',
                'confidence': r'"confidence":\s*([0-9.]+)',
                'explanation': r'"explanation":\s*"([^"]*)"',
                'reasoning': r'"reasoning":\s*"([^"]*)"'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    value = match.group(1)
                    # Convert to float if it's a number
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
            
            if result:
                result['parsing_method'] = 'regex_extraction'
                result['original_response'] = content[:200]
                return result
                
        except Exception as e:
            print(f"Regex extraction failed: {e}")
        
        # Strategy 5: Return structured fallback
        return {
            "analysis": "LLM analysis completed but response format was unclear",
            "confidence": 0.5,
            "parsing_method": "fallback",
            "raw_response": content[:300],
            "note": "Response could not be parsed as JSON",
            "status": "completed_with_issues"
        }
    
    def analyze_bias_with_llm(self, text):
        """Use LLM to analyze bias in text"""
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            prompt_template = self.get_default_prompt("bias_detection")
            prompt = prompt_template.format(text=text[:1500])  # Shorter text
            
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a media bias analyst. Always respond with ONLY a valid JSON object. No additional text, explanations, or markdown formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,  # Shorter response
                temperature=0.1,  # More deterministic
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            print(f"LLM Bias Response: {content[:100]}...")  # Debug
            return self._parse_llm_response(content)
            
        except Exception as e:
            logging.error(f"Error in LLM bias analysis: {e}")
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def classify_fact_opinion_with_llm(self, text):
        """Use LLM to classify facts vs opinions"""
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            prompt_template = self.get_default_prompt("fact_opinion")
            prompt = prompt_template.format(text=text[:1500])
            
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a fact-checking expert. Always respond with ONLY a valid JSON object. No additional text, explanations, or markdown formatting."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1,
                top_p=0.9
            )
            
            content = response.choices[0].message.content
            print(f"LLM Fact/Opinion Response: {content[:100]}...")  # Debug
            return self._parse_llm_response(content)
            
        except Exception as e:
            logging.error(f"Error in LLM fact/opinion analysis: {e}")
            return {"error": f"LLM analysis failed: {str(e)}"}
    
    def compare_sources_with_llm(self, source1, source2):
        """Compare two sources using LLM"""
        if not self.client:
            return {"error": "OpenAI client not available"}
        
        try:
            prompt_template = self.load_prompt_template("source_comparison")
            prompt = prompt_template.format(
                source1=source1[:1500],  # Limit length
                source2=source2[:1500]
            )
            
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert media analyst. Compare sources objectively and provide detailed analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            content = response.choices[0].message.content
            return self._parse_llm_response(content)
            
        except Exception as e:
            logging.error(f"Error in LLM source comparison: {e}")
            return {"error": f"Source comparison failed: {str(e)}"}
    
    def comprehensive_llm_analysis(self, text, additional_sources=None):
        """Perform comprehensive analysis using LLM"""
        results = {}
        
        # Bias analysis
        results["bias_analysis"] = self.analyze_bias_with_llm(text)
        
        # Fact vs opinion analysis
        results["fact_opinion_analysis"] = self.classify_fact_opinion_with_llm(text)
        
        # Source comparison if additional sources provided
        if additional_sources:
            results["source_comparisons"] = []
            for i, source in enumerate(additional_sources):
                comparison = self.compare_sources_with_llm(text, source)
                results["source_comparisons"].append({
                    "source_index": i,
                    "comparison": comparison
                })
        
        return results

source_comparison_analyzer = SourceComparisonAnalyzer()