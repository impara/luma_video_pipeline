"""
Voice Optimizer module for automatically enhancing voice parameters based on script content.
Analyzes text to determine optimal voice settings for different types of content.
Streamlined to focus only on parameter optimization for the Daniel voice.
"""

import re
import logging
from typing import Dict, Any, List, Optional
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceOptimizer:
    """
    Analyzes script content to determine optimal voice parameters automatically.
    Streamlined to focus only on parameter optimization for the Daniel voice.
    """
    
    def __init__(self):
        """Initialize the voice optimizer with content analysis capabilities."""
        # Emotion keywords for sentiment analysis
        self.emotion_keywords = {
            "positive": [
                "happy", "excited", "joyful", "amazing", "wonderful", "great", "good", "love", 
                "beautiful", "fantastic", "excellent", "perfect", "brilliant", "delighted"
            ],
            "negative": [
                "sad", "angry", "upset", "terrible", "horrible", "bad", "hate", "awful", 
                "disappointed", "unfortunate", "tragic", "gloomy", "depressing"
            ],
            "calm": [
                "peaceful", "calm", "serene", "gentle", "quiet", "relaxed", "soothing",
                "tranquil", "still", "composed", "steady", "balanced"
            ],
            "energetic": [
                "energetic", "vibrant", "dynamic", "lively", "exciting", "thrilling",
                "fast", "quick", "rapid", "swift", "hurried", "bustling"
            ],
            "serious": [
                "serious", "important", "critical", "crucial", "significant", "essential",
                "grave", "solemn", "formal", "stern", "severe", "strict"
            ],
            "humorous": [
                "funny", "humorous", "amusing", "hilarious", "comical", "laughable",
                "witty", "joke", "comedy", "laugh", "humor", "jest"
            ]
        }
        
        # Content type patterns
        self.content_patterns = {
            "educational": [
                r"learn", r"understand", r"explain", r"concept", r"guide", r"tutorial",
                r"how to", r"step by step", r"instruction", r"lesson", r"teach"
            ],
            "storytelling": [
                r"once", r"story", r"adventure", r"journey", r"character", r"world",
                r"tale", r"legend", r"myth", r"narrative", r"plot", r"scene"
            ],
            "promotional": [
                r"amazing", r"incredible", r"best", r"revolutionary", r"offer", r"buy", 
                r"discount", r"limited time", r"exclusive", r"special", r"deal", r"sale"
            ],
            "informational": [
                r"fact", r"information", r"data", r"research", r"study", r"report",
                r"analysis", r"statistics", r"findings", r"evidence", r"conclusion"
            ],
            "inspirational": [
                r"inspire", r"motivate", r"achieve", r"success", r"dream", r"goal",
                r"passion", r"purpose", r"vision", r"mission", r"destiny", r"future"
            ]
        }
        
    def adapt_to_available_voices(self, available_voices: List[str]) -> None:
        """
        Simplified method that no longer needs to adapt voice catalog.
        Kept for compatibility with existing code.
        
        Args:
            available_voices: List of available voice names
        """
        logger.info("Voice adaptation not needed - using Daniel voice only")
        return
            
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for content type, emotion, complexity, and other characteristics.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing analysis results
        """
        # Remove any SSML tags for analysis
        clean_text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize text
        normalized_text = clean_text.lower()
        # Remove extra whitespace
        normalized_text = ' '.join(normalized_text.split())
        
        logger.info(f"Analyzing text: '{normalized_text[:50]}...'")
        
        # Calculate basic text metrics
        word_count = len(normalized_text.split())
        sentence_count = len(re.split(r'[.!?]+', normalized_text))
        avg_words_per_sentence = word_count / max(1, sentence_count)
        
        # Count punctuation
        exclamation_count = normalized_text.count('!')
        question_count = normalized_text.count('?')
        comma_count = normalized_text.count(',')
        
        # Detect dialogue
        has_dialogue = '"' in clean_text or "'" in clean_text or ":" in clean_text
        
        # Detect content types
        content_types = {}
        for content_type, patterns in self.content_patterns.items():
            matches = sum(1 for pattern in patterns if re.search(pattern, normalized_text))
            content_types[content_type] = matches / len(patterns)
        
        # Log content type scores for debugging
        logger.info(f"Content type scores: {', '.join([f'{k}: {v:.2f}' for k, v in content_types.items()])}")
        
        # Determine primary content type
        if content_types:
            # Sort by score and then by name for deterministic results when scores are equal
            primary_content_type = sorted(content_types.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            primary_content_type = "narrative"
        
        # Detect emotions
        emotions = {}
        for emotion, keywords in self.emotion_keywords.items():
            emotion_score = 0
            for keyword in keywords:
                if keyword in normalized_text:
                    # Count occurrences and weight by word count
                    occurrences = normalized_text.count(keyword)
                    emotion_score += occurrences / max(1, word_count)
            emotions[emotion] = emotion_score
        
        # Log emotion scores for debugging
        logger.info(f"Emotion scores: {', '.join([f'{k}: {v:.2f}' for k, v in emotions.items()])}")
        
        # Determine primary emotion
        if emotions:
            # Sort by score and then by name for deterministic results when scores are equal
            primary_emotion = sorted(emotions.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            primary_emotion = "neutral"
        
        # Calculate text complexity
        complexity_score = self._calculate_complexity(clean_text)
        
        logger.info(f"Analysis results - Content: {primary_content_type}, Emotion: {primary_emotion}, Complexity: {complexity_score:.2f}")
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_words_per_sentence": avg_words_per_sentence,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "comma_count": comma_count,
            "has_dialogue": has_dialogue,
            "content_types": content_types,
            "primary_content_type": primary_content_type,
            "emotions": emotions,
            "primary_emotion": primary_emotion,
            "complexity_score": complexity_score
        }
    
    def _calculate_complexity(self, text: str) -> float:
        """
        Calculate text complexity based on sentence length, word length, and punctuation.
        
        Args:
            text: The text to analyze
            
        Returns:
            Complexity score (0.0-1.0)
        """
        # Normalize text
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Calculate average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Calculate sentence length
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / max(1, len(sentences))
        
        # Count complex punctuation
        semicolons = text.count(';')
        colons = text.count(':')
        dashes = text.count('-') + text.count('—')
        
        # Calculate complexity score (0.0-1.0)
        word_length_factor = min(1.0, avg_word_length / 8.0)  # Normalize to 0-1
        sentence_length_factor = min(1.0, avg_sentence_length / 25.0)  # Normalize to 0-1
        punctuation_factor = min(1.0, (semicolons + colons + dashes) / 10.0)  # Normalize to 0-1
        
        # Weighted average
        complexity_score = (
            0.4 * word_length_factor + 
            0.4 * sentence_length_factor + 
            0.2 * punctuation_factor
        )
        
        return complexity_score
    
    def optimize_voice_parameters(self, text: str) -> Dict[str, Any]:
        """
        Determine optimal voice parameters based on text analysis.
        Optimized specifically for the Daniel voice.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dict containing optimized voice parameters
        """
        # Analyze the text
        analysis = self.analyze_text(text)
        
        # Default parameters for Daniel voice
        params = {
            "voice_name": "Daniel",  # Always use Daniel
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.5,
            "use_speaker_boost": True,
            "model": "eleven_turbo_v2"
        }
        
        # Adjust based on content type
        content_type = analysis["primary_content_type"]
        logger.info(f"Optimizing Daniel voice for content type: {content_type}")
        
        if content_type == "educational":
            params["stability"] = 0.7  # More stable/clear
            params["style"] = 0.3      # Less style variation
            
        elif content_type == "storytelling":
            params["stability"] = 0.4  # More expression for stories
            params["style"] = 0.7      # More style variation
            
        elif content_type == "promotional":
            params["stability"] = 0.4  # More expressive
            params["style"] = 0.8      # More style variation
            
        elif content_type == "informational":
            params["stability"] = 0.6  # Balanced stability
            params["style"] = 0.4      # Moderate style
            
        elif content_type == "inspirational":
            params["stability"] = 0.3  # More expressive
            params["style"] = 0.8      # More style variation
        
        # Adjust based on emotion
        emotion = analysis["primary_emotion"]
        logger.info(f"Adjusting for emotion: {emotion}")
        
        # Store original parameters for logging
        original_params = params.copy()
        
        if emotion == "positive":
            params["stability"] = max(0.3, params["stability"] - 0.1)  # More expressive
            params["style"] = min(0.9, params["style"] + 0.1)          # More style
            
        elif emotion == "negative":
            params["stability"] = max(0.3, params["stability"] - 0.1)  # More expressive
            params["style"] = min(0.9, params["style"] + 0.1)          # More style
            
        elif emotion == "calm":
            params["stability"] = min(0.8, params["stability"] + 0.1)  # More stable
            params["style"] = max(0.2, params["style"] - 0.1)          # Less style
            
        elif emotion == "energetic":
            params["stability"] = max(0.2, params["stability"] - 0.2)  # More expressive
            params["style"] = min(0.9, params["style"] + 0.2)          # More style
            
        elif emotion == "serious":
            params["stability"] = min(0.8, params["stability"] + 0.1)  # More stable
            params["style"] = max(0.3, params["style"] - 0.1)          # Less style
            
        elif emotion == "humorous":
            params["stability"] = max(0.3, params["stability"] - 0.1)  # More expressive
            params["style"] = min(0.9, params["style"] + 0.1)          # More style
        
        # Adjust for text complexity
        complexity = analysis["complexity_score"]
        logger.info(f"Adjusting for complexity: {complexity:.2f}")
        
        if complexity > 0.7:  # High complexity
            params["stability"] = min(0.8, params["stability"] + 0.1)  # More stable for clarity
            params["model"] = "eleven_multilingual_v2"                 # Higher quality model
            
        elif complexity < 0.3:  # Low complexity
            params["stability"] = max(0.3, params["stability"] - 0.1)  # More expressive
            
        # Adjust for dialogue
        if analysis["has_dialogue"]:
            logger.info("Adjusting for dialogue")
            params["style"] = min(0.9, params["style"] + 0.1)  # More style for dialogue
            
        # Adjust for questions and exclamations
        if analysis["question_count"] > 0 or analysis["exclamation_count"] > 0:
            question_exclamation_ratio = (analysis["question_count"] + analysis["exclamation_count"]) / max(1, analysis["sentence_count"])
            if question_exclamation_ratio > 0.3:  # If more than 30% of sentences are questions/exclamations
                logger.info(f"Adjusting for questions/exclamations (ratio: {question_exclamation_ratio:.2f})")
                params["stability"] = max(0.3, params["stability"] - 0.1)  # More expressive
                params["style"] = min(0.9, params["style"] + 0.1)          # More style
        
        # Model selection based on length and complexity
        if analysis["word_count"] < 100 and complexity < 0.5:
            params["model"] = "eleven_turbo_v2"  # Faster for short, simple content
        else:
            params["model"] = "eleven_multilingual_v2"  # Higher quality for longer or complex content
            
        # Log parameter changes
        logger.info(f"Voice parameter adjustments for Daniel:")
        logger.info(f"  Content type '{content_type}'")
        logger.info(f"  Stability: {original_params['stability']:.2f} → {params['stability']:.2f}")
        logger.info(f"  Style: {original_params['style']:.2f} → {params['style']:.2f}")
        logger.info(f"  Model: {params['model']}")
        
        return params
    
    def get_voice_settings_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert voice parameters to the format expected by ElevenLabs API.
        
        Args:
            params: Voice parameters from optimize_voice_parameters
            
        Returns:
            Dict formatted for ElevenLabs API
        """
        return {
            "stability": params["stability"],
            "similarity_boost": params["similarity_boost"],
            "style": params["style"],
            "use_speaker_boost": params["use_speaker_boost"]
        } 