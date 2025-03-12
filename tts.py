"""
Text-to-Speech module using ElevenLabs API.
Handles:
- Converting text scripts to natural-sounding voiceovers
- Managing different voices and speech parameters
- Saving generated audio files
- Providing word-level timing data for karaoke captions
"""

import os
import uuid
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from elevenlabs.client import ElevenLabs
import logging
from pydub import AudioSegment

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSError(Exception):
    """Base exception class for Text-to-Speech errors."""
    pass

class TextToSpeech:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the TTS client.
        
        Args:
            api_key: Optional API key. If not provided, will look for ELEVENLABS_API_KEY env var.
            
        Raises:
            ValueError: If API key is missing or invalid
        """
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError("ELEVENLABS_API_KEY environment variable or api_key parameter is required")
        if not self.api_key.strip():
            raise ValueError("API key cannot be empty")
        if not (self.api_key.startswith("el_") or self.api_key.startswith("sk_")):
            raise ValueError("Invalid API key format. ElevenLabs API keys should start with 'el_' or 'sk_'")
        
        try:
            # Initialize ElevenLabs client
            self.client = ElevenLabs(api_key=self.api_key)
            # Try to list voices to verify API key works
            response = self.client.voices.get_all()
            if not response.voices:
                raise TTSError("No voices available. API key may be invalid or rate limited.")
            logger.info(f"Successfully connected to ElevenLabs API. Found {len(response.voices)} voices.")
            self.available_voices = response.voices
        except Exception as e:
            raise TTSError(f"Failed to initialize ElevenLabs client: {str(e)}")
        
        # Create output directory
        self.output_dir = Path("generated_audio")
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache setup
        self.cache_file = self.output_dir / "tts_cache.json"
        self._load_cache()
        
        # Voice settings for different content types
        self.voice_settings = {
            "narrative": {
                "voice_settings": {
                    "stability": 0.7,
                    "similarity_boost": 0.8,
                    "style": 0.65,
                    "use_speaker_boost": True
                }
            },
            "descriptive": {
                "voice_settings": {
                    "stability": 0.85,
                    "similarity_boost": 0.7,
                    "style": 0.35,
                    "use_speaker_boost": True
                }
            },
            "dialogue": {
                "voice_settings": {
                    "stability": 0.6,
                    "similarity_boost": 0.9,
                    "style": 0.8,
                    "use_speaker_boost": True
                }
            }
        }
        
        # Default voice and model settings
        self.default_voice = "Bella"  # Warm, engaging storyteller voice
        self.default_model = "eleven_turbo_v2"
        
        # Validate default voice exists
        available_voice_names = [v.name for v in self.available_voices]
        if self.default_voice not in available_voice_names:
            logger.warning(f"Default voice '{self.default_voice}' not found. Available voices: {', '.join(available_voice_names)}")
            # Fall back to first available voice
            self.default_voice = available_voice_names[0]
            logger.info(f"Using fallback voice: {self.default_voice}")

    def _load_cache(self):
        """Load the TTS cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            logger.warning(f"Failed to load TTS cache: {e}")
            self.cache = {}
            
    def _save_cache(self):
        """Save the TTS cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save TTS cache: {e}")
            
    def _get_cache_key(self, text: str, voice: str) -> str:
        """Generate a cache key from text and voice."""
        # Normalize text (strip whitespace, lowercase)
        text = text.strip().lower()
        voice = voice.strip().lower()
        
        # Create key string
        key_str = f"{text}|{voice}"
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def _get_cached_audio(self, text: str, voice: str) -> Optional[Dict]:
        """Check if audio exists in cache."""
        cache_key = self._get_cache_key(text, voice)
        
        if cache_key in self.cache:
            cache_data = self.cache[cache_key]
            audio_path = Path(cache_data["path"])
            if audio_path.exists():
                logger.info(f"Found cached audio: {audio_path}")
                return {
                    "path": str(audio_path),
                    "word_timings": cache_data.get("word_timings", [])
                }
            else:
                # Clean up invalid cache entry
                del self.cache[cache_key]
                self._save_cache()
        return None

    def _detect_content_type(self, text: str) -> str:
        """
        Detect the type of content to optimize voice settings.
        
        Args:
            text: The text to analyze
            
        Returns:
            str: Content type ('narrative', 'descriptive', or 'dialogue')
        """
        # Check for scene description markers
        if any(marker in text.lower() for marker in ["scene", "visual", "we see", "camera shows"]):
            return "descriptive"
        
        # Check for dialogue markers
        if any(marker in text for marker in ['"', "'", ":", "says", "exclaims", "asks"]):
            return "dialogue"
            
        # Default to narrative for other content
        return "narrative"

    def generate_speech(self, text: str, voice_name: str = None) -> Dict:
        """
        Converts text to speech using ElevenLabs API with optimized settings.
        Args:
            text: The text to convert to speech
            voice_name: The voice name to use (defaults to self.default_voice)
        Returns:
            Dict: Contains path to the generated audio file and word timing data
                {
                    "path": str,  # Path to audio file
                    "word_timings": List[Tuple[str, float, float]]  # (word, start_time, end_time)
                }
        Raises:
            TTSError: If speech generation or saving fails
        """
        if not text.strip():
            raise TTSError("Cannot generate speech from empty text")

        try:
            # Use default voice if none provided
            voice_name = voice_name or self.default_voice
            
            # Check cache first
            if cached_data := self._get_cached_audio(text, voice_name):
                return cached_data
            
            logger.info(f"Generating speech for text: '{text[:50]}...' using voice: {voice_name}")
            
            # Get content-specific voice settings
            content_type = self._detect_content_type(text)
            settings = self.voice_settings[content_type]["voice_settings"]
            
            # Generate unique filename
            filename = f"voiceover_{uuid.uuid4()}.mp3"
            output_path = self.output_dir / filename
            
            # Find voice by name
            voice = next((v for v in self.available_voices if v.name == voice_name), None)
            if not voice:
                raise TTSError(f"Voice '{voice_name}' not found")
            
            # Generate audio with optimized settings and word timings
            word_timings = []
            
            try:
                # Use the standard TTS endpoint since timestamps aren't working
                logger.info("Generating speech with ElevenLabs API")
                
                # Use the standard text-to-speech endpoint
                audio_generator = self.client.text_to_speech.convert(
                    text=text,
                    voice_id=voice.voice_id,
                    model_id="eleven_turbo_v2",
                    voice_settings={
                        "stability": settings["stability"],
                        "similarity_boost": settings["similarity_boost"],
                        "style": settings.get("style", 0.0),
                        "use_speaker_boost": settings.get("use_speaker_boost", True)
                    },
                    output_format="mp3_44100_128"
                )

                # Save audio file
                with open(output_path, "wb") as f:
                    # Consume the generator and write chunks to file
                    for chunk in audio_generator:
                        f.write(chunk)
                
                # Generate approximate word timings based on audio duration
                # This is a fallback since the API timestamps aren't working
                audio = AudioSegment.from_file(output_path)
                duration = len(audio) / 1000.0  # Convert ms to seconds
                
                # Generate approximate word timings
                word_timings = self._generate_approximate_word_timings(text, duration)
                logger.info(f"Generated approximate word timings for {len(word_timings)} words")
                
            except Exception as e:
                logger.warning(f"Failed to generate speech: {e}. Trying alternative method.")
                # Try an alternative approach if the first one fails
                try:
                    # Use the standard text-to-speech endpoint with a different approach
                    response = self.client.generate(
                        text=text,
                        voice=voice.voice_id,
                        model="eleven_turbo_v2",
                        voice_settings={
                            "stability": settings["stability"],
                            "similarity_boost": settings["similarity_boost"],
                            "style": settings.get("style", 0.0),
                            "use_speaker_boost": settings.get("use_speaker_boost", True)
                        },
                        output_format="mp3_44100_128"
                    )
                    
                    # Save the audio file
                    with open(output_path, "wb") as f:
                        f.write(response.content)
                    
                    # Generate approximate word timings
                    audio = AudioSegment.from_file(output_path)
                    duration = len(audio) / 1000.0  # Convert ms to seconds
                    
                    # Generate approximate word timings
                    word_timings = self._generate_approximate_word_timings(text, duration)
                    logger.info(f"Generated approximate word timings for {len(word_timings)} words")
                    
                except Exception as e2:
                    logger.error(f"Both TTS methods failed: {e}, then {e2}")
                    raise TTSError(f"Failed to generate speech: {e}, then {e2}")
            
            # Get audio duration if not already calculated
            if not duration:
                audio = AudioSegment.from_file(output_path)
                duration = len(audio) / 1000.0  # Convert ms to seconds
            
            # Create result dictionary
            result = {
                "path": str(output_path),
                "word_timings": word_timings,
                "duration": duration
            }
            
            # Update cache
            self._cache_audio(text, voice_name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            raise TTSError(f"Failed to generate speech: {str(e)}")
            
    def _generate_approximate_word_timings(self, text: str, duration: float) -> List[Tuple[str, float, float]]:
        """
        Generate approximate word timings based on text length and audio duration.
        This is a fallback when API-provided timestamps are not available.
        
        Args:
            text: The text that was converted to speech
            duration: The duration of the audio in seconds
            
        Returns:
            List of (word, start_time, end_time) tuples
        """
        words = text.split()
        if not words:
            return []
            
        # Calculate average word duration
        avg_word_duration = duration / len(words)
        
        # Generate evenly spaced word timings
        word_timings = []
        current_time = 0.0
        
        for word in words:
            # Adjust duration based on word length (longer words take more time)
            word_length_factor = len(word) / 5  # Assuming average word length is 5 characters
            word_length_factor = max(0.5, min(2.0, word_length_factor))  # Limit between 0.5x and 2x
            
            word_duration = avg_word_duration * word_length_factor
            
            # Add word timing
            word_timings.append((word, current_time, current_time + word_duration))
            
            # Move to next word
            current_time += word_duration
        
        # Normalize to ensure the last word ends at the audio duration
        if word_timings:
            last_word_end = word_timings[-1][2]
            if last_word_end != duration:
                scale_factor = duration / last_word_end
                word_timings = [(word, start * scale_factor, end * scale_factor) 
                               for word, start, end in word_timings]
        
        return word_timings

    def _cache_audio(self, text: str, voice: str, result: Dict):
        """Cache the generated audio file and word timings."""
        cache_key = self._get_cache_key(text, voice)
        
        # Ensure duration is included in the cache
        if "duration" not in result:
            try:
                audio = AudioSegment.from_file(result["path"])
                result["duration"] = len(audio) / 1000.0  # Convert ms to seconds
            except Exception as e:
                logger.warning(f"Failed to get audio duration for cache: {e}")
                result["duration"] = 0.0  # Default to 0 if we can't get the duration
        
        self.cache[cache_key] = {
            "path": result["path"],
            "word_timings": result.get("word_timings", []),
            "duration": result.get("duration", 0.0)
        }
        self._save_cache() 