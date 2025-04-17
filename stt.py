"""
Speech-to-Text functionality for voice recognition.
"""
import io
import logging
import os
from typing import Dict, Optional

import speech_recognition as sr
# Change this:
# import whisper
# To this:
import whisper
import numpy as np

logger = logging.getLogger(__name__)

# Rest of the code

class SpeechToText:
    """Speech-to-Text handler for voice recognition."""
    
    def __init__(self, config: Dict):
        """
        Initialize the STT module.
        
        Args:
            config: STT configuration dictionary
        """
        self.config = config
        self.recognizer = sr.Recognizer()
        self.model_type = config.get("model", "whisper")
        self.language = config.get("language", "ru")
        
        # Initialize the model based on configuration
        if self.model_type == "whisper":
            self.model = whisper.load_model("base")
            logger.info("Loaded Whisper STT model")
        elif self.model_type == "vosk":
            # Implementation for Vosk would go here
            logger.info("Vosk not implemented yet, falling back to Whisper")
            self.model_type = "whisper"
            self.model = whisper.load_model("base")
        else:
            logger.warning(f"Unknown STT model '{self.model_type}', falling back to Whisper")
            self.model_type = "whisper"
            self.model = whisper.load_model("base")
    
    async def transcribe_audio(self, audio_data: bytes) -> Optional[str]:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_data: Audio data in bytes
            
        Returns:
            Transcribed text or None if recognition failed
        """
        try:
            if self.model_type == "whisper":
                # Convert audio bytes to numpy array
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Transcribe with Whisper
                result = self.model.transcribe(
                    audio_np, 
                    language=self.language,
                    fp16=False
                )
                
                return result["text"].strip()
            
            elif self.model_type == "vosk":
                # Placeholder for Vosk implementation
                pass
                
            else:
                # Fallback to Google Speech Recognition
                with sr.AudioFile(io.BytesIO(audio_data)) as source:
                    audio = self.recognizer.record(source)
                    return self.recognizer.recognize_google(audio, language=self.language)
                    
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}", exc_info=True)
            return None

    def is_activation_word(self, text: str) -> bool:
        """
        Check if the transcribed text contains an activation keyword.
        
        Args:
            text: Transcribed text
            
        Returns:
            True if activation word is present, False otherwise
        """
        if not text:
            return False
            
        text = text.lower()
        activation_keywords = [kw.lower() for kw in self.config.get("activation_keywords", [])]
        
        return any(keyword in text for keyword in activation_keywords)
        
    def is_deactivation_command(self, text: str) -> bool:
        """
        Check if the transcribed text contains a deactivation command.
        
        Args:
            text: Transcribed text
            
        Returns:
            True if deactivation command is present, False otherwise
        """
        if not text:
            return False
            
        text = text.lower()
        deactivation_commands = [cmd.lower() for cmd in self.config.get("deactivation_commands", [])]
        
        return any(command in text for command in deactivation_commands)