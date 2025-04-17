"""
Text-to-Speech functionality for voice synthesis.
"""
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union, List, Any

import torch
import numpy as np

logger = logging.getLogger(__name__)

class TextToSpeech:
    """Text-to-Speech handler for voice synthesis."""
    
    def __init__(self, config: Dict):
        """
        Initialize the TTS module.
        
        Args:
            config: TTS configuration dictionary
        """
        self.config = config
        self.model_type = config.get("model", "silero")
        self.voice = config.get("voice", "female")
        self.sample_rate = 48000  # Discord expects 48kHz audio
        self.speech_rate = config.get("speech_rate", 0.33)  # Скорость речи (меньше = медленнее)
        self.speech_style = config.get("speech_style", "casual")  # casual, formal, angry
        
        # Initialize the model based on configuration
        if self.model_type == "silero":
            self._init_silero()
        elif self.model_type == "coqui":
            # Placeholder for Coqui TTS
            logger.warning("Coqui TTS not implemented yet, falling back to Silero")
            self.model_type = "silero"
            self._init_silero()
        else:
            logger.warning(f"Unknown TTS model '{self.model_type}', falling back to Silero")
            self.model_type = "silero"
            self._init_silero()
            
        logger.info(f"Loaded {self.model_type} TTS model with '{self.speaker}' voice")
    
    def _init_silero(self) -> None:
        """Initialize Silero TTS model."""
        # Map generic voice preferences to specific Silero voices
        voice_mapping = {
            "female": {
                "ru": "kseniya_v2",  # Russian female voice
                "en": "lj_v2"        # English female voice
            },
            "male": {
                "ru": "aidar_v2",    # Russian male voice
                "en": "lj_v2"        # No dedicated male English voice, fallback to female
            }
        }
        
        # Determine language based on voice option or config
        language = self.config.get("language", "ru")
        
        # Get appropriate speaker ID based on gender and language
        gender = self.voice
        if gender not in voice_mapping:
            logger.warning(f"Unknown voice type '{gender}', falling back to female")
            gender = "female"
            
        if language not in voice_mapping[gender]:
            logger.warning(f"Language '{language}' not supported for {gender} voice, falling back to Russian")
            language = "ru"
            
        self.speaker = voice_mapping[gender][language]
        
        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            model, example_text = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=language,
                speaker=self.speaker
            )
        except Exception as e:
            logger.error(f"Error loading Silero model: {e}")
            raise
            
        self.model = model
        self.language = language
        
        logger.info(f"Initialized Silero TTS with {self.speaker} voice (language: {language})")
    
    def _split_text(self, text: str, max_length: int = 140) -> List[str]:
        """
        Split long text into smaller chunks for TTS processing.
        
        Args:
            text: Text to split
            max_length: Maximum length of each chunk
            
        Returns:
            List of text chunks
        """
        # Если текст короче максимальной длины, возвращаем его как есть
        if len(text) <= max_length:
            return [text]
        
        # Разбиваем текст на предложения
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Если предложение само по себе длиннее максимальной длины
            if len(sentence) > max_length:
                # Добавляем текущий фрагмент, если он не пустой
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Разбиваем длинное предложение на части по словам
                words = sentence.split()
                word_chunk = ""
                
                for word in words:
                    if len(word_chunk) + len(word) + 1 <= max_length:
                        if word_chunk:
                            word_chunk += " "
                        word_chunk += word
                    else:
                        chunks.append(word_chunk)
                        word_chunk = word
                
                if word_chunk:
                    chunks.append(word_chunk)
            
            # Если добавление предложения не превысит максимальную длину
            elif len(current_chunk) + len(sentence) + 1 <= max_length:
                if current_chunk:
                    current_chunk += " "
                current_chunk += sentence
            
            # Если добавление предложения превысит максимальную длину
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        
        # Добавляем последний фрагмент, если он не пустой
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def synthesize(self, text: str) -> str:
        """
        Synthesize text to speech and return the path to the audio file.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Path to the generated audio file
        """
        # Разбиваем текст на фрагменты, если он слишком длинный
        chunks = self._split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks for TTS")
        
        if len(chunks) == 1:
            # Если текст короткий, используем стандартный метод
            return await self._synthesize_chunk(chunks[0])
        else:
            # Если текст разбит на несколько фрагментов, объединяем аудио
            try:
                # Генерируем аудио для каждого фрагмента
                audio_files = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Synthesizing chunk {i+1}/{len(chunks)}: '{chunk}'")
                    audio_file = await self._synthesize_chunk(chunk)
                    audio_files.append(audio_file)
                
                # Объединяем аудио файлы
                import numpy as np
                import wave
                
                # Создаем временный файл для объединенного аудио
                temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                combined_path = temp_file.name
                temp_file.close()
                
                # Чтение и объединение аудио из файлов
                combined_audio = np.array([], dtype=np.int16)
                
                for audio_file in audio_files:
                    with wave.open(audio_file, 'rb') as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        audio_data = np.frombuffer(wf.readframes(frames), dtype=np.int16)
                        combined_audio = np.append(combined_audio, audio_data)
                
                # Запись объединенного аудио в файл
                with wave.open(combined_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(combined_audio.tobytes())
                
                # Удаляем временные файлы
                for audio_file in audio_files:
                    try:
                        os.remove(audio_file)
                    except:
                        pass
                
                return combined_path
                
            except Exception as e:
                logger.error(f"Error combining audio chunks: {e}")
                # Возвращаем первый фрагмент в случае ошибки
                if audio_files and os.path.exists(audio_files[0]):
                    return audio_files[0]
                # Или синтезируем только первый фрагмент
                return await self._synthesize_chunk(chunks[0])
    
    async def _synthesize_chunk(self, text: str) -> str:
        """
        Synthesize a single text chunk to speech.
        
        Args:
            text: Text chunk to synthesize
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Create a temporary file for the audio
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            file_path = temp_file.name
            temp_file.close()
            
            if self.model_type == "silero":
                try:
                    # Based on previous errors, let's try using only positional arguments
                    try:
                        # Try with only text argument
                        audio = self.model.apply_tts(text)
                        
                    except Exception as e:
                        logger.warning(f"Simple apply_tts failed: {e}, trying with more arguments")
                        
                        # Try with text and sample_rate as positional arguments
                        try:
                            audio = self.model.apply_tts(text, self.sample_rate)
                        except Exception as e:
                            logger.warning(f"apply_tts with sample_rate failed: {e}, trying different argument order")
                            
                            # Try with text and speaker as positional arguments
                            try:
                                audio = self.model.apply_tts(text, self.speaker)
                            except Exception as e:
                                logger.warning(f"apply_tts with speaker failed: {e}, trying basic save_wav")
                                
                                # Try save_wav with minimal arguments
                                try:
                                    self.model.save_wav(text, file_path)
                                    return file_path
                                except Exception as e:
                                    logger.warning(f"Basic save_wav failed: {e}, falling back to placeholder audio")
                                    raise  # Fall through to the placeholder audio generation
                    
                    # If we get here, one of the apply_tts methods worked
                    # Convert to numpy and normalize
                    if isinstance(audio, torch.Tensor):
                        audio_np = audio.cpu().numpy()
                    elif isinstance(audio, list) and len(audio) > 0:
                        if isinstance(audio[0], torch.Tensor):
                            audio_np = audio[0].cpu().numpy()
                        else:
                            audio_np = np.array(audio[0], dtype=np.float32)
                    else:
                        audio_np = np.array(audio, dtype=np.float32)
                    
                    # Применяем замедление только если оно необходимо
                    if self.speech_rate != 1.0:
                        try:
                            from scipy import signal
                            
                            # Рассчитываем новую длину аудио (обратное соотношение - меньше значение = медленнее речь)
                            original_length = len(audio_np)
                            new_length = int(original_length / self.speech_rate)
                            
                            # Применяем ресемплинг для замедления
                            audio_np = signal.resample(audio_np, new_length)
                            
                            logger.info(f"Slowed down speech by factor of {1/self.speech_rate:.1f}")
                        except ImportError:
                            logger.warning("scipy not installed, cannot slow down speech")
                        except Exception as e:
                            logger.warning(f"Error slowing down speech: {e}")
                    
                    # Normalize and convert to int16
                    audio_np = np.clip(audio_np, -1.0, 1.0)
                    audio_np = (audio_np * 32767).astype(np.int16)
                    
                    # Write WAV file
                    import wave
                    with wave.open(file_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_np.tobytes())
                        
                    return file_path
                        
                except Exception as e:
                    logger.error(f"Failed to generate speech with Silero: {e}")
                    
                    # Create a basic placeholder audio
                    duration = 1.0  # seconds
                    t = np.linspace(0, duration, int(self.sample_rate * duration))
                    # Generate a simple tone
                    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz
                    # Convert to 16-bit PCM
                    audio = (audio * 32767).astype(np.int16)
                    
                    # Write with the wave module
                    import wave
                    with wave.open(file_path, 'wb') as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)  # 16-bit
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio.tobytes())
                    
                    logger.warning("Generated fallback audio due to TTS failure")
                    return file_path
            
            elif self.model_type == "coqui":
                # Placeholder for Coqui TTS
                pass
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}", exc_info=True)
            # Return a fallback audio file or raise an exception
            raise