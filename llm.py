"""
Language Model functionality for generating responses.
"""
import json
import logging
import aiohttp
import asyncio
import time
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class LanguageModel:
    """Language model handler for generating text responses."""
    
    def __init__(self, config: Dict):
        """
        Initialize the language model.
        
        Args:
            config: LLM configuration dictionary
        """
        self.config = config
        self.provider = config.get("provider", "ollama")
        self.model_name = config.get("model", "mixtral:8x7b")
        self.api_url = config.get("api_url", "http://localhost:11434/api/chat")
        self.context_length = config.get("context_length", 5)
        self.timeout = config.get("timeout", 90)  # Увеличиваем до 60 секунд (1 минута)
        self.max_retries = config.get("max_retries", 1)  # Уменьшаем количество повторов до 1
        
        logger.info(f"Initialized {self.provider} LLM with model '{self.model_name}'")
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a response based on the conversation history.
        
        Args:
            messages: List of conversation messages
                     
        Returns:
            Generated text response
        """
        start_time = time.time()
        
        # Получаем последнее сообщение пользователя
        last_user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_message = msg["content"]
                break
        
        if not last_user_message:
            return "Пожалуйста, задайте вопрос."
        
        logger.info(f"Generating response for: '{last_user_message}'")
        
        # Пробуем получить ответ от модели с несколькими попытками
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                if self.provider == "ollama":
                    response = await asyncio.wait_for(
                        self._generate_ollama(messages),
                        timeout=self.timeout
                    )
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Generated response in {elapsed:.2f}s: '{response[:50]}...'")
                    return response
                    
                elif self.provider == "text-generation-webui":
                    response = await asyncio.wait_for(
                        self._generate_tgweb(messages),
                        timeout=self.timeout
                    )
                    
                    elapsed = time.time() - start_time
                    logger.info(f"Generated response in {elapsed:.2f}s: '{response[:50]}...'")
                    return response
                    
                else:
                    logger.warning(f"Unknown provider '{self.provider}'")
                    return f"Провайдер '{self.provider}' не поддерживается. Пожалуйста, настройте Ollama."
                    
            except asyncio.TimeoutError:
                retries += 1
                last_error = f"Timeout after {self.timeout}s on attempt {retries}"
                logger.warning(last_error)
                
                # Небольшая пауза перед следующей попыткой
                await asyncio.sleep(1)
                
            except Exception as e:
                retries += 1
                last_error = f"Error on attempt {retries}: {str(e)}"
                logger.error(last_error)
                
                # Небольшая пауза перед следующей попыткой
                await asyncio.sleep(1)
        
        # Если все попытки исчерпаны, возвращаем ошибку
        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        
        return (
            "Не удалось сгенерировать ответ. Убедитесь, что Ollama запущена и доступна. "
            f"Последняя ошибка: {last_error}"
        )
    
    async def _generate_ollama(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using Ollama API.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Generated text response
        """
        try:
            # Format messages for Ollama (which follows OpenAI format)
            formatted_messages = []
            
            # Добавляем системный промпт для направления модели
            system_message = {
                "role": "system",
                "content": (
                    "Ты полезный и дружелюбный ассистент. Давай точные и понятные ответы на вопросы. "
                    "Важно: отвечай кратко (обычно не более 3-4 предложений), чтобы ответ был удобен для прослушивания. "
                    "Избегай длинных списков и сложных структур в ответах."
                )
            }
            formatted_messages.append(system_message)
            
            # Добавляем сообщения из контекста беседы, но ограничиваем их количество
            # для повышения скорости генерации
            recent_messages = messages[-self.context_length:] if len(messages) > self.context_length else messages
            
            for msg in recent_messages:
                formatted_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                if "name" in msg:
                    formatted_msg["name"] = msg["name"]
                    
                formatted_messages.append(formatted_msg)
            
            # Prepare the request payload
            payload = {
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "max_tokens": 300,  # Уменьшаем максимальную длину ответа
                    'keep_alive': True
                }
            }
            
            # Вычисляем URL для API
            chat_url = self.api_url.replace("generate", "chat")
            logger.debug(f"Sending request to: {chat_url}")
            
            # Отправляем запрос к API с таймаутом
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    chat_url,
                    json=payload,
                    timeout=self.timeout - 5  # Оставляем запас в 5 секунд
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        return f"Ошибка API Ollama (код {response.status}): {error_text}"
                    
                    result = await response.json()
                    return result.get("message", {}).get("content", "")
                
        except aiohttp.ClientConnectorError:
            logger.error("Cannot connect to Ollama API. Is Ollama running?")
            raise ConnectionError("Не удалось подключиться к API Ollama. Убедитесь, что Ollama запущена.")
            
        except Exception as e:
            logger.error(f"Error in Ollama generation: {e}")
            raise
    
    async def _generate_tgweb(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using text-generation-webui API.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Generated text response
        """
        # Не реализовано
        return "Поддержка text-generation-webui пока не реализована."