"""
Discord bot implementation with voice assistant capabilities.
"""
import asyncio
import logging
import random
import time
import os  # Добавляем импорт os
from typing import Dict, List, Optional, Set, Tuple

import discord
from discord.ext import commands, tasks

from stt import SpeechToText
from tts import TextToSpeech
from llm import LanguageModel

logger = logging.getLogger(__name__)
class DiscordBot:
    """Discord bot with AI voice capabilities."""
    
    def __init__(self, config: dict):
        """
        Initialize Discord bot with the given configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        self.discord_config = config["discord"]
        self.voice_config = config["voice"]
        
        intents = discord.Intents.default()
        intents.message_content = True
        intents.voice_states = True
        
        self.bot = commands.Bot(
            command_prefix=self.discord_config["command_prefix"],
            intents=intents
        )
        
        # Voice client tracking
        self.voice_clients: Dict[int, discord.VoiceClient] = {}
        self.listening_guilds: Set[int] = set()
        
        # User activity tracking
        self.last_activity: Dict[int, float] = {}
        self.conversation_contexts: Dict[int, List[Dict[str, str]]] = {}
        
        # Initialize components
        self.stt = SpeechToText(config["stt"])
        self.tts = TextToSpeech(config["tts"])
        self.llm = LanguageModel(config["llm"])
        
        # Register commands and events
        self._setup_commands()
        self._setup_events()
        
        # Start background tasks
        self.check_silence_background.start()
    
    def _setup_commands(self) -> None:
        """Set up bot commands."""
        @self.bot.command(name="join")
        async def join_voice(ctx: commands.Context):
            """Join the voice channel of the command author."""
            if not ctx.author.voice:
                await ctx.send("You need to be in a voice channel first.")
                return
            
            channel = ctx.author.voice.channel
            guild_id = ctx.guild.id
            
            if guild_id in self.voice_clients:
                await self.voice_clients[guild_id].move_to(channel)
                await ctx.send(f"Moved to {channel.name}")
            else:
                voice_client = await channel.connect()
                self.voice_clients[guild_id] = voice_client
                self.listening_guilds.add(guild_id)
                self.last_activity[guild_id] = time.time()
                self.conversation_contexts[guild_id] = []
                
                await ctx.send(f"Joined {channel.name}")
                await self._speak_in_guild(guild_id, "Привет! Я готов вам помочь.")
                
                # Start listening
                asyncio.create_task(self._listen_in_guild(guild_id, voice_client))
        
        @self.bot.command(name="leave")
        async def leave_voice(ctx: commands.Context):
            """Leave the voice channel in the current guild."""
            guild_id = ctx.guild.id
            
            if guild_id in self.voice_clients:
                await self.voice_clients[guild_id].disconnect()
                self.voice_clients.pop(guild_id)
                self.listening_guilds.discard(guild_id)
                self.last_activity.pop(guild_id, None)
                self.conversation_contexts.pop(guild_id, None)
                
                await ctx.send("Disconnected from voice channel.")
    
    def _setup_events(self) -> None:
        """Set up bot events."""
        @self.bot.event
        async def on_ready():
            logger.info(f"Bot logged in as {self.bot.user.name} ({self.bot.user.id})")
            await self.bot.change_presence(activity=discord.Game(name="Голосовой ассистент"))
        
        
        @self.bot.event
        async def on_message(message: discord.Message):
            # Ignore messages from the bot itself
            if message.author.bot:
                return

            # First, check if this is a command and process it if so
            await self.bot.process_commands(message)
                
            # Then check for activation keywords or bot mentions
            activation_keywords = self.voice_config.get("activation_keywords", [])
            content_lower = message.content.lower()
            
            bot_mentioned = self.bot.user in message.mentions
            keyword_mentioned = any(keyword.lower() in content_lower for keyword in activation_keywords)
            
            # If the bot is mentioned or an activation keyword is used
            if (bot_mentioned or keyword_mentioned) and message.guild:
                # Update activity timestamp
                self.last_activity[message.guild.id] = time.time()
                
                # Extract the actual message content
                user_message = message.content
                
                # Remove bot mention if present
                if bot_mentioned:
                    user_message = user_message.replace(f"<@{self.bot.user.id}>", "").strip()
                
                # Log the activation
                logger.info(f"Bot activation detected in message: '{message.content}'")
                logger.info(f"Processing user message: '{user_message}'")
                
                # Send typing indicator to show the bot is working
                async with message.channel.typing():
                    # Optionally send a temporary message
                    temp_msg = await message.channel.send("⏳ Генерирую ответ, пожалуйста, подождите...")
                    
                    # Process the message with LLM
                    response = await self._process_message(message.guild.id, message.author.name, user_message)
                    
                    # Delete the temporary message
                    try:
                        await temp_msg.delete()
                    except:
                        pass
                    
                    # Log the response
                    logger.info(f"Generated response: '{response}'")
                    
                    # Check if bot is in a voice channel in this guild
                    if message.guild.id in self.voice_clients:
                        # Send text response
                        await message.channel.send(f"**Ответ:** {response}")
                        # Speak the response
                        await self._speak_in_guild(message.guild.id, response)
                    else:
                        # Just send text response if not in voice
                        await message.channel.send(f"**Ответ:** {response}")
        
    async def _listen_in_guild(self, guild_id: int, voice_client: discord.VoiceClient) -> None:
        """
        Listen to audio in a voice channel and process speech.
        
        Args:
            guild_id: ID of the guild
            voice_client: Discord voice client
        """
        # This would require custom FFmpeg setup to capture audio
        # For simplicity, we'll just outline the steps here
        
        # 1. Set up a sink to record audio from the voice channel
        # 2. Process audio chunks with the STT model
        # 3. When activation keyword is detected, start recording until silence
        # 4. Process the complete utterance with LLM
        # 5. Generate and play back the response
        
        # For implementation details, you'd need to use discord.py's audio sinks
        # or a custom solution using PyAudio with FFmpeg
        
        logger.info(f"Started listening in guild {guild_id}")
        
        # Example implementation placeholder
        while guild_id in self.listening_guilds:
            # This would be replaced with actual audio processing
            await asyncio.sleep(1)
            
            # Check if we need to disconnect (example condition)
            if voice_client.is_connected() is False:
                self.listening_guilds.discard(guild_id)
                logger.info(f"Disconnected from guild {guild_id}")
                break
    
    async def _process_message(self, guild_id: int, username: str, message: str) -> str:
        """
        Process a message with the LLM and update conversation context.
        """
        start_time = time.time()
        
        # Add user message to context
        if guild_id not in self.conversation_contexts:
            self.conversation_contexts[guild_id] = []
            
        context = self.conversation_contexts[guild_id]
        context.append({"role": "user", "name": username, "content": message})
        
        # Trim context if needed
        max_context = self.config["llm"]["context_length"]
        if len(context) > max_context:
            context = context[-max_context:]
            self.conversation_contexts[guild_id] = context
        
        # Generate response with extended timeout
        try:
            response = await asyncio.wait_for(
                self.llm.generate_response(context),
                timeout=65.0  # Устанавливаем таймаут чуть больше минуты
            )
        except asyncio.TimeoutError:
            logger.warning("LLM response generation timed out after 65 seconds")
            response = "Извините, генерация ответа заняла слишком много времени. Пожалуйста, попробуйте задать вопрос иначе или повторите попытку позже."
        
        # Add response to context
        context.append({"role": "assistant", "content": response})
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated response in {elapsed_time:.2f} seconds: '{response[:50]}...'")
        
        return response
    
    async def _speak_in_guild(self, guild_id: int, text: str) -> None:
        """
        Generate speech from text and play it in the voice channel.
        
        Args:
            guild_id: ID of the guild
            text: Text to speak
        """
        if guild_id not in self.voice_clients:
            logger.warning(f"Attempted to speak in guild {guild_id} but not connected")
            return
        
        voice_client = self.voice_clients[guild_id]
        
        try:
            # Generate audio from text
            audio_file = await self.tts.synthesize(text)
            
            # Play audio in voice channel
            if voice_client.is_playing():
                voice_client.stop()
                
            # Check operating system
            import platform
            if platform.system() == "Windows":
                # Try to find ffmpeg in common locations
                ffmpeg_locations = [
                    r"C:\ffmpeg\bin\ffmpeg.exe",
                    r"C:\ffmpeg\ffmpeg.exe",
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    r"C:\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe",
                ]
                
                ffmpeg_executable = None
                for location in ffmpeg_locations:
                    if os.path.exists(location):
                        ffmpeg_executable = location
                        break
                        
                if ffmpeg_executable:
                    logger.info(f"Using FFmpeg from: {ffmpeg_executable}")
                    voice_client.play(
                        discord.FFmpegPCMAudio(audio_file, executable=ffmpeg_executable),
                        after=lambda e: logger.error(f"Player error: {e}") if e else None
                    )
                else:
                    # Try using ffmpeg from PATH
                    logger.warning("FFmpeg not found in common locations, trying from PATH")
                    voice_client.play(
                        discord.FFmpegPCMAudio(audio_file),
                        after=lambda e: logger.error(f"Player error: {e}") if e else None
                    )
            else:
                # On non-Windows platforms, assume ffmpeg is in PATH
                voice_client.play(
                    discord.FFmpegPCMAudio(audio_file),
                    after=lambda e: logger.error(f"Player error: {e}") if e else None
                )
            
            # Update activity timestamp
            self.last_activity[guild_id] = time.time()
            
        except Exception as e:
            logger.error(f"Error speaking in guild {guild_id}: {e}", exc_info=True)
    
    @tasks.loop(seconds=30)
    async def check_silence_background(self) -> None:
        """Periodically check for silence in voice channels and generate random utterances."""
        for guild_id in list(self.listening_guilds):
            if guild_id not in self.last_activity:
                continue
                
            silence_time = time.time() - self.last_activity[guild_id]
            silence_timeout = self.voice_config["silence_timeout"]
            
            # If there's been silence for longer than the timeout
            if silence_time > silence_timeout:
                # Only speak with a certain probability
                if random.random() < self.voice_config["silence_utterance_chance"]:
                    # Generate a random utterance
                    context = [{"role": "system", "content": 
                        "Generate a short, friendly comment to break the silence. Keep it under 20 words."}]
                    utterance = await self.llm.generate_response(context)
                    
                    # Speak the utterance
                    await self._speak_in_guild(guild_id, utterance)
    
    @check_silence_background.before_loop
    async def before_check_silence(self) -> None:
        """Wait until the bot is ready before starting the silence check task."""
        await self.bot.wait_until_ready()
    
    async def start(self) -> None:
        """Start the Discord bot."""
        await self.bot.start(self.discord_config["token"])
    
    async def close(self) -> None:
        """Close the Discord bot and all voice connections."""
        for guild_id in list(self.voice_clients.keys()):
            try:
                await self.voice_clients[guild_id].disconnect()
            except:
                pass
                
        self.voice_clients.clear()
        self.listening_guilds.clear()
        self.check_silence_background.cancel()
        
        await self.bot.close()