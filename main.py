#!/usr/bin/env python3
"""
Discord AI Voice Assistant
A voice-enabled AI assistant for Discord using STT, TTS, and LLM technologies.
"""
import os
import sys
import yaml
import asyncio
import logging
from pathlib import Path

from bot import DiscordBot
from utils.logger import setup_logger

def load_config() -> dict:
    """Load configuration from YAML file."""
    config_path = Path("PATH_TO_CONFIG")
    
    if not config_path.exists():
        # Create default config if file doesn't exist
        print(f"Configuration file not found at {config_path}")
        print("Creating a default configuration file...")
        
        # Ensure directory exists
        config_path.parent.mkdir(exist_ok=True)
        
        # Default configuration
        # In the load_config function where the default config is defined:  
        default_config = {
            "discord": {
                "token": "YOUR_TOKEN_HERE",
                "command_prefix": "/"
            },
            "voice": {
                "activation_keywords": ["бот", "bot", "ассистент"],
                "deactivation_commands": ["замолчи", "стоп", "stop"],
                "silence_timeout": 300,
                "silence_utterance_chance": 0.3
            },
            "stt": {
                "model": "google",
                "language": "ru"
            },
            "tts": {
                "model": "silero",
                "voice": "female",
                "language": "ru"  # Add this line
            },
            "llm": {
                "provider": "ollama",
                "model": "mixtral:8x7b",
                "api_url": "http://localhost:11434/api/generate",
                "context_length": 50
            },
            "logging": {
                "level": "INFO",
                "file": "bot.log"
            }
        }
        
        # Write default config
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Default configuration created at {config_path}")
        print("Please edit the file to add your Discord bot token and customize settings.")
        print("Then run the program again.")
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    # Verify token is set
    if config["discord"]["token"] == "YOUR_DISCORD_BOT_TOKEN":
        print("Please set your Discord bot token in config/config.yaml")
        sys.exit(1)
        
    return config

async def main() -> None:
    """Main entry point for the application."""
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logger(
        level=getattr(logging, config["logging"]["level"]),
        log_file=config["logging"]["file"]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Discord AI Voice Assistant")
    
    # Initialize and run the bot
    bot = DiscordBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Shutting down bot...")
        await bot.close()
    except Exception as e:
        logger.error(f"Error running bot: {e}", exc_info=True)
        await bot.close()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
