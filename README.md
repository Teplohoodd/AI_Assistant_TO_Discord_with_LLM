# AI Assistant to Discord with LLM

Голосовой Discord-бот, подключённый к языковой модели (LLM), который распознаёт речь, генерирует осмысленные ответы и озвучивает их обратно участникам голосового чата.

## Возможности

- **Speech-to-Text (STT):** преобразует голос пользователя в текст.
- **Language Model Integration:** обрабатывает текст через LLM (например, GPT).
- **Text-to-Speech (TTS):** озвучивает ответы модели в голосовом канале Discord.
- **Интеграция с Discord:** подключается к серверам, слушает участников, и взаимодействует голосом.

## Структура проекта

```
discord-ai-voice-assistant2/
│
├── bot.py                 # Основная логика Discord-бота
├── main.py                # Точка запуска
├── llm.py                 # Взаимодействие с языковой моделью
├── stt.py                 # Модуль распознавания речи
├── tts.py                 # Модуль синтеза речи
├── config/config.yaml     # Конфигурационные параметры
├── utils/logger.py        # Логирование
├── .gitignore
├── requirements.txt
└── src/
    ├── ai/
    │   └── __init__.py
    └── audio/
        └── __init__.py
```

## Установка

1. Клонируй репозиторий:
   ```bash
   git clone https://github.com/Teplohoodd/AI_Assistant_TO_Discord_with_LLM.git
   cd AI_Assistant_TO_Discord_with_LLM
   ```

2. Установи зависимости:
   ```bash
   pip install -r requirements.txt
   ```

3. Установи и запусти локальный сервер с моделью Mixtral, используя CUDA:
   - Скачай модель Mixtral (например, с HuggingFace).
   - Запусти сервер с использованием GPU:
     ```bash
     python -m llama_cpp.server --model ./path_to_mixtral_model.gguf --n_gpu_layers 100
     ```
   Это позволит распределить нагрузку между ЦП и GPU и повысить производительность.

4. Настрой `config/config.yaml` — укажи токен Discord-бота, параметры STT/LLM/TTS и другие настройки.

5. Запусти бота:
   ```bash
   python main.py
   ```

## Технологии

- Python
- Discord.py (или Pycord / nextcord — в зависимости от импорта)
- OpenAI / Llama / иные LLM API
- SpeechRecognition / Whisper
- gTTS / pyttsx3 / или другой TTS-движок

## Планы на будущее

- Управление ботов голосом
- Переход между режимами общения
- Поддержка нескольких пользователей одновременно
- Кеширование LLM-ответов
- Web-интерфейс для конфигурации
