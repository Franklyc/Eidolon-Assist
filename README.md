# Eidolon Assist

A personal assistant with voice and image capabilities, featuring a modern UI with rounded corners, gradients, and blur effects.

## Features

- **Voice Input**: Press Ctrl+[ to start recording and Ctrl+] to stop recording and process your request
- **Text Input**: Type your query directly in the text input area
- **Screenshot Capture**: Press Ctrl+P to capture a screenshot to include with your query
- **Text-to-Speech**: Hear responses through high-quality Edge TTS
- **Conversation History**: Keep track of your interactions with the AI
- **Customizable Settings**: Configure API keys, models, voice, hotkeys and more

## Requirements

- Python 3.7+
- PyQt6
- Groq API key (for Whisper transcription)
- OpenAI-compatible API key (for language model responses)

## Installation

1. Clone this repository
2. Install the required packages:

```
pip install -r requirements.txt
```

3. Run the application:

```
python eidolon_assist.py
```

## Configuration

On first run, open the Settings dialog to configure:

1. API Keys:
   - Groq API Key for audio transcription
   - OpenAI-compatible API Key and Base URL for AI responses

2. Models:
   - Whisper model for transcription
   - LLM model for AI responses

3. TTS Voice:
   - Choose from various Edge TTS voices

4. Hotkeys:
   - Customize keyboard shortcuts

5. General:
   - Select microphone
   - Set conversation history limit
   - Customize system prompt

## Hotkeys

- **Ctrl+[**: Start recording
- **Ctrl+]**: Stop recording and process
- **Ctrl+P**: Take a screenshot
- **Ctrl+\\**: Delete last screenshot

## Usage Tips

1. **Voice Commands**: Start recording with Ctrl+[ and speak your query. Stop with Ctrl+].
2. **Mixed Input**: Use screenshots and voice/text together for multimodal queries.
3. **Text Input**: Type questions directly when voice input isn't needed.
4. **Settings**: Configure your API keys before first use.