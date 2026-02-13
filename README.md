# 🦙 Llama 3.3 Local AI Voice Agent (AMD ROCm + vLLM)

A fully local, GPU-accelerated AI voice assistant powered by vLLM, Gradio, OpenAI Whisper, and Microsoft Edge TTS — running entirely on AMD ROCm hardware.

No cloud. No API keys. Just fast, local inference.

## 🚀 Overview

This project builds a sarcastic, voice-enabled AI assistant named Eva, running locally using:

- 🧠 LLM: Llama 3.3 8B Instruct (via vLLM)

- 🎙️ Speech-to-Text: Whisper (base)

- 🔊 Text-to-Speech: Edge-TTS (AriaNeural voice)

- 🌐 UI: Gradio web interface

- ⚡ Inference Engine: vLLM

- 🖥️ GPU Platform: AMD ROCm

Everything runs 100% locally on an AMD GPU with ROCm support.

## 🧠 Features

- 💬 Text-based chat

- 🎙️ Voice input (microphone → Whisper → LLM)

- 🔊 AI voice responses (Edge-TTS)

- ⚡ High-speed inference with vLLM

- 🧩 Custom personality system prompt

- 🖥️ Fully local GPU execution

- 🔁 Persistent chat history within session

## 🏗 Architecture

**Pipeline Flow:**

Microphone → Whisper → Llama 3.3 (vLLM) → Edge-TTS → Audio Playback

**Core Components**

- Model loading via vllm.LLM

- Chat template handling with Hugging Face tokenizer

- Async TTS wrapped for synchronous use

- Gradio Blocks UI with:

- Chatbot display

- Text input

- Microphone input

- Autoplay audio responses

## ⚙️ Model Configuration
```python
MODEL_ID = "DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning"

SamplingParams(
    max_tokens=128,
    temperature=0.8,
    top_p=0.9
)
```

- Short, sharp responses

- Dry humor personality

- Optimized for speed and responsiveness

## 🖥 Hardware & Platform

Tested on:

- AMD Radeon™ AI PRO R9700 (RDNA4)

- ROCm 7.2

- Ubuntu 22.04 / 24.04

- PyTorch 2.11 (Preview)

- vLLM 0.14

Designed specifically for AMD GPU acceleration.

## 🚀 Installation

### 1️⃣ **Update and install** the Python environment
```bash
sudo apt update
sudo apt install ffmpeg -y
python3 -m pip install --upgrade pip wheel --break-system-packages
python3 -m pip install gradio --break-system-packages
python3 -m pip install git+https://github.com/openai/whisper.git --break-system-packages
python3 -m pip install asyncio --break-system-packages
python3 -m pip install edge-tts --break-system-packages
```

### 2️⃣ **Download** the Chat Agent script
```bash
wget https://raw.githubusercontent.com/JoergR75/Voice-to-Voice-Chat-Agent/refs/heads/main/chat_agent_transformers.py
```

### 3️⃣ **Run** the Chat Agent
```bash
python3 chat_agent_transformers.py
```

### 4️⃣ Launch the Gradio web Agent from another device connected to same network

First, SSH into the web server and forward port **7860**:
```echo
ssh -L 7860:0.0.0.0:7860 ai1@pc1
```
or use the the server IP address
```echo
ssh -L 7860:0.0.0.0:7860 ai1@192.168.178.xxx
```
Now you can open **http://localhost:7860** in your local browser to access the Gradio Web Agent.

<img width="943" height="1262" alt="{41C95E6D-D768-44D1-B856-A1A43B5B96B3}" src="https://github.com/user-attachments/assets/05730fcf-f9e4-4dee-a2a0-2ff7888ec693" />
