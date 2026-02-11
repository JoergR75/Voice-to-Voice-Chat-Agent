### 1️⃣ **Update and install** the Python environment
```bash
sudo apt update
python3 -m pip install --upgrade pip wheel
python3 -m pip install gradio
python3 -m pip install git+https://github.com/openai/whisper.git
python3 -m pip install asyncio
python3 -m pip install edge-tts
```

### 2️⃣ **Download** the Chat Agent script
```bash
wget https://raw.githubusercontent.com/JoergR75/Voice-to-Voice-Chat-Agent/refs/heads/main/chat_agent_transformers.py
```

### 3️⃣ **Start** the Chat Agent
```bash
python3 chat_agent_transformers.py
```

### 4️⃣ Launch the Gradio web Agent from another device connected to same network

First, SSH into the web server and forward port **7860**:
```echo
ssh -L 7860:127.0.0.1:7860 ai1@pc1
```
or use the the server IP address
```echo
ssh -L 7860:127.0.0.1:7860 ai1@192.168.178.xxx
```
Now you can open **http://127.0.0.1:7860** in your local browser to access the Gradio Web Agent.

<img width="943" height="1262" alt="{41C95E6D-D768-44D1-B856-A1A43B5B96B3}" src="https://github.com/user-attachments/assets/05730fcf-f9e4-4dee-a2a0-2ff7888ec693" />
