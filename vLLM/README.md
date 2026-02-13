### 1️⃣ **System preperation**
Install the latest **RDNA4** architecture docker vLLM container for Ubuntu 24.04
```bash
docker pull rocm/vllm-dev:rocm7.2_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.14.0rc0
```

### 2️⃣ **Start the vLLM container**
```bash
sudo docker run -it \
    -p 7860:7860 \
    --device=/dev/kfd \
    --device=/dev/dri/card2 \
    --device=/dev/dri/renderD129 \
    --security-opt seccomp=unconfined \
    --group-add video \
    rocm/vllm-dev:rocm7.2_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.14.0rc0
```

| Flag / Option | Purpose |
|---------------|---------|
| `-p 7860:7860` | Exposes port 7860 (commonly used for web UIs or API endpoints). |
| `--device=/dev/kfd` | Grants access to the ROCm kernel driver (required for compute). |
| `--device=/dev/dri/card2` | Passes the physical GPU device into the container. |
| `--device=/dev/dri/renderD129` | Enables render node access for compute workloads. |
| `--security-opt seccomp=unconfined` | Required to avoid ROCm-related syscall restrictions. |
| `--group-add video` | Ensures proper GPU access permissions inside the container. |


rocm/vllm-dev:...
Uses the ROCm 7.2 vLLM development image with:
Ubuntu 24.04
Python 3.12
PyTorch 2.9
vLLM 0.14.0rc0

Notes
Adjust /dev/dri/cardX and /dev/dri/renderDX if your GPU uses different device IDs (ls /dev/dri/ to verify).
Ensure Docker is installed and the ROCm driver is properly configured on the host system.
For production use, consider adding volume mounts for model storage and persistent data.


### 3️⃣ **Update and install** the container environment
```bash
sudo apt update
sudo apt install nano -y
sudo apt install ffmpeg -y
python3 -m pip install --upgrade pip wheel
python3 -m pip install gradio
python3 -m pip install git+https://github.com/openai/whisper.git
python3 -m pip install asyncio
python3 -m pip install edge-tts
```

### 4️⃣ **Download** the Chat Agent script
```bash
wget https://raw.githubusercontent.com/JoergR75/Voice-to-Voice-Chat-Agent/refs/heads/main/vLLM/chat_agent_vllm.py
```

### 3️⃣ **Run** the Chat Agent
```bash
python3 chat_agent_vllm.py
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

