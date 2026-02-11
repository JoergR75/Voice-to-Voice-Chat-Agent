### 1️⃣ **System preperation**
Install **Ubuntu 22.04.5 LTS** or **Ubuntu 24.04.3 LTS** (Server or Desktop version).
```bash
sudo apt update
sudo apt install nano -y
python3 -m pip install --upgrade pip wheel
python3 -m pip install gradio
python3 -m pip install git+https://github.com/openai/whisper.git
python3 -m pip install asyncio
python3 -m pip install edge-tts
```

### 2️⃣ **Lounch the vLLM container**
```bash
sudo docker run -it \
    -p 7860:7860 \
    --device /dev/snd:/dev/snd \
    --group-add audio \
    --device=/dev/kfd \
    --device=/dev/dri/card2 \
    --device=/dev/dri/renderD129 \
    --security-opt seccomp=unconfined \
    --group-add video \
    rocm/vllm-dev:rocm7.2_navi_ubuntu24.04_py3.12_pytorch_2.9_vllm_0.14.0rc0
```

What this does
-p 7860:7860
Exposes port 7860 (commonly used for web UIs or API endpoints).
--device=/dev/kfd
Grants access to the ROCm kernel driver (required for compute).
--device=/dev/dri/card2
Passes the physical GPU device into the container.
--device=/dev/dri/renderD129
Enables render node access for compute workloads.
--security-opt seccomp=unconfined
Required to avoid ROCm-related syscall restrictions.
--group-add video
Ensures proper GPU access permissions inside the container.

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

<img width="967" height="275" alt="{88EE1CD5-AD5C-43E7-BF71-84CAB636FC7B}" src="https://github.com/user-attachments/assets/2aee5ebd-1b4b-4066-b030-d5e58d06ddbe" />

### 3️⃣ **Run the Installer**
```bash
bash script_module_ROCm_720_Ubuntu_22.04-24.04_pytorch_server.sh
```
**⚠️ Note**: Entering the user password may be required.

<img width="956" height="327" alt="{036AB9F1-945C-461E-B497-C8F977804405}" src="https://github.com/user-attachments/assets/df538538-32d7-40c3-a67e-cc065c5b8bc0" />

The installation takes ~15 minutes depending on internet speed and hardware performance.
