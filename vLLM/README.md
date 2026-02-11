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


### 2️⃣ **Download the Script from the Repository**
```bash
wget https://raw.githubusercontent.com/JoergR75/rocm-7.2.0-pytorch-docker-cdna-rdna-automated-deployment/refs/heads/main/script_module_ROCm_720_Ubuntu_22.04-24.04_pytorch_server.sh
```

<img width="967" height="275" alt="{88EE1CD5-AD5C-43E7-BF71-84CAB636FC7B}" src="https://github.com/user-attachments/assets/2aee5ebd-1b4b-4066-b030-d5e58d06ddbe" />

### 3️⃣ **Run the Installer**
```bash
bash script_module_ROCm_720_Ubuntu_22.04-24.04_pytorch_server.sh
```
**⚠️ Note**: Entering the user password may be required.

<img width="956" height="327" alt="{036AB9F1-945C-461E-B497-C8F977804405}" src="https://github.com/user-attachments/assets/df538538-32d7-40c3-a67e-cc065c5b8bc0" />

The installation takes ~15 minutes depending on internet speed and hardware performance.
