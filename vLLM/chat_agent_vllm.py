import gradio as gr
import whisper
import tempfile
import asyncio
import edge_tts

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# -----------------------------
# Model configuration
# -----------------------------
MODEL_ID = "DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# -----------------------------
# Whisper (Speech → Text)
# -----------------------------
whisper_model = whisper.load_model("base")

# -----------------------------
# Text to Speech (Edge-TTS)
# -----------------------------
VOICE_NAME = "en-US-AriaNeural"

async def speak_async(text):
    """Generate speech using edge-tts and save to a temp file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name
    communicate = edge_tts.Communicate(text, voice=VOICE_NAME)
    await communicate.save(path)
    return path

def speak(text):
    """Sync wrapper for edge-tts"""
    return asyncio.run(speak_async(text))

# -----------------------------
# LLM Chat Function with personality (vLLM)
# -----------------------------
def chat_llama(llm, user_input, history):
    messages = []

    system_prompt = (
        "You are Eva, Jörg’s fast, local AI assistant running on AMD Ryzen AI hardware. "
        "Respond with sharp wit and dry humor. Keep replies short, clear, and confident. "
        "Be helpful first, funny second. "
        "Occasionally reference speed, efficiency, or running locally when relevant. "
        "No long explanations unless requested."
    )

    messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    sampling_params = SamplingParams(
        max_tokens=128,
        temperature=0.8,
        top_p=0.9,
    )

    outputs = llm.generate([prompt], sampling_params)
    answer = outputs[0].outputs[0].text.strip()

    audio_path = speak(answer)

    return answer, audio_path

# -----------------------------
# Audio Input Handler
# -----------------------------
def text_to_chat(llm, text, history):
    answer, audio_out = chat_llama(llm, text, history)

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})

    return history, history, audio_out

# -----------------------------
# Speech Input Handler
# -----------------------------
def speech_to_chat(llm, audio, history):
    result = whisper_model.transcribe(audio)
    text = result["text"]

    answer, audio_out = chat_llama(llm, text, history)

    history.append({"role": "user", "content": text})
    history.append({"role": "assistant", "content": answer})

    return history, history, audio_out

# -----------------------------
# Main (REQUIRED for vLLM spawn)
# -----------------------------
if __name__ == "__main__":

    llm = LLM(
        model=MODEL_ID,
        dtype="float16",
        max_model_len=88000,
    )

    # -----------------------------
    # Gradio UI
    # -----------------------------
    with gr.Blocks(title="Llama 3.3 Local AI Agent (AMD ROCm 7.2)") as demo:
        gr.Markdown(
            "# 🤖🎙️ Sarcastic & Funny AI Chat Agent\n"
            "## Model: 🦙 Llama 3.3 8B Instruct\n"
            "## ASR (automatic speech recognition): OpenAI Whisper - base 74M parameters\n"
            "[![ROCm](https://img.shields.io/badge/ROCm-7.2.0-ff6b6b?logo=amd)](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html)"
            "[![Whisper GitHub repo](https://img.shields.io/badge/Whisper-GitHub_repo-blue)](https://github.com/JoergR75/whisper_rocm_transcribe/tree/main/whisper_gradio_web_ui)"
            "[![Gradio Quickstart](https://img.shields.io/badge/Gradio-Quickstart-blue)](https://www.gradio.app/guides/quickstart)"
            "[![Docker](https://img.shields.io/badge/Docker-29.2.0-blue?logo=docker)](https://www.docker.com/)"
            "[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04%20%7C%2024.04-e95420?logo=ubuntu)](https://ubuntu.com/download/server)"
            "[![AMD Radeon AI PRO R9700](https://img.shields.io/badge/AMD-RDNA4%20Radeon(TM)%20AI%20PRO%20R9700-8B0000?logo=amd)](https://www.amd.com/)"
        )

        gr.Markdown("Talk or type. Audio runs fully local on one Radeon(TM) AI PRO R9700 GPU.")

        chatbot = gr.Chatbot()
        state = gr.State([])

        txt = gr.Textbox(
            label="Type your message",
            placeholder="Press Enter to send...",
            lines=1
        )

        mic = gr.Audio(
            label="Speak",
            type="filepath",
            sources=["microphone"]
        )

        audio_out = gr.Audio(label="AI Voice Reply", autoplay=True)

        # TEXT
        txt.submit(
            lambda text, hist: text_to_chat(llm, text, hist),
            inputs=[txt, state],
            outputs=[chatbot, state, audio_out]
        ).then(lambda: "", None, txt)

        # SPEECH
        mic.stop_recording(
            lambda audio, hist: speech_to_chat(llm, audio, hist),
            inputs=[mic, state],
            outputs=[chatbot, state, audio_out]
        )

    demo.launch(server_name="localhost", server_port=7860)
