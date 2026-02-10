import gradio as gr
import torch
import whisper
import tempfile
import asyncio
import edge_tts
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# -----------------------------
# Model config
# -----------------------------
MODEL_ID = "DavidAU/Llama3.3-8B-Instruct-Thinking-Heretic-Uncensored-Claude-4.5-Opus-High-Reasoning"
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# -----------------------------
# Whisper model
# -----------------------------
whisper_model = whisper.load_model("base")

# -----------------------------
# TTS
# -----------------------------
VOICE_NAME = "en-US-AriaNeural"

async def speak_async(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
        path = f.name
    communicate = edge_tts.Communicate(text, voice=VOICE_NAME)
    await communicate.save(path)
    return path

def speak(text):
    return asyncio.run(speak_async(text))

# -----------------------------
# vLLM Model & Chat function
# -----------------------------
def init_llm():
    return LLM(
        model=MODEL_ID,
        dtype="float16",
        max_model_len=88000,
    )

def chat_llama(llm, user_input, history):
    messages = []
    system_prompt = (
        "You are Eva, Joergs fast, local AI assistant running on AMD Ryzen AI hardware. "
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
# Main script entry
# -----------------------------
if __name__ == "__main__":
    llm = init_llm()

    # Gradio setup (everything else stays the same)
    with gr.Blocks(title="Llama 3.3 Local AI Agent (AMD ROCm 7.2)") as demo:
        gr.Markdown(
            "# 🤖🎙️ Sarcastic & Funny AI Chat Agent\n"
            "## Model: 🦙 Llama 3.3 8B Instruct\n"
            "## ASR (automatic speech recognition): OpenAI Whisper - base 74M parameters\n"
        )
        chatbot = gr.Chatbot()
        state = gr.State([])

        txt = gr.Textbox(label="Type your message", placeholder="Press Enter to send...", lines=1)
        mic = gr.Audio(label="Speak", type="filepath", sources=["microphone"])
        audio_out = gr.Audio(label="AI Voice Reply", autoplay=True)

        # Text input
        txt.submit(
            lambda text, hist: text_to_chat(llm, text, hist),
            inputs=[txt, state],
            outputs=[chatbot, state, audio_out]
        ).then(lambda: "", None, txt)

        # Speech input
        mic.stop_recording(
            lambda audio, hist: speech_to_chat(llm, audio, hist),
            inputs=[mic, state],
            outputs=[chatbot, state, audio_out]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)
