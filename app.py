import os

# NOTE: This script is designed to run together with SCRIBAL
assert os.path.isdir("utils"), "utils SCRIBAL directory not found!"
assert os.path.isdir("config"), "config SCRIBAL directory not found!"

# HOME = "/home/ppastells/projects/tts"
HOME = "/media/clic/tts"
os.environ["PATH"] += f":{HOME}/espeak-ng/bin"
os.environ["LD_LIBRARY_PATH"] = f"{HOME}/espeak-ng/lib"
os.environ["ESPEAK_DATA_PATH"] = f"{HOME}/espeak-ng/espeak-ng-data"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import re  # noqa: E402
from argparse import Namespace  # noqa: E402

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.utils.quantization_config import BitsAndBytesConfig  # noqa: E402

from matcha_vocos_inference import matxa_alvocat, tts  # noqa: E402
from utils import global_vars  # noqa: E402

args = Namespace()
args.debug = False
args.log_level = "info"
args.keep_audio_files = False
args.language_tool = True
args.secure_api = False
args.preload_models = None
args.context = False
args.translation_context = False
args.logprob_threshold = -0.7

global_vars.init_globals(args)

from utils.utils import transcriber_inference  # noqa: E402

history_visible = gr.State(False)
SPEAKER_ID_DICT = {
    "Balear": {"Quim": 0, "Olga": 1},
    "Central": {"Grau": 2, "Èlia": 3},
    "Nord-occidental": {"Pere": 4, "Emma": 5},
    "Valencià": {"Lluc": 6, "Gina": 7},
}

# Read CSS from file
with open("static/style.css", "r") as f:
    custom_css = f.read()


class ChatBot:
    def __init__(self):
        self.llama_model = None
        self.tokenizer = None
        self.matcha_model = None
        self.vocoder = None
        self.messages = []
        self.is_initialized = False

        # Load configuration
        df = pd.read_csv("secret.config", header=None)
        self.HF_TOKEN = df.iloc[1, 1]

        self.INITIAL_PROMPT = """Ets un xatbot del grup CLiC de la Universitat de Barcelona, que respon educadament en català.
Et dius CLiC-Bot, i has de respondre a les preguntes dels nens que et parlin, siguent amable i divertit.
Avui és la festa de la ciència, i t'estem ensenyant juntament SCRIBAL, el transcriptor del grup.
Les teves respostes passen per un vocoder per ser escoltades en veu alta, han de ser breus.
"""

    def load_models(self, progress=gr.Progress()):
        """Load all models with progress tracking"""
        if self.is_initialized:
            return "Models ja carregats!"

        progress(0.1, desc="Carregant Llama 🦙...")
        self.llama_model, self.tokenizer = self.load_llama()

        progress(0.7, desc="Carregant Matxa 🍵 + alVoCat 🥑...")
        self.matcha_model, self.vocoder = matxa_alvocat()

        progress(1.0, desc="Models carregats!")
        self.is_initialized = True
        self.reset_conversation()
        return "🤖 Models carregats! Llest per conversar!"

    def load_llama(self):
        """Load Llama model"""
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=nf4_config,
            token=self.HF_TOKEN,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=self.HF_TOKEN)
        return llama_model, tokenizer

    def llama_response(self, messages, max_tokens=256):
        """Generate response from Llama"""
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.llama_model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = self.llama_model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        response = outputs[0][input_ids.shape[-1] :]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        # Clean up response
        response = re.sub(r"\s*\*[^*]+\*", "", response)
        response = re.sub(r"\s*\*\*\*[^*]+\*\*", "", response)
        response = re.sub(r"$ [^)]* $ ", "", response)
        response = re.sub("clic", "CLiC", response)
        response = response.strip()
        response = re.sub(r"\(.*\)", "", response)
        return response

    def transcribe_audio(self, audio_data):
        """Transcribe audio using your transcriber"""
        print(f"transcribe_audio called with: {type(audio_data)}")

        if audio_data is None:
            print("Audio data is None")
            return "No audio data received"

        try:
            # Handle different audio input formats from Gradio 5.x
            sample_rate = 16000  # Default
            audio_array = None

            if isinstance(audio_data, dict):
                print(f"Dict format: {audio_data.keys()}")
                if "array" in audio_data:
                    audio_array = audio_data["array"]
                    sample_rate = audio_data.get("sampling_rate", 16000)
                    print(
                        f"Extracted from dict: array shape {audio_array.shape if audio_array is not None else 'None'}"
                    )
                else:
                    return "Dict format but no 'array' key found"

            elif isinstance(audio_data, tuple) and len(audio_data) == 2:
                print(f"Tuple format: length {len(audio_data)}")
                sample_rate, audio_array = audio_data
                print(
                    f"Extracted from tuple: sr={sample_rate}, array shape {audio_array.shape if audio_array is not None else 'None'}"
                )

            elif isinstance(audio_data, np.ndarray):
                print("Direct numpy array")
                audio_array = audio_data
            else:
                error_msg = f"Unexpected audio format: {type(audio_data)}"
                print(error_msg)
                return error_msg

            # Validate audio array
            if audio_array is None:
                return "Audio array is None"

            if hasattr(audio_array, "size") and audio_array.size == 0:
                return "Audio array is empty"

            # Convert to numpy if tensor
            if torch.is_tensor(audio_array):
                audio_array = audio_array.detach().cpu().numpy()
                print("Converted tensor to numpy")

            # Check if we actually have audio data
            if len(audio_array) == 0:
                return "Empty audio data after processing"

            # Handle multi-channel audio
            original_shape = audio_array.shape
            if len(audio_array.shape) > 1:
                print(f"Multi-channel audio detected: {audio_array.shape}")
                if audio_array.shape[0] > audio_array.shape[1]:
                    # More samples than channels, audio is (samples, channels)
                    audio_array = audio_array[:, 0]
                else:
                    # More channels than samples, audio is (channels, samples)
                    audio_array = audio_array[0, :]
                print(f"After channel selection: {audio_array.shape}")

            print(f"Original dtype: {audio_array.dtype}")
            print(
                f"Original range before conversion: [{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]"
            )

            if audio_array.dtype == np.int16:
                # Convert int16 to float32
                audio_array = audio_array.astype(np.float32) / 32768.0
                print("Converted int16 to float32")
            elif audio_array.dtype == np.int32:
                # Convert int32 to float32
                audio_array = audio_array.astype(np.float32) / 2147483648.0
                print("Converted int32 to float32")
            elif audio_array.dtype == np.float64:
                audio_array = audio_array.astype(np.float32)
                print("Converted float64 to float32")
            elif audio_array.dtype != np.float32:
                # Convert any other type to float32
                audio_array = audio_array.astype(np.float32)
                print(f"Converted {audio_array.dtype} to float32")

            print(f"After dtype conversion: {audio_array.dtype}")
            print(f"After dtype conversion: [{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]")

            # Check for silent audio BEFORE normalization
            max_amplitude = np.max(np.abs(audio_array))
            print(f"Max amplitude: {max_amplitude:.6f}")

            if max_amplitude < 1e-6:  # Very small threshold
                return f"Audio appears to be silent (max amplitude: {max_amplitude:.2e}). Please check your microphone permissions and try speaking louder."

            # Only normalize if the audio is too loud, not too quiet
            if max_amplitude > 1.0:
                audio_array = audio_array / max_amplitude
                print(f"Normalized loud audio by factor {max_amplitude}")
            elif max_amplitude < 0.01:  # If very quiet, amplify it
                amplification = min(10.0, 0.1 / max_amplitude)  # Max 10x amplification
                audio_array = audio_array * amplification
                print(f"Amplified quiet audio by factor {amplification}")

            print(
                f"Final audio: shape={audio_array.shape}, dtype={audio_array.dtype}, "
                f"range=[{np.min(audio_array):.6f}, {np.max(audio_array):.6f}], "
                f"sample_rate={sample_rate}"
            )

            # Final check after processing
            if np.max(np.abs(audio_array)) < 1e-4:
                return "Processed audio is still too quiet. Please check microphone settings."

            if sample_rate != 16000:
                print(f"Resampling from {sample_rate} to 16000 Hz")

                try:
                    # Try using librosa for better quality resampling
                    import librosa

                    audio_array = librosa.resample(
                        audio_array, orig_sr=sample_rate, target_sr=16000
                    ).astype(np.float32)  # Ensure float32
                    sample_rate = 16000
                    print("Resampled using librosa")
                except ImportError:
                    print("librosa not available, using simple interpolation")
                    # Simple resampling fallback
                    target_length = int(len(audio_array) * 16000 / sample_rate)
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array), target_length, dtype=np.float32),
                        np.arange(len(audio_array), dtype=np.float32),
                        audio_array,
                    ).astype(np.float32)  # Force float32
                    sample_rate = 16000

                # Double-check the dtype after resampling
                print(f"After resampling: dtype={audio_array.dtype}, shape={audio_array.shape}")

            # Final dtype check before calling transcriber
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
                print("Final conversion to float32 before transcription")

            print(
                f"About to call transcriber with: dtype={audio_array.dtype}, shape={audio_array.shape}"
            )
            # Call transcription
            transcription, _ = transcriber_inference(
                audio=audio_array,
                language="ca",
                target_language="ca",
                task="transcribe",
                turbo=True,
                context=None,
            )

            print(f"Transcription successful: '{transcription}'")

            if transcription and transcription.strip():
                return transcription.strip()
            else:
                return "La transcripció ha fallat"

        except Exception as e:
            error_msg = f"Transcription error: {str(e)}"
            print(f"Exception in transcribe_audio: {e}")
            import traceback

            traceback.print_exc()
            return error_msg

    def convert_audio_to_numpy(self, audio_output):
        """Convert audio tensor to numpy array for Gradio 5.x compatibility"""
        if audio_output is None:
            return None

        try:
            if isinstance(audio_output, tuple) and len(audio_output) == 2:
                sample_rate, audio_data = audio_output

                # Convert tensor to numpy if needed
                if torch.is_tensor(audio_data):
                    audio_data = audio_data.detach().cpu().numpy()

                # Ensure proper format for Gradio 5.x
                if audio_data.dtype != np.float32:
                    if audio_data.dtype == np.int16:
                        audio_data = audio_data.astype(np.float32) / 32767.0
                    else:
                        audio_data = audio_data.astype(np.float32)

                # Ensure audio_data is 1D
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.squeeze()

                return (sample_rate, audio_data)
            else:
                return audio_output

        except Exception as e:
            print(f"Error converting audio: {e}")
            return None

    def generate_tts(self, text, speaker_id=3):
        """Generate TTS audio"""
        try:
            audio_response = tts(
                self.matcha_model,
                self.vocoder,
                text,
                spk_id=speaker_id,
            )
            # Convert to format expected by Gradio Audio component
            audio_output = (22050, audio_response)
            return self.convert_audio_to_numpy(audio_output)
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

    def reset_conversation(self):
        """Reset the conversation"""
        self.messages = [{"role": "system", "content": self.INITIAL_PROMPT}]
        return "🔄 conversa reiniciada!"

    def get_conversation_history(self):
        """Format conversation history as readable text"""
        if not self.messages:
            return "No conversation yet..."

        history_text = ""
        for msg in self.messages[-20:]:  # Show last 20 messages
            if msg["role"] == "user":
                history_text += f"👤 Tu: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                history_text += f"🤖 CLiC-Bot: {msg['content']}\n\n"

        return history_text.strip()

    def process_message(self, user_input, speaker_id=3, max_tokens=256):
        """Process a text message"""
        if not self.is_initialized:
            return "❌ Please load models first!", None, self.get_conversation_history()

        if not user_input.strip():
            return "Please provide some input.", None, self.get_conversation_history()

        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        # Generate response
        response = self.llama_response(self.messages, max_tokens=max_tokens)

        # Add bot response
        self.messages.append({"role": "assistant", "content": response})

        # Generate TTS
        audio_output = self.generate_tts(response, speaker_id)

        return response, audio_output, self.get_conversation_history()

    def process_audio_message(self, audio_data, speaker_id=3):
        """Process an audio message - optimized for Gradio 5.x"""
        if not self.is_initialized:
            return "❌ Please load models first!", None, self.get_conversation_history()

        # More detailed audio validation
        if audio_data is None:
            return (
                "No audio received. Please record some audio first.",
                None,
                self.get_conversation_history(),
            )

        # Handle different audio formats
        audio_info = ""
        if isinstance(audio_data, dict):
            audio_info = f"Dict format: keys={list(audio_data.keys())}"
            if "array" in audio_data and audio_data["array"] is not None:
                audio_info += f", array_shape={audio_data['array'].shape}"
        elif isinstance(audio_data, tuple):
            audio_info = f"Tuple format: length={len(audio_data)}"
            if len(audio_data) == 2 and audio_data[1] is not None:
                audio_info += f", array_shape={audio_data[1].shape}"

        print(f"Audio data received: {audio_info}")

        # Transcribe audio
        transcription = self.transcribe_audio(audio_data)
        print(f"Transcription result: '{transcription}'")

        if not transcription or transcription.startswith("Error") or transcription.strip() == "":
            return (
                f"⚠️ Could not understand the audio. Please try again. ({transcription})",
                None,
                self.get_conversation_history(),
            )

        # Add transcription info to response
        response_prefix = f"🎤 Transcripció: '{transcription}'\n\n"

        # Process the transcribed text
        bot_response, audio_output, history = self.process_message(
            transcription,
            speaker_id,
        )

        # Combine transcription info with bot response
        full_response = response_prefix + bot_response

        return full_response, audio_output, history

    def generate_speech(self, text, speaker_id):
        """Generate speech for given text and speaker_id"""
        try:
            audio = self.generate_tts(text, speaker_id=speaker_id)
            return audio
        except Exception as e:
            print(f"TTS generation error: {e}")
            return None


def get_speaker_id(dialect, person):
    """Get speaker ID from dialect and person selection"""
    if dialect in SPEAKER_ID_DICT and person in SPEAKER_ID_DICT[dialect]:
        return SPEAKER_ID_DICT[dialect][person]
    return 0  # Default fallback


def update_person_choices(dialect):
    """Update person dropdown based on selected dialect"""
    if dialect in SPEAKER_ID_DICT:
        persons = list(SPEAKER_ID_DICT[dialect].keys())
        return gr.update(choices=persons, value=persons[0])
    return gr.update(choices=[], value=None)


def process_text_with_speaker_selection(message, dialect, person):
    speaker_id = get_speaker_id(dialect, person)
    response_text, audio, conv_display = chatbot.process_message(
        message, max_tokens=256, speaker_id=speaker_id
    )
    # Store both the response text and the input text for regeneration
    return conv_display, response_text, audio, {"type": "text", "content": message}


def process_audio_with_speaker_selection(audio_input, dialect, person):
    if not chatbot.is_initialized or not audio_input:
        return "", None, "", None

    speaker_id = get_speaker_id(dialect, person)
    print(f"{audio_input=}")
    response_text, audio, conv_display = chatbot.process_audio_message(
        audio_input, speaker_id=speaker_id
    )
    # Store both the response text and the audio input for regeneration
    return conv_display, response_text, audio, {"type": "audio", "content": audio_input}


# Initialize chatbot
chatbot = ChatBot()


def embed_image_base64(path, alt="", class_=""):
    import base64

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}" class="{class_}"/>'


# Create Gradio interface
def create_interface():
    def regenerate_tts(last_input_state, dialect, person):
        """Regenerate TTS with new voice settings"""
        if not last_input_state:
            return gr.update(), gr.update(), None  # Return proper updates

        speaker_id = get_speaker_id(dialect, person)

        try:
            if last_input_state["type"] == "text":
                # Re-process text input
                response_text, audio, conv_display = chatbot.process_message(
                    last_input_state["content"], max_tokens=256, speaker_id=speaker_id
                )
                return conv_display, response_text, audio
            elif last_input_state["type"] == "audio":
                # Re-process audio input
                print(f"{last_input_state=}")
                response_text, audio, conv_display = chatbot.process_audio_message(
                    last_input_state["content"], speaker_id=speaker_id
                )
                return conv_display, response_text, audio
        except Exception as e:
            print(f"TTS regeneration error: {e}")
            return gr.update(), gr.update(), None

    def update_person_and_regenerate_tts(dialect, last_input_state):
        person_update = update_person_choices(dialect)
        if not last_input_state:
            return person_update, gr.update(), gr.update(), None

        # Get the first person for the new dialect
        first_person = (
            list(SPEAKER_ID_DICT.get(dialect, {}).keys())[0] if dialect in SPEAKER_ID_DICT else None
        )
        if first_person:
            conv_display, response_text, audio = regenerate_tts(
                last_input_state, dialect, first_person
            )
            return person_update, conv_display, response_text, audio
        return person_update, gr.update(), gr.update(), None

    def person_change_regenerate_tts(person, dialect, last_input_state):
        if not last_input_state:
            return gr.update(), gr.update(), None

        conv_display, response_text, audio = regenerate_tts(last_input_state, dialect, person)
        return conv_display, response_text, audio

    def reset_and_clear_last_response():
        reset_result = chatbot.reset_conversation()
        return reset_result, "", None, None  # Clear last_input_state too

    with gr.Blocks(title="CLiC-Bot", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.HTML(
            """
            <h1 class="custom-title">🤖 CLiC Xatbot - Festa de la Ciència</h1>
            <p class="custom-subtitle">Parla amb el nostre xatbot català per text o per veu!</p>
            """,
            elem_id="header-block",
        )
        last_input_state = gr.State(value=None)

        with gr.Row():
            with gr.Column(scale=2):
                # Model loading section
                with gr.Group():
                    gr.Markdown("### 🔧 Gestió de Models")
                    load_btn = gr.Button("Carregar models", variant="primary")
                    status_text = gr.Textbox(
                        label="Estat", value="Models no carregats", interactive=False
                    )
                    reset_btn = gr.Button("🔄 Nova Conversa", variant="secondary")

                # Settings
                with gr.Group():
                    gr.Markdown("### 🎭 Selecció de Veu")
                    with gr.Row():
                        dialect_dropdown = gr.Dropdown(
                            choices=list(SPEAKER_ID_DICT.keys()),
                            value="Central",
                            label="Dialecte",
                            info="Escull el dialecte català",
                        )
                        person_dropdown = gr.Dropdown(
                            choices=list(SPEAKER_ID_DICT["Central"].keys()),
                            value="Grau",
                            label="Persona",
                            info="Escull la veu específica",
                        )
                # Conversation history
                with gr.Group():
                    show_history_btn = gr.Button("📜 Mostrar/Amagar Historial")
                    history_visible = gr.State(False)
                    conversation_display = gr.Textbox(
                        label="Historial de conversa",
                        lines=10,
                        interactive=False,
                        visible=False,
                    )

            with gr.Column(scale=3):
                # Chat interface
                with gr.Group():
                    gr.Markdown("### 💬 Xateja")

                    # Text input
                    with gr.Row():
                        with gr.Column(scale=7):
                            text_input = gr.Textbox(
                                label="Escriu",
                                placeholder="Escriu el teu missatge aquí...",
                                scale=3,
                                max_lines=3,
                            )
                        with gr.Column(scale=1, min_width=120, elem_classes=["button-column"]):
                            text_send_btn = gr.Button("📤 Envia Text", variant="primary")

                    # Audio input
                    with gr.Row():
                        with gr.Column(scale=7):
                            audio_input = gr.Audio(
                                label="Grava la teva veu",
                                sources=["microphone"],
                                type="numpy",
                                show_download_button=False,
                                elem_id="audio_input",
                                interactive=True,
                            )
                        with gr.Column(scale=1, min_width=120, elem_classes=["button-column"]):
                            audio_send_btn = gr.Button("🎵 Envia Àudio", variant="primary")

                    # Response area
                    response_text = gr.Textbox(
                        label="Resposta CLiC-Bot", interactive=False, lines=3
                    )

                    # Audio response
                    audio_output = gr.Audio(label="Resposta CLiC-Bot (veu)", autoplay=True)

        gr.HTML(
            f"""
            <div id="footer">
                <div class="footer-logos">
                    {embed_image_base64("static/clic_logo.png", "CLiC Logo", "footer-logo")}
                    {embed_image_base64("static/ub_logo.png", "UB Logo", "footer-logo")}
                </div>
                <div class="footer-text">
                    <!-- Gran Via de les Corts Catalanes, 585<br> -->
                    <!-- Edifici Josep Carner, 5è pis<br> -->
                    <!-- 08007 Barcelona<br> -->
                    <a href="http://www.ub.edu/filologia" target="_blank" class="footer-link">
                        Facultat de Filologia i Comunicació - Universitat de Barcelona
                    </a>
                </div>
            </div>
            """,
            elem_id="footer-container",
        )

        # Event handlers
        load_btn.click(fn=chatbot.load_models, outputs=[status_text])

        text_send_btn.click(
            fn=process_text_with_speaker_selection,
            inputs=[text_input, dialect_dropdown, person_dropdown],
            outputs=[
                conversation_display,
                response_text,
                audio_output,
                last_input_state,
            ],
        ).then(fn=lambda: "", outputs=[text_input])

        text_input.submit(
            fn=process_text_with_speaker_selection,
            inputs=[text_input, dialect_dropdown, person_dropdown],
            outputs=[
                conversation_display,
                response_text,
                audio_output,
                last_input_state,
            ],
        ).then(fn=lambda: "", outputs=[text_input])

        audio_send_btn.click(
            fn=process_audio_with_speaker_selection,
            inputs=[audio_input, dialect_dropdown, person_dropdown],
            outputs=[
                conversation_display,
                response_text,
                audio_output,
                last_input_state,
            ],
        ).then(fn=lambda: None, outputs=[audio_input])

        # Event handlers for voice changes (auto TTS regeneration)
        dialect_dropdown.change(
            fn=update_person_and_regenerate_tts,
            inputs=[dialect_dropdown, last_input_state],
            outputs=[person_dropdown, conversation_display, response_text, audio_output],
        )
        person_dropdown.change(
            fn=person_change_regenerate_tts,
            inputs=[person_dropdown, dialect_dropdown, last_input_state],
            outputs=[conversation_display, response_text, audio_output],
        )

        reset_btn.click(
            fn=reset_and_clear_last_response,
            outputs=[conversation_display, response_text, audio_output, last_input_state],
        )

        # Fixed toggle function for conversation history
        def toggle_history_visibility(current_visible_state):
            new_state = not current_visible_state
            return gr.update(visible=new_state), new_state

        show_history_btn.click(
            fn=toggle_history_visibility,
            inputs=[history_visible],
            outputs=[conversation_display, history_visible],
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8123,
        share=False,
        debug=True,
        favicon_path="static/bot_base.ico",
    )
