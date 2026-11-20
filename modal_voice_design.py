"""VoiceDesign test: design a voice from a description, no preset, no clip.

Qwen3-TTS 1.7B VoiceDesign (Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign).
API, from the official Qwen3-TTS README:
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    wavs, sr = model.generate_voice_design(
        text=..., language="Chinese", instruct=...)

Here the T4 has no native bf16, so this uses fp16 and sdpa, the same
settings as the working CustomVoice runs in modal_app.py. Same size
class as CustomVoice 1.7B, so VRAM should be similar, about 4.2 GB.

Voice designed for this test, in Chinese so it matches the text:
    台湾口音的中年女性声音，语调温和柔软，
    语速缓慢平稳，像台湾人讲国语那样，不带儿化音。
    用缓慢平稳的语气朗读。
In English: a middle-aged female voice with a Taiwanese accent,
gentle and soft intonation, slow and steady pace, speaking Mandarin
the way Taiwanese people do, without erhua. Read aloud in a slow
and steady tone.

Note: generate_voice_design has a single instruct argument covering
both voice description and speaking style, so the description and
the slow-reading instruction are combined into that one string.

The first VoiceDesign test used a soft, warm, gentle young adult
female voice at a moderate pace, and Nicholas said it was too fast
and sounded too young. A second test used an older, slower
middle-aged voice, saved as h01-older-slower.wav, still slightly fast.
This third test asks for a Taiwanese Mandarin accent, saved as
h01-taiwanese.wav so earlier clips are not overwritten.

Caveat, from the official README: VoiceDesign invents a voice from the
description on each call, there is no persistent speaker id. To reuse a
designed voice, the official workflow is Voice Design then Clone: use
the designed clip as the reference for the Base model
(create_voice_clone_prompt, then generate_voice_clone). That second
step is not done in this test.

Run through the GitHub Actions bridge, as modal_app.py runs are:
    modal run modal_voice_design.py
    modal run modal_voice_design.py --variation v01
Variations live in tests/voice_design_variations.json, so the Chinese
instruct never has to travel through bridge shell quoting.
"""

import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "mandarin-tts-voice-design"
VOLUME_NAME = "mandarin-tts-cache"
MODEL_DIR = "/data/models"
OUTPUT_DIR = "/data/outputs/voice-design"

VOICE_DESIGN_INSTRUCT = (
    "台湾口音的中年女性声音，语调温和柔软，"
    "语速缓慢平稳，像台湾人讲国语那样，不带儿化音。"
    "用缓慢平稳的语气朗读。"
)
TEXT = "我每天早上七点起床，然后去跑步。"
OUT_FILE = "h01-taiwanese.wav"

VARIATIONS_FILE = Path(__file__).parent / "tests" / "voice_design_variations.json"


def load_variation(variation_id):
    data = json.loads(VARIATIONS_FILE.read_text(encoding="utf-8"))
    entry = data["variations"][variation_id]
    return data["text"], entry["instruct"], entry["out_file"]

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env({"HF_HOME": "/data/hf", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install("qwen-tts", "soundfile", "torch", "transformers", "accelerate")
)


@app.function(
    image=image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=3600,
    cpu=2,
    memory=8192,
)
def synth_voice_design(text=TEXT, instruct=VOICE_DESIGN_INSTRUCT, out_file=OUT_FILE):
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        device_map="cuda:0",
        dtype=torch.float16,
        attn_implementation="sdpa",
    )
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
            flush=True,
        )

    s0 = time.time()
    out_path = f"{OUTPUT_DIR}/{out_file}"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wavs, sr = model.generate_voice_design(
        text=text,
        language="Chinese",
        instruct=instruct,
    )
    sf.write(out_path, wavs[0], sr)
    info = sf.info(out_path)
    print(
        f"[voice-design] {out_file} done in {time.time() - s0:.1f}s, "
        f"audio {info.duration:.2f}s: {text}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(
            f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
            flush=True,
        )
    vol.commit()
    print(f"ALL DONE voice-design in {time.time() - t0:.0f}s", flush=True)


@app.local_entrypoint()
def main(variation: str = None, instruct: str = None, out_file: str = None):
    if variation:
        text, v_instruct, v_out = load_variation(variation)
        synth_voice_design.remote(text, v_instruct, v_out)
    else:
        synth_voice_design.remote(
            TEXT, instruct or VOICE_DESIGN_INSTRUCT, out_file or OUT_FILE
        )
