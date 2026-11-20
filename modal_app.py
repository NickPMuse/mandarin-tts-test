"""Mandarin TTS test harness, Modal T4 only.

Runs open-source Mandarin TTS models on Nicholas's own Modal account.
T4 GPU only (USD 0.59/hr, 16 GB VRAM), fp16 / half precision (the T4 has
no native bf16), single stream, no urgency. Model weights are cached in
a Modal Volume so re-runs do not re-download them. Audio is written to
the same Volume and fetched back with `modal volume get`.

No third-party TTS API is used anywhere. Only open-source models running
on Nicholas's own Modal account.

Usage, from the repo root (run through the GitHub Actions bridge, since
the Muse VM blocks direct gRPC to Modal; see run_via_bridge.sh):

    modal run modal_app.py --model cosyvoice3
    modal run modal_app.py --model cosyvoice3 --sentences s01,s08,s10,s11
    modal run modal_app.py --model qwen3-tts
    modal run modal_app.py --model qwen3-tts --sentences h01,h02,h03,h04,h05,h06 --speaker Serena
    modal run modal_app.py --model indextts2

With no --sentences flag, the four-sentence comparison set is used
(s01, s08, s10, s11), so every model reads exactly the same sentences.
"""

import json
import os
import time

import modal

APP_NAME = "mandarin-tts-test"
VOLUME_NAME = "mandarin-tts-cache"
MODEL_DIR = "/data/models"
OUTPUT_DIR = "/data/outputs"

# Same reference voice for every zero-shot model: the Mandarin female
# prompt clip shipped in the Apache-2.0 CosyVoice repo, cloned into the
# image at build time. Transcript: 希望你以后能够做的比我还好呦。
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
PROMPT_WAV = "/opt/CosyVoice/asset/zero_shot_prompt.wav"

# Qwen3-TTS CustomVoice needs no reference clip; it has built-in speakers.
# Vivian is the default Chinese preset voice. Serena is the softer preset:
# warm, gentle young female voice, per the official Qwen3-TTS model card.
# The speaker is a --speaker parameter on the entrypoint, default Vivian,
# so a Serena run writes to a separate output folder and never overwrites
# the Vivian clips.
QWEN_DEFAULT_SPEAKER = "Vivian"
QWEN_INSTRUCT = "用自然平稳的语气朗读。"

vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)

common_env = {
    "HF_HOME": "/data/hf",
    "TOKENIZERS_PARALLELISM": "false",
}

base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env(common_env)
)

cosyvoice_image = (
    base_image
    .run_commands(
        "git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice",
        # openai-whisper is installed separately below: its setup.py imports
        # pkg_resources, which setuptools 81 removed, so it cannot build in
        # an isolated build env. Installed with --no-build-isolation against
        # a pinned setuptools<81. It is genuinely needed: the CosyVoice
        # frontend calls whisper.log_mel_spectrogram on the prompt clip.
        "cd /opt/CosyVoice && grep -v openai-whisper requirements.txt > requirements-tts.txt && pip install -r requirements-tts.txt",
        "pip install 'setuptools<81' wheel numpy",
        "pip install --no-build-isolation openai-whisper==20231117",
        # DeepSpeed ships with the CosyVoice requirements, but on this image
        # it breaks the import chain: transformers imports deepspeed, and
        # deepspeed tries to compile CUDA ops, which fails because the image
        # has no CUDA toolkit (CUDA_HOME does not exist). DeepSpeed is only
        # used for training, not TTS inference, so remove it.
        "pip uninstall -y deepspeed",
        "pip install huggingface_hub soundfile torchaudio",
    )
    .env({"PYTHONPATH": "/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS"})
    .add_local_file("tests/test_sentences.json", "/app/test_sentences.json")
)

qwen_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env(common_env)
    .pip_install("qwen-tts", "soundfile", "torch", "transformers", "accelerate")
    .add_local_file("tests/test_sentences.json", "/app/test_sentences.json")
)

indextts_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1")
    .env(common_env)
    .pip_install("huggingface_hub", "soundfile")
    .run_commands(
        # Reference voice clip for zero-shot cloning, from the CosyVoice repo.
        "git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice",
        # openai-whisper first, on its own: its setup.py imports pkg_resources,
        # removed in setuptools 81, so it cannot build in an isolated env.
        "pip install 'setuptools<81' wheel numpy",
        "pip install --no-build-isolation 'openai-whisper>=20231117'",
        # Official index-tts repo, not the third-party indextts2-inference
        # wrapper: the wrapper targets transformers 5.x, whose beam search
        # passes attention_mask=None into the GPT forward and crashes it.
        # The official repo pins transformers 4.52.1 and torch 2.8.
        "git clone --depth 1 https://github.com/index-tts/index-tts.git /opt/index-tts",
        "cd /opt/index-tts && pip install -e .",
    )
    .add_local_file("tests/test_sentences.json", "/app/test_sentences.json")
)


def _load_sentences(sentence_ids):
    with open("/app/test_sentences.json", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {s["id"]: s for s in data["sentences"]}
    return [(sid, by_id[sid]["text"]) for sid in sentence_ids]


def _report(model, sid, text, seconds, audio_seconds):
    rtf = seconds / audio_seconds if audio_seconds else 0
    print(
        f"[{model}] {sid} done in {seconds:.1f}s, audio {audio_seconds:.1f}s, "
        f"RTF {rtf:.2f}: {text}",
        flush=True,
    )


@app.function(
    image=cosyvoice_image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=3600,
    cpu=2,
    memory=8192,
)
def synth_cosyvoice3(sentence_ids: list[str]):
    import torch
    import torchaudio
    from huggingface_hub import snapshot_download
    from cosyvoice.cli.cosyvoice import AutoModel

    t0 = time.time()
    model_dir = snapshot_download(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        local_dir=f"{MODEL_DIR}/cosyvoice3",
    )
    print(f"model ready in {time.time() - t0:.0f}s: {model_dir}", flush=True)
    print(f"prompt wav exists: {os.path.exists(PROMPT_WAV)}", flush=True)
    cosyvoice = AutoModel(model_dir=model_dir)
    print(f"model loaded in {time.time() - t0:.0f}s total", flush=True)
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
            flush=True,
        )

    for sid, text in _load_sentences(sentence_ids):
        s0 = time.time()
        out_path = f"{OUTPUT_DIR}/cosyvoice3/{sid}.wav"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        made = False
        for i, chunk in enumerate(
            cosyvoice.inference_zero_shot(text, PROMPT_TEXT, PROMPT_WAV, stream=False)
        ):
            torchaudio.save(out_path, chunk["tts_speech"], cosyvoice.sample_rate)
            made = True
        if not made:
            raise RuntimeError(f"CosyVoice3 produced no audio for {sid}")
        import soundfile as sf

        info = sf.info(out_path)
        _report("cosyvoice3", sid, text, time.time() - s0, info.duration)
        if torch.cuda.is_available():
            print(
                f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                flush=True,
            )
    vol.commit()
    print(f"ALL DONE cosyvoice3 in {time.time() - t0:.0f}s", flush=True)


@app.function(
    image=qwen_image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=3600,
    cpu=2,
    memory=8192,
)
def synth_qwen(sentence_ids: list[str], speaker: str = QWEN_DEFAULT_SPEAKER):
    import torch
    import soundfile as sf
    from qwen_tts import Qwen3TTSModel

    # Vivian keeps the original output folder. Any other speaker gets its
    # own folder, so a Serena run cannot overwrite the Vivian clips.
    output_dir = (
        "qwen3-tts"
        if speaker.lower() == QWEN_DEFAULT_SPEAKER.lower()
        else f"qwen3-tts-{speaker.lower()}"
    )
    print(f"Qwen speaker: {speaker}, output dir: {output_dir}", flush=True)

    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
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

    for sid, text in _load_sentences(sentence_ids):
        s0 = time.time()
        out_path = f"{OUTPUT_DIR}/{output_dir}/{sid}.wav"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="Chinese",
            speaker=speaker,
            instruct=QWEN_INSTRUCT,
        )
        sf.write(out_path, wavs[0], sr)
        info = sf.info(out_path)
        _report(output_dir, sid, text, time.time() - s0, info.duration)
        if torch.cuda.is_available():
            print(
                f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                flush=True,
            )
    vol.commit()
    print(f"ALL DONE {output_dir} ({speaker}) in {time.time() - t0:.0f}s", flush=True)


@app.function(
    image=indextts_image,
    gpu="T4",
    volumes={"/data": vol},
    timeout=3600,
    cpu=2,
    memory=8192,
)
def synth_indextts2(sentence_ids: list[str]):
    import torch
    import soundfile as sf
    from huggingface_hub import snapshot_download
    from indextts.infer_v2 import IndexTTS2

    t0 = time.time()
    print(f"prompt wav exists: {os.path.exists(PROMPT_WAV)}", flush=True)
    model_dir = snapshot_download(
        "IndexTeam/IndexTTS-2", local_dir=f"{MODEL_DIR}/indextts2"
    )
    print(f"model ready in {time.time() - t0:.0f}s: {model_dir}", flush=True)
    tts = IndexTTS2(
        cfg_path=f"{model_dir}/config.yaml",
        model_dir=model_dir,
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_qwen_emo=False,
    )
    print(f"model loaded in {time.time() - t0:.0f}s", flush=True)
    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}, "
            f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB",
            flush=True,
        )

    for sid, text in _load_sentences(sentence_ids):
        s0 = time.time()
        out_path = f"{OUTPUT_DIR}/indextts2/{sid}.wav"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        tts.infer(spk_audio_prompt=PROMPT_WAV, text=text, output_path=out_path)
        info = sf.info(out_path)
        _report("indextts2", sid, text, time.time() - s0, info.duration)
        if torch.cuda.is_available():
            print(
                f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                flush=True,
            )
    vol.commit()
    print(f"ALL DONE indextts2 in {time.time() - t0:.0f}s", flush=True)


@app.local_entrypoint()
def main(model: str, sentences: str = "", speaker: str = QWEN_DEFAULT_SPEAKER):
    with open("tests/test_sentences.json", encoding="utf-8") as f:
        data = json.load(f)
    if sentences:
        ids = [s.strip() for s in sentences.split(",") if s.strip()]
    else:
        ids = data["comparison_set"]
    known = {s["id"] for s in data["sentences"]}
    unknown = [i for i in ids if i not in known]
    if unknown:
        raise ValueError(f"unknown sentence ids: {unknown}")
    print(f"model={model} sentences={ids} speaker={speaker}", flush=True)
    if model == "cosyvoice3":
        synth_cosyvoice3.remote(ids)
    elif model == "qwen3-tts":
        synth_qwen.remote(ids, speaker=speaker)
    elif model == "indextts2":
        synth_indextts2.remote(ids)
    elif model == "s1-mini":
        raise NotImplementedError(
            "OpenAudio S1-mini is not wired up yet: its self-hosted path runs "
            "through the Fish Speech repo CLI, not a pip package, and the "
            "fish-audio-sdk on PyPI is the hosted Fish Audio API, which this "
            "test does not use. See README."
        )
    else:
        raise ValueError(f"unknown model: {model} (cosyvoice3, qwen3-tts, indextts2)")
