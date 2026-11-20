# Generating our own Mandarin voices, 20 Nov 2026

Three own-voice paths in Qwen3-TTS 1.7B, all Apache-2.0, all inference on a Modal T4 in fp16 unless noted. No third-party TTS API.

## 1. VoiceDesign: design a voice from a description, no audio needed

Model: Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign. API, from the official README:
`model.generate_voice_design(text=..., language="Chinese", instruct=...)`
where `instruct` is a natural-language voice description. Same 1.7B size class as CustomVoice.

Tested here, it works on a T4. Voice description used, in Chinese:
温柔、温暖的年轻成年女性声音，音色柔和清晰，语速适中，像在温柔地讲故事。
In English: a soft, warm, gentle young adult female Mandarin voice, smooth and clear, moderate pace, gentle storytelling delivery.

Clip: `audio/voice-design/h01.wav`, h01: 我每天早上七点起床，然后去跑步。
24 kHz mono WAV, 3.20 s, peak 0.36, verified non-silent.
Run: bridge run 44826933410, completed in 1m39s. Model load 29 s (first download, weights cached in the Modal Volume after this). GPU Tesla T4, VRAM allocated 4.18 GB, peak 4.44 GB. Synthesis 10.2 s, whole function 46 s, roughly $0.01 at $0.59/hr, plus small CPU and memory charges; exact Modal billing was not checked. No failures. Code: `modal_voice_design.py`, a separate file so it does not touch `modal_app.py`.

Caveat: VoiceDesign invents a voice from the description on each call, there is no persistent speaker id, so the same description may not give the identical voice next time. The official reuse path is Voice Design then Clone: use the designed clip as the reference for the Base model (see below). That second step was not done in this test.

## 2. Base cloning: clone a voice from 3 seconds of reference audio

Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (0.6B also exists). API, from the official README:
`model.generate_voice_clone(text=..., language="Chinese", ref_audio=..., ref_text=...)`
`ref_audio` can be a file path, URL, base64 string, or numpy tuple. `ref_text` is the transcript of the reference clip. For reuse across many sentences, build the prompt once with `create_voice_clone_prompt` and pass it as `voice_clone_prompt`.
Not tested in this run. It differs from CustomVoice presets (fixed 9 speakers, no clip) and from CosyVoice 3 zero-shot cloning (which already clones the CosyVoice repo female reference clip, transcript 希望你以后能够做的比我还好呦, in this repo). Base cloning would let Nicholas clone any voice he has a clean 3 second clip of, including a VoiceDesign clip or his own voice, with no training.

## 3. Fine-tuning / LoRA: train a persistent custom voice

Not tested, no fine-tune run was done. The official path is the Qwen3-TTS-Finetuning repo, fine-tuning the Base model on a clean single-speaker dataset. That gives a persistent voice that does not need a reference clip at inference, but it needs recorded training audio and GPU training time, unlike VoiceDesign and Base cloning, which are inference only and need no training. Cost and time were not measured here.
