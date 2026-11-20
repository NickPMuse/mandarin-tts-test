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

## Revision, same day: older, slower voice

Nicholas said the first designed voice was too fast and sounded too young, so a second VoiceDesign test used a mature, steady middle-aged female voice, warm and full timbre, slow and steady pace, like telling a story calmly, with the reading instruction 用缓慢平稳的语气朗读。 (read aloud in a slow and steady tone). Note: `generate_voice_design` has a single `instruct` argument covering both voice description and speaking style, so the description and the slow-reading instruction were combined into that one string. Code: `modal_voice_design.py`, commit a8428f7.

Clip: `audio/voice-design/h01-older-slower.wav`, same h01 sentence. 24 kHz mono WAV, 3.44 s, peak 0.85, verified non-silent. Bridge run 44827516652, completed in about 1m10s. Synthesis 11.0 s, whole function 19 s, roughly $0.01 at $0.59/hr; exact Modal billing was not checked. The first clip, `audio/voice-design/h01.wav`, was not overwritten.

Plain assessment: the new clip is only slightly slower by duration, 3.44 s against 3.20 s for the first clip, still under 3.5 s for the same 17 characters, about 4.9 characters per second against 5.3. That is a small slowdown, not a clearly slow reading. Whether it sounds older is for Nicholas to judge by listening; the duration alone does not prove it. If it still sounds too young or too fast, the next levers are a stronger age cue in the description (for example, an older adult voice rather than middle-aged), a longer test sentence where pacing differences show up more, or cloning a real older voice with the Base model instead of designing one.

## 2. Base cloning: clone a voice from 3 seconds of reference audio

Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (0.6B also exists). API, from the official README:
`model.generate_voice_clone(text=..., language="Chinese", ref_audio=..., ref_text=...)`
`ref_audio` can be a file path, URL, base64 string, or numpy tuple. `ref_text` is the transcript of the reference clip. For reuse across many sentences, build the prompt once with `create_voice_clone_prompt` and pass it as `voice_clone_prompt`.
Not tested in this run. It differs from CustomVoice presets (fixed 9 speakers, no clip) and from CosyVoice 3 zero-shot cloning (which already clones the CosyVoice repo female reference clip, transcript 希望你以后能够做的比我还好呦, in this repo). Base cloning would let Nicholas clone any voice he has a clean 3 second clip of, including a VoiceDesign clip or his own voice, with no training.

## Taiwanese test, same day: VoiceDesign with a Taiwan accent cue

Nicholas said the older, slower clip was still slightly fast, and asked for a Taiwanese accent. Third VoiceDesign test, same h01 sentence, same model, same T4, fp16. Voice description used, in Chinese, exactly as specified:
台湾口音的中年女性声音，语调温和柔软，语速缓慢平稳，像台湾人讲国语那样，不带儿化音。用缓慢平稳的语气朗读。
In English: a middle-aged female voice with a Taiwanese accent, gentle and soft intonation, slow and steady pace, speaking Mandarin the way Taiwanese people do, without erhua. Read aloud in a slow and steady tone. As before, `generate_voice_design` has a single `instruct` argument covering both voice description and speaking style, so the description and the slow-reading instruction are combined into that one string. Code: `modal_voice_design.py`, output `h01-taiwanese.wav`.

Clip: `audio/voice-design/h01-taiwanese.wav`, same h01 sentence. 24 kHz mono WAV, 10.56 s total, peak 0.68, verified non-silent. Run: bridge run 44828391264, completed in 1m24s. Model load 7 s (weights cached in the Modal Volume). GPU Tesla T4, VRAM allocated 4.18 GB, peak 4.51 GB. Synthesis 25.1 s, whole function 33 s, roughly $0.01 at $0.59/hr, plus small CPU and memory charges; exact Modal billing was not checked. No failures. The earlier clips were not overwritten.

Read the duration with care: the file is 10.56 s, but speech energy runs only from about 1.16 s to about 5.30 s, roughly 4.1 s of speech, with about 1.2 s of near silence at the start and about 5.2 s of silence trailing. The spoken part is slower than both earlier clips (speech about 3.1 s in each), but the 10.56 s file length overstates it. Whether it sounds Taiwanese, and whether the slow pace sounds natural, is for Nicholas to judge by listening. If it still does not sound Taiwanese enough, the next candidates are genuine Taiwanese models, not run in this task: BreezyVoice (MediaTek-Research/BreezyVoice, open source, Taiwanese Mandarin, built on CosyVoice) and BlueMagpie-TTS (OpenFormosa, Taiwanese Mandarin, based on VoxCPM, Apache-2.0).

## Taiwanese reference clone: not done, blocked in this run

The clean reference source was confirmed to exist: JacobLinCool/common_voice_25_zh-TW_taiwan_accent on Hugging Face, CC0 like its Common Voice source, 80,556 validated rows, each with a Taiwan accent probability and an alignment score plus age and gender fields where given. No clip was picked and no clone run was done. A programmatic scan of the validated rows for a female speaker in her thirties to fifties, high Taiwan probability and alignment score, 3 to 7 s long, upvoted and not flagged, timed out reading the datasets-server API after 60 s, and the retry still needs approval, so it was not retried. No reference audio was downloaded, no speaker was identified, and nothing from the dataset was committed. If the scan is approved and a clip is picked, the plan is one short validated clip only, used as `ref_audio` (URL or a local file kept out of the repo) with its sentence as `ref_text` for Qwen3-TTS Base `generate_voice_clone`, or CosyVoice 3 zero-shot, on h01 only, saved under `audio/taiwanese-clone/`, with source, licence, and sentence recorded here.

## 3. Fine-tuning / LoRA: train a persistent custom voice

Not tested, no fine-tune run was done. The official path is the Qwen3-TTS-Finetuning repo, fine-tuning the Base model on a clean single-speaker dataset. That gives a persistent voice that does not need a reference clip at inference, but it needs recorded training audio and GPU training time, unlike VoiceDesign and Base cloning, which are inference only and need no training. Cost and time were not measured here.
