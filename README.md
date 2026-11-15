# Mandarin TTS test

Comparing open-source Mandarin text-to-speech models for narrating study notes, judged by listening, not by metrics.

The pain point being tested: Microsoft TTS sometimes sounds inauthentic, gets tones wrong, and misreads polyphonic characters (多音字). The deciding question is which model reads hard Mandarin correctly and sounds the most natural, and whether a misreading can be fixed with pinyin.

## Shortlist (from the research report)

Research: `research_notes/mandarin-tts-models-2026-20261115-2057/report.md` (in the agent workspace, not in this repo).

| Rank | Model | Hugging Face repo | Why | Pinyin fix? | Licence |
|------|-------|-------------------|-----|-------------|---------|
| 1 | CosyVoice 3 | FunAudioLLM/Fun-CosyVoice3-0.5B-2512 | Best balance of naturalness, fixable polyphones, numbers handling | Yes, pronunciation inpainting | Apache-2.0 |
| 2 | IndexTTS-2 | IndexTeam/IndexTTS-2 | Best first-impression Mandarin naturalness, best polyphone override | Yes, inline pinyin plus custom polyphone dictionary | Custom bilibili licence, free for small users |
| 3 | Qwen3-TTS 1.7B CustomVoice | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | Most stable quality, built-in preset voices, no reference clip needed | No | Apache-2.0 |
| 4 | OpenAudio S1-mini | fishaudio/openaudio-s1-mini | Cheap baseline, about 5 GB VRAM | No | CC-BY-NC-SA-4.0 |

Not tested here: Fish S2-Pro (needs 24 GB VRAM, does not fit a T4), ChatTTS (dialogue, not narration), Chatterbox V3 (worst polyphone score, retired).

## How the test runs

- GPU: Modal T4 only, USD 0.59/hr, 16 GB VRAM, fp16 (the T4 has no native bf16). Slowest and cheapest Modal GPU on purpose. Single stream, no urgency.
- No third-party TTS API is used. Only open-source models running on Nicholas's own Modal account.
- Model weights are cached in a Modal Volume (`mandarin-tts-cache`), so re-runs do not re-download them.
- The Muse VM blocks direct gRPC to Modal, so jobs run through the GitHub Actions bridge repo `NickPMuse/muse-modal-bridge`, which holds the Modal token as an Actions secret. See `run_via_bridge.sh`.
- Audio is fetched back from the Modal Volume with `modal volume get` and saved under `audio/<model>/<id>.wav`.

## Listening

The same four sentences are read by every model, so clips compare directly:

- s01, polyphone 长: 这条路很长，走路要花很长时间，他长大以后就明白了。
- s08, tone sandhi 不, 一, third tone: 我不想去，你也不想去，我们都想好好休息一下。
- s10, erhua: 下午我们去小院儿里坐坐，那儿很好玩儿，还有花儿和鸟儿。
- s11, date, time, decimal: 会议定在2026年11月15日下午3点半，费用是12.5元，请提前十分钟到。

The full set is 14 sentences in `tests/test_sentences.json` (also in `tests/test_sentences.md`), covering polyphones (行, 长, 重, 乐, 还, 得, 着, 差), a surname (单, 曾), 不/一 sandhi, a third-tone run, erhua, a date, a decimal, an identifier string, and a proper name. All sentences are generic and written for this test; none of Nicholas's private notes are in this repo.

Voices: CosyVoice 3 and IndexTTS-2 are zero-shot cloners, so both use the same reference clip, the Mandarin female prompt clip shipped in the Apache-2.0 CosyVoice repo (`asset/zero_shot_prompt.wav`, transcript 希望你以后能够做的比我还好呦). Qwen3-TTS CustomVoice uses its built-in preset speaker Vivian, since it needs no reference clip.

Judge by ear: correct readings first, then naturalness. No WER or CER score is used, because those scores do not catch polyphone errors.

## Run it

From the repo root, through the bridge (see `run_via_bridge.sh` for the exact commands):

```
modal run modal_app.py --model cosyvoice3
modal run modal_app.py --model cosyvoice3 --sentences s01,s08,s10,s11
modal run modal_app.py --model qwen3-tts
modal run modal_app.py --model indextts2
```

With no `--sentences` flag, the four-sentence comparison set is used.

Fetch the audio back:

```
modal volume get mandarin-tts-cache /outputs audio
```

## Status

See `results/` for per-run notes (timings, cost, failures). The deliverable is the audio in `audio/` plus a short note per model, not a long report.

## OpenAudio S1-mini

Not wired up yet. Its self-hosted path runs through the Fish Speech repo CLI, not a pip package, and the `fish-audio-sdk` package on PyPI is the hosted Fish Audio API, which this test does not use.
