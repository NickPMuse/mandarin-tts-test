#!/bin/bash
# Run a Modal job through the GitHub Actions bridge.
# The Muse VM blocks direct gRPC to Modal, so the job runs in GitHub
# Actions, where the Modal token lives as an Actions secret in
# NickPMuse/muse-modal-bridge. No token or secret appears in this file.
#
# Usage:
#   ./run_via_bridge.sh cosyvoice3            # comparison set, 4 sentences
#   ./run_via_bridge.sh cosyvoice3 s01,s08    # chosen sentences only
#   ./run_via_bridge.sh qwen3-tts
#   ./run_via_bridge.sh qwen3-tts h01,h02,h03,h04,h05,h06 Serena
#   ./run_via_bridge.sh indextts2
#
# The third argument is the Qwen3-TTS speaker only (default Vivian).
# A non-Vivian speaker writes to outputs/qwen3-tts-<speaker>, so it never
# overwrites the Vivian clips in outputs/qwen3-tts.
#
# Fetching audio back: `modal volume get` with a directory remote path
# fails with Errno 21 unless the destination directory already exists,
# and it downloads the remote directory as a subfolder of the destination.
# So: mkdir the destination first, then get outputs/<model> into it, with
# a per-file fallback loop. Files land in bridge-output/<model>/*.wav.
#
# The bridge workflow uploads whatever lands in bridge-output/ as an
# artifact named modal-output. Download it with:
#   gh run download <run-id> --repo NickPMuse/muse-modal-bridge -n modal-output -D ./bridge-output
# Then copy the wav files into audio/<model>/.

set -euo pipefail

MODEL="${1:?usage: run_via_bridge.sh <model> [s01,s08,...] [speaker]}"
SENTENCES="${2:-}"
SPEAKER="${3:-}"
if [ -n "$SENTENCES" ]; then
  SENTENCE_ARGS="--sentences $SENTENCES"
  SENTENCE_LIST="$(echo "$SENTENCES" | tr ',' ' ')"
else
  SENTENCE_ARGS=""
  SENTENCE_LIST="s01 s08 s10 s11"
fi
if [ -n "$SPEAKER" ]; then
  SPEAKER_ARGS="--speaker $SPEAKER"
else
  SPEAKER_ARGS=""
fi
# Output folder in the Modal Volume, mirroring modal_app.py.
OUTPUT_DIR="$MODEL"
if [ "$MODEL" = "qwen3-tts" ] && [ -n "$SPEAKER" ]; then
  LOWER="$(echo "$SPEAKER" | tr '[:upper:]' '[:lower:]')"
  if [ "$LOWER" != "vivian" ]; then
    OUTPUT_DIR="qwen3-tts-$LOWER"
  fi
fi

CMD="git clone --depth 1 https://github.com/NickPMuse/mandarin-tts-test.git tts && cd tts && pip install -q modal && modal run modal_app.py --model $MODEL $SENTENCE_ARGS $SPEAKER_ARGS && mkdir -p \$GITHUB_WORKSPACE/bridge-output && modal volume get mandarin-tts-cache outputs/$OUTPUT_DIR \$GITHUB_WORKSPACE/bridge-output || for f in $SENTENCE_LIST; do modal volume get mandarin-tts-cache outputs/$OUTPUT_DIR/\$f.wav \$GITHUB_WORKSPACE/bridge-output/$OUTPUT_DIR/\$f.wav; done"

gh workflow run modal.yml --repo NickPMuse/muse-modal-bridge -f command="$CMD"
echo "Dispatched. Watch with: gh run list --repo NickPMuse/muse-modal-bridge --limit 3"
