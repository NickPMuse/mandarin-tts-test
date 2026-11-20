#!/bin/bash
# Run VoiceDesign variations v01 to v10, one bridge run each, sequentially.
# Each run clones the repo (with the variations JSON), runs
# modal_voice_design.py --variation vXX, and fetches that one wav back
# through the bridge-output artifact. Verified clips land in
# audio/voice-design/taiwanese-vXX/h01.wav. Failures are logged and the
# loop continues with the next variation.
# Usage: ./run_voice_design_10.sh [v01 v02 ...]  (default: all ten)
set -uo pipefail
cd "$(dirname "$0")"

VARIATIONS="${@:-v01 v02 v03 v04 v05 v06 v07 v08 v09 v10}"
LOG="voice_design_10_log.txt"
: > "$LOG"

verify() {
  python3 - "$1" << 'PY'
import sys, wave, struct
p = sys.argv[1]
w = wave.open(p)
n = w.getnframes(); fr = w.getframerate()
frames = w.readframes(n); w.close()
samples = struct.unpack('<' + 'h' * n, frames)
peak = max(abs(s) for s in samples) / 32768.0
thr = 200
first = next((i for i, s in enumerate(samples) if abs(s) > thr), 0)
last = next((i for i in range(n - 1, -1, -1) if abs(samples[i]) > thr), n - 1)
print(f"duration={n/fr:.2f}s peak={peak:.2f} speech={first/fr:.2f}s-{last/fr:.2f}s")
sys.exit(0 if peak > 0.05 else 1)
PY
}

for V in $VARIATIONS; do
  echo "=== $V $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
  CMD="git clone --depth 1 https://github.com/NickPMuse/mandarin-tts-test.git tts && cd tts && pip install -q modal && modal run modal_voice_design.py --variation $V && mkdir -p \$GITHUB_WORKSPACE/bridge-output/voice-design/taiwanese-$V && modal volume get mandarin-tts-cache outputs/voice-design/taiwanese-$V/h01.wav \$GITHUB_WORKSPACE/bridge-output/voice-design/taiwanese-$V/h01.wav"
  gh workflow run modal.yml --repo NickPMuse/muse-modal-bridge -f command="$CMD"
  sleep 15
  RUN_ID=$(gh run list --repo NickPMuse/muse-modal-bridge --limit 1 --json databaseId --jq '.[0].databaseId')
  echo "$V bridge run $RUN_ID" | tee -a "$LOG"
  if ! gh run watch "$RUN_ID" --repo NickPMuse/muse-modal-bridge --exit-status >> "$LOG" 2>&1; then
    echo "$V FAILED bridge run $RUN_ID" | tee -a "$LOG"
    continue
  fi
  rm -rf "bridge-output-$V"
  if ! gh run download "$RUN_ID" --repo NickPMuse/muse-modal-bridge -n modal-output -D "bridge-output-$V" >> "$LOG" 2>&1; then
    echo "$V FAILED download run $RUN_ID" | tee -a "$LOG"
    continue
  fi
  SRC="bridge-output-$V/voice-design/taiwanese-$V/h01.wav"
  DEST="audio/voice-design/taiwanese-$V/h01.wav"
  if [ ! -f "$SRC" ]; then
    echo "$V FAILED missing $SRC" | tee -a "$LOG"
    continue
  fi
  mkdir -p "audio/voice-design/taiwanese-$V"
  cp "$SRC" "$DEST"
  if INFO=$(verify "$DEST"); then
    echo "$V OK run $RUN_ID $DEST $INFO" | tee -a "$LOG"
  else
    echo "$V FAILED verify run $RUN_ID $DEST $INFO" | tee -a "$LOG"
  fi
done
echo "=== done ===" | tee -a "$LOG"
grep -E '^(v[0-9]+ (OK|FAILED))' "$LOG" || true
