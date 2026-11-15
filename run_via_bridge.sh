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
#   ./run_via_bridge.sh indextts2
#
# The bridge workflow uploads whatever lands in bridge-output/ as an
# artifact named modal-output. Download it with:
#   gh run download <run-id> --repo NickPMuse/muse-modal-bridge -n modal-output -D ./bridge-output
# Then copy the wav files into audio/<model>/.

set -euo pipefail

MODEL="${1:?usage: run_via_bridge.sh <model> [s01,s08,...]}"
SENTENCES="${2:-}"

if [ -n "$SENTENCES" ]; then
  CMD="git clone --depth 1 https://github.com/NickPMuse/mandarin-tts-test.git tts && cd tts && pip install -q modal && modal run modal_app.py --model $MODEL --sentences $SENTENCES && modal volume get mandarin-tts-cache /outputs \$GITHUB_WORKSPACE/bridge-output"
else
  CMD="git clone --depth 1 https://github.com/NickPMuse/mandarin-tts-test.git tts && cd tts && pip install -q modal && modal run modal_app.py --model $MODEL && modal volume get mandarin-tts-cache /outputs \$GITHUB_WORKSPACE/bridge-output"
fi

gh workflow run modal.yml --repo NickPMuse/muse-modal-bridge -f command="$CMD"
echo "Dispatched. Watch with: gh run list --repo NickPMuse/muse-modal-bridge --limit 3"
