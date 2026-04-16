#!/usr/bin/env bash

set -euo pipefail

TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-pose_publishers}"

if tmux has-session -t "${TMUX_SESSION_NAME}" 2>/dev/null; then
    echo "tmux session '${TMUX_SESSION_NAME}' already exists"
    echo "attach with: tmux attach -t ${TMUX_SESSION_NAME}"
    exit 0
fi

CONDA_SETUP=""
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SETUP="source ${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_SETUP="source ${HOME}/anaconda3/etc/profile.d/conda.sh"
else
    echo "could not find conda.sh under ~/miniconda3 or ~/anaconda3" >&2
    exit 1
fi

tmux new-session -d -s "${TMUX_SESSION_NAME}" -n sam3
tmux send-keys -t "${TMUX_SESSION_NAME}:sam3" "${CONDA_SETUP}" C-m
tmux send-keys -t "${TMUX_SESSION_NAME}:sam3" "conda activate sam3" C-m
tmux send-keys -t "${TMUX_SESSION_NAME}:sam3" "cd ~/ws/sam3" C-m
tmux send-keys -t "${TMUX_SESSION_NAME}:sam3" "python -m publisher.sam3_publisher" C-m

tmux new-window -t "${TMUX_SESSION_NAME}" -n foundationpose
tmux send-keys -t "${TMUX_SESSION_NAME}:foundationpose" "${CONDA_SETUP}" C-m
tmux send-keys -t "${TMUX_SESSION_NAME}:foundationpose" "conda activate foundationpose" C-m
tmux send-keys -t "${TMUX_SESSION_NAME}:foundationpose" "cd ~/ws/FoundationPose" C-m
tmux send-keys -t "${TMUX_SESSION_NAME}:foundationpose" "python -m publisher.foundationpose_publisher" C-m

echo "started tmux session '${TMUX_SESSION_NAME}'"
echo "attach with: tmux attach -t ${TMUX_SESSION_NAME}"
