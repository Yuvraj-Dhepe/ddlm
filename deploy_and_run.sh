#!/bin/bash

# Script to deploy DDLM codebase to remote machine, install dependencies, and run training/generation/GIF creation
# Usage: ./deploy_and_run.sh <ssh_address> <ssh_key_path> [run_mode] [user_prompt]

set -e  # Exit on any error

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <ssh_address> <ssh_key_path> [run_mode] [user_prompt]"
    echo "  ssh_address: e.g., user@remote.host"
    echo "  ssh_key_path: path to SSH private key"
    echo "  run_mode: quick or budget_100 (default: quick)"
    echo "  user_prompt: prompt for generation (default: 'Once upon a time')"
    exit 1
fi

SSH_ADDRESS=$1
SSH_KEY=$2
RUN_MODE=${3:-quick}
USER_PROMPT=${4:-"Once upon a time"}

REMOTE_DIR="~/ddlm"

echo "Deploying to $SSH_ADDRESS using key $SSH_KEY"
echo "Run mode: $RUN_MODE"
echo "User prompt: $USER_PROMPT"

# Push code to remote machine
echo "Pushing code to remote machine..."
rsync -avz -e "ssh -i $SSH_KEY" --exclude='.git' --exclude='__pycache__' --exclude='checkpoints' --exclude='*.pyc' . $SSH_ADDRESS:$REMOTE_DIR

# SSH to remote and run commands
ssh -i $SSH_KEY $SSH_ADDRESS << EOF
    set -e

    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="\$HOME/.cargo/bin:\$PATH"

    echo "Navigating to project directory..."
    cd $REMOTE_DIR

    echo "Installing dependencies..."
    uv sync

    echo "Running training, generation, and GIF creation..."
    uv run python main.py --run-mode $RUN_MODE --train --generate --render --user-prompt "$USER_PROMPT"

    echo "Process completed. Check checkpoints/ and inference*.gif files."
EOF

echo "Deployment and execution completed successfully!"