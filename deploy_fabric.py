#!/usr/bin/env python3
"""
Fabric-based deployment script for DDLM.

This script deploys the codebase to a remote machine, installs dependencies,
and runs the training/generation/GIF pipeline.
"""

import subprocess

import click
from fabric import Connection


@click.command()
@click.option("--address", required=True, help="SSH address (e.g., user@host)")
@click.option("--ssh-key", required=True, help="Path to SSH private key")
@click.option("--run-mode", default="quick", help="Run mode: quick or budget_100")
@click.option(
    "--user-prompt", default="Once upon a time", help="User prompt for generation"
)
def deploy_and_run(address: str, ssh_key: str, run_mode: str, user_prompt: str):
    """
    Deploy DDLM to remote machine and run the full pipeline.

    Args:
        address: SSH address
        ssh_key: Path to SSH key
        run_mode: Training mode
        user_prompt: Generation prompt
    """
    remote_dir = "~/ddlm"

    print(f"Deploying to {address} using key {ssh_key}")
    print(f"Run mode: {run_mode}")
    print(f"User prompt: {user_prompt}")

    # Push code using rsync
    print("Pushing code to remote machine...")
    rsync_cmd = [
        "rsync",
        "-avz",
        "-e",
        f"ssh -i {ssh_key}",
        "--exclude=.git",
        "--exclude=__pycache__",
        "--exclude=checkpoints",
        "--exclude=*.pyc",
        ".",
        f"{address}:{remote_dir}",
    ]
    subprocess.run(rsync_cmd, check=True)

    # Connect via Fabric
    with Connection(
        host=address, user=None, connect_kwargs={"key_filename": ssh_key}
    ) as c:
        print("Installing uv...")
        c.run("curl -LsSf https://astral.sh/uv/install.sh | sh")
        c.run('export PATH="$HOME/.cargo/bin:$PATH"')

        print("Navigating to project directory...")
        with c.cd(remote_dir):
            print("Installing dependencies...")
            c.run("uv sync")

            print("Running training, generation, and GIF creation...")
            cmd = f'uv run python main.py --run-mode {run_mode} --train --generate --render --user-prompt "{user_prompt}"'
            c.run(cmd)

        print(
            "Process completed. Check checkpoints/ and inference*.gif files on remote machine."
        )

    print("Deployment and execution completed successfully!")


if __name__ == "__main__":
    deploy_and_run()
