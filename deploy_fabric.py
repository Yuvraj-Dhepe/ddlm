#!/usr/bin/env python3
"""
Fabric-based deployment script for DDLM.

This script deploys the codebase to a remote machine, installs dependencies,
and runs the training/generation/GIF pipeline or hyperparameter sweeps.
"""

import datetime
import os
import subprocess
import tarfile
from typing import Optional

import click
from fabric import Connection

UV_BIN = "~/.local/bin/uv"


def setup_wandb(c, wandb_api_key: Optional[str] = None):
    """Setup Weights & Biases login on remote machine."""
    if not wandb_api_key:
        # Check if WANDB_API_KEY env var is set
        c.run(
            'if [ -z "$WANDB_API_KEY" ]; then '
            'echo "WARNING: WANDB_API_KEY not set. W&B logging will not work."; '
            "fi"
        )


def get_remote_output(c, remote_dir: str, output_type: str = "outputs"):
    """Download and extract outputs or sweeps from remote machine."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_type == "sweeps":
        local_dir = f"sweeps/{timestamp}"
        remote_tar = "sweeps.tar.gz"
    else:
        local_dir = f"outputs/{timestamp}"
        remote_tar = "outputs.tar.gz"

    os.makedirs(local_dir, exist_ok=True)

    with c.cd(remote_dir):
        print(f"Compressing {output_type}...")
        c.run(f"tar czf {remote_tar} {output_type}/")

    print(f"Downloading {output_type}...")
    c.get(f"{remote_dir}/{remote_tar}", remote_tar)

    print(f"Extracting to local {local_dir}...")
    with tarfile.open(remote_tar, "r:gz") as tar:
        tar.extractall(local_dir)

    os.remove(remote_tar)
    print(f"{output_type.capitalize()} extracted to: {local_dir}")
    return local_dir


@click.command()
@click.option("--address", required=True, help="SSH address (e.g., user@host)")
@click.option("--ssh-key", required=True, help="Path to SSH private key")
@click.option(
    "--run-mode", default="quick", help="Run mode: quick or budget_100"
)
@click.option(
    "--user-prompt",
    default="Once upon a time",
    help="User prompt for generation",
)
@click.option(
    "--run-sweep",
    is_flag=True,
    help="Run hyperparameter sweep instead of training",
)
@click.option(
    "--sweep-trials", default=3, type=int, help="Number of sweep trials"
)
@click.option(
    "--wandb-api-key", default=None, help="W&B API key for remote login"
)
def deploy_and_run(
    address: str,
    ssh_key: str,
    run_mode: str,
    user_prompt: str,
    run_sweep: bool,
    sweep_trials: int,
    wandb_api_key: Optional[str] = None,
):
    """
    Deploy DDLM to remote machine and run the full pipeline or sweep.

    Args:
        address: SSH address
        ssh_key: Path to SSH key
        run_mode: Training mode
        user_prompt: Generation prompt
        run_sweep: Whether to run hyperparameter sweep
        sweep_trials: Number of trials for sweep
        wandb_api_key: W&B API key (optional, can use WANDB_API_KEY env var)
    """
    print(f"Deploying to {address} using key {ssh_key}")
    print(f"Run mode: {run_mode}")
    print(f"User prompt: {user_prompt}")
    print(f"Run sweep: {run_sweep}")
    if run_sweep:
        print(f"Sweep trials: {sweep_trials}")

    # Parse address
    try:
        user, host = address.split("@", 1)
    except ValueError as exc:
        raise click.ClickException(
            "Address must be in the form user@host"
        ) from exc

    with Connection(
        host=host,
        user=user,
        port=22,
        connect_kwargs={"key_filename": ssh_key},
    ) as c:
        print("Creating deployment archive...")
        subprocess.run(
            (
                "find . "
                "\\( -name '*.py' -o -name 'pyproject.toml' -o -name '*.yaml' \\) "
                "-not -path './.venv/*' "
                "| tar czf ddlm.tar.gz -T -"
            ),
            shell=True,
            check=True,
        )

        print("Pushing code to remote machine...")
        home = c.run("echo $HOME", hide=True).stdout.strip()
        remote_dir = f"{home}/ddlm"

        c.run(f"mkdir -p {remote_dir}")
        c.put("ddlm.tar.gz", f"{remote_dir}/ddlm.tar.gz")

        with c.cd(remote_dir):
            c.run("tar xzf ddlm.tar.gz && rm ddlm.tar.gz")

        os.remove("ddlm.tar.gz")

        print("Installing uv...")
        c.run("curl -LsSf https://astral.sh/uv/install.sh | sh")

        with c.cd(remote_dir):
            print("Ensuring virtual environment exists...")
            c.run(f"{UV_BIN} venv --clear")

            print("Installing dependencies...")
            c.run(f"{UV_BIN} sync")

            # Setup W&B
            setup_wandb(c, wandb_api_key)

            if run_sweep:
                print("Running hyperparameter sweep...")
                # Select sweep config based on run_mode
                if run_mode == "quick":
                    sweep_config = "src/config/sweeps/sweep_quick.yaml"
                else:
                    sweep_config = "src/config/sweeps/sweep.yaml"
                
                print(f"Using sweep config: {sweep_config}")
                
                # Set WANDB_API_KEY environment variable if provided
                if wandb_api_key:
                    cmd = (
                        f'export WANDB_API_KEY="{wandb_api_key}" && '
                        f"{UV_BIN} run python scripts/sweep.py "
                        f"--sweep-config {sweep_config} "
                        f"--trials {sweep_trials} "
                        f"--run-mode {run_mode} "
                        f'--user-prompt "{user_prompt}"'
                    )
                else:
                    cmd = (
                        f"{UV_BIN} run python scripts/sweep.py "
                        f"--sweep-config {sweep_config} "
                        f"--trials {sweep_trials} "
                        f"--run-mode {run_mode} "
                        f'--user-prompt "{user_prompt}"'
                    )
                c.run(cmd)

                # Download sweeps
                get_remote_output(c, remote_dir, "sweeps")
            else:
                print("Running training, generation, and GIF creation...")
                cmd = (
                    f"{UV_BIN} run python main.py "
                    f"--run-mode {run_mode} "
                    f"--train --generate --render "
                    f'--user-prompt "{user_prompt}"'
                )
                c.run(cmd)

                # Download outputs
                get_remote_output(c, remote_dir, "outputs")

    print("Deployment and execution completed successfully!")


if __name__ == "__main__":
    deploy_and_run()
