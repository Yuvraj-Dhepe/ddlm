#!/usr/bin/env python3
"""
Fabric-based deployment script for DDLM.

This script deploys the codebase to a remote machine, installs dependencies,
and runs the training/generation/GIF pipeline.
"""

import datetime
import os
import subprocess
import tarfile

import click
from fabric import Connection

UV_BIN = "~/.local/bin/uv"


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
    print(f"Deploying to {address} using key {ssh_key}")
    print(f"Run mode: {run_mode}")
    print(f"User prompt: {user_prompt}")

    # Parse address
    try:
        user, host = address.split("@", 1)
    except ValueError as exc:
        raise click.ClickException("Address must be in the form user@host") from exc

    with Connection(
        host=host,
        user=user,
        port=23223,
        connect_kwargs={"key_filename": ssh_key},
    ) as c:
        print("Creating deployment archive...")
        subprocess.run(
            (
                "find . "
                "\\( -name '*.py' -o -name 'pyproject.toml' \\) "
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
            c.run(f"{UV_BIN} venv")

            print("Installing dependencies...")
            c.run(f"{UV_BIN} sync")

            print("Running training, generation, and GIF creation...")
            cmd = (
                f"{UV_BIN} run python main.py "
                f"--run-mode {run_mode} "
                f"--train --generate --render "
                f'--user-prompt "{user_prompt}"'
            )
            c.run(cmd)

            print("Compressing outputs...")
            c.run("tar czf outputs.tar.gz outputs/")

        print("Downloading outputs...")
        c.get(f"{remote_dir}/outputs.tar.gz", "outputs.tar.gz")

        print("Extracting to local outputs folder...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        with tarfile.open("outputs.tar.gz", "r:gz") as tar:
            tar.extractall(output_dir)

        os.remove("outputs.tar.gz")

        print(f"Outputs extracted to: {output_dir}")

    print("Deployment and execution completed successfully!")


if __name__ == "__main__":
    deploy_and_run()
