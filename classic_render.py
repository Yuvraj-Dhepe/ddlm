"""
Classic animation-style terminal GIF renderer.

This module renders terminal-style diffusion outputs with a
hand-animated, old-film / storybook aesthetic.
"""

import os
import random
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ============================================================
# FONT
# ============================================================


def get_mono_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


# ============================================================
# TEXT UTILITIES
# ============================================================


def wrap_text_to_width(text: str, max_chars: int = 90) -> List[str]:
    out = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.rstrip()
        if not paragraph:
            out.append("")
            continue
        while len(paragraph) > max_chars:
            out.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
        out.append(paragraph)
    return out


def make_chat_lines(user_msg: str, assistant_text: str) -> List[str]:
    header = "============================== multi-turn chat mode =============================="
    sub = "<Starting a new chat. Type your message.>"
    lines = [header, sub, ""]
    lines += ["[You]:", user_msg, ""]
    lines += ["[Assistant]:"]
    if "<|assistant|>" in assistant_text:
        assistant_text = assistant_text.split("<|assistant|>", 1)[1]
    assistant_text = assistant_text.replace("<|end|>", "").strip()
    lines += wrap_text_to_width(assistant_text, max_chars=90)
    return lines


# ============================================================
# CLASSIC HAND-ANIMATED RENDERER
# ============================================================


def render_terminal_frame_classic(
    lines: List[str],
    width: int = 1200,
    height: int = 700,
    font_size: int = 20,
    margin: int = 30,
    line_spacing: int = 8,
    jitter: int = 1,
) -> Image.Image:
    """
    Render a classical hand-animated / old-film style terminal frame.
    """

    # Colors
    paper = (245, 238, 220)
    ink = (20, 15, 10)
    shadow = (140, 120, 90)

    img = Image.new("RGB", (width, height), paper)
    draw = ImageDraw.Draw(img)
    font = get_mono_font(font_size)

    # Paper grain
    grain = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    gp = grain.load()
    for y in range(height):
        for x in range(width):
            v = random.randint(0, 18)
            gp[x, y] = (0, 0, 0, v)

    img = Image.alpha_composite(img.convert("RGBA"), grain).convert("RGB")

    draw = ImageDraw.Draw(img)
    y = margin

    for line in lines:
        jx = random.randint(-jitter, jitter)
        jy = random.randint(-jitter, jitter)

        # Ink shadow / bleed
        draw.text(
            (margin + jx + 1, y + jy + 1),
            line,
            font=font,
            fill=shadow,
        )

        draw.text(
            (margin + jx, y + jy),
            line,
            font=font,
            fill=ink,
        )

        y += font_size + line_spacing
        if y > height - margin:
            break

    # Vignette
    vignette = Image.new("L", img.size, 255)
    vdraw = ImageDraw.Draw(vignette)
    max_r = max(width, height)
    for r in range(0, max_r, 30):
        alpha = int(180 * (r / max_r))
        vdraw.ellipse(
            [-r, -r, width + r, height + r],
            fill=255 - alpha,
        )

    img = Image.composite(
        img,
        Image.new("RGB", img.size, (200, 190, 170)),
        vignette,
    )

    # Slight blur for drawn feel
    img = img.filter(ImageFilter.GaussianBlur(radius=0.25))

    return img


# ============================================================
# GIF CREATOR
# ============================================================


def create_classic_gif(
    frames: List[Tuple[int, str]],
    diffusion_steps: int,
    user_prompt: str,
    gif_path: str = "outputs/inference_classic.gif",
) -> None:
    """
    Create a hand-animated style GIF from diffusion frames.
    """
    gif_frames = []

    for s, decoded in frames:
        lines = make_chat_lines(user_prompt, decoded)
        lines.insert(2, f"(diffusion step {s:03d}/{diffusion_steps:03d})")
        img = render_terminal_frame_classic(lines)
        gif_frames.append(np.array(img))

    imageio.mimsave(gif_path, gif_frames, duration=0.09)
    print("Saved:", gif_path)
