"""
Rendering module for creating terminal-style GIFs.

This module contains functions to render frames and create GIFs from diffusion generation.
"""

import os
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_mono_font(size: int):
    """
    Get a monospace font.

    Args:
        size (int): Font size.

    Returns:
        ImageFont: Font object.
    """
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_terminal_frame(
    lines: List[str],
    width: int = 1200,
    height: int = 700,
    font_size: int = 20,
    margin: int = 20,
    line_spacing: int = 6,
) -> Image.Image:
    """
    Render a terminal frame.

    Args:
        lines: List of lines to render.
        width (int): Image width.
        height (int): Image height.
        font_size (int): Font size.
        margin (int): Margin.
        line_spacing (int): Line spacing.

    Returns:
        PIL Image: Rendered image.
    """
    bg = (10, 10, 10)
    fg = (230, 230, 230)

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    font = get_mono_font(font_size)

    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=fg)
        y += font_size + line_spacing
        if y > height - margin:
            break
    return img


def wrap_text_to_width(text: str, max_chars: int = 90) -> List[str]:
    """
    Wrap text to width.

    Args:
        text (str): Text to wrap.
        max_chars (int): Max characters per line.

    Returns:
        list: Wrapped lines.
    """
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
    """
    Make chat lines for rendering.

    Args:
        user_msg (str): User message.
        assistant_text (str): Assistant text.

    Returns:
        list: Lines for rendering.
    """
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


def create_gif(
    frames: List[Tuple[int, str]],
    diffusion_steps: int,
    user_prompt: str,
    gif_path: str = "outputs/inference.gif",
) -> None:
    """
    Create a GIF from frames.

    Args:
        frames: List of (step, decoded) tuples.
        diffusion_steps (int): Total diffusion steps.
        user_prompt (str): User prompt.
        gif_path (str): Path to save GIF.
    """
    gif_frames = []
    for s, decoded in frames:
        lines = make_chat_lines(user_prompt, decoded)
        lines.insert(2, f"(diffusion step {s:03d}/{diffusion_steps:03d})")
        img = render_terminal_frame(lines)
        gif_frames.append(np.array(img))

    imageio.mimsave(gif_path, gif_frames, duration=0.08)
    print("Saved:", gif_path)


def render_terminal_frame_neon(
    lines: List[str],
    width: int = 1200,
    height: int = 700,
    font_size: int = 20,
    margin: int = 20,
    line_spacing: int = 6,
) -> Image.Image:
    """
    Render a neon-style terminal frame.

    Args:
        lines: List of lines.
        width (int): Image width.
        height (int): Image height.
        font_size (int): Font size.
        margin (int): Margin.
        line_spacing (int): Line spacing.

    Returns:
        PIL Image: Rendered image.
    """
    # === NEON COLORS ===
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    neon_green = (57, 255, 20)
    orange = (255, 165, 0)
    dim = (80, 80, 100)
    text_color = (200, 200, 220)

    # Gradient background
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for py in range(height):
        t = py / height
        r = int(12 * (1 - t) + 4 * t)
        g = int(12 * (1 - t) + 4 * t)
        b = int(28 * (1 - t) + 12 * t)
        for px in range(width):
            pixels[px, py] = (r, g, b)

    draw = ImageDraw.Draw(img)
    font = get_mono_font(font_size)
    font_small = get_mono_font(font_size - 4)

    # === CORNER BRACKETS ===
    cs = 25
    draw.line([(6, 6 + cs), (6, 6), (6 + cs, 6)], fill=cyan, width=2)
    draw.line(
        [(width - 6 - cs, 6), (width - 6, 6), (width - 6, 6 + cs)], fill=cyan, width=2
    )
    draw.line(
        [(6, height - 6 - cs), (6, height - 6), (6 + cs, height - 6)],
        fill=magenta,
        width=2,
    )
    draw.line(
        [
            (width - 6 - cs, height - 6),
            (width - 6, height - 6),
            (width - 6, height - 6 - cs),
        ],
        fill=magenta,
        width=2,
    )

    # === PARSE LINES FOR STEP INFO ===
    step_current, step_total = None, None
    for line in lines:
        if "diffusion step" in line.lower():
            import re

            match = re.search(r"(\d+)/(\d+)", line)
            if match:
                step_current, step_total = int(match.group(1)), int(match.group(2))
                break

    y = margin

    for i, line in enumerate(lines):
        if "====" in line:
            # Header panel
            header_text = (
                line.replace("=", "").strip() or "◈ DIFFUSION LANGUAGE MODEL ◈"
            )
            draw.rounded_rectangle(
                [margin, y - 2, width - margin, y + font_size + 8],
                radius=5,
                fill=(15, 15, 30),
                outline=cyan,
                width=2,
            )
            draw.line([(margin + 10, y), (width - margin - 10, y)], fill=cyan, width=1)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    draw.text(
                        (margin + 15 + dx, y + 3 + dy),
                        header_text,
                        font=font,
                        fill=(0, 80, 80),
                    )
            draw.text((margin + 15, y + 3), header_text, font=font, fill=cyan)
            y += font_size + line_spacing + 8

        elif "diffusion step" in line.lower() and step_current is not None:
            # Progress bar
            progress = step_current / step_total
            bar_x, bar_y = margin, y
            bar_w, bar_h = width - margin * 2 - 200, 20

            draw.rounded_rectangle(
                [bar_x, bar_y, bar_x + bar_w, bar_y + bar_h],
                radius=bar_h // 2,
                fill=(20, 20, 35),
                outline=dim,
                width=1,
            )

            seg_w, gap = 8, 3
            num_segs = (bar_w - 6) // (seg_w + gap)
            filled = int(num_segs * progress)

            for si in range(num_segs):
                sx = bar_x + 3 + si * (seg_w + gap)
                if si < filled:
                    t = si / max(num_segs - 1, 1)
                    seg_color = (
                        int(cyan[0] * (1 - t) + magenta[0] * t),
                        int(cyan[1] * (1 - t) + magenta[1] * t),
                        int(cyan[2] * (1 - t) + magenta[2] * t),
                    )
                else:
                    seg_color = (25, 25, 40)
                draw.rounded_rectangle(
                    [sx, bar_y + 3, sx + seg_w, bar_y + bar_h - 3],
                    radius=2,
                    fill=seg_color,
                )

            step_text = f"STEP {step_current:03d}/{step_total:03d}"
            tx = width - margin - 180
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    draw.text(
                        (tx + dx, bar_y + dy),
                        step_text,
                        font=font_small,
                        fill=(80, 50, 0),
                    )
            draw.text((tx, bar_y), step_text, font=font_small, fill=orange)
            y += bar_h + line_spacing + 5

        elif line.startswith("[You]:"):
            draw.text((margin, y), "▶ USER", font=font_small, fill=dim)
            y += font_size + line_spacing

        elif line.startswith("[Assistant]:"):
            draw.text((margin, y), "◀ ASSISTANT", font=font_small, fill=dim)
            y += font_size + line_spacing

        elif "<Starting a new chat" in line:
            draw.text((margin, y), line, font=font_small, fill=dim)
            y += font_size + line_spacing

        elif line.strip() == "":
            y += line_spacing

        else:
            in_user = False
            in_assistant = False
            for prev_line in lines[:i]:
                if "[You]:" in prev_line:
                    in_user, in_assistant = True, False
                elif "[Assistant]:" in prev_line:
                    in_user, in_assistant = False, True

            if in_user:
                color, glow = cyan, (0, 60, 60)
            elif in_assistant:
                color, glow = neon_green, (15, 60, 5)
            else:
                color, glow = text_color, None

            if glow:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            draw.text((margin + dx, y + dy), line, font=font, fill=glow)
            draw.text((margin, y), line, font=font, fill=color)
            y += font_size + line_spacing

        if y > height - margin:
            break

    # Scanlines
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    scanline_draw = ImageDraw.Draw(overlay)
    for sy in range(0, height, 2):
        scanline_draw.line([(0, sy), (width, sy)], fill=(0, 0, 0, 12))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return img


def create_cool_gif(
    frames: List[Tuple[int, str]],
    diffusion_steps: int,
    user_prompt: str,
    gif_path: str = "outputs/inference_cool.gif",
) -> None:
    """
    Create a cool neon GIF.

    Args:
        frames: List of (step, decoded) tuples.
        diffusion_steps (int): Total steps.
        user_prompt (str): User prompt.
        gif_path (str): Path to save GIF.
    """
    gif_frames = []
    for s, decoded in frames:
        lines = make_chat_lines(user_prompt, decoded)
        lines.insert(2, f"(diffusion step {s:03d}/{diffusion_steps:03d})")
        img = render_terminal_frame_neon(lines)
        gif_frames.append(np.array(img))
    imageio.mimsave(gif_path, gif_frames, duration=0.08)
    print("Saved:", gif_path)
