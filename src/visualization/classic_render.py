"""
Modern minimal AGI-aesthetic terminal GIF renderer.

Clean, dark, typographic. Feels like peering into the mind of an
intelligence that is just barely emerging:
  • Deep charcoal / near-black canvas
  • Single accent colour: electric blue (#3B82F6) — the "thought" colour
  • Razor-thin 1 px border, generous whitespace
  • Hairline progress bar — no segments, just a smooth fill
  • Sparse floating "neuron" dots that drift subtly per frame
  • No grain, no noise, no blur — pixel-perfect sharpness
  • Section labels in subdued grey; content in crisp white / blue
"""

import math
import os
import random
import re
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# FONT
# ============================================================


def _get_mono_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


get_mono_font = _get_mono_font  # public alias


# ============================================================
# PALETTE — modern minimal
# ============================================================

_BG        = (13, 13, 15)       # near-black
_SURFACE   = (20, 20, 24)       # slightly lighter panels
_BORDER    = (38, 38, 45)       # thin border / separator
_DIM       = (72, 72, 82)       # muted labels
_TEXT      = (220, 220, 228)    # primary text — cool white
_BLUE      = (59, 130, 246)     # electric blue accent
_BLUE_DIM  = (30, 65, 120)     # blue at 50 % for glows / track
_BLUE_FAINT = (20, 42, 76)    # faintest blue
_USER_CLR  = (167, 139, 250)   # soft violet for user
_ASST_CLR  = (96, 205, 255)    # bright cyan-blue for assistant
_WHITE     = (248, 248, 252)


# ============================================================
# DRAWING HELPERS
# ============================================================


def _gradient_bg(w: int, h: int) -> Image.Image:
    """Subtle vertical gradient: _BG at top → slightly warmer at bottom."""
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / h
        r = int(_BG[0] + 4 * t)
        g = int(_BG[1] + 3 * t)
        b = int(_BG[2] + 2 * t)
        for x in range(w):
            px[x, y] = (r, g, b)
    return img


def _draw_neuron_dots(draw: ImageDraw.Draw, w: int, h: int,
                      seed: int = 0) -> None:
    """Sparse, softly glowing dots — like stray neurons firing."""
    rng = random.Random(seed)
    n = rng.randint(6, 14)
    for _ in range(n):
        x = rng.randint(40, w - 40)
        y = rng.randint(30, h - 30)
        r = rng.choice([1, 1, 2, 2, 3])
        # outer glow
        if r >= 2:
            draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2],
                         fill=_BLUE_FAINT)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=_BLUE_DIM)
        # bright centre
        draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=_BLUE)


def _draw_progress_bar(draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont,
                       x: int, y: int, w: int, h: int,
                       progress: float,
                       step_cur: int, step_total: int) -> None:
    """Hairline smooth progress bar with numeric label."""
    # track
    draw.rounded_rectangle([x, y, x + w, y + h],
                           radius=h // 2, fill=_SURFACE, outline=_BORDER)
    # fill
    fill_w = max(h, int(w * progress))
    draw.rounded_rectangle([x, y, x + fill_w, y + h],
                           radius=h // 2, fill=_BLUE)
    # bright leading edge
    if fill_w > h:
        draw.ellipse([x + fill_w - 3, y, x + fill_w + 3, y + h],
                     fill=_WHITE)

    # label
    pct = f"{progress * 100:5.1f}%"
    label = f"step {step_cur}/{step_total}  {pct}"
    draw.text((x + w + 14, y - 1), label, font=font, fill=_DIM)


def _draw_thin_border(draw: ImageDraw.Draw, w: int, h: int,
                      m: int = 16) -> None:
    """Single 1 px rounded rectangle border."""
    draw.rounded_rectangle([m, m, w - m, h - m],
                           radius=8, outline=_BORDER, width=1)


def _draw_header_line(draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont,
                      x: int, y: int, w: int) -> int:
    """Thin separator with centred 'AGI' badge. Returns new y."""
    # line
    draw.line([(x, y), (x + w, y)], fill=_BORDER, width=1)
    # badge
    badge = " DIFFUSION LM "
    tw = draw.textlength(badge, font=font)
    bx = x + (w - tw) / 2
    draw.rectangle([bx - 6, y - 8, bx + tw + 6, y + 8], fill=_BG)
    draw.text((bx, y - 7), badge, font=font, fill=_DIM)
    return y + 18


# ============================================================
# TEXT UTILITIES
# ============================================================


def wrap_text_to_width(text: str, max_chars: int = 90) -> List[str]:
    out: List[str] = []
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
# MODERN MINIMAL AGI RENDERER
# ============================================================


def render_terminal_frame_classic(
    lines: List[str],
    width: int = 1200,
    height: int = 700,
    font_size: int = 18,
    margin: int = 36,
    line_spacing: int = 7,
    jitter: int = 0,           # kept for API compat — unused
    _frame_seed: int = 0,      # varies per frame for dot animation
) -> Image.Image:
    """Render one frame — modern, minimal, AGI aesthetic."""

    img = _gradient_bg(width, height)
    draw = ImageDraw.Draw(img)
    font = _get_mono_font(font_size)
    font_sm = _get_mono_font(max(11, font_size - 4))
    font_xs = _get_mono_font(max(10, font_size - 6))

    # floating neuron dots (subtle, behind everything)
    _draw_neuron_dots(draw, width, height, seed=_frame_seed)

    # thin border
    _draw_thin_border(draw, width, height, m=14)

    # parse step info
    step_cur = step_total = None
    for line in lines:
        if "diffusion step" in line.lower():
            m = re.search(r"(\d+)/(\d+)", line)
            if m:
                step_cur, step_total = int(m.group(1)), int(m.group(2))
                break

    text_left = margin + 10
    text_right = width - margin - 10
    content_w = text_right - text_left
    y = margin + 4

    for idx, line in enumerate(lines):
        if "====" in line:
            # header separator with badge
            y = _draw_header_line(draw, font_xs, text_left, y + 4, content_w)

        elif "diffusion step" in line.lower() and step_cur is not None:
            progress = step_cur / step_total
            _draw_progress_bar(draw, font_xs, text_left, y,
                               content_w - 200, 6, progress,
                               step_cur, step_total)
            y += 6 + line_spacing + 8

        elif line.startswith("[You]:"):
            draw.text((text_left, y), "USER", font=font_xs, fill=_USER_CLR)
            # thin accent line under label
            lw = draw.textlength("USER", font=font_xs)
            draw.line([(text_left, y + font_size - 2),
                       (text_left + lw, y + font_size - 2)],
                      fill=(*_USER_CLR[:3], 80) if False else _USER_CLR,  # PIL RGB
                      width=1)
            y += font_size + line_spacing

        elif line.startswith("[Assistant]:"):
            draw.text((text_left, y), "ASSISTANT", font=font_xs, fill=_ASST_CLR)
            lw = draw.textlength("ASSISTANT", font=font_xs)
            draw.line([(text_left, y + font_size - 2),
                       (text_left + lw, y + font_size - 2)],
                      fill=_ASST_CLR, width=1)
            y += font_size + line_spacing

        elif "<Starting a new chat" in line:
            draw.text((text_left, y), line, font=font_xs, fill=_DIM)
            y += font_size + line_spacing

        elif line.strip() == "":
            y += line_spacing

        else:
            # determine section
            in_user = in_asst = False
            for prev in lines[:idx]:
                if "[You]:" in prev:
                    in_user, in_asst = True, False
                elif "[Assistant]:" in prev:
                    in_user, in_asst = False, True

            if in_user:
                clr = _WHITE
            elif in_asst:
                clr = _TEXT
            else:
                clr = _DIM

            draw.text((text_left, y), line, font=font, fill=clr)
            y += font_size + line_spacing

        if y > height - margin:
            break

    # bottom-right subtle branding
    tag = "diffusion-lm"
    tw = draw.textlength(tag, font=font_xs)
    draw.text((width - 24 - tw, height - 28), tag, font=font_xs, fill=_BORDER)

    # pulsing blue dot — bottom left "status alive"
    dot_r = 3
    dx, dy = 28, height - 26
    draw.ellipse([dx - dot_r - 3, dy - dot_r - 3,
                  dx + dot_r + 3, dy + dot_r + 3], fill=_BLUE_FAINT)
    draw.ellipse([dx - dot_r, dy - dot_r, dx + dot_r, dy + dot_r], fill=_BLUE)

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
    """Create a modern-minimal AGI-aesthetic GIF from diffusion frames."""
    gif_frames: List[np.ndarray] = []

    for i, (s, decoded) in enumerate(frames):
        lines = make_chat_lines(user_prompt, decoded)
        lines.insert(2, f"(diffusion step {s:03d}/{diffusion_steps:03d})")
        img = render_terminal_frame_classic(lines, _frame_seed=i)
        gif_frames.append(np.array(img))

    imageio.mimsave(gif_path, gif_frames, duration=0.09)
    print("Saved:", gif_path)
