"""Deterministic synthetic scenes from the recovered Background template.

Nature/urban categories also differ in palette. This is a compound background
appearance intervention, not an isolated semantic-context intervention.
"""
import hashlib
import numpy as np
from PIL import Image, ImageDraw
IMAGE_SIZE = (224, 224)

def rng_for(label):
    digest = hashlib.sha256(label.encode()).digest()
    seed = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed)


def gradient(top, bottom):
    h, w = IMAGE_SIZE[1], IMAGE_SIZE[0]
    top = np.array(top, dtype=np.float32)
    bottom = np.array(bottom, dtype=np.float32)

    t = np.linspace(0, 1, h)[:, None]
    rows = top[None, :] * (1 - t) + bottom[None, :] * t
    arr = np.repeat(rows[:, None, :], w, axis=1)

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def nature_background(seed):
    """
    Synthetic Green/Nature scene with deterministic variations.
    """

    rng = rng_for(f"nature::{seed}")

    img = gradient(
        [
            int(rng.integers(125, 195)),
            int(rng.integers(185, 230)),
            int(rng.integers(215, 250)),
        ],
        [
            int(rng.integers(20, 65)),
            int(rng.integers(95, 155)),
            int(rng.integers(25, 75)),
        ],
    )

    draw = ImageDraw.Draw(img, "RGBA")

    ground = int(IMAGE_SIZE[1] * 0.72)
    draw.rectangle([0, ground, 224, 224], fill=(35, 115, 45, 255))

    # Hills
    for layer in range(3):
        base = int(IMAGE_SIZE[1] * (0.48 + 0.08 * layer))
        pts = [(0, 224), (0, base)]
        for x in range(0, 240, 16):
            pts.append((x, base + int(rng.integers(-25, 25))))
        pts.append((224, 224))
        draw.polygon(
            pts,
            fill=(
                int(rng.integers(25, 80)),
                int(rng.integers(80, 150)),
                int(rng.integers(25, 80)),
                255,
            ),
        )

    # Trees
    for _ in range(int(rng.integers(5, 10))):
        x = int(rng.integers(5, 219))
        y = int(rng.integers(45, 145))
        trunk_h = int(rng.integers(25, 65))
        draw.rectangle(
            [x - 3, y, x + 3, y + trunk_h],
            fill=(95, 65, 35, 255),
        )

        r = int(rng.integers(14, 28))
        greens = [
            (25, 90, 35, 240),
            (35, 125, 45, 240),
            (55, 145, 55, 235),
        ]
        color = greens[int(rng.integers(0, len(greens)))]
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=color,
        )

    # Grass texture
    for _ in range(220):
        x = int(rng.integers(0, 224))
        y = int(rng.integers(ground, 224))
        length = int(rng.integers(3, 11))
        draw.line(
            [x, y, x + int(rng.integers(-3, 4)), y - length],
            fill=(25, 80, 30, 190),
            width=1,
        )

    return img


def urban_indoor_background(seed):
    """
    Synthetic Blue/Urban-Indoor scene with deterministic variations.
    """

    rng = rng_for(f"urban::{seed}")

    img = gradient(
        [
            int(rng.integers(30, 85)),
            int(rng.integers(75, 130)),
            int(rng.integers(135, 210)),
        ],
        [
            int(rng.integers(20, 60)),
            int(rng.integers(45, 95)),
            int(rng.integers(85, 155)),
        ],
    )

    draw = ImageDraw.Draw(img, "RGBA")

    floor_y = int(IMAGE_SIZE[1] * 0.72)

    # Indoor floor / urban ground
    draw.rectangle(
        [0, floor_y, 224, 224],
        fill=(55, 75, 105, 255),
    )

    # Wall panels
    panels = int(rng.integers(3, 7))
    pw = 224 // panels

    for i in range(panels):
        x0 = i * pw
        draw.rectangle(
            [x0, 0, x0 + pw - 2, floor_y],
            fill=(
                int(rng.integers(50, 95)),
                int(rng.integers(85, 135)),
                int(rng.integers(145, 215)),
                255,
            ),
        )

    # Windows
    for _ in range(int(rng.integers(2, 5))):
        x = int(rng.integers(5, 175))
        y = int(rng.integers(15, 105))
        w = int(rng.integers(25, 55))
        h = int(rng.integers(25, 55))

        x1 = min(223, x + w)
        y1 = min(floor_y - 5, y + h)

        draw.rectangle(
            [x, y, x1, y1],
            fill=(35, 110, 190, 255),
            outline=(185, 205, 225, 255),
            width=2,
        )

        draw.line(
            [x + w // 2, y, x + w // 2, y1],
            fill=(165, 195, 220, 220),
        )

        draw.line(
            [x, y + h // 2, x1, y + h // 2],
            fill=(165, 195, 220, 220),
        )

    # Building / urban silhouettes
    for _ in range(int(rng.integers(3, 7))):
        x = int(rng.integers(0, 190))
        w = int(rng.integers(20, 55))
        h = int(rng.integers(30, 100))
        draw.rectangle(
            [x, floor_y - h, min(223, x + w), floor_y],
            fill=(25, 50, 85, 120),
        )

    # Floor lines
    for y in range(floor_y + 8, 224, 15):
        draw.line(
            [0, y, 224, y],
            fill=(125, 150, 180, 90),
        )

    return img


def make_background(background_type, seed):
    if background_type == "nature":
        return nature_background(seed)
    if background_type == "urban_indoor":
        return urban_indoor_background(seed)
    raise ValueError(background_type)
