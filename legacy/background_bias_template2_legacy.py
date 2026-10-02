from pathlib import Path
import csv
import hashlib
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import torch

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


# ============================================================
# TEMPLATE 2: BACKGROUND BIAS
# ============================================================
# Uses the SAME 10 frozen classes as Template 1.
#
# Required experiment:
#   Group A (classes 0-4):
#       90% Green/Nature
#       10% Blue/Urban-Indoor
#
#   Group B (classes 5-9):
#       90% Blue/Urban-Indoor
#       10% Green/Nature
#
# Evaluation modes:
#   1. correlated
#   2. balanced
#   3. counterfactual
#
# Foreground:
#   SAM automatic mask generation.
#
# Background:
#   Multiple deterministic synthetic Green/Nature and
#   Blue/Urban-Indoor backgrounds.
#
# The original archive is READ ONLY.
# No model training is performed.
# ============================================================


# -----------------------------
# Paths
# -----------------------------

ROOT = Path(__file__).resolve().parent

SOURCE_DIR = ROOT / "archive"
METADATA_FILE = ROOT / "project_metadata" / "selected_classes.json"

OUTPUT_DIR = ROOT / "background_bias_dataset"
REPORT_DIR = ROOT / "reports" / "background_bias"

CHECKPOINT = ROOT / "checkpoints" / "sam_vit_b_01ec64.pth"

IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42

MODES = ("correlated", "balanced", "counterfactual")
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


# -----------------------------
# Device
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# METADATA / DATASET
# ============================================================

def load_frozen_classes():
    """Load exactly the 10 frozen classes used in Template 1."""

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing frozen metadata:\n{METADATA_FILE}"
        )

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    classes = metadata.get("classes")

    if not isinstance(classes, list) or len(classes) != 10:
        raise ValueError(
            "selected_classes.json must contain exactly 10 classes."
        )

    classes = sorted(classes, key=lambda x: int(x["index"]))

    if [int(x["index"]) for x in classes] != list(range(10)):
        raise ValueError("Class indices must be exactly 0 through 9.")

    return classes


def resolve_class_source(item):
    """Resolve the source directory of a frozen class."""

    if item.get("path"):
        p = Path(item["path"])
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p

    class_id = str(item["id"])
    direct = SOURCE_DIR / class_id
    if direct.exists():
        return direct

    matches = [
        p for p in SOURCE_DIR.rglob(class_id)
        if p.is_dir()
    ]

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Could not uniquely resolve class '{class_id}' inside {SOURCE_DIR}"
    )


def image_files(directory):
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def output_name(source):
    key = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:12]
    return f"{source.stem}__{key}.png"


# ============================================================
# SAM
# ============================================================

def load_sam():
    """Load official SAM ViT-B."""

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found:\n{CHECKPOINT}\n\n"
            "Expected: checkpoints/sam_vit_b_01ec64.pth"
        )

    print(f"\nLoading SAM ViT-B on {DEVICE}...")

    sam = sam_model_registry["vit_b"](
        checkpoint=str(CHECKPOINT)
    )
    sam.to(device=DEVICE)
    sam.eval()

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=8,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

    print("SAM loaded.")
    return generator


def center_score(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0.0

    h, w = mask.shape
    cx = xs.mean() / w
    cy = ys.mean() / h

    d = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
    dmax = np.sqrt(0.5 ** 2 + 0.5 ** 2)

    return float(1.0 - d / dmax)


def border_fraction(mask):
    if mask.sum() == 0:
        return 1.0

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    return float((mask & border).sum() / mask.sum())


def area_score(mask):
    frac = float(mask.mean())

    if 0.05 <= frac <= 0.80:
        return 1.0

    if frac < 0.05:
        return frac / 0.05

    return max(0.0, (1.0 - frac) / 0.20)


def choose_foreground_mask(annotations):
    """
    SAM returns multiple masks. The project specification does not
    define a manual point/box prompt or a mask-selection rule.
    Therefore this implementation records and uses a deterministic
    candidate-selection heuristic based on SAM scores plus:
      - centrality
      - object-sized area
      - boundary penalty
    """

    candidates = []

    for ann in annotations:
        mask = ann["segmentation"].astype(bool)
        if mask.sum() == 0:
            continue

        iou = float(ann.get("predicted_iou", 0.0))
        stability = float(ann.get("stability_score", 0.0))
        center = center_score(mask)
        area = area_score(mask)
        border = border_fraction(mask)

        score = (
            2.0 * iou
            + 2.0 * stability
            + 1.5 * center
            + 1.0 * area
            - 1.5 * border
        )

        candidates.append({
            "mask": mask,
            "score": score,
            "area_fraction": float(mask.mean()),
            "predicted_iou": iou,
            "stability_score": stability,
            "center_score": center,
            "border_fraction": border,
            "area_score": area,
        })

    if not candidates:
        raise RuntimeError("SAM returned no usable mask.")

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best = candidates[0]

    scores = np.array([x["score"] for x in candidates], dtype=float)
    lo, hi = scores.min(), scores.max()
    best["selection_confidence"] = (
        1.0 if hi == lo else float((best["score"] - lo) / (hi - lo))
    )
    best["num_sam_masks"] = len(annotations)

    return best["mask"], best


# -----------------------------
# Cache one SAM mask per image
# -----------------------------

def cache_paths(source):
    key = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:16]
    cache = OUTPUT_DIR / "_sam_cache"
    return cache / f"{key}.npy", cache / f"{key}.json"


def get_mask(source, generator):
    mask_file, info_file = cache_paths(source)

    if mask_file.exists() and info_file.exists():
        mask = np.load(mask_file).astype(bool)
        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        return mask, info

    image = np.asarray(load_rgb(source))
    annotations = generator.generate(image)

    mask, info = choose_foreground_mask(annotations)

    info["source"] = str(source)
    info["source_size"] = list(load_rgb(source).size)

    mask_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_file, mask.astype(np.uint8))

    json_safe_info = {}

    for key, value in info.items():
        if isinstance(value, np.ndarray):
            continue
        elif isinstance(value, np.generic):
            json_safe_info[key] = value.item()
        else:
            json_safe_info[key] = value

    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(json_safe_info, f, indent=4)

    return mask, info


# ============================================================
# SYNTHETIC BACKGROUNDS
# ============================================================

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


# ============================================================
# COMPOSITING
# ============================================================

def mask_to_224(mask):
    m = Image.fromarray(mask.astype(np.uint8) * 255, "L")
    m = m.resize(IMAGE_SIZE, Image.Resampling.NEAREST)
    return m.filter(ImageFilter.GaussianBlur(radius=1.2))


def replace_background(original, mask, background):
    original = original.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    background = background.resize(IMAGE_SIZE, Image.Resampling.BICUBIC)
    alpha = mask_to_224(mask)
    return Image.composite(original, background, alpha)


# ============================================================
# ASSIGNMENTS
# ============================================================

def correlated_assignments(paths, class_index):
    paths = list(paths)
    rng = random.Random(f"{RANDOM_SEED}:corr:{class_index}")
    rng.shuffle(paths)

    n_corr = int(len(paths) * 0.90)

    if class_index < 5:
        corr = "nature"
        mismatch = "urban_indoor"
    else:
        corr = "urban_indoor"
        mismatch = "nature"

    out = {}

    for p in paths[:n_corr]:
        out[p] = (corr, True)

    for p in paths[n_corr:]:
        out[p] = (mismatch, False)

    return out


def balanced_assignments(paths, class_index):
    """
    Approximately / exactly balanced within each class:
    n//2 nature and the remainder urban_indoor.
    """

    paths = list(paths)
    rng = random.Random(f"{RANDOM_SEED}:balanced:{class_index}")
    rng.shuffle(paths)

    n_nature = len(paths) // 2

    out = {}

    for p in paths[:n_nature]:
        out[p] = ("nature", None)

    for p in paths[n_nature:]:
        out[p] = ("urban_indoor", None)

    return out


def counterfactual_assignments(paths, class_index):
    bg = "urban_indoor" if class_index < 5 else "nature"
    return {p: (bg, False) for p in paths}


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(classes, sam_generator):
    """
    SAM is executed ONCE per source image.
    The same mask is reused for all three modes.
    """

    stats = {
        mode: {}
        for mode in MODES
    }

    manifest = []

    for item in classes:
        class_index = int(item["index"])
        class_id = str(item["id"])
        source_dir = resolve_class_source(item)
        paths = image_files(source_dir)

        if not paths:
            raise RuntimeError(f"No images found for {source_dir}")

        print(
            f"\nClass {class_index} | {class_id} | "
            f"{len(paths)} source images"
        )

        assignments = {
            "correlated": correlated_assignments(paths, class_index),
            "balanced": balanced_assignments(paths, class_index),
            "counterfactual": counterfactual_assignments(paths, class_index),
        }

        for mode in MODES:
            stats[mode][class_id] = {
                "class_index": class_index,
                "total": len(paths),
                "nature": 0,
                "urban_indoor": 0,
                "correlated": 0,
                "mismatch": 0,
            }

        for i, source in enumerate(paths, start=1):

            mask, mask_info = get_mask(
                source,
                sam_generator
            )

            for mode in MODES:
                background_type, correlation = assignments[mode][source]

                seed = (
                    f"{RANDOM_SEED}|{mode}|{class_id}|"
                    f"{source.name}"
                )

                destination = (
                    OUTPUT_DIR
                    / mode
                    / class_id
                    / output_name(source)
                )

                if not destination.exists():
                    output = replace_background(
                        load_rgb(source),
                        mask,
                        make_background(background_type, seed)
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    # Publish only complete PNGs, so an interrupted write is
                    # regenerated on the next run instead of being skipped.
                    temporary = destination.with_suffix(".png.tmp")
                    output.save(temporary, format="PNG")
                    temporary.replace(destination)

                stats[mode][class_id][background_type] += 1

                if correlation is True:
                    stats[mode][class_id]["correlated"] += 1
                elif correlation is False:
                    stats[mode][class_id]["mismatch"] += 1

                manifest.append({
                    "source": str(source),
                    "output": str(destination),
                    "class_index": class_index,
                    "class_id": class_id,
                    "mode": mode,
                    "background_type": background_type,
                    "correlated": (
                        "" if correlation is None
                        else str(correlation)
                    ),
                    "sam_area_fraction": mask_info["area_fraction"],
                    "sam_predicted_iou": mask_info["predicted_iou"],
                    "sam_stability": mask_info["stability_score"],
                    "sam_center_score": mask_info["center_score"],
                    "sam_border_fraction": mask_info["border_fraction"],
                    "sam_selection_confidence": mask_info["selection_confidence"],
                })

            if i == 1 or i % 25 == 0 or i == len(paths):
                print(f"  processed {i}/{len(paths)}")

        for mode in MODES:
            show_class_stats(
                mode,
                class_index,
                class_id,
                stats[mode][class_id]
            )

    save_manifest(manifest)

    return stats


def show_class_stats(mode, class_index, class_id, info):
    n = info["total"]

    print(
        f"  {mode:14s} | "
        f"Class {class_index:2d} | {class_id} | "
        f"Nature {info['nature']:5d} "
        f"({100 * info['nature'] / n:6.2f}%) | "
        f"Urban/Indoor {info['urban_indoor']:5d} "
        f"({100 * info['urban_indoor'] / n:6.2f}%)"
    )


def save_manifest(rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "manifest.csv"

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest saved: {path}")


# ============================================================
# AUDITS
# ============================================================

def numerical_audit(stats):
    audit = {}

    for mode in MODES:
        audit[mode] = {}

        overall = {
            "total": 0,
            "nature": 0,
            "urban_indoor": 0,
            "correlated": 0,
            "mismatch": 0,
        }

        for class_id, info in stats[mode].items():
            total = info["total"]

            record = dict(info)
            record["nature_percentage"] = 100 * info["nature"] / total
            record["urban_indoor_percentage"] = (
                100 * info["urban_indoor"] / total
            )

            if mode == "correlated":
                record["correlated_percentage"] = (
                    100 * info["correlated"] / total
                )
                record["mismatch_percentage"] = (
                    100 * info["mismatch"] / total
                )

            audit[mode][class_id] = record

            for key in overall:
                overall[key] += info[key]

        audit[mode]["__overall__"] = {
            **overall,
            "nature_percentage": (
                100 * overall["nature"] / overall["total"]
            ),
            "urban_indoor_percentage": (
                100 * overall["urban_indoor"] / overall["total"]
            ),
        }

        if mode == "correlated":
            audit[mode]["__overall__"].update({
                "correlated_percentage": (
                    100 * overall["correlated"] / overall["total"]
                ),
                "mismatch_percentage": (
                    100 * overall["mismatch"] / overall["total"]
                ),
            })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_DIR / "numerical_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(audit, f, indent=4)

    return audit


def integrity_audit(classes):
    results = {}

    for mode in MODES:
        result = {
            "files_checked": 0,
            "bad_files": 0,
            "wrong_size": 0,
            "wrong_mode": 0,
        }

        for item in classes:
            class_id = str(item["id"])
            directory = OUTPUT_DIR / mode / class_id

            if not directory.exists():
                continue

            for path in image_files(directory):
                result["files_checked"] += 1

                try:
                    with Image.open(path) as img:
                        if img.size != IMAGE_SIZE:
                            result["wrong_size"] += 1
                        if img.mode != "RGB":
                            result["wrong_mode"] += 1
                except Exception:
                    result["bad_files"] += 1

        results[mode] = result

    with open(
        OUTPUT_DIR / "image_integrity_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4)

    return results


def mask_audit(classes):
    records = []

    for item in classes:
        source_dir = resolve_class_source(item)

        for source in image_files(source_dir):
            cached = load_cached_info(source)

            if cached is None:
                continue

            info = cached

            records.append({
                "class_index": int(item["index"]),
                "class_id": str(item["id"]),
                "source": str(source),
                "area_fraction": info["area_fraction"],
                "predicted_iou": info["predicted_iou"],
                "stability_score": info["stability_score"],
                "center_score": info["center_score"],
                "border_fraction": info["border_fraction"],
                "selection_confidence": info["selection_confidence"],
                "num_sam_masks": info["num_sam_masks"],
            })

    with open(
        OUTPUT_DIR / "sam_mask_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(records, f, indent=4)

    return records


def load_cached_info(source):
    _, info_file = cache_paths(source)

    if not info_file.exists():
        return None

    with open(info_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# VISUAL VERIFICATION
# ============================================================

def create_visual_samples(classes, samples_per_class=1):
    out_dir = REPORT_DIR / "visual_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for item in classes:
        source_dir = resolve_class_source(item)

        for sample_index, source in enumerate(
            image_files(source_dir)[:samples_per_class]
        ):

            cached = load_cached_mask(source)

            if cached is None:
                continue

            mask, _ = cached

            original = load_rgb(source).resize(
                IMAGE_SIZE,
                Image.Resampling.BICUBIC
            )

            mask_img = mask_to_224(mask)

            foreground = Image.new(
                "RGB",
                IMAGE_SIZE,
                "black"
            )
            foreground.paste(
                original,
                mask=mask_img
            )

            class_index = int(item["index"])

            corr_bg_type = (
                "nature" if class_index < 5
                else "urban_indoor"
            )

            counter_bg_type = (
                "urban_indoor" if class_index < 5
                else "nature"
            )

            correlated = replace_background(
                load_rgb(source),
                mask,
                make_background(
                    corr_bg_type,
                    f"visual|corr|{item['id']}|{sample_index}"
                )
            )

            counterfactual = replace_background(
                load_rgb(source),
                mask,
                make_background(
                    counter_bg_type,
                    f"visual|counter|{item['id']}|{sample_index}"
                )
            )

            panels = [
                ("ORIGINAL", original),
                ("SAM MASK", mask_img.convert("RGB")),
                ("FOREGROUND", foreground),
                ("CORRELATED", correlated),
                ("COUNTERFACTUAL", counterfactual),
            ]

            label_h = 35
            w, h = IMAGE_SIZE

            canvas = Image.new(
                "RGB",
                (w * len(panels), h + label_h),
                "white"
            )

            draw = ImageDraw.Draw(canvas)

            for i, (label, img) in enumerate(panels):
                x = i * w
                draw.text((x + 8, 8), label, fill="black")
                canvas.paste(img.convert("RGB"), (x, label_h))

            output = (
                out_dir
                / f"class_{class_index}_{item['id']}_sample_{sample_index}.png"
            )

            canvas.save(output)

    print(f"Visual samples saved: {out_dir}")


# ============================================================
# GRAPHS
# ============================================================

def create_distribution_graphs(audit):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for mode in MODES:

        entries = [
            (k, v)
            for k, v in audit[mode].items()
            if k != "__overall__"
        ]
        entries.sort(key=lambda x: x[1]["class_index"])

        labels = [
            f"Class {v['class_index']}"
            for _, v in entries
        ]

        nature = [
            v["nature_percentage"]
            for _, v in entries
        ]

        urban = [
            v["urban_indoor_percentage"]
            for _, v in entries
        ]

        x = np.arange(len(labels))
        width = 0.38

        plt.figure(figsize=(12, 6))

        plt.bar(
            x - width / 2,
            nature,
            width,
            label="Green / Nature"
        )

        plt.bar(
            x + width / 2,
            urban,
            width,
            label="Blue / Urban-Indoor"
        )

        if mode == "correlated":
            plt.axhline(
                90,
                linestyle="--",
                linewidth=1,
                label="90% target"
            )
            plt.axhline(
                10,
                linestyle=":",
                linewidth=1,
                label="10% target"
            )

        if mode == "balanced":
            plt.axhline(
                50,
                linestyle="--",
                linewidth=1,
                label="50% reference"
            )

        plt.xticks(x, labels)
        plt.ylim(0, 100)
        plt.xlabel("Class")
        plt.ylabel("Background percentage")
        plt.title(f"Background Bias — {mode.capitalize()}")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / f"{mode}_background_distribution.png",
            dpi=200
        )

        plt.close()


# ============================================================
# METADATA / REPORT
# ============================================================

def save_metadata(classes):
    metadata = {
        "template": "BACKGROUND_BIAS",
        "source_dataset": str(SOURCE_DIR),
        "image_resolution": [224, 224],
        "output_mode": "RGB",
        "segmentation": {
            "model": "SAM",
            "model_type": "vit_b",
            "checkpoint": str(CHECKPOINT),
            "mask_generation": "automatic",
            "selection_rule": (
                "predicted IoU + stability + center proximity "
                "+ object-sized area preference - border penalty"
            ),
        },
        "backgrounds": {
            "nature": "synthetic Green/Nature backgrounds",
            "urban_indoor": (
                "synthetic Blue/Urban-Indoor backgrounds"
            ),
        },
        "group_a": {
            "classes": [0, 1, 2, 3, 4],
            "correlated": "90% nature / 10% urban_indoor",
            "counterfactual": "100% urban_indoor",
        },
        "group_b": {
            "classes": [5, 6, 7, 8, 9],
            "correlated": "90% urban_indoor / 10% nature",
            "counterfactual": "100% nature",
        },
        "balanced": "approximately 50% / 50% per class",
        "evaluation_modes": list(MODES),
        "random_seed": RANDOM_SEED,
        "classes": classes,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_DIR / "experiment_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)


def create_report(classes, audit, integrity, mask_records):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Template 2 — BACKGROUND Bias",
        "",
        "## 1. Objective",
        "",
        "Introduce a label-correlated background factor while "
        "retaining the foreground object selected from the original image.",
        "",
        "## 2. Frozen classes",
    ]

    for item in classes:
        group = "A" if int(item["index"]) < 5 else "B"
        lines.append(
            f"- Class {item['index']}: "
            f"{item.get('name', item['id'])} (Group {group})"
        )

    lines += [
        "",
        "## 3. Foreground segmentation",
        "",
        "SAM automatic mask generation is used. Because the project "
        "specification does not define a manual SAM prompt or a "
        "per-image mask selection rule, a deterministic selection "
        "heuristic is applied using SAM scores, centrality, "
        "object-sized area preference, and border penalty.",
        "",
        "## 4. Background replacement",
        "",
        "Two synthetic background categories are generated:",
        "",
        "- Green / Nature",
        "- Blue / Urban-Indoor",
        "",
        "Multiple deterministic variants are generated inside each "
        "category. The selected foreground is retained and the "
        "remaining background is replaced.",
        "",
        "## 5. Correlation design",
        "",
        "Group A (classes 0–4): 90% Nature, 10% Urban/Indoor.",
        "",
        "Group B (classes 5–9): 90% Urban/Indoor, 10% Nature.",
        "",
        "## 6. Evaluation modes",
        "",
        "### Correlated",
        "The 90/10 class-background relationship is applied.",
        "",
        "### Balanced",
        "Nature and Urban/Indoor backgrounds are approximately balanced "
        "within each class.",
        "",
        "### Counterfactual",
        "The original class-background relationship is reversed.",
        "",
        "## 7. Correlated numerical audit",
    ]

    overall = audit["correlated"]["__overall__"]

    lines += [
        f"- Total images: {overall['total']}",
        f"- Correlated assignments: {overall['correlated']}",
        f"- Mismatch assignments: {overall['mismatch']}",
        f"- Correlated percentage: {overall['correlated_percentage']:.2f}%",
        f"- Mismatch percentage: {overall['mismatch_percentage']:.2f}%",
        "",
        "## 8. Image integrity",
    ]

    for mode, result in integrity.items():
        lines += [
            f"### {mode}",
            f"- Files checked: {result['files_checked']}",
            f"- Bad files: {result['bad_files']}",
            f"- Wrong size: {result['wrong_size']}",
            f"- Wrong mode: {result['wrong_mode']}",
        ]

    lines += [
        "",
        "## 9. SAM mask audit",
        "",
        f"- Mask records: {len(mask_records)}",
    ]

    if mask_records:
        lines += [
            f"- Mean mask area fraction: "
            f"{np.mean([x['area_fraction'] for x in mask_records]):.4f}",
            f"- Mean predicted IoU: "
            f"{np.mean([x['predicted_iou'] for x in mask_records]):.4f}",
            f"- Mean stability: "
            f"{np.mean([x['stability_score'] for x in mask_records]):.4f}",
        ]

    lines += [
        "",
        "## 10. Visual verification",
        "",
        "Visual samples are saved under "
        "`reports/background_bias/visual_samples/`.",
        "",
        "Each sample shows:",
        "",
        "- Original",
        "- SAM Mask",
        "- Foreground",
        "- Correlated Background",
        "- Counterfactual Background",
        "",
        "## 11. Generated artifacts",
        "",
        "- `background_bias_dataset/correlated/`",
        "- `background_bias_dataset/balanced/`",
        "- `background_bias_dataset/counterfactual/`",
        "- `background_bias_dataset/_sam_cache/`",
        "- `background_bias_dataset/manifest.csv`",
        "- `background_bias_dataset/numerical_audit.json`",
        "- `background_bias_dataset/sam_mask_audit.json`",
        "- `background_bias_dataset/image_integrity_audit.json`",
        "- `reports/background_bias/`",
        "",
        "## 12. Model training",
        "",
        "No model training is performed in this stage.",
    ]

    report = REPORT_DIR / "Template_2_Background_Bias_Report.md"

    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("TEMPLATE 2 — BACKGROUND BIAS")
    print("=" * 80)

    print(f"Device: {DEVICE}")
    print(f"Source: {SOURCE_DIR}")
    print(f"Metadata: {METADATA_FILE}")
    print(f"SAM checkpoint: {CHECKPOINT}")

    classes = load_frozen_classes()

    print("\nFrozen classes:")
    for item in classes:
        group = "A" if int(item["index"]) < 5 else "B"
        print(
            f"  {item['index']} | {item['id']} | Group {group}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sam_generator = load_sam()

    print("\n" + "=" * 80)
    print("GENERATING CORRELATED / BALANCED / COUNTERFACTUAL DATASETS")
    print("=" * 80)

    stats = generate_dataset(
        classes,
        sam_generator
    )

    audit = numerical_audit(stats)

    print("\n" + "=" * 80)
    print("NUMERICAL AUDIT")
    print("=" * 80)

    for mode in MODES:
        overall = audit[mode]["__overall__"]
        print(f"\n{mode.upper()}")
        print(
            f"  Total: {overall['total']}"
        )
        print(
            f"  Nature: {overall['nature']} "
            f"({overall['nature_percentage']:.2f}%)"
        )
        print(
            f"  Urban/Indoor: {overall['urban_indoor']} "
            f"({overall['urban_indoor_percentage']:.2f}%)"
        )

        if mode == "correlated":
            print(
                f"  Correlated: {overall['correlated']} "
                f"({overall['correlated_percentage']:.2f}%)"
            )
            print(
                f"  Mismatch: {overall['mismatch']} "
                f"({overall['mismatch_percentage']:.2f}%)"
            )

    integrity = integrity_audit(classes)

    print("\n" + "=" * 80)
    print("IMAGE INTEGRITY")
    print("=" * 80)

    for mode, result in integrity.items():
        print(
            f"{mode:14s} | "
            f"checked={result['files_checked']} | "
            f"bad={result['bad_files']} | "
            f"wrong_size={result['wrong_size']} | "
            f"wrong_mode={result['wrong_mode']}"
        )

    records = mask_audit(classes)

    print("\n" + "=" * 80)
    print("SAM MASK AUDIT")
    print("=" * 80)
    print(f"Mask records: {len(records)}")

    create_visual_samples(classes)
    create_distribution_graphs(audit)

    save_metadata(classes)

    report = create_report(
        classes,
        audit,
        integrity,
        records
    )

    print("\n" + "=" * 80)
    print("TEMPLATE 2 — BACKGROUND BIAS COMPLETE")
    print("=" * 80)

    for mode in MODES:
        print(OUTPUT_DIR / mode)

    print(f"\nReports: {REPORT_DIR}")
    print(f"Report: {report}")
    print("\nOriginal archive was NOT modified.")
    print("Model training was NOT performed.")


if __name__ == "__main__":
    main()
