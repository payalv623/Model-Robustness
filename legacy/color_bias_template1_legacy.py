from pathlib import Path
import json
import random
import csv

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import matplotlib.pyplot as plt


# ============================================================
# TEMPLATE 1: COLOR BIAS
# ============================================================
#
# This script:
#   1. Uses the frozen 10 classes in selected_classes.json.
#   2. Reads the original ImageNet-100 images from archive/.
#   3. Standardizes generated images to 224 x 224 RGB.
#   4. Applies 70% original + 30% red/blue color overlay.
#   5. Creates the four required Color evaluation modes:
#        - correlated
#        - randomized
#        - reversed
#        - grayscale
#   6. Performs numerical auditing.
#   7. Creates visual comparison images.
#   8. Creates correlation/distribution graphs.
#   9. Creates a Markdown research report from actual results.
#
# IMPORTANT:
#   - archive/ is READ ONLY.
#   - No model training is performed here.
#   - The script does NOT choose or replace the frozen classes.
#     selected_classes.json must already exist.
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SOURCE_DIR = PROJECT_ROOT / "archive"

METADATA_FILE = (
    PROJECT_ROOT
    / "project_metadata"
    / "selected_classes.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "color_bias_dataset"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "color_bias"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

# 70% original + 30% color overlay
ORIGINAL_WEIGHT = 0.70
TINT_WEIGHT = 0.30

# Intended training correlation
CORRELATED_RATIO = 0.90

# Reproducibility
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ============================================================
# COLOR DEFINITIONS
# ============================================================

RED_TINT = np.array(
    [255, 0, 0],
    dtype=np.float32
)

BLUE_TINT = np.array(
    [0, 0, 255],
    dtype=np.float32
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# METADATA
# ============================================================

def load_frozen_classes():
    """
    Load the already-frozen 10 classes.

    The code intentionally does NOT select classes automatically.
    This prevents the experiment from silently using a different
    set of classes.
    """

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "\nselected_classes.json was not found.\n\n"
            "Expected:\n"
            f"{METADATA_FILE}\n\n"
            "Create/verify the frozen 10-class metadata first."
        )

    with open(
        METADATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    classes = metadata.get("classes")

    if classes is None:
        raise ValueError(
            "selected_classes.json does not contain 'classes'."
        )

    if len(classes) != 10:
        raise ValueError(
            f"Exactly 10 classes are required. Found {len(classes)}."
        )

    expected_indices = list(range(10))

    actual_indices = sorted(
        int(item["index"])
        for item in classes
    )

    if actual_indices != expected_indices:
        raise ValueError(
            "Class indices must be exactly 0 through 9.\n"
            f"Found: {actual_indices}"
        )

    classes = sorted(
        classes,
        key=lambda item: int(item["index"])
    )

    for item in classes:
        required = {"index", "id"}

        missing = required - set(item.keys())

        if missing:
            raise ValueError(
                f"Missing fields {missing} in class metadata:\n"
                f"{item}"
            )

    return classes


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_images(directory):
    """
    Recursively find supported image files.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Source directory does not exist:\n{directory}"
        )

    return sorted(
        [
            p
            for p in directory.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower() in IMAGE_EXTENSIONS
            )
        ]
    )


def resolve_class_source(class_info):
    """
    Resolve the frozen class directory.

    The metadata may contain an explicit 'path'. If it does,
    that path is used. Otherwise the class ID is resolved
    underneath archive/.
    """

    if "path" in class_info:
        path = Path(class_info["path"])

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if path.exists():
            return path

    class_id = str(class_info["id"])

    direct = SOURCE_DIR / class_id

    if direct.exists():
        return direct

    # Search recursively for an exact directory name.
    matches = [
        p
        for p in SOURCE_DIR.rglob(class_id)
        if p.is_dir()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple source directories match class ID '{class_id}':\n"
            + "\n".join(str(p) for p in matches)
        )

    raise FileNotFoundError(
        f"Could not locate frozen class '{class_id}' inside:\n"
        f"{SOURCE_DIR}"
    )


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

def load_image(path):
    """
    Load, convert to RGB, and resize to 224 x 224.
    """

    with Image.open(path) as img:
        img = img.convert("RGB")
        return img.resize(
            IMAGE_SIZE,
            Image.Resampling.BICUBIC
        )


def apply_color_tint(image, tint_color):
    """
    Required Color Bias transformation:

        output = 0.70 * original
               + 0.30 * solid color

    This is a pixel-wise RGB blend.
    """

    original = np.asarray(
        image
    ).astype(np.float32)

    overlay = np.zeros_like(
        original
    )

    overlay[:, :, :] = tint_color

    output = (
        ORIGINAL_WEIGHT * original
        + TINT_WEIGHT * overlay
    )

    output = np.clip(
        output,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(
        output,
        mode="RGB"
    )


def make_red(image):
    return apply_color_tint(
        image,
        RED_TINT
    )


def make_blue(image):
    return apply_color_tint(
        image,
        BLUE_TINT
    )


def make_grayscale(image):
    """
    Grayscale is converted back to RGB so all generated
    outputs remain three-channel.
    """

    return ImageOps.grayscale(
        image
    ).convert("RGB")


# ============================================================
# CORRELATED ASSIGNMENT
# ============================================================

def exact_correlated_assignment(
    image_paths,
    class_index
):
    """
    Group A: classes 0-4
        90% red
        10% blue

    Group B: classes 5-9
        90% blue
        10% red

    The random ordering is reproducible.
    """

    image_paths = list(image_paths)

    random.shuffle(
        image_paths
    )

    total = len(image_paths)

    correlated_count = int(
        total * CORRELATED_RATIO
    )

    correlated_files = image_paths[
        :correlated_count
    ]

    mismatch_files = image_paths[
        correlated_count:
    ]

    assignments = []

    if class_index < 5:
        correlated_color = "red"
        mismatch_color = "blue"
    else:
        correlated_color = "blue"
        mismatch_color = "red"

    for path in correlated_files:
        assignments.append(
            {
                "source": path,
                "color": correlated_color,
                "correlated": True
            }
        )

    for path in mismatch_files:
        assignments.append(
            {
                "source": path,
                "color": mismatch_color,
                "correlated": False
            }
        )

    return assignments


# ============================================================
# DATASET RECORD
# ============================================================

def save_image(
    source_path,
    destination_path,
    mode,
    color=None
):
    """
    Generate and save one image.
    """

    image = load_image(
        source_path
    )

    if mode in {"correlated", "randomized", "reversed"}:

        if color == "red":
            output = make_red(image)

        elif color == "blue":
            output = make_blue(image)

        else:
            raise ValueError(
                f"Color must be 'red' or 'blue', got {color}"
            )

    elif mode == "grayscale":

        output = make_grayscale(
            image
        )

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.save(
        destination_path,
        format="PNG"
    )


# ============================================================
# MODE 1: CORRELATED
# ============================================================

def generate_correlated(
    classes
):
    """
    Generate the main biased Color dataset.

    Group A:
        90% red, 10% blue

    Group B:
        90% blue, 10% red
    """

    mode = "correlated"

    statistics = {}

    for item in classes:

        index = int(item["index"])
        class_id = str(item["id"])

        source_dir = resolve_class_source(
            item
        )

        image_paths = get_images(
            source_dir
        )

        if not image_paths:
            raise RuntimeError(
                f"No images found for class {class_id}:\n"
                f"{source_dir}"
            )

        assignments = exact_correlated_assignment(
            image_paths,
            index
        )

        destination_dir = (
            OUTPUT_DIR
            / mode
            / class_id
        )

        red_count = 0
        blue_count = 0
        correlated_count = 0
        mismatch_count = 0

        manifest_rows = []

        for assignment in assignments:

            source = assignment["source"]
            color = assignment["color"]
            correlated = assignment["correlated"]

            destination = (
                destination_dir
                / source.name
            ).with_suffix(".png")

            save_image(
                source,
                destination,
                mode,
                color
            )

            if color == "red":
                red_count += 1
            else:
                blue_count += 1

            if correlated:
                correlated_count += 1
            else:
                mismatch_count += 1

            manifest_rows.append(
                {
                    "source": str(source),
                    "output": str(destination),
                    "class_index": index,
                    "class_id": class_id,
                    "mode": mode,
                    "color": color,
                    "correlated": correlated
                }
            )

        save_manifest(
            destination_dir,
            manifest_rows
        )

        total = len(assignments)

        statistics[class_id] = {
            "class_index": index,
            "total": total,
            "red": red_count,
            "blue": blue_count,
            "correlated": correlated_count,
            "mismatch": mismatch_count,
            "correlated_percentage": (
                100.0 * correlated_count / total
            ),
            "mismatch_percentage": (
                100.0 * mismatch_count / total
            )
        }

        print_class_result(
            mode,
            item,
            statistics[class_id]
        )

    return statistics


# ============================================================
# MODE 2: RANDOMIZED
# ============================================================

def generate_randomized(
    classes,
    correlated_statistics
):
    """
    Remove class-color correlation while preserving the
    total number of red/blue assignments.

    The red/blue assignments produced by the correlated mode
    are collected and then randomly shuffled across ALL images.

    This avoids introducing a new fixed color ratio assumption.
    """

    mode = "randomized"

    all_records = []

    for item in classes:

        index = int(item["index"])
        class_id = str(item["id"])

        source_dir = resolve_class_source(
            item
        )

        image_paths = get_images(
            source_dir
        )

        for source in image_paths:

            all_records.append(
                {
                    "source": source,
                    "class_index": index,
                    "class_id": class_id
                }
            )

    if not all_records:
        raise RuntimeError(
            "No images found for randomized mode."
        )

    # Build the exact global color pool using the
    # correlated-mode totals.
    total_red = sum(
        info["red"]
        for info in correlated_statistics.values()
    )

    total_blue = sum(
        info["blue"]
        for info in correlated_statistics.values()
    )

    if total_red + total_blue != len(all_records):
        raise RuntimeError(
            "Color pool size does not match total image count."
        )

    colors = (
        ["red"] * total_red
        + ["blue"] * total_blue
    )

    random.shuffle(
        colors
    )

    destination_root = (
        OUTPUT_DIR
        / mode
    )

    stats = {}

    for record, color in zip(
        all_records,
        colors
    ):

        class_id = record["class_id"]
        index = record["class_index"]
        source = record["source"]

        destination = (
            destination_root
            / class_id
            / source.name
        ).with_suffix(".png")

        save_image(
            source,
            destination,
            mode,
            color
        )

        if class_id not in stats:
            stats[class_id] = {
                "class_index": index,
                "total": 0,
                "red": 0,
                "blue": 0
            }

        stats[class_id]["total"] += 1
        stats[class_id][color] += 1

    for class_id, info in stats.items():

        total = info["total"]

        info["red_percentage"] = (
            100.0 * info["red"] / total
        )

        info["blue_percentage"] = (
            100.0 * info["blue"] / total
        )

    return stats


# ============================================================
# MODE 3: REVERSED
# ============================================================

def generate_reversed(
    classes
):
    """
    Directly reverse the class-color correlation:

    Classes 0-4 -> BLUE
    Classes 5-9 -> RED
    """

    mode = "reversed"

    statistics = {}

    for item in classes:

        index = int(item["index"])
        class_id = str(item["id"])

        source_dir = resolve_class_source(
            item
        )

        image_paths = get_images(
            source_dir
        )

        color = (
            "blue"
            if index < 5
            else "red"
        )

        destination_dir = (
            OUTPUT_DIR
            / mode
            / class_id
        )

        for source in image_paths:

            destination = (
                destination_dir
                / source.name
            ).with_suffix(".png")

            save_image(
                source,
                destination,
                mode,
                color
            )

        statistics[class_id] = {
            "class_index": index,
            "total": len(image_paths),
            "color": color
        }

        print(
            f"{mode:11s} | "
            f"class {index:2d} | "
            f"{class_id} | "
            f"{color.upper()}"
        )

    return statistics


# ============================================================
# MODE 4: GRAYSCALE
# ============================================================

def generate_grayscale(
    classes
):
    """
    Remove color information by converting every image to
    grayscale, while keeping the final images RGB.
    """

    mode = "grayscale"

    statistics = {}

    for item in classes:

        index = int(item["index"])
        class_id = str(item["id"])

        source_dir = resolve_class_source(
            item
        )

        image_paths = get_images(
            source_dir
        )

        destination_dir = (
            OUTPUT_DIR
            / mode
            / class_id
        )

        for source in image_paths:

            destination = (
                destination_dir
                / source.name
            ).with_suffix(".png")

            save_image(
                source,
                destination,
                mode
            )

        statistics[class_id] = {
            "class_index": index,
            "total": len(image_paths)
        }

        print(
            f"{mode:11s} | "
            f"class {index:2d} | "
            f"{class_id} | "
            f"grayscale"
        )

    return statistics


# ============================================================
# MANIFEST
# ============================================================

def save_manifest(
    directory,
    rows
):
    """
    Save the exact transformation applied to each image.
    """

    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_file = (
        directory
        / "manifest.csv"
    )

    with open(
        manifest_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLASS RESULT PRINTING
# ============================================================

def print_class_result(
    mode,
    item,
    info
):
    """
    Print numerical evidence to the terminal.
    """

    print(
        f"{mode:11s} | "
        f"class {int(item['index']):2d} | "
        f"{item['id']} | "
        f"total={info['total']} | "
        f"red={info['red']} | "
        f"blue={info['blue']} | "
        f"corr={info['correlated_percentage']:.2f}% | "
        f"mismatch={info['mismatch_percentage']:.2f}%"
    )


# ============================================================
# DATASET AUDIT
# ============================================================

def audit_correlated(
    statistics
):
    """
    Audit the main biased dataset.
    """

    total = 0
    correlated = 0
    mismatch = 0

    for info in statistics.values():

        total += info["total"]
        correlated += info["correlated"]
        mismatch += info["mismatch"]

    audit = {
        "total_images": total,
        "correlated_images": correlated,
        "mismatch_images": mismatch,
        "correlated_percentage": (
            100.0 * correlated / total
        ),
        "mismatch_percentage": (
            100.0 * mismatch / total
        ),
        "per_class": statistics
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_DIR / "correlated_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            audit,
            f,
            indent=4
        )

    return audit


# ============================================================
# VERIFY OUTPUT IMAGES
# ============================================================

def verify_generated_images(
    classes
):
    """
    Verify the generated images are:
      - readable
      - 224x224
      - RGB
    """

    modes = [
        "correlated",
        "randomized",
        "reversed",
        "grayscale"
    ]

    results = {}

    for mode in modes:

        mode_result = {
            "files_checked": 0,
            "bad_files": 0,
            "wrong_size": 0,
            "wrong_mode": 0
        }

        for item in classes:

            class_id = str(item["id"])

            class_dir = (
                OUTPUT_DIR
                / mode
                / class_id
            )

            if not class_dir.exists():
                continue

            for path in get_images(
                class_dir
            ):

                mode_result["files_checked"] += 1

                try:
                    with Image.open(path) as img:

                        if img.size != IMAGE_SIZE:
                            mode_result["wrong_size"] += 1

                        if img.mode != "RGB":
                            mode_result["wrong_mode"] += 1

                except Exception:
                    mode_result["bad_files"] += 1

        results[mode] = mode_result

    with open(
        OUTPUT_DIR / "image_integrity_audit.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            indent=4
        )

    return results


# ============================================================
# VISUAL COMPARISON
# ============================================================

def create_visual_comparison(
    classes,
    samples_per_class=1
):
    """
    Create per-class visual panels:

        ORIGINAL
        RED
        BLUE
        CORRELATED
        REVERSED
        GRAYSCALE

    The same original image is used for each panel so that the
    visual difference is directly attributable to the
    transformation.
    """

    output_dir = (
        REPORT_DIR
        / "visual_samples"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for item in classes:

        source_dir = resolve_class_source(
            item
        )

        image_paths = get_images(
            source_dir
        )

        if not image_paths:
            continue

        for sample_index in range(
            min(samples_per_class, len(image_paths))
        ):

            source = image_paths[
                sample_index
            ]

            original = load_image(
                source
            )

            red = make_red(
                original
            )

            blue = make_blue(
                original
            )

            # Class-correlated color
            if int(item["index"]) < 5:
                correlated = red
                reversed_image = blue
            else:
                correlated = blue
                reversed_image = red

            grayscale = make_grayscale(
                original
            )

            panels = [
                ("ORIGINAL", original),
                ("RED", red),
                ("BLUE", blue),
                ("CORRELATED", correlated),
                ("REVERSED", reversed_image),
                ("GRAYSCALE", grayscale)
            ]

            width = IMAGE_SIZE[0]
            height = IMAGE_SIZE[1]

            label_h = 35

            canvas = Image.new(
                "RGB",
                (
                    width * 3,
                    (height + label_h) * 2
                ),
                "white"
            )

            draw = ImageDraw.Draw(
                canvas
            )

            for i, (
                label,
                image
            ) in enumerate(panels):

                row = i // 3
                col = i % 3

                x = col * width
                y = row * (height + label_h)

                draw.text(
                    (x + 8, y + 8),
                    label,
                    fill="black"
                )

                canvas.paste(
                    image,
                    (
                        x,
                        y + label_h
                    )
                )

            file_name = (
                f"class_{item['index']}_"
                f"{item['id']}_"
                f"sample_{sample_index}.png"
            )

            canvas.save(
                output_dir / file_name
            )

    print(
        f"\nVisual samples saved to:\n"
        f"{output_dir}"
    )


# ============================================================
# CORRELATION GRAPH
# ============================================================

def create_correlation_graph(
    statistics
):
    """
    Plot per-class correlated/mismatch percentages.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    indices = []
    correlated = []
    mismatch = []

    for class_id, info in sorted(
        statistics.items(),
        key=lambda x: x[1]["class_index"]
    ):

        indices.append(
            info["class_index"]
        )

        correlated.append(
            info["correlated_percentage"]
        )

        mismatch.append(
            info["mismatch_percentage"]
        )

    x = np.arange(
        len(indices)
    )

    width = 0.36

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        x - width / 2,
        correlated,
        width,
        label="Correlated"
    )

    plt.bar(
        x + width / 2,
        mismatch,
        width,
        label="Mismatch"
    )

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

    plt.xticks(
        x,
        [f"Class {i}" for i in indices]
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Color Bias: Correlated vs Mismatch"
    )

    plt.ylim(
        0,
        100
    )

    plt.legend()

    plt.tight_layout()

    output = (
        REPORT_DIR
        / "correlation_distribution.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()

    print(
        f"Correlation graph saved to:\n{output}"
    )


# ============================================================
# MODE COMPARISON GRAPH
# ============================================================

def create_mode_summary_graph(
    correlated_stats,
    randomized_stats
):
    """
    Visualize the difference between the correlated and
    randomized modes.

    For randomized mode, the graph reports the actual per-class
    red percentage produced by the shuffle.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    classes_sorted = sorted(
        correlated_stats.items(),
        key=lambda x: x[1]["class_index"]
    )

    labels = [
        f"Class {info['class_index']}"
        for _, info in classes_sorted
    ]

    corr_values = [
        info["correlated_percentage"]
        for _, info in classes_sorted
    ]

    random_red_values = [
        randomized_stats[class_id]["red_percentage"]
        for class_id, _ in classes_sorted
    ]

    x = np.arange(
        len(labels)
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        x,
        corr_values,
        marker="o",
        label="Correlated condition: correlated-color %"
    )

    plt.plot(
        x,
        random_red_values,
        marker="s",
        label="Randomized condition: red %"
    )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1,
        label="50% reference"
    )

    plt.xticks(
        x,
        labels
    )

    plt.xlabel(
        "Class"
    )

    plt.ylabel(
        "Percentage"
    )

    plt.title(
        "Color Assignment Comparison: Correlated vs Randomized"
    )

    plt.ylim(
        0,
        100
    )

    plt.legend()

    plt.tight_layout()

    output = (
        REPORT_DIR
        / "correlated_vs_randomized.png"
    )

    plt.savefig(
        output,
        dpi=200
    )

    plt.close()


# ============================================================
# SAVE EXPERIMENT METADATA
# ============================================================

def save_experiment_metadata(
    classes
):
    """
    Record the exact design of Template 1.
    """

    metadata = {
        "template": "COLOR_BIAS",
        "source_dataset_directory": str(SOURCE_DIR),
        "image_resolution": [224, 224],
        "image_channels": 3,
        "original_weight": 0.70,
        "color_overlay_weight": 0.30,
        "correlation_target": 0.90,
        "mismatch_target": 0.10,
        "group_a": {
            "classes": [0, 1, 2, 3, 4],
            "correlated_color": "red",
            "mismatch_color": "blue"
        },
        "group_b": {
            "classes": [5, 6, 7, 8, 9],
            "correlated_color": "blue",
            "mismatch_color": "red"
        },
        "evaluation_modes": [
            "correlated",
            "randomized",
            "reversed",
            "grayscale"
        ],
        "random_seed": RANDOM_SEED,
        "classes": classes
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_DIR / "experiment_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )


# ============================================================
# REPORT
# ============================================================

def create_report(
    classes,
    correlated_audit,
    integrity_audit
):
    """
    Create a report using the actual generated counts.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    lines = []

    lines.append(
        "# Template 1 — COLOR Bias"
    )

    lines.append("")

    lines.append(
        "## 1. Objective"
    )

    lines.append(
        "Introduce a label-correlated color factor while "
        "retaining the original image content."
    )

    lines.append("")

    lines.append(
        "## 2. Image Standardization"
    )

    lines.append(
        "- Resolution: 224 × 224 pixels"
    )

    lines.append(
        "- Output channels: RGB"
    )

    lines.append(
        "- Original dataset directory: `archive/`"
    )

    lines.append(
        "- Original files are not modified."
    )

    lines.append("")

    lines.append(
        "## 3. Color Transformation"
    )

    lines.append(
        "The generated color-biased image uses:"
    )

    lines.append("")

    lines.append(
        "**Output = 0.70 × Original + 0.30 × Color Overlay**"
    )

    lines.append("")

    lines.append(
        "Two color overlays are used:"
    )

    lines.append(
        "- Warm / Red"
    )

    lines.append(
        "- Cool / Blue"
    )

    lines.append("")

    lines.append(
        "## 4. Correlation Rule"
    )

    lines.append(
        "### Group A — Classes 0–4"
    )

    lines.append(
        "90% correlated with red and 10% mismatched with blue."
    )

    lines.append("")

    lines.append(
        "### Group B — Classes 5–9"
    )

    lines.append(
        "90% correlated with blue and 10% mismatched with red."
    )

    lines.append("")

    lines.append(
        "## 5. Frozen Classes"
    )

    for item in classes:

        group = (
            "A"
            if int(item["index"]) < 5
            else "B"
        )

        lines.append(
            f"- Class {item['index']}: "
            f"{item.get('name', item['id'])} "
            f"(Group {group})"
        )

    lines.append("")

    lines.append(
        "## 6. Evaluation Modes"
    )

    lines.append(
        "Four modes were generated:"
    )

    lines.append(
        "1. Correlated"
    )

    lines.append(
        "2. Randomized"
    )

    lines.append(
        "3. Reversed Tint"
    )

    lines.append(
        "4. Grayscale"
    )

    lines.append("")

    lines.append(
        "### Correlated"
    )

    lines.append(
        f"- Total images: {correlated_audit['total_images']}"
    )

    lines.append(
        f"- Correlated images: "
        f"{correlated_audit['correlated_images']}"
    )

    lines.append(
        f"- Mismatch images: "
        f"{correlated_audit['mismatch_images']}"
    )

    lines.append(
        f"- Actual correlated percentage: "
        f"{correlated_audit['correlated_percentage']:.2f}%"
    )

    lines.append(
        f"- Actual mismatch percentage: "
        f"{correlated_audit['mismatch_percentage']:.2f}%"
    )

    lines.append("")

    lines.append(
        "### Randomized"
    )

    lines.append(
        "The color assignments are shuffled across all selected "
        "images while preserving the total red/blue assignment "
        "counts from the correlated construction."
    )

    lines.append("")

    lines.append(
        "### Reversed Tint"
    )

    lines.append(
        "Classes 0–4 receive blue and classes 5–9 receive red."
    )

    lines.append("")

    lines.append(
        "### Grayscale"
    )

    lines.append(
        "Images are converted to grayscale and retained as "
        "three-channel RGB output."
    )

    lines.append("")

    lines.append(
        "## 7. Integrity Verification"
    )

    for mode, result in integrity_audit.items():

        lines.append(
            f"### {mode}"
        )

        lines.append(
            f"- Files checked: {result['files_checked']}"
        )

        lines.append(
            f"- Bad/unreadable files: {result['bad_files']}"
        )

        lines.append(
            f"- Wrong size: {result['wrong_size']}"
        )

        lines.append(
            f"- Wrong channel mode: {result['wrong_mode']}"
        )

    lines.append("")

    lines.append(
        "## 8. Visual Verification"
    )

    lines.append(
        "Visual comparison panels are stored in:"
    )

    lines.append(
        "`reports/color_bias/visual_samples/`"
    )

    lines.append("")

    lines.append(
        "Each panel compares:"
    )

    lines.append(
        "- Original"
    )

    lines.append(
        "- Red"
    )

    lines.append(
        "- Blue"
    )

    lines.append(
        "- Class-correlated color"
    )

    lines.append(
        "- Reversed color"
    )

    lines.append(
        "- Grayscale"
    )

    lines.append("")

    lines.append(
        "## 9. Numerical and Visual Artifacts"
    )

    lines.append(
        "- `color_bias_dataset/correlated/`"
    )

    lines.append(
        "- `color_bias_dataset/randomized/`"
    )

    lines.append(
        "- `color_bias_dataset/reversed/`"
    )

    lines.append(
        "- `color_bias_dataset/grayscale/`"
    )

    lines.append(
        "- `color_bias_dataset/correlated_audit.json`"
    )

    lines.append(
        "- `color_bias_dataset/image_integrity_audit.json`"
    )

    lines.append(
        "- `reports/color_bias/correlation_distribution.png`"
    )

    lines.append(
        "- `reports/color_bias/correlated_vs_randomized.png`"
    )

    lines.append(
        "- `reports/color_bias/visual_samples/`"
    )

    lines.append("")

    lines.append(
        "## 10. Model Training"
    )

    lines.append(
        "No model training is performed in this stage."
    )

    lines.append(
        "The generated datasets are prepared for the later "
        "model-training and evaluation stage."
    )

    report_file = (
        REPORT_DIR
        / "Template_1_Color_Bias_Report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )

    return report_file


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print("TEMPLATE 1 — COLOR BIAS")
    print("=" * 75)

    print(
        "\nThis stage does dataset generation, audit, "
        "and visual verification only."
    )

    print(
        "\nOriginal dataset:"
    )
    print(SOURCE_DIR)

    print(
        "\nFrozen metadata:"
    )
    print(METADATA_FILE)

    # --------------------------------------------------------
    # Load frozen classes
    # --------------------------------------------------------

    classes = load_frozen_classes()

    print("\nFrozen classes:")

    for item in classes:

        group = (
            "A"
            if int(item["index"]) < 5
            else "B"
        )

        print(
            f"  {item['index']} | "
            f"{item['id']} | "
            f"Group {group}"
        )

    # --------------------------------------------------------
    # Generate correlated mode
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("MODE 1 — CORRELATED")
    print("=" * 75)

    correlated_statistics = (
        generate_correlated(
            classes
        )
    )

    # --------------------------------------------------------
    # Audit correlated mode
    # --------------------------------------------------------

    correlated_audit = audit_correlated(
        correlated_statistics
    )

    print("\nCorrelated audit:")
    print(
        f"  Total      : "
        f"{correlated_audit['total_images']}"
    )
    print(
        f"  Correlated : "
        f"{correlated_audit['correlated_images']} "
        f"({correlated_audit['correlated_percentage']:.2f}%)"
    )
    print(
        f"  Mismatch   : "
        f"{correlated_audit['mismatch_images']} "
        f"({correlated_audit['mismatch_percentage']:.2f}%)"
    )

    # --------------------------------------------------------
    # Generate randomized mode
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("MODE 2 — RANDOMIZED")
    print("=" * 75)

    randomized_statistics = (
        generate_randomized(
            classes,
            correlated_statistics
        )
    )

    # --------------------------------------------------------
    # Generate reversed mode
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("MODE 3 — REVERSED TINT")
    print("=" * 75)

    reversed_statistics = (
        generate_reversed(
            classes
        )
    )

    # --------------------------------------------------------
    # Generate grayscale mode
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("MODE 4 — GRAYSCALE")
    print("=" * 75)

    grayscale_statistics = (
        generate_grayscale(
            classes
        )
    )

    # --------------------------------------------------------
    # Save experiment metadata
    # --------------------------------------------------------

    save_experiment_metadata(
        classes
    )

    # --------------------------------------------------------
    # Verify generated files
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("IMAGE INTEGRITY VERIFICATION")
    print("=" * 75)

    integrity_audit = (
        verify_generated_images(
            classes
        )
    )

    for mode, result in integrity_audit.items():

        print(
            f"{mode:11s} | "
            f"checked={result['files_checked']} | "
            f"bad={result['bad_files']} | "
            f"wrong_size={result['wrong_size']} | "
            f"wrong_mode={result['wrong_mode']}"
        )

    # --------------------------------------------------------
    # Visual comparisons
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("VISUAL VERIFICATION")
    print("=" * 75)

    create_visual_comparison(
        classes
    )

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("GRAPH GENERATION")
    print("=" * 75)

    create_correlation_graph(
        correlated_statistics
    )

    create_mode_summary_graph(
        correlated_statistics,
        randomized_statistics
    )

    # --------------------------------------------------------
    # Research report
    # --------------------------------------------------------

    report_file = create_report(
        classes,
        correlated_audit,
        integrity_audit
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("TEMPLATE 1 COMPLETE")
    print("=" * 75)

    print(
        "\nGenerated datasets:"
    )

    for mode in [
        "correlated",
        "randomized",
        "reversed",
        "grayscale"
    ]:
        print(
            f"  {OUTPUT_DIR / mode}"
        )

    print(
        "\nReports:"
    )

    print(
        f"  {REPORT_DIR}"
    )

    print(
        f"\nResearch report:\n  {report_file}"
    )

    print(
        "\nOriginal archive was not modified."
    )

    print(
        "Model training was NOT performed."
    )

    print(
        "\nTemplate 1 Color Bias is now ready "
        "for the later training/evaluation stage."
    )


if __name__ == "__main__":
    main()
