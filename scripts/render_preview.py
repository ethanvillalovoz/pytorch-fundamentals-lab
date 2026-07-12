"""Render the animated README preview from committed reference metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as font_manager
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1200, 675
BACKGROUND = "#f4f4f2"
INK = "#171717"
MUTED = "#6b6b67"
BLUE = "#2563eb"
GREEN = "#138a62"
CORAL = "#d95d45"
FONT_PATH = font_manager.findfont("DejaVu Sans")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def base_scene(kicker: str, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((72, 58), kicker.upper(), fill=MUTED, font=font(20))
    draw.text((72, 94), title, fill=INK, font=font(48))
    draw.text((72, 158), subtitle, fill=MUTED, font=font(24))
    return image, draw


def rounded_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=8, fill="#ffffff", outline="#d8d8d4", width=2)


def overview_scene() -> Image.Image:
    image, draw = base_scene(
        "PyTorch Fundamentals Lab",
        "From tensor shape to measured behavior",
        "Five notebooks. One tested implementation path.",
    )
    lessons = (
        ("01", "Tensors", "shape / dtype / device", BLUE),
        ("02", "Operations", "reshape / broadcast", GREEN),
        ("03", "Autograd", "loss / backward / step", CORAL),
        ("04", "Iris MLP", "deterministic reference", BLUE),
        ("05", "MNIST CNN", "sample-weighted metrics", GREEN),
    )
    for index, (number, title, detail, color) in enumerate(lessons):
        y = 238 + index * 73
        draw.ellipse((76, y, 110, y + 34), fill=color)
        draw.text((84, y + 5), number, fill="#ffffff", font=font(15))
        draw.text((132, y - 3), title, fill=INK, font=font(25))
        draw.text((360, y + 1), detail, fill=MUTED, font=font(21))
    rounded_panel(draw, (758, 246, 1125, 546))
    draw.text((800, 282), "x", fill=MUTED, font=font(20))
    matrix = ((0.2, 1.4, 3.1, 2.0), (1.7, 0.1, 2.6, 3.3), (2.2, 1.1, 0.4, 2.8))
    colors = ("#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa")
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            x = 800 + column * 70
            y = 326 + row * 62
            draw.rounded_rectangle((x, y, x + 54, y + 46), radius=5, fill=colors[column])
            draw.text((x + 10, y + 11), f"{value:.1f}", fill=INK, font=font(17))
    draw.text((800, 520), "shape = (3, 4)", fill=MUTED, font=font(18))
    return image


def curve_scene(metrics: dict[str, object], progress: float) -> Image.Image:
    test = metrics["test"]
    assert isinstance(test, dict)
    accuracy = float(test["accuracy"])
    raw_losses = metrics["train_loss"]
    assert isinstance(raw_losses, list)
    losses = [float(value) for value in raw_losses]
    image, draw = base_scene(
        "04 / Iris MLP",
        "A result you can reproduce",
        "Seeded split, train-only scaling, serialized predictions.",
    )
    rounded_panel(draw, (72, 232, 830, 592))
    left, top, right, bottom = 125, 278, 786, 535
    draw.line((left, bottom, right, bottom), fill="#b8b8b2", width=2)
    draw.line((left, top, left, bottom), fill="#b8b8b2", width=2)
    count = max(2, int(len(losses) * progress))
    shown = losses[:count]
    maximum = max(losses)
    points = []
    for index, value in enumerate(shown):
        x = left + (right - left) * index / (len(losses) - 1)
        y = bottom - (bottom - top) * (value / maximum)
        points.append((x, y))
    draw.line(points, fill=BLUE, width=5, joint="curve")
    draw.text((125, 548), "epoch", fill=MUTED, font=font(18))
    draw.text((90, 270), "loss", fill=MUTED, font=font(18))
    rounded_panel(draw, (868, 232, 1126, 592))
    draw.text((902, 278), "held-out", fill=MUTED, font=font(18))
    draw.text((902, 312), f"{accuracy:.1%}", fill=GREEN, font=font(49))
    draw.text((902, 372), "accuracy", fill=INK, font=font(21))
    draw.text((902, 447), "30", fill=INK, font=font(38))
    draw.text((902, 493), "test examples", fill=MUTED, font=font(18))
    return image


def confusion_scene(metrics: dict[str, object]) -> Image.Image:
    test = metrics["test"]
    dataset = metrics["dataset"]
    assert isinstance(test, dict) and isinstance(dataset, dict)
    matrix = test["confusion_matrix"]
    names = dataset["class_names"]
    assert isinstance(matrix, list) and isinstance(names, list)
    image, draw = base_scene(
        "04 / Iris MLP",
        "Inspect the errors, not just the score",
        "Every row is a true class; every column is a prediction.",
    )
    rounded_panel(draw, (72, 230, 1127, 594))
    start_x, start_y, size = 425, 292, 86
    maximum = max(max(row) for row in matrix)
    for index, name in enumerate(names):
        draw.text(
            (start_x - 166, start_y + index * size + 29), str(name), fill=MUTED, font=font(18)
        )
        draw.text(
            (start_x + index * size + 4, start_y - 38), str(name)[:4], fill=MUTED, font=font(17)
        )
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            intensity = value / maximum if maximum else 0
            fill = "#2563eb" if intensity > 0.7 else "#bfdbfe" if intensity > 0 else "#eeeeeb"
            x = start_x + column_index * size
            y = start_y + row_index * size
            draw.rounded_rectangle((x, y, x + 68, y + 68), radius=6, fill=fill)
            text_fill = "#ffffff" if intensity > 0.7 else INK
            draw.text((x + 24, y + 18), str(value), fill=text_fill, font=font(25))
    draw.text((786, 326), "The reference run commits", fill=INK, font=font(24))
    draw.text((786, 367), "metrics.json", fill=BLUE, font=font(24))
    draw.text((786, 408), "predictions.csv", fill=GREEN, font=font(24))
    draw.text((786, 466), "No mystery checkpoint.", fill=MUTED, font=font(19))
    return image


def cnn_scene() -> Image.Image:
    image, draw = base_scene(
        "05 / MNIST CNN",
        "One implementation, bounded when needed",
        "Use batch limits for smoke tests; remove them for a full run.",
    )
    stages = (
        ("1 x 28 x 28", "input", BLUE),
        ("8 x 14 x 14", "conv + pool", GREEN),
        ("16 x 7 x 7", "conv + pool", CORAL),
        ("64", "linear", BLUE),
        ("10 logits", "output", GREEN),
    )
    for index, (shape, label, color) in enumerate(stages):
        x = 74 + index * 220
        rounded_panel(draw, (x, 280, x + 174, 444))
        draw.rounded_rectangle((x + 21, 307, x + 153, 353), radius=6, fill=color)
        draw.text((x + 35, 318), shape, fill="#ffffff", font=font(18))
        draw.text((x + 27, 384), label, fill=INK, font=font(20))
        if index < len(stages) - 1:
            draw.line((x + 175, 362, x + 211, 362), fill=MUTED, width=3)
            draw.polygon(((x + 211, 362), (x + 201, 355), (x + 201, 369)), fill=MUTED)
    draw.text((74, 503), "CrossEntropyLoss expects logits.", fill=INK, font=font(22))
    draw.text(
        (74, 542), "Metrics are weighted by examples, not batches.", fill=MUTED, font=font(22)
    )
    return image


def crossfade(scenes: list[Image.Image]) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for index, scene in enumerate(scenes):
        frames.extend([scene] * 5)
        next_scene = scenes[(index + 1) % len(scenes)]
        for step in range(1, 7):
            frames.append(Image.blend(scene, next_scene, step / 7))
    return frames


def main() -> None:
    metrics_path = ROOT / "artifacts" / "iris-reference" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    scenes = [
        overview_scene(),
        curve_scene(metrics, 0.25),
        curve_scene(metrics, 0.6),
        curve_scene(metrics, 1.0),
        confusion_scene(metrics),
        cnn_scene(),
    ]
    frames = crossfade(scenes)
    output = ROOT / "docs" / "media" / "lab-preview.webp"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        quality=82,
        method=6,
    )


if __name__ == "__main__":
    main()
