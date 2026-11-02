"""Evaluate YOLOv7-tiny against the hand-labeled game frames.

Run: .venv/bin/python src/training/testing.py
The default COCO weights score Player (person) and Ball (sports ball).
With --weights pointing to a fine-tuned checkpoint, Referee is scored too.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
YOLO_REPO = ROOT / "media" / "yolov7"
WEIGHTS = ROOT / "media" / "yolov7-tiny.pt"
BASELINE_CLASSES = {0: 0, 32: 1}  # COCO person and sports ball
ALL_CLASSES = {0: "Player", 1: "Ball", 2: "Referee"}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def read_labels(path, width, height):
    boxes = defaultdict(list)
    for line in path.read_text().splitlines():
        cls, xc, yc, w, h = map(float, line.split())
        xc, w = xc * width, w * width
        yc, h = yc * height, h * height
        boxes[int(cls)].append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return boxes


def score(predictions, truth, threshold, classes, ignore_referees):
    """Greedily pair each prediction with at most one same-class annotation."""
    counts = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in classes}
    for cls in classes:
        unmatched = set(range(len(truth[cls])))
        for box, confidence in sorted(predictions[cls], key=lambda item: -item[1]):
            if ignore_referees and cls == 0 and any(iou(box, ref) >= threshold for ref in truth[2]):
                continue  # A COCO person detection on a referee is not a player error.
            best = max(unmatched, key=lambda j: iou(box, truth[cls][j]), default=None)
            if best is not None and iou(box, truth[cls][best]) >= threshold:
                unmatched.remove(best)
                counts[cls]["tp"] += 1
            else:
                counts[cls]["fp"] += 1
        counts[cls]["fn"] = len(unmatched)
    return counts


def summarize(counts, classes):
    result = {}
    for cls, name in classes.items():
        c = counts[cls]
        precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
        result[name] = {**c, "precision": round(precision, 4), "recall": round(recall, 4)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=0.01)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--games", type=int, nargs="+", choices=(1, 2), default=(1, 2))
    parser.add_argument("--limit", type=int, help="Limit labeled frames per game for a quick run")
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="YOLOv7 checkpoint; defaults to pretrained COCO weights")
    args = parser.parse_args()
    weights = args.weights.resolve()
    if not YOLO_REPO.is_dir() or not weights.is_file():
        parser.error("Missing media/yolov7 or the selected weights; see README.md for setup")
    baseline = weights == WEIGHTS.resolve()
    class_map = BASELINE_CLASSES if baseline else {0: 0, 1: 1, 2: 2}
    classes = {label: ALL_CLASSES[label] for label in class_map.values()}

    sys.path.insert(0, str(YOLO_REPO))
    # These pickle weights are loaded only from the official YOLOv7 release.
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model = checkpoint["ema"] if checkpoint.get("ema") else checkpoint["model"]
    model = model.float().eval().autoshape()
    model.conf = args.confidence
    model.classes = list(class_map)

    report = {"weights": str(weights), "confidence": args.confidence, "match_iou": args.match_iou, "image_size": args.image_size, "games": {}}
    for game in args.games:
        label_dir = ROOT / "training" / f"Game {game}" / "obj_train_data"
        image_dir = ROOT / "media" / "frames" / f"games_{game}"
        labels = sorted(label_dir.glob("frame_*.txt"))[: args.limit]
        if not labels:
            parser.error(f"No annotations found in {label_dir}")
        totals = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in classes}
        for label in labels:
            image_path = image_dir / f"{label.stem}.jpg"
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                truth = read_labels(label, *image.size)
                with torch.inference_mode():
                    detections = model(image, size=args.image_size).xyxy[0].cpu().tolist()
            predictions = defaultdict(list)
            for x1, y1, x2, y2, confidence, coco_class in detections:
                predictions[class_map[int(coco_class)]].append(((x1, y1, x2, y2), confidence))
            frame_counts = score(predictions, truth, args.match_iou, classes, baseline)
            for cls in classes:
                for metric in ("tp", "fp", "fn"):
                    totals[cls][metric] += frame_counts[cls][metric]
        report["games"][f"Game {game}"] = {"frames": len(labels), "classes": summarize(totals, classes)}
        print(f"Game {game}: {len(labels)} labeled frames")
        for name, metrics in report["games"][f"Game {game}"]["classes"].items():
            print(f"  {name}: {metrics}")

    output = ROOT / "media" / ("yolov7_annotation_results.json" if baseline else "yolov7_finetuned_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
