"""Fine-tune official YOLOv7-tiny on the annotated games using the CPU.

Run: .venv/bin/python training/player_detector/train.py
"""

import argparse
import os
import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "media" / "yolov7"
WEIGHTS = ROOT / "media" / "yolov7-tiny.pt"
DATASET = ROOT / "media" / "yolo_player_detector_train_game_1"


def mapped_head_weights(destination: Path):
    """Keep COCO box/person/ball weights when changing the head to 3 classes."""
    from models.yolo import Model

    source = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["model"].float()
    target = Model(source.yaml, ch=3, nc=3)
    source_state = source.state_dict()
    target_state = target.state_dict()
    target_state.update({
        key: value for key, value in source_state.items()
        if key in target_state and value.shape == target_state[key].shape
    })
    target.load_state_dict(target_state)
    src_head, dst_head = source.model[-1], target.model[-1]
    indices = [0, 1, 2, 3, 4, 5, 37, 5]  # xywh, objectness, person, sports ball, person
    with torch.no_grad():
        for src_conv, dst_conv in zip(src_head.m, dst_head.m):
            for field in ("weight", "bias"):
                src = getattr(src_conv, field).view(src_head.na, src_head.no, *getattr(src_conv, field).shape[1:])
                dst = getattr(dst_conv, field).view(dst_head.na, dst_head.no, *getattr(dst_conv, field).shape[1:])
                dst.copy_(src[:, indices])
            # Referees begin as person-like, but less likely than players.
            dst_conv.bias.view(dst_head.na, dst_head.no)[:, 7] -= 2.0
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": target, "optimizer": None, "epoch": -1}, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--train-game", type=int, choices=(1, 2), default=1)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate this run's best checkpoint on the other game")
    args = parser.parse_args()
    if not REPO.is_dir() or not WEIGHTS.is_file():
        parser.error("Download media/yolov7 and media/yolov7-tiny.pt first; see README.md")
    dataset = DATASET.with_name(f"yolo_player_detector_train_game_{args.train_game}")
    if not (dataset / "data.yaml").is_file():
        parser.error(f"Missing {dataset / 'data.yaml'}; run prepare_dataset.py first")

    # Refresh existing generated copies too, without touching source labels.
    for split, game in (("train", args.train_game), ("val", 3 - args.train_game)):
        source = ROOT / "training" / f"Game {game}" / "obj_train_data"
        target = dataset / "labels" / split
        changed = False
        for label in sorted(source.glob("frame_*.txt")):
            rows = list(dict.fromkeys(label.read_text().splitlines()))
            content = "\n".join(rows) + ("\n" if rows else "")
            output = target / label.name
            if output.read_text() != content:
                output.write_text(content)
                changed = True
        if changed:
            (dataset / "labels" / f"{split}.cache").unlink(missing_ok=True)

    # The original YOLOv7 trainer expects absolute train/val paths and an `nc` key.
    data_file = dataset / "yolov7.yaml"
    data_file.write_text(
        "nc: 3\n"
        "names: [Player, Ball, Referee]\n"
        f"train: {dataset / 'images/train'}\n"
        f"val: {dataset / 'images/val'}\n"
    )
    hyp = yaml.safe_load((REPO / "data/hyp.scratch.tiny.yaml").read_text())
    hyp["lr0"] = 0.001
    hyp["warmup_bias_lr"] = 0.001
    hyp_file = dataset / "hyp.cpu.finetune.yaml"
    hyp_file.write_text(yaml.safe_dump(hyp, sort_keys=False))

    # YOLOv7 predates newer PyTorch/NumPy APIs. These compatibility settings
    # apply only to this process, using the trusted official weight file.
    original_load = torch.load

    def load_checkpoint(*load_args, **load_kwargs):
        load_kwargs.setdefault("weights_only", False)
        return original_load(*load_args, **load_kwargs)

    torch.load = load_checkpoint
    if not hasattr(np, "int"):
        np.int = int
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid

    sys.path.insert(0, str(REPO))
    mapped_weights = mapped_head_weights(dataset / "yolov7_tiny_mapped_head.pt")
    run_name = f"train_game_{args.train_game}_cpu_{args.image_size}_{args.epochs}ep_head"
    if args.evaluate:
        run_name += "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = ROOT / "media/yolov7_training" / run_name
    sys.argv = [
        str(REPO / "train.py"),
        "--weights", str(mapped_weights),
        "--data", str(data_file),
        "--hyp", str(hyp_file),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--img-size", str(args.image_size), str(args.image_size),
        "--device", "cpu",
        "--workers", "0",
        "--freeze", "77",
        "--noautoanchor",
        "--project", str(ROOT / "media/yolov7_training"),
        "--name", run_name,
    ]
    os.chdir(REPO)
    runpy.run_path(str(REPO / "train.py"), run_name="__main__")
    if args.evaluate:
        best = run_dir / "weights/best.pt"
        if not best.is_file():
            raise FileNotFoundError(f"Training finished without a best checkpoint: {best}")
        print(f"Evaluating {best} on Game {3 - args.train_game}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "src/training/testing.py"),
             "--weights", str(best), "--games", str(3 - args.train_game),
             "--image-size", str(args.image_size)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
