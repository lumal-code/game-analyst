"""Pair the hand-labeled game frames for YOLO object detection.

Run from anywhere with: python training/player_detector/prepare_dataset.py
The generated dataset lives under media/ and is ignored by Git.
"""

from pathlib import Path
import argparse
import shutil


ROOT = Path(__file__).resolve().parents[2]
CLASSES = ("Player", "Ball", "Referee")


def prepare(train_game: int) -> Path:
    output = ROOT / "media" / f"yolo_player_detector_train_game_{train_game}"
    if output.exists():
        raise FileExistsError(f"{output} already exists; move it aside to rebuild")

    sources = {}
    for game in (1, 2):
        label_dir = ROOT / "training" / f"Game {game}" / "obj_train_data"
        image_dir = ROOT / "media" / "frames" / f"games_{game}"
        names = tuple((ROOT / "training" / f"Game {game}" / "obj.names").read_text().splitlines())
        if names != CLASSES:
            raise ValueError(f"Unexpected class order for Game {game}: {names}")
        labels = sorted(label_dir.glob("frame_*.txt"))
        if not labels:
            raise ValueError(f"No labels found for Game {game}")
        for label in labels:
            if not (image_dir / f"{label.stem}.jpg").is_file():
                raise FileNotFoundError(f"Missing image for {label}")
        sources[game] = (image_dir, labels)

    for split, game in (("train", train_game), ("val", 3 - train_game)):
        image_dir, labels = sources[game]
        out_images = output / "images" / split
        out_labels = output / "labels" / split
        out_images.mkdir(parents=True)
        out_labels.mkdir(parents=True)
        for label in labels:
            shutil.copy2(image_dir / f"{label.stem}.jpg", out_images / f"{label.stem}.jpg")
            # Some exports repeat identical rows. YOLOv7 rejects the whole
            # frame when that happens, so deduplicate only the dataset copy.
            rows = list(dict.fromkeys(label.read_text().splitlines()))
            (out_labels / label.name).write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"{split}: Game {game}, {len(labels)} labeled frames")

    (output / "data.yaml").write_text(
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASSES))
    )
    return output / "data.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-game", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    print(f"Dataset config: {prepare(args.train_game)}")
