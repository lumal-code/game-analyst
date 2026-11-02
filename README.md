# game-analyst
Remember to set up venv.

## YOLOv7 annotation baseline

`src/training/testing.py` evaluates the hand-labeled Game 1 and Game 2 frames
with the official pretrained YOLOv7-tiny COCO model. It uses a 0.01 detection
confidence threshold and counts matches at 0.5 box IoU. It prints per-game
precision and recall and writes `media/yolov7_annotation_results.json`.

The model's `person` class is compared with `Player`, and `sports ball` with
`Ball`. Referee is excluded because the pretrained COCO model has no distinct
referee class. A person detection overlapping a referee annotation is excluded
from player false positives. These are baseline detection counts, not mAP.

```sh
source .venv/bin/activate
pip install -r requirements.txt
git clone --depth 1 https://github.com/WongKinYiu/yolov7.git media/yolov7
curl -L --fail --output media/yolov7-tiny.pt https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt
python src/training/testing.py
```

For a quick smoke test, add `--limit 1`. The source and weights are downloaded
under the ignored `media/` directory. The evaluator uses only frames with a
matching annotation file.

## Fine-tuning on CPU

The training wrapper uses YOLOv7-tiny, Game 1 for training, and Game 2 for
validation. It defaults to 50 epochs, 640-pixel images, and batch size 2.
It maps the pretrained COCO person and sports-ball detector weights into the
three-class head, freezes the backbone, and uses a lower learning rate. It
keeps the pretrained anchors during training.
Exact duplicate annotation rows are removed in generated dataset copies so
YOLOv7 does not skip those frames; the original annotations remain untouched.

```sh
python training/player_detector/train.py --evaluate
```

This creates a uniquely named run, evaluates its `best.pt` on Game 2 at
confidence 0.01, and writes `media/yolov7_finetuned_results.json`.

To reverse the split, first run
`python training/player_detector/prepare_dataset.py --train-game 2`, then
`python training/player_detector/train.py --train-game 2 --evaluate`. Training outputs
are stored under the ignored `media/yolov7_training/` directory.

# TODO
* Evaluate YOLOv7 using 30 image dataset
* Develop custom models to find 
  * Field Boundaries
  * Soccer ball 
  * Members of each team (via jersey color so programmatically determine the two different teams)
* Somehow track each player on the Field
* Programmatically determine when the ball gets close to a chosen player
* Take all these clips
