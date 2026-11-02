# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: .venv (3.12.0)
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append('..')

# %%
import cv2
import numpy as np

def make_collage(batch):
    while len(batch) != 9:
        batch.append(np.zeros_like(batch[0]))
    row1 = cv2.hconcat(batch[:3])
    row2 = cv2.hconcat(batch[3:6])
    row3 = cv2.hconcat(batch[6:])

    return cv2.vconcat([row1, row2, row3])


# %%
from src.video.extract_frames import extract_frames
from collections import deque

deque(
    extract_frames(
        '../media/videos/first-half.mp4',
        interval=15,
        output_folder='../media/frames/first-half/',
        save=True
    ),
    maxlen=0
)
# x = 0
# for img in batches:
#     cv2.imwrite(f"../media/frames/first-half/image{x}.png", img[3])
#     x += 1

# %%
import cv2

image = cv2.imread("../media/frames/first-half/image0.png")

# Crop using pixel coordinates
cropped = image[400:900, 600:1200]

cv2.imwrite("../media/frames/first-half/cropped.jpg", cropped)

# %% [markdown]
# Extract 30 frames from each video link
# ffmpeg -i "https://vj.hudl.com/92125/521561/14e/6abe7879166b7faab44eb14e/video.ondemand.m3u8?v=90A2139DD31FDF08" \
#   -vf "fps=1/180" \
#   -frames:v 30 \
#   frame_%03d.jpg

# %%
import subprocess

urls = [
    'https://vj.hudl.com/92125/521561/dfb/6aabffc49b130ae65ac5adfb/video.ondemand.m3u8?v=F05D8BA2D214DF08',
    'https://vj.hudl.com/92125/521561/e81/6ac8649f9528cb2650defe81/video.ondemand.m3u8?v=2098F740BB25DF08',
    'https://vj.hudl.com/92125/521561/14e/6abe7879166b7faab44eb14e/video.ondemand.m3u8?v=90A2139DD31FDF08',
    'https://vj.hudl.com/92125/521561/3ae/6aafe3e83b1e0a21ec0123ae/video.ondemand.m3u8?v=808CCF133E17DF08',
    'https://vj.hudl.com/92125/521561/fd3/6ab92b656b39d59fc14b7fd3/video.ondemand.m3u8?v=103DA2B7AE1CDF08'
]

# %%
import subprocess
from pathlib import Path

for i, link in enumerate(urls):
    output_dir = Path(f"../media/frames/game_{i+1}")
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        "ffmpeg",
        "-i", link,
        "-vf", "fps=1/180",
        "-frames:v", "30",
        "-q:v", "2",
        str(output_dir / "frame_%03d.jpg")
    ], check=True)

# %%
from pathlib import Path
import shutil

src_root = Path("../media/frames")
dst_root = src_root / "dataset"
dst_root.mkdir(exist_ok=True)

for game_dir in src_root.glob("game_*"):
    if game_dir.is_dir():
        game_name = game_dir.name   # like game_1
        for img in game_dir.glob("frame_*.jpg"):
            new_name = f"{game_name}_{img.name}"   # game_1_frame_001.jpg
            shutil.move(str(img), str(dst_root / new_name))
