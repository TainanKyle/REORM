import os
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPONENT_DIR = REPO_ROOT / "component"
GSA_DIR = COMPONENT_DIR / "Grounded-Segment-Anything"


@contextmanager
def temporary_cwd(path: str | Path):
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def dilate_images(input_dir, output_dir, kernel_size, iterations):
    """Dilate images in the specified directory."""
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_image = cv2.dilate(image, kernel, iterations=iterations)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, dilated_image)