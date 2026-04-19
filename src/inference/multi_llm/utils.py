import json
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


def strip_location(text):
    """Removes prepositional phrases describing location from a string."""
    location_prepositions = [
        " on ",
        " in ",
        " at ",
        " under ",
        " behind ",
        " near ",
        " above ",
        " below ",
        " across ",
        " from ",
    ]
    for prep in location_prepositions:
        if prep in text:
            text = text[: text.find(prep)]
    return text.strip()


def format_target_objects(all_results, exclusion_keywords=None):
    """
    Extracts, cleans, and formats target objects for each successful entry.

    Args:
        all_results (list): The list of result dictionaries returned by the detector.
        exclusion_keywords (set[str] | list[str] | None): If provided, drop objects
            containing any of these keywords after location stripping.

    Returns:
        dict: Key is image filename and value is a period-separated object string.
    """
    formatted_data = {}
    invalid_values = {"object", "none"}
    exclusions = {k.lower() for k in (exclusion_keywords or [])}

    for result in all_results or []:
        if result.get("status") != "success":
            continue
        try:
            filename = result["original_entry"]["input"]
            target_objects = result["detection_results"]["target_objects"]

            cleaned_objects = []
            for obj in target_objects:
                stripped = strip_location(str(obj)).strip()
                if not stripped or stripped.lower() in invalid_values:
                    continue
                lowered = stripped.lower()
                if exclusions and any(keyword in lowered for keyword in exclusions):
                    continue
                cleaned_objects.append(stripped)

            formatted_data[filename] = ". ".join(cleaned_objects)
        except (KeyError, TypeError) as e:
            print(f"Skipping a result due to unexpected format: {e}")

    return formatted_data


def load_results_from_json(file_path):
    """Loads processing results from a specified JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None