import os
from pathlib import Path
import json
import argparse

from src.inference.multi_llm.utils import dilate_images

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run image object removal with self-correction pipeline.')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to the folder containing input images')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to the JSON file with instructions')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output folder')

    args = parser.parse_args()

    image_folder = str(Path(args.image_folder).expanduser().resolve())
    json_path = str(Path(args.json_path).expanduser().resolve())
    output_folder = str(Path(args.output_folder).expanduser().resolve())
    SEED = 44
    
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    
    #
    # run SmartEraser and get the edited images
    #

    masks_dir = os.path.join(output_folder, "masks/")
    dilate_images(masks_dir, masks_dir, kernel_size=5, iterations=5)

    results_path = os.path.join(output_folder, "results/")

    from src.tools.run_objectclear import inference_objectclear_dataset
    inference_objectclear_dataset(image_folder, masks_dir, results_path)