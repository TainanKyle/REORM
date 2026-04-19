import os
import sys
from pathlib import Path
import json
import argparse
from src.inference.multi_llm.utils import (
    GSA_DIR,
    format_target_objects,
    load_results_from_json,
    temporary_cwd,
)



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

    llm_results = load_results_from_json(os.path.join(output_folder, "records.json"))

    target_objects = format_target_objects(llm_results, exclusion_keywords={"wall", "road", "environment"})


    #
    # run Grounded-SAM in its directory
    #
    with temporary_cwd(GSA_DIR):
        if str(GSA_DIR) not in sys.path:    
            sys.path.insert(0, str(GSA_DIR)) 

        from src.tools.run_grounded_sam import inference_sam_dataset

        inference_sam_dataset(image_folder, output_folder, entries, target_objects, box_threshold=0.32)
