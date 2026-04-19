import os
import sys
from pathlib import Path
import json
import argparse

from src.inference.api_llm.api_llm_client import api_llm_detector_dataset, ResultStorage
from src.inference.api_llm.utils import GSA_DIR, temporary_cwd


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run image object removal with self-correction pipeline.')
    parser.add_argument('--image_folder', type=str, required=True,
                        help='Path to the folder containing input images')
    parser.add_argument('--json_path', type=str, required=True,
                        help='Path to the JSON file with instructions')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Path to the output folder')
    parser.add_argument('--api_key', type=str,
                        help='OpenAI API key for GPT access (if not provided, will try to read from OPENAI_API_KEY environment variable)')

    args = parser.parse_args()

    # Use the parsed arguments
    image_folder = str(Path(args.image_folder).expanduser().resolve())
    json_path = str(Path(args.json_path).expanduser().resolve())
    output_folder = str(Path(args.output_folder).expanduser().resolve())

    # Get API key from command line argument or environment variable
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided either as a command line argument or as the OPENAI_API_KEY environment variable")
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "correction"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "correction", "initial_results"), exist_ok=True)

    #
    # get the target objects
    #

    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    print('--'*20)
    print('Starting LLM Target Object Analysis')
    print('--'*20)

    api_llm_detector_dataset(image_folder, entries, api=api_key)
    ResultStorage.save_to_json(os.path.join(output_folder, "records.json"))


    #
    # run Grounded-SAM in its directory
    #
    print('--'*20)
    print('Starting Initial Mask Generation')
    print('--'*20)
    with temporary_cwd(GSA_DIR):
        if str(GSA_DIR) not in sys.path:    
            sys.path.insert(0, str(GSA_DIR)) 

        from src.tools.run_grounded_sam import inference_sam_dataset

        inference_sam_dataset(image_folder, output_folder, entries, ResultStorage.initial_targets, box_threshold=0.32)