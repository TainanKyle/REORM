import os
import argparse
import json

from src.inference.api_llm.api_llm_client import ResultStorage
from src.inference.api_llm.utils import dilate_images


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
    image_folder = args.image_folder
    json_path = args.json_path
    output_folder = args.output_folder

    # Get API key from command line argument or environment variable
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("API key must be provided either as a command line argument or as the OPENAI_API_KEY environment variable")
    
    ResultStorage.load_from_json(os.path.join(output_folder, "records.json"))

    
    #
    # run diffusion model and get the edited images
    #
    print('--'*20)
    print('Starting Diffusion Model Image Editing')
    print('--'*20)

    masks_dir = os.path.join(output_folder, "masks/")
    dilate_images(masks_dir, masks_dir, kernel_size=5, iterations=3)

    initial_results_folder = os.path.join(output_folder, "correction/initial_results/")
    os.makedirs(initial_results_folder, exist_ok=True)

    from src.tools.run_objectclear import inference_objectclear_dataset
    inference_objectclear_dataset(image_folder, masks_dir, initial_results_folder)
