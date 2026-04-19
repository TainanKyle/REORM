import os
import sys
import json
import argparse

from src.inference.api_llm.api_llm_client import api_llm_examiner_dataset, ResultStorage
from src.inference.api_llm.utils import GSA_DIR, dilate_images, temporary_cwd
from src.tools.attentive_eraser import AttentiveEraser


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
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)


    #
    # run GPT to self-correct the images
    #
    correction_folder = os.path.join(output_folder, "correction/")
    initial_results_images = os.path.join(correction_folder, "initial_results/")
    
    print('--'*20)
    print('Starting LLM Examination')
    print('--'*20)
    
    api_llm_examiner_dataset(initial_results_images, entries, ResultStorage.descriptions, api=api_key, detail="auto")
    ResultStorage.save_to_json(os.path.join(output_folder, "records.json"))

    print('--'*20)
    print('Starting Mask Generation for Self-Correction')
    print('--'*20)
    with temporary_cwd(GSA_DIR):
        if str(GSA_DIR) not in sys.path:    
            sys.path.insert(0, str(GSA_DIR)) 

        from src.tools.run_grounded_sam import inference_sam_dataset


        inference_sam_dataset(initial_results_images, correction_folder, entries, ResultStorage.objects_to_correct, box_threshold=0.45)


    #
    # run Attentive Eraser to correct the images
    #
    print('--'*20)
    print('Starting Attentive Eraser Correction')
    print('--'*20)

    masks_dir = os.path.join(correction_folder, "masks/")

    # If no objects need correction, Grounded-SAM may produce no mask folder.
    # In that case, skip dilation and let correct_dataset copy initial results.
    has_masks = os.path.isdir(masks_dir) and any(
        name.lower().endswith((".png", ".jpg", ".jpeg")) for name in os.listdir(masks_dir)
    )
    if has_masks:
        dilate_images(masks_dir, masks_dir, kernel_size=5, iterations=3)
    else:
        print("No correction masks generated. Skipping mask dilation.")
    
    final_folder = os.path.join(output_folder, "results/") 
    attentive_eraser = AttentiveEraser()
    attentive_eraser.correct_dataset(entries, initial_results_images, masks_dir, final_folder)