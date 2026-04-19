import os
import json
import traceback
import argparse

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

    image_folder = args.image_folder
    json_path = args.json_path
    output_folder = args.output_folder
    SEED = 44
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "results"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)
    
    #
    # Run Multi-LLM to get the target objects
    #
    
    from src.inference.multi_llm.hybrid_object_detector import HybridObjectDetector, set_seed

    set_seed(SEED)
    HybridAssistant = HybridObjectDetector(
        vlm_model_id="llava-hf/llava-v1.6-vicuna-13b-hf",
        llm_model_id="meta-llama/Llama-3.1-8B-Instruct",
        precision_vlm="4bit",
        precision_llm="8bit"
    )

    output_json_path = os.path.join(output_folder, "records.json")
    try:
        llm_results = HybridAssistant.process_dataset(
            entries=entries,
            image_folder=image_folder,
            output_json_path=output_json_path
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
