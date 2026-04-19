import ast
import json
import os
import random
import re

import numpy as np
import requests
import torch
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)


class HybridObjectDetector:
    """
    A hybrid assistant that leverages a Vision-Language Model (LLaVA-NeXT) for
    image analysis and a powerful Large Language Model for all
    subsequent text-based reasoning, parsing, and refinement tasks.
    """

    def __init__(
        self,
        vlm_model_id="llava-hf/llava-v1.6-vicuna-13b-hf",
        llm_model_id="Qwen/Qwen2-7B-Instruct",
        precision_vlm="8bit",
        precision_llm="8bit",
    ):
        """Initializes the assistant by loading both the VLM and LLM."""
        print("Initializing HybridObjectDetector...")
        print(f"Loading VLM: {vlm_model_id} at {precision_vlm} precision.")
        self.vlm_model, self.vlm_processor = self._load_vlm(vlm_model_id, precision_vlm)

        print(f"Loading LLM: {llm_model_id} at {precision_llm} precision.")
        self.llm_model, self.llm_tokenizer = self._load_llm(llm_model_id, precision_llm)

        self.device = self.vlm_model.device
        print("HybridObjectDetector is ready.")

    def _create_quantization_config(self, precision):
        """Helper function to create the correct BitsAndBytesConfig."""
        torch_dtype = torch.float16
        if precision == "4bit":
            print("Setting up 4-bit quantization...")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
            )
        if precision == "8bit":
            print("Setting up 8-bit quantization...")
            return BitsAndBytesConfig(load_in_8bit=True)
        if precision == "16bit":
            print("Loading in 16-bit (half-precision).")
            return None
        raise ValueError("Precision must be '4bit', '8bit', or '16bit'.")

    def _load_vlm(self, model_id, precision):
        """Loads the Vision-Language Model (LLaVA)."""
        quantization_config = self._create_quantization_config(precision)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return model, processor

    def _load_llm(self, model_id, precision):
        """Loads the text-only Large Language Model."""
        quantization_config = self._create_quantization_config(precision)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer

    def _run_vlm_inference(self, prompt_text, images, max_new_tokens=256, temperature=0.1):
        """Runs inference using the VLM (LLaVA)."""
        inputs = self.vlm_processor(text=prompt_text, images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=self.vlm_processor.tokenizer.pad_token_id,
            )
        full_output = self.vlm_processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        parts = re.split(r"ASSISTANT:", full_output, flags=re.IGNORECASE)
        return parts[-1].strip() if len(parts) > 1 else full_output.strip()

    def _run_llm_inference(self, user_prompt, max_new_tokens=256, temperature=0.1):
        """Runs inference using the LLM, applying the chat template."""
        messages = [{"role": "user", "content": user_prompt}]
        prompt = self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            terminators = [
                self.llm_tokenizer.eos_token_id,
                self.llm_tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            outputs = self.llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=terminators,
            )
        response = self.llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        return response.strip()

    def _load_images(self, image_sources):
        """Loads an image from a source (URL, path, or PIL object)."""
        if not image_sources:
            return None
        if not isinstance(image_sources, list):
            image_sources = [image_sources]
        images = []
        for src in image_sources:
            if isinstance(src, Image.Image):
                images.append(src.convert("RGB"))
            elif str(src).startswith("http"):
                response = requests.get(src)
                images.append(Image.open(BytesIO(response.content)).convert("RGB"))
            else:
                images.append(Image.open(src).convert("RGB"))
        return images

    def _parse_final_list(self, raw_output_str):
        """Safely parses a string that should contain a Python list."""
        match = re.search(r"(\[.*?\])", raw_output_str, re.DOTALL)
        if not match:
            print(f"Warning: Could not find a list in the output: {raw_output_str}")
            return []
        list_str = match.group(1)
        try:
            return ast.literal_eval(list_str)
        except (ValueError, SyntaxError):
            print(f"Warning: ast.literal_eval failed for: {list_str}. Attempting manual parsing.")
            try:
                content = list_str.strip()[1:-1]
                items = [item.strip().strip("'\"") for item in content.split(",")]
                return [item for item in items if item]
            except Exception as e:
                print(f"Error: Manual parsing also failed for '{list_str}': {e}")
                return []

    def detect_removal_targets_multi_step(self, image_source, user_command):
        """Performs a multi-step inference using the best model for each task."""
        prompt1 = (
            f"Instruction: \"{user_command}\".\n\n"
            f"Based on this instruction, what are the primary objects to be removed? "
            f"Respond with only the object's full name. Do not write a sentence."
        )
        primary_target = self._run_llm_inference(prompt1, max_new_tokens=50).strip().strip('"')

        image = self._load_images(image_source)
        prompt2 = (
            f"<image>\nUSER: The user wants to remove the '{primary_target}'. "
            f"Analyze the image and provide a detailed, objective description of the '{primary_target}' AND all its associated elements. "
            f"Your description must systematically identify the following, if present:\n"
            f"1.  **Lighting-dependent effects:** Identify any shadows it casts or reflections it creates.\n"
            f"2.  **Physically connected objects:** Describe any objects it is physically touching, holding, wearing, or supporting.\n"
            f"3.  **Function-generated elements:** Describe anything being actively produced or emitted (e.g., water from a sprinkler).\n"
            f"4.  **Contextually linked objects:** Describe any nearby standalone objects that are related to the primary target (e.g., a sign that describes it, a mailbox belonging to it).\n\n"
            f"Focus strictly on these points. Do not describe the general background or other unrelated items.\nASSISTANT:"
        )
        image_description_str = self._run_vlm_inference(prompt2, image, max_new_tokens=256)

        prompt3 = (
            f"You are a meticulous logical reasoner. Your primary goal is to analyze an image description to determine which objects, besides the primary target, must be removed to ensure the final image is physically plausible and contextually coherent.\n\n"
            f"Follow these strict rules:\n"
            f"1.  Find out all elements that should also be removed if the primary target is removed. Including Lighting-dependent, Physically connected, Produced, and Contextually linked objects.\n"
            f"2.  Your analysis must be based EXCLUSIVELY on the facts in the 'Image Description' below. Do not invent details.\n"
            f"3.  Do NOT include static background items that the target is merely on or in front of (e.g., floor, stairs, walls, railings).\n\n"
            f"Now, apply these rules to the following scenario:\n\n"
            f"Primary Target for Removal: '{primary_target}'\n"
            f"Image Description: \"{image_description_str}\"\n\n"
            f"Based exclusively on the 'Image Description', write your short reasoning paragraph now."
        )
        reasoning_str = self._run_llm_inference(prompt3, max_new_tokens=300)

        prompt4 = (
            f"Based on the reasoning provided, give a brief summary and then a Python list of all objects to be removed.\n\n"
            f"Follow these rules:\n"
            f"1. Do not include any abstract concepts or motions.\n"
            f"2. Avoid complex terms. Specify the name of each object clearly.\n"
            f"3. Do NOT include static background items (e.g., floor, wall, stairs).\n\n"
            f"--- Example ---\n"
            f"Reasoning: The person is carrying a backpack and has a shadow on the floor. These should be removed. The person is sitting on the staircase, leaning on the wall.\n\n"
            f"Output:\n"
            f"Summary: The person, the backpack he is carrying, and his shadow on the floor should be removed. The staircase and the wall are background object.\n"
            f"Target Objects: [\"person\", \"backpack\", \"person's shadow\"]\n"
            f"--- End Example ---\n\n"
            f"Now, process the following reasoning:\n"
            f"Reasoning: \"{reasoning_str}\"\n\n"
            f"Output:"
        )
        summary_str = self._run_llm_inference(prompt4, max_new_tokens=200)

        extracted_list = self._parse_final_list(summary_str)

        if " " in primary_target:
            prompt5 = (
                f"Transform the text. Follow the pattern.\n\n"
                f"'person and plant' -> 'person. plant'\n"
                f"'black cat' -> 'black cat'\n"
                f"'a running dog' -> 'a running dog'\n"
                f"'a red bicycle and a blue helmet' -> 'a red bicycle. a blue helmet'\n\n"
                f"'{primary_target}' ->"
            )
            refined_item_str = self._run_llm_inference(prompt5, max_new_tokens=50)
        else:
            refined_item_str = primary_target

        extracted_list.append(refined_item_str.strip())

        return {
            "primary_target": primary_target,
            "image_description": image_description_str,
            "reasoning": reasoning_str,
            "summary_and_list": summary_str,
            "target_objects": list(dict.fromkeys(extracted_list)),
        }

    def process_dataset(self, entries, image_folder, output_json_path):
        """Processes a dataset of images and instructions."""
        all_results = []
        for i, entry in enumerate(tqdm(entries, desc="Phase 1: Detector")):
            print(f"\n--- Processing entry {i + 1}/{len(entries)} ---")
            try:
                image_filename = entry.get("input")
                user_command = entry.get("instruction")
                if not image_filename or not user_command:
                    print(f"Skipping entry {i+1} due to missing 'input' or 'instruction'.")
                    continue

                image_filename = str(image_filename)
                if os.path.isabs(image_filename):
                    image_path = os.path.join(image_folder, os.path.basename(image_filename))
                else:
                    image_path = os.path.join(image_folder, image_filename)

                if not os.path.exists(image_path):
                    image_path = os.path.join(image_folder, os.path.basename(image_filename))

                print(f"Image Path: {image_path}")
                print(f"User Command: {user_command}")

                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found at: {image_path}")

                detection_output = self.detect_removal_targets_multi_step(
                    image_source=image_path,
                    user_command=user_command,
                )

                result_record = {
                    "original_entry": entry,
                    "status": "success",
                    "detection_results": detection_output,
                }

            except Exception as e:
                print(f"!!! An error occurred while processing entry {i + 1}: {e}")
                result_record = {
                    "original_entry": entry,
                    "status": "failed",
                    "error": str(e),
                }

            all_results.append(result_record)

        print(f"\n--- Dataset processing complete. Saving results to {output_json_path} ---")
        try:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print("Successfully saved results.")
        except Exception as e:
            print(f"!!! Failed to save results to JSON file: {e}")
        return all_results


def set_seed(seed: int):
    """Sets the seed for reproducibility for all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)