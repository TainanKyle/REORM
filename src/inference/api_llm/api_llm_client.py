import ast
import base64
import json
import os
import re

from openai import OpenAI
from tqdm import tqdm


class ResultStorage:
    detector_reasoning = {}
    initial_targets = {}
    descriptions = {}
    examination_results = {}
    objects_to_correct = {}

    @classmethod
    def store_detector_results(cls, image_id, target_object, description):
        cls.initial_targets[image_id] = target_object
        cls.descriptions[image_id] = description

    @classmethod
    def store_detector_reasoning(cls, image_id, reasoning):
        cls.detector_reasoning[image_id] = reasoning

    @classmethod
    def store_examiner_results(cls, image_id, result):
        cls.examination_results[image_id] = result

    @classmethod
    def store_objects_to_correct(cls, image_id, objects):
        cls.objects_to_correct[image_id] = objects

    @classmethod
    def save_to_json(cls, file_path: str):
        print(f"Saving results to {file_path}...")

        all_results = {}
        all_keys = (
            set(cls.initial_targets.keys())
            | set(cls.descriptions.keys())
            | set(cls.objects_to_correct.keys())
            | set(cls.examination_results.keys())
        )

        for image_id in all_keys:
            all_results[image_id] = {
                "initial_targets": cls.initial_targets.get(image_id, ""),
                "detector_reasoning": cls.detector_reasoning.get(image_id, ""),
                "description": cls.descriptions.get(image_id, ""),
                "examination_result": cls.examination_results.get(image_id, ""),
                "objects_to_correct": cls.objects_to_correct.get(image_id, ""),
            }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
            print("Results saved successfully.")
        except IOError as e:
            print(f"Error saving file: {e}")

    @classmethod
    def load_from_json(cls, file_path: str):
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Starting with empty results.")
            return

        print(f"Loading results from {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)

            cls.initial_targets.clear()
            cls.descriptions.clear()
            cls.objects_to_correct.clear()
            cls.examination_results.clear()

            for image_id, data in all_results.items():
                if data.get("initial_targets"):
                    cls.initial_targets[image_id] = data["initial_targets"]
                if data.get("description"):
                    cls.descriptions[image_id] = data["description"]
                if data.get("objects_to_correct"):
                    cls.objects_to_correct[image_id] = data["objects_to_correct"]
                if data.get("examination_result"):
                    cls.examination_results[image_id] = data["examination_result"]

            print("Results loaded successfully.")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading file: {e}")


class LLMImageProcessor:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o"

    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def strip_location(phrase: str) -> str:
        return re.sub(
            r"\b(?:in|on|at|under|above|beneath|beside|by|near|inside|outside)\b.*$",
            "",
            phrase,
            flags=re.IGNORECASE,
        ).strip()


class ObjectDetector(LLMImageProcessor):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.prompt_template = """
# Identity
Helpful assistant that can help the user to identify the target objects and associated elements that should be removed from the image, ensuring the final result is contextually coherent.

# Your task
Given an image and a prompt, you need to analyze and identify the target objects and all related objects and effects that should be removed in the image. You need to provide a list of objects that should be removed in the image.

# Process Steps
1. Read the user prompt and identify the target objects with the attributes.
2. Carefully examine the image to identify any other objects or effects whose presence would become **physically implausible or contextually illogical** if the primary target object were removed. This includes, but is not limited to:
    * **Direct physical effects:** shadows, reflections.
    * **Interacting objects:** items being held, worn, or physically supported by the target.
3. Explain your understanding and then format your answer as shown in the examples.
4. Avoid vague or abstract noun phrases followed by relative clauses, such as "the object that..." or "any item that...". Use specific object names only. For example, write "the dog's toy" instead of "the object it is interacting with".
5. Avoid describing subtle background changes or textures, such as 'water ripples'

# Examples

<product_review id="example-1">
Remove the person.
</product_review>

<assistant_response id="example-1">
Reasoning: "The target object is the person. If the person is removed, his shadow and his scooter would appear contextually inconsistent, as the scooter would appear to stand upright without a rider."
Target Objects: ["person", "the person's shadow", "the scooter"]
</assistant_response>

<product_review id="example-2">
Remove the white dog.
</product_review>

<assistant_response id="example-2">
Reasoning: "The target object is a white dog. Removing the dog would make its reflection in the water and the toy it is playing with appear contextually inconsistent, as the toy would not reasonably be there without the dog."
Target Objects: ["white dog", "the white dog's reflection", "the dog toy"]
</assistant_response>

<product_review id="example-3">
Remove the person.
</product_review>

<assistant_response id="example-3">
Reasoning: "The target object is a person. Without the person, the bags and the cup he is holding would appear to float in midair, which is physically implausible."
Target Objects: ["person", "the bags", "the cup"]
</assistant_response>
"""

    def process_dataset(self, image_folder, entries, detail="low"):
        initial_targets = []
        descriptions = []
        empty_n = 0

        for idx, entry in enumerate(tqdm(entries, desc="API: Analyzer & Detector"), start=1):
            print(f"Processing image {idx}/{len(entries)}......")

            image_file = os.path.join(image_folder, entry["input"])
            prompt = entry["instruction"]
            image_id = entry["input"]

            base64_image = self.encode_image(image_file)

            response = self.client.responses.create(
                model=self.model,
                instructions=self.prompt_template,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low",
                            },
                        ],
                    }
                ],
            )

            ResultStorage.store_detector_reasoning(image_id, response.output_text)

            match = re.search(r"Objects:\s*(\[[^\]]*\])", response.output_text)
            if not match:
                initial_targets.append("")
                empty_n += 1
                ResultStorage.store_detector_results(image_id, "", "")
                continue

            objects_str = match.group(1)
            objects = ast.literal_eval(objects_str)
            cleaned_objects = [self.strip_location(obj.strip()) for obj in objects]
            cleaned_objects = ". ".join(obj.strip() for obj in cleaned_objects)
            initial_targets.append(cleaned_objects)

            descriptor = ImageDescriptor(api_key=self.client.api_key)
            description = descriptor.generate_description(image_file, objects_str, detail=detail)
            descriptions.append(description)

            ResultStorage.store_detector_results(image_id, cleaned_objects, description)

        print(f"Total images: {len(entries)}, empty target objects: {empty_n}")
        return initial_targets, descriptions


class ImageDescriptor(LLMImageProcessor):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.prompt_template = """
# Identity
You are a vision-language assistant that describes how the picture will look after the requested objects are removed

# Your task
Given one image and the user's removal request, describe the resulting image.

# Output Rules
1. Describe all visible elements in the resulting image as detailed as possible. Include all elements in the background, foreground, and any other visible parts of the image.
2. Avoid abstract emotions or psychology.
3. Do NOT mention the original image, the removal process, or phrases like "in the given image."
"""

    def generate_description(self, image_path, objects_str, detail="low"):
        base64_image = self.encode_image(image_path)

        response = self.client.responses.create(
            model=self.model,
            instructions=self.prompt_template,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Objects to be removed: " + objects_str},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": detail,
                        },
                    ],
                }
            ],
        )

        description = response.output_text.strip()
        ResultStorage.store_detector_results(os.path.basename(image_path), objects_str, description)
        return description


class ImageExaminer(LLMImageProcessor):
    def __init__(self, api_key=None):
        super().__init__(api_key)
        self.prompt_template = """
# Identity
You are a vision-language assistant that checks whether an edited image matches the description.

# Your task
Given one image and a description, examine whether the image contains any unexpected objects or elements not mentioned in the description. If there are such elements, list them. Otherwise return an empty list.

# Output Rules
1. Provide the reasoning first, and then list the objects that should be removed from the image.
2. Only include objects that should be removed, not objects that should be added.
3. Ignore discrepancies in the quantity of objects. For example, if the description says "a cup" and the image shows two cups, you should not list "additional cup" for removal.
4. Only include the specific name of the object, not its location. State the name directly (e.g., "a blue vase"), not in a vague way (e.g., "an object that looks like a vase"). Include the color or shape if you are certain about it.
5. If you are uncertain about an object's precise identity, do not list it.
6. Avoid using vague or large-scale location words such as "area", "space", or "region". Focus only on specific, tangible objects

# Examples

<product_review id="example-1">
The edited image features a kitchen with a stove, a sink, and a refrigerator. The stove is clean and shiny, the sink is empty, and the refrigerator is closed. There are no people or pets in the image.
</product_review>

<assistant_response id="example-1">
Reasoning: "The image shows a cozy kitchen. It contains a stove, a chair, and a refrigerator. The chair is not mentioned in the description, so it should be removed. There's something like a hand in the image, which doesn't appear i the description."
Objects to be removed: ["the chair, the hand, arm"]
</assistant_response>
"""

    def examine_dataset(self, image_folder, entries, descriptions, detail="auto"):
        objects_to_correct = []

        for idx, entry in enumerate(tqdm(entries, desc="API: Examiner"), start=1):
            print(f"Processing image {idx}/{len(entries)}......")

            image_file = os.path.join(image_folder, entry["input"])
            image_id = entry["input"]
            prompt = descriptions[image_id]

            base64_image = self.encode_image(image_file)

            response = self.client.responses.create(
                model=self.model,
                instructions=self.prompt_template,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": detail,
                            },
                        ],
                    }
                ],
            )

            match = re.search(r"Objects to be removed:\s*(\[[^\]]*\])", response.output_text)
            if not match:
                objects_to_correct.append("")
                ResultStorage.store_objects_to_correct(image_id, "")
                continue

            objects_str = match.group(1)
            objects = ast.literal_eval(objects_str)
            cleaned_objects = [self.strip_location(obj.strip()) for obj in objects if obj is not None]
            cleaned_objects = ". ".join(obj.strip() for obj in cleaned_objects)
            objects_to_correct.append(cleaned_objects)

            ResultStorage.store_objects_to_correct(image_id, cleaned_objects)
            ResultStorage.store_examiner_results(image_id, response.output_text)

        return objects_to_correct


def api_llm_detector_dataset(image_folder, entries, api=None, model="gpt-4o"):
    detector = ObjectDetector(api_key=api)
    detector.model = model
    initial_targets, descriptions = detector.process_dataset(image_folder, entries)
    return initial_targets, descriptions


def api_llm_examiner_dataset(image_folder, entries, descriptions, api=None, detail="auto", model="gpt-4o"):
    examiner = ImageExaminer(api_key=api)
    examiner.model = model
    objects_to_correct = examiner.examine_dataset(image_folder, entries, descriptions, detail=detail)
    return objects_to_correct