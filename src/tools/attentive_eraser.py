import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import os
from tqdm import tqdm
from PIL import Image
import shutil

class AttentiveEraser:
    def __init__(self, model_path="stabilityai/stable-diffusion-xl-base-1.0"):
        self.dtype = torch.float16
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.scheduler = DDIMScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            clip_sample=False, 
            set_alpha_to_one=False
        )
        
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_path,
            custom_pipeline="pipeline_stable_diffusion_xl_attentive_eraser",
            scheduler=self.scheduler,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        
        self.generator = torch.Generator(device=self.device).manual_seed(123)
    
    def preprocess_image(self, image_path):
        image = to_tensor((load_image(image_path)))
        image = image.unsqueeze_(0).float() * 2 - 1  # [0,1] --> [-1,1]
        if image.shape[1] != 3:
            image = image.expand(-1, 3, -1, -1)
        image = F.interpolate(image, (1024, 1024))
        image = image.to(self.dtype).to(self.device)
        return image

    def preprocess_mask(self, mask_path):
        mask = to_tensor((load_image(mask_path, convert_method=lambda img: img.convert('L'))))
        mask = mask.unsqueeze_(0).float()  # 0 or 1
        mask = F.interpolate(mask, (1024, 1024))
        mask = gaussian_blur(mask, kernel_size=(77, 77))
        mask[mask < 0.1] = 0
        mask[mask >= 0.1] = 1
        mask = mask.to(self.dtype).to(self.device)
        return mask
    
    def set_seed(self, seed):
        self.generator = torch.Generator(device=self.device).manual_seed(seed)   
        
    def remove_object(self, source_image_path, mask_path, do_resize, output_path, 
                      strength=0.8, rm_guidance_scale=9, ss_steps=9, 
                      ss_scale=0.3, num_inference_steps=50, prompt=""):
        source_image = self.preprocess_image(source_image_path)
        mask = self.preprocess_mask(mask_path)

        image = self.pipeline(
            prompt=prompt, 
            image=source_image,
            mask_image=mask,
            height=1024,
            width=1024,
            AAS=True,
            strength=strength,
            rm_guidance_scale=rm_guidance_scale,
            ss_steps=ss_steps,
            ss_scale=ss_scale,
            AAS_start_step=0,
            AAS_start_layer=34,
            AAS_end_layer=70,
            num_inference_steps=num_inference_steps,
            generator=self.generator,
            guidance_scale=1,
        ).images[0]

        if do_resize:
            original_size = load_image(source_image_path).size
            image =  image.resize(original_size, resample=Image.BILINEAR)
        
        if output_path:
            image.save(output_path)
        return image
    
    def process_dataset(self, input_folder, mask_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        mask_files = [f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in tqdm(image_files):
            mask_file = image_file
            if mask_file not in mask_files:
                print(f"Mask file {mask_file} not found for image {image_file}. Skipping.")
                continue
            
            source_image_path = os.path.join(input_folder, image_file)
            mask_path = os.path.join(mask_folder, mask_file)
            output_path = os.path.join(output_folder, image_file)

            self.remove_object(source_image_path, mask_path, do_resize=True, output_path=output_path)
    
    def correct_dataset(self, entries, input_folder, mask_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for entry in tqdm(entries, desc="Attentive Eraser"):
            image_name = entry['input']
            image_path = os.path.join(input_folder, image_name)
            mask_path = os.path.join(mask_folder, image_name)
            output_path = os.path.join(output_folder, image_name)

            if not os.path.exists(mask_path):
                # copy the original image to the output folder 
                shutil.copy(image_path, output_path)
                continue
                
            self.remove_object(image_path, mask_path, do_resize=True, output_path=output_path)


def inference_attentive_eraser_dataset(input_folder, mask_folder, output_folder):
    eraser = AttentiveEraser()
    eraser.process_dataset(input_folder, mask_folder, output_folder)

