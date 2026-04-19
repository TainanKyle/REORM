import os
import argparse
import glob
import sys
from pathlib import Path
import torch
from PIL import Image
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPONENT_DIR = REPO_ROOT / "component"
OBJECTCLEAR_DIR = COMPONENT_DIR / "ObjectClear"

if str(OBJECTCLEAR_DIR) not in sys.path:
    sys.path.insert(0, str(OBJECTCLEAR_DIR))

from objectclear.pipelines import ObjectClearPipeline
from objectclear.utils import resize_by_short_side


def inference_objectclear_dataset(image_folder, mask_folder, output_folder, 
                               use_fp16=True, steps=20, guidance_scale=2.5, 
                               seed=42, no_agf=False, cache_dir=None):
    """
    Inference ObjectClear on a dataset of images and masks.
    
    Parameters
    ----------
    image_folder : str
        Path to the folder containing input images.
    mask_folder : str
        Path to the folder containing input masks.
    output_folder : str
        Path to the folder where output images will be saved.
    steps : int
        Number of diffusion inference steps. Default is 20.
    guidance_scale : float
        CFG guidance scale. Default is 2.5.
    use_fp16 : bool
        Whether to use float16 for inference. Default is True.
    seed : int
        Random seed for torch.Generator. Default is 42.
    no_agf : bool
        If True, disable Attention Guided Fusion. Default is False.
    cache_dir : str or None
        Path to cache directory. Default is None.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_folder, exist_ok=True)
    args = argparse.Namespace()
    args.input_path = image_folder
    args.mask_path = mask_folder
    args.output_path = output_folder
    args.cache_dir = cache_dir
    args.use_fp16 = use_fp16
    args.steps = steps
    args.guidance_scale = guidance_scale
    args.seed = seed
    args.no_agf = no_agf

    # ------------------ set up ObjectClear pipeline -------------------
    torch_dtype = torch.float16 if args.use_fp16 else torch.float32
    variant = "fp16" if args.use_fp16 else None
    generator = torch.Generator(device=device).manual_seed(args.seed)
    use_agf = not args.no_agf
    pipe = ObjectClearPipeline.from_pretrained_with_custom_modules(
        "jixin0101/ObjectClear",
        torch_dtype=torch_dtype,
        apply_attention_guided_fusion=use_agf,
        cache_dir=args.cache_dir,
        variant=variant,
    )
    pipe.to(device)
    
    input_img_list = sorted(glob.glob(os.path.join(args.input_path, '*.[jpJP][pnPN]*[gG]')))
    input_mask_list = sorted(glob.glob(os.path.join(args.mask_path, '*.[jpJP][pnPN]*[gG]')))

    # -------------------- start to processing ---------------------
    # for i, (img_path, mask_path) in enumerate(zip(input_img_list, input_mask_list)):
    from tqdm import tqdm
    for i, (img_path, mask_path) in enumerate(tqdm(zip(input_img_list, input_mask_list), desc="Phase 3: Editing Images")):
        img_name = os.path.basename(img_path)
        basename, ext = os.path.splitext(img_name)
        print(f'[{i+1}/{len(input_img_list)}] Processing: {img_name}')
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image_or = image.copy()
        
        # Our model was trained on 512×512 resolution.
        # Resizing the input so that the **shorter side is 512** helps achieve the best performance.
        image = resize_by_short_side(image, 512, resample=Image.BICUBIC)
        mask = resize_by_short_side(mask, 512, resample=Image.NEAREST)
        
        w, h = image.size
    
        result = pipe(
            prompt="remove the instance of object",
            image=image,
            mask_image=mask,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=h,
            width=w,
            return_attn_map=False,
        )
        
        fused_img_pil = result.images[0]

        # save results
        save_path = os.path.join(args.output_path, f'{basename}.png')
        fused_img_pil = fused_img_pil.resize(image_or.size)
        fused_img_pil.save(save_path)

    print(f'\nAll results are saved in {args.output_path}')
