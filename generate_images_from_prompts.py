import os
import csv
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from tqdm import tqdm
from utils.visualization_utils import get_prompts
# Model descriptions
MODEL_LIST = {
    "dreamshaper": "Lykon/dreamshaper-8",  # Versatile model with good concept understanding
    "dreamlike_photoreal": "dreamlike-art/dreamlike-photoreal-2.0",  # Photorealistic image generation model
    "realistic_vision": "SG161222/Realistic_Vision_V5.1_noVAE",  # Specialized for realistic images and portraits
    "openjourney": "prompthero/openjourney",  # Midjourney-style images with vibrant colors
    "sd_v1_5": "runwayml/stable-diffusion-v1-5",  # Original stable diffusion model, good for general purpose
    "sd_v2_1": "stabilityai/stable-diffusion-2-1",  # Improved version of SD with better image quality
}
SAVE_DIR = "data/generated_images"   # Path to save generated images
METADATA_FILE = os.path.join(SAVE_DIR, "metadata.csv")  # Metadata file
NUM_IMAGES_PER_PROMPT = 8       # Number of images to generate per prompt
IMG_HEIGHT = 512
IMG_WIDTH = 512


torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

CUSTOM_HF_CACHE_DIR = "G:/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = CUSTOM_HF_CACHE_DIR

def load_pipeline(model_name):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        cache_dir=CUSTOM_HF_CACHE_DIR,
    )
    pipe = pipe.to(device)
    return pipe

def generate_images(prompts):

    os.makedirs(SAVE_DIR, exist_ok=True)

    with open(METADATA_FILE, mode='w', newline='') as csvfile:
        fieldnames = ['model', 'prompt_idx', 'prompt', 'image_idx', 'filename', 'seed', 'height', 'width']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for model_key, model_path in MODEL_LIST.items():
            print(f"Loading model: {model_key}")
            pipe = load_pipeline(model_path)

            model_save_dir = os.path.join(SAVE_DIR, model_key)
            os.makedirs(model_save_dir, exist_ok=True)

            for idx, prompt in enumerate(tqdm(prompts, desc=f"Generating with {model_key}")):
                for i in range(NUM_IMAGES_PER_PROMPT):
                    seed = 42 + idx * 10 + i
                    generator = torch.manual_seed(seed)
                    
                    # Try to generate images, if NSFW content is detected, retry
                    max_retries = 10
                    for retry in range(max_retries):
                        try:
                            if "sdxl" in model_key:
                                result = pipe(prompt, num_inference_steps=30, height=IMG_HEIGHT, width=IMG_WIDTH, guidance_scale=7.5, generator=generator)
                            else:
                                result = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, generator=generator)
                            
                            # Check if there is NSFW content
                            if hasattr(result, "nsfw_content_detected") and result.nsfw_content_detected is not None and result.nsfw_content_detected[0]:
                                print(f"NSFW content detected for prompt {idx}, image {i}. Retrying with new seed...")
                                seed = seed + 1000  # Use new seed
                                generator = torch.manual_seed(seed)
                                continue
                            
                            # Ensure result.images is not None and has content
                            if result.images is None or len(result.images) == 0:
                                print(f"No images generated for prompt {idx}, image {i}. Retrying with new seed...")
                                seed = seed + 1000
                                generator = torch.manual_seed(seed)
                                continue
                                
                            image = result.images[0]
                            break
                        except Exception as e:
                            print(f"Error generating image: {e}. Retrying with new seed...")
                            seed = seed + 1000
                            generator = torch.manual_seed(seed)
                    else:
                        # If all retries fail, use a blank image
                        print(f"Failed to generate safe image after {max_retries} attempts for prompt {idx}, image {i}")
                        from PIL import Image
                        image = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color='white')

                    filename = f"{model_key}_{idx}_{i}.png"
                    save_path = os.path.join(model_save_dir, filename)
                    image.save(save_path)

                    # Write metadata
                    writer.writerow({
                        'model': model_key,
                        'prompt_idx': idx,
                        'prompt': prompt,
                        'image_idx': i,
                        'filename': os.path.join(model_key, filename),
                        'seed': seed,
                        'height': IMG_HEIGHT,
                        'width': IMG_WIDTH
                    })

if __name__ == "__main__":
    from datasets import load_dataset

    dataset = load_dataset("zhwang/HPDv2", split="test")
    prompts = get_prompts(dataset, 10, 10)
    generate_images(prompts)
