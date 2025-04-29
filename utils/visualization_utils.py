import os
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
def get_prompts(dataset, n,m):
    prompts = []
    for i in range(n,n+m):
        prompts.append(dataset[i]['prompt'])
    return prompts

def get_all_sorted_pics(dataset, n):
    rank = dataset[n]['rank']
    images_path = dataset[n]['image_path']
    sorted_images_path = [images_path[i] for i in sorted(range(len(rank)), key=lambda i: rank[i])]
    sorted_images_path = ["data/test/" + path for path in sorted_images_path]
    prompt = dataset[n]['prompt']
    return sorted_images_path, prompt

def get_best_pic(dataset, n):
    sorted_images_path, prompt = get_all_sorted_pics(dataset, n)
    images_path = sorted_images_path[0]
    return images_path, prompt

def get_worst_pic(dataset, n):
    sorted_images_path, prompt = get_all_sorted_pics(dataset, n)
    images_path = sorted_images_path[-1]
    return images_path, prompt

def get_all_pics(dataset, n):
    images_path = dataset[n]['image_path']
    images_path = ["data/test/" + path for path in images_path]
    prompt = dataset[n]['prompt']
    return images_path, prompt

def show_tensor_with_preprocess_config(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    
    # Handle both single image and batch of images
    if tensor.dim() == 4 and tensor.size(0) > 1:
        # For batch show all images
        imgs = tensor.detach().cpu()
        num_images = imgs.size(0)
        fig, axes = plt.subplots(1, num_images, figsize=(4*num_images, 4))
        
        for i in range(num_images):
            img = imgs[i] * std + mean
            img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
            if num_images > 1:
                axes[i].imshow(img_np)
                axes[i].axis("off")
                axes[i].set_title(f"Image {i+1}")
            else:
                axes.imshow(img_np)
                axes.axis("off")
                axes.set_title(f"Image {i+1}")
        
        plt.suptitle("Images restored from preprocess")
        plt.tight_layout()
        plt.show()
    else:
        # For single image
        img = tensor.squeeze(0).detach().cpu()
        img = img * std + mean
        img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
        plt.imshow(img_np)
        plt.axis("off")
        plt.title("Image restored from preprocess")
        plt.show()



def images_to_tensor(image_paths, model):
    image_tensors = []
    for image_path in image_paths:
        image_tensor = model.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(model.device, dtype=torch.float32)
        image_tensors.append(image_tensor)
    return torch.cat(image_tensors, dim=0)  # Concatenate all tensors along batch dimension


def save_attacked_images(adv_images: torch.Tensor, image_paths: list[str], output_dir: str):
    """
    Save a batch of adversarial images to a fixed output directory without quality loss.

    Args:
        adv_images: Tensor of shape (B, 3, H, W) containing normalized images.
        image_paths: List of original image paths to derive filenames.
    """
    # Fixed output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # CLIP normalization parameters
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    
    save_paths = []

    # Check if adv_images is a sparse tensor and convert to dense if needed
    if adv_images.is_sparse:
        adv_images = adv_images.to_dense()
        
    # For batch show all images
    imgs = adv_images.detach().cpu()
    num_images = imgs.size(0)
    for i in range(num_images):
        img = imgs[i] * std + mean
        img = img.squeeze(0)
        img_np = img.permute(1, 2, 0).clamp(0, 1).numpy()
        pil_img = Image.fromarray((img_np * 255).astype(np.uint8))
        orig_filename = os.path.basename(image_paths[i])
        filename, _ = os.path.splitext(orig_filename)
        # Save as PNG to avoid JPEG compression artifacts
        save_path = os.path.join(output_dir, f"{filename}.png")
        pil_img.save(save_path, format='PNG')
        save_paths.append(save_path)
    return save_paths
