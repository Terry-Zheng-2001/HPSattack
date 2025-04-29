from datasets import load_dataset
import torch
from utils.visualization_utils import get_all_pics, save_attacked_images, images_to_tensor
from models.hpsv2_model import HPSv2Module
import hpsv2
import os
import pandas as pd
from utils.attacks import NESAttack
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#load the model
model = HPSv2Module("weights/HPS_v2_compressed.pt", device="cuda")
print("model loaded")
#load the dataset
dataset = load_dataset("zhwang/HPDv2", split="test")
print("dataset loaded")
#load the attackers
NESAttacker = NESAttack(model, eps=8/255, alpha=2/255, steps=40, nes_samples=50, device="cuda")
#NESAttacker = NESAttack(model, eps=8/255, alpha=2/255, steps=15, nes_samples=25, device="cuda")
#make output directory
output_dir = "outputs/black_box_attack"
os.makedirs(output_dir, exist_ok=True)
#make score directory
score_dir = "outputs/black_box_attack/scores1"
os.makedirs(score_dir, exist_ok=True)
#make attacked image directory
NES_attacked_image_dir = "outputs/black_box_attack/NES_attacked_images1"
os.makedirs(NES_attacked_image_dir, exist_ok=True)

for i in tqdm(range(100)):
    image_paths, prompt = get_all_pics(dataset, i)
    print("attack image paths: ", image_paths)
    #get the original score(v2.0)
    score2_0 = hpsv2.score(image_paths, prompt, hps_version="v2.0")#list
    save_paths_NES = []
    for j in range(len(image_paths)):
        image_tensor = images_to_tensor([image_paths[j]], model)
        #generate the adversarial image
        adv_images_NES = NESAttacker.attack(image_tensor, prompt)
        #save the adversarial image
        save_paths_NES.append(save_attacked_images(adv_images_NES, [image_paths[j]], NES_attacked_image_dir)[0])
        print("save_paths_NES: ", save_paths_NES[j])
    #get the attacked score(v2.0)
    score2_0_attacked_NES = hpsv2.score(save_paths_NES, prompt, hps_version="v2.0")#list
    # Create a dataframe for this iteration and save to individual file
    results_df = pd.DataFrame({
        "image_path": image_paths, 
        "prompt": [prompt for _ in range(len(score2_0))], 
        "score2_0_original": score2_0, 
        "score2_0_attacked_NES": score2_0_attacked_NES,
        "sample_id": i
    })
    
    # Save to individual CSV file
    individual_file_path = os.path.join(score_dir, f"results_{i}.csv")
    results_df.to_csv(individual_file_path, index=False)














