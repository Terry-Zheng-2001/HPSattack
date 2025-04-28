from datasets import load_dataset
import torch
from utils.visualization_utils import get_all_pics, save_attacked_images, images_to_tensor
from models.hpsv2_model import HPSv2Module
import hpsv2
import os
import pandas as pd
from utils.attacks import PGDAttack, FGSMAttack
import numpy as np
from tqdm import tqdm

#load the model
model = HPSv2Module("weights/HPS_v2_compressed.pt", device="cuda")

#load the dataset
dataset = load_dataset("zhwang/HPDv2", split="test")

#load the attackers
FGSMAttacker = FGSMAttack(model, eps=4/255)
PGDAttacker = PGDAttack(model, eps=4/255, alpha=2/255, steps=4)

#make output directory
output_dir = "outputs/white_box_attack"
os.makedirs(output_dir, exist_ok=True)
#make score directory
score_dir = "outputs/white_box_attack/scores"
os.makedirs(score_dir, exist_ok=True)
#make attacked image directory
PGD_attacked_image_dir = "outputs/white_box_attack/PGD_attacked_images"
FGSM_attacked_image_dir = "outputs/white_box_attack/FGSM_attacked_images"
os.makedirs(PGD_attacked_image_dir, exist_ok=True)
os.makedirs(FGSM_attacked_image_dir, exist_ok=True)

for i in tqdm(range(100)):
    image_paths, prompt = get_all_pics(dataset, i)
    #get the original score(v2.0 and v2.1)
    score2_0 = hpsv2.score(image_paths, prompt, hps_version="v2.0")#list
    score2_1 = hpsv2.score(image_paths, prompt, hps_version="v2.1")#list
    image_tensors = images_to_tensor(image_paths, model)
    #generate the adversarial image
    adv_images_PGD = PGDAttacker.attack(image_tensors, prompt)
    adv_images_FGSM = FGSMAttacker.attack(image_tensors, prompt)
    #save the adversarial image
    save_paths_PGD = save_attacked_images(adv_images_PGD, image_paths, PGD_attacked_image_dir)
    save_paths_FGSM = save_attacked_images(adv_images_FGSM, image_paths, FGSM_attacked_image_dir)

    #get the attacked score(v2.0 and v2.1)
    score2_0_attacked_PGD = hpsv2.score(save_paths_PGD, prompt, hps_version="v2.0")#list
    score2_1_attacked_PGD = hpsv2.score(save_paths_PGD, prompt, hps_version="v2.1")#list
    score2_0_attacked_FGSM = hpsv2.score(save_paths_FGSM, prompt, hps_version="v2.0")#list
    score2_1_attacked_FGSM = hpsv2.score(save_paths_FGSM, prompt, hps_version="v2.1")#list
    # Create a dataframe for this iteration and save to individual file
    results_df = pd.DataFrame({
        "image_path": image_paths, 
        "prompt": [prompt for _ in range(len(score2_0))], 
        "score2_0_original": score2_0, 
        "score2_1_original": score2_1,
        "score2_0_attacked_PGD": score2_0_attacked_PGD,
        "score2_1_attacked_PGD": score2_1_attacked_PGD,
        "score2_0_attacked_FGSM": score2_0_attacked_FGSM,
        "score2_1_attacked_FGSM": score2_1_attacked_FGSM,
        "sample_id": i
    })
    
    # Save to individual CSV file
    individual_file_path = os.path.join(score_dir, f"results_{i}.csv")
    results_df.to_csv(individual_file_path, index=False)














