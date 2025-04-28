import os
import csv
import pandas as pd
from datasets import load_dataset
import torch
from models.hpsv2_model import HPSv2Module
from utils.attacks import FGSMAttack, PGDAttack
from utils.visualization_utils import images_to_tensor, save_attacked_images
import hpsv2
from tqdm import tqdm
# Model descriptions
MODEL_LIST = ["dreamshaper", "dreamlike_photoreal", "realistic_vision", "openjourney", "sd_v1_5", "sd_v2_1"]
#load the model
model = HPSv2Module("weights/HPS_v2_compressed.pt", device="cuda")

#load the attackers
FGSMAttacker = FGSMAttack(model, eps=4/255)
PGDAttacker = PGDAttack(model, eps=4/255, alpha=2/255, steps=4)
image_path = f"data/generated_images/"
df = pd.read_csv("data/generated_images/metadata.csv")


for model_name in MODEL_LIST:
    score_dir = f"outputs/attack_images_generated_by_different_models/scores/{model_name}"
    os.makedirs(score_dir, exist_ok=True)

    df_model = df[df["model"] == model_name]
    
    output_dir = f"outputs/attack_images_generated_by_different_models/{model_name}"
    os.makedirs(output_dir, exist_ok=True)
    #make attacked image directory
    PGD_attacked_image_dir = f"outputs/attack_images_generated_by_different_models/{model_name}/PGD_attacked_images"
    FGSM_attacked_image_dir = f"outputs/attack_images_generated_by_different_models/{model_name}/FGSM_attacked_images"
    os.makedirs(PGD_attacked_image_dir, exist_ok=True)
    os.makedirs(FGSM_attacked_image_dir, exist_ok=True)
    print(f"Processing {model_name}...")
    for i in tqdm(range(max(df_model["prompt_idx"])+1)):
        df_prompt = df_model[df_model["prompt_idx"] == i]
        image_paths = df_prompt["filename"].tolist()
        image_paths = [os.path.join(image_path, path) for path in image_paths]
        prompt = df_prompt["prompt"].tolist()[0]

        #get the original score(v2.0 and v2.1)
        score2_0 = hpsv2.score(image_paths, prompt, hps_version="v2.0")#list
        score2_1 = hpsv2.score(image_paths, prompt, hps_version="v2.1")#list

        image_tensors = images_to_tensor(image_paths, model)
        adv_images_PGD = PGDAttacker.attack(image_tensors, prompt)
        adv_images_FGSM = FGSMAttacker.attack(image_tensors, prompt)
        save_paths_PGD = save_attacked_images(adv_images_PGD, image_paths, PGD_attacked_image_dir)
        save_paths_FGSM = save_attacked_images(adv_images_FGSM, image_paths, FGSM_attacked_image_dir)
        #get the attacked score(v2.0 and v2.1)
        score2_0_attacked_PGD = hpsv2.score(save_paths_PGD, prompt, hps_version="v2.0")#list
        score2_1_attacked_PGD = hpsv2.score(save_paths_PGD, prompt, hps_version="v2.1")#list
        score2_0_attacked_FGSM = hpsv2.score(save_paths_FGSM, prompt, hps_version="v2.0")#list
        score2_1_attacked_FGSM = hpsv2.score(save_paths_FGSM, prompt, hps_version="v2.1")#list
        #save the score
        df_score = pd.DataFrame({
            "image_path": image_paths,
            "prompt": [prompt for _ in range(len(score2_0_attacked_PGD))],
            "score2_0_original": score2_0,
            "score2_1_original": score2_1,
            "score2_0_attacked_PGD": score2_0_attacked_PGD,
            "score2_1_attacked_PGD": score2_1_attacked_PGD,
            "score2_0_attacked_FGSM": score2_0_attacked_FGSM,
            "score2_1_attacked_FGSM": score2_1_attacked_FGSM
        })
        df_score.to_csv(f"{score_dir}/results_{model_name}_{i}.csv", index=False)