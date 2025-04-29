# HPS Attack Research

This repository contains research code for attacking the Human Preference Score (HPS) model, including both white-box and black-box attack implementations.

## Project Overview

This project focuses on exploring vulnerabilities in the HPS model through various attack methods. It includes implementations of:
- White-box attacks
- Black-box attacks using NES (Natural Evolution Strategies)
- Image generation and attack evaluation

## Project Structure

```
.
├── data/                  # Dataset storage
│   └── test/             # Test images directory
├── models/               # Model definitions
├── outputs/              # Attack results and generated images
├── utils/                # Utility functions
├── weights/              # Model weights
├── MLtask.ipynb          # Machine learning tasks notebook (for testing various functions)
├── data_process.ipynb    # Data processing notebook (for processing experimental data)
├── black_box_attack.py   # Black-box attack implementation
├── white_box_attack.py   # White-box attack implementation
├── generate_images_from_prompts.py  # Image generation script
├── attack_images_generated_by_different_models.py  # Multi-model attack evaluation
└── requirements.txt      # Python dependencies
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Terry-Zheng-2001/HPSattack
cd HPSattack
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required files:
- Download the HPS model: [HPS_v2_compressed.pt](https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt)
  - Place the downloaded .pt file in the `weights/` directory
- Download test images: [Test Images](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155172150_link_cuhk_edu_hk/EVnjOngvDO1MhIp7hVr8GXgBmxVDcSk7s9Xuu9srO4YLbA?e=8PqYud)
  - Place the downloaded test images in the `data/test/` directory

## Notebook Usage

- `MLtask.ipynb`: This notebook is used for testing various functions related to the model and attacks
- `data_process.ipynb`: This notebook is used for processing and analyzing the experimental data obtained from the attacks

## Usage
### White-box Attack
```python
python white_box_attack.py
```
### Black-box Attack
```python
python black_box_attack.py
```
### Multi-model Attack
```python
python attack_images_generated_by_different_models.py
```


### Image Generation
```python
python generate_images_from_prompts.py
```

