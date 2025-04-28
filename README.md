# HPS v2 Project Template

This project runs inference on Human Preference Score v2 using CLIP ViT-H/14.

## Structure

- `main_infer.py`: Runs basic inference on test image and prompt
- `models/hps_clip.py`: Loads and runs the HPS model
- `utils/select_best_image.py`: Extracts best-ranked image from dataset
- `weights/`: Place your `HPS_v2_compressed.pt` here
- `data/test/`: Place test images here

## Install

```
pip install -r requirements.txt
```

## Run

```
python main_infer.py
```
