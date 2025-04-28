import torch
import open_clip
import torch
import torch.nn as nn
from PIL import Image

# In order to use attack, we need to take prompt as input of init

class HPSv2Module(nn.Module):
    def __init__(self, ckpt_path: str, device="cuda"):
        super().__init__()
        self.device = device

        # Load the OpenCLIP model structure and preprocess method
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device)
        self.model.to(device).eval().half()

        # Load the trained weights
        state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)

        # Text encoder
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def forward(self, image_tensor: torch.Tensor, prompt: str):
        """
        Forward the image tensor and return the score (float)
        """
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)

        # encode image
        img_feat = self.model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # encode prompt
        token = self.tokenizer(prompt).to(self.device)
        with torch.no_grad():
            text_feat = self.model.encode_text(token)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # similarity score
        score = (text_feat @ img_feat.T) * 100  # shape: (1, B)
        return score.squeeze(0)  # shape: (B,)

    def score(self, image_tensor: torch.Tensor, prompt: str):
        scores = self.forward(image_tensor, prompt)
        if scores.numel() == 1:
            return scores.item()
        else:
            return scores.detach().cpu()
