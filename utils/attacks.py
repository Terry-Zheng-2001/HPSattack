import torch
import torch.nn as nn
from tqdm import tqdm
import hpsv2

class BaseAttack:
    def __init__(self, model: nn.Module, device="cuda"):
        self.model = model
        self.device = device

    def attack(self, images: torch.Tensor, prompt: str):
        raise NotImplementedError("You must implement this method.")


class PGDAttack(BaseAttack):
    def __init__(self, model, eps=4/255, alpha=1/255, steps=1, random_start=True, device="cuda"):
        super().__init__(model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        
        # CLIP mean / std
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def unnormalize(self, x):
        return x * self.std + self.mean

    def normalize(self, x):
        return (x - self.mean) / self.std

    def attack(self, images, prompt):
        # Make sure the original input is in the normalized space
        images = images.clone().detach().to(self.device)
        ori_images = images.clone().detach()
        
        # Convert to unnormalized space [0,1]
        images_unnorm = self.unnormalize(images)
        ori_images_unnorm = images_unnorm.clone()

        if self.random_start:
            # Add random noise in the unnormalized space
            images_unnorm = images_unnorm + torch.empty_like(images_unnorm).uniform_(-self.eps, self.eps) * self.std
            images_unnorm = torch.clamp(images_unnorm, 0, 1).detach()
            # Convert back to normalized space
            images = self.normalize(images_unnorm)

        for _ in range(self.steps):
            images.requires_grad = True
            score = self.model(images, prompt)  # (B,)
            loss = - score.mean()
            grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
            
            # Convert current image to unnormalized space
            images_unnorm = self.unnormalize(images.detach())
            # Apply perturbation in unnormalized space
            images_unnorm = images_unnorm + self.alpha * grad.sign() * self.std
            # Clamp the perturbation
            delta_unnorm = torch.clamp(images_unnorm - ori_images_unnorm, -self.eps * self.std, self.eps * self.std)
            images_unnorm = torch.clamp(ori_images_unnorm + delta_unnorm, 0, 1).detach()
            # Convert back to normalized space
            images = self.normalize(images_unnorm)

        return images


class FGSMAttack(BaseAttack):
    def __init__(self, model, eps=8/255, device="cuda"):
        super().__init__(model, device)
        self.eps = eps

        # CLIP mean / std
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def unnormalize(self, x):
        return x * self.std + self.mean

    def normalize(self, x):
        return (x - self.mean) / self.std

    def attack(self, images, prompt, increase=False):
        """
        increase: if True, increase the score
        """

        # Make sure the original input is in the normalized space
        images = images.clone().detach().to(self.device).requires_grad_(True)

        # Calculate the score and gradient
        score = self.model(images, prompt)
        if increase:
            loss = score.mean()
        else:
            loss = -score.mean()
        grad = torch.autograd.grad(loss, images)[0]

        # 1. Convert the normalized image to [0,1] original image
        images_unnorm = self.unnormalize(images)

        # 2. Do FGSM perturbation in the original image space
        adv_unnorm = images_unnorm + self.eps * grad.sign() * self.std  # Note: multiply std to scale the perturbation

        # 3. Clamp to [0,1]
        adv_unnorm = torch.clamp(adv_unnorm, 0, 1)

        # 4. Normalize the image back to the normalized space
        adv_images = self.normalize(adv_unnorm).detach()

        return adv_images

class NESAttack(BaseAttack):
    def __init__(self, model, eps=4/255, alpha=1/255, steps=1000, nes_samples=50, fd_eps=0.02, device="cuda"):
        super().__init__(model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.nes_samples = nes_samples  # Number of random directions sampled per step
        self.fd_eps = fd_eps

        # CLIP mean / std
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def unnormalize(self, x):
        return x * self.std + self.mean

    def normalize(self, x):
        return (x - self.mean) / self.std

    def attack(self, images: torch.Tensor, prompt: str):
        images = images.clone().detach().to(self.device)
        ori_images = images.clone().detach()

        images_unnorm = self.unnormalize(images)
        ori_images_unnorm = images_unnorm.clone()

        B, C, H, W = images.shape
        assert B == 1, "NESAttack only supports batch size 1."

        for step in tqdm(range(self.steps), desc="NES Attack"):
            images_unnorm.requires_grad = False

            grad_estimate = torch.zeros_like(images_unnorm)

            for _ in range(self.nes_samples):
                noise = torch.randn_like(images_unnorm)
                # noise = noise / (noise.view(noise.size(0), -1).norm(dim=1).view(-1, 1, 1, 1) + 1e-12)  # Normalize noise

                # + epsilon direction
                plus_images = torch.clamp(images_unnorm + self.fd_eps * noise, 0, 1)
                plus_score = self.model.score(self.normalize(plus_images), prompt)

                # - epsilon direction
                minus_images = torch.clamp(images_unnorm - self.fd_eps * noise, 0, 1)
                minus_score = self.model.score(self.normalize(minus_images), prompt)
                print(plus_score - minus_score)
                grad_estimate += (plus_score - minus_score) * noise

            grad_estimate /= (2 * self.fd_eps * self.nes_samples)

            # Gradient descent step
            images_unnorm = images_unnorm - self.alpha * grad_estimate.sign() * self.std

            # Clamp perturbation
            delta_unnorm = torch.clamp(images_unnorm - ori_images_unnorm, -self.eps * self.std, self.eps * self.std)
            images_unnorm = torch.clamp(ori_images_unnorm + delta_unnorm, 0, 1).detach()

        # Final output: normalized back
        adv_images = self.normalize(images_unnorm)
        return adv_images