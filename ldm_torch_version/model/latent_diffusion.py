import torch
import torch.nn as nn

from model.autoencoder import VAE
from model.unet import UNet
from model.gaussian_diffusion import GaussianDiffusion

class LatentDiffusion(nn.Module):

    def __init__(self,
                 first_stage_model: VAE,
                 unet_model: UNet,
                 sampler: GaussianDiffusion):
        super().__init__(LatentDiffusion)
        self.first_stage_model = first_stage_model
        self.unet_model = unet_model
        self.sampler = sampler

    def encode(self, x):
        mu, std = self.first_stage_model.encode(x)
        return self.first_stage_model.reparameterize(mu, std)

    def decode(self, x):
        return self.first_stage_model.decode(x)

    def q_sample(self, x_0, t):
        return self.sampler.q_sample(x_0,t,None)

    def p_sample(self, model, x, t):
        return self.sampler.p_sample(
            model=model,
            x=x,
            t=t,
            clip_denoised=True  # 将结果限制在[-1, 1]范围内
        )

    




