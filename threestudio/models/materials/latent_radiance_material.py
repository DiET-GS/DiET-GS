import random
from dataclasses import dataclass, field
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.materials.base import BaseMaterial
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import dot, get_activation
from threestudio.utils.typing import *


@threestudio.register("latent-radiance-material")
class LatentRadianceMaterial(BaseMaterial):
    @dataclass
    class Config(BaseMaterial.Config):
        input_feature_dims: int = 16
        n_latent_dims: int = 4
        n_image_dims: int = 3

        dir_encoding_config: dict = field(
            default_factory=lambda: {"otype": "SphericalHarmonics", "degree": 4}
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )

    cfg: Config

    def configure(self) -> None:
        self.encoding = get_encoding(3, self.cfg.dir_encoding_config)
        self.n_input_dims = self.cfg.input_feature_dims + self.encoding.n_output_dims  # type: ignore
        # self.n_input_dims = self.cfg.input_feature_dims   # type: ignore
        self.network = get_mlp(self.n_input_dims, self.cfg.n_image_dims, self.cfg.mlp_network_config)

        
    def forward(
        self,
        features: Float[Tensor, "*B Nf"],
        viewdirs: Float[Tensor, "*B 3"],
        **kwargs,
    ) -> Float[Tensor, "*B C"]:
        # viewdirs and normals must be normalized before passing to this function
        viewdirs = (viewdirs + 1.0) / 2.0  # (-1, 1) => (0, 1)
        viewdirs_embd = self.encoding(viewdirs.view(-1, 3))
        network_inp = torch.cat([features.view(-1, features.shape[-1]), viewdirs_embd], dim=-1)
        # network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.cfg.n_image_dims).float()

        return color
    

    def get_latents(self,images,guidance,zoom, img_wh_base):
        images = images.permute(0,3,1,2) #[BCHW]
        images = images * 2 - 1.0 # [-1,1]
        wz,hz = images.shape[-2:]
        w,h = img_wh_base

        filename = f'latents_{w}_x{zoom}.pt'
        if os.path.exists(filename):
            latents = torch.load(filename)
            print(f"loaded cached {filename} of shape: {latents.shape}")
            return latents

        # get latents     
        print(f"Encoding images...")
        with torch.no_grad():
            latents = []
            for image in tqdm(images):
                upscaled_image = F.interpolate(image.unsqueeze(0), (int(hz*4),int(wz*4)), mode="bilinear", align_corners=True) #upscaled img (1,3,3200,3200)
                latent = guidance.encode_images(upscaled_image)
                latents.append(latent)
            latents = torch.cat(latents)
        latents = latents.permute(0,2,3,1).reshape(-1,hz*wz,4) #[B HW C]

        torch.save(latents, filename)
        print(f"created latents of shape: {latents.shape}")
        print(f"saved latents to {filename}")

        return latents

