# --------------------------------------------------------
# Code adapted from:
# RAE: https://github.com/bytetriper/RAE
# --------------------------------------------------------

from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from transformers import AutoConfig, AutoImageProcessor
from typing import Optional
from math import sqrt
from typing import Protocol
from PIL import Image
import torch.nn.functional as F
import os

# !wget https://huggingface.co/nyu-visionx/RAE-collections/resolve/main/stats/dinov2/wReg_base/imagenet1k/stat.pt
# !wget https://huggingface.co/facebook/dinov2-with-registers-base/resolve/main/model.safetensors

class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)

        # self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        self.encoder.requires_grad_(False)
        self.encoder.eval()
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x, output_hidden_states=True)
        unused_token_num = 5  # 1 CLS + 4 register tokens
        image_features = x.last_hidden_state[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)

class RAE(nn.Module):
    def __init__(
        self, 
        # ---- encoder configs ----
        encoder_cls: str = 'Dinov2withNorm',
        encoder_config_path: str = 'facebook/dinov2-with-registers-base',
        encoder_input_size: int = 224,
        encoder_params: dict = {'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
        # ---- noising, reshaping and normalization-----
        noise_tau: float = 0.0,
        reshape_to_2d: bool = True,
        normalization_stat_path: Optional[str] = '/path/to/latentforcing/normalization_stat_path.pt',
        match_pixel_norm: float = 0.485, # Empirically calculated
        eps: float = 1e-5,
    ):
        super().__init__()
        self.encoder = Dinov2withNorm(**encoder_params)
        print(f"encoder_config_path: {encoder_config_path}")
        proc = AutoImageProcessor.from_pretrained(encoder_config_path)
        self.encoder_mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1)
        self.encoder_std = torch.tensor(proc.image_std).view(1, 3, 1, 1)
        encoder_config = AutoConfig.from_pretrained(encoder_config_path)
        # see if the encoder has patch size attribute            
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = self.encoder.patch_size
        self.latent_dim = self.encoder.hidden_size
        assert self.encoder_input_size % self.encoder_patch_size == 0, f"encoder_input_size {self.encoder_input_size} must be divisible by encoder_patch_size {self.encoder_patch_size}"
        self.base_patches = (self.encoder_input_size // self.encoder_patch_size) ** 2 # number of patches of the latent
        
        # noising
        self.noise_tau = noise_tau
        self.reshape_to_2d = reshape_to_2d
        
        if normalization_stat_path is not None:
            if not os.path.exists(normalization_stat_path):
                assert False, "wget RAE stats: wget https://huggingface.co/nyu-visionx/RAE-collections/resolve/main/stats/dinov2/wReg_base/imagenet1k/stat.pt"
            stats = torch.load(normalization_stat_path, map_location='cpu', weights_only=True)
            self.latent_mean = stats.get('mean', None)
            self.latent_var = stats.get('var', None)
            self.do_normalization = True
            self.eps = eps
            print(f"Loaded normalization stats from {normalization_stat_path}")
        else:
            self.do_normalization = False
        self.match_pixel_norm = match_pixel_norm

    def noising(self, x: torch.Tensor) -> torch.Tensor:
        noise_sigma = self.noise_tau * torch.rand((x.size(0),) + (1,) * (len(x.shape) - 1), device=x.device)
        noise = noise_sigma * torch.randn_like(x)
        return x + noise

    @torch.compile()
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input
        _, _, h, w = x.shape
        if h != self.encoder_input_size or w != self.encoder_input_size:
            x = nn.functional.interpolate(x, size=(self.encoder_input_size, self.encoder_input_size), mode='bicubic', align_corners=False)
        x = (x - self.encoder_mean.to(x.device)) / self.encoder_std.to(x.device)
        z = self.encoder(x)
        if self.training and self.noise_tau > 0:
            z = self.noising(z)
        if self.reshape_to_2d:
            b, n, c = z.shape
            h = w = int(sqrt(n))
            z = z.transpose(1, 2).view(b, c, h, w)
        if self.do_normalization:
            latent_mean = self.latent_mean.to(z.device) if self.latent_mean is not None else 0
            latent_var = self.latent_var.to(z.device) if self.latent_var is not None else 1
            z = (z - latent_mean) / torch.sqrt(latent_var + self.eps)
            z = z * self.match_pixel_norm
        return z