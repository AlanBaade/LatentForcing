## Single file reimplementation of
# https://github.com/facebookresearch/fairseq/blob/main/examples/data2vec/README.md
# Modified from SyllableLM: https://github.com/AlanBaade/SyllableLM

from typing import List, Tuple

import torch
from torch import nn
import math
import torch.nn.functional as F


class Fp32GroupNorm(nn.GroupNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


def norm_block(is_layer_norm, dim, affine=True):
    if is_layer_norm:
        mod = nn.Sequential(
            TransposeLast(),
            Fp32LayerNorm(dim, elementwise_affine=affine),
            TransposeLast(),
        )
    else:
        mod = Fp32GroupNorm(1, dim, affine=affine)

    return mod


class SamePad(nn.Module):
    def __init__(self, kernel_size, causal=False):
        super().__init__()
        if causal:
            self.remove = kernel_size - 1
        else:
            self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


try:
    from apex.normalization import FusedLayerNorm as _FusedLayerNorm

    has_fused_layernorm = True


    class FusedLayerNorm(_FusedLayerNorm):
        @torch.jit.unused
        def forward(self, x):
            if not x.is_cuda:
                return super().forward(x)
            else:
                with torch.cuda.device(x.device):
                    return super().forward(x)
except ImportError:
    has_fused_layernorm = False


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        export = True
    if not export and torch.cuda.is_available() and has_fused_layernorm:
        return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

class SamePad2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        assert len(x.size()) == 4
        if self.remove > 0:
            x = x[:, :, : -self.remove, : -self.remove]
        return x
    

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class D2vDecoderConfig:
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 5
    decoder_layers: int = 5
    input_dropout: float = 0.1

    add_positions_masked: bool = False
    add_positions_all: bool = False

    decoder_residual: bool = True
    projection_layers: int = 1
    projection_ratio: float = 2.0


class FixedPositionalEncoder(nn.Module):
    def __init__(self, pos_embed):
        super().__init__()
        self.positions = pos_embed

    def forward(self, x, padding_mask):
        return self.positions


class TextFeatPositionalEncoder(nn.Module):
    """
    Original encoder expects (B, T) long input. This module wraps it to take
    local_encoder output which are (B, T, D) float tensors
    """

    def __init__(self, pos_encoder):
        super().__init__()
        self.pos_encoder = pos_encoder

    def forward(self, x, padding_mask):
        # assume padded token embeddings are 0s
        # TODO: consider using padding_mask as input
        return self.pos_encoder(x[..., 0])


class BlockEncoder(nn.Module):
    def __init__(self, blocks, norm_layer, layer_norm_first, layerdrop, dropout):
        super().__init__()
        self.blocks = blocks
        self.norm = norm_layer
        self.layer_norm_first = layer_norm_first
        self.layerdrop = layerdrop
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x, padding_mask, alibi_bias, alibi_scale):
        if self.norm is not None and not self.layer_norm_first:
            x = self.norm(x)

        x = self.dropout(x)

        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.layerdrop == 0
                or (np.random.random() > self.layerdrop)
            ):
                ab = alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)
                x, _ = blk(x, padding_mask, ab)

        if self.norm is not None and self.layer_norm_first:
            x = self.norm(x)

        return x


class DecoderBase(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig):
        super().__init__()

        self.decoder_cfg = cfg

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def add_residual(self, x, residual, i, mask_info):
        if (
            residual is None
            or not self.decoder_cfg.decoder_residual
            or residual.size(1) != x.size(1)
        ):
            return x

        ret = x + residual

        return ret


class Decoder1d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim):
        super().__init__(cfg)

        def make_block(in_dim):
            block = [
                nn.Conv1d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad(cfg.decoder_kernel),
                TransposeLast(),
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        projs = []
        curr_dim = cfg.decoder_dim
        for i in range(cfg.projection_layers - 1):
            next_dim = int(curr_dim * cfg.projection_ratio) if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim
        projs.append(nn.Linear(curr_dim, input_dim))
        if len(projs) == 1:
            self.proj = projs[0]
        else:
            self.proj = nn.Sequential(*projs)

    def forward(self, x, mask_info):

        x = x.transpose(1, 2)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


class Decoder2d(DecoderBase):
    def __init__(self, cfg: D2vDecoderConfig, input_dim, h_size, w_size):
        super().__init__(cfg)

        self.h_size = h_size
        self.w_size = w_size

        def make_block(in_dim):
            block = [
                nn.Conv2d(
                    in_dim,
                    cfg.decoder_dim,
                    kernel_size=cfg.decoder_kernel,
                    padding=cfg.decoder_kernel // 2,
                    groups=cfg.decoder_groups,
                ),
                SamePad2d(cfg.decoder_kernel),
                TransposeLast(tranpose_dim=-3),
                LayerNorm(cfg.decoder_dim, elementwise_affine=False),
                TransposeLast(tranpose_dim=-3),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(input_dim if i == 0 else cfg.decoder_dim)
                for i in range(cfg.decoder_layers)
            ]
        )

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def forward(self, x, mask_info):
        B, T, C = x.shape

        x = x.transpose(1, 2).reshape(B, C, self.h_size, self.w_size)

        residual = x

        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = self.add_residual(x, residual, i, mask_info)
            residual = x

        x = x.reshape(B, -1, T).transpose(1, 2)
        x = self.proj(x)
        return x


class TransformerDecoder(nn.Module):
    decoder_cfg: D2vDecoderConfig

    def __init__(self, cfg: D2vDecoderConfig, input_dim, encoder):
        super().__init__()

        self.decoder_cfg = cfg

        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)

        self.encoder = encoder

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        from fairseq.modules.transformer_sentence_encoder import init_bert_params

        self.apply(init_bert_params)

    def forward(self, x, mask_info):
        x = self.input_proj(x)
        x = self.encoder(x, None, None, 1)
        x = self.proj(x)
        return x


class AltBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        ffn_targets=False,
        cosine_attention=False,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first
        self.ffn_targets = ffn_targets

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(dim)
        self.attn = AltAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)

    def forward(self, x, padding_mask=None, alibi_bias=None):
        if self.layer_norm_first:
            x = x + self.drop_path(self.attn(self.norm1(x), padding_mask, alibi_bias))
            r = x = self.mlp(self.norm2(x))  # LatentForcing Authors: Lol og d2v2 is bugged
            t = x
            x = r + self.drop_path(self.post_mlp_dropout(x))
            if not self.ffn_targets:
                t = x
        else:
            x = x + self.drop_path(self.attn(x, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            t = x
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))
            if not self.ffn_targets:
                t = x

        return x, t


class AltAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cosine_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

    def forward(self, x, padding_mask=None, alibi_bias=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # qkv x B x H x L x D
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        if self.cosine_attention:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(
                self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
            ).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias

        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)  #
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncDecAttention(nn.Module):
    def __init__(
        self,
        q_dim,
        kv_dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cosine_attention=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = q_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, q_dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(kv_dim, 2 * q_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(q_dim, q_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.cosine_attention = cosine_attention

        if cosine_attention:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True
            )

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        B, N, C = q.shape

        q = (
            self.q_proj(q)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )  # B x H x L x D
        kv = (
            self.kv_proj(kv)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # kv x B x H x L x D
        k, v = (
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        dtype = q.dtype

        if self.cosine_attention:
            # cosine attention
            attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
            logit_scale = torch.clamp(
                self.logit_scale, max=torch.log(torch.tensor(1.0 / 0.01))
            ).exp()
            attn = attn * logit_scale
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

        if alibi_bias is not None:
            attn = attn.type_as(alibi_bias)
            attn[:, : alibi_bias.size(1)] += alibi_bias

        if padding_mask is not None and padding_mask.any():
            attn = attn.masked_fill(
                padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )

        attn = attn.softmax(dim=-1, dtype=torch.float32).to(dtype=dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2)  #
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EncDecBlock(nn.Module):
    def __init__(
        self,
        q_dim,
        kv_dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        mlp_drop=0.0,
        post_mlp_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        layer_norm_first=True,
        cosine_attention=False,
        first_residual=True,
    ):
        super().__init__()

        self.layer_norm_first = layer_norm_first

        from timm.models.vision_transformer import DropPath, Mlp

        self.norm1 = norm_layer(q_dim)
        self.attn = EncDecAttention(
            q_dim,
            kv_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            cosine_attention=cosine_attention,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(q_dim)
        mlp_hidden_dim = int(q_dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=q_dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )
        self.post_mlp_dropout = nn.Dropout(post_mlp_drop, inplace=False)
        self.first_residual = first_residual

    def forward(self, q, kv, padding_mask=None, alibi_bias=None):
        r = q if self.first_residual else 0
        if self.layer_norm_first:
            x = r + self.drop_path(
                self.attn(self.norm1(q), kv, padding_mask, alibi_bias)
            )
            r = x = self.mlp(self.norm2(x))
            x = r + self.drop_path(self.post_mlp_dropout(x))
        else:
            x = r + self.drop_path(self.attn(q, kv, padding_mask, alibi_bias))
            r = x = self.norm1(x)
            x = self.mlp(x)
            x = self.norm2(r + self.drop_path(self.post_mlp_dropout(x)))

        return x


class EncDecTransformerDecoder(nn.Module):
    def __init__(self, cfg: D2vDecoderConfig, input_dim):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, cfg.decoder_dim)

        self.blocks = nn.Sequential(
            *[
                EncDecBlock(
                    q_dim=cfg.decoder_dim,
                    kv_dim=input_dim,
                    num_heads=8,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    qk_scale=None,
                    drop=0.0,
                    attn_drop=0.0,
                    mlp_drop=0.0,
                    post_mlp_drop=0.0,
                    drop_path=0.0,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    layer_norm_first=False,
                    cosine_attention=False,
                    first_residual=i > 0,
                )
                for i in range(cfg.decoder_layers)
            ]
        )

        self.proj = nn.Linear(cfg.decoder_dim, input_dim)

    def reset_parameters(self):
        from fairseq.modules.transformer_sentence_encoder import init_bert_params

        self.apply(init_bert_params)

    def forward(self, x, kv):
        x = self.input_proj(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x, kv)

        x = self.proj(x)
        return x
    
from enum import Enum, auto

class Modality(Enum):
    AUDIO = auto()
    IMAGE = auto()
    TEXT = auto()


@dataclass
class D2vDecoderConfig:
    decoder_dim: int = 384
    decoder_groups: int = 16
    decoder_kernel: int = 5
    decoder_layers: int = 5
    input_dropout: float = 0.1

    add_positions_masked: bool = False
    add_positions_all: bool = False

    decoder_residual: bool = True
    projection_layers: int = 1
    projection_ratio: float = 2.0

    channel_mult: object = (1, 0.5, 0.25, 0.25, 0.25)  # tuple[float]
    decoder_transformer_layers: int = 4


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from omegaconf import MISSING, II
from typing import Optional, Callable
# from fairseq.data.data_utils import compute_mask_indices
# from fairseq.modules import GradMultiply
# from fairseq.utils import index_put

logger = logging.getLogger(__name__)


@dataclass
class D2vModalityConfig:
    type: Modality = MISSING
    prenet_depth: int = 4
    prenet_layerdrop: float = 0
    prenet_dropout: float = 0
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0

    num_extra_tokens: int = 0
    init_extra_token_zero: bool = True

    mask_noise_std: float = 0.01
    mask_prob_min: Optional[float] = None
    mask_prob: float = 0.7
    inverse_mask: bool = False
    mask_prob_adjust: float = 0
    keep_masked_pct: float = 0

    mask_length: int = 5
    add_masks: bool = False
    remove_masks: bool = False
    mask_dropout: float = 0.0
    encoder_zero_mask: bool = True

    mask_channel_prob: float = 0.0
    mask_channel_length: int = 64

    ema_local_encoder: bool = False  # used in data2vec_multi
    local_grad_mult: float = 1.0

    use_alibi_encoder: bool = False
    alibi_scale: float = 1.0
    learned_alibi: bool = False
    alibi_max_pos: Optional[int] = None
    learned_alibi_scale: bool = False
    learned_alibi_scale_per_head: bool = False
    learned_alibi_scale_per_layer: bool = False

    num_alibi_heads: int = II("model.num_heads")
    model_depth: int = II("model.depth")

    decoder: Optional[D2vDecoderConfig] = D2vDecoderConfig()


MaskSeed = namedtuple("MaskSeed", ["seed", "update", "ids"])
MaskInfo = namedtuple("MaskInfo", ["x_unmasked", "mask", "ids_restore", "ids_keep"])


class ModalitySpecificEncoder(nn.Module):
    def __init__(
            self,
            modality_cfg: D2vModalityConfig,
            embed_dim: int,
            local_encoder: nn.Module,
            project_features: nn.Module,
            fixed_positional_encoder: Optional[nn.Module],
            relative_positional_encoder: Optional[nn.Module],
            context_encoder: nn.Module,
            decoder: nn.Module,
            get_alibi_bias: Optional[Callable[[int, int, str, str], torch.Tensor]],
    ):
        super().__init__()

        self.modality_cfg = modality_cfg
        self.local_encoder = local_encoder
        self.project_features = project_features
        self.fixed_positional_encoder = fixed_positional_encoder
        self.relative_positional_encoder = relative_positional_encoder
        self.context_encoder = context_encoder

        self.decoder = decoder
        self.get_alibi_bias = get_alibi_bias if modality_cfg.use_alibi_encoder else None

        self.local_grad_mult = self.modality_cfg.local_grad_mult

        self.extra_tokens = None
        if modality_cfg.num_extra_tokens > 0:
            self.extra_tokens = nn.Parameter(
                torch.zeros(1, modality_cfg.num_extra_tokens, embed_dim)
            )
            if not modality_cfg.init_extra_token_zero:
                nn.init.normal_(self.extra_tokens)
            elif self.extra_tokens.size(1) > 1:
                nn.init.normal_(self.extra_tokens[:, 1:])

        self.alibi_scale = None
        if self.get_alibi_bias is not None:
            self.alibi_scale = nn.Parameter(
                torch.full(
                    (
                        (modality_cfg.prenet_depth + modality_cfg.model_depth)
                        if modality_cfg.learned_alibi_scale_per_layer
                        else 1,
                        1,
                        self.modality_cfg.num_alibi_heads
                        if modality_cfg.learned_alibi_scale_per_head
                        else 1,
                        1,
                        1,
                    ),
                    modality_cfg.alibi_scale,
                    dtype=torch.float,
                ),
                requires_grad=modality_cfg.learned_alibi_scale,
            )

        if modality_cfg.learned_alibi and self.get_alibi_bias is not None:
            assert modality_cfg.alibi_max_pos is not None
            alibi_bias = self.get_alibi_bias(
                batch_size=1,
                time_steps=modality_cfg.alibi_max_pos,
                heads=modality_cfg.num_alibi_heads,
                scale=1.0,
                dtype=torch.float,
                device="cpu",
            )
            self.alibi_bias = nn.Parameter(alibi_bias)
            self.get_alibi_bias = partial(
                _learned_alibi_bias, alibi_bias=self.alibi_bias
            )

    def upgrade_state_dict_named(self, state_dict, name):
        k = f"{name}.alibi_scale"
        if k in state_dict and state_dict[k].dim() == 4:
            state_dict[k] = state_dict[k].unsqueeze(0)

        return state_dict

    def convert_padding_mask(self, x, padding_mask):
        return padding_mask

    def decoder_input(self, x, mask_info: MaskInfo):
        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        num_extra = self.modality_cfg.num_extra_tokens

        if mask_info is not None:
            num_masked = mask_info.ids_restore.shape[1] - x.shape[1] + num_extra

            mask_tokens = x.new_empty(
                x.size(0),
                num_masked,
                x.size(-1),
            ).normal_(0, self.modality_cfg.mask_noise_std)

            x_ = torch.cat([x[:, num_extra:], mask_tokens], dim=1)
            x = torch.gather(x_, dim=1, index=mask_info.ids_restore)

            if self.modality_cfg.decoder.add_positions_masked:
                assert self.fixed_positional_encoder is not None
                pos = self.fixed_positional_encoder(x, None)
                x = x + (pos * mask_info.mask.unsqueeze(-1))
        else:
            x = x[:, num_extra:]

        if self.modality_cfg.decoder.add_positions_all:
            assert self.fixed_positional_encoder is not None
            x = x + self.fixed_positional_encoder(x, None)

        return x, mask_info

    def local_features(self, features):
        if self.local_grad_mult > 0:
            if self.local_grad_mult == 1.0:
                x = self.local_encoder(features)
            else:
                x = GradMultiply.apply(
                    self.local_encoder(features), self.local_grad_mult
                )
        else:
            with torch.no_grad():
                x = self.local_encoder(features)

        x = self.project_features(x)
        return x

    def contextualized_features(
            self,
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch: int = 1,
            mask_seeds: Optional[torch.Tensor] = None,
            precomputed_mask=None,
    ):

        if padding_mask is not None:
            padding_mask = self.convert_padding_mask(x, padding_mask)

        local_features = x
        if mask and clone_batch == 1:
            local_features = local_features.clone()

        orig_B, orig_T, _ = x.shape
        pre_mask_B = orig_B
        mask_info = None

        x_pos = None
        if self.fixed_positional_encoder is not None:
            x = x + self.fixed_positional_encoder(x, padding_mask)

        if mask:
            if clone_batch > 1:
                x = x.repeat_interleave(clone_batch, 0)
                if mask_seeds is not None:
                    clone_hash = [
                        int(hash((mask_seeds.seed, ind)) % 1e10)
                        for ind in range(clone_batch - 1)
                    ]
                    clone_hash = torch.tensor([0] + clone_hash).long().view(1, -1)

                    id = mask_seeds.ids
                    id = id.repeat_interleave(clone_batch, 0)
                    id = id.view(-1, clone_batch) + clone_hash.to(id)
                    id = id.view(-1)
                    mask_seeds = MaskSeed(
                        seed=mask_seeds.seed, update=mask_seeds.update, ids=id
                    )
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat_interleave(clone_batch, 0)

            x, mask_info = self.compute_mask(
                x,
                padding_mask,
                mask_seed=mask_seeds,
                apply=self.relative_positional_encoder is not None or not remove_masked,
                precomputed_mask=precomputed_mask,
            )

        if self.relative_positional_encoder is not None:
            x_pos = self.relative_positional_encoder(x)

        masked_padding_mask = padding_mask
        if mask and remove_masked:
            x = mask_info.x_unmasked
            if x_pos is not None:
                x = x + gather_unmasked(x_pos, mask_info)

            if padding_mask is not None and padding_mask.any():
                masked_padding_mask = gather_unmasked_mask(padding_mask, mask_info)
                if not masked_padding_mask.any():
                    masked_padding_mask = None
            else:
                masked_padding_mask = None

        elif x_pos is not None:
            x = x + x_pos

        alibi_bias = orig_alibi_bias = None
        alibi_scale = self.alibi_scale

        if self.get_alibi_bias is not None:
            orig_alibi_bias = alibi_bias = self.get_alibi_bias(
                batch_size=pre_mask_B,
                time_steps=orig_T,
                heads=self.modality_cfg.num_alibi_heads,
                dtype=torch.float32,
                device=x.device,
            )

            if alibi_scale is not None:
                alibi_scale = alibi_scale.clamp_min(0)
                if alibi_scale.size(0) == 1:
                    alibi_bias = alibi_bias * alibi_scale.squeeze(0).type_as(alibi_bias)
                    alibi_scale = None

            if clone_batch > 1:
                alibi_bias = alibi_bias.repeat_interleave(clone_batch, 0)

            if mask_info is not None and remove_masked:
                alibi_bias = masked_alibi(alibi_bias, mask_info)

        if self.extra_tokens is not None:
            num = self.extra_tokens.size(1)
            x = torch.cat([self.extra_tokens.expand(x.size(0), -1, -1), x], dim=1)
            if masked_padding_mask is not None:
                # B x T
                masked_padding_mask = F.pad(masked_padding_mask, (num, 0))
            if alibi_bias is not None:
                # B x H x T x T
                alibi_bias = F.pad(alibi_bias, (num, 0, num, 0))

        x = self.context_encoder(
            x,
            masked_padding_mask,
            alibi_bias,
            alibi_scale[: self.modality_cfg.prenet_depth]
            if alibi_scale is not None
            else None,
        )

        return {
            "x": x,
            "local_features": local_features,
            "padding_mask": masked_padding_mask,
            "alibi_bias": alibi_bias,
            "orig_alibi_bias": orig_alibi_bias,
            "alibi_scale": alibi_scale[self.modality_cfg.prenet_depth:]
            if alibi_scale is not None and alibi_scale.size(0) > 1
            else alibi_scale,
            "encoder_mask": mask_info,
        }

    def forward(
            self,
            features,
            padding_mask,
            mask: bool,
            remove_masked: bool,
            clone_batch: int = 1,
            mask_seeds: Optional[torch.Tensor] = None,
            precomputed_mask=None,
    ):
        x = self.local_features(features)
        return self.contextualized_features(
            x,
            padding_mask,
            mask,
            remove_masked,
            clone_batch,
            mask_seeds,
            precomputed_mask,
        )

    def reset_parameters(self):
        pass

    def compute_mask(
            self,
            x,
            padding_mask,
            mask_seed: Optional[MaskSeed],
            apply,
            precomputed_mask,
    ):
        if precomputed_mask is not None:
            mask = precomputed_mask
            mask_info = self.make_maskinfo(x, mask)
        else:
            B, T, C = x.shape
            cfg = self.modality_cfg

            mask_prob = cfg.mask_prob

            if (
                    cfg.mask_prob_min is not None
                    and cfg.mask_prob_min >= 0
                    and cfg.mask_prob_min < mask_prob
            ):
                mask_prob = np.random.uniform(cfg.mask_prob_min, mask_prob)

            if mask_prob > 0:
                if cfg.mask_length == 1:
                    mask_info = random_masking(x, mask_prob, mask_seed)
                else:
                    if self.modality_cfg.inverse_mask:
                        mask_prob = 1 - mask_prob

                    mask = compute_mask_indices(
                        (B, T),
                        padding_mask,
                        mask_prob,
                        cfg.mask_length,
                        min_masks=1,
                        require_same_masks=True,
                        mask_dropout=cfg.mask_dropout,
                        add_masks=cfg.add_masks,
                        seed=mask_seed.seed if mask_seed is not None else None,
                        epoch=mask_seed.update if mask_seed is not None else None,
                        indices=mask_seed.ids if mask_seed is not None else None,
                    )

                    mask = torch.from_numpy(mask).to(device=x.device)
                    if self.modality_cfg.inverse_mask:
                        mask = 1 - mask
                    mask_info = self.make_maskinfo(x, mask)
            else:
                mask_info = None

        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def make_maskinfo(self, x, mask, shape=None):
        if shape is None:
            B, T, D = x.shape
        else:
            B, T, D = shape

        mask = mask.to(torch.uint8)
        ids_shuffle = mask.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1).unsqueeze(-1).expand(-1, -1, D)

        len_keep = T - mask[0].sum()
        if self.modality_cfg.keep_masked_pct > 0:
            len_keep += round((T - int(len_keep)) * self.modality_cfg.keep_masked_pct)

        ids_keep = ids_shuffle[:, :len_keep]

        mask = mask.new_zeros(mask.shape)  # Alan addition, mask should update to represent new kept
        mask.scatter_(index=ids_shuffle[:, len_keep:], dim=1, value=1)

        if shape is not None:
            x_unmasked = None
        else:
            ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
            x_unmasked = torch.gather(x, dim=1, index=ids_keep)

        mask_info = MaskInfo(
            x_unmasked=x_unmasked,
            mask=mask,
            ids_restore=ids_restore,
            ids_keep=ids_keep,
        )
        return mask_info

    def apply_mask(self, x, mask_info):
        cfg = self.modality_cfg
        B, T, C = x.shape

        if mask_info is not None:
            mask = mask_info.mask
            if cfg.encoder_zero_mask:
                x = x * (1 - mask.type_as(x).unsqueeze(-1))
            else:
                num_masks = mask.sum().item()
                masks = x.new_empty(num_masks, x.size(-1)).normal_(
                    0, cfg.mask_noise_std
                )
                x = index_put(x, mask, masks)
        if cfg.mask_channel_prob > 0:
            mask_channel = compute_mask_indices(
                (B, C),
                None,
                cfg.mask_channel_prob,
                cfg.mask_channel_length,
            )
            mask_channel = (
                torch.from_numpy(mask_channel)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
            )
            x = index_put(x, mask_channel, 0)
        return x

    def remove_pretraining_modules(self, keep_decoder=False):
        if not keep_decoder:
            self.decoder = None


def get_annealed_rate(start, end, curr_step, total_steps):
    if curr_step >= total_steps:
        return end
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


# adapted from MAE
def random_masking(x, mask_ratio, mask_seed: Optional[MaskSeed]):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    generator = None
    if mask_seed is not None:
        seed = int(
            hash((mask_seed.seed, mask_seed.update, mask_seed.ids.sum().item())) % 1e6
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)

    noise = torch.rand(N, L, generator=generator, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = noise.argsort(dim=1)  # ascend: small is keep, large is remove
    ids_restore = ids_shuffle.argsort(dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_keep = ids_keep.unsqueeze(-1).expand(-1, -1, D)
    x_unmasked = torch.gather(x, dim=1, index=ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], dtype=x.dtype, device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    ids_restore = ids_restore.unsqueeze(-1).expand(-1, -1, D)

    return MaskInfo(
        x_unmasked=x_unmasked, mask=mask, ids_restore=ids_restore, ids_keep=ids_keep
    )


def gather_unmasked(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep,
    )


def gather_unmasked_mask(x: torch.Tensor, mask_info: MaskInfo) -> torch.Tensor:
    return torch.gather(
        x,
        dim=1,
        index=mask_info.ids_keep[..., 0],  # ignore the feature dimension
    )


def get_alibi(
        max_positions: int,
        attention_heads: int,
        dims: int = 1,
        distance: str = "manhattan",
):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some
        # a. This function has some good properties that only occur when
        # the input is a power of 2. To maintain that even when the number
        # of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    maxpos = max_positions
    attn_heads = attention_heads
    slopes = torch.Tensor(get_slopes(attn_heads))

    if dims == 1:
        # prepare alibi position linear bias. Note that wav2vec2 is non
        # autoregressive model so we want a symmetric mask with 0 on the
        # diagonal and other wise linear decreasing valuees
        pos_bias = (
                torch.abs(
                    torch.arange(maxpos).unsqueeze(0) - torch.arange(maxpos).unsqueeze(1)
                )
                * -1
        )
    elif dims == 2:
        if distance == "manhattan":
            df = lambda x1, y1, x2, y2: abs(x1 - x2) + abs(y1 - y2)
        elif distance == "euclidean":
            df = lambda x1, y1, x2, y2: math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

        n = math.sqrt(max_positions)
        assert n.is_integer(), n
        n = int(n)

        pos_bias = torch.zeros((max_positions, max_positions))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        new_x = i * n + j
                        new_y = k * n + l
                        pos_bias[new_x, new_y] = -df(i, j, k, l)

    else:
        raise Exception(f"unsupported number of alibi dims: {dims}")

    alibi_bias = slopes.unsqueeze(1).unsqueeze(1) * pos_bias.unsqueeze(0).expand(
        attn_heads, -1, -1
    )

    return alibi_bias


def get_alibi_bias(
        alibi_biases,
        batch_size,
        time_steps,
        heads,
        dtype,
        device,
        dims=1,
        distance="manhattan",
):
    cache_key = f"{dims}_{heads}_{distance}"

    buffered = alibi_biases.get(cache_key, None)

    target_size = heads * batch_size
    if (
            buffered is None
            or buffered.size(0) < target_size
            or buffered.size(1) < time_steps
            or buffered.dtype != dtype
            or buffered.device != device
    ):
        bt = max(time_steps, buffered.size(1) if buffered is not None else 0)
        bn = max(target_size, buffered.size(0) if buffered is not None else 0) // heads

        buffered = (
            get_alibi(bt, heads, dims=dims, distance=distance)
                .to(dtype=dtype, device=device)
                .repeat(bn, 1, 1)
        )

        alibi_biases[cache_key] = buffered

    b = buffered[:target_size, :time_steps, :time_steps]
    b = b.view(batch_size, heads, time_steps, time_steps)
    return b


def _learned_alibi_bias(
        alibi_bias,
        batch_size,
        time_steps,
        heads,
        scale,
        dtype,
        device,
):
    assert alibi_bias.size(1) == heads, alibi_bias.shape
    assert alibi_bias.dtype == dtype, alibi_bias.dtype
    assert alibi_bias.device == device, alibi_bias.device

    if alibi_bias.size(-1) < time_steps:
        psz = math.ceil((time_steps - alibi_bias.size(-1)) / 2)
        alibi_bias = F.pad(alibi_bias, (psz, psz, psz, psz), mode="replicate")

    alibi_bias = alibi_bias.expand(batch_size, -1, -1, -1) * scale
    return alibi_bias[..., :time_steps, :time_steps]


def masked_alibi(alibi_bias, mask_info):
    H = alibi_bias.size(1)

    orig_bias = alibi_bias

    index = mask_info.ids_keep.unsqueeze(1)[..., 0].unsqueeze(-1)
    alibi_bias = torch.gather(
        orig_bias,
        dim=-2,
        index=index.expand(-1, H, -1, mask_info.ids_restore.size(1)),
    )
    alibi_bias = torch.gather(
        alibi_bias,
        dim=-1,
        index=index.transpose(-1, -2).expand(-1, H, alibi_bias.size(-2), -1),
    )

    return alibi_bias


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from dataclasses import dataclass
from typing import Callable, Dict, Optional

def to_2tuple(x):
    return (x,x)

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x




def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

@dataclass
class D2vImageConfig(D2vModalityConfig):
    type: Modality = Modality.IMAGE

    input_size: int = 224
    in_chans: int = 3
    patch_size: int = 16
    embed_dim: int = 768

    alibi_dims: int = 2
    alibi_distance: str = "manhattan"

    fixed_positions: bool = True

    transformer_decoder: bool = False
    enc_dec_transformer: bool = False


class ImageEncoder(ModalitySpecificEncoder):

    modality_cfg: D2vImageConfig

    def __init__(
        self,
        modality_cfg: D2vImageConfig,
        embed_dim: int,
        make_block: Callable[[float, Optional[int], Optional[int]], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases: Dict,
        task, #Optional[FairseqTask],
    ):

        img_size = to_2tuple(modality_cfg.input_size)
        patch_size = to_2tuple(modality_cfg.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        local_encoder = PatchEmbed(
            modality_cfg.input_size,
            modality_cfg.patch_size,
            modality_cfg.in_chans,
            modality_cfg.embed_dim,
        )

        w = local_encoder.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        if modality_cfg.embed_dim != embed_dim:
            local_encoder = nn.Sequential(
                local_encoder,
                nn.Linear(modality_cfg.embed_dim, embed_dim),
            )

        project_features = nn.Identity()

        pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )

        side_n = int(num_patches ** 0.5)

        emb = get_2d_sincos_pos_embed(
            pos_embed.shape[-1],
            side_n,
            cls_token=False,
        )
        pos_embed.data.copy_(torch.from_numpy(emb).float().unsqueeze(0))
        fixed_positional_encoder = (
            FixedPositionalEncoder(pos_embed) if modality_cfg.fixed_positions else None
        )

        dpr = np.linspace(
            modality_cfg.start_drop_path_rate,
            modality_cfg.end_drop_path_rate,
            modality_cfg.prenet_depth,
        )

        context_encoder = BlockEncoder(
            nn.ModuleList(make_block(dpr[i]) for i in range(modality_cfg.prenet_depth)),
            norm_layer(embed_dim) if not layer_norm_first else None,
            layer_norm_first,
            modality_cfg.prenet_layerdrop,
            modality_cfg.prenet_dropout,
        )

        if modality_cfg.transformer_decoder:
            if modality_cfg.enc_dec_transformer:
                decoder = EncDecTransformerDecoder(modality_cfg.decoder, embed_dim)
            else:
                dec_enc = BlockEncoder(
                    nn.ModuleList(
                        make_block(0, modality_cfg.decoder.decoder_dim, 8)
                        for _ in range(modality_cfg.decoder.decoder_layers)
                    ),
                    None,
                    layer_norm_first,
                    0,
                    0,
                )
                decoder = TransformerDecoder(modality_cfg.decoder, embed_dim, dec_enc)
        else:
            decoder = (
                Decoder2d(modality_cfg.decoder, embed_dim, side_n, side_n)
                if modality_cfg.decoder is not None
                else None
            )

        alibi_bias_fn = partial(
            get_alibi_bias,
            alibi_biases=alibi_biases,
            heads=modality_cfg.num_alibi_heads,
            dims=modality_cfg.alibi_dims,
            distance=modality_cfg.alibi_distance,
        )

        super().__init__(
            modality_cfg=modality_cfg,
            embed_dim=embed_dim,
            local_encoder=local_encoder,
            project_features=project_features,
            fixed_positional_encoder=fixed_positional_encoder,
            relative_positional_encoder=None,
            context_encoder=context_encoder,
            decoder=decoder,
            get_alibi_bias=alibi_bias_fn,
        )

    def reset_parameters(self):
        super().reset_parameters()
        if self.decoder is not None:
            self.decoder.reset_parameters()

    @torch.no_grad()
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.modality_cfg.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    @torch.no_grad()
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.modality_cfg.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def compute_mask(
        self,
        x,
        padding_mask,
        mask_seed: Optional[MaskSeed],
        apply,
        shape=None,
        precomputed_mask=None,
    ):
        mlen = self.modality_cfg.mask_length
        if mlen <= 1:
            return super().compute_mask(
                x, padding_mask, mask_seed, apply, precomputed_mask
            )

        if precomputed_mask is not None:
            mask = precomputed_mask
        else:
            from fairseq.data.data_utils import compute_block_mask_2d

            if shape is not None:
                B, L, D = shape
            else:
                B, L, D = x.shape

            mask = compute_block_mask_2d(
                shape=(B, L),
                mask_prob=self.modality_cfg.mask_prob,
                mask_length=self.modality_cfg.mask_length,
                mask_prob_adjust=self.modality_cfg.mask_prob_adjust,
                inverse_mask=self.modality_cfg.inverse_mask,
                require_same_masks=True,
                mask_dropout=self.modality_cfg.mask_dropout,
            )

        mask_info = self.make_maskinfo(x, mask, shape)
        if apply:
            x = self.apply_mask(x, mask_info)

        return x, mask_info

    def decoder_input(self, x, mask_info):
        if (
            not self.modality_cfg.transformer_decoder
            or not self.modality_cfg.enc_dec_transformer
        ):
            return super().decoder_input(x, mask_info)

        inp_drop = self.modality_cfg.decoder.input_dropout
        if inp_drop > 0:
            x = F.dropout(x, inp_drop, training=self.training, inplace=True)

        kv = x[:, self.modality_cfg.num_extra_tokens :]

        assert self.fixed_positional_encoder is not None
        pos = self.fixed_positional_encoder(x, None).expand(x.size(0), -1, -1)

        mask = mask_info.mask.bool()
        if self.modality_cfg.decoder.add_positions_all:
            kv = kv + pos[~mask].view(kv.shape)

        q = pos[mask].view(x.size(0), -1, x.size(-1))

        return q, kv
    

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable
from functools import partial
import numpy as np

from omegaconf import II

import torch
import torch.nn as nn

# from syllablelm.data2vec.data.modality import Modality

# from syllablelm.data2vec.models.modalities.base import (
#     MaskSeed,
#     D2vModalityConfig,
#     ModalitySpecificEncoder,
#     get_annealed_rate,
# )
# from syllablelm.data2vec.models.modalities.modules import (
#     D2vDecoderConfig,
#     AltBlock,
# )


# from syllablelm.data2vec.models.modalities.audio import (
#     D2vAudioConfig,
#     AudioEncoder,
# )


@dataclass
class D2vModalitiesConfig:
    image: D2vImageConfig = D2vImageConfig()


@dataclass
class Data2VecMultiConfig:

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )

    depth: int = 8
    start_drop_path_rate: float = 0
    end_drop_path_rate: float = 0
    num_heads: int = 12
    norm_eps: float = 1e-6
    norm_affine: bool = True
    encoder_dropout: float = 0.1
    post_mlp_drop: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    dropout_input: float = 0.0
    layerdrop: float = 0.0
    embed_dim: int = 768
    mlp_ratio: float = 4
    layer_norm_first: bool = False

    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    end_of_block_targets: bool = False

    clone_batch: int = 1

    layer_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_same_dtype: bool = True
    log_norms: bool = True
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_encoder_only: bool = field(
        default=True,
        metadata={
            "help": "whether to momentum update only the shared transformer encoder"
        },
    )

    max_update: int = II("optimization.max_update")

    modalities: D2vModalitiesConfig = D2vModalitiesConfig()

    shared_decoder: Optional[D2vDecoderConfig] = None

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    supported_modality: Optional[Modality] = None
    mae_init: bool = False

    seed: int = II("common.seed")

    skip_ema: bool = False

    cls_loss: float = 0
    recon_loss: float = 0
    d2v_loss: float = 1

    decoder_group: bool = False


class Data2VecMultiModel(nn.Module):
    def make_modality_encoder(
        self,
        cfg: D2vModalityConfig,
        embed_dim: int,
        make_block: Callable[[float], nn.ModuleList],
        norm_layer: Callable[[int], nn.LayerNorm],
        layer_norm_first: bool,
        alibi_biases,
        task,
    ) -> ModalitySpecificEncoder:
        # if cfg.type == Modality.AUDIO:
        # enc_cls = AudioEncoder
        # elif cfg.type == Modality.IMAGE:
        enc_cls = ImageEncoder
        # elif cfg.type == Modality.TEXT:
        #     enc_cls = TextEncoder
        #     if hasattr(task, "text_task"):
        #         task = task.text_task
        # else:
        #     raise Exception(f"unsupported modality {cfg.type}")

        return enc_cls(
            cfg,
            embed_dim,
            make_block,
            norm_layer,
            layer_norm_first,
            alibi_biases,
            task,
        )

    def __init__(self, cfg: Data2VecMultiConfig, modalities, skip_ema=False, task=None):
        super().__init__()
        self.cfg = cfg
        self.modalities = modalities
        self.task = task

        make_layer_norm = partial(
            nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.modality_encoders = nn.ModuleDict()
        for mod in self.modalities:
            mod_cfg = getattr(cfg.modalities, mod.name.lower())
            enc = self.make_modality_encoder(
                mod_cfg,
                cfg.embed_dim,
                make_block,
                make_layer_norm,
                cfg.layer_norm_first,
                self.alibi_biases,
                task,
            )
            self.modality_encoders[mod.name] = enc

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth)

        self.blocks = nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        # if self.cfg.mae_init:
        #     self.apply(self._init_weights)
        # else:
        #     from fairseq.modules.transformer_sentence_encoder import init_bert_params
        #
        #     self.apply(init_bert_params)

        # for mod_enc in self.modality_encoders.values():
        #     mod_enc.reset_parameters()

        # if not skip_ema:
        #     self.ema = self.make_ema_teacher(cfg.ema_decay)
        #     self.shared_decoder = (
        #         Decoder1d(cfg.shared_decoder, cfg.embed_dim)
        #         if self.cfg.shared_decoder is not None
        #         else None
        #     )
        #     if self.shared_decoder is not None:
        #         self.shared_decoder.apply(self._init_weights)
        #
        #     self.recon_proj = None
        #     if cfg.recon_loss > 0:
        #         self.recon_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        for pn, p in self.named_parameters():
            if len(p.shape) == 1 or pn.endswith(".bias") or "alibi_scale" in pn:
                p.optim_overrides = {"optimizer": {"weight_decay_scale": 0}}
            if cfg.decoder_group and "decoder" in pn:
                p.param_group = "decoder"

        self.num_updates = 0

    def _init_weights(self, m):

        try:
            from apex.normalization import FusedLayerNorm

            fn = FusedLayerNorm
        except:
            fn = nn.LayerNorm

        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, fn):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # @torch.no_grad()
    # def make_ema_teacher(self, ema_decay):
    #     ema_config = EMAModuleConfig(
    #         ema_decay=ema_decay,
    #         ema_fp32=True,
    #         log_norms=self.cfg.log_norms,
    #         add_missing_params=False,
    #     )
    #
    #     model_copy = self.make_target_model()
    #
    #     return EMAModule(
    #         model_copy,
    #         ema_config,
    #         copy_model=False,
    #     )

    def make_target_model(self):
        logger.info("making target model")

        model_copy = Data2VecMultiModel(
            self.cfg, self.modalities, skip_ema=True, task=self.task
        )

        if self.cfg.ema_encoder_only:
            model_copy = model_copy.blocks
            for p_s, p_t in zip(self.blocks.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)
        else:
            for p_s, p_t in zip(self.parameters(), model_copy.parameters()):
                p_t.data.copy_(p_s.data)

            for mod_enc in model_copy.modality_encoders.values():
                mod_enc.decoder = None
                if not mod_enc.modality_cfg.ema_local_encoder:
                    mod_enc.local_encoder = None
                    mod_enc.project_features = None

        model_copy.requires_grad_(False)
        return model_copy

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is not None and (
            (self.num_updates == 0 and num_updates > 1)
            or self.num_updates >= num_updates
        ):
            pass
        elif self.training and self.ema is not None:
            ema_weight_decay = None
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay, weight_decay=ema_weight_decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.blocks if self.cfg.ema_encoder_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        k = prefix + "_ema"
        if self.ema is not None:
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        elif k in state_dict:
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecMultiConfig, task=None):
        """Build a new model instance."""
        if task is None or not hasattr(task, "supported_modalities"):
            modalities = (
                [cfg.supported_modality]
                if cfg.supported_modality is not None
                else [
                    Modality.AUDIO,
                    Modality.IMAGE,
                    Modality.TEXT,
                ]
            )
        else:
            modalities = task.supported_modalities
        return cls(cfg, modalities, task=task, skip_ema=cfg.skip_ema)

    def forward(
        self,
        source,
        target=None,
        id=None,
        mode=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        force_remove_masked=False,
        remove_extra_tokens=True,
        precomputed_mask=None,
        out_layer=None,  ## NEGATIVE
    ):
        if mode is None:
            assert self.cfg.supported_modality is not None
            mode = self.cfg.supported_modality

        if isinstance(mode, Modality):
            mode = mode.name

        feature_extractor = self.modality_encoders[mode]

        mask_seeds = None
        if id is not None:
            mask_seeds = MaskSeed(seed=self.cfg.seed, update=self.num_updates, ids=id)

        extractor_out = feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        xs = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = (
                        alibi_scale[i]
                        if alibi_scale.size(0) > 1
                        else alibi_scale.squeeze(0)
                    )
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                if features_only:
                    layer_results.append(lr)
                    xs.append(x)
                if out_layer is not None and i == len(self.blocks) + out_layer:
                    break

        if self.norm is not None:
            x = self.norm(x)

        if features_only:
            if remove_extra_tokens:
                x = x[:, feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return {
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
                "xs": xs
            }
        

from types import SimpleNamespace


d2v2_config = SimpleNamespace(**{'_name': 'data2vec_multi',
 'cosine_loss_temp': 0.0,
 'loss_beta': 0.0,
 'loss_scale': None,
 'mean_loss': False,
 'reconstruct_all': False,
 'depth': 24,
 'start_drop_path_rate': 0.0,
 'end_drop_path_rate': 0.0,
 'num_heads': 16,
 'norm_eps': 1e-06,
 'norm_affine': True,
 'encoder_dropout': 0.0,
 'post_mlp_drop': 0.0,
 'attention_dropout': 0.0,
 'activation_dropout': 0.0,
 'dropout_input': 0.0,
 'layerdrop': 0.0,
 'embed_dim': 1024,
 'mlp_ratio': 4.0,
 'layer_norm_first': False,
 'average_top_k_layers': 18,
 'end_of_block_targets': False,
 'clone_batch': 16,
 'layer_norm_target_layer': False,
 'batch_norm_target_layer': False,
 'instance_norm_target_layer': True,
 'instance_norm_targets': False,
 'layer_norm_targets': True,
 'ema_decay': 0.9998,
 'ema_same_dtype': True,
 'log_norms': True,
 'ema_end_decay': 1.0,
 'ema_anneal_end_step': 500000,
 'ema_encoder_only': False,
 'max_update': 750000,
 'modalities': SimpleNamespace(**{'audio': SimpleNamespace(**{'type': 'AUDIO',
   'prenet_depth': 4,
   'prenet_layerdrop': 0.0,
   'prenet_dropout': 0.0,
   'start_drop_path_rate': 0.0,
   'end_drop_path_rate': 0.0,
   'num_extra_tokens': 0,
   'init_extra_token_zero': True,
   'mask_from_extra': False,
   'mask_from_extra_detached': False,
   'mask_noise_std': 0.01,
   'mask_prob_min': None,
   'mask_prob': 0.7,
   'inverse_mask': False,
   'mask_prob_adjust': 0.0,
   'keep_masked_pct': 0.0,
   'mask_length': 5,
   'add_masks': False,
   'remove_masks': False,
   'mask_dropout': 0.0,
   'encoder_zero_mask': True,
   'mask_channel_prob': 0.0,
   'mask_channel_length': 64,
   'ema_local_encoder': False,
   'local_grad_mult': 1.0,
   'use_alibi_encoder': False,
   'alibi_scale': 1.0,
   'learned_alibi': False,
   'alibi_max_pos': None,
   'learned_alibi_scale': False,
   'learned_alibi_scale_per_head': False,
   'learned_alibi_scale_per_layer': False,
   'num_alibi_heads': 16,
   'model_depth': 24,
   'decoder': None,
   'max_alibi_scale': 0.0,
   'max_alibi_grad': 0.0,
   'max_alibi_val': 0.0,
   'extractor_mode': 'layer_norm',
   'feature_encoder_spec': '[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]',
   'conv_pos_width': 95,
   'conv_pos_groups': 16,
   'conv_pos_depth': 5,
   'conv_pos_pre_ln': False,
   'mlp_encoder': False,
   'mlp_n_in': 320,
   'mlp_dim': None,
   'mlp_layers': 9}),
  'image': SimpleNamespace(**{'type': 'IMAGE',
   'prenet_depth': 0,
   'prenet_layerdrop': 0.0,
   'prenet_dropout': 0.0,
   'start_drop_path_rate': 0.0,
   'end_drop_path_rate': 0.0,
   'num_extra_tokens': 1,
   'init_extra_token_zero': False,
   'mask_from_extra': False,
   'mask_from_extra_detached': False,
   'mask_noise_std': 0.01,
   'mask_prob_min': None,
   'mask_prob': 0.75,
   'inverse_mask': True,
   'mask_prob_adjust': 0.1,
   'keep_masked_pct': 0.0,
   'mask_length': 3,
   'add_masks': False,
   'remove_masks': False,
   'mask_dropout': 0.0,
   'encoder_zero_mask': True,
   'mask_channel_prob': 0.0,
   'mask_channel_length': 64,
   'ema_local_encoder': True,
   'local_grad_mult': 1.0,
   'use_alibi_encoder': False,
   'alibi_scale': 1.0,
   'learned_alibi': False,
   'alibi_max_pos': None,
   'learned_alibi_scale': False,
   'learned_alibi_scale_per_head': False,
   'learned_alibi_scale_per_layer': False,
   'num_alibi_heads': 16,
   'model_depth': 24,
   'decoder': SimpleNamespace(**{'decoder_dim': 1024,
    'decoder_groups': 16,
    'decoder_kernel': 5,
    'decoder_layers': 3,
    'input_dropout': 0.0,
    'add_positions_masked': False,
    'add_positions_all': False,
    'final_layer_norm': False,
    'tanh_scale': 0.0,
    'project_first_residual': False,
    'decoder_residual': True,
    'projection_layers': 1,
    'projection_ratio': 2.0,
    'residual_scale': 1.0,
    'remove_residual_noise': False,
    'post_residual_ln': False}),
   'max_alibi_scale': 0.0,
   'max_alibi_grad': 0.0,
   'max_alibi_val': 0.0,
   'input_size': 224,
   'in_chans': 3,
   'patch_size': 16,
   'embed_dim': 1024,
   'fix_masks': False,
   'exact_mask_pct': False,
   'unmask_focal': False,
   'focal_length': 1,
   'alibi_dims': 2,
   'alibi_distance': 'manhattan',
   'fixed_positions': True,
   'conv_pos_cfg': None,
   'transformer_decoder': False,
   'enc_dec_transformer': False,
   'conv_mae': False,
   'conv_mae_multiscale': True,
   'conv_mae_masking': True}),
  'text': SimpleNamespace(**{'type': 'TEXT',
   'prenet_depth': 4,
   'prenet_layerdrop': 0.0,
   'prenet_dropout': 0.0,
   'start_drop_path_rate': 0.0,
   'end_drop_path_rate': 0.0,
   'num_extra_tokens': 0,
   'init_extra_token_zero': True,
   'mask_from_extra': False,
   'mask_from_extra_detached': False,
   'mask_noise_std': 0.01,
   'mask_prob_min': None,
   'mask_prob': 0.7,
   'inverse_mask': False,
   'mask_prob_adjust': 0.0,
   'keep_masked_pct': 0.0,
   'mask_length': 5,
   'add_masks': False,
   'remove_masks': False,
   'mask_dropout': 0.0,
   'encoder_zero_mask': True,
   'mask_channel_prob': 0.0,
   'mask_channel_length': 64,
   'ema_local_encoder': False,
   'local_grad_mult': 1.0,
   'use_alibi_encoder': False,
   'alibi_scale': 1.0,
   'learned_alibi': False,
   'alibi_max_pos': None,
   'learned_alibi_scale': False,
   'learned_alibi_scale_per_head': False,
   'learned_alibi_scale_per_layer': False,
   'num_alibi_heads': 16,
   'model_depth': 24,
   'decoder': None,
   'max_alibi_scale': 0.0,
   'max_alibi_grad': 0.0,
   'max_alibi_val': 0.0,
   'max_source_positions': 512,
   'learned_pos': True,
   'dropout': 0.1,
   'no_scale_embedding': True,
   'layernorm_embedding': True,
   'no_token_positional_embeddings': False})}),
 'shared_decoder': None,
 'min_target_var': 0.0,
 'min_pred_var': 0.0,
 'supported_modality': 'IMAGE',
 'mae_init': False,
 'bert_init': True,
 'seed': 1,
 'skip_ema': False,
 'cls_loss': 0.01,
 'alt_cls_targets': False,
 'recon_loss': 0.0,
 'recon_dim': 0,
 'd2v_loss': 1.0,
 'qk_scale': None,
 'cosine_attention': False,
 'decoder_group': False,
 'extra_tokens_group': False,
 'shift_targets_down_updates': 0,
 'shift_targets_down_scale': 1.0,
 'modality_discrim_weight': 1.0,
 'modality_discrim_ema': 0.0,
 'modality_discrim_depth': 0}
)

d2v2_12_std = torch.tensor([1.7143195867538452, 0.5259653329849243, 1.5721157789230347, 0.4863377809524536, 0.6252964735031128, 0.4615776240825653, 0.5931997299194336, 2.6008753776550293, 0.4766261577606201, 0.7057560086250305, 0.4928278625011444, 0.5091769099235535, 0.4916788935661316, 0.5836999416351318, 0.6225807666778564, 0.7759029269218445, 0.4879513382911682, 0.4539593458175659, 0.4719198942184448, 0.5205592513084412, 0.4852038025856018, 0.4613896310329437, 0.5035895109176636, 0.48355862498283386, 0.44846242666244507, 0.5513873100280762, 0.5081468224525452, 0.4227955937385559, 0.4746731221675873, 0.4817580580711365, 0.6914331316947937, 0.45948556065559387, 0.5387393236160278, 0.505884051322937, 0.48521363735198975, 0.753070592880249, 0.4427732229232788, 0.434841126203537, 0.5120889544487, 1.4472752809524536, 0.4952249526977539, 0.5811933875083923, 0.48664823174476624, 0.4907647371292114, 0.4814741909503937, 0.47892898321151733, 0.47738510370254517, 0.7923340201377869, 0.48127198219299316, 0.528940737247467, 0.5406062006950378, 0.46323928236961365, 0.509406328201294, 1.6873691082000732, 0.5071007013320923, 0.5055264234542847, 0.49880826473236084, 0.43403762578964233, 0.49112436175346375, 0.4450030028820038, 0.43282032012939453, 0.4765056371688843, 0.4321625530719757, 0.4534757435321808, 0.4717271327972412, 0.44052624702453613, 0.48420190811157227, 0.4530673325061798, 0.45030713081359863, 0.5342947840690613, 0.5588785409927368, 0.4449504613876343, 0.4566808342933655, 0.46198931336402893, 0.5224624872207642, 0.4669182300567627, 0.4822517931461334, 0.46546873450279236, 0.4966888129711151, 0.4612712562084198, 0.4631820023059845, 0.48109281063079834, 0.4849053621292114, 0.4877570867538452, 0.46163490414619446, 0.4580410122871399, 0.5507311820983887, 0.5067227482795715, 0.4839591085910797, 0.5700198411941528, 0.5161567330360413, 0.5065191388130188, 0.7466540932655334, 0.5204383730888367, 0.49725955724716187, 0.45053333044052124, 0.45386290550231934, 0.49756768345832825, 0.43598827719688416, 0.9835352897644043, 0.6921523213386536, 0.44924989342689514, 0.5286270380020142, 0.4264791011810303, 0.46355965733528137, 0.4822981655597687, 0.5389453172683716, 0.5515199899673462, 0.48042014241218567, 0.4820408523082733, 0.5036085247993469, 0.5223432779312134, 0.4824448525905609, 0.4729730784893036, 0.44685330986976624, 0.47218573093414307, 0.6641924977302551, 0.48100611567497253, 1.126521110534668, 0.513288676738739, 0.44053104519844055, 0.47927331924438477, 0.5130708813667297, 0.5322798490524292, 0.47631359100341797, 0.5077064037322998, 0.48212069272994995, 0.561772346496582, 0.5715526342391968, 0.49431174993515015, 0.48231303691864014, 0.48703232407569885, 0.5067303776741028, 0.48998862504959106, 0.4815555214881897, 0.4682222306728363, 0.4604851007461548, 0.5476630926132202, 0.5084977746009827, 0.5257272124290466, 0.46838968992233276, 0.6670439839363098, 1.2565149068832397, 0.4703105688095093, 0.4606296420097351, 0.5144451260566711, 0.4910828471183777, 0.5846068263053894, 0.5760313272476196, 0.5263540148735046, 0.4685674011707306, 0.4976310431957245, 0.4621015787124634, 0.47825148701667786, 0.6305429935455322, 0.47755852341651917, 0.4775390923023224, 0.5188104510307312, 0.4935683608055115, 0.5691671967506409, 0.4652363955974579, 0.49891287088394165, 0.4542810618877411, 0.46017053723335266, 0.4837132394313812, 0.5070484280586243, 0.4865330755710602, 1.0575475692749023, 0.4816477596759796, 0.508073627948761, 0.49342143535614014, 0.47756722569465637, 0.5670822858810425, 0.44835561513900757, 0.45536044239997864, 0.4533184766769409, 0.48844069242477417, 0.5458412766456604, 0.5309258103370667, 0.4736350476741791, 0.5019723773002625, 0.4984630048274994, 0.5119196176528931, 0.5157008171081543, 0.798039436340332, 0.5182589292526245, 0.6656643152236938, 0.46326732635498047, 0.44803303480148315, 0.483267605304718, 0.49832379817962646, 0.5008751749992371, 0.48150113224983215, 0.4767361879348755, 0.4693645238876343, 0.4722767770290375, 0.48409777879714966, 0.5605290532112122, 0.5168891549110413, 0.48367545008659363, 0.4747414290904999, 0.45997846126556396, 0.5248360633850098, 0.4801252782344818, 0.502997100353241, 0.4442753195762634, 0.5766912698745728, 0.49531903862953186, 0.4710557460784912, 0.4776116907596588, 0.5039357542991638, 0.45637789368629456, 0.4691314697265625, 0.4629024565219879, 0.45744043588638306, 0.4918842911720276, 0.47092992067337036, 0.5282542109489441, 0.46780771017074585, 0.5005477070808411, 0.4671836793422699, 0.48168376088142395, 0.5249614119529724, 0.5219925045967102, 0.4925771951675415, 0.5265450477600098, 0.47455480694770813, 0.4746258556842804, 0.4943791329860687, 0.5639381408691406, 0.5006301403045654, 0.4600648880004883, 0.7884655594825745, 0.45157888531684875, 0.49258995056152344, 0.4554096460342407, 0.47146615386009216, 0.48878592252731323, 1.370406150817871, 0.5422217845916748, 0.47739189863204956, 0.48758262395858765, 0.5129379630088806, 0.46059730648994446, 0.48720160126686096, 0.4833666682243347, 0.4442417621612549, 1.156221628189087, 0.5083333253860474, 0.4794164001941681, 0.4867599904537201, 0.4733979403972626, 0.4798559248447418, 0.4900586009025574, 0.4843064248561859, 0.46826520562171936, 0.5120558738708496, 0.7884657382965088, 0.7686752080917358, 0.5162506103515625, 0.4596789479255676, 0.4822338819503784, 1.9407657384872437, 1.113363265991211, 0.7145654559135437, 0.7095075845718384, 0.47234517335891724, 0.525841474533081, 0.4457598626613617, 0.47877761721611023, 0.5388484597206116, 0.45341625809669495, 0.50823575258255, 0.4471467137336731, 0.6442512273788452, 0.5073826909065247, 0.45795631408691406, 0.5068398118019104, 0.7288253903388977, 0.45713338255882263, 1.1342384815216064, 0.5874901413917542, 0.4846671223640442, 0.5578012466430664, 0.48686134815216064, 0.5187036991119385, 0.4797843098640442, 0.4908222258090973, 0.490058034658432, 0.5224190950393677, 0.7109644412994385, 0.8504059314727783, 0.49434739351272583, 0.4630945920944214, 0.5358626842498779, 0.4878246784210205, 0.5161843299865723, 0.5501636862754822, 0.5060940980911255, 1.6289795637130737, 0.4568762183189392, 0.48491740226745605, 0.46110469102859497, 0.4440530836582184, 0.5609595775604248, 0.45634812116622925, 0.4718032777309418, 0.49983006715774536, 0.6584033966064453, 0.49749547243118286, 0.5355739593505859, 0.48821574449539185, 0.46715062856674194, 0.47845736145973206, 0.5217816233634949, 0.5425538420677185, 0.4946318566799164, 0.4999028444290161, 0.49655821919441223, 0.4677058160305023, 0.4791293442249298, 0.47235921025276184, 0.5182938575744629, 0.45551738142967224, 0.46992582082748413, 0.5855816602706909, 0.7935413122177124, 0.4993453025817871, 0.5571065545082092, 0.4685541093349457, 0.4701991677284241, 0.5942508578300476, 0.48560968041419983, 0.452267050743103, 0.496084064245224, 0.4978652894496918, 0.4386063814163208, 0.4717290699481964, 0.7243639826774597, 0.5742682814598083, 0.5175004005432129, 0.5207015872001648, 0.8056126236915588, 0.4912734031677246, 0.4541911482810974, 0.48336127400398254, 0.505936324596405, 0.4492543637752533, 0.4781171381473541, 0.46399012207984924, 0.469990074634552, 0.4933171570301056, 0.4650290608406067, 1.603661298751831, 0.49281808733940125, 0.5204745531082153, 0.5459663271903992, 0.503679096698761, 0.49803367257118225, 0.46632346510887146, 0.5230693221092224, 0.48688921332359314, 0.6994965672492981, 0.710091233253479, 0.46950769424438477, 0.47352951765060425, 0.5062945485115051, 0.5609676241874695, 0.47082993388175964, 0.5530094504356384, 0.47155067324638367, 0.6285647749900818, 0.4847359359264374, 0.45148804783821106, 0.5897446274757385, 0.5226767063140869, 0.48509177565574646, 0.5371697545051575, 0.4426495134830475, 0.46084049344062805, 0.42679816484451294, 0.5114015936851501, 0.48137491941452026, 0.4700334072113037, 0.4966558516025543, 0.5112631916999817, 0.5225161910057068, 0.4824070632457733, 0.4684094488620758, 0.5094033479690552, 0.5266973376274109, 0.444451242685318, 0.8306864500045776, 0.45353269577026367, 0.45405489206314087, 0.44770511984825134, 0.5055490732192993, 0.47452130913734436, 0.4645853042602539, 0.47740301489830017, 0.6399964690208435, 0.4376465380191803, 0.5197327136993408, 0.5075460076332092, 0.5041786432266235, 0.44067180156707764, 0.4592853784561157, 0.49302271008491516, 0.442242830991745, 0.4856396019458771, 0.5900024175643921, 0.4669262766838074, 0.4879051744937897, 0.45391884446144104, 0.4541222155094147, 0.5103352069854736, 1.3686574697494507, 0.5377334952354431, 0.5051239132881165, 0.4849715530872345, 0.7293565273284912, 0.47750943899154663, 0.46601954102516174, 0.49212729930877686, 0.48072442412376404, 0.4580775797367096, 0.5051493048667908, 0.5279922485351562, 0.47266629338264465, 0.5666463375091553, 0.5059764981269836, 0.45071184635162354, 0.48440760374069214, 0.5042374134063721, 0.44882336258888245, 0.46657636761665344, 0.539040207862854, 0.5102065801620483, 0.5319273471832275, 0.49278172850608826, 0.5185022354125977, 0.46593594551086426, 0.4747137129306793, 0.5575792789459229, 0.5266367197036743, 0.4586792588233948, 0.5988649129867554, 0.4451603889465332, 0.49485039710998535, 1.7206718921661377, 0.4386351406574249, 0.5050587058067322, 0.45971888303756714, 0.5004608631134033, 0.47299548983573914, 0.43869781494140625, 0.4678645730018616, 0.4679088890552521, 0.6300318837165833, 0.753396213054657, 0.6139176487922668, 0.5561637878417969, 0.5477942228317261, 0.5092222094535828, 0.5256072282791138, 2.364285707473755, 0.42428916692733765, 0.5055098533630371, 0.5207569599151611, 0.4485221207141876, 0.4452337920665741, 0.7421566843986511, 0.4373774528503418, 0.5449707508087158, 0.45802250504493713, 0.5054560303688049, 0.4974212348461151, 0.564220666885376, 0.953541100025177, 0.46974459290504456, 0.4643647372722626, 0.4630602300167084, 0.46019184589385986, 1.8659111261367798, 0.47847434878349304, 0.5419135689735413, 0.46304401755332947, 0.4400555193424225, 1.7135529518127441, 0.48105165362358093, 0.5118847489356995, 0.5050941705703735, 0.4896504282951355, 0.47518274188041687, 0.5816757678985596, 0.4787048101425171, 0.5308335423469543, 0.635687530040741, 0.44677048921585083, 0.6222075819969177, 0.722343385219574, 0.6105208992958069, 0.5123047232627869, 0.45688432455062866, 0.5044951438903809, 0.4576584994792938, 0.5247951149940491, 0.5132135152816772, 0.4752810001373291, 0.5661938786506653, 0.4644637703895569, 0.7011630535125732, 0.5488810539245605, 0.5066341757774353, 0.6102062463760376, 0.7828695774078369, 0.6851696372032166, 0.4962502717971802, 0.6077042818069458, 0.4852491617202759, 0.5063872337341309, 0.7891708016395569, 0.47399649024009705, 2.0862016677856445, 0.4031859040260315, 2.073993682861328, 0.6723452806472778, 0.48080742359161377, 0.45877841114997864, 0.4839378595352173, 0.563006579875946, 0.45509228110313416, 0.47186577320098877, 0.5084784626960754, 0.6210505366325378, 0.4578529894351959, 0.7113475799560547, 0.9522448182106018, 0.5020246505737305, 0.49119672179222107, 0.4538820683956146, 0.5293563604354858, 0.44640985131263733, 0.4746414124965668, 0.4663568139076233, 0.45513737201690674, 0.5550380349159241, 0.5028210282325745, 0.5612569451332092, 0.47359442710876465, 0.45705077052116394, 0.4846446216106415, 0.480704128742218, 0.4821990430355072, 0.5038185715675354, 0.4535871148109436, 0.4516525864601135, 0.5227063298225403, 0.496787965297699, 0.8547360301017761, 0.5540058016777039, 0.5173356533050537, 0.5105699300765991, 0.4827985465526581, 0.4893546402454376, 0.7615548372268677, 0.43440234661102295, 0.3955163061618805, 0.4956596791744232, 0.4536210894584656, 0.485153466463089, 0.48681268095970154, 0.4955998659133911, 0.8241665363311768, 0.7879904508590698, 0.6351007223129272, 0.450047105550766, 0.47135230898857117, 0.48718225955963135, 0.4611225426197052, 0.45962485671043396, 0.5492035746574402, 0.5757979154586792, 0.487103670835495, 0.4468379616737366, 0.4743455648422241, 0.48034289479255676, 0.47651177644729614, 0.4714882969856262, 0.4921356439590454, 0.4492872655391693, 0.46641290187835693, 0.5037959218025208, 0.7829825282096863, 0.6555920243263245, 0.543789803981781, 0.681760847568512, 0.47474873065948486, 0.4829057455062866, 0.47285664081573486, 1.0807654857635498, 0.5704447031021118, 0.6478289365768433, 0.5059822201728821, 0.5279452800750732, 0.5167149901390076, 0.4765699505805969, 0.4631853401660919, 0.4612647593021393, 0.5574339032173157, 0.4820546507835388, 0.48392289876937866, 0.7944263219833374, 0.46294066309928894, 0.48678451776504517, 0.5015594959259033, 0.4763072729110718, 0.47896134853363037, 0.49020496010780334, 0.48924291133880615, 0.4838624894618988, 0.47531118988990784, 0.4739695191383362, 1.2851684093475342, 0.4523046016693115, 0.5969529151916504, 0.47299978137016296, 2.3670201301574707, 0.46520885825157166, 0.4932715594768524, 0.4781786799430847, 2.132633686065674, 0.5860868096351624, 0.44218069314956665, 0.546772837638855, 0.5163512229919434, 0.4516104459762573, 0.5526682138442993, 0.5880622267723083, 0.5949016809463501, 0.4575449824333191, 0.46610787510871887, 0.9822306036949158, 0.46154606342315674, 0.5054338574409485, 0.47108423709869385, 1.2631464004516602, 0.5091187953948975, 0.4628240466117859, 0.47116267681121826, 0.43597468733787537, 0.4703264534473419, 0.5862725377082825, 0.4828304648399353, 0.4565570056438446, 0.48461347818374634, 0.5099529027938843, 0.48228171467781067, 0.4427153766155243, 3.704069137573242, 0.46763876080513, 0.4578510820865631, 0.4523693323135376, 0.4475661516189575, 0.4897129535675049, 0.5146257877349854, 0.4608882963657379, 0.4514702558517456, 0.44966694712638855, 0.47952407598495483, 0.5892067551612854, 0.4672599136829376, 0.46047112345695496, 0.5010252594947815, 0.5344521403312683, 0.8863028883934021, 0.43064019083976746, 0.46389639377593994, 0.4710347056388855, 0.4350631833076477, 0.5363197922706604, 0.781825065612793, 0.5033189058303833, 0.46451857686042786, 0.4328738749027252, 0.47528955340385437, 0.47890013456344604, 0.5057220458984375, 0.466420978307724, 0.45531734824180603, 0.4554806649684906, 0.5286865234375, 0.48163220286369324, 0.4720601737499237, 0.47363728284835815, 0.44506365060806274, 0.45607462525367737, 0.4826270043849945, 0.47209885716438293, 0.4890591502189636, 0.5013560056686401, 0.5308541655540466, 0.4566815495491028, 0.4363518953323364, 0.7729146480560303, 0.4754297435283661, 0.6910618543624878, 0.554892361164093, 0.5937984585762024, 0.6783701181411743, 0.46519404649734497, 0.5033023953437805, 0.4644238352775574, 0.5104886889457703, 0.4816296100616455, 0.4625292420387268, 0.47262510657310486, 0.5118705630302429, 0.5532336235046387, 0.48408597707748413, 0.5403879284858704, 0.4736521542072296, 0.5151403546333313, 0.4984828531742096, 0.5110184550285339, 0.45306169986724854, 0.4695259630680084, 0.44828543066978455, 0.4951188564300537, 1.814292073249817, 0.549314558506012, 1.3951635360717773, 0.5004539489746094, 0.5896664261817932, 0.5123594999313354, 0.5060793161392212, 0.45690587162971497, 0.47075578570365906, 0.475929856300354, 0.4832924008369446, 0.4888384938240051, 0.6179359555244446, 0.4718579649925232, 0.45997172594070435, 0.4687635004520416, 0.7270686626434326, 0.4504535496234894, 0.8148441910743713, 0.452122300863266, 0.6689842343330383, 0.4700954556465149, 0.772312343120575, 0.48133304715156555, 0.5219054222106934, 0.4940274655818939, 0.6401433348655701, 0.49335020780563354, 0.4877510368824005, 0.4971371591091156, 0.5144386291503906, 0.4551357626914978, 0.5344761610031128, 0.454338937997818, 0.4774194359779358, 0.4393724799156189, 0.49745434522628784, 0.5141054391860962, 0.5378813743591309, 0.5369637608528137, 0.5634642839431763, 0.6758680939674377, 0.5192216634750366, 0.5622823238372803, 0.49013128876686096, 0.6156094670295715, 0.46798253059387207, 1.14351224899292, 0.5133482217788696, 0.448995977640152, 0.7465838193893433, 0.5518608093261719, 0.48753467202186584, 0.5113651156425476, 0.7031078934669495, 0.953072726726532, 0.47600725293159485, 0.7040034532546997, 0.46785253286361694, 0.4445611238479614, 0.6399282217025757, 0.4575897753238678, 0.48878687620162964, 1.4834003448486328, 0.4779414236545563, 0.4910161793231964, 0.46135684847831726, 1.3734321594238281, 0.4972502589225769, 0.49210160970687866, 0.4565756320953369, 0.529870867729187, 0.5088825821876526, 0.5335147976875305, 0.44477152824401855, 0.4478742480278015, 0.44028013944625854, 0.4568280577659607, 0.5216493010520935, 2.2291831970214844, 0.7445442080497742, 0.49455171823501587, 0.457772433757782, 0.4775288701057434, 0.4859618544578552, 0.5122065544128418, 0.8448947668075562, 0.4837675392627716, 1.6177254915237427, 0.5670027732849121, 0.47022196650505066, 0.45723384618759155, 0.4800226390361786, 2.0360965728759766, 0.5337859988212585, 0.4833858013153076, 0.5079479813575745, 0.4605746865272522, 0.4806421399116516, 0.5130828022956848, 0.4642304480075836, 0.45646899938583374, 0.47867801785469055, 0.5349822640419006, 2.519371747970581, 1.0007580518722534, 0.519834041595459, 0.5896478891372681, 0.4900038540363312, 0.5648201704025269, 0.46167439222335815, 0.48918256163597107, 0.43410223722457886, 0.5950461030006409, 0.459888219833374, 0.5375059247016907, 0.48990997672080994, 0.610268235206604, 0.47169050574302673, 0.5882740020751953, 0.5093563199043274, 0.4547959268093109, 0.5232970714569092, 0.43655502796173096, 0.5238208770751953, 0.5083205699920654, 0.4932748079299927, 0.6159178018569946, 0.458647221326828, 0.48161348700523376, 0.47138136625289917, 0.4813978672027588, 0.47822806239128113, 0.4929860234260559, 0.4818013608455658, 0.46341657638549805, 0.9617831110954285, 0.5094396471977234, 0.4648878276348114, 0.466753751039505, 0.565567135810852, 0.482160747051239, 0.7597511410713196, 0.4450536370277405, 0.46436870098114014, 0.4779229164123535, 0.5174615979194641, 0.481147438287735, 0.48676249384880066, 0.4619966447353363, 0.4609915018081665, 0.5079206228256226, 0.4454151690006256, 0.4981209337711334, 0.5782195925712585, 0.6200931072235107, 0.49036210775375366, 0.5139603614807129, 0.4816179871559143, 0.4520772099494934, 0.5169569849967957, 0.47529542446136475, 0.5097917318344116, 0.8718745708465576, 0.49256432056427, 0.5045002102851868, 0.5436437129974365, 0.4640907347202301, 0.4697455167770386, 0.4749232828617096, 0.5493507981300354, 0.4782836437225342, 0.4545508623123169, 2.23856520652771, 0.48218613862991333, 0.4588022828102112, 0.44289448857307434, 0.444560706615448, 0.47700318694114685, 0.7012177109718323, 0.6278716921806335, 0.4348026216030121, 0.5007299780845642, 0.7808961868286133, 0.4844789505004883, 0.48333898186683655, 0.4560331702232361, 0.5098503828048706, 0.44185400009155273, 0.4861728549003601, 0.4515114426612854, 0.4835145175457001, 0.458003431558609, 0.44936028122901917, 0.5041279196739197, 0.5952136516571045, 0.6254212260246277, 0.48669740557670593, 0.5057520270347595, 0.4814610481262207, 0.5232632756233215, 1.4803334474563599, 0.41525688767433167, 0.45741111040115356, 0.7565503120422363, 0.5248284935951233, 0.4632813632488251, 1.7220245599746704, 0.5016604065895081, 0.5004728436470032, 0.46836715936660767, 0.5251286625862122, 0.45814049243927, 0.4422759711742401, 0.4951963722705841, 0.4671713709831238, 0.4721435606479645, 0.4727913737297058, 1.1001667976379395, 0.502860426902771, 0.5172538161277771, 0.49039602279663086, 2.651355028152466, 0.6416285634040833, 0.46989187598228455, 0.44577422738075256, 0.4484672546386719, 0.4606108069419861, 0.4488101005554199, 0.5466633439064026, 0.46715623140335083, 0.6134126782417297, 0.5117906928062439, 0.47845658659935, 0.478168785572052, 0.4779745936393738, 0.5648555755615234, 0.4413744807243347, 0.5637167096138, 0.5142650604248047, 0.4533678889274597, 0.49838733673095703, 0.4952107071876526, 0.4458698630332947, 0.4704954922199249, 0.4861122667789459, 0.49636730551719666, 0.4843072295188904, 0.6693789958953857, 0.5417760610580444, 0.5006742477416992, 0.5568727850914001, 1.5601212978363037, 0.4715024530887604, 0.47964420914649963, 0.4625881314277649, 1.4857171773910522, 0.5071725845336914, 0.46170368790626526, 0.562862753868103, 0.46990063786506653, 0.45146626234054565, 0.4428105056285858, 0.5697567462921143, 0.6790448427200317, 0.5176102519035339, 0.4887544512748718, 0.47981056571006775, 0.509110152721405, 0.46126842498779297, 0.5214576125144958, 0.4823669493198395, 0.4995634853839874, 0.45886462926864624, 0.49055221676826477, 0.5810911655426025, 0.5243300795555115, 0.49301451444625854, 0.5319560170173645, 0.4702531099319458, 0.7205012440681458, 0.517248272895813, 0.42819878458976746, 0.43674615025520325, 0.4969038963317871, 0.4862322211265564, 0.5139387249946594, 0.4678921699523926, 0.461713582277298, 0.49103397130966187, 0.5678456425666809, 0.48420295119285583, 0.4996723532676697, 0.6130819320678711, 0.33194148540496826])
d2v2_12_mean= torch.tensor([-0.8445490598678589, 0.15477554500102997, 0.1479628086090088, -0.012875348329544067, -0.00959821417927742, -0.14145195484161377, -0.16924743354320526, 0.28247660398483276, -0.18270152807235718, 0.0362468883395195, -0.043754030019044876, -0.10893438756465912, 0.024627281352877617, -0.1898491382598877, -0.2520236372947693, 0.055126920342445374, 0.09561359137296677, 0.05958176031708717, -0.0722939670085907, 0.01751611940562725, -0.16159650683403015, 0.06363600492477417, -0.060686495155096054, -0.14250193536281586, -0.16858132183551788, 0.4064728319644928, -0.07800662517547607, -0.14705410599708557, -0.053696367889642715, 0.15630966424942017, -0.06898672133684158, 0.0009478782303631306, -0.2274111956357956, -0.03304455056786537, 0.11067400127649307, 0.1144108697772026, -0.11052116006612778, -0.2106057107448578, -0.09839481860399246, 1.1298072338104248, -0.0717640072107315, -0.14146064221858978, -0.00893676932901144, 0.24302232265472412, -0.07973659783601761, 0.12806400656700134, -0.002310041803866625, -0.2723897099494934, -0.08584631979465485, -0.2108689844608307, 0.06734177470207214, -0.13045662641525269, 0.11781768500804901, 0.7238938808441162, 0.0573393851518631, -0.2730674147605896, -0.03648456558585167, 0.095210500061512, 0.019918208941817284, 0.04819764941930771, 0.17749345302581787, -0.12661275267601013, -0.005215555429458618, -0.2614291310310364, -0.03966040164232254, 0.07833414524793625, 0.0732194110751152, 0.014712007716298103, 0.22669008374214172, 0.05586092174053192, -0.00425039604306221, 0.15713584423065186, -0.05266328155994415, -0.05664915591478348, 0.12135616689920425, -0.01717097871005535, 0.014845183119177818, -0.011201516725122929, -0.1819431483745575, -0.0643933042883873, 0.015416871756315231, -0.0005943977739661932, -0.15755878388881683, 0.29096102714538574, 0.10248798131942749, -0.11695277690887451, -0.026268957182765007, -0.10052245855331421, 0.035421766340732574, 0.2699345648288727, 0.059824585914611816, 0.05285051092505455, 0.3161073625087738, 0.05405240133404732, 0.04731271043419838, -0.03235234320163727, 0.11628707498311996, -0.1486985981464386, 0.0987134724855423, -0.7184066772460938, -0.15163099765777588, -0.024483611807227135, -0.15971730649471283, 0.27661463618278503, 0.1445642113685608, -0.05150650814175606, -0.007072856649756432, 0.07546184957027435, 0.1749076247215271, -0.08801151067018509, 0.20440682768821716, 0.1597638875246048, -0.10933873802423477, -0.03562159836292267, 0.20883364975452423, -0.24780000746250153, 0.02772662043571472, 0.057330187410116196, 0.15691137313842773, 0.42944878339767456, -0.24678725004196167, 0.03220215067267418, -0.061677683144807816, -0.03575187548995018, -0.15043602883815765, -0.03516335040330887, 0.24423569440841675, -0.13067597150802612, -0.217870831489563, -0.033758845180273056, 0.34374162554740906, -0.00967290811240673, 0.02247617207467556, 0.08189468085765839, 0.20979104936122894, 0.10807905346155167, 0.19447416067123413, -0.1409398913383484, 0.12768907845020294, 0.03716008737683296, 0.0021732032764703035, -0.2560007572174072, -1.874603033065796, -0.12753160297870636, -0.08492974191904068, 0.013074194081127644, -0.11672014743089676, -0.036020390689373016, -0.17761622369289398, 0.061674486845731735, -0.02790697291493416, 0.07406199723482132, -0.0032570534385740757, -0.19673414528369904, 0.11930065602064133, -0.04399847239255905, -0.09272271394729614, -0.2081010639667511, -0.002098511205986142, -0.042446356266736984, 0.059154950082302094, 0.057901859283447266, -0.1587245613336563, -0.04555685073137283, 0.2305472195148468, 0.09826702624559402, -0.055072762072086334, 0.2962173819541931, 0.21129360795021057, 0.18628625571727753, 0.050699979066848755, 0.1457941234111786, 0.015229769982397556, -0.08103931695222855, 0.0055203065276145935, -0.16322000324726105, 0.04453017935156822, -0.05867299810051918, 0.2172413021326065, 0.13606886565685272, -0.1220957413315773, -0.12071136385202408, 0.059450216591358185, 0.13059170544147491, -0.0356246642768383, 0.02094026282429695, 0.005404170602560043, 0.07833825796842575, 0.00698512140661478, 0.012099028564989567, -0.3199264705181122, -0.018686894327402115, 0.0734044760465622, 0.003674923675134778, -0.05136981979012489, -0.05351101607084274, 0.09411972761154175, 0.024584582075476646, 0.054113004356622696, -0.01687043160200119, 0.1792021095752716, -0.29141226410865784, 0.03886707127094269, 0.040815021842718124, 0.15273182094097137, 0.06170301511883736, 0.24220561981201172, -0.027352357283234596, 0.004404406528919935, -0.12855353951454163, -0.02040134370326996, -0.120488740503788, 0.059480905532836914, -0.04969759285449982, 0.06766819208860397, 0.03608453646302223, -0.08582524955272675, -0.13464316725730896, -0.32880648970603943, 0.17053748667240143, 0.1792914867401123, -0.13457749783992767, 0.11908376216888428, 0.2577522397041321, 0.1601325273513794, 0.1921854317188263, 0.0034633935429155827, -0.015824513509869576, -0.051314469426870346, -0.05444558337330818, -0.014407279901206493, -0.07279602438211441, 0.07534149289131165, -0.0380297414958477, 0.038350533694028854, -0.03191142901778221, -0.17107194662094116, 0.0903211236000061, -1.2181367874145508, 0.05850519612431526, -0.021382810547947884, 0.07452606409788132, -0.01873892918229103, 0.2827099561691284, 0.04561880603432655, -0.009015308693051338, 0.10861527919769287, 0.3351198732852936, 0.20056669414043427, 0.16314861178398132, 0.005558560602366924, 0.16721634566783905, 0.1488446444272995, 0.18224681913852692, -0.20586714148521423, 0.0628218948841095, -0.0656806081533432, 0.15415386855602264, 0.1587245613336563, -0.03736284002661705, -0.1772986650466919, -0.09916575998067856, 0.41828736662864685, -0.2414931356906891, -0.2162836343050003, -0.09419085085391998, -0.0956873670220375, 0.007806259207427502, -0.09913568943738937, 0.14439259469509125, -0.005536556243896484, -0.11259220540523529, 0.1362684965133667, 0.0594821460545063, 0.2679970860481262, 0.028254549950361252, 0.06634741276502609, -0.0029462575912475586, -0.07736074924468994, 0.14197885990142822, 0.4345760643482208, -0.2020304948091507, -0.0010640262626111507, 0.05979091301560402, 0.03655930981040001, 0.16570168733596802, -0.1493098884820938, 0.30591392517089844, -0.0008919507963582873, 0.006789154373109341, 0.39694106578826904, -0.1711343377828598, 0.028724567964673042, 0.15062591433525085, 0.017815327271819115, -0.041598640382289886, -0.21915651857852936, 0.01465645246207714, 0.1927042454481125, 0.4096967279911041, 0.027582693845033646, 0.09645799547433853, -0.16929423809051514, 0.0881900042295456, 0.09085336327552795, -0.02864488773047924, 0.023667143657803535, -0.09770514816045761, 0.047121476382017136, 0.21574990451335907, -0.11459439992904663, -0.041989054530858994, 0.03712983429431915, 0.12038883566856384, -0.11584555357694626, -0.08551936596632004, 0.1041744127869606, 0.07127261906862259, -0.2337905466556549, -0.054954756051301956, 0.0005469890311360359, 0.0779273733496666, 0.07319425046443939, 0.03509003296494484, -0.116678886115551, 0.14345191419124603, -0.10108901560306549, 0.13205856084823608, 0.0896797925233841, 0.04309305548667908, 0.24502378702163696, -0.30975469946861267, -0.039526503533124924, 0.029324229806661606, 0.11558385193347931, -0.012482828460633755, -0.1790064573287964, -0.0218930896371603, 0.24379107356071472, 0.031124385073781013, 0.13053616881370544, 0.05846831575036049, 0.23702464997768402, 0.1376716047525406, 0.09364847838878632, -0.15054501593112946, 0.04164839908480644, 0.1707388013601303, -0.10396779328584671, -0.13381235301494598, 0.10667286068201065, 0.10330016165971756, 0.020754998549818993, -0.08466359972953796, -0.020055970177054405, -0.07652970403432846, 0.11816558241844177, 0.10620634257793427, 0.21490712463855743, 0.25069403648376465, -0.10153841972351074, 0.05963607504963875, -0.08308887481689453, 0.36662110686302185, -0.11763312667608261, 0.01628938503563404, 0.033135559409856796, 0.21466147899627686, -0.06953262537717819, 0.21789926290512085, 0.11016954481601715, -0.19336064159870148, -0.114857979118824, 0.24600471556186676, 0.17461362481117249, 0.18354323506355286, 0.2188737690448761, -0.16680769622325897, 0.060453157871961594, -0.07126133143901825, -0.08601352572441101, 0.06916996091604233, 0.09058178216218948, 0.03222448006272316, -0.015391743741929531, -0.13590385019779205, 0.26023268699645996, 0.10382590442895889, -0.2886732816696167, 0.21987700462341309, -0.023214835673570633, 0.0040930709801614285, -0.6097663044929504, 0.11081039905548096, 0.10324278473854065, 0.026337923482060432, 0.1973511427640915, 0.02286018617451191, 0.030826536938548088, -0.024001507088541985, -0.10969903320074081, 0.17059072852134705, -0.16891612112522125, 0.06691977381706238, 0.27960842847824097, -0.11351358145475388, 0.09596642106771469, 0.0065834298729896545, 0.11085321754217148, 0.12932661175727844, -0.05050988867878914, 0.11936096847057343, -0.2244863212108612, -0.13197189569473267, 0.1447179615497589, 0.06551679223775864, -1.7304353713989258, -0.22641177475452423, -0.12514112889766693, -0.014618978835642338, -0.3161837160587311, 0.09143440425395966, 0.13555113971233368, -0.0067017762921750546, -0.10783296823501587, 0.02020445466041565, 0.2442036122083664, 0.018006181344389915, 0.016069630160927773, 0.22023127973079681, 0.02868618071079254, -0.020548144355416298, 0.04743817076086998, 0.0644717812538147, -0.2704446017742157, 0.010333421640098095, -0.22452013194561005, 0.13541989028453827, -0.07610991597175598, -0.07130058109760284, 0.22120793163776398, 0.18342120945453644, -0.04193746671080589, -0.28123462200164795, 0.038611672818660736, -0.02652411162853241, -0.11206822097301483, 0.03753988817334175, 0.10176384449005127, 0.3127118647098541, 0.13192647695541382, -0.1332424283027649, 0.025999603793025017, -0.0297206062823534, 0.09915894269943237, -0.00741238659247756, 0.05430569127202034, -0.07141061127185822, 0.11121369898319244, -0.20333462953567505, 0.12529104948043823, -0.1322224885225296, -0.1235935166478157, 0.08391006290912628, 0.08043882250785828, 0.4550285339355469, -0.023028263822197914, 0.1926816701889038, 0.05183548480272293, -0.048523660749197006, 0.0413498617708683, -0.14106185734272003, -0.15076670050621033, 0.09211515635251999, 0.04743967205286026, 0.0702684298157692, -0.011284889653325081, 0.15503041446208954, -0.901027262210846, -0.04322120547294617, -0.17077285051345825, 0.09425774216651917, -0.06306365877389908, -0.00796472281217575, 0.061786726117134094, -0.10845784097909927, 0.062385473400354385, 0.05262959748506546, -0.03335389867424965, 0.11734876036643982, 0.16126300394535065, -0.20785565674304962, -0.020762024447321892, 0.14799556136131287, 0.325826495885849, -0.13190029561519623, 0.014980807900428772, -0.27564164996147156, 0.07653529942035675, 0.2549634277820587, -0.23832343518733978, -0.2469640076160431, 0.10145370662212372, -0.12924465537071228, -0.2761044502258301, -0.005283933598548174, -0.2118120640516281, -0.20622335374355316, 0.18322135508060455, 0.08178088068962097, -0.046768296509981155, -0.12437871098518372, -0.1391221433877945, 0.09672881662845612, -0.021970516070723534, 0.056165095418691635, -0.1870168298482895, -0.13727709650993347, -0.1437152475118637, 0.33679673075675964, 0.15810300409793854, -0.23313167691230774, 0.0663490742444992, -0.5235679149627686, 0.0192865040153265, 0.15918642282485962, 0.08968383818864822, -0.1532541960477829, 0.033680614084005356, 0.06845748424530029, -0.15886548161506653, 0.06118592992424965, 0.0983414500951767, 0.2649596333503723, 0.10000412166118622, 0.03881782293319702, 0.11414411664009094, 0.6607146859169006, 0.11007039248943329, -0.07139710336923599, -0.23660200834274292, 0.0392739437520504, 0.11997213959693909, 0.22678335011005402, 0.07658374309539795, -0.01556254643946886, 0.2743735611438751, 0.012383624911308289, -0.07808534801006317, 0.08655733615159988, 0.1491595208644867, -0.006859424524009228, 0.01840372011065483, 0.13533659279346466, 0.14917439222335815, 0.0598544217646122, 0.16099070012569427, 0.06139180809259415, -0.24226419627666473, 0.0717514306306839, 0.14673717319965363, -0.04542732611298561, 0.010947526432573795, 0.06482171267271042, 0.11830328404903412, 0.1022295281291008, -0.29227855801582336, 0.19561409950256348, 0.036290667951107025, 0.10403750091791153, -0.00859000626951456, -0.10664793103933334, 0.2975868582725525, 0.552247941493988, -0.46724721789360046, 0.07517696917057037, 0.0715513527393341, -0.047059882432222366, 0.13012097775936127, -0.09668860584497452, -0.21834495663642883, 0.03172594681382179, -0.1224948838353157, -0.09849369525909424, 0.013571962714195251, 0.0034300265833735466, -0.17341937124729156, 0.1312016099691391, 0.20813360810279846, 0.10314654558897018, 0.2468613088130951, 0.09093679487705231, 0.10203087329864502, -0.2456960380077362, -0.2514178454875946, -0.18465346097946167, -0.1498902142047882, -0.03580593317747116, 0.08461683243513107, -0.0006586553063243628, 0.5949753522872925, 0.14165900647640228, 0.1625135838985443, -0.006949069909751415, -0.06246544048190117, -0.00028190913144499063, -0.11991438269615173, -0.04193628951907158, -0.06952331215143204, 0.20920827984809875, -0.1057458147406578, 0.046941448003053665, -0.13279688358306885, 0.1717553436756134, 0.1639775186777115, -0.015425769612193108, 0.1755506545305252, -0.01686210185289383, -0.06358611583709717, 0.10559795051813126, -0.19137772917747498, 0.25459200143814087, 0.06346885114908218, 0.9036686420440674, 0.02398480288684368, 0.1973147988319397, 0.07620291411876678, -0.5488176941871643, 0.11005258560180664, 0.10293637216091156, -0.10208804160356522, 0.1164490357041359, -0.11841949820518494, -0.061322446912527084, 0.018436115235090256, 0.022870097309350967, 0.13118824362754822, 0.040751297026872635, 0.22772592306137085, 0.19854333996772766, 0.08086393773555756, -0.1333560049533844, 0.7559105157852173, -0.13534308969974518, 0.019866665825247765, 0.0700300857424736, 0.1365208625793457, -0.051371295005083084, -0.1077364832162857, -0.13930995762348175, -0.0599895715713501, -0.11562544107437134, -0.2478330135345459, -0.10010915249586105, 0.03089774027466774, -0.12649819254875183, -0.04620908573269844, 0.11014717817306519, -0.04969263821840286, 1.0526455640792847, 0.23638315498828888, 0.026545163244009018, 0.05107760429382324, 0.10764535516500473, 0.12211740016937256, 0.07334145903587341, -0.00446389289572835, 0.06259845197200775, 0.011424693278968334, -0.1843453347682953, 0.13518008589744568, 0.18232055008411407, -0.042984433472156525, 0.024526886641979218, -0.0071172756142914295, 0.06657469272613525, 0.09048233926296234, -0.16478487849235535, -0.04143023118376732, 0.13292671740055084, 0.03184891492128372, -0.7014108300209045, 0.09668384492397308, -0.05508294329047203, 0.04383660480380058, 0.20652073621749878, -0.06461852788925171, -0.230263814330101, -0.07614465802907944, 0.3184964656829834, 0.1366632580757141, 0.0766044408082962, -0.12462374567985535, -0.12288132309913635, -0.07332617044448853, 0.17577920854091644, 0.12563827633857727, -0.018648812547326088, 0.04812198504805565, -0.14978821575641632, 0.19991940259933472, 0.3222958743572235, 0.08404771238565445, 0.23359836637973785, 0.22805729508399963, 0.30151817202568054, 0.23059846460819244, -0.23631881177425385, 0.14536763727664948, 0.03582177311182022, -0.011799263767898083, 0.08513657003641129, -0.05516097694635391, -0.0010567419230937958, 0.01551414281129837, 0.1010451465845108, 0.08427987992763519, -0.032723624259233475, -0.03431139141321182, 0.10978620499372482, 0.013465614058077335, 0.2616085112094879, 0.16259005665779114, 0.007394095417112112, -0.054159149527549744, 0.18618223071098328, -0.021604051813483238, -0.13664071261882782, 0.15484924614429474, -1.114260196685791, -0.09255217015743256, -0.150362029671669, 0.10356032103300095, -0.11431733518838882, 0.11052753031253815, -0.23798763751983643, -0.11784473061561584, 0.06017211079597473, 0.06543270498514175, 0.014401035383343697, 0.03486153855919838, -0.009327865205705166, -0.08877372741699219, 0.07055442035198212, 0.007601728662848473, -0.49325481057167053, -0.01288359984755516, -0.014851824380457401, 0.021932240575551987, -0.070746049284935, 0.07173822075128555, -0.6186736822128296, -0.24001555144786835, -0.1263154000043869, 0.020288772881031036, 0.031523704528808594, -0.0047748517245054245, 0.13927286863327026, 0.06948943436145782, 0.06816467642784119, 0.08762255311012268, 0.1053456962108612, 0.0018840389093384147, 0.19601546227931976, 0.09798485040664673, -0.010989036411046982, 0.08550485968589783, 0.24333767592906952, 0.2737734317779541, 0.046426158398389816, 0.34961146116256714, 0.04133942723274231, 0.08023710548877716, -0.01787559688091278, -0.0794718936085701, 0.24315613508224487, -1.2990648746490479, 0.08658092468976974, 0.12367833405733109, 0.2519993484020233, -0.08053990453481674, 0.11181644350290298, -0.10090555995702744, -0.2099817991256714, 0.3210836946964264, -0.14471988379955292, -0.00426506670191884, -0.2376311719417572, -0.23223677277565002, 0.0526169016957283, 0.04388944432139397, -0.12047185003757477, 0.5245551466941833, 0.24377785623073578, -0.027087949216365814, 0.11108861118555069, 0.34955909848213196, -0.03127533942461014, -0.04248346760869026, 0.004625745117664337, 0.2592698633670807, 0.04236878454685211, 0.12087925523519516, 0.06612562388181686, 0.011113086715340614, -0.0634494349360466, 0.010739226825535297, 0.10765198618173599, 0.8526321649551392, 0.34834790229797363, 0.020113129168748856, 0.007320327218621969, 0.053625281900167465, 0.18152177333831787, -0.051044292747974396, 0.2680436670780182, -0.05812135711312294, -0.3479052782058716, -0.22538644075393677, 0.20293886959552765, -0.16643691062927246, 0.0780436098575592, -3.0136191844940186, -0.0927017405629158, 0.00024295749608427286, 0.38275372982025146, 0.19686579704284668, 0.1968500316143036, 0.17080847918987274, 0.08802902698516846, 0.13146907091140747, 0.07391877472400665, -0.24193842709064484, 1.4558964967727661, 0.19380387663841248, -0.14377161860466003, -0.14984393119812012, 0.12691937386989594, -0.1778971403837204, 0.062859907746315, 0.021760528907179832, -0.06198273226618767, -0.06930188834667206, -0.04429561272263527, 0.018568959087133408, -0.07124610245227814, 0.19855953752994537, -0.09196225553750992, -0.27120038866996765, -0.030319204553961754, -0.030473966151475906, 0.0976199358701706, -0.14971381425857544, 0.11234097182750702, 0.06487865000963211, 0.25760066509246826, 0.19012205302715302, -0.040539391338825226, -0.10177810490131378, 0.0847669243812561, -0.03171961382031441, 0.13029538094997406, -0.03099048137664795, -0.03300574794411659, -0.21379731595516205, 1.0768510103225708, 0.21650119125843048, -0.2287798523902893, 0.11582594364881516, -0.13233378529548645, 0.04885410517454147, -0.268283486366272, -0.00415433757007122, -0.010069411247968674, 0.041968245059251785, 0.22821548581123352, -0.17108625173568726, -0.0002718085306696594, 0.007501760497689247, -0.004065374843776226, -0.07860611379146576, -0.07059704512357712, 0.14434024691581726, 0.16371499001979828, 0.16337352991104126, 0.04942641407251358, -0.06155156344175339, -0.0066118426620960236, 0.04072374477982521, 0.037810876965522766, 0.15541303157806396, 0.07265131175518036, -0.40169665217399597, 0.03473773971199989, 0.1241707056760788, 0.2250819206237793, 0.1772988736629486, 0.007892053574323654, 0.025328097864985466, -0.2533957362174988, -0.04060492292046547, 0.016430048272013664, 2.3855881690979004, 0.010697080753743649, 0.1009281799197197, -0.18615604937076569, 0.10835379362106323, 0.04297171160578728, 0.11062152683734894, -0.40139439702033997, -0.053486138582229614, -0.09042327105998993, -0.8064281940460205, 0.019155189394950867, -0.085041843354702, -0.04772993549704552, -0.06539168208837509, -0.06284204870462418, 0.028300529345870018, 0.01986808516085148, 0.03274732828140259, 0.20445159077644348, -0.0727270171046257, 0.03830054774880409, 0.017316222190856934, 0.18921920657157898, -0.13147591054439545, -0.12268750369548798, 0.10003738105297089, -0.1906248778104782, 1.7822238206863403, 0.1206187829375267, 0.08617936074733734, 0.023826532065868378, 0.07733946293592453, -0.031185690313577652, 1.3047239780426025, -0.32383987307548523, 0.01737128756940365, -0.020766548812389374, 0.07776961475610733, 0.25921395421028137, -0.17001695930957794, -0.18313485383987427, -0.03091002069413662, 0.22724363207817078, 0.23229312896728516, -0.5283453464508057, -0.11016305536031723, -0.019952058792114258, -0.17550057172775269, 2.042447566986084, 0.09368857741355896, -0.006777285132557154, 0.018436778336763382, 0.04023877903819084, 0.133881613612175, -0.068699911236763, 0.1710049957036972, 0.0702514797449112, -0.30020514130592346, 0.11922767013311386, -0.1604119837284088, -0.18455520272254944, -0.07134006172418594, -0.007721440400928259, 0.16034968197345734, 0.0590018592774868, -0.01028922013938427, -0.07523156702518463, 0.23960834741592407, 0.16343781352043152, 0.18971772491931915, 0.059112273156642914, 0.3484516739845276, -0.02429223619401455, -0.1255527138710022, 0.3698309659957886, 0.2351343035697937, -0.05096381902694702, 0.06948117911815643, -0.48164957761764526, -0.07132791727781296, -0.10910650342702866, 0.004644968546926975, -0.7036786675453186, -0.10545079410076141, -0.03259032219648361, -0.07465822994709015, -0.10871358215808868, 0.12677182257175446, -0.07570789754390717, -0.07952741533517838, -0.026477813720703125, 0.06984280794858932, -0.22945941984653473, 0.015926821157336235, -0.14885887503623962, -0.011332892812788486, 0.08493077754974365, -0.15522490441799164, -0.06845367699861526, -0.23213626444339752, -0.07301405817270279, -0.11880737543106079, 0.003320008981972933, 0.17502792179584503, 0.41226688027381897, 0.04537675902247429, 0.014825945720076561, 0.09765269607305527, 0.02394934371113777, -0.1532425433397293, 0.1607235074043274, 0.08105559647083282, -0.22956547141075134, -0.0916622057557106, -0.06985598057508469, 0.18260440230369568, 0.3137093484401703, 0.09711585938930511, 0.17369917035102844, 0.0399029515683651, -0.2348783016204834])

import os
class Data2Vec2Encoder(nn.Module):
    def __init__(
        self,
        model_ckpt_path: str = '/path/to/your/latentforcing/checkpoint/',
        match_pixel_norm: float = 0.485,
    ):
        super().__init__()

        self.register_buffer("latent_std", d2v2_12_std.clone().float())
        self.register_buffer("latent_mean", d2v2_12_mean.clone().float())

        self.register_buffer("pixel_std", torch.tensor((0.229, 0.224, 0.225)))
        self.register_buffer("pixel_mean", torch.tensor((0.485, 0.456, 0.406)))

        self.match_pixel_norm = match_pixel_norm

        if not os.path.exists(model_ckpt_path): # TODO should be home dir local
            assert False, "Download D2V2 https://dl.fbaipublicfiles.com/fairseq/data2vec2/large_imagenet.pt"
        state_dict = torch.load(model_ckpt_path, map_location="cpu", weights_only=False)
        d2v2_model = Data2VecMultiModel(d2v2_config, [Modality.IMAGE])
        d2v2_model.load_state_dict(state_dict['model'])
        
        d2v2_pos_196 = d2v2_model.modality_encoders["IMAGE"].fixed_positional_encoder.positions
        d2v2_pos_256 = d2v2_pos_196.clone().reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
        d2v2_pos_256 = torch.nn.functional.interpolate(d2v2_pos_256, size=(16, 16), mode='bicubic', align_corners=False)
        d2v2_pos_256 = d2v2_pos_256.permute(0, 2, 3, 1).flatten(1, 2) # Returns (1, 256, 1024)
        d2v2_model.modality_encoders["IMAGE"].fixed_positional_encoder.positions = nn.Parameter(d2v2_pos_256)

        d2v2_model.requires_grad_(False)
        d2v2_model.eval()

        self.d2v2_model = d2v2_model

    @torch.compile()
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # normalize input
        # x : b c h w
        
        x = (x - self.pixel_mean.view(1,3,1,1)) / self.pixel_std.view(1,3,1,1)

        z = self.d2v2_model(x, mode=None, mask=False, features_only=True, remove_extra_tokens=True, out_layer=-12)
        z = z['xs'][12][:,1:]
        z = (z - self.latent_mean.view(1,1,-1)) / self.latent_std.view(1,1,-1)
        z = z.clamp(-5, 5)

        z = z * self.match_pixel_norm
        z = z.view(-1,16,16,1024).permute(0,3,1,2) # b hw d --> b d h w

        return z
