""" CLIP Model

Adapted from https://github.com/mlfoundations/open_clip.

Originally MIT License, Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
Ludwig Schmidt
"""

import numpy as np
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from modeling.module import (_ntuple, ResidualAttentionBlock, LayerNorm,
                    QuickGELUActivation, CLIPTextCfg, CLIPVisionCfg,
                    ModifiedResNet)


to_2tuple = _ntuple(2)


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisualTransformer(nn.Module):
    def __init__(
        self,
        batch_size: int,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        output_dim: int,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            self.image_size[0] // self.patch_size[0],
            self.image_size[1] // self.patch_size[1],
        )
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads, mlp_ratio, act_layer=act_layer
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert (
            unlocked_groups == 0
        ), 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        class_embedding = self.class_embedding.unsqueeze(0).unsqueeze(0)
        class_embedding = torch.concat([class_embedding for _ in range(self.batch_size)], dim=0)
        x = torch.cat([class_embedding, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        quick_gelu: bool = False,
        batch_size: int = 1,
    ):
        super().__init__()
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)

        act_layer = QuickGELUActivation if quick_gelu else nn.GELU

        # Vision Transformer
        # layers name: visual....
        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisualTransformer(
            batch_size=batch_size,
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=embed_dim,
            act_layer=act_layer,
        )

    def encode_image(self, image):
        return self.visual(image)

    def forward(self, image):
        return self.encode_image(image)


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        batch_size: int = 1,
    ):
        super().__init__()
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        self.context_length = text_cfg.context_length
        act_layer = QuickGELUActivation if quick_gelu else nn.GELU

        # General Transformer
        self.transformer = Transformer(
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            act_layer=act_layer,
        )

        self.vocab_size = text_cfg.vocab_size
        self.token_embedding = nn.Embedding(text_cfg.vocab_size, text_cfg.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_cfg.width))
        self.ln_final = LayerNorm(text_cfg.width)

        self.text_projection = nn.Parameter(torch.empty(text_cfg.width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        proj_std = (self.transformer.width**-0.5) * (
            (2 * self.transformer.layers) ** -0.5
        )
        attn_std = self.transformer.width**-0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width**-0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # # x.shape = [batch_size, n_ctx, transformer.width]
        # # take features from the eot embedding (eot_token is the highest number in each sequence)
        return x, self.text_projection

    def forward(self, text):
        return self.encode_text(text)
