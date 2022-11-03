# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt

from model_optimization import optimize_model
from modeling.openclip import CLIPTextTransformer, CLIPVisionTransformer


# TODO: `batch_size`
class ORG_CLIPTextTransformer():
    def __init__(self, batch_size = 1, **kwargs):
        # initialize
        embed_dim = 768
        text_cfg = {
            'layers': 12,
            'context_length': 77,
            'vocab_size': 49408,
            'width': 768,
            'heads': 12,
        }
        quick_gelu = False
        self._model = CLIPTextTransformer(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            batch_size=batch_size
        )

    def __call__(self, text):
        return self._model(text)


class ORG_CLIPVisionTransformer():
    def __init__(self, batch_size, **kwargs):
        # initialize
        embed_dim = 768
        vision_cfg = {
            'image_size': 224,
            'layers': 24,
            'width': 1024,
            'patch_size': 14,
        }
        quick_gelu = False
        self._model = CLIPVisionTransformer(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            batch_size=batch_size
        )

    def __call__(self, image):
        return self._model(image)


class OPT_CLIPTextTransformer(ORG_CLIPTextTransformer):
    def __init__(self, input, batch_size, **kwargs):
        # initialize
        super().__init__(batch_size, **kwargs)
        # optimize
        self._run = optimize_model(
            original_model=self._model,
            example_inputs=input,
        )

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)


class OPT_CLIPVisionTransformer(ORG_CLIPVisionTransformer):
    def __init__(self, input, batch_size, **kwargs):
        # initialize
        super().__init__(batch_size, **kwargs)
        # optimize
        self._run = optimize_model(
            original_model=self._model,
            example_inputs=input,
        )

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)
