# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt

import torch

from clip_server.model.pretrained_models import get_model_url_md5, download_model
from clip_server.model.model import load_openai_model, load_openclip_model

from model_optimization import optimize_model
from modeling.openclip import CLIPTextTransformer


# TODO: CLIP
# class OpenCLIPModel(CLIPModel):
#     def __init__(self, name: str, device: str = 'cpu', jit: bool = False, **kwargs):
#         super().__init__(name, **kwargs)

#         if '::' in name:
#             model_name, pretrained = name.split('::')
#         else:
#             model_name = name
#             pretrained = 'openai'

#         self._model_name = model_name

#         model_url, md5sum = get_model_url_md5(name)
#         model_path = download_model(model_url, md5sum=md5sum)

#         if pretrained == 'openai':
#             self._model = load_openai_model(model_path, device=device, jit=jit)
#         else:
#             self._model = load_openclip_model(
#                 self._model_name, model_path=model_path, device=device, jit=jit
#             )

#     @staticmethod
#     def get_model_name(name: str):
#         if '::' in name:
#             model_name, pretrained = name.split('::')
#         else:
#             model_name = name
#         if model_name == 'ViT-L/14@336px':
#             return 'ViT-L-14-336'
#         return model_name.replace('/', '-')

#     def encode_text(self, input_ids: 'torch.Tensor', **kwargs):
#         return self._model.encode_text(input_ids)

#     def encode_image(self, pixel_values: 'torch.Tensor', **kwargs):
#         return self._model.encode_image(pixel_values)


# class OpenCLIPModel_opt(OpenCLIPModel):
#     def __init__(self, name: str, device: str = 'cpu', jit: bool = False, example_inputs = None, **kwargs):
#         super().__init__(name, device, jit, **kwargs)
#         self._model = self._model.eval().cuda()
#         print(f"Load Model: {type(self._model)}")
        
#         self._model_opt = optimize_model(original_model=self._model, example_inputs=example_inputs)
#         print(f"Load Model: {type(self._model_opt)}")

class OPT_CLIPTextTransformer():
    def __init__(self, input):
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
            quick_gelu=quick_gelu
        )
        # optimize
        self._run = optimize_model(
            original_model=self._model,
            example_inputs=input,
        )

    def __call__(self, *args, **kwargs):
        return self._run(*args, **kwargs)