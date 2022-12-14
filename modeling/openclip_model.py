# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt

import torch
from clip_server.model.pretrained_models import get_model_url_md5, download_model
from clip_server.model.model import load_openai_model

from modeling.clip_model import CLIPModel
from modeling.builder import load_openclip_model_seperately


class OpenCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str = 'cpu', jit: bool = False, batch_size: int = 1, **kwargs):
        super().__init__(name, **kwargs)

        if '::' in name:
            model_name, pretrained = name.split('::')
        else:
            model_name = name
            pretrained = 'openai'

        self._model_name = model_name

        model_url, md5sum = get_model_url_md5(name)
        model_path = download_model(model_url, md5sum=md5sum)

        if pretrained == 'openai':
            self._model = load_openai_model(model_path, device=device, jit=jit)
        else:
            self._model_text, self._model_vision = load_openclip_model_seperately(
                self._model_name, model_path=model_path, device=device, jit=jit, batch_size=batch_size
            )
