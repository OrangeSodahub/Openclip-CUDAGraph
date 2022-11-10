import torch
from copy import deepcopy
from collections import OrderedDict

from open_clip.factory import _MODEL_CONFIGS
from modeling.model import (CLIPTextCfg, CLIPVisionCfg,
                            CLIPTextTransformer, CLIPVisionTransformer)


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


""" Seperately load model """

def load_openclip_model_seperately(
    model_name: str,
    model_path: str,
    device: torch.device = torch.device('cpu'),
    jit: bool = False,
    batch_size: int = 1,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model_name = model_name.replace(
        '/', '-'
    )  # for callers using old naming with / in ViT names

    if model_name in _MODEL_CONFIGS:
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    if pretrained_image:
        if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert (
                False
            ), 'pretrained image towers currently only supported for timm models'

    # begin building model
    embed_dim = model_cfg['embed_dim']
    quick_gelu = model_cfg.get('quick_gelu', False)
    text_cfg = model_cfg['text_cfg']
    text_model = CLIPTextTransformer(embed_dim, text_cfg, quick_gelu, batch_size)
    text_model.eval()
    vision_cfg = model_cfg['vision_cfg']
    vision_model = CLIPVisionTransformer(embed_dim, vision_cfg, quick_gelu, batch_size)
    vision_model.eval()

    # load params
    state_dict = load_state_dict(model_path)
    text_state_dict = OrderedDict()
    vision_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('visual'):
            vision_state_dict[k] = v
        else:
            text_state_dict[k] = v

        # Must be fp16 on CUDA
        if v.dtype == torch.float32:
            v.data == v.data.half()

    text_model.load_state_dict(text_state_dict)
    vision_model.load_state_dict(vision_state_dict)
    text_model.to(device=device)
    vision_model.to(device=device)
    if jit:
        text_model = torch.jit.script(text_model)
        vision_model = torch.jit.script(vision_model)

    return text_model, vision_model