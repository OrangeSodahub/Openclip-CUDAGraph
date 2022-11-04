import time
import torch
import logging
import numpy as np

from optimized_model import (ORG_CLIPTextTransformer, OPT_CLIPTextTransformer,
                             ORG_CLIPVisionTransformer, OPT_CLIPVisionTransformer,
                             OPT_CLIPModel)

from clip_server.model.openclip_model import OpenCLIPModel


N = 1        # times
B = 8        # batch_size

def benchmark(mode):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model: mock input
    if mode == 'text':
        example_input = torch.randint(0, 10, (B, 77)).long().cuda()
        opt_model = OPT_CLIPTextTransformer(example_input, B)
        org_model = ORG_CLIPTextTransformer()
    elif mode == 'image':
        example_input = torch.randint(0, 10, (B, 3, 224, 224)).float().cuda()
        opt_model = OPT_CLIPVisionTransformer(example_input, B)
        org_model = ORG_CLIPVisionTransformer()
    elif mode == 'all':
        name='ViT-L-14::laion2b-s32b-b82k'
        example_input_text = torch.randint(0, 10, (B, 77)).long().cuda()
        example_input_image = torch.randint(0, 10, (B, 3, 224, 224)).float().cuda()
        opt_model = OPT_CLIPModel(
            name=name,
            device='cuda',
            batch_size=B,
            example_inputs_text=example_input_text,
            example_inputs_image=example_input_image,
            mode='image', # TODO: Remove this
        )
        org_model = OpenCLIPModel(name=name, device='cuda')

    # Benchmark
    complete_time_baseline = 0
    score_baseline = 0
    complete_time_optimized = 0
    score_optimize = 0
    
    if mode == 'text':
        inputs = torch.randint(0, 10, (N, B, 77)).long().cuda()
    elif mode == 'image':
        inputs = torch.randint(0, 10, (N, B, 3, 224, 224)).cuda()
    elif mode == 'all':
        inputs_text = torch.randint(0, 10, (N, B, 77)).long().cuda()
        inputs_image = torch.randint(0, 10, (N, B, 3, 224, 224)).cuda()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        for input in inputs_image:

            torch.cuda.synchronize()
            start = time.perf_counter()
            _1 = org_model.encode_image(input)
            torch.cuda.synchronize()
            complete_time_baseline += time.perf_counter() - start

            start = time.perf_counter()
            _2 = opt_model.encode_image(input)
            torch.cuda.synchronize()
            complete_time_optimized += time.perf_counter() - start
    
    print(f"{complete_time_baseline=:.5f}s")
    print(f"{complete_time_optimized=:.5f}s")
    show_diff(_1, _2)
    

def show_diff(a, b):
    from matplotlib import pyplot as plt
    print(a)
    print(b)
    
    a = a.cpu().numpy()[0]
    b = b[0].detach().cpu().numpy()[0]
    plt.plot(np.arange(768), a-b)
    plt.show()
    

if __name__ == "__main__":
    benchmark('all')
