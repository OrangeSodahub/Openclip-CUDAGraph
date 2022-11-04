import time
import torch
import logging
import numpy as np

from optimized_model import (ORG_CLIPTextTransformer, OPT_CLIPTextTransformer,
                             ORG_CLIPVisionTransformer, OPT_CLIPVisionTransformer,
                             OPT_CLIPModel)

from clip_server.model.openclip_model import OpenCLIPModel


def benchmark(use_dynamo, N, B):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model: mock input
    name='ViT-L-14::laion2b-s32b-b82k'
    org_model = OpenCLIPModel(name=name, device='cuda')
    example_input_image = None
    example_input_text = None
    if not use_dynamo:
        example_input_text = torch.randint(0, 10, (B, 77)).long().cuda()
        example_input_image = torch.randint(0, 10, (B, 3, 224, 224)).float().cuda()
    opt_model = OPT_CLIPModel(
        name=name,
        device='cuda',
        batch_size=B,
        example_inputs_text=example_input_text,     # `None` if use dynamo
        example_inputs_image=example_input_image,   # `None` if use dynamo
    )

    # Benchmark
    complete_time_baseline = 0
    complete_time_optimized = 0
    
    inputs_text = torch.randint(0, 10, (N, B, 77)).long().cuda()
    inputs_image = torch.randint(0, 10, (N, B, 3, 224, 224)).half().cuda()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        for input in inputs_image:

            torch.cuda.synchronize()
            start = time.perf_counter()
            _1 = org_model.encode_image(input)
            torch.cuda.synchronize()
            complete_time_baseline += time.perf_counter() - start

            torch.cuda.synchronize()
            start = time.perf_counter()
            _2 = opt_model.encode_image(input)
            torch.cuda.synchronize()
            complete_time_optimized += time.perf_counter() - start
    
    print(f"{complete_time_baseline=:.5f}s")
    print(f"{complete_time_optimized=:.5f}s")
    return complete_time_baseline, complete_time_optimized
    

def show_diff(a, b):
    from matplotlib import pyplot as plt
    print(a)
    print(b)
    
    a = a.cpu().numpy()[0]
    b = b.cpu().numpy()[0]
    plt.plot(np.arange(768), a-b)
    plt.show()
    

if __name__ == "__main__":
    complete_time_baseline = []
    complete_time_optimized = []
    saved_time_percent = []
    # warm up
    for _ in range(2):
        _, _ = benchmark(True, 1, 1)
    # benchmark
    for N in [10, 1000, 10000]:
        for B in [1, 2, 4, 8]:
            print(f"Runing on N={N}, B={B}")
            complete_time_baseline_, complete_time_optimized_ = benchmark(True, N, B)
            complete_time_baseline.append(complete_time_baseline_)
            complete_time_optimized.append(complete_time_optimized_)
            saved_time_percent_ = (complete_time_optimized_-complete_time_baseline_)/complete_time_baseline_
            saved_time_percent.append(saved_time_percent_)
            print(f"Saved time:{saved_time_percent_}")
    print(complete_time_baseline)
    print(complete_time_optimized)
    print(saved_time_percent)