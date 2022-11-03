import click
import time
import torch
import logging
import numpy as np

from openclip_model import (ORG_CLIPTextTransformer, OPT_CLIPTextTransformer,
                            ORG_CLIPVisionTransformer, OPT_CLIPVisionTransformer)

N = 1000
B = 1

def benchmark(mode):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model
    if mode == 'text':
        example_input = torch.randint(0, 10, (B, 77)).long()
        opt_model = OPT_CLIPTextTransformer(example_input, B)
        org_model = ORG_CLIPTextTransformer()
    elif mode == 'image':
        example_input = torch.randint(0, 10, (B, 3, 224, 224)).float()
        opt_model = OPT_CLIPVisionTransformer(example_input, B)
        org_model = ORG_CLIPVisionTransformer()

    # Benchmark
    complete_time_baseline = 0
    score_baseline = 0
    complete_time_optimized = 0
    score_optimize = 0
    
    if mode == 'text':
        inputs = torch.randint(0, 10, (N, B, 77)).long()
    elif mode == 'image':
        inputs = torch.randint(0, 10, (N, B, 3, 224, 224)).float()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        for input in inputs:
            # TODO: devices
            # input = input.to("cuda")

            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = org_model(input)
            torch.cuda.synchronize()
            complete_time_baseline += time.perf_counter() - start

            start = time.perf_counter()
            _ = opt_model(input)
            torch.cuda.synchronize()
            complete_time_optimized += time.perf_counter() - start
    
    print(f"{complete_time_baseline=:.2f}s")
    print(f"{complete_time_optimized=:.2f}s")

if __name__ == "__main__":
    benchmark('text')
