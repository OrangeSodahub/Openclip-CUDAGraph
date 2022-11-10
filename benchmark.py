import time
import torch
import logging
import numpy as np

from optimized_model import (ORG_CLIPTextTransformer, OPT_CLIPTextTransformer,
                             ORG_CLIPVisionTransformer, OPT_CLIPVisionTransformer,
                             OPT_CLIPModel)

from clip_server.model.openclip_model import OpenCLIPModel


def benchmark(use_dynamo = False, pt = True, N = 1, B = 1, mode = 'text'):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model: mock input
    name='ViT-L-14::laion2b-s32b-b82k'
    example_inputs = None
    if not use_dynamo:
        if mode == 'text':
            example_inputs = torch.randint(0, 10, (B, 77)).long().cuda()
        elif mode == 'image':
            example_inputs = torch.randint(0, 10, (B, 3, 224, 224)).float().cuda()
    opt_model = OPT_CLIPModel(
        name=name,
        device='cuda',
        batch_size=B,
        example_inputs=example_inputs     # `None` if use dynamo
    )
    if pt:
        org_model = OpenCLIPModel(name=name, device='cuda')

    # Benchmark
    complete_time_baseline = 0
    complete_time_optimized = 0
    
    if mode == 'text':
        input = torch.randint(0, 10, (B, 77)).long().cuda()
    elif mode == 'image':
        input = torch.randint(0, 10, (B, 3, 224, 224)).half().cuda()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        # set up encode fn
        if mode == 'text':
            org_encode = org_model.encode_text if pt else None
            opt_encode = opt_model.encode_text
        elif mode == 'image':
            org_encode = org_model.encode_image if pt else None
            opt_encode = opt_model.encode_image
        
        # warm up
        for _ in range(5):

            _1 = org_encode(input) if pt else None
            _2 = opt_encode(input)
        
        # benchmark
        for _ in range(N):

            if pt:
                torch.cuda.synchronize()
                start = time.perf_counter()
                _1 = org_encode(input)
                torch.cuda.synchronize()
                complete_time_baseline += time.perf_counter() - start

            torch.cuda.synchronize()
            start = time.perf_counter()
            _2 = opt_encode(input)
            torch.cuda.synchronize()
            complete_time_optimized += time.perf_counter() - start
    
    print(f"{complete_time_baseline=:.5f}s")
    print(f"{complete_time_optimized=:.5f}s")
    if pt:
        mean_diff = (_1-_2).abs().mean().cpu().numpy()
    else:
        mean_diff = 0.
    print(f"{mean_diff=:.5f}")
    return complete_time_baseline, complete_time_optimized, mean_diff
    

if __name__ == "__main__":
    import time
    complete_time_baseline = []
    complete_time_optimized = []
    mean_diff = []
    speed_up = []
    # benchmark
    for N in [10]:
        for B in [1, 2, 4, 8, 16]:
            print(f"Runing on N={N}, B={B}")
            complete_time_baseline_, complete_time_optimized_, mean_diff_ = benchmark(True, False, N, B, 'image')
            complete_time_baseline.append(complete_time_baseline_)
            complete_time_optimized.append(complete_time_optimized_)
            mean_diff.append(mean_diff_)
            speed_up_ = complete_time_baseline_/complete_time_optimized_
            speed_up.append(speed_up_)
            print(f"Speed up:{speed_up_}\n")
            import time
            time.sleep(10)
    print(complete_time_baseline)
    print(complete_time_optimized)
    print(mean_diff)
    print(speed_up)
