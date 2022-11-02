import click
import time
import torch
import logging
import numpy as np

from openclip_model import ORG_CLIPTextTransformer, OPT_CLIPTextTransformer

N = 1000

@click.command()
@click.option("--batch-size", default=1, help="batch size")
def benchmark(batch_size):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model
    example_input = torch.randint(0, 10, (1, 77), dtype=torch.int64).long()
    opt_model = OPT_CLIPTextTransformer(example_input)
    org_model = ORG_CLIPTextTransformer()

    # Benchmark
    complete_time_baseline = 0
    score_baseline = 0
    complete_time_optimized = 0
    score_optimize = 0
    inputs = torch.randint(0, 10, (N, 1, 77), dtype=torch.int64).long()
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
    benchmark()
