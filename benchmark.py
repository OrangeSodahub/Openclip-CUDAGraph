import click
import time
import torch
import logging
import numpy as np

from modeling.openclip import OpenCLIPModel_opt


@click.command()
@click.option("--batch-size", default=1, help="batch size")
def benchmark(batch_size):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model
    openclip = OpenCLIPModel_opt(name='ViT-L-14::laion2b-s32b-b82k', device='cuda')
    start = time.perf_counter()
    input = torch.randint(0, 10, (1, 77), dtype=torch.int64).long().cuda()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        # TODO: Modified model.forward()
        _ = openclip._model_opt(input)
    

if __name__ == "__main__":
    benchmark()
