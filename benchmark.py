import click
import time
import torch
import logging
import numpy as np

from openclip_model import OPT_CLIPTextTransformer


@click.command()
@click.option("--batch-size", default=1, help="batch size")
def benchmark(batch_size):
    logging.getLogger().setLevel(logging.INFO)
    np.random.seed(0)
    torch.manual_seed(4896)

    # Load Model
    input = torch.randint(0, 10, (1, 77), dtype=torch.int64).long()
    openclip = OPT_CLIPTextTransformer(input)
    start = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        # TODO: Modified model.forward()
        _ = openclip(input)
        print(_)
    

if __name__ == "__main__":
    benchmark()
