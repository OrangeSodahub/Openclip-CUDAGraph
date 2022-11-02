from typing import Callable, List

import torch
import torchdynamo

from modeling.openclip import CLIPTextTransformer
from kernl.implementations.cuda_graph import cuda_graphs_wrapper
from kernl.optimizer.dynamo_backend import dynamo_backend_ofi
from torch.fx._symbolic_trace import symbolic_trace


def optimize_model(original_model: CLIPTextTransformer, example_inputs: List[torch.Tensor], pool = torch.cuda.graph_pool_handle()) -> Callable:
    """
    Optimizes a given model. Optimization is done in two steps:
    *  first step is to convert the given model to fx graph.
    *  second step is to replace patterns found in the graph in order to optimize the model.

    @return: returns the optimized model (and the original model is modified, so it can not be used after optimization).

    """
    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = (example_inputs,)
    def compiler(model: CLIPTextTransformer, example_inputs: List[torch.Tensor]):
        # Generate gm
        gm: torch.fx.GraphModuel = symbolic_trace(model)
        print("Called with FX Graph:")
        gm.graph.print_tabular()
        # Optimize model
        # dynamo_backend_ofi(gm)
        # Build CUDAGraph and return callable `run`
        return cuda_graphs_wrapper(gm, example_inputs, pool=pool)

    forward = compiler(original_model, example_inputs)

    def run(*args, **kwargs):
        return forward(*args, **kwargs)

    return run
