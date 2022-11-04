import torch
from typing import Callable, Union, List

from modeling.model import CLIPTextTransformer, CLIPVisionTransformer
from modeling.cuda_graph import cuda_graphs_wrapper
from torch.fx._symbolic_trace import symbolic_trace


def optimize_model(
        original_model: Union[CLIPTextTransformer, CLIPVisionTransformer],
        example_inputs: List[torch.Tensor],
        pool = torch.cuda.graph_pool_handle()
    ) -> Callable:
    """
    Optimizes a given model. Optimization is done in two steps:
    *  first step is to convert the given model to fx graph.
    *  second step is to replace patterns found in the graph in order to optimize the model.

    @return: returns the optimized model (and the original model is modified, so it can not be used after optimization).

    """
    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = (example_inputs,)
        
    def compiler(model: Union[CLIPTextTransformer, CLIPVisionTransformer], example_inputs: List[torch.Tensor]):
        # Generate gm
        gm: torch.fx.GraphModule = symbolic_trace(model)
        # Build CUDAGraph and return callable `run`
        return cuda_graphs_wrapper(gm, example_inputs, pool=pool)

    forward = compiler(original_model, example_inputs)

    def run(*args, **kwargs):
        return forward(*args, **kwargs)

    return run


def optimize_model_dynamo(
        original_model: Union[CLIPTextTransformer, CLIPVisionTransformer],
        pool = torch.cuda.graph_pool_handle()
    ) -> Callable:
    import torchdynamo
    
    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        return cuda_graphs_wrapper(gm, example_inputs, pool=pool)

    @torchdynamo.optimize(compiler)
    def run(*args, **kwargs):
        return original_model.forward(*args, **kwargs)

    return run
