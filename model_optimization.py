from typing import Callable, List

import torch
import torchdynamo

from kernl.implementations.cuda_graph import cuda_graphs_wrapper
from kernl.optimizer.dynamo_backend import dynamo_backend_ofi


def optimize_model(original_model, pool = torch.cuda.graph_pool_handle()) -> Callable:
    """
    Optimizes a given model. Optimization is done in two steps:
    *  first step is to convert the given model to fx graph.
    *  second step is to replace patterns found in the graph in order to optimize the model.

    @return: returns the optimized model (and the original model is modified, so it can not be used after optimization).

        Example:
            from kernl.model_optimization import optimize_model

            model_name = "BaptisteDoyen/camembert-base-xnli"
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            model = model.eval().cuda()
2
            optimized_model = optimize_model(model)
    """
    original_model.forward2 = original_model.forward

    def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
        dynamo_backend_ofi(gm)
        return cuda_graphs_wrapper(gm, example_inputs, pool=pool)

    def run(*args, **kwargs):
        with torchdynamo.optimize(compiler):
            return original_model.forward2(*args, **kwargs)

    def forward_with_exception(*args, **kwargs):
        raise Exception("Original model can not be used after optimization")

    original_model.forward = forward_with_exception

    return run
