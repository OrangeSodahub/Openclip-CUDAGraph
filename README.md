## Openclip-CUDAGraph
Tips: There are three steps from the original model to final callable optimized model:
1. Trace the torch model through `torch.fx._symbolic_trace` to generate `torch.fx.GraphModuel` object.
2. Replace some modules in this object to optimize it **(not included in this version)** and output is a `torch.fx.GraphModuel` object.
3. Build static graph model through `torch.cuda.CUDAGraph`.
And also not use `torchdynamo.optimize()`.

### Benchmark

Tested on RTX3080:
--------------------------------------------------------------
shape     | model                      | pt (ms)  | graph (ms)
----------|----------------------------|----------|-----------
(1, 77)   |ViT-L-14::laion2b-s32b-b82k |6.3988    |2.4668    |
(2, 77)   |                            |6.9618    |3.3369    |
(4, 77)   |                            |7.3371    |4.7601    |
(8, 77)   |                            |9.3325    |8.9874    |
(16, 77)  |                            |10.0143   |17.3033   |
(1, 3, 224, 224)|                      |16.98     |16.75     |
(2, 3, 224, 224)|                      |18.66     |28.13     |
(4, 3, 224, 224)|                      |43.23     |50.18     |

