## Openclip-kernl
Tips: There are three steps from the original model to final callable optimized model:
1. Trace the torch model through `torch.fx._symbolic_trace` to generate `torch.fx.GraphModuel` object.
2. Replace some modules in this object to optimize it **(not included in this version)** and output is a `torch.fx.GraphModuel` object.
3. Build static graph model through `torch.cuda.CUDAGraph`.

### Benchmark
Run and output (1000 inputs to textclip model):
```
complete_time_baseline=52.18s
complete_time_optimized=0.14s
```