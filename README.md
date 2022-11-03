## Openclip-kernl
Tips: There are three steps from the original model to final callable optimized model:
1. Trace the torch model through `torch.fx._symbolic_trace` to generate `torch.fx.GraphModuel` object.
2. Replace some modules in this object to optimize it **(not included in this version)** and output is a `torch.fx.GraphModuel` object.
3. Build static graph model through `torch.cuda.CUDAGraph`.
And also not use `torchdynamo.optimize()`.

### Benchmark
textclip model (1000 * shape=(1, 77)):
```
complete_time_baseline=6.39887s
complete_time_optimized=2.46688s
```
visionclip model (1000 * shape=(1, 3, 224, 224)):
```
TBD...
```