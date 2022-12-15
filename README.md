## Openclip-CUDAGraph
Tips: There are three steps from the original model to final callable optimized model:
1. Trace the torch model through `torch.fx._symbolic_trace` to generate `torch.fx.GraphModuel` object.
2. Replace some modules in this object to optimize it **(not included in this version)** and output is a `torch.fx.GraphModuel` object.
3. Build static graph model through `torch.cuda.CUDAGraph`.
And also not use `torchdynamo.optimize()`.

### Benchmark

Tested Model (on RTX3080): `ViT-L-14::laion2b-s32b-b82k`

Using dynamo:

<p float="left">
    <img src=./assets/1.png width=60%>
</p>

Using symbolic_trace: we need to specify whethter TextModel or VisionModel, if not it generate both example inputs and build graphs and causes a large GPU memory occupancxy.

**graph**: `[torch dynamo]/[symbolic trace]` for torch dynamo's time, it includes building graphs and inference while for symbolic trace, building graph's cost is not been counted. It saves the time cost on starting the model multiply times. When N gets larger and batch_size gets smaller, it performs better.

**GPU**: `[pytorch]/[torch dynamo]/[symbolic trace]`

**RES**: `[torch dynamo]/symbolic trace]` Since data needs to be created on the CPU and the moved to the GPU, the value of RES will get a peak when the program starts and then decrease. Here record the peak value.

-------------------------------------------------------------------------------------------------------
times   |shape           | pt (s)   | graph (s)         |    speed up     | GPU(MB)           | RES(MB)
--------|----------------|----------|-------------------|-----------------|-------------------|---------
|1      |(1, 77)         |0.0068    |0.00182/0.00309    |3.6558/2.1231    |1857/3011/2993     |2515/3539
|       |(2, 77)         |0.0073    |0.00216/0.00419    |3.0058/1.6446    |1859/3005/3132     |2511/3534
|       |(4, 77)         |0.0070    |0.00294/0.00753    |2.2076/0.9188    |1859/3087/3043     |2510/3543
|       |(8, 77)         |0.0071    |0.00450/0.01200    |2.3033/0.5882    |1863/3369/3315     |2512/3538
|       |(16, 77)        |0.0087    |0.00791/0.02126    |1.0675/0.4427    |1883/3619/3591     |2511/3548
|       |(1, 3, 224, 224)|0.0134    |                   |                 |              |
|       |(2, 3, 224, 224)|0.0352    |                   |                 |              |
|       |(4, 3, 224, 224)|0.0323    |                   |                 |              |
|       |(8, 3, 224, 224)|0.0416    |                   |                 |              |
|100    |(1, 77)         |0.6099    |1.4387/0.16603     |3.6735/2.3807    |              |
|       |(2, 77)         |0.6519    |1.4735/0.19833     |3.2871/2.0116    |              |
|       |(4, 77)         |0.6750    |1.5138/0.28739     |2.3488/0.9679    |              |
|       |(8, 77)         |0.6821    |1.6625/0.38165     |1.7872/0.6017    |              |
|       |(16, 77)        |0.8649    |2.1333/0.76007     |1.1379/0.4570    |              |
|       |(1, 3, 224, 224)|0.1328    |                   |                 |              |
|       |(2, 3, 224, 224)|0.1507    |                   |                 |              |
|       |(4, 3, 224, 224)|0.3066    |                   |                 |              |
|       |(8, 3, 224, 224)|0.3949    |                   |                 |              |
|1000   |(1, 77)         |6.3198    |1.66601/2.76445    |3.7943/2.3402    |              |
|       |(2, 77)         |6.4785    |1.97298/3.82585    |3.2836/1.8046    |              |
|       |(4, 77)         |6.5346    |2.69592/7.11847    |2.4238/0.9867    |              |
|       |(8, 77)         |6.5828    |3.75718/11.70292   |1.1383/0.6002    |              |
|       |(16, 77)        |8.1642    |7.47373/20.32973   |1.0923/0.4300    |              |
|       |(1, 3, 224, 224)|13.0253   |                   |                 |              |
|       |(2, 3, 224, 224)|15.0432   |                   |                 |              |
|       |(4, 3, 224, 224)|-         |-                  |-                |              |
|10000  |(1, 77)         |62.5197   |16.89150/28.17511  |3.7012/2.2981    |              |
|       |(2, 77)         |64.2329   |19.49905/38.36930  |3.2941/1.7738    |              |
|       |(4, 77)         |65.2058   |27.15045/71.31565  |2.4016/0.9664    |              |
|       |(8, 77)         |65.7108   |37.89167/118.55823 |1.7252/0.5885    |              |
|       |(16, 77)        |81.8811   |74.14199/205.51851 |1.0796/0.4290    |              |
|       |(1, 3, 224, 224)|-         |-                  |-                |              |
|       |(2, 3, 224, 224)|-         |-                  |-                |              |
|       |(4, 3, 224, 224)|-         |-                  |-                |              |
