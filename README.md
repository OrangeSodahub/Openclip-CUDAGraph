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

-------------------------------------------------------------------------------------------------
times   |shape           | pt (s)   | graph (s)         | speed up | GPU(MB)           | RES(MB)
--------|----------------|----------|-------------------|----------|-------------------|---------
|1      |(1, 77)         |0.0068    |1.1747/0.00182     |0.00575/3.6558   |1857/3105/3011     |2515/3539
|       |(2, 77)         |0.0073    |1.1755/0.00216     |0.00605/3.0058   |1859/3518/3005     |2511/3534
|       |(4, 77)         |0.0070    |1.1780/0.00294     |0.00583/2.2076   |1859/3796/3087     |2510/3543
|       |(8, 77)         |0.0071    |1.2095/0.00450     |0.00575/2.3033   |1863/4253/3369     |2512/3538
|       |(16, 77)        |0.0087    |1.1905/0.00791     |0.00796/1.0675   |1883/4849/3619     |2511/3548
|       |(1, 3, 224, 224)|0.0134    |3.9472             |0.00339          |              |
|       |(2, 3, 224, 224)|0.0352    |4.0213             |0.00875          |              |
|       |(4, 3, 224, 224)|0.0323    |4.0170             |0.00804          |              |
|       |(8, 3, 224, 224)|0.0416    |4.1213             |0.01009          |              |
|100    |(1, 77)         |0.6099    |1.4387/0.16603     |0.44370/3.6735   |              |
|       |(2, 77)         |0.6519    |1.4735/0.19833     |0.44843/3.2871   |              |
|       |(4, 77)         |0.6750    |1.5138/0.28739     |0.43349/2.3488   |              |
|       |(8, 77)         |0.6821    |1.6625/0.38165     |0.42214/1.7872   |              |
|       |(16, 77)        |0.8649    |2.1333/0.76007     |0.37915/1.1379   |              |
|       |(1, 3, 224, 224)|0.1328    |3.9772             |0.03339          |              |
|       |(2, 3, 224, 224)|0.1507    |4.0623             |0.03709          |              |
|       |(4, 3, 224, 224)|0.3066    |4.2564             |0.07203          |              |
|       |(8, 3, 224, 224)|0.3949    |4.4650             |0.08844          |              |
|1000   |(1, 77)         |6.3198    |3.7619/1.66601     |1.67443/3.7943   |              |
|       |(2, 77)         |6.4785    |4.1487/1.97298     |1.55306/3.2836   |              |
|       |(4, 77)         |6.5346    |4.3992/2.69592     |1.47605/2.4238   |              |
|       |(8, 77)         |6.5828    |5.7641/3.75718     |1.13833/1.7520   |              |
|       |(16, 77)        |8.1642    |7.8367/7.47373     |0.82446/1.0923   |              |
|       |(1, 3, 224, 224)|13.0253   |11.9307            |1.09174          |              |
|       |(2, 3, 224, 224)|15.0432   |17.7647            |0.84680          |              |
|       |(4, 3, 224, 224)|-         |-                  |-                |              |
|10000  |(1, 77)         |62.5197   |18.0050/16.89150   |3.50177/3.7012   |              |
|       |(2, 77)         |64.2329   |21.7872/19.49905   |3.00574/3.2941   |              |
|       |(4, 77)         |65.2058   |24.3028/27.15045   |2.67424/2.4016   |              |
|       |(8, 77)         |65.7108   |37.9319/37.89167   |1.73254/1.7252   |              |
|       |(16, 77)        |81.8811   |77.0026/74.14199   |1.06335/1.0796   |              |
|       |(1, 3, 224, 224)|-         |-                  |-                |              |
|       |(2, 3, 224, 224)|-         |-                  |-                |              |
|       |(4, 3, 224, 224)|-         |-                  |-                |              |
