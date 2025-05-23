# Awesome AI Everything

## AI LLM Survey
*   [A Survey on Test-Time Scaling in Large Language Models:What, How, Where, and How Well](https://arxiv.org/pdf/2503.24235)
*   [Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models](https://arxiv.org/pdf/2503.16419)

*   [Thus Spake Long-Context Large Language Model](https://arxiv.org/pdf/2502.17129) |24 Feb 2025| Shanghai AI Lab\&huawei\&fudan
*   [A Survey on Inference Optimization Techniques for Mixture of Experts Models](https://arxiv.org/pdf/2412.14219) |Dec 2024 | CUHK Univresity\&shang hai jiao tong University
*   [A Survey on Mixture of Experts](https://arxiv.org/pdf/2407.06204) | 8 Aug 2024 | UK Univesity
*   [A Survey of Low-bit Large Language Models\:Basics, Systems, and Algorithms](https://arxiv.org/pdf/2409.16694) | 30 Sep 2024 | Beihang University\&ETH Zuric\&SenseTime\&CUHK
*   [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/pdf/2404.14294) |  22 Apr 2024 | Infinigence-AI
*   [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863) | 23 May 2024 | AWS
*   [Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models](https://arxiv.org/abs/2401.00625) | 1 Jan 2024 |
*   [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234) | 23 Dec 2023 | CMU
*   [Challenges and Applications of Large Language Models](https://arxiv.org/pdf/2307.10169.pdf) | 19 Jul 2023 | University College London
*   [Towards Efficient Mixture of Experts: A Holistic Study of Compression Techniques](https://arxiv.org/pdf/2406.02500)
*   [Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding](https://arxiv.org/pdf/2401.07851v2)

## AI State of Art Model

paper\&technical paper

*   [deepseek r1](https://arxiv.org/abs/2501.12948) | 22 Jan 2025 | DeepSeek
*   [deepseek v3](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) |26 Dec 2024 | DeepSeek

## AI benchmark

*   [LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators](https://arxiv.org/abs/2411.00136) | 31 Oct 2024 | Argonne National Laboratory

## AI Algorithm

### Network Architecture

*   [Transformer](https://arxiv.org/pdf/1706.03762): Attention Is All You Need | 2 Aug 2023 | Google
*   [RWKV](https://arxiv.org/pdf/2305.13048): Reinventing RNNs for the Transformer Era | 11 Dec 2023 | Generative AI Commons
*   [Mamba](https://arxiv.org/abs/2312.00752): Linear-Time Sequence Modeling with Selective State Spaces | 31 May 2024 | CMU

### MoE

*   [MoE review](https://arxiv.org/pdf/2209.01667)
*   [DeepSeek MoE](https://arxiv.org/pdf/2401.06066)
*   [MiniMax 01](https://arxiv.org/pdf/2501.08313)
*   [Mixtral 8x7b](https://arxiv.org/pdf/2401.04088)
*   [GROK]()

## AI Chips

Chips Survey

*   2024 [LARGE LANGUAGE MODEL INFERENCE ACCELERATION\:A COMPREHENSIVE HARDWARE PERSPECTIVE](https://arxiv.org/pdf/2410.04466)
*   2023 [Lincoln AI Computing Survey (LAICS) Update](https://arxiv.org/pdf/2310.09145)
*   2022 [AI and ML Accelerator Survey and Trendse](https://arxiv.org/pdf/2210.04055)
*   2021 [AI Accelerator Survey and Trends](https://arxiv.org/pdf/2109.08957)
*   2020 [Survey of Machine Learning Accelerators](https://arxiv.org/pdf/2009.00993)
*   2019 [Survey and Benchmarking of Machine Learning Accelerators](https://arxiv.org/pdf/1908.11348)

### GPU

*   NVIDIA

| #     | FLOPs dense fp16 | HBM   | Bandwidth | L2 cache | NV link | PCIe    | Architecture |
| ----- | ---------------- | ----- | --------- | -------- | ------- | ------- | ------------ |
| GB200 | 5P               | 384GB | 8.0TB/s   |          | 1.8TB/s | 128GB/s | blackwell    |
| GH200 | 985T             | 141GB | 4.8TB/s   | 60MB     | 900GB/s | 128GB/s | hopper       |
| H100  | 985T             | 80GB  | 3.35TB/s  | 50MB     | 900GB/s | 128GB/s | hopper       |
| H800  | 985T             | 80GB  | 3.35TB/s  | 50MB     | 400GB/s | 64GB/s  | hopper       |
| A100  | 312T             | 80GB  | 2.0TB/s   | 40MB     | 600GB/s | 64GB/s  | ampere       |
| A800  | 312T             | 80GB  | 2.0TB/s   | 80MB     | 400GB/s | 128GB/s | ampere       |
| H20   | 148T             | 141GB | 4.0TB/s   | 60MB     | 900GB/s | 128GB/s | hopper       |
| L40s  | 362T             | 48GB  | 846GB/s   | 96MB     | /       | 64GB/s  | Ada Lovelace |
| 4090  | 330T             | 24GB  | 1.0TB/s   | 72MB     | /       | 64GB/s  | Ada Lovelace |

*   AMD
*   [MI400X]()
*   [MI350X]()
*   [MI325X]()
*   [MI300X]()
*   BIREN
*   Enflame
*   MOORE

### ASIC

*   [gaudi3](https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)
*   [trainium](https://aws.amazon.com/cn/ai/machine-learning/trainium/)
*   [D-matrix](https://www.d-matrix.ai/)
*   [sohu](https://www.etched.com/announcing-etched)
*   [tensortorrent](https://tenstorrent.com/en)
*   [sambanova RDU](https://sambanova.ai/)
*   [Groq LDU](https://groq.com/)
*   [cerebras](https://cerebras.ai/product-chip/)
*   [graphcore IPU](https://www.graphcore.ai/)
*   [google tpu v4](https://arxiv.org/abs/2304.01433): TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings
*   [google tpu v3](https://arxiv.org/pdf/1909.09756): Scale MLPerf-0.6 models on Google TPU-v3 Pods
*   [google tpu v2](https://arxiv.org/pdf/1811.06992): Image Classification at Supercomputer Scale
*   [google tpu v1](https://arxiv.org/pdf/1704.04760): In-Datacenter Performance Analysis of a Tensor Processing Unit

### FPGA

### PIM/NDP

*   [PIM Is All You Need: A CXL-Enabled GPU-Free System for Large Language Model Inference](https://arxiv.org/pdf/2502.07578)
*   [Make LLM Inference Affordable to Everyone: Augmenting GPU Memory with NDP-DIMM](https://arxiv.org/pdf/2502.16963)

## AI Training Optimization

### MoE training

*   [MoC-System: Efficient Fault Tolerance for Sparse Mixture-of-Experts Model Training](https://arxiv.org/pdf/2408.04307)

### Finetune

*   [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685) |16 Oct 2021 | microsoft\&cmu

### Parallelism training

*   [WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training](https://arxiv.org/pdf/2503.17924) | 23 Mar 2025 | Meta\&University of California, San Diego
*   [Mist: Efficient Distributed Training of Large Language Models via Memory-Parallelism Co-Optimization](https://arxiv.org/pdf/2503.19050)

#### TP

*   [Training Multi-Billion Parameter Language Models Using Model Parallelism(Megatron-LM)](https://arxiv.org/pdf/1909.08053.pdf) | 13 Mar 2020 | Nvidia

#### PP

*   [ping-pong pipeline parallelism](https://arxiv.org/pdf/2504.02263)
*   [DualPipe](https://arxiv.org/pdf/2412.19437)

#### DP

*   [FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html#getting-started-with-fully-sharded-data-parallel-fsdp)

#### SP/CP

*   [CSPS: A Communication-Efficient Sequence-Parallelism based Serving System for Transformer based Models with Long Prompts](https://arxiv.org/pdf/2409.15104) | 23 Sep 2024 | University of Virginia
*   [TRAINING ULTRA LONG CONTEXT LANGUAGE MODEL WITH FULLY PIPELINED DISTRIBUTED TRANSFORMER](https://arxiv.org/pdf/2408.16978) | 30 Aug 2024 | Microsoft
*   [ring attention]()
*   [Ulysses]()

#### EP

*   [deepEP](https://github.com/deepseek-ai/DeepEP) | deepseek

*   [megascale infer:](https://arxiv.org/pdf/2504.02263) | 7 Apr 2025 | byte dance seed

## AI Inference optimization

### KV cache optimization

*   [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) | Nov 2022 | Google
*   [Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching](https://arxiv.org/pdf/2504.06319)

### Quantization

*   [QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving](https://arxiv.org/pdf/2405.04532)
*   [COMET: Towards Practical W4A4KV4 LLMs Serving](https://arxiv.org/pdf/2410.12168)

### Pruning

*   [layer levlel]()
*   [head level]()
*   [channel level]()

### Decomposition

*   [ESPACE: Dimensionality Reduction of Activations for Model Compression](https://arxiv.org/pdf/2410.05437)

### Distilling

*   [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](https://arxiv.org/pdf/2410.13276)

### Sparse

*   [LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention](https://arxiv.org/pdf/2502.14866)
*   [SpInfer: Leveraging Low-Level Sparsity for Efficient Large Language Model Inference on GPUs](https://www.cse.ust.hk/\~weiwa/papers/eurosys25-fall-spinfer.pdf)

### Fusion

*   [FFN FUSION: RETHINKING SEQUENTIAL COMPUTATION IN LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2503.18908)
*   
### Heterogeneous Speculative Decoding
*   [Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM inference](https://arxiv.org/pdf/2412.18934)
*   [DuoDecoding: Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting](https://arxiv.org/pdf/2503.00784)

### Overlapping

Communication & Compute: tensor parallelism & communication

*   [CuBLASMp](https://docs.nvidia.com/cuda/cublasmp/release_notes/index.html) | nvidia
*   [FlashOverlap: A Lightweight Design for Efficiently Overlapping Communication and Computation](https://arxiv.org/pdf/2504.19519) | 28 Apr 2025 | pk\&infini-ai
*   [Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler](https://arxiv.org/pdf/2504.19442) | 4 May 2025 | PK\&seed
*   [TileLink: Generating Efficient Compute-Communicatio](https://arxiv.org/pdf/2503.20313)
*   [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/pdf/2408.12757) | 22 Aug 2024 | UW
*   [FLUX: FAST SOFTWARE-BASED COMMUNICATION OVERLAP ON GPUS THROUGH KERNEL FUSION](https://arxiv.org/pdf/2406.06858) | 23 Oct 2024 | ByteDance
*   [ISO: Overlap of Computation and Communication within Seqenence For LLM Inference](https://arxiv.org/pdf/2409.11155) | 4 Sep 2024 | Baichuan inc
*   [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/pdf/2409.15241v1) | 23 Sep 2024 |Microsoft
*   [T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives](https://arxiv.org/pdf/2401.16677) | 30 Jan 2024 | AMD

MoE: overlapping of alltoall & compute & inference system

*   [LANCET: ACCELERATING MIXTURE-OF-EXPERTS TRAINING VIA WHOLE G RAPH C OMPUTATION -C OMMUNICATION O VERLAPPING](https://arxiv.org/pdf/2404.19429) | 30 Apr 2024 | AWS
*   [TUTEL: ADAPTIVE MIXTURE-OF-EXPERTS AT SCALE](https://arxiv.org/pdf/2206.03382) | 5 Jun 2023 | Microsoft
*   [MegaScale infer](https://arxiv.org/pdf/2504.02263) | byte dance seed
*   [Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts](https://arxiv.org/pdf/2502.19811)
*   [Klotski: Efficient Mixture-of-Expert Inference via Expert-Aware Multi-Batch Pipeline](https://arxiv.org/pdf/2502.06888)
*   [MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators](https://arxiv.org/pdf/2504.02658)
*   [Delta Decompression for MoE-based LLMs Compression](https://arxiv.org/pdf/2502.17298)
*   [Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models](https://arxiv.org/pdf/2402.14800)
*   [DAOP: Data-Aware Offloading and Predictive Pre-Calculation for Efficient MoE Inference](http://arxiv.org/pdf/2501.10375)

### MoE route

*   [Dynamic Language Group-based MoE: Enhancing Code-Switching Speech Recognition with Hierarchical Routing](https://arxiv.org/pdf/2407.18581)

### Offloading

*   [PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization](https://arxiv.org/pdf/2503.01328) | |
*   [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/pdf/2411.11217) | 18 Nov 2024 | UC Berkeley
*   [ZeRO-offload++](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-offloadpp) |November 2023| Microsoft
*   [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://www.usenix.org/conference/atc21/presentation/ren-jie) | 18 Jan 2021 | Microsoft
*   [Efficient and Economic Large Language Model Inference with Attention Offloading](https://arxiv.org/pdf/2405.01814) | 3 May 2024 | TsingHua University
*   [InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference](https://arxiv.org/pdf/2409.04992) | 8 Sep 2024 |
*   [Neo: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference](https://arxiv.org/pdf/2411.01142) | 2 Nov 2024 | PeKing University
*   [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/pdf/2312.17238) | 28 Dec 2023 | Moscow Institute of Physics and Technology
*   [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/pdf/2411.01433) |6 Nov 2024 | shanghai jiaotong University\&CUHK
*   [FlexInfer: Breaking Memory Constraint via Flexible and Efficient Offloading for On-Device LLM Inference](https://arxiv.org/pdf/2503.03777)
*   [MOE-INFINITY: Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache](https://arxiv.org/pdf/2401.14361) | 12 Mar 2025 | The University of Edinburgh
*   [Smart-Infinity: Fast Large Language Model Training using Near-Storage Processing on a Real System](https://arxiv.org/pdf/2403.06664) | 11 Mar 2024 |
*   [Glinthawk: A Two-Tiered Architecture for Offline LLM Inference](https://arxiv.org/pdf/2501.11779) | 11 Feb 2025 | microsoft
*   [Fast State Restoration in LLM Serving with HCache](https://arxiv.org/pdf/2410.05004)
*   [Efficient and Economic Large Language Model Inference with Attention Offloading](https://arxiv.org/html/2405.01814v1)

### hybrid batches

*   [POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference](https://arxiv.org/pdf/2410.18038)

### Parameter Sharing

*   [MHA]()
*   [GQA](https://arxiv.org/pdf/2305.13245)
*   [MLA](https://arxiv.org/pdf/2405.04434)

### LORA

*   [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/pdf/2402.09353)

### Attention optimization

The quadratic complexity of self-attention in a vanilla Transformer is well-known, and there has been much research on how to optimize attention to a linear-complexity algorithm.

1.Efficient attention algorithms: the use of faster, modified memory-efficient attention algorithms

2.Removing some attention heads, called attention head pruning

3.Approximate attention

4.Next-gen architectures

5.Code optimizations: op level rewriting optimizaiton

*   [FLEX ATTENTION: A PROGRAMMING MODEL FOR GENERATING OPTIMIZED ATTENTION KERNELS](https://arxiv.org/pdf/2412.05496) | 7 Dec 2024 | Meta
*   [STAR ATTENTION: EFFICIENT LLM INFERENCE OVER LONG SEQUENCES](https://arxiv.org/pdf/2411.17116) | 26 Nov 2024 | Nvidia
*   [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA)](https://arxiv.org/pdf/2405.04434) |19 Jun 2024 | Deepseek
*   [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/pdf/2407.08608) | 12 Jul 2024 |
*   [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691) | 17 Jul 2023 | Princeton University\&Stanford University
*   [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) | 23 Jun 2022 | stanford
*   [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245) | 23 Dec 2023 | Google
*   [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/pdf/1911.02150) |6 Nov 2019 | Google
*   [Multi-Head Attention: Collaborate Instead of Concatenate](https://arxiv.org/pdf/2006.16362) | 20 May 2021 |
*   [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180) | 12 Sep 2023 | UC Berkeley
*   [Slim attention](https://arxiv.org/pdf/2503.05840) | 7 Mar 2025 | openmachine
*   [Do We Really Need the KVCache for All Large Language Models](https://yywangcs.notion.site/Do-We-Really-Need-the-KVCache-for-All-Large-Language-Models-0c27c6c8f9d04420b899a09702980045) | blog
*   [SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs](https://arxiv.org/pdf/2410.13276) | 17 Feb 2025 | UHK\&UW\&Microsoft\&nvidia

### MHA2MLA

*   [Towards Economical Inference: Enabling DeepSeek’s Multi-Head Latent Attention in Any Transformer-based LLMs](https://arxiv.org/pdf/2502.14837)
*   [TransMLA: Multi-Head Latent Attention Is All You Need](https://arxiv.org/pdf/2502.07864)
*   [X-EcoMLA: Upcycling Pre-Trained Attention into MLA for Efficient and Extreme KV Compression](https://arxiv.org/pdf/2503.11132)

### Prefill Decode Disaggregated

*   [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | 9 Jul 2024 | Moonshot AI
*   [MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool](https://arxiv.org/abs/2406.17565) | 26 Jun 2024 | huawei cloud
*   [Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads](https://arxiv.org/abs/2401.11181) | 20 Jan 2024 | huawei cloud
*   [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/pdf/2311.18677) | 20 May 2024 | Microsoft
*   [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/pdf/2401.09670) | 6 Jun 2024 | PK\&StepFun
*   [semi-PD: TOWARDS EFFICIENT LLM SERVING VIA PHASE-WISE
    DISAGGREGATED COMPUTATION AND UNIFIED STORAGE](https://arxiv.org/pdf/2504.19867)

### Prefill optimization

*   [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/pdf/2308.16369) | 31 Aug 2023 | Microsoft
*   [Cachegen: Fast context loading for language model applications (prefix KV cache)](https://arxiv.org/abs/2310.07240) |August 2024 | Microsoft
*   [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://arxiv.org/abs/2312.07104) | 6 Jun 2024 | Stanford University

### DeepSeek Open Day

<https://github.com/deepseek-ai/open-infra-index?tab=readme-ov-file#day-6---one-more-thing-deepseek-v3r1-inference-system-overview>

*   [DeepSeek-NSA]()
*   [FlashMLA]()
*   [DualPipe]()
*   [DeepEP]()
*   [3FS]()
*   [DeepSeek inference system]()
*   [EPLB]()
*   [DeepGEMM]()

## AI Inference\&serving Framework

*   [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft
*   [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Nvidia
*   [Accelerate](https://huggingface.co/docs/accelerate/index) | Hugging Face
*   [Ray-LLM](https://github.com/ray-project/ray-llm) | Ray
*   [LLaVA](https://github.com/haotian-liu/LLaVA)
*   [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Nvidia
*   [NeMo](https://github.com/NVIDIA/NeMo) | Nvidia
*   [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch
*   [vLLM](https://github.com/vllm-project/vllm) | UCB
*   [SGLang](https://github.com/sgl-project/sglang) | UCB
*   [llama.cpp](https://github.com/ggerganov/llama.cpp)
*   [PRIMA.CPP](https://arxiv.org/pdf/2504.08791)

## AI Complier

*   [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/pdf/2002.03794) | 28 Aug 2020 | Beihang University\&Tsinghua University
*   [MLIR: A Compiler Infrastructure for the End of Moore’s Law](https://arxiv.org/pdf/2002.11054) | 1 Mar 2020 | Google
*   [XLA](https://www.tensorflow.org/xla?hl=fr) | 2017 | Google
*   [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/pdf/1802.04799) | 5 Oct 2018 | University of Washington

## AI Infrastructure

*   [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/pdf/2402.15627) | 23 Feb 2024 | ByteDance
*   [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068) | 21 Jun 2022 | Meta
*   [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/research/70b-infrastructure/) | Imbue
*   [Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures](https://www.arxiv.org/pdf/2505.09343) | 14 May 2025 | DeepSeek

