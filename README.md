# Awesome AI

## AI LLM Survey
* [Thus Spake Long-Context Large Language Model](https://arxiv.org/pdf/2502.17129) |24 Feb 2025| Shanghai AI Lab&huawei&fudan
* [A Survey on Inference Optimization Techniques for Mixture of Experts Models](https://arxiv.org/pdf/2412.14219) |Dec 2024 | CUHK Univresity&shang hai jiao tong University
* [A Survey on Mixture of Experts](https://arxiv.org/pdf/2407.06204) | 8 Aug 2024 | UK Univesity
* [A Survey of Low-bit Large Language Models:Basics, Systems, and Algorithms](https://arxiv.org/pdf/2409.16694) | 30 Sep 2024 | Beihang University&ETH Zuric&cSenseTime&CUHK
* [A Survey on Efficient Inference for Large Language Models](https://arxiv.org/pdf/2404.14294) |  22 Apr 2024 | Infinigence-AI
* [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863) | 23 May 2024 | AWS
* [Beyond Efficiency: A Systematic Survey of Resource-Efficient Large Language Models](https://arxiv.org/abs/2401.00625) | 1 Jan 2024 |
* [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234) | 23 Dec 2023 | CMU
* [Challenges and Applications of Large Language Models](https://arxiv.org/pdf/2307.10169.pdf) | 19 Jul 2023 | University College London
* 
## AI State of Art Model
paper&technical paper
* [deepseek r1](https://arxiv.org/abs/2501.12948) | 22 Jan 2025 | DeepSeek
* [deepseek v3](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/DeepSeek_V3.pdf) |26 Dec 2024 | DeepSeek

## AI benchmark
* [LLM-Inference-Bench: Inference Benchmarking of Large Language Models on AI Accelerators](https://arxiv.org/abs/2411.00136) | 31 Oct 2024 | Argonne National Laboratory

## AI System 

## AI Algorithm

### Network Architecture
* [Transformer](https://arxiv.org/pdf/1706.03762): Attention Is All You Need | 2 Aug 2023 | Google
* [RWKV](https://arxiv.org/pdf/2305.13048): Reinventing RNNs for the Transformer Era | 11 Dec 2023 | Generative AI Commons
* [Mamba](https://arxiv.org/abs/2312.00752): Linear-Time Sequence Modeling with Selective State Spaces | 31 May 2024 | CMU
### MoE
* [deepseek]()
* [minimax]()
* [mistral]()
* [GROK]()

## AI Chips 
Chips Survey
* 2024 [LARGE LANGUAGE MODEL INFERENCE ACCELERATION:A COMPREHENSIVE HARDWARE PERSPECTIVE](https://arxiv.org/pdf/2410.04466)
* 2023 [Lincoln AI Computing Survey (LAICS) Update](https://arxiv.org/pdf/2310.09145)
* 2022 [AI and ML Accelerator Survey and Trendse](https://arxiv.org/pdf/2210.04055)
* 2021 [AI Accelerator Survey and Trends](https://arxiv.org/pdf/2109.08957)
* 2020 [Survey of Machine Learning Accelerators](https://arxiv.org/pdf/2009.00993)
* 2019 [Survey and Benchmarking of Machine Learning Accelerators](https://arxiv.org/pdf/1908.11348)

### GPU
* NVIDIA

| #     |FLOPs dense fp16 | HBM  | Bandwidth | L2 cache |NV link| PCIe|Architecture|
| ---   | ---  | ----  | ---     | --- |----|-----|-----|
| GB200 | 5P   | 192GB | 8.0TB/s |  | 1.8TB/s | 128GB/s|blackwell|
| GH200 | 985T | 141GB | 4.8TB/s | 60MB | 900GB/s | 128GB/s| hopper  |
| H100  | 985T | 80GB | 3.35TB/s| 50MB | 900GB/s | 128GB/s| hopper  |
| H800  | 985T | 80GB | 3.35TB/s| 50MB | 400GB/s | 64GB/s | hopper  |
| A100  | 312T | 80GB | 2.0TB/s | 40MB | 600GB/s | 64GB/s | ampere  |
| A800  | 312T | 80GB | 2.0TB/s | 80MB | 400GB/s | 128GB/s| ampere  |
| H20   | 148T | 141GB  | 4.0TB/s | 60MB | 900GB/s | 128GB/s| hopper  |
| L40s  | 362T | 48GB | 846GB/s | 96MB | /       | 64GB/s | Ada Lovelace  |
| 4090  | 330T | 24GB | 1.0TB/s | 72MB | /       | 64GB/s | Ada Lovelace  |

* AMD
* [MI400X]()
* [MI350X]()
* [MI325X]()
* [MI300X]()
* BIREN
* Enflame
* MOORE

### ASIC
* [gaudi3](https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html)
* [trainium](https://aws.amazon.com/cn/ai/machine-learning/trainium/)
* [D-matrix](https://www.d-matrix.ai/)
* [sohu](https://www.etched.com/announcing-etched)
* [tensortorrent](https://tenstorrent.com/en)
* [sambanova RDU](https://sambanova.ai/)
* [Groq LDU](https://groq.com/)
* [cerebras](https://cerebras.ai/product-chip/)
* [graphcore IPU](https://www.graphcore.ai/)
* [google tpu v4](https://arxiv.org/abs/2304.01433): TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings
* [google tpu v3](https://arxiv.org/pdf/1909.09756): Scale MLPerf-0.6 models on Google TPU-v3 Pods
* [google tpu v2](https://arxiv.org/pdf/1811.06992): Image Classification at Supercomputer Scale
* [google tpu v1](https://arxiv.org/pdf/1704.04760): In-Datacenter Performance Analysis of a Tensor Processing Unit

### FPGA

### PIM/NDP

## AI Training Optimization
### Finetune
*
### Parallelism training

#### TP
* [Training Multi-Billion Parameter Language Models Using Model Parallelism(Megatron-LM)](https://arxiv.org/pdf/1909.08053.pdf) | 13 Mar 2020 | Nvidia
#### PP

#### DP

#### SP&CP
* [CSPS: A Communication-Efficient Sequence-Parallelism based Serving System for Transformer based Models with Long Prompts](https://arxiv.org/pdf/2409.15104) | 23 Sep 2024 | University of Virginia
* [TRAINING ULTRA LONG CONTEXT LANGUAGE MODEL WITH FULLY PIPELINED DISTRIBUTED TRANSFORMER](https://arxiv.org/pdf/2408.16978) | 30 Aug 2024 | Microsoft
* [ring attention]()
* [Ulysses]()
#### EP
* [deepEP](https://github.com/deepseek-ai/DeepEP) | deepseek
 
* [megascale infer:](https://arxiv.org/pdf/2504.02263) | 7 Apr 2025 | byte dance seed

## AI Inference optimization

### KV cache optimization
* [Efficiently Scaling Transformer Inference]( https://arxiv.org/abs/2211.05102 ) | Nov 2022 | Google

### Quantization
*

### Pruning
layer levlel
head level

### Distilling
*

### Sparse
*

### Fusion
*

### Overlapping
Communication & Compute: tensor parallelism & communication
* [NanoFlow: Towards Optimal Large Language Model Serving Throughput](https://arxiv.org/pdf/2408.12757) | 22 Aug 2024 | UW
* [FLUX: FAST SOFTWARE-BASED COMMUNICATION OVERLAP ON GPUS THROUGH KERNEL FUSION](https://arxiv.org/pdf/2406.06858) | 23 Oct 2024 | ByteDance
* [ISO: Overlap of Computation and Communication within Seqenence For LLM Inference](https://arxiv.org/pdf/2409.11155) | 4 Sep 2024 | Baichuan inc
* [Domino: Eliminating Communication in LLM Training via Generic Tensor Slicing and Overlapping](https://arxiv.org/pdf/2409.15241v1) | 23 Sep 2024 |Microsoft
* [T3: Transparent Tracking & Triggering for Fine-grained Overlap of Compute & Collectives](https://arxiv.org/pdf/2401.16677) | 30 Jan 2024 | AMD

MoE: overlapping of alltoall & compute
* [LANCET: ACCELERATING MIXTURE-OF-EXPERTS TRAINING VIA WHOLE G RAPH C OMPUTATION -C OMMUNICATION O VERLAPPING](https://arxiv.org/pdf/2404.19429) | 30 Apr 2024 | AWS
* [TUTEL: ADAPTIVE MIXTURE-OF-EXPERTS AT SCALE](https://arxiv.org/pdf/2206.03382) | 5 Jun 2023 | Microsoft 
*  [MegaScale infer]()

### Offloading
* [PipeOffload: Improving Scalability of Pipeline Parallelism with Memory Optimization](https://arxiv.org/pdf/2503.01328) | |
* [MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs](https://arxiv.org/pdf/2411.11217) | 18 Nov 2024 | UC Berkeley
* [ZeRO-offload++](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-offloadpp) |November 2023| Microsoft
* [ZeRO-Offload: Democratizing Billion-Scale Model Training](https://www.usenix.org/conference/atc21/presentation/ren-jie) | 18 Jan 2021 | Microsoft
* [Efficient and Economic Large Language Model Inference with Attention Offloading](https://arxiv.org/pdf/2405.01814) | 3 May 2024 | TsingHua University
* [InstInfer: In-Storage Attention Offloading for Cost-Effective Long-Context LLM Inference](https://arxiv.org/pdf/2409.04992) | 8 Sep 2024 | 
* [Neo: Saving GPU Memory Crisis with CPU Offloading for Online LLM Inference](https://arxiv.org/pdf/2411.01142) | 2 Nov 2024 | PeKing University
* [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/pdf/2312.17238) | 28 Dec 2023 | Moscow Institute of Physics and Technology
* [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/pdf/2411.01433) |6 Nov 2024 | shanghai jiaotong University&CUHK

### Parameter Sharing
* [MQA]()
* [GQA]()
* 
### Attention optimization
The quadratic complexity of self-attention in a vanilla Transformer is well-known, and there has been much research on how to optimize attention to a linear-complexity algorithm.

1.Efficient attention algorithms: the use of faster, modified memory-efficient attention algorithms

2.Removing some attention heads, called attention head pruning

3.Approximate attention 

4.Next-gen architectures

5.Code optimizations: op level rewriting optimizaiton

* [FLEX ATTENTION: A PROGRAMMING MODEL FOR GENERATING OPTIMIZED ATTENTION KERNELS](https://arxiv.org/pdf/2412.05496) | 7 Dec 2024 | Meta
* [STAR ATTENTION: EFFICIENT LLM INFERENCE OVER LONG SEQUENCES](https://arxiv.org/pdf/2411.17116) | 26 Nov 2024 | Nvidia
* [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA)](https://arxiv.org/pdf/2405.04434) |19 Jun 2024 | Deepseek
* [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/pdf/2407.08608) | 12 Jul 2024 |
* [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/pdf/2307.08691) | 17 Jul 2023 | Princeton University&Stanford University
* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135) | 23 Jun 2022 | stanford
* [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/pdf/2305.13245) | 23 Dec 2023 | Google
* [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/pdf/1911.02150) |6 Nov 2019 | Google
* [Multi-Head Attention: Collaborate Instead of Concatenate](https://arxiv.org/pdf/2006.16362) | 20 May 2021 |
* [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/pdf/2309.06180) | 12 Sep 2023 | UC Berkeley 

### Prefill decode disaggregated
* [Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving](https://arxiv.org/abs/2407.00079) | 9 Jul 2024 | Moonshot AI
* [MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool](https://arxiv.org/abs/2406.17565) | 26 Jun 2024 | huawei cloud
* [Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads](https://arxiv.org/abs/2401.11181) | 20 Jan 2024 | huawei cloud
* [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/pdf/2311.18677) | 20 May 2024 | Microsoft
* [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/pdf/2401.09670) | 6 Jun 2024 | PK&StepFun
* 
### Prefill optimization
* [SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills](https://arxiv.org/pdf/2308.16369) | 31 Aug 2023 | Microsoft
* [Cachegen: Fast context loading for language model applications (prefix KV cache)](https://arxiv.org/abs/2310.07240) |August 2024 | Microsoft
* [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://arxiv.org/abs/2312.07104) | 6 Jun 2024 | Stanford University

## AI Inference&serving Framework
* [DeepSpeed](https://github.com/microsoft/DeepSpeed) | Microsoft
* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Nvidia
* [Accelerate](https://huggingface.co/docs/accelerate/index) | Hugging Face
* [Ray-LLM](https://github.com/ray-project/ray-llm) | Ray
* [LLaVA](https://github.com/haotian-liu/LLaVA)
* [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Nvidia
* [NeMo](https://github.com/NVIDIA/NeMo) | Nvidia
* [torchtitan](https://github.com/pytorch/torchtitan) | PyTorch
* [vLLM](https://github.com/vllm-project/vllm) | UCB
* [SGLang](https://github.com/sgl-project/sglang) | UCB
* [llama.cpp](https://github.com/ggerganov/llama.cpp)

## AI Complier
* [The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/pdf/2002.03794) | 28 Aug 2020 | Beihang University&Tsinghua University
* [MLIR: A Compiler Infrastructure for the End of Moore’s Law](https://arxiv.org/pdf/2002.11054) | 1 Mar 2020 | Google
* [XLA](https://www.tensorflow.org/xla?hl=fr) | 2017 | Google
* [TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://arxiv.org/pdf/1802.04799) | 5 Oct 2018 | University of Washington

## AI Infrastructure
* [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/pdf/2402.15627) | 23 Feb 2024 | ByteDance
* [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/pdf/2205.01068) | 21 Jun 2022 | Meta
