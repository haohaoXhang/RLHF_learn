1️⃣ LLaMA-Factory
LLAMA Factory 是一个基于 Hugging Face Transformers 的开源项目，专注于为大型语言模型提供高效、灵活且用户友好的微调框架，是一个封装比较完善的LLM微调工具，它能够帮助用户快速地训练和微调大多数LLM模型。（有前端UI，傻瓜操作）
2️⃣ Megatron
Megatron 是由 NVIDIA 开发的一个强大的开源工具集，专门用于大规模训练巨型Transformer模型。它提供了两个主要组件：Megatron-LM 参考实现为训练最先进的基础模型提供了开箱即用的解决方案；而Megatron Core 可组合库则为开发者提供了高度优化的基础模块，用以构建自定义的、高性能的训练框架。

英伟达 Megatron 支持模型并行化技术（Deepspeed及FSDP本质都是数据并行），追求极致吞吐。但是，Megatron 与人们常用的 Hugging Face 软件库不兼容。
3️⃣ ms-swift
ms-swift 是 ModelScope 社区官方推出的轻量级可扩展微调框架。它支持超过 450 个纯文本大型模型和 150 多个多模态大型模型，覆盖了从模型训练到部署的整个流程。该框架支持包括 LoRA、QLoRA 在内的多种轻量级微调方法，以及分布式训练、量化训练和强化学习人类反馈训练。
4️⃣ unsloth
unsloth 旨在显著加快大型语言模型的微调速度并降低内存使用。它支持包括 Llama、DeepSeek、Gemma 和 Mistral 等多种模型。unsloth 利用高度优化的 OpenAI Triton 内核和手动反向传播引擎来实现性能提升。它支持全参数微调、预训练以及 4 位、8 位和 16 位的高效训练，并声称在加速和减少内存使用的同时不会损失精度。
5️⃣ trl
TRL（Transformers Reinforcement Learning） 是 Hugging Face 推出的一个专门用于大语言模型对齐和微调的库。它建立在 Transformers 和 Accelerate 之上，兼容 Hugging Face 生态（Datasets、PEFT 等），并提供了简单易用的接口来实现.
6️⃣ verl
Verl 是一个灵活、高效且可用于生产环境的强化学习训练库，由字节seed团队开发，专为大语言模型设计；基于HybridFlow的架构，基于FSDP, Megatron-LM, vLLM, SGLang等构建的框架。
7️⃣ OpenRLHF
OpenRLHF 是一个基于 Ray、DeepSpeed 和 HF Transformers 构建的高性能 RLHF 框架。OpenRLHF 是目前可用的最简单的高性能 RLHF 库之一，无缝兼容 Huggingface 模型和数据集。 RLHF 训练中 80% 的时间用于样本生成阶段。
