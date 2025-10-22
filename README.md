# **Project: Autoregressive Modeling in a Visual-Symbolic Space**

## **Background and Motivation**

### **The Computational Bottleneck of Long-Context Modeling**

LLM 的核心架构 Transformer 的 self-attention 具有与输入序列长度O(n²)相关的计算与内存复杂度。这一特性从根本上限制了模型在单次交互中能够处理的上下文长度。因此，对于书籍、长篇财报或完整代码库等超长文档的深度理解与分析，现有 LLM 架构面临难以逾越的瓶颈。

### **Insights and Limitations of Prior Work**

本研究受到以下两项前沿工作的启发，旨在解决其固有局限性：

* **Glyph (Tsinghua & Zhipu AI):** 该工作验证了 visual context extension 范式的可行性——通过将文本渲染为图像，利用 VLM 进行处理，从而压缩信息长度。它证明了在视觉空间中表征并理解语义信息是一条有效路径。其核心局限在于：它依赖一个通用的、为图文对设计的 VLM，而没有构建一个为这种新型数据形态原生设计的、在 latent space 中直接进行推理的模型。
* **DeepSeek-OCR (DeepSeek AI):** 该工作提供了一个性能卓越的 visual-text codec。其 Encoder 能够以极高的压缩率将文档图像转换为离散的 visual tokens，而其 Decoder 则能高保真地还原文本。这为我们提供了一个在人类文本世界与高效、压缩的潜空间之间进行双向”翻译“的强大工具。然而，其本身作为 OCR 工具，不具备在压缩后的符号空间中进行上下文推理的能力。

### **Central Thesis**

现有工作揭示了一个独特的机遇。我们的核心议题是：**我们能否定义并操作一个全新的 Visual-Symbolic Space？**在此空间里，视觉符号不再是图像的中间表征，而是构成一种新语言的基本单位。我们的目标是构建一个能直接在该空间中进行 native autoregressive reasoning 的模型，从而从根本上改变处理长文本的计算范式，绕开传统文本序列的二次方复杂度限制。

---

## **Method: Visual-Symbolic Space Autoregressive Modeling**

我们将构建一个名为 Visual-Symbolic Space Autoregressive Modeling 的全新模型框架。该框架并非对现有模型的微调，而是一种全新的端到端学习范式。

### **Framework Architecture**

该框架由三个核心组件构成：

1. **The Translators (Frozen):**
   * **Encoder:** 一个预训练且**冻结的（frozen）**DeepSeek-OCR Encoder。它将渲染后的文档图片高效地"翻译"为视觉-符号空间中的离散符号序列。将其冻结是为了利用其强大的、经过充分训练的视觉表征能力，并大幅降低核心模型的训练成本。
   * **Decoder:** 一个预训练且**冻结的（frozen）**DeepSeek-OCR Decoder。它负责在推理结束后，将模型在潜空间中生成的符号序列"翻译"回人类可读的文本。
2. **The Native Speaker (Trainable):**
   * **Visual-Symbolic Transformer:** 这是我们工作的核心创新——一个只处理视觉符号的 decoder-only 自回归 Transformer 模型。它的唯一任务是通过学习预测序列中的下一个视觉符号，隐式地掌握这个全新空间的内在语法、逻辑和语义关系。

### **Core Mechanism: The Unified Reading Path**

为了让模型能够学习，我们必须为空间数据建立清晰的 causal structure。我们的方法是定义一个”统一阅读路径“，具体操作如下：

1. 一篇长文档被顺序渲染成 M 张标准化的图片。
2. Encoder 将每张图片 i 转换为一个视觉符号序列 V_i。
3. 所有序列按原始文档顺序拼接成单一的超长视觉符号序列 V_full = [V_1, V_2, ..., V_M]。

Visual-Symbolic Transformer 的 autoregressive objective 就是沿着这个一维路径，从左至右预测下一个视觉符号 v_t，即最大化条件概率 P(v_t | v_<t)。这个过程类似于图像生成模型中的光栅扫描顺序（raster scan order），它为原本在空间上并行的视觉数据提供了结构化的串行学习路径，使得标准的因果语言模型（Causal LM）训练范式可以被直接应用。

---

## **Phased Research Plan & Goals**

**第一阶段：Foundation & Verification**

* **目标：**建立完整的 data-processing pipeline，量化 translator 的性能，确保信息保真度。
* **产出：**
  1. 可稳定部署的 DeepSeek-OCR Encoder & Decoder，并完成保真度测试（例如，原文 → 渲染 → 编码 → 解码 → 原文的 BLEU/ROUGE 分数）。
  2. 一个自动化的 rendering engine，能将任意长文本按序、批量地转换为标准化图片序列。
  3. 一个高效的 data conversion pipeline，能将大型文本语料库（如 The Pile）转换为以高效格式（如 memmapped numpy arrays）存储的视觉符号序列数据集。

**第二阶段：Core Model Pre-training**

* **目标：**在大规模视觉符号数据集上预训练 Visual-Symbolic Transformer，使其学习该空间的内在规律。
* **产出：**

  1. 一个预训练好的 Visual-Symbolic Transformer (Base) 基础模型。
  2. 

  ## **推荐训练目标（continuous AR）的三重损失**

  令 $x_t \in \mathbb{R}^D$ 为视觉嵌入序列，模型预测 $\hat x_t$。


  * **向量级回归损失（稳住几何）**
    $\mathcal{L}_{reg} = \lambda_1 \| \hat x_t - x_t \|_2^2 + \lambda_2 \big(1 - \cos(\hat x_t, x_t)\big)$
  * **流形一致性损失（稳住语义）**
    把 $\hat X = [\hat x_{t-k..t}]$ 送进 **冻结解码器** （你的 DeepSeek-OCR decoder 端），计算与真值文本 y 的 **交叉熵** ：
    $\mathcal{L}_{dec} = \mathrm{CE}\big( \mathrm{Decoder}(\hat X),\, y \big)$
    这相当于“感知器约束”，强迫预测的嵌入对解码器来说仍然是“合法”的。
  * **范数与分布正则（防漂）**
    给预测加 LayerNorm 或单位球投影正则，或 penalty $\lambda_3(\|\hat x_t\|_2 - \alpha)^2$。

  1. 一套完整的评估体系，用于在训练过程中监控模型在视觉-符号空间中的学习情况（核心指标：Perplexity）。
  2. 关于模型收敛性和扩展法则（scaling laws）的初步分析报告。

**第三阶段：Instruction Fine-tuning & Evaluation**

* **目标：**将预训练模型微调为指令遵循模型，并在权威的长上下文基准测试中系统性地评估其性能与计算效率。
* **产出：**
  1. 一个经过指令微调的对话模型——Visual-Symbolic Chat。
  2. 在 LongBench、MRCR 等长上下文基准测试上的详尽评测报告，与 SOTA（State-of-the-Art）长文本模型进行对比。
  3. 一份关于模型效率的量化分析报告，具体包括处理等量文本时在内存占用（memory footprint）、吞吐量（throughput）和端到端延迟（latency）方面的表现。

---

## **Detailed Implementation Plan**

[Work List](https://www.notion.so/Work-List-293fef87c245801f983ada117ed45ab5?pvs=21)

**Phase 1: TODOs**

1. **环境配置：**部署开源的 DeepSeek-OCR 模型，并进行推理验证。
2. **渲染引擎开发：**
   * 利用 headless browser 或 Pillow/PyCairo 等库开发脚本。
   * 输入：长文本文件。
   * 输出：output/doc_id/page_001.png, page_002.png, ...
   * 关键参数：字体、字号、行距、页面分辨率、页边距等需标准化。
3. **数据处理管线：**
   * **数据源：**优先选择 The Pile 或 C4 等大型、多样化的公开文本数据集。
   * **批处理渲染：**编写分布式脚本，将数据集并行渲染成图片。
   * **批处理编码：**使用 Encoder 将图片序列批量转换为视觉符号序列，并以 memmapped 或 HDF5 格式存储，以便在训练中高效读取。

**Phase 2: TODOs**

1. **模型架构设计：**
   * 基于 PyTorch 和 Hugging Face Transformers 等现有库，实现标准的 decoder-only Transformer 架构。
2. **训练实现：**
   * 实现自定义的 Dataset 和 Dataloader，能够高效地从 memmapped 文件中读取和拼接超长序列 V_full。
   * 训练脚本采用标准的 Causal LM 目标，损失函数为 Cross-Entropy Loss。
3. **模型训练：**
   * 在多机多卡的 GPU 集群上启动预训练。
   * 使用 wandb 持续监控训练损失和 Perplexity，确保训练稳定。

**Phase 3: TODOs**

1. **SFT 数据准备：**
   * 选取高质量的指令微调数据集（如 Alpaca、ShareGPT）。
   * 编写脚本，将数据集中的每一条样本（指令、上下文、答案）分别渲染并编码，然后按照 [V_instruction, V_context, V_response] 的格式拼接成 SFT 训练样本。
2. **SFT（Supervised Fine-tuning）训练：**
   * 在 Visual-Symbolic Transformer (Base) 模型的基础上进行全参数或 LoRA 微调。
3. **评估框架搭建：**
   * 编写端到端（end-to-end）评估脚本，封装从**文本输入 → 渲染 → 编码 → 模型推理 → 解码 → 文本输出**的完整流程。
4. **执行评估与分析：**
   * 在 LongBench 等基准上运行评估脚本，收集性能数据。
   * 与 GPT-4 (long context)、Claude、Gemini 等顶尖长上下文模型进行性能和效率的横向对比，并撰写最终分析报告。

---

## **Key Challenges & Discussion**

### **Learning Hierarchical Dependencies within a Unified Causal Framework**

* **问题陈述：**模型处理的视觉符号序列 V_full 具有内在的层次结构。在微观层面（单张图片内部 V_i），符号间的关系主要是空间上的（二维布局），并无严格的因果性。而在宏观层面（不同图片 V_i 和 V_{i+1} 之间），则存在由原始文本决定的强烈语义因果性。模型如何在单一的自回归目标下同时学会这两种不同尺度的依赖关系？
* **核心假设（Central Hypothesis）：**我们并不将此视为矛盾，而是将其视为**统一的学习任务** 。我们提出的"统一阅读路径"为模型提供了单一、简洁的一维自回归目标。我们的核心假设是：一个足够强大的 Transformer 模型，通过其自注意力机制，能够**自适应地学会（adaptively learn）**处理这种具有层次化结构的数据流。在训练过程中，模型将涌现（emerge）出不同的注意力模式（attention patterns）来分别处理局部空间依赖（学习"阅读"页面布局）和全局语义依赖（学习"思考"文档逻辑）。这一设计的优雅之处在于，模型的复杂行为是从最简单的单一目标和复杂的数据结构之间涌现出来的，而非通过两种手动设计的、分离的模式。该范式与图像生成模型通过施加光栅扫描顺序来学习复杂空间分布的原理一脉相承，具有坚实的理论基础。

### **Other Potential Challenges**

* **信息保真度（Information Fidelity）：**在"文本 → 图像 → 符号"的编码过程中，以及在"符号 → 文本"的解码过程中，可能会出现信息损失。第一阶段需要对这一损失进行精确量化，以确保其在可接受范围内。
* **计算资源需求：**虽然本方法旨在解决推理时的长度瓶颈，但对海量语料进行一次性的渲染和编码本身是一个计算密集型任务，需要强大的工程实现和计算资源。
* **误差累积（Error Accumulation）：**在自回归生成长序列时，解码过程中的微小错误可能会被累积和放大，导致最终生成的文本质量下降。这需要在评估阶段重点关注。
