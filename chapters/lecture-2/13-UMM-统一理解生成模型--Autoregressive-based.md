[Previous](12-UMM-统一理解生成模型--Diffusion-based.md) | [Contents](../../README.md) | [Next](14-UMM-统一理解生成模型--AR--Diffusion.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=13)

## 5.2 <span style="color: rgb(36,91,219); background-color: inherit">Autoregressive-based</span>

### 5.2.1 **<span style="color: rgb(36,91,219); background-color: inherit">Chameleon（b-1）</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

![](../../images/视觉多模态讲义（下）-image-138.png)

架构基本和 LLaMA-2 一致：

> * 使&#x7528;**`RMSNorm`**&#x8FDB;行归一化
>
> * 激活函数采&#x7528;**`SwiGLU`**
>
> * 位置编码采&#x7528;**`RoPE`**

但<span style="color: rgb(216,57,49); background-color: inherit">标准的 LLaMA 在训练中后期因范数缓慢增长而出现复杂发散</span>，作者发现发散根源在于 softmax 操作在多模态训练中存在问题：<span style="color: rgb(216,57,49); background-color: inherit">由于 softmax 具有平移不变性，即</span>$$\mathrm{softmax}(\mathbf{z}) = \mathrm{softmax}(\mathbf{z} + c)$$<span style="color: rgb(216,57,49); background-color: inherit">，当不同模态的熵差异显著时，softmax 输出会失衡。由于模型在所有模态间共享权重，各模态会通过略微增大自身范数竞争主导地位；虽然训练初期无碍，但一旦超出 BF16 的有效表示范围，就会导致发散</span>。在单模态场景中，这个问题也被称&#x4E3A;**`logit drift`**。下图(a)展示了随着训练进行，最后一层 Transformer 输出范数的变化：尽管发散可能在训练进度&#x8FBE;**`20–30%`**&#x540E;才显现，但输出范数的失控增长与未来损失发散高度相关。

![      (a)                                                          (b)                                                          (c)](../../images/视觉多模态讲义（下）-image-135.png)

Transformer 中 softmax 出现在两个位置：注意力机制内部和最终 logits 的 softmax。作者修改 LLaMA 架构，<span style="color: rgb(100,37,208); background-color: inherit">引入</span>**<span style="color: rgb(100,37,208); background-color: inherit">QK-Norm</span>**<span style="color: rgb(100,37,208); background-color: inherit">。QK-Norm 通过对注意力中的 query 和 key 向量施加 LayerNorm，直接控制输入 softmax 的范数增长</span>。上图(b)显示了 Chameleon-7B 在启用与禁用 QK-Norm 下的训练损失曲线，禁用 QK-Norm 在&#x7EA6;**`20%`**&#x8BAD;练周期后发散。

为稳定 Chameleon-7B，除 QK-Norm 外，还需在注意力层和前馈层后引入 dropout，如上图(c)所示。但这一策略不足以稳定 Chameleon-34B，还需对归一化顺序进行调整：<span style="color: rgb(100,37,208); background-color: inherit">在 Transformer 块内采用 Swin Transformer 的归一化策略，能约束前馈块的范数增长</span>。设$$h$$为时间步$$t$$的隐藏向量，在对输入$$x$$应用自注意力后：

**<span style="color: rgb(216,57,49); background-color: inherit">LLaMA-2</span>**

$$\begin{aligned}
h &= x + \text{attention}(\text{attention\_norm}(x))\\
\text{output} &= h + \text{feed\_forward}(\text{ffn\_norm}(h))
\end{aligned}$$

**<span style="color: rgb(46,161,33); background-color: inherit">Chameleon-34B</span>**

$$\begin{aligned}
h &= x + \text{attention\_norm}(\text{attention}(x))\\
\text{output} &= h + \text{ffn\_norm}(\text{feed\_forward}(h))
\end{aligned}$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：几点配置细节
>
> 1. 从头训练时，在 LLaMA-2 参数化发散前，两种归一化顺序在困惑度上无显著差异
>
> 2. 这种归一化与 dropout 组合效果不佳，因&#x6B64;**`Chameleon-34B`**<span style="color: rgb(100,37,208); background-color: inherit">不使用 dropout</span>
>
> 3. 若采用归一化重排，**`Chameleon-7B`**&#x4E5F;可在无 dropout 下稳定训练，但 QK-Norm 在两种情况下均不可或缺

![](../../images/视觉多模态讲义（下）-image-162.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

**Chameleon&#x20;**&#x4E0D;仅处理文本，还<span style="color: rgb(100,37,208); background-color: inherit">将图像表示为一系列离散的 token</span>，并利用自回归 Transformer 的缩放定律。在训练过程中<span style="color: rgb(100,37,208); background-color: inherit">采用任意顺序混合图像与文本，包括纯文本、单个图文对，以及完整的交错式图文文档</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">分词</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">图像分词</span>**

基于 **[Make-A-Scene](https://arxiv.org/pdf/2203.13131)&#x20;**&#x7684;方法训练新的图像分词器，将$$512 \times 512$$的图像编码为$$1024$$个离散 token，这些 token 来自大小为$$8192$$的 codebook。

**<span style="color: rgb(222,120,2); background-color: inherit">文本分词</span>**

使用 sentencepiece 库在训练数据的一个子集上训练了一个新的 BPE 分词器，其词汇表大小&#x4E3A;**`65,536`**，其中包括上&#x8FF0;**`8192`**&#x4E2A;图像 codebook token。

> 由于生成人脸比较重要，在预训练阶段将含有人脸的图像比例上采样 2 倍
>
> 图像分词器在重建包含大量文本的图像时表现不佳，限制了模型在 OCR 相关任务上的能力上限

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

将预训练阶段划分为两个独立阶段：<span style="color: rgb(100,37,208); background-color: inherit">第一阶段占总训练过程的前 80%，第二阶段占后 20%。对于所有图文对进行轮换，使得 50% 的情况下图在前、文在后</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">Stage 1</span>**：使用大规模、完全无监督的数据集混合：

> * **<span style="color: rgb(36,91,219); background-color: inherit">纯文本</span>**：使用多种文本数据集，包括用于训练 LLaMA-2 和 CodeLLaMA 的预训练数据组合，总计&#x7EA6;**`2.9T`**&#x7EAF;文本 token。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图文对</span>**：所有图像被调整大小并中心裁剪为$$512 \times 512$$以供分词。总计包&#x542B;**`1.4B`**&#x56FE;文对，产生&#x7EA6;**`1.5T`**&#x56FE;文 token。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">交错图文</span>**：从公开网络来源获取数据，总计&#x7EA6;**`400B`**&#x4EA4;错图文 token，应用与 Text-to-Image 相同的过滤策略。

**<span style="color: rgb(222,120,2); background-color: inherit">Stage 2</span>**：将第一阶段数据的权重降低 50%，同时混入更高质量的数据集，保持图文 token 的大致比例不变。并加入一组大规模指令微调数据集中经过筛选的训练子集。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练设置 </span>**

虽然 QK-Norm 有助于缓解 Transformer 内部 softmax 的问题，但无法解决最终 softmax &#x7684;**`logit shift`**&#x95EE;题，因此引入 z-loss 正则化：对 softmax 函数 $$\sigma(x)_i = \frac{e^{x_i}}{Z}$$ 的配分函数$$Z = \sum_i e^{x_i}$$，在损失函数中添加项 $$10^{-5} \log^2 Z$$。

训练使用 AdamW 优化器，其中$$\beta_1 = 0.9$$，$$\beta_2 = 0.95$$，$$\epsilon = 10^{-5}$$。学习率采用$$4000$$步线性 warm-up，随后指数衰减至$$0$$。权重衰减设为$$0.1$$，全局梯度裁剪阈值为$$1.0$$。在完整训练数据上训练$$2.1$$个 epoch，总计处&#x7406;**`9.2T`** token。

**<span style="color: rgb(222,120,2); background-color: inherit">Chameleon-7B</span>**

使用 dropout 率$$0.1$$提高训练稳定性

同时使用 dropout 和 z-loss 实现稳定

使用全局 batch size 为$$2^{23}$$（约 8M）token

**<span style="color: rgb(222,120,2); background-color: inherit">Chameleon-34B</span>**

不使用 dropout

仅需 z-loss 即可实现稳定

使用全局 batch size$$3 \times 2^{22}$$（约 12M）token

* **<span style="color: rgb(36,91,219); background-color: inherit">对齐</span>**

采用轻量级对齐阶段，基于精心策划的高质量数据集进行 SFT。数据涵盖多个类型，以充分展现模型能力并提升安全性。

1. **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

将 SFT 数据集分为以下类别：**文本**、**代码**、**视觉对话**、**图像生成**、**交错图文生成** 和 **安全**。

**<span style="color: rgb(36,91,219); background-color: inherit">文本 SFT 数据</span>**：继承自 LLaMA-2

**<span style="color: rgb(36,91,219); background-color: inherit">代码 SFT 数据</span>**：继承自 CodeLLaMA

![](../../images/视觉多模态讲义（下）-image-164.png)

**<span style="color: rgb(36,91,219); background-color: inherit">安全数据</span>**：一系列可能诱使模型生成不安全内容的 prompt，并搭配拒绝的回答，如`我无法协助此事`。这些 prompt 覆盖暴力、管制物质、隐私、色情内容等广泛敏感话题。

**<span style="color: rgb(36,91,219); background-color: inherit">图像生成数据</span>**：使用 LAION-AI 的审美预测器 **[aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)&#x20;**&#x5BF9;图像进行筛选，选取评分至少为$$6$$的图像，并从中选出最接近$$512 \times 512$$尺寸与宽高比的前 64K 张图像。

* **<span style="color: rgb(36,91,219); background-color: inherit">微调方法</span>**

在 SFT 阶段平衡各模态至关重要。<span style="color: rgb(216,57,49); background-color: inherit">若模态配对严重失衡，模型会学习到无条件生成该模态的先验，导致某一模态生成被抑制或过度放大</span>。

SFT 采用余弦学习率调度，初始学习率为$$1 \times 10^{-5}$$，权重衰减为$$0.1$$。batch size 设为$$128$$，支持最长$$4096$$ token 的序列。每个训练样本由一个 prom 及其对应 response 组成。为提升效率，将多个 prompt-response 对打包进同一序列，并插入特殊 token 标记 prompt 结束与 response 开始。采用自回归训练目标，仅对回答 token 计算损失。这种针对性优化带来轻微整体增益。此外，使用 dropout 率$$0.05$$，并保留预训练阶段使用的 z-loss。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：在 SFT 中，**<span style="color: rgb(100,37,208); background-color: inherit">prompt 中的图像</span>**<span style="color: rgb(100,37,208); background-color: inherit">通过边界填充调整大小，以保留全部信息；</span>**<span style="color: rgb(100,37,208); background-color: inherit">response 中的图像</span>**<span style="color: rgb(100,37,208); background-color: inherit">则采用中心裁剪，以确保生成图像的视觉质量</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理</span>**

自回归、多模态混合生成在推理时会存在很多问题，包括：

> * **<span style="color: rgb(36,91,219); background-color: inherit">每步数据依赖</span>**：由于解码逻辑取决于当前步生成的是图像还是文本，每步都需检查 token 以引导控制流，即以阻塞方式从 GPU 复制到 CPU
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">模态约束生成的 mask</span>**：为支持单一模态生成，需对非目标模态的 token 进行掩码并在 de-tokenizing 时忽略
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">固定大小的文本单元</span>**：与可变长度的纯文本生成不同，基于 token 的图像生成产生固定大小的 token 块

**Chameleon&#x20;**&#x63A8;理实现支持文本与图像的流式生成。<span style="color: rgb(100,37,208); background-color: inherit">流式生成时，每步需执行 token 依赖的条件逻辑；非流式生成时，图像 token 块可融合生成，无需条件计算</span>。在所有情况下，token 掩码消除了 GPU 上的分支。即使在非流式设置下，生成文本时仍需检查每个输出 token 是否为图像起始 token，以触发图像专用的解码增强。

### 5.2.2 **<span style="color: rgb(36,91,219); background-color: inherit">Emu3（b-1）</span>**

Emu3 从零开始在语言、图像和视频数据的混合上进行训练。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">语言数据</span>**：使用与 Aquila 相同的语言数据，包含中文和英文的高质量语料库。

**<span style="color: rgb(36,91,219); background-color: inherit">图像数据</span>**：包含开源网络数据、AI 生成数据以及高质量的内部数据。过滤过程包括以下几个关键步骤： &#x20;

> 1. 应用分辨率过滤器，剔除分辨率低于$$512 \times 512$$像素的样本
>
> 2. 使用 LAION-AI 的审美预测器 [aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)**&#x20;**<span style="color: rgb(100,37,208); background-color: inherit">对每张图像的审美质量进行评估</span>，剔除评分低于$$5.5$$的图像，以确保整体审美质量
>
> 3) 对于未通过审美过滤的图像，<span style="color: rgb(100,37,208); background-color: inherit">采用文本检测和颜色过滤，保留非单色且文本内容极少的图像</span>，从而提升对开放世界图像的过滤召回率
>
> 4) 还为图像理解任务准备了补充数据。按照 DenseFusion 的数据处理流程，<span style="color: rgb(100,37,208); background-color: inherit">从多样化的开源网络数据中提取百万张具有代表性的图像</span>，涵盖图表、表格、富含文本的内容等多种类别。

为标注过滤后的数据集，基于 Emu2 开发了一个图像描述模型，用于生成密集的合成 captions。<span style="color: rgb(100,37,208); background-color: inherit">利用 GPT-4V 并结合详细 prompt，生成约</span>$$100$$<span style="color: rgb(100,37,208); background-color: inherit">万对图像-描述数据</span>。这个标注数据集随后用于微调 Emu2-17B 模型，作为图像 captioner。使用 vLLM 加速标注过程。

**<span style="color: rgb(36,91,219); background-color: inherit">视频数据</span>**：收集多种类别的视频，如<span style="color: rgb(220,155,4); background-color: inherit">风景、动物</span> <span style="color: rgb(220,155,4); background-color: inherit">、植物、游戏和动作</span>等。这些视频通过一个复杂的预处理流程进行处理，包含以下四个阶段： &#x20;

> 1. 使用 [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)**&#x20;**&#x5C06;视频分割为场景，其中 ContentDetector 用于识别内容变化，ThresholdDetector 用于检测淡入/淡出事件
>
> 2. 使用 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 进行文本检测，并移除文本覆盖过多的片段。为降低计算开销，以 2 FPS 采样视频帧，并将短边缩放至 256 像素
>
> 3) 进一步计算光流以剔除运动过少或极端剧烈的片段。与上一步类似的，对视频帧进行采样和缩放以提高效率。<span style="color: rgb(100,37,208); background-color: inherit">光流得分定义为所有像素平均光流幅值与短边长度的比值，剔除光流得分超出可接受范围的片段</span>； &#x20;
>
> 4) 使用 LAION-AI 审美预测器 [aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)**&#x20;**&#x8BC4;估每个片段的审美质量。<span style="color: rgb(100,37,208); background-color: inherit">对每个片段采样三帧并获得三个得分，若最低得分小于</span>$$5$$<span style="color: rgb(100,37,208); background-color: inherit">，则丢弃该片段</span>。

使用基于上述图像 captioner 训练得到的视频 captioner 对过滤后的视频片段进行标注。训练数据最初由 GPT-4V 标注：<span style="color: rgb(100,37,208); background-color: inherit">对每个视频片段采样</span>$$8$$<span style="color: rgb(100,37,208); background-color: inherit">帧，并构造详细 prompt 让 GPT-4V 描述这些帧中的内容和运动。部分标注数据经过人工更正</span>。随后在标注数据上微调图像 captioner，从而得到视频 captioner。为支持大规模部署，使用 vLLM 加速 captioning 生成：对于时长不足 20 秒的片段，使用 12 帧均匀采样进行描述；更长的片段则被分割为 10–20 秒的子片段，分别独立描述。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉分词器 Vision Tokenizer</span>**

基于 [SBER-MoVQGAN](https://github.com/sberbank-ai/SBER-MoVQGAN) 训练视觉分词器，可将一段$$4 \times 512 \times 512$$的视频片段或一张$$512 \times 512$$的图像编码为$$4096$$个离散 token，这些 token 来自大小为$$32,768$$的 codebook。<span style="color: rgb(100,37,208); background-color: inherit">分词器在时间维度上实现</span>$$4\times$$<span style="color: rgb(100,37,208); background-color: inherit">压缩，在空间维度上实现</span>$$8\times8$$<span style="color: rgb(100,37,208); background-color: inherit">压缩，适用于任意时间和空间分辨率。在 MoVQGAN 架构的基础上，在编码器和解码器模块中均引入了两个带有 3D 卷积核的时间残差层，以增强视频分词能力</span>。分词器在 [LAION-High-Resolution](https://laion.ai/blog/laion-high-resolution) 图像数据集和 InternVid 视频数据集上进行端到端训练，目标函数结合了$$L_2$$损失、LPIPS 感知损失、GAN 损失和 commitment 损失。

右表展示了在 [Pexels](https://www.pexels.com) 提供&#x7684;**`3,172`**&#x4E2A;视频构成的评估集上的 LPIPS、PSNR 和 SSIM 分数。视频重建时长为 5 秒，并保持原始宽高比。评估过程中，原始视频与重建视频均根据短边进行缩放和裁剪，并以 12 FPS 均匀采样 8 帧。

![](../../images/视觉多模态讲义（下）-image-159.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

Emu3 保留了 LLaMA-2 的架构框架，主要改动：

> * 扩展 Embedding 层以容纳离散视觉 token
>
> * 使&#x7528;**`RMSNorm`**&#x8FDB;行归一化
>
> * 采&#x7528;**`GQA`**&#x4F5C;为注意力机制
>
> * 使&#x7528;**`SwiGLU`**&#x6FC0;活函数
>
> * 使用旋转位置编&#x7801;**`RoPE`**
>
> * 移除 QKV 投影层和线性投影层中的偏置项
>
> * 为提升训练稳定性，使用$$0.1$$的 dropout 率
>
> * 多语言文本使用 [QwenTokenizer](https://github.com/QwenLM/Qwen) 进行分词

![](../../images/视觉多模态讲义（下）-image-160.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

- **<span style="color: rgb(36,91,219); background-color: inherit">训练数据</span>**

首先需要定义多模态数据格式。Emu3 原生集成文本条件信息用于图像视频生成，<span style="color: rgb(100,37,208); background-color: inherit">将图像视频在保持宽高比的前提下缩放到面积接近</span>$$512 \times 512$$<span style="color: rgb(100,37,208); background-color: inherit">的尺寸，然后使用视觉分词器生成视觉 token。接着引入五个特殊 token 来融合文本与视觉数据，构建类似文档的输入格式用于训练</span>。最终训练数据结构如下：

$$[\text{BOS}]~\{\text{caption text}\}~[\text{SOV}]~\{\text{meta text}\}~[\text{SOT}]~\{\text{vision tokens}\}~[\text{EOV}]~[\text{EOS}]$$

> * **`[BOS]`**&#x548C;**`[EOS]`**&#x662F;文本分词器原有的特殊 token；**`[SOV]`**&#x6807;记视觉输入的开始；**`[SOT]`**&#x6807;记视觉 token 的开始；**`[EOV]`**&#x8868;示视觉输入的结束
>
> * 在视觉 token 中插&#x5165;**`[EOL]`**&#x548C;**`[EOF]`**&#x5206;别表示换行和帧边界
>
> * **`meta text`**&#x4EE5;纯文本形式包含图像的分辨率信息；对于视频，则还包括分辨率、帧率和时长
>
> * 将部分数据集中&#x7684;**`caption text`**&#x5B57;段移&#x81F3;**`[EOV]`**&#x4E4B;后，从而构建面向视觉理解任务的数据

* **<span style="color: rgb(36,91,219); background-color: inherit">训练细节</span>**

由于 Emu3 中的视觉信号已完全转换为离散 token，仅&#x9700;**<span style="color: rgb(100,37,208); background-color: inherit">使用标准交叉熵损失进行 next-token 预测</span>**&#x4EFB;务。<span style="color: rgb(100,37,208); background-color: inherit">为防止视觉 token 主导学习过程，对与视觉 token 相关的损失施加</span>$$0.5$$<span style="color: rgb(100,37,208); background-color: inherit">的权重</span>。

Emu3 在预训练中采用极长上下文长度以处理视频数据，<span style="color: rgb(100,37,208); background-color: inherit">结合</span>**<span style="color: rgb(100,37,208); background-color: inherit">张量并行 TP</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">上下文并行 CP</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 和</span>**<span style="color: rgb(100,37,208); background-color: inherit">数据并行 DP</span>**<span style="color: rgb(100,37,208); background-color: inherit">。将文本-图像数据打包至最大上下文长度以充分利用计算资源，同时确保完整图像在打包过程中不被分割</span>。预训练分为两个阶段：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Step 1</span>**：不使用视频数据，从零开始&#x4EE5;**`5120`**&#x7684;上下文长度训练文本和图像数据
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Step 2</span>**：引入视频数据，上下文长度提升&#x81F3;**`131072`**

两个阶段均使用$$5 \times 10^{-5}$$的初始学习率，并采用余弦退火策略将学习率降至零。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

- **<span style="color: rgb(36,91,219); background-color: inherit">视觉生成</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">质量微调 QFT（Quality Fine-Tuning）</span>**

在预训练之后，针对视觉生成任务进行后训练以提升生成质量。使用高质量数据进行质量微调，模型继续<span style="color: rgb(100,37,208); background-color: inherit">使用标准交叉熵损失进行 next-token 预测，但仅对视觉 token 施加监督</span>。对于 QFT 中的图像数据，选择多样化的高质量来源，并基于三种主流偏好评分的平均值进行筛选：**HPSv2.1**、**MPS&#x20;**&#x548C; **LAION** 审美评分。在 QFT 过程中，<span style="color: rgb(100,37,208); background-color: inherit">将训练数据分辨率从 512 像素提升至 720 像素以改善生成质量</span>。对于视频数据，从高质量来源采样，并施加严格的分辨率和光流过滤以确保质量。在训练末期，采用退火策略将学习率线性衰减至零。

**<span style="color: rgb(222,120,2); background-color: inherit">直接偏好优化 DPO</span>**

利用人类偏好数据提升模型性能。数据集构建分为三步： &#x20;

> 1. 对每个用户收集的 prompt $$p$$，使用 QFT 模型执行 8–10 次推理，构建初始数据池$$x$$ &#x20;
>
> 2. 由三位评审员对每个 prompt 的输出进行评估，重点关注视觉吸引力和与提示的一致性 &#x20;
>
> 3) 根据评分，选取最高分样本作&#x4E3A;**`chosen`**&#x6837;本，最低分作&#x4E3A;**`rejected`**&#x6837;本，形成三元组 $$(p_i, x^{\text{chosen}}_i, x^{\text{rejected}}_i)$$ 用于后续训练。
>
> > 这里直接存储数据构建过程中生成的 token，以便在后续训练中复用，从而避免因重新分词导致的重建差异。

Emu3-DPO 通过<span style="color: rgb(100,37,208); background-color: inherit">联合最小化 DPO 损失和 next-token 预测的交叉熵损失来微调 QFT 模型</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉-语言理解</span>**

预训练模型还需经过两阶段后训练以提升视觉-语言理解能力： &#x20;

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">图像到文本训练</span>**：将图像理解数据与纯语言数据混合训练，且在纯文本预测任务中忽略与视觉 token 相关的损失。每张图像在保持原始宽高比的前提下缩放至约$$512 \times 512$$分辨率
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">指令微调</span>**：从 LLaVA-OneVision 中采样子集问答对，以增强模型遵循视觉指令的能力。对于分辨率低于$$512 \times 512$$或高于$$1024 \times 1024$$的图像，分别缩放至下限或上限分辨率并保持宽高比；其余图像则保留原始分辨率

### 5.2.3 <span style="color: rgb(36,91,219); background-color: inherit">OmniGen2（b-2）</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

在原始的 OmniGen 中，<span style="color: rgb(100,37,208); background-color: inherit">对文本采用自回归建模，对图像生成采用基于扩散的方法</span>，二者均在 Transformer 架构中实现，并使&#x7528;**`phi-3`**&#x521D;始化。OmniGen2 进行了进一步的探索：

> 1. 将 phi-3 替换为 Qwen 模型，但<span style="color: rgb(216,57,49); background-color: inherit">使用了更强的 LLM 后，图像生成质量有所下降</span>
>
> 2. 探索混合专家 MoE 策略，以独立路由文本和图像参数，类似于 LMfusion 中采用的方法。但是若<span style="color: rgb(216,57,49); background-color: inherit">将图像分支的参数初始化为从文本分支导出的参数，其性能反而不如直接对图像通路进行随机初始化</span>

这些结果表明，**<span style="color: rgb(100,37,208); background-color: inherit">为文本优化的参数并不适用于图像建模</span>**。因此在 OmniGen2 中，<span style="color: rgb(100,37,208); background-color: inherit">将扩散过程解耦，并对其参数进行随机初始化</span>。

**`MetaQuery`**&#x548C;**`BLIP-3o`**&#x7B49;方法采用可学习的 Query token 来编码用于扩散生成的条件信息。这些方法<span style="color: rgb(216,57,49); background-color: inherit">将所有条件信息压缩为固定数量的 token</span>，这不可避免地<span style="color: rgb(216,57,49); background-color: inherit">限制了表示能力并导致信息损失</span>。并且这种基于 token 的压缩方式<span style="color: rgb(216,57,49); background-color: inherit">难以处理长文本渲染任务</span>。因此，**<span style="color: rgb(100,37,208); background-color: inherit">OmniGen2 采用多模态交错条件在 MLLM 中产生的隐藏状态作为扩散解码器的输入</span>**，而非依赖一组固定的可学习 Query token。

现有 MLLM 主要使用 ViT 进行图像建模，但 <span style="color: rgb(216,57,49); background-color: inherit">ViT 往往难以捕捉细粒度的视觉细节，从而在图像生成任务中降低图像保真度</span>。虽然对 ViT 特征进行端到端训练可在一定程度上缓解这一限制，但会<span style="color: rgb(216,57,49); background-color: inherit">引入额外的复杂性，特别是在平衡图像理解与生成任务之间</span>。近期的工作&#x5982;**`BAGEL`**&#x548C;**`Mogao`**&#x901A;过双重编码图像，在模型中同时引入 VAE 和 ViT 特征来解决这个问题。但这种<span style="color: rgb(216,57,49); background-color: inherit">双编码方法需要大量的架构修改以及复杂的注意力机制，从而显著增加了开发复杂度</span>。此外，<span style="color: rgb(216,57,49); background-color: inherit">将模型适配到这种新架构还需要重新训练</span>，以恢复其图像理解能力。因此 **<span style="color: rgb(100,37,208); background-color: inherit">OmniGen2 仅将 VAE 作为扩散解码器的输入，而没有将其集成到 MLLM 中</span>**。该策略保留了 MLLM 的架构简洁性，并在无需大规模重训练的前提下维持其多模态理解能力。

![](../../images/视觉多模态讲义（下）-image-158.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态大语言模型 MLLM</span>**

OmniGen2 <span style="color: rgb(100,37,208); background-color: inherit">利用 MLLM Transformer 来处理文本和视觉输入。对于文本生成任务，采用自回归语言头；而图像生成则通过专用的扩散模块完成</span>。Transformer 主干网络&#x4EE5;**`Qwen2.5-VL-3B`**&#x521D;始化，并引入一个特殊 token **`<|img|>`**，用于<span style="color: rgb(100,37,208); background-color: inherit">在输出序列中显式指示图像生成。当模型遇到该 token 时，将触发 Diffusion Decoder 合成对应的图像，MLLM 产生的隐藏状态作为扩散解码器的条件输入</span>。但由于这些隐藏状态可能缺乏详细的视觉信息，进一步<span style="color: rgb(100,37,208); background-color: inherit">通过从输入图像中提取的 VAE 特征对解码器进行增强</span>。随后，Diffusion Decoder 采&#x7528;**`Rectified Flow`**&#x751F;成图像。

![多模态 Reflection Flow 图像生成](../../images/视觉多模态讲义（下）-image-161.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Diffusion Transformer</span>**

OmniGen2 采用简单的 Diffusion Transformer 架构，<span style="color: rgb(100,37,208); background-color: inherit">直接拼接来自 MLLM、VAE 和噪声的特征，从而在这些模态上执行联合注意力机制</span>。借鉴 Lumina-Image 2.0 的思想，多个输入条件首先通过一&#x4E2A;**`Refiner Network`**&#x8FDB;行处理，以确保对齐后再送入 Transformer 层。Diffusion Decoder 包&#x542B;**`32`**&#x5C42;，隐藏层维度&#x4E3A;**`2520`**，总参数量约&#x4E3A;**`4B`**。由于显式引入了 VAE 特征，MLLM 中对应图像的隐藏状态变得不那么关键。<span style="color: rgb(100,37,208); background-color: inherit">为降低计算开销，在 MLLM 中丢弃与图像相关的隐藏状态，仅保留与文本 token 相关的部分</span>。此外在 Diffusion Transformer 中采用了 3D RoPE，是对 Qwen MRoPE 的改进。

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态旋转位置编码 MRoPE</span>**

OmniGen2 提出新的位置编码：Omni-RoPE，用于多样化且复杂的任务需求，尤其是图像编辑和上下文内生成。Omni-RoPE 被分解为三个独立模块：

![](../../images/视觉多模态讲义（下）-image-163.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">序列与模态标识符</span>**$$\text{id}_\text{seq}$$：主要作用是区分来自不同模态和序列的 token。关键在于<span style="color: rgb(100,37,208); background-color: inherit">将每张图像视为一个完整的语义单元，因此属于同一图像的所有 token 被赋予相同且恒定的 ID</span>。对于文本 token，该 ID <span style="color: rgb(100,37,208); background-color: inherit">随每个后续 token 单调递增</span>，起到标准一维位置索引的作用以保留词序。这里等同于 Qwen2-VL 中原始的 MRoPE。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">二维空间高度坐标</span>$$h$$**：表示图像 token 的归一化垂直位置。
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">二维空间宽度坐标</span>$$w$$**：表示图像 token 的归一化水平位置。对于所有非图像 token，两个空间坐标$$(h, w)$$均设为零。

这里的难点在于这些组件如何协同工作。对于每个图像实体，无论是源图像还是目标图像，其<span style="color: rgb(100,37,208); background-color: inherit">空间坐标</span>$$(h, w)$$<span style="color: rgb(100,37,208); background-color: inherit">均从</span>$$(0, 0)$$<span style="color: rgb(100,37,208); background-color: inherit">独立计算，确保了处于对应位置的 token 具有相同的空间 Embedding，从而在编辑过程中强烈鼓励一致性并保留未修改区域</span>。尽管空间坐标是局部定义的，但<span style="color: rgb(100,37,208); background-color: inherit">唯一的序列与模态标识符 </span>$$\text{id}_\text{seq}$$<span style="color: rgb(100,37,208); background-color: inherit"> 提供了一种明确机制，用以区分不同的图像实体</span>。这种整体设计<span style="color: rgb(46,161,33); background-color: inherit">在纯文本输入时可自然退化为标准的一维位置编码，使 MRoPE 成为一个灵活且鲁棒的框架，能够有效支持全系的多模态操作</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练数据</span>**

> - **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**：直接采&#x7528;**`LLaVA-OneVision`**&#x7684;数据。
>
> - **<span style="color: rgb(36,91,219); background-color: inherit">文生图 T2I 训练</span>**：整合&#x7EA6;**`140M`**&#x5F00;源图像（来自 Recap-DataComp、LAION-Aesthetic、BLIP3-o 等多个高质量来源）+ **`10M`**&#x81EA;采图像（使用 Qwen2.5-VL-72B 自动生成合成标注）
>
> - **<span style="color: rgb(36,91,219); background-color: inherit">图像编辑任务</span>**：在现有公开编辑数据集（&#x5982;**`SEED-Edit`**、**`UltraEdit`**&#x7B49;）基础上，<span style="color: rgb(100,37,208); background-color: inherit">自建高质量、高多样性、指令精准的新编辑数据集</span>，以弥补开源数据在质量、指令准确性和任务覆盖上的不足

* **<span style="color: rgb(36,91,219); background-color: inherit">上下文感知数据 In-Context Data</span>**

支持模型从给定图像中提取视觉概念，如特定物体或人物，并在新图像中复现，即免微调的个性化生成

**<span style="color: rgb(222,120,2); background-color: inherit">上下文生成 In-Context Generation</span>**

利用视频帧序列，同一主体在不同姿态、光照、视角下的自然变化，流程：

> 1. 从视频中抽取关键帧，选定基准帧
>
> 2. 用 Qwen2.5-VL 识别语义显著主体，结合 GroundingDINO 定位 + SAM2 跟踪
>
> 3) 选取外观差异最大的有效帧对，确保语义一致但视觉多样
>
> 4) 引入 VLM 过滤跟踪错误，用 FLUX.1-Fill-dev 更换背景以增强泛化
>
> 5. 基于 DINO 相似度和 VLM 质量评估筛选样本
>
> 6. 自动生成对象描述与自然语言指令

输出`(指令, 改写输入图, 原始目标图)`三元组，用于多主体上下文生成训练

**<span style="color: rgb(222,120,2); background-color: inherit">上下文编辑 In-Context Editing</span>**

从上下文图像中提取元素，用于编辑另一张目标图像，构建方式：

> * 同一视频中选两帧：一作上下文，保留主体+新背景；一作目标，移除主体保留背景
>
> * 使用 FLUX.1-Fill-dev 分别做 outpainting（上下文）和 inpainting（目标）；
>
> * 由 Qwen2.5-VL-72B 生成描述“如何从输入图变为目标图”的自然语言指令，并融合上下文中的对象描述。

实现了**跨图像元素迁移+指令化编辑**的联合监督。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像编辑数据</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">高质量 Inpainting 数据</span>**

现有 inpainting 数据集存在图像质量差、指令与图像不匹配两大缺陷。解决方案：

> * 从高质量 T2I 图像出发，用 FLUX.1-Fill-dev **随机 inpaint（不输入指令）**，得到 (inpainted → original) 对；
>
> * 再用 MLLM（如 Qwen2.5-VL）**反向生成精准编辑指令**；
>
> * 结果：指令与图像高度对齐，构建出高精度编辑数据集。

**<span style="color: rgb(222,120,2); background-color: inherit">视频驱动的编辑对</span>**

传统 inpainting 无法支持动作、表情、物体移动等动态编辑。方法：

> * 按场景分割视频，避免跨场景配对；
>
> * 利用 RGB/HSV 差异检测场景边界；
>
> * 在同场景内筛选**仅局部变化**的帧对（通过 DINOv2 + CLIP 过滤大/小变化对）；
>
> * 提出**分块颜色直方图比对法**：高效判断视角是否一致，过滤因摄像机运动导致的无效对；
>
> * 最终用 Qwen2.5-VL-72B 为保留的帧对生成精确编辑指令。

实现了**真实世界动态编辑**，如<span style="color: rgb(220,155,4); background-color: inherit">挥手、转头、物体位移</span>的数据构建。

* **<span style="color: rgb(36,91,219); background-color: inherit">交错多模态序列数据 Interleaved Data</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">交错视频帧 Interleaved Frames</span>**

从视频中提取**同场景内**或**跨场景**的最多5帧序列；用 MLLM（Qwen2.5-VL-7B-Instruct）为每对连续帧生成描述性 caption，涵盖：

> * 动作/行为变化
>
> * 环境/背景差异
>
> * 外观变化

构建 **800K 条**多模态交错序列，用于预训练模型处理连续视觉-语言流的能力。

**<span style="color: rgb(222,120,2); background-color: inherit">反思数据 Reflection Data</span>**

借鉴 LLM 的 test-time scaling 与 self-reflection 机制。让多模态模型具备“自我批评-迭代优化”能力。构建流程：

> 1. 用当前模型生成 T2I 图像
>
> 2. 用强大多模态模型（如 Doubao-1.5-pro）评估图像是否符合指令
>
> 3) 若不符，生成结构化反思：① 指出缺陷；② 提出修改建议
>
> 4) 将 (原始指令, 生成图, 反思文本) 组成训练样本，微调模型
>
> 5. 迭代多轮，形成多阶反思数据

首次系统探索**反思机制在多模态生成中的应用**，为未来引入在线强化学习奠定基础。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练策略</span>**

MLLM 以 Qwen2.5-VL 初始化，<span style="color: rgb(100,37,208); background-color: inherit">训练过程中大部分参数保持冻结，以保留其多模态理解能力，仅更新新引入的特殊 token </span>**`<|img|>`**。扩散模型从零开始训练，<span style="color: rgb(100,37,208); background-color: inherit">首先在文本到图像 T2I 生成任务上进行训练，随后采用混合任务训练策略以兼顾多个目标</span>。在反思训练 reflection training 阶段，<span style="color: rgb(100,37,208); background-color: inherit">所有模型参数均被解冻</span>，使模型能够生成反思性的文本描述，并迭代优化图像输出。

### 5.2.4 <span style="color: rgb(36,91,219); background-color: inherit">X-Omni（b-2）</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

**X-Omni** 基于一个自回归模型构建，包含一&#x4E2A;**`SigLIP-VQ tokenizer`**&#x4EE5;及用于图像生成的扩散解码&#x5668;**`Diffusion Decoder`**。这使图像和文本输入输出均可在统一的自回归框架内实现类文本的分词 Tokenization 与逆分词Detokenization，从而在统一的自回归架构中整合图像与文本的 token。



1. **<span style="color: rgb(36,91,219); background-color: inherit">图像分词 Image Tokenization</span>**

![](../../images/视觉多模态讲义（下）-image-154.png)

图像 tokenizer 是在视觉理解任务上进行训练的。为了将连续的图像转换为离散 token 的同时保留丰富的语义信息，作者<span style="color: rgb(100,37,208); background-color: inherit">选择预训练的</span>**`SigLIP2-g ViT`**<span style="color: rgb(100,37,208); background-color: inherit">作为视觉语义提取器。在 ViT 编码器基础上，引入一个向量量化器作为图像 tokenizer，并将其在视觉理解任务上与预训练的</span>**`Qwen2.5-1.5B`**<span style="color: rgb(100,37,208); background-color: inherit">对齐</span>。该向量量化器采用大小&#x4E3A;**`16,384`**、维度&#x4E3A;**`2,048`**&#x7684; codebook，并在视觉 tokenizer 与 LLM 之间使用残差块作为适配器。<span style="color: rgb(100,37,208); background-color: inherit">视觉编码器与向量量化器共同构成 SigLIP-VQ 图像 tokenizer</span>。在<span style="color: rgb(100,37,208); background-color: inherit">后续训练阶段中，这两个组件均保持冻结状态，以确保分词过程的稳定性与一致性</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">自回归建模 Autoregressive Modeling</span>**

将图像分词为离散 token 后，<span style="color: rgb(100,37,208); background-color: inherit">视觉 token 与语言 token 可在自回归架构下通过多模态建模自然地统一起来</span>。作者采&#x7528;**`Qwen2.5-7B`**&#x4F5C;为基础预训练模型。为了将视觉感知能力融入纯文本的语言模型中，在<span style="color: rgb(100,37,208); background-color: inherit">原始 Transformer 层前后各插入四个</span>**<span style="color: rgb(100,37,208); background-color: inherit">随机初始化的、专用于视觉的</span>**<span style="color: rgb(100,37,208); background-color: inherit">模块</span>。这些视觉专用模块<span style="color: rgb(100,37,208); background-color: inherit">采用与标准 Transformer 块相同的结构配置，仅对图像 token 进行操作，不影响文本 token</span>。此外还<span style="color: rgb(100,37,208); background-color: inherit">为图像 token 引入了随机初始化的 Embedding 层和分类头</span>。相比原始语言模型，这些架构的修改并未增加额外的基础设施复杂性，同时完全兼容张量并行、流水线并行和上下文并行等分布式训练策略。

对于视觉生成与理解任务，视觉 token 和语言 token 被拼接成一个统一的多模态序列，并输入自回归模型进行下一 token 预测训练。

> * &#x5728;**<span style="color: rgb(36,91,219); background-color: inherit">理解任务</span>**&#x4E2D;，仅监督语言 token
>
> * &#x5728;**<span style="color: rgb(36,91,219); background-color: inherit">生成任务</span>**&#x4E2D;，仅监督视觉 token

为支持任意图像分辨率，在视觉 token 前添加一个分辨率信息前缀，格式为：

$$\text{language tokens } \langle \text{SOM} \rangle\ \text{height width } \langle Image \rangle\ \text{visual tokens } \langle \text{EOM} \rangle\ \text{language tokens}$$

其中<span style="color: rgb(100,37,208); background-color: inherit">特殊 token </span>$$\langle \text{SOM} \rangle$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\langle \text{EOM} \rangle$$<span style="color: rgb(100,37,208); background-color: inherit">分别表示多模态序列中的起始与结束标记。</span>**`height`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`width`**<span style="color: rgb(100,37,208); background-color: inherit">是代表二维图像 token 空间尺寸的文本 token。在特殊token</span>$$\langle \text{Image} \rangle$$<span style="color: rgb(100,37,208); background-color: inherit">之后，提供展平后的图像 token，其长度为</span>$$\text{height}\times\text{width}$$<span style="color: rgb(100,37,208); background-color: inherit">个 token。作者使用与原始语言模型一致的一维</span>**`RoPE`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">扩散解码器 Diffusion Decoder</span>**

使用一个充分预训练的扩散模型作为视觉解码器，从离散的语义 token 中重建图像像素。具体来说就是<span style="color: rgb(100,37,208); background-color: inherit">添加一个线性层，将语义 token 映射到</span>**`FLUX.1-dev`**<span style="color: rgb(100,37,208); background-color: inherit">的特征通道维度，并将其集成到中间层特征中</span>。扩散解码器的训练目标是重建图像，<span style="color: rgb(100,37,208); background-color: inherit">以图像 tokenizer 提取的语义 token 作为输入</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">GRPO 强化学习训练</span>**

为了<span style="color: rgb(100,37,208); background-color: inherit">弥补训练过程中扩散解码器所使用的语义 token 与自回归模型生成的语义 token 之间的分布差异，采用强化学习训练自回归模型</span>。这种贯穿整个采样过程的引导<span style="color: rgb(46,161,33); background-color: inherit">有助于缓解误差传播，同时确保自回归模型的输出分布符合扩散解码器的期望</span>。强训练流程如下图：

![](../../images/视觉多模态讲义（下）-image-157.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">RL 算法</span>**

采用 GRPO 算法，对于每个文本 prompt$$p \sim D$$，使用<span style="color: rgb(100,37,208); background-color: inherit">旧策略</span>$$  \pi_{\theta_{\text{old}}}  $$<span style="color: rgb(100,37,208); background-color: inherit">生成一组</span>$$  G  $$<span style="color: rgb(100,37,208); background-color: inherit">条轨迹</span>$$\{o_1, o_2, ..., o_G\}$$<span style="color: rgb(100,37,208); background-color: inherit">。这些轨迹随后由固定的扩散解码器解码，得到对应的图像</span>$$\{I_1, I_2, ..., I_G\}$$<span style="color: rgb(100,37,208); background-color: inherit">，每幅图像通过奖励函数评分，获得标量值</span>$$\{r_1, ..., r_G\}$$。作者按照原始 GRPO 流程对该组奖励进行归一化，计算优势值$$A_i$$。策略模型$$  \pi_\theta  $$通过最大化以下目标函数进行优化：

$$J_{\text{GRPO}}(\theta) = \mathbb{E}_{p \sim D,\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|p)} \left[ \frac{1}{G} \sum_{i=1}^{G} \left( \min \left( \frac{\pi_\theta(o_i|p)}{\pi_{\theta_{\text{old}}}(o_i|p)} A_i,\ \text{clip} \left( \frac{\pi_\theta(o_i|p)}{\pi_{\theta_{\text{old}}}(o_i|p)},\ 1 - \epsilon,\ 1 + \epsilon \right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\theta_{\text{ref}}}) \right) \right]$$

其中$$  \epsilon  $$和$$  \beta  $$是超参数，$$\pi_{\theta_{\text{ref}}}$$是参考策略，$$D_{\text{KL}}$$使用无偏估计进行估算。这能够在<span style="color: rgb(46,161,33); background-color: inherit">最大化奖励的同时控制与稳定参考模型的偏离程度，从而高效地微调策略</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">奖励函数设计</span>**

奖励函数包含多个部分，每个部分可以监督图像生成质量的不同方面。这在强化学习过程中可以提供全面指导，<span style="color: rgb(100,37,208); background-color: inherit">涵盖</span>**<span style="color: rgb(100,37,208); background-color: inherit">美学质量</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">图文对齐</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本渲染准确性</span>**<span style="color: rgb(100,37,208); background-color: inherit">等关键维度</span>。各个奖励信号<span style="color: rgb(100,37,208); background-color: inherit">通过加权聚合机制组合，形成最终的奖励分数，指导强化学习的优化过程</span>。这种多维度的方法<span style="color: rgb(46,161,33); background-color: inherit">确保模型能够同时平衡多种质量标准，生成在多个维度上符合人类预期的高保真图像</span>。

> * **<span style="color: rgb(36,91,219); background-color: inherit">人类偏好得分</span>**：<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**`HPSv2`**<span style="color: rgb(100,37,208); background-color: inherit">来评估美学质量和人类偏好一致性</span>。能<span style="color: rgb(46,161,33); background-color: inherit">有效预测人类对生成图像的偏好，并在不同图像分布上表现出强泛化能力</span>，是朝向美观且符合人类审美方向优化的关键组件。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">统一奖励得分</span>**：**`HPSv2`**&#x5728;**`224×224`**&#x5206;辨率下运行，但模型<span style="color: rgb(216,57,49); background-color: inherit">专注于高分辨率图像生成</span>。为此<span style="color: rgb(100,37,208); background-color: inherit">引入统一奖励模型来进行人类对齐评估</span>。将多个质量维度聚合为一个统一分数，为强化学习提供整体反馈。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图文对齐得分</span>**：为确保输入 prompt 与生成图像之间的语义一致性，利&#x7528;**`Qwen2.5-VL-32B`**&#x8BA1;算对齐奖励。借助其强大的图像理解能力，可以<span style="color: rgb(100,37,208); background-color: inherit">判断生成图像是否准确反映了 prompt 中描述的内容</span>。这个对齐分数<span style="color: rgb(100,37,208); background-color: inherit">量化了文本描述与视觉内容之间的对应关系，鼓励生成上下文相关的图像，同时减少语义幻觉</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">OCR 准确率得分</span>**：对于需要在图像中生成文字的场景，实现了基于 OCR 的奖励机制，用于量化渲染文本相对于真实标签的保真度。作者<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**`GOT-OCR2.0`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`PaddleOCR`**<span style="color: rgb(100,37,208); background-color: inherit">联合评估图像，并计算文本渲染的准确率得分</span>。这为文本到图像合成提供了关键指导，能够可靠地生成清晰且精确的文本内容。

### 5.2.5 <span style="color: rgb(36,91,219); background-color: inherit">Qwen-Image（b-2）</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

图像生成模型，包括文本到<span style="color: rgb(216,57,49); background-color: inherit">图像生成</span> **<span style="color: rgb(216,57,49); background-color: inherit">T2I</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ext-</span>**<span style="color: rgb(216,57,49); background-color: inherit">to</span>**<span style="color: rgb(216,57,49); background-color: inherit">-</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage generation）</span>和<span style="color: rgb(216,57,49); background-color: inherit">图像编辑 </span>**<span style="color: rgb(216,57,49); background-color: inherit">TI2I</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ext-</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage-</span>**<span style="color: rgb(216,57,49); background-color: inherit">to</span>**<span style="color: rgb(216,57,49); background-color: inherit">-</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage）</span>使计算机能够根据文本提示合成或修改视觉上精彩且语义连贯的内容。过去几年取得了显著进展，尤其是基于扩散模型的架构，能够在捕获细粒度语义细节的同时生成高分辨率图像。

尽管取得了这些进展，但仍存在两个关键挑战：

> 1. 对于文本到图像生成，将模型输出与复杂、多方面的提示对齐仍然是一个重大障碍。即使目前最好的模型，例&#x5982;**`GPT Image 1`**&#x548C;**`Seedream 3.0`**，<span style="color: rgb(216,57,49); background-color: inherit">在面对需要多行文本渲染、中文等非字母语言渲染、本地化文本插入或文本与视觉元素无缝集成的任务时，会遇到困难</span>
>
> 2. 对于图像编辑，实现编辑输出与原始图像之间的精确对齐面临双重挑战：
>
>    1. 视觉一致性，仅修改目标区域，同时保留所有其他视觉细节，例如<span style="color: rgb(220,155,4); background-color: inherit">在不改变面部细节的情况下更改头发颜色</span>
>
>    2. 语义一致性，在结构变化期间必须保留全局语义，例如<span style="color: rgb(220,155,4); background-color: inherit">在保持身份和场景一致性的同时修改人物的姿势</span>

Qwen-Image 旨在处理复杂的文本提示并进行精确的图像编辑，通过以下关键技术实现卓越的性能：

> * **<span style="color: rgb(36,91,219); background-color: inherit">数据处理流程</span>**：为了解决提示对齐的挑战，使用了一套先进的数据流程。通过课程学习策略，从简单的文本渲染任务逐步过渡到更复杂的段落和布局敏感的描述，增强了模型对多种语言，特别是中文的理解能力
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多任务学习框架</span>**：Qwen-Image 采用了一个创新的多任务学习框架来应对图像配准问题。<span style="color: rgb(100,37,208); background-color: inherit">将</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本到图像</span>`T2I`**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像到图像</span>`I2I`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本-图像到图像</span>`TI2I`**<span style="color: rgb(100,37,208); background-color: inherit">三种任务无缝整合</span>。通过<span style="color: rgb(100,37,208); background-color: inherit">将输入图像编码为</span>**<span style="color: rgb(100,37,208); background-color: inherit">语义特征</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">重构特征</span>**<span style="color: rgb(100,37,208); background-color: inherit">两种互补表示</span>，结&#x5408;**`MMDiT`**&#x67B6;构，实现了同时保持语义连贯性和视觉一致性的能力。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">高效大规模训练</span>**：为了确保训练的效率和稳定性，模型<span style="color: rgb(100,37,208); background-color: inherit">采用了一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">生产者-消费者框架</span>**<span style="color: rgb(100,37,208); background-color: inherit">，并利用</span>**`TensorPipe`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`Megatron`**<span style="color: rgb(100,37,208); background-color: inherit">进行分布式数据加载和模型训练</span>。这套系统还配备了全面的监控工具，以确保大规模训练过程中的可靠收敛和调试。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

Qwen-Image 架构基于三个核心组件构建，它们协同工作，实现高保真度的文本转图像生成：

1. **VLM** 是条件编码器，从文本输入中提取特征

2. **VAE** 是图像标记器，将输入图像压缩为紧凑的潜在表示，并在推理过程中将其解码回来

![](../../images/视觉多模态讲义（下）-image-155.png)

* **MMDiT** 充当主干扩散模型，在文本引导下对噪声和图像潜在特征之间的复杂联合分布进行建模

总参数量达到&#x4E86;**`27B`**（**`7B VLM`** + **`20B MMDiT`**），整体结构如下图：

![](../../images/视觉多模态讲义（下）-image-153.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">条件编码器：</span>`Qwen2.5-VL`**

负责理解用户的输入，包括纯文本提示和图文结合的复杂指令。选&#x7528;**`Qwen2.5-VL`**&#x4F5C;为文本和视觉特征提取器有三大优势：

> * 其语言和视觉空间已经预先对齐，天生适合图生文任务
>
> * 语言能力强大
>
> * 支持多模态输入，为图像编辑等高级功能奠定了基础

给定用户输入，例如提示和图像，采用 Qwen2.5-VL 来提取特征。为了更好地引导模型生成精细的潜在表征，同时兼顾不同任务中不同的输入模式，<span style="color: rgb(100,37,208); background-color: inherit">分别针对纯文本输入和文本与图像输入设计了不同的系统提示</span>。最后<span style="color: rgb(100,37,208); background-color: inherit">利用 Qwen2.5-VL 主干网络最后一层隐藏状态的潜在表征作为用户输入的表征</span>。

![](../../images/视觉多模态讲义（下）-image-152.png)

![](../../images/视觉多模态讲义（下）-image-151.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">图像分词器：</span>`VAE`**

将高清图像压缩成紧凑的潜在表示，并在生成过程结束时将其解码回高清图像。为了同时兼顾图像和未来视频应用，<span style="color: rgb(100,37,208); background-color: inherit">模型采用了一个独特的单编码器、双解码器的 VAE 架构，使用一个兼容图像和视频的共享编码器，以及针对每种模态的独立专用解码器</span>，这可以作为视频模型的骨干网络。具体而言，采用&#x4E86;**`Wan-2.1-VAE`**&#x7684;架构，冻结了其编码器，并专门微调了图像解码器。

> 现有的联合图像-视频 VAE，例如 Wan-2.1-VAE，通常会牺牲性能，导致图像重建能力下降

同时，为了提升对微小文字和细节的重建保真度，团队在一个包含大量富文本文档（PDF、PPT等）的自建数据集上专门微调了图像解码器，最终在文本图像重建质量上取得了 SOTA 表现

> 在训练过程中：
>
> * 平衡重建损失和感知损失可以有效减少网格伪影，这种伪影通常出现在灌木丛等重复纹理中
>
> * 随着重建质量的提高，对抗损失变得无效，因为判别器无法提供有效的指导
>
> 基于以上两点，Qwen-Image 仅使用重建损失和感知损失，并在微调过程中动态调整它们的比例。并且仅微调解码器可以有效增强细节并改善小文本的渲染效果。

* **<span style="color: rgb(36,91,219); background-color: inherit">骨干扩散模型：</span>`MMDiT`**

MMDiT 是一个拥&#x6709;**`60`**&#x5C42;、**`20B`**&#x53C2;数的庞大 Transformer 模型，负责在 VAE 的潜在空间中，根据 Qwen2.5-VL 提供的指引，从随机噪声中逐步生成目标图像。

Qwen-Image 设计了一种<span style="color: rgb(216,57,49); background-color: inherit">多模态可伸缩旋转位置编码 </span>**<span style="color: rgb(216,57,49); background-color: inherit">MSRoPE</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ultimodal </span>**<span style="color: rgb(216,57,49); background-color: inherit">S</span>**<span style="color: rgb(216,57,49); background-color: inherit">calable </span>**<span style="color: rgb(216,57,49); background-color: inherit">RoPE</span>**<span style="color: rgb(216,57,49); background-color: inherit">）</span>。文本输入被视为二维张量，并在两个维度上应用相同的位置 ID，文本被概念化为沿图像对角线连接。这种设计使 MSRoPE 能够利用图像端的分辨率缩放优势，同时在文本端保持与 1D-RoPE 的功能等效，从而无需确定文本的最佳位置编码。

> * 在传统的 MMDiT 块中，文本标记在扁平化图像位置嵌入后直接连接
>
> * Seedream 3.0 引入了 Scaling RoPE，其中图像位置编码移至图像的中心区域，并且文本标记被视为形状为 **`[1, L]`**&#x7684;二维 token，然后使用 2D RoPE 进行图像-文本联合位置编码。虽然这种调整有助于分辨率缩放训练，但文本和图像的某些位置编码行会变得同构，使得模型更难区分文本标记和第 0 行中间行中的图像潜在 token

![](../../images/视觉多模态讲义（下）-image-150.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

模型性能的飞跃离不开高质量的数据。Qwen-Image团队构建了一个包含数十亿图文对的庞大数据集，并实施了一套先进的数据处理流程。

**<span style="color: rgb(222,120,2); background-color: inherit">数据收集</span>**

![](../../images/视觉多模态讲义（下）-image-156.png)

数据集被分为四大领域：

* **<span style="color: rgb(36,91,219); background-color: inherit">自然</span>** (Nature)：占比&#x7EA6;**`55%`**，作为通用图像生成的基础

* **<span style="color: rgb(36,91,219); background-color: inherit">设计</span>** (Design)：占比&#x7EA6;**`27%`**，包含海报、UI、艺术作品等，用于提升艺术风格和复杂布局的遵循能力

* **<span style="color: rgb(36,91,219); background-color: inherit">人物</span>** (People)：占比&#x7EA6;**`13%`**，用于生成高质量、多样化的人物图像

* **<span style="color: rgb(36,91,219); background-color: inherit">合成数据</span>** (Synthetic Data)：占比&#x7EA6;**`5%`**，专门用于强化文本渲染能力

**<span style="color: rgb(222,120,2); background-color: inherit">数据过滤</span>**

![](../../images/视觉多模态讲义（下）-image-179.png)

包含7个阶段的渐进式过滤流程，在模型训练的不同时期，对数据进行越来越严格的筛选，确保数据质量和分布的持续优化。

* **<span style="color: rgb(36,91,219); background-color: inherit">初期 S1-S2</span>**：进行基础清洗，如去除损坏文件、低分辨率图像和重复数据

* **<span style="color: rgb(36,91,219); background-color: inherit">中期 S3-S4</span>**：提升图文对齐性，并引入合成的文本渲染数据，专门攻克文本生成难题

* **<span style="color: rgb(36,91,219); background-color: inherit">后期 S5-S7</span>**：聚焦高分辨率和美学质量，进行类别平衡和多尺度训练，确保模型在细节和泛化性上的卓越表现

**<span style="color: rgb(222,120,2); background-color: inherit">数据标注</span>**

![](../../images/视觉多模态讲义（下）-image-178.png)

利用 Qwen2.5-VL 来同时执行两项任务：

1. **<span style="color: rgb(36,91,219); background-color: inherit">生成详细的图像描述</span>**：字幕模型能够捕捉图像中的关键细节，包括<span style="color: rgb(100,37,208); background-color: inherit">对象属性、空间关系、上下文以及可见文本</span>

2. **<span style="color: rgb(36,91,219); background-color: inherit">提取结构化元数据</span>**：同时，模型以 JSON 等结构化格式输出关键信息，如<span style="color: rgb(220,155,4); background-color: inherit">图像类型、风格、是否存在水印或二维码等异常元素</span>

这种方法将字幕生成和元数据提取合并为一个流程，大大提高了效率，并能够大规模处理数据集。此外还结合了专家规则和轻量级分类模型，以进一步完善水印验证和内容过滤等关键任务。

**<span style="color: rgb(222,120,2); background-color: inherit">数据合成</span>**

![](../../images/视觉多模态讲义（下）-image-176.png)

针对自然图像中文本稀疏且不均衡的问题，采用三种策略来“凭空创造”高质量的文本渲染训练数据（如图13所示）：

1. **<span style="color: rgb(36,91,219); background-color: inherit">纯净背景渲染</span>**：在纯色背景上渲染文本，让模型首先学会认识字符本身

2. **<span style="color: rgb(36,91,219); background-color: inherit">情景组合渲染</span>**：将合成文本无缝嵌入到真实场景图片中，如纸上、木板上，并用 Qwen-VL 生成描述性标题，教会模型理解在情景中的文字

3. **<span style="color: rgb(36,91,219); background-color: inherit">结构化模板渲染</span>**：基于PPT、UI等预定义模板，程序化地填充内容，训练模型理解并执行复杂的布局指令

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

采用流匹配 flow matching 训练目标来预训练 Qwen-Image，它通过常微分方程 ODE 稳定学习过程，同时保持与最大似然目标的等价性。形式上，设$$x_0$$是输入图像的潜在向量 latent。潜在表示$$z$$是通过 VAE 编码器$$\mathcal{E}$$对$$x$$进行编码得到的，即$$z=\mathcal{E}(x)$$，其中$$\mathcal{E}:x↦z$$。接下来，从标准多元正态分布中采样一个随机噪声向量$$x_1$$，即 $$x_1∼N(0,I)$$。对于用户输入$$S$$，可能包含文本或提示词以及图像，引导潜在向量$$h$$通过 VLM $$ϕ$$获得，即 $$h=ϕ(S)$$，其中$$ϕ:S↦h$$。此外，扩散时间步长$$t$$从对数正态分布中采样，其中$$t∈[0,1]$$。根据 Rectified Flow 的思想，时间步长$$t$$处的中间潜在变量$$x_t$$及其对应的速度$$v_t$$可以计算为：

$$\begin{cases}
x_t = tx_0 + (1-t)x_1 \\
v_t = \frac{dx_t}{dt} = x_0 - x_1
\end{cases}$$

然后，训练模型来预测目标速度，损失函数定义为模型预测输出$$f_θ(x_t,t)$$与真实速度$$v_t$$之间的均方误差：

$$L=\mathbb{E}_{(x_0,h)∼\mathcal{D},x_1,t}∥v_θ(x_t,t,h)−v_t∥^2$$

其中，$$v_θ(x_t,t,h)$$是模型预测的速度，$$\mathcal{D}$$表示训练数据集。

**<span style="color: rgb(222,120,2); background-color: inherit">生产者-消费者框架</span>**

为了在扩展到大规模 GPU 集群时<span style="color: rgb(100,37,208); background-color: inherit">确保高吞吐量和训练稳定性，采用了一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">生产者-消费者框架</span>**<span style="color: rgb(100,37,208); background-color: inherit">，将数据预处理与模型训练解耦</span>。这种设计使两个阶段能够异步且以最佳效率运行，同时还支持在不中断正在进行的训练过程的情况下对数据流水线进行即时更新。

> * **<span style="color: rgb(36,91,219); background-color: inherit">生产者</span>**：原始图像-标题对<span style="color: rgb(100,37,208); background-color: inherit">首先根据预定义的标准进行过滤</span>，如图像分辨率和检测操作符。然后，<span style="color: rgb(100,37,208); background-color: inherit">使用 Qwen2.5 VL 和 VAE 将筛选出的数据编码为潜在表示</span>。处理后的图像按分辨率分组，放入快速访问缓存桶中，并<span style="color: rgb(100,37,208); background-color: inherit">存储在一个共享的、位置感知的存储区中，这使得消费者无需排队等待即可立即获取数据</span>。生产者和消费者之间的连接通过一个特定的 HTTP 传输层实现，原生支持异步、零拷贝调度所需的 RPC 语义。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">消费者</span>**：部署在 GPU 密集型集群上，专门用于模型训练。通过将所有数据处理卸载给生产者，消费者节点可以将其全部计算预算用于训练 MMDiT 模型。MMDiT 参数以 4 路张量并行布局分布在这些节点上，每个数据并行组直接从生产者异步拉取预处理的批次数据。

**<span style="color: rgb(222,120,2); background-color: inherit">分布式训练优化</span>**

由于 Qwen-Image 庞大的参数量，仅使&#x7528;**`FSDP`**&#x4E0D;足以让模型适应每个 GPU。因此利&#x7528;**`Megatron-LM`**&#x8FDB;行训练

* **<span style="color: rgb(36,91,219); background-color: inherit">混合并行策略</span>**：结合了数据并行和张量并行，以高效地扩展跨大型 GPU 集群的训练。为了实现张量并行，<span style="color: rgb(46,161,33); background-color: inherit">使用 Transformer-Engine 库构建了 MMDiT 模型，允许在不同程度的张量并行之间进行无缝和自动切换</span>。对于多头自注意力块，<span style="color: rgb(46,161,33); background-color: inherit">采用</span>**<span style="color: rgb(46,161,33); background-color: inherit">头级并行</span>**<span style="color: rgb(46,161,33); background-color: inherit">（head-wise parallelism），以减少与沿头部维度的张量并行相比的同步和通信开销</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">分布式优化器和激活检查点</span>**：通过对 256 多分辨率图像训练设置的经验比较，作者发现启用激活检查点可以将每个 GPU 的内存消耗减&#x5C11;**`11.3%`**：&#x4ECE;**`71GB`**&#x964D;&#x81F3;**`63GB`**，但代价是将每次迭代时间增加了 **`3.75`** 倍：&#x4ECE;**`2`**&#x79D2;增加&#x5230;**`7.5`**&#x79D2;。基于这种权衡，<span style="color: rgb(100,37,208); background-color: inherit">禁用激活检查点，仅依赖分布式优化器</span>。在训练期间，<span style="color: rgb(46,161,33); background-color: inherit">所有</span>**`all-gather`**<span style="color: rgb(46,161,33); background-color: inherit">操作都以</span>**`bfloat16`**<span style="color: rgb(46,161,33); background-color: inherit">格式执行</span>**<span style="color: rgb(46,161,33); background-color: inherit">，</span>`gradient reduce-scatter`**<span style="color: rgb(46,161,33); background-color: inherit">使用</span>**`float32`**<span style="color: rgb(46,161,33); background-color: inherit">，这既确保了计算效率又提高了数值稳定性</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">训练策略</span>**

采用<span style="color: rgb(100,37,208); background-color: inherit">多阶段预训练策略</span>，旨在逐步提高数据质量、图像分辨率和模型性能：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">提升分辨率</span>**：逐步提升多分辨率、多长宽比输入的尺寸，从初始&#x7684;**`256×256`**&#x50CF;素，包&#x542B;**`1:1`**、**`2:3`**、**`3:2`**、**`3:4`**、**`4:3`**、**`9:16`**、**`16:9`**、**`1:3`**&#x548C;**`3:1`**&#x591A;种长宽比，然后增加&#x5230;**`640×640`**&#x50CF;素，最后达&#x5230;**`1328×1328`**&#x50CF;素。<span style="color: rgb(46,161,33); background-color: inherit">通过提升图像分辨率，模型可以捕捉更精细的特征，从而获得更好的性能。更丰富的特征空间有助于提高对未见数据的泛化能力</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">从低分辨率到高分辨率的花卉图像过渡，使模型能够辨别出花瓣纹理和颜色渐变等更精细的细节</span>。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">融合文本渲染</span>**：解决传统视觉数据集中文本内容有限以及由此导致的字形生成性能不佳的问题，特别是对于中文字符，<span style="color: rgb(100,37,208); background-color: inherit">逐步引入包含叠加在自然背景上的渲染文本的图像</span>。这使模型能够首先学习通用的视觉表示，然后逐步获得文本渲染能力。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">细化数据质量</span>**：在预训练早期，利用大规模数据集使模型获得基本的视觉生成能力。随着训练的进行，<span style="color: rgb(100,37,208); background-color: inherit">逐步采用越来越严格的数据过滤机制来选择更高质量的数据</span>。这种渐进式的数据细化确保<span style="color: rgb(46,161,33); background-color: inherit">仅利用最相关和最高质量的样本，以保证训练效率和模型性能</span>。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">平衡数据分布</span>**：在整个训练过程中，<span style="color: rgb(100,37,208); background-color: inherit">逐步平衡数据集中域和图像分辨率的分布</span>。这种调整降低了模型对特定域或分辨率过拟合的风险，否则这可能会损害在代表性不足的设置中生成图像的逼真度和精细细节。<span style="color: rgb(46,161,33); background-color: inherit">通过保持更均匀的数据分布，促进了模型在不同域和分辨率上的稳健泛化</span>。
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">通过合成数据增强</span>**：某些数据在真实世界数据集中代表性不足甚至缺失，如<span style="color: rgb(220,155,4); background-color: inherit">超现实主义风格或包含大量文本内容的高分辨率图像</span>。此外，一些高质量数据样本的可用性本身就有限。为了弥补这些差距，<span style="color: rgb(100,37,208); background-color: inherit">采用数据合成技术来生成补充样本，从而丰富数据集并确保更全面地覆盖不同的视觉域</span>。这种增强策略<span style="color: rgb(46,161,33); background-color: inherit">提升了模型在更广泛场景中进行泛化和稳健表现的能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">有监督微调 SFT</span>**

构建了一个分层组织的语义类别数据集，并采用细致的人工标注来解决模型的特定缺陷。这里要求筛选出的图像清晰、细节丰富、明亮且具有逼真感。这种方法可以引导模型生成更具真实感和更精细细节的内容。

**<span style="color: rgb(222,120,2); background-color: inherit">强化学习 RL</span>**

采用了两种不同的 **RL** 策略：

> **<span style="color: rgb(36,91,219); background-color: inherit">直接偏好优化 DPO</span>**：DPO 擅长流匹配（一步）在线偏好建模，并且计算效率高
>
> **<span style="color: rgb(36,91,219); background-color: inherit">组相对策略优化 GRPO</span>**：在训练期间执行 on-policy 采样并使用奖励模型评估每个轨迹

为了利用离线偏好学习的可扩展性优势，<span style="color: rgb(100,37,208); background-color: inherit">使用 DPO 进行相对大规模的 RL，并保留 GRPO 用于小规模的精细化 RL 改进</span>

**<span style="color: rgb(36,91,219); background-color: inherit">直接偏好优化 DPO</span>**：给定相同的 prompt，会使用不同的随机初始化种子生成多张图像。然后，<span style="color: rgb(100,37,208); background-color: inherit">人工标注员从这些候选中选择</span>**<span style="color: rgb(100,37,208); background-color: inherit">最佳</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">最差</span>**<span style="color: rgb(100,37,208); background-color: inherit">的图像</span>。数据分为两类：

> **<span style="color: rgb(36,91,219); background-color: inherit">有参考图像的 prompt</span>**：标注员首先将生成的输出与参考图像进行比较。如果存在显著差异，标注员会将最差的生成结果指定为被拒绝的样本。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">没有参考图像的 prompt</span>**：标注员需要从生成的图像中选择最佳和最差的样本，或者指出所有生成结果都不令人满意。

给定文本隐藏状态$$h$$、被选中的生成图像$$x_0^{win}$$和被拒绝的生成图像$$x_0^{lose}$$，采样时间步长$$t∼(0,1)$$，以此构建输入潜在变量$$x_t^{win}$$和$$x_t^{lose}$$及其对应的速度$$v_t^{win}$$和$$v_t^{lose}$$。然后基于流匹配训练准则构建 DPO 目标：

$$\begin{aligned}
\text{Diff}_{\text{policy}} &= \left( \left\| v_{\theta}(x_t^{\text{win}}, h, t) - v_t^{\text{win}} \right\|_2^2 - \left\| v_{\theta}(x_t^{\text{lose}}, h, t) - v_t^{\text{lose}} \right\|_2^2 \right) \\
\text{Diff}_{\text{ref}} &= \left( \left\| v_{\text{ref}}(x_t^{\text{win}}, h, t) - v_t^{\text{win}} \right\|_2^2 - \left\| v_{\text{ref}}(x_t^{\text{lose}}, h, t) - v_t^{\text{lose}} \right\|_2^2 \right) \\
\mathcal{L}_{\text{DPO}} &= -\mathbb{E}_{h, (x_0^{\text{win}}, x_0^{\text{lose}}) \sim \mathcal{D}, t \sim \mathcal{U}(0, 1)} \left[ \log \sigma \left( -\beta \left( \text{Diff}_{\text{policy}} - \text{Diff}_{\text{ref}} \right) \right) \right]
\end{aligned}$$

其中，$$\text{Diff}_\text{policy}$$和$$\text{Diff}_\text{ref}$$分别表示由策略模型和参考模型计算的偏好差异，$$β$$是一个缩放参数，$$σ(⋅)$$表示 sigmoid 函数。

**<span style="color: rgb(36,91,219); background-color: inherit">组相对策略优化 GRPO</span>**：在使用 DPO 训练之后，按&#x7167;**`Flow-GRPO`**&#x6846;架进行进一步的精细化训练。给定文本隐藏状态$$h$$，流模型预测一组$$G$$张图像$$\{x_0^i\}_{i=1}^G$$和对应的轨迹$$\{x_T^i,x_{T−1}^i,\cdots,x_0^i\}_{i=1}^G$$。在每个组内，优势函数可以表述为：

$$A_i = \frac{R(x_0^i, h) - \text{mean}\left(\{R(x_0^j, h)\}_{j=1}^G\right)}{\text{std}\left(\{R(x_0^j, h)\}_{j=1}^G\right)}$$

其中$$R$$是奖励模型。然后，GRPO 的训练目标为：

$$\mathcal{L}_{\text{GRPO}}(\theta) = \mathbb{E}_{h \sim \mathcal{D}_h, \{x_0^i, \dots, x_T^i\}_{i=1}^G \sim \pi_{\theta}}$$

$$\frac{1}{G} \sum_{i=1}^G \frac{1}{T} \sum_{t=0}^{T-1} \left( \min(r_t^i(\theta) A_i, \text{clip}(r_t^i(\theta), 1-\epsilon, 1+\epsilon) A_i) - \beta D_{\text{KL}}(\pi_{\theta} \| \pi_{\text{ref}}) \right)$$

其中 $$r_t^i(\theta) = \frac{p_{\theta}(x_t^i | x_{t-1}^i, h)}{p_{\theta_{\text{old}}}(x_t^i | x_{t-1}^i, h)}$$

当采样轨迹$$\{x_T^i,\cdots,x_0^i\}_{i=1}^G∼π_θ$$时，有$$dx_t=v_td_t$$用于流匹配采样，其中$$v_t=v_θ(x_t,t,h)$$是预测的速度。然而这种采样策略没有随机性，不适合探索。因此将采样过程重新表述为随机微分方程过程，以增加随机性。SDE 采样过程可以写为：

$$\mathrm{d}x_t = \left( v_t + \frac{\sigma_t^2}{2t} \left( x_t + (1-t)v_t \right) \right) \mathrm{d}t + \sigma_t \, \mathrm{d}w$$

使用 Euler-Maruyama 离散化，得到：

$$x_{t+\Delta t} = x_t + \left[ v_{\theta}(x_t, t, h) + \frac{\sigma_t^2}{2t} \left( x_t + (1-t)v_{\theta}(x_t, t, h) \right) \right] \Delta t + \sigma_t \sqrt{\Delta t} \epsilon$$

使用上述方程来采样轨迹。公式 (5) 中的 KL 散度可以闭合形式求解：

$$D_{\text{KL}}(\pi_{\theta} \| \pi_{\text{ref}}) = \frac{\Delta t}{2} \left( \frac{\sigma_t (1-t)}{2t} + \frac{1}{\sigma_t} \right)^2 \| v_{\theta}(x_t, t, h) - v_{\text{ref}}(x_t, t, h) \|^2$$

**<span style="color: rgb(222,120,2); background-color: inherit">多任务训练</span>**

除了文本到图像 T2I 生成，<span style="color: rgb(100,37,208); background-color: inherit">Qwen-Image 还包含文本和图像输入的多模态图像生成任务，包括基于指令的图像编辑、新视角合成和计算机视觉任务，如深度估计</span>。可以将这些任务广义地视为通用图像编辑任务。

Qwen2.5-VL 原生支持图像输入：从用户提供的图像中提取的视觉块通过 ViT 进行编码，并与文本 token 拼接在一起，形成输入序列。然后提取输入图像和文本指令作为 Qwen-Image MMDiT 文本流的输入：

![](../../images/视觉多模态讲义（下）-image-175.png)

为了使模型能够区分多张图像，<span style="color: rgb(100,37,208); background-color: inherit">除了用于定位单个图像内图像块的高度和宽度之外，还引入一个额外的帧维度来扩展 MSRoPE</span>。作者发现<span style="color: rgb(46,161,33); background-color: inherit">提供来自 VLM 的视觉语义嵌入能够更好地遵循指令，而引入像素级 VAE 嵌入则进一步增强了模型保持视觉保真度和与用户提供图像结构一致性的能力</span>。

### 5.2.6 <span style="color: rgb(36,91,219); background-color: inherit">SEED（b-3）</span>

![](../../images/视觉多模态讲义（下）-image-173.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉分词</span>**

视觉分词主要是将图像表示为一系列离散的 token，目前的一些工作比&#x5982;**`VQ-VAE`**<span style="color: rgb(220,155,4); background-color: inherit">通过重建图像像素来训练向量量化 </span>**<span style="color: rgb(220,155,4); background-color: inherit">VAE；</span>`BEiT v2`**<span style="color: rgb(220,155,4); background-color: inherit">提出了向量量化知识蒸馏 </span>**<span style="color: rgb(220,155,4); background-color: inherit">VQ-KD</span>**<span style="color: rgb(220,155,4); background-color: inherit">（</span>**<span style="color: rgb(220,155,4); background-color: inherit">V</span>**<span style="color: rgb(220,155,4); background-color: inherit">ector-</span>**<span style="color: rgb(220,155,4); background-color: inherit">Q</span>**<span style="color: rgb(220,155,4); background-color: inherit">uantized </span>**<span style="color: rgb(220,155,4); background-color: inherit">K</span>**<span style="color: rgb(220,155,4); background-color: inherit">nowledge </span>**<span style="color: rgb(220,155,4); background-color: inherit">D</span>**<span style="color: rgb(220,155,4); background-color: inherit">istillation），通过重建教师模型的高层特征来训练视觉分词器</span>。

作者提出<span style="color: rgb(100,37,208); background-color: inherit">基于向量量化的图像分词器 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SEED</span>**<span style="color: rgb(100,37,208); background-color: inherit">，用于生成具有</span>**<span style="color: rgb(100,37,208); background-color: inherit">一维因果依赖性</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">高层语义</span>**<span style="color: rgb(100,37,208); background-color: inherit">的离散视觉代码</span>。其由以下模块组成：**<span style="color: rgb(100,37,208); background-color: inherit">ViT Encoder</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">Causal Q-Former</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">VQ Codebook</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">Reverse Q-Former </span>**<span style="color: rgb(100,37,208); background-color: inherit">以及 </span>**<span style="color: rgb(100,37,208); background-color: inherit">UNet Decoder</span>**。其中，ViT Encoder 和 UNet Decoder 分别直接来自预训练&#x7684;**`BLIP-2`**&#x548C;**`Stable Diffusion`**。

![](../../images/视觉多模态讲义（下）-image-177.png)

总的来说，训练主要有以下 3 步：

> 1. 训练一个 **<span style="color: rgb(100,37,208); background-color: inherit">Causal Q-Former</span>**，将 ViT Encoder 输出的$$16 \times 16$$ 个 token 二维光栅顺序特征转换为一串具有因果依赖性的 32 个语义 Embedding
>
> 2. 训练一个<span style="color: rgb(100,37,208); background-color: inherit">视觉 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Codebook</span>**，将这些 Casual Embedding 离散化为具有因果依赖的$$32$$个量化视觉代码
>
> 3. 使用 **<span style="color: rgb(100,37,208); background-color: inherit">Reverse Q-Former</span>** 将这些视觉代码解码为$$77$$个生成 Embedding，使其与预训练 Stable Diffusion 模型的潜在空间对齐。

**<span style="color: rgb(36,91,219); background-color: inherit">训练阶段 1：Casual Q-Former</span>**

首先<span style="color: rgb(100,37,208); background-color: inherit">一组</span>$$32$$<span style="color: rgb(100,37,208); background-color: inherit">个可学习的 Query Embedding 与预训练 ViT 图像 Encoder 的特征被输入到 Casual Q-Former 中，用来编码输入图像的固定的</span>$$32$$<span style="color: rgb(100,37,208); background-color: inherit">个 Casual Embedding</span>。Query Embedding 仅能通过带 causal mask 的自注意力层与之前的 Query 交互，并通过交叉注意力层与冻结的图像特征交互。

这里采用对比学习策略，在包含 CC3M、Unsplash 和 COCO &#x7684;**`5M`**&#x56FE;像-文本对上，对 BLIP-2 中的 Q-Former 进行优化。**<span style="color: rgb(100,37,208); background-color: inherit">最大化最终 Casual Embedding 与对应图像描述文本特征之间的相似度，同时最小化其与其他批次中非匹配描述文本特征的相似度</span>**。最终 Casual Q-Former 在综合指标 Recall@mean 上优于 BLIP-2，表明<span style="color: rgb(46,161,33); background-color: inherit">具有因果依赖的输出 Query Embedding 在性能上不比 BLIP-2 中使用双向注意力机制的输出 Embedding 差</span>。

![](../../images/视觉多模态讲义（下）-image-174.png)

**<span style="color: rgb(36,91,219); background-color: inherit">训练阶段 2：De-tokenization</span>**

在包含 CC3M、Unsplash 和 COCO &#x7684;**`5M`**&#x56FE;像-文本对上训练一个 VQ Codebook，将$$32$$个 Casual Embedding 离散化为$$32$$个量化视觉代码。<span style="color: rgb(100,37,208); background-color: inherit">Quantizer 为每个 Casual Embedding 在 Codebook 中查找最近邻并获取对应代码</span>。这里采用一个多层 Transformer 作为解码器，从离散代码中重建连续的 Casual Embedding。训练过程中，**<span style="color: rgb(100,37,208); background-color: inherit">最大化解码器输出与原始 Casual Embedding 之间的余弦相似度</span>**。

然后引入一个 Reverse Q-Former，从离散代码中重建冻结的 Stable Diffusion 模型的文本特征。一组$$77$$个可学习的 Query Embedding 被输入到 Reverse Q-Former 中。这些 <span style="color: rgb(100,37,208); background-color: inherit">Query Embedding 通过自注意力层相互交互，并通过交叉注意力层与</span>$$32$$<span style="color: rgb(100,37,208); background-color: inherit">个 Casual code 交互，最终输出</span>$$77$$<span style="color: rgb(100,37,208); background-color: inherit">个生成 Embedding</span>。训练&#x65F6;**<span style="color: rgb(100,37,208); background-color: inherit">最小化生成 Embedding 与 SD 文本特征之间的均方误差 MSE 损失</span>**。推理时，这些生成 Embedding 可直接输入 UNet 以解码出逼真的图像。如上表，离散的 SEED token 表现出与 BLIP-2 相当的效果。

如对于图像生成，按照 GILL 的做法，使用 CLIP 相似度作为评估语义一致性的指标，<span style="color: rgb(46,161,33); background-color: inherit">与理论上限 SD 相比，SEED 的性能仅有轻微下降，且在图像生成任务上优于 GILL</span>。

![](../../images/视觉多模态讲义（下）-image-170.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态自回归建模</span>**

基于预训练的 SEED 分词器，通过在 OPT-2.7B 模型上 LoRA 微调，并使用包含 CC3M、Unsplash 和 COCO 的 5M 图像-文本对，构建了 SEED-OPT2.7B 模型。如图 4 所示，我们执行图像到文本和文本到图像的自回归预训练，以实现统一的多模态理解与生成。

![](../../images/视觉多模态讲义（下）-image-169.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像到文本自回归</span>**

这个阶段主要是对齐预训练 VQ Codebook 的词汇表与 OPT-2.7B，使用一个<span style="color: rgb(100,37,208); background-color: inherit">全连接层将视觉分词器输出的 Casual code 线性投影到与 OPT-2.7B Word Embedding 相同的维度</span>。投影后的 Casual code 与前缀`A photo of`的 Word Embedding 拼接后作为 OPT-2.7B 的输入，对应图像描述的文本 token 作为生成目标。训练&#x65F6;**<span style="color: rgb(100,37,208); background-color: inherit">冻结 OPT-2.7B，进行 LoRA 微调，目标是预测下一个文本 token</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">文本到图像自回归</span>**

联合执行图像到文本和文本到图像的自回归训练，使 LLM 不仅能生成文本 token，还能生成视觉 token。在文本到图像的自回归预训练中，前缀`Generate an image`与图像描述的 Word Embedding 被输入 OPT-2.7B，而来自预训练分词器对应图像的视觉 code 作为生成目标。同样，**<span style="color: rgb(100,37,208); background-color: inherit">OPT-2.7B 被冻结，进行 LoRA 微调，目标是预测下一个视觉 token</span>**。推理时，给定 prompt `Generate an image`和一段文本描述，SEED-OPT2.7B 自回归地预测视觉 token。输出的视觉 token 被送入 Reverse Q-Former 以获得生成 Embedding，再通过 UNet 解码生成逼真图像。

* **<span style="color: rgb(36,91,219); background-color: inherit">性能</span>**

![](../../images/视觉多模态讲义（下）-image-171.png)

BLIP-2 &#x5728;**`129M`**&#x56FE;像-文本对上训练，SEED-OPT2.7B &#x5728;**`5M`**&#x56FE;像-文本对上训练，<span style="color: rgb(46,161,33); background-color: inherit">借助 SEED 离散视觉 token，取得了有竞争力的零样本图像描述和视觉问答结果</span>。这里 SEED-OPT2.7B 仅通过前缀`A photo of`”的图像到文本自回归预训练，即可理解自由形式的问题并预测开放式答案，从而实现 Zero-Shot 视觉问答。

SEED 能有效促进视觉 token 与大语言模型之间的对齐。SEED-OPT2.7B 在经过 LoRA 微调后，已<span style="color: rgb(46,161,33); background-color: inherit">具备执行文本到图像和图像到文本生成任务的能力</span>。

### 5.2.7 <span style="color: rgb(36,91,219); background-color: inherit">MiniGPT-5（b-3）</span>

近年来，跨模态生成模型成为研究热点。在这一背景下，**MiniGPT-5** 产生，多模态生成领域迈入了一个全新的阶段。它<span style="color: rgb(46,161,33); background-color: inherit">不仅能够生成高质量的图像和文本，还能在两者之间实现无缝切换，为用户提供了更加自然、直观的交互体验</span>。

**MiniGPT-5** 的设计灵感源自对人类视觉与语言处理机制的深刻理解。<span style="color: rgb(100,37,208); background-color: inherit">通过结合大规模语言模型 LLM 与扩散模型 Diffusion Model，该模型实现了从单一模态到多模态生成的跨越式发展</span>。

![MiniGPT-5：交错 视觉-语言 理解和生成的统一模型](../../images/视觉多模态讲义（下）-image-172.png)

为了赋予大型语言模型多模态生成能力，**<span style="color: rgb(100,37,208); background-color: inherit">MiniGPT-5 </span>**<span style="color: rgb(100,37,208); background-color: inherit">引入了一种新框架，该框架整合了预训练的多模态大语言模型和文本到图像生成模型。核心是引入</span>**<span style="color: rgb(216,57,49); background-color: inherit">生成视觉标记 Generative Vokens</span>**，这些特殊的视觉标记在训练过程中有效地连接了文本和视觉领域。此外，作者采用两阶段训练方法结&#x5408;**<span style="color: rgb(216,57,49); background-color: inherit">无分类器引导 Classifier-Free Guidance</span>** 策略来提升生成输出的质量和连贯性。下图展示了模型结构概览。**MiniGPT-5** 主要由两个模块组成：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉语言编码模块</span>**，利用预训练的多模态大语言模&#x578B;**`MiniGPT-4`**&#x5904;理多模态输入
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">多模态生成模块</span>**，使&#x7528;**`Stable Diffusion`**&#x751F;成视觉输出

![](../../images/视觉多模态讲义（下）-image-168.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解模块</span>**

近年来，多模态大语言模型的进步主要集中于多模态理解，&#x5982;**`MiniGPT-4`**，使模型能够将图像作为序列化输入进行处理。<span style="color: rgb(100,37,208); background-color: inherit">视觉-语言编码模块可以扩展大语言模型的能力，从单纯的多模态理解转向多模态上下文中的主动生成</span>。生成视觉标记 Generative **Voken**s 在该模块中起着关键作用，它将原始视觉输入转换为大语言模型可以处理并用于后续生成任务的格式。

1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态编码</span>**

每个文本 Token 被嵌入为向量$$e_{\text{text}} \in \mathbb{R}^d$$，而预训练的视觉编码器将每个输入图像转换为特征$$e_{\text{img}} \in \mathbb{R}^{32 \times d}$$。这些 Embedding 被拼接以创建输入提示特征。

* **<span style="color: rgb(36,91,219); background-color: inherit">生成视觉标记 Generative Vokens</span>**

由于原始大语言模型的词汇表$$  V  $$仅包含文本 Token，因此需要在大语言模型和生成模型之间构建桥梁：在大语言模型的词汇表$$  V  $$中引入一组特殊 Token $$V_{\text{img}} = \{\text{[IMG1]}, \text{[IMG2]}, \dots, \text{[IMGn]}\}$$(默认$$n = 8$$)作为生成视觉标记 Generative Vokens。这些 Voken 的输出隐藏状态被用于后续的图像生成，其位置可以表示插入交错图像的位置。在固定 **MiniGPT-4** 的所有预训练权重$$  \theta_{\text{pretrained}}  $$的情况下，可训练参数包括额外的输入 Embedding $$  \theta_{\text{voken\_input}}  $$和输出 Embedding $$\theta_{\text{voken\_output}}$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">参数高效微调 PEFT</span>**

PEFT 在训练 LLM 中至关重要，它被用来使大语言模型适应下游任务而无需大规模重新训练。在 PEFT 时，不是更新模型的所有参数，而是<span style="color: rgb(100,37,208); background-color: inherit">仅训练一小部分参数。这部分通常包括特定任务组件或添加到原始模型架构中的轻量级层</span>。**MiniGPT-5** 将 PEFT 应用&#x4E8E;**`MiniGPT-4`**&#x7F16;码器，增强其基于给定指令或提示处理和生成多模态内容的能力。具体来说，是在整个语言编码&#x5668;**`Vicuna`**&#x4E0A;应&#x7528;**`Prefix Tuning`**&#x548C;**`LoRA`**。此外，<span style="color: rgb(100,37,208); background-color: inherit">在 Transformer 解码器的输入处实现可学习 Query</span>，这是一种常见的序列到序列 Transformer 架构设置，以进一步提高模型的多模态生成能力。<span style="color: rgb(100,37,208); background-color: inherit">解码器中的可学习 Query 使模型能够拥有动态、自适应的表示，从而启动生成过程</span>。当模型需要基于视觉和文本混合输入生成输出时，这种方法尤其有用。<span style="color: rgb(46,161,33); background-color: inherit">结合指令调优，它显著提升了模型在各种数据集上的多模态生成性能</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态生成模块</span>**

为了将生成视觉标记 Generative Vokens 与文本到图像生成模型对齐，**MiniGPT-5** 设计了一个映射模块用于维度匹配，并引入了 voken 定位损失和 voken 对齐损失：

> **<span style="color: rgb(36,91,219); background-color: inherit">voken 定位损失</span>**&#x5E2E;助模型学习 token 的正确位置
>
> **<span style="color: rgb(36,91,219); background-color: inherit">voken 对齐损失</span>**&#x5219;直接将 voken 与扩散模型的适当条件生成特征对齐

由于 <span style="color: rgb(100,37,208); background-color: inherit">Generative Vokens 的特征梯度可以直接从图像中计算得出，因此不需要对图像进行详细描述，从而实现了无描述学习</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">Voken 定位</span>**

首先通过自回归语言模型中的 Next Token Prediction 在文本空间中联合生成文本和 voken。在训练过程中，将 voken $$  V_{\text{img}}  $$ 添加到真实图像的位置，并训练模型在文本生成过程中预测这些 voken。具体来说，生成的 token 表示为：

$$W = \{w_1, w_2, \dots, w_m\}$$

其中$$w_i \in V \cup V_{\text{img}}$$。因果语言建模损失定义为：

$$L_{\text{text}} := -\sum_{i=1}^m \log p(w_i | e_{\text{text}}, e_{\text{img}}, w_1, \dots, w_{i-1}; \theta_{\text{pretrained}}, \theta_{\text{voken\_input}}, \theta_{\text{voken\_output}})$$

其中$$w_i \in V \cup V_{\text{img}}$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">用于图像生成的 Voken 对齐</span>**

接下来，将输出隐藏状态$$h_{\text{voken}}$$与文本到图像生成模型的条件特征空间对齐。为了将 voken 特征$$  h_{\text{voken}}  $$映射到可行的图像生成条件特征$$e_{\text{text\_encoder}} \in \mathbb{R}^{L \times \hat{d}}$$，其中$$  L  $$是文本到图像生成模型中文本编码器的最大输入长度，$$\hat{d}$$是该模型中编码器输出特征的维度，**<span style="color: rgb(100,37,208); background-color: inherit">MiniGPT-5</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 构建了一个特征映射模块，包括一个两层多层感知机模型</span>$$\theta_{\text{MLP}}$$<span style="color: rgb(100,37,208); background-color: inherit">、一个四层编码器-解码器 Transformer 模型</span>$$  \theta_{\text{enc-dec}}  $$<span style="color: rgb(100,37,208); background-color: inherit">和一个可学习的解码器特征序列</span>$$q$$。映射特征$$  \hat{h}_{\text{voken}}  $$定义为：

$$\hat{h}_{\text{voken}} := \theta_{\text{enc-dec}}(\theta_{\text{MLP}}(h_{\text{voken}}), q) \in \mathbb{R}^{L \times \hat{d}}$$

为了生成适当的图像，映射特征$$  \hat{h}_{\text{voken}}  $$被用作去噪过程中的条件输入。$$\hat{h}_{\text{voken}}$$应表示与扩散模型生成真实图像对应的条件特征。**<span style="color: rgb(100,37,208); background-color: inherit">MiniGPT-5</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 采用潜在扩散模型</span>**`LDM`**<span style="color: rgb(100,37,208); background-color: inherit">损失作为 voken 对齐损失来训练图像生成模块</span>。在训练过程中，<span style="color: rgb(100,37,208); background-color: inherit">真实图像首先通过预训练的变分自编码器</span>**`VAE`**<span style="color: rgb(100,37,208); background-color: inherit">转换为潜在特征</span>$$z_0$$<span style="color: rgb(100,37,208); background-color: inherit">。然后，通过对</span>$$  z_0  $$<span style="color: rgb(100,37,208); background-color: inherit">添加噪声</span>$$  \epsilon  $$<span style="color: rgb(100,37,208); background-color: inherit">来获得带噪声的潜在特征</span>$$z_t$$。使用预训练&#x7684;**`U-Net`**&#x6A21;型$$  \epsilon_\theta  $$计算条件 **LDM** 损失为：

$$L_{\text{LDM}} := \mathbb{E}_{\epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, \hat{h}_{\text{voken}})\|_2^2 \right]$$

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Voken 定位损失</span>**&#x4F7F;模型能够学习 Token 的精确定位。如果没有这一组件，模型将缺乏在推理过程中预测何时生成 Voken 的基本能力。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">Voken 对齐损失</span>**&#x786E;保了 Voken 与扩散模型的适当条件生成特征之间的直接对应关系。如果没有这一损失，模型将无法直接从图像中学习语义 Voken。

这种综合方法<span style="color: rgb(46,161,33); background-color: inherit">利用了预训练模型、专用标记和创新的训练技术，确保了对文本和视觉元素的理解和生成的连贯性</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练策略</span>**

鉴于文本和图像领域之间不可忽视的领域差异，在有限的交错文本-图像数据集上直接训练会导致生成的文本和图像对齐不佳以及图像质量下降。因此，**MiniGPT-5** 采用两阶段训练策略：

> 1. 预训练阶段，专注于<span style="color: rgb(100,37,208); background-color: inherit">单模态生成的粗略特征对齐</span>
>
> 2. 微调阶段，致力于<span style="color: rgb(100,37,208); background-color: inherit">多模态生成的复杂特征学习</span>

此外，为了在整个扩散过程中增强生成 token 的有效性，在整个训练过程中引入了无分类器引导 **<span style="color: rgb(216,57,49); background-color: inherit">CFG</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">C</span>**<span style="color: rgb(216,57,49); background-color: inherit">lassifier-</span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">ree </span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">uidance）</span>技术。

**<span style="color: rgb(36,91,219); background-color: inherit">两阶段训练策略</span>**

认识到纯文本生成与文本-图像生成之间的显著领域差异，作者提出了一种两阶段训练策略：预训练阶段和微调阶段。首先，在单文本-图像对数据集中，&#x5982;**`CC3M`**，<span style="color: rgb(100,37,208); background-color: inherit">将 voken 特征与图像生成特征对齐</span>。这些数据集中每个样本仅包含一个文本和一个图像，且文本通常是图像的描述。在此阶段，<span style="color: rgb(100,37,208); background-color: inherit">利用描述作为 LLM 的输入，使 LLM 能够生成 voken</span>。由于这些数据集包含图像描述信息，因此<span style="color: rgb(100,37,208); background-color: inherit">引入了一个辅助损失以帮助 voken 对齐</span>，最小化生成特征$$  \hat{h}_{\text{voken}}  $$和文本到图像生成模型中文本编码器$$  \tau_\theta  $$的描述特征之间的距离：

$$L_{\text{CAP}} := \text{MSE}(\hat{h}_{\text{voken}}, \tau_\theta(c))$$

预训练阶段的总损失表示为：

$$L_{\text{Pretrain}} = \lambda_1 \cdot L_{\text{text}} + \lambda_2 \cdot L_{\text{LDM}} + \lambda_3 \cdot L_{\text{CAP}}$$

其中选择的权重值为$$\lambda_1 = 0.01$$、$$\lambda_2 = 1$$、$$\lambda_3 = 0.1$$，以将损失调整到相似的数值范围。

在预训练阶段之后，模型能够为单一文本描述生成图像，但<span style="color: rgb(216,57,49); background-color: inherit">在交错视觉-语言生成任务中表现不佳</span>。这类任务<span style="color: rgb(216,57,49); background-color: inherit">包含多个文本-图像对，并需要复杂的推理来生成文本和图像</span>。为了解决这一问题，在微调阶段，通过交错视觉-语言数据集进一步微调模型，&#x5982;**`VIST`**，使用 **PEFT&#x20;**&#x8FDB;行优化。在这些数据集中，每个样本包含多个步骤，并且文本和图像顺序相关。在此阶段，**MiniGPT-5** 从数据集中构建三种任务类型：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">仅文本生成</span>**：给定下一张图像，生成相关文本
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">仅图像生成</span>**：给定下一个文本，生成相关图像
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">多模态生成</span>**：根据上下文生成文本-图像对

微调阶段的损失定义为：

$$L_{\text{Fine-tune}} = \lambda_1 \cdot L_{\text{text}} + \lambda_2 \cdot L_{\text{LDM}}$$

**<span style="color: rgb(36,91,219); background-color: inherit">无分类器引导 CFG</span>**

为了增强生成文本和图像之间的一致性，首先利用无分类器引导的思想进行多模态生成。无分类器引导被引入到文本到图像的扩散过程中。该方法表明，<span style="color: rgb(100,37,208); background-color: inherit">通过对条件生成和无条件生成进行联合训练，生成模型</span>$$  P_\theta  $$<span style="color: rgb(100,37,208); background-color: inherit">可以获得改进的条件结果</span>。在这个场景中，我们<span style="color: rgb(100,37,208); background-color: inherit">希望模型直接关注来自 LLM 的输出特征</span>$$h_{\text{voken}}$$。相比于原始的稳定扩散无条件分布，即丢弃$$\hat{h}_{\text{voken}}$$，在无条件过程中也需要包含整个特征映射模块。因此要突出可训练条件$$h_{\text{voken}}$$，并固定生成模型。在训练过程中，&#x4EE5;**`10%`**&#x7684;概率将$$  h_{\text{voken}}  $$替换为零特征$$h_0 \in \mathbb{R}^{n \times d}$$，从而获得无条件特征：

$$\hat{h}_0 = \theta_{\text{enc-dec}}(\theta_{\text{MLP}}(h_0), q)$$

在推理过程中，$$\hat{h}_0$$被用作负提示，改进后的去噪过程为：

$$\log\hat{P_\theta}(\epsilon_t | z_{t+1}, \hat{h}_{\text{voken}}, \hat{h}_0) = \log P_\theta(\epsilon_t | z_{t+1}, \hat{h}_0) + \gamma \left( \log P_\theta(\epsilon_t | z_{t+1}, \hat{h}_{\text{voken}}) - \log P_\theta(\epsilon_t | z_{t+1}, \hat{h}_0) \right)$$

其中$$  \gamma  $$是控制引导强度的超参数。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**MiniGPT-5** 是一款创新的多模态生成模型，<span style="color: rgb(100,37,208); background-color: inherit">通过引入</span>**<span style="color: rgb(100,37,208); background-color: inherit">生成式 Voken</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 概念，成功构建了文本与图像特征空间的桥梁，实现了高质量的跨模态生成</span>。其独特的<span style="color: rgb(100,37,208); background-color: inherit">两阶段训练方法显著提升了模型在数据稀缺场景下的表现，同时结合扩散模型和语言模型的优势，能够生成连贯且贴切的图文内容</span>。无论是智能助手、创意设计还是教育领域，**MiniGPT-5** 展现了广泛的应用潜力，为多模态生成技术开辟了新方向。



### 5.2.8 <span style="color: rgb(36,91,219); background-color: inherit">BLIP3-o（b-3）</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

能够同时支持图像理解和生成的统一多模态架构已成为一个非常有前景的研究方向。**`Janus`**、**`Show-o`**、**`MetaMorph`** 、**`Janus-Pro`**&#x548C;**`LMFusion`**&#x7B49;模型进行了在单一框架内融合图像理解与生成能力的方法。**`GPT-4o`**&#x53C8;进一步，在图像生成和多模态理解方面均展现出很强的能力。

**<span style="color: rgb(222,120,2); background-color: inherit">推理与指令遵循</span>**

将图像生成能力集成到多模态 Autoregressive Model 中，可以<span style="color: rgb(100,37,208); background-color: inherit">继承其预训练知识、推理能力以及指令遵循能力</span>。传统图像生成模型难以达到很强的推理能力和世界知识水平，但当多模态大模型的指令遵循能力被整合进统一架构时，这些能力可能迁移到图像生成过程中。

**<span style="color: rgb(222,120,2); background-color: inherit">上下文学习</span>**

同时支持图像理解和生成的统一模型天然具备上下文学习能力。<span style="color: rgb(100,37,208); background-color: inherit">多模态输出可作为后续生成的上下文，从而无缝支持迭代式图像编辑、视觉对话和逐步视觉推理</span>。这避免了模式切换或对外部处理流程的依赖，使模型能够保持连贯性和任务连续性。

**<span style="color: rgb(222,120,2); background-color: inherit">通用人工智能</span>**

未来的通用人工智能系统需超越纯文本能力，实现对多模态内容的无缝感知、解读与生成。实现这一目标要求<span style="color: rgb(100,37,208); background-color: inherit">从纯文本架构转向统一的多模态架构，使其能够在多种模态间进行推理与生成</span>。此类模型对于构建能以整体性、类人方式与世界交互的通用智能至关重要。

目前用的比较多的统一多模态模型模式如下：

$$\text{Tokens} \longrightarrow [\text{Autoregressive Model}] \longrightarrow [\text{Diffusion Model}] \longrightarrow \text{Image Pixels}$$

有一些工作认为 GPT-4o 也用了这种模式，Autogress Model 与 Diffusion Model 联合利用，可以融合两者的优势。<span style="color: rgb(100,37,208); background-color: inherit">BLIP3-o 也采用这种</span>**`Autoregressive + Diffusion`**<span style="color: rgb(100,37,208); background-color: inherit">的框架</span>。但为了细化模型结构，还有两个关键问题需要解决：

> 1. 应使用 VAE 还是 CLIP 将图像编码为连续特征？
>
> 2. 如何对齐 Autoregressive Model 生成的视觉特征与真实图像特征？ 如何建模这些连续视觉特征的分布，采用简单的 MSE 损失，还是采用基于 Diffusion 的方法？

* **<span style="color: rgb(36,91,219); background-color: inherit">模型选型</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像编码与重建</span>**

<span style="color: rgb(100,37,208); background-color: inherit">图像生成一般首先通过编码器将图像编码为连续潜在 Embedding ，再由解码器从该 Embedding 重建图像。这种</span>**`Encoding-Decoding`**<span style="color: rgb(100,37,208); background-color: inherit">流程可有效降低图像生成中输入空间的维度，促进高效训练</span>。一般有以下两&#x79CD;**`Encoder-Decoder`**&#x8303;式。

**<span style="color: rgb(222,120,2); background-color: inherit">VAE</span>**

VAE 学习将图像编码到结构化的连续潜在空间中。<span style="color: rgb(100,37,208); background-color: inherit">Encoder 近似给定输入图像下潜在变量的后验分布，Decoder 则从该潜在分布中采样以重建图像</span>。Latent Diffusion Model 在此框架基础上，学习建模压缩后的潜在表示分布，而非原始像素。<span style="color: rgb(46,161,33); background-color: inherit">通过在 VAE 潜在空间中操作，这些模型显著降低了输出空间维度，从而减少计算开销并提升训练效率</span>。去噪步骤完成后，VAE 解码器将生成的潜在 Embedding 映射回原始图像像素。

**<span style="color: rgb(222,120,2); background-color: inherit">CLIP Encoder + Diffusion Decoder</span>**

CLIP 是图像理解任务的基础编码器。但将其用于图像生成比较困难，因为 CLIP 不是为重建任务设计的。**`Emu2`**&#x5C06; CLIP Encoder 与 Diffusion Decoder 配对：<span style="color: rgb(100,37,208); background-color: inherit">使用 EVA-CLIP 将图像编码为连续视觉 Embedding，通过基于 SDXL-base 初始化的 Diffusion Model 进行重建</span>。训练 Diffusion Decoder 时，<span style="color: rgb(100,37,208); background-color: inherit">以 EVA-CLIP 的视觉 Embedding 为条件，从高斯噪声中恢复原始图像，EVA-CLIP 保持冻结</span>。这有效将 CLIP 与 Diffusion 结合为图像自编码器：CLIP Encoder 将图像压缩为语义丰富的潜在 Embedding ，Diffusion Decoder 从中重建图像。虽然 Deocder 基于 Diffusion，但<span style="color: rgb(100,37,208); background-color: inherit">训练采用重建损失而非概率采样目标，因此推理时执行确定性重建</span>。

VAE 与 CLIP-Diffusion 这两种 Encoder-Deocder 架构代表了图像编码与重建的不同范式，各有优势与权衡。

> * **<span style="color: rgb(216,57,49); background-color: inherit">VAE</span>** 将图像编码为低层像素特征，重建质量更优，且作为现成模型易于集成到图像生成训练流程中
>
> * **<span style="color: rgb(216,57,49); background-color: inherit">CLIP-Diffusion</span>** 需额外训练以适配不同 CLIP Encoder，但在图像压缩率方面优势显著

> **<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：
>
> **`CLIP-Diffusion`**&#x67B6;构代&#x8868;**`Emu2`**&#x65E0;论分辨率如何，每张图像均可编码为固定长&#x5EA6;**`64`**&#x7684;连续向量，提供紧凑且语义丰富的潜在 Embedding 。
>
> **`VAE`** Encoder 对高分辨率输入通常生成更长的潜在 Embedding 序列，增加训练计算负担。

* **<span style="color: rgb(36,91,219); background-color: inherit">潜在图像表示</span>**

获得连续图像 Embedding 后，BLIP3-o 使用 Autoregressive 架构对其进行建模。给定用户 prompt，首先通过 Autoregressive Model 的输入 Embedding 层将其编码为 Embedding 序列$$\mathbf{C}$$，并在其后附加一个可学习的 Query 向量$$\mathbf{Q}$$，$$\mathbf{Q}$$向量随机初始化并在训练中优化。当组合序列$$[\mathbf{C}; \mathbf{Q}]$$通过 Autoregressive Transformer 时，$$\mathbf{Q}$$学会关注并从 prompt $$\mathbf{C}$$中提取相关语义信息。最终得到的$$\mathbf{Q}$$为 Autoregressive Model 生成的中间视觉特征或潜在表示，并用来训练以逼近来自 VAE 或 CLIP 的真实图像特征$$\mathbf{X}$$。BLIP3-o 使用均方误差 MSE 和流匹配 Flow Matching 两个目标函数将$$\mathbf{Q}$$与真实图像 Embedding $$\mathbf{X}$$对齐。

**<span style="color: rgb(222,120,2); background-color: inherit">MSE 损失</span>**<span style="color: rgb(222,120,2); background-color: inherit"> </span>&#x20;

给定 Autoregressive Model 预测的视觉特征$$\mathbf{Q}$$与真实图像特征$$\mathbf{X}$$，损失定义为：

$$\mathcal{L}_{\text{MSE}} = \|\mathbf{X} - \mathbf{W}\mathbf{Q}\|_2^2$$

其中$$\mathbf{W}$$为可学习投影矩阵，这里使用可学习的线性投影以对齐二者维度。

**<span style="color: rgb(222,120,2); background-color: inherit">流匹配 Flow Matching</span>**<span style="color: rgb(222,120,2); background-color: inherit">  </span>

MSE 损失仅将预测特征$$\mathbf{Q}$$与目标分布的均值对齐。理想的训练目标应建模连续图像表示的概率分布。BLIP3-o 采用流匹配，从先验分布迭代传输样本来采样目标连续分布。<span style="color: rgb(100,37,208); background-color: inherit">给定真实图像特征</span>$$\mathbf{X}_1$$<span style="color: rgb(100,37,208); background-color: inherit">与 Autoregressive Model 编码的条件</span>$$\mathbf{Q}$$<span style="color: rgb(100,37,208); background-color: inherit">，在每步训练中，采样时间步</span>$$t \sim \mathcal{U}(0,1)$$<span style="color: rgb(100,37,208); background-color: inherit">和噪声</span>$$\mathbf{X}_0 \sim \mathcal{N}(0,1)$$<span style="color: rgb(100,37,208); background-color: inherit">。 Diffusion Transformer 学习预测在时间步</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">、以</span>$$\mathbf{Q}$$<span style="color: rgb(100,37,208); background-color: inherit">为条件、朝向</span>$$\mathbf{X}_1$$<span style="color: rgb(100,37,208); background-color: inherit">方向的速度</span>$$\mathbf{V}_t = \frac{d\mathbf{X}_t}{dt}$$。BLIP3-o 通过$$\mathbf{X}_0$$与$$\mathbf{X}_1$$的线性插值得到$$\mathbf{X}_t$$：

$$\mathbf{X}_t = t \mathbf{X}_1 + (1 - t) \mathbf{X}_0$$

其解析速度为：

$$\mathbf{V}_t = \frac{d\mathbf{X}_t}{dt} = \mathbf{X}_1 - \mathbf{X}_0$$

最终训练目标定义为：

$$\mathcal{L}_{\text{Flow}}(\theta) = \mathbb{E}_{(\mathbf{X}_1,\mathbf{Q}) \sim \mathcal{D},\, t \sim \mathcal{U}(0,1),\, \mathbf{X}_0 \sim \mathcal{N}(0,1)} \left[ \left\| \mathbf{V}_\theta(\mathbf{X}_t, \mathbf{Q}, t) - \mathbf{V}_t \right\|_2^2 \right]$$

其中$$\theta$$为 Diffusion Transformer 的参数，$$\mathbf{V}_\theta(\mathbf{X}_t, \mathbf{Q}, t)$$表示基于样本$$(\mathbf{X}_1, \mathbf{Q})$$、时间步$$t$$和噪声$$\mathbf{X}_0$$预测的速度。

> 与离散 token 不同，连续表示本身不支持基于采样的多样化生成路径探索。在 MSE 训练目标下，给定 prompt 的预测视觉特征$$\mathbf{Q}$$几乎是确定性的，导致<span style="color: rgb(216,57,49); background-color: inherit">无论视觉解码器基于 VAE 还是 CLIP+ Diffusion 架构，多次推理生成的图像几乎完全相同</span>。这种确定性凸显了 MSE 目标的关键局限：它<span style="color: rgb(216,57,49); background-color: inherit">迫使模型对每个 prompt 仅生成单一固定输出，限制了生成多样性</span>。
>
> Flow Matching 使模型继承 Diffusion 过程的随机性，从而<span style="color: rgb(46,161,33); background-color: inherit">在相同 prompt 条件下生成多样化的图像样本，促进对输出空间的更广探索</span>。但此灵活性以增加模型复杂度为代价：<span style="color: rgb(216,57,49); background-color: inherit">流匹配引入了比 MSE 更多的可学习参数</span>。
>
> 综上，BLIP3-o 采用 DiT，扩大其容量可显著提升性能。

* **<span style="color: rgb(36,91,219); background-color: inherit">组合设计选择</span>**

不同图像 Encoder–Decoder 架构与训练目标的组合，产生了多种图像生成模型设计选择：

![](../../images/视觉多模态讲义（下）-image-165.png)

**<span style="color: rgb(222,120,2); background-color: inherit">CLIP + MSE</span>** &#x20;

沿&#x7528;**`Emu2`**、**`Seed-X`**&#x548C; **`MetaMorph`**&#x7684;做法，使用 CLIP 将图像编码&#x4E3A;**`64`**&#x7EF4;固定长度、语义丰富的视觉 Embedding。<span style="color: rgb(100,37,208); background-color: inherit">训练自回归模型以最小化预测视觉特征</span>$$\mathbf{Q}$$<span style="color: rgb(100,37,208); background-color: inherit">与真实 CLIP Embedding </span>$$\mathbf{X}$$<span style="color: rgb(100,37,208); background-color: inherit">之间的 MSE 损失</span>。推理时，给定文本 prompt $$\mathbf{C}$$，自回归模型预测潜在视觉特征$$\mathbf{Q}$$，随后传入基于 Diffusion 的视觉 Decoder 重建真实图像。

**<span style="color: rgb(222,120,2); background-color: inherit">CLIP + Flow Matching</span>**

采用 Flow Matching 损失训练模型预测真实 CLIP Embedding。给定 prompt $$\mathbf{C}$$，<span style="color: rgb(100,37,208); background-color: inherit">自回归模型生成视觉特征序列</span>$$\mathbf{Q}$$<span style="color: rgb(100,37,208); background-color: inherit">，用作 Diffusion 过程的条件，以逼近真实 CLIP 特征</span>。推理流程包含两个 Diffusion 阶段：第一阶段以$$\mathbf{Q}$$为条件，迭代去噪生成 CLIP Embedding（支持随机采样，提升生成多样性）；第二阶段通过 Diffusion Decoder 将 CLIP Embedding 转换为真实图像。

**<span style="color: rgb(222,120,2); background-color: inherit">VAE + Flow Matching</span>**

使用 Flow Matching 损失预测真实 VAE 特征，类似 **`MetaQuery`**。推理时，给定 prompt $$\mathbf{C}$$， 自回归模型生成$$\mathbf{Q}$$，随后以$$\mathbf{Q}$$为条件，通过 VAE 解码器迭代去噪生成真实图像。



为比较不同设计选择，BLIP3-o 采&#x7528;**`Llama-3.2-1B-Instruct`**&#x4F5C;为 Autoregressive Model。训练数据包括 **`CC12M`**、**`SA-1B`**&#x548C;**`JourneyDB`**，总计&#x7EA6;**`25M`**&#x6837;本。CC12M 与 SA-1B 使用 LLaVA 生成的详细描述，JourneyDB 使用原始描述。通过实验发现：**<span style="color: rgb(100,37,208); background-color: inherit">CLIP + Flow Matching</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 在 GenEval 与 DPG-Bench 上取得最佳 prompt 对齐分数，</span>**<span style="color: rgb(100,37,208); background-color: inherit">是最有效的设计选择</span>**<span style="color: rgb(100,37,208); background-color: inherit">；</span>**<span style="color: rgb(100,37,208); background-color: inherit">VAE + Flow Matching</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 获得最优 FID，表明其美学质量更优</span>。

![](../../images/视觉多模态讲义（下）-image-166.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**
>
> 1. CLIP 特征比 VAE 特征更紧凑且语义更丰富，带来更高训练效率
>
> 2. 自回归模型在学习语义级特征 (CLIP) 方面比像素级特征 (VAE) 更有效
>
> 3. Flow Matching 作为训练目标能更好建模图像分布，提升样本多样性与视觉质量

* **<span style="color: rgb(36,91,219); background-color: inherit">训练策略</span>**

在经过上面的探究之后，BLIP3-o 采&#x7528;**`CLIP + Flow Matching`**&#x4F5C;为图像生成模块。由于图像理解也在 CLIP Embedding 空间中操作，因此可以将这两项任务对齐至同一语义空间，实现统一。这里有两种训练策略可以选择：

![](../../images/视觉多模态讲义（下）-image-167.png)

**<span style="color: rgb(222,120,2); background-color: inherit">联合训练</span>**

最近一些工作的常见做法，&#x5982;**`MetaMorph`**、**`Janus-Pro`**、**`Show-o`**。虽然这些方法采用不同生成架构，但均通过混合图像理解与生成数据进行多任务学习。

**<span style="color: rgb(222,120,2); background-color: inherit">顺序训练</span>** &#x20;

两阶段方法：第一阶段<span style="color: rgb(100,37,208); background-color: inherit">仅训练图像理解模块</span>；第二阶段冻结 MLLM 主干，<span style="color: rgb(100,37,208); background-color: inherit">仅训练图像生成模块</span>，类&#x4F3C;**`LMFusion`**&#x4E0E;**`MetaQuery`**。

在联合训练中，两项任务可能相互促进，这在MetaMorph中被证明，但其协同效应受两个关键因素影响：

> * 总数据量
>
> * 图像理解与生成数据的比例

相比之下，顺序训练更具灵活性：<span style="color: rgb(46,161,33); background-color: inherit">可冻结 Autoregressive 主干以保留图像理解能力，并将全部训练能力专注于图像生成，避免联合训练中的任务干扰</span>。&#x4E0E;**`LMFusion`**&#x548C;**`MetaQuery`**&#x7C7B;似的，BLIP3-o 选择顺序训练构建统一多模态模型。

* **<span style="color: rgb(36,91,219); background-color: inherit">BLIP3-o 总体概览</span>**

![](../../images/视觉多模态讲义（下）-image-194.png)

通过上面的探究，BLIP3-o 采&#x7528;**`CLIP + Flow Matching`**&#x4E0E;`顺序训练`，开发&#x4E86;**`4B`**&#x548C;**`8B`**&#x4E24;种规模模型。由于市面上已有强大开源图像理解模型，比&#x5982;**`Qwen2.5-VL`**，BLIP3-o 跳过图像理解训练阶段，直接在上面构建图像生成模块：

> * **<span style="color: rgb(36,91,219); background-color: inherit">8B 参数模型</span>**：使用私有数据训练。<span style="color: rgb(100,37,208); background-color: inherit">冻结 Qwen2.5-VL-7B-Instruct 主干，仅训练 Diffusion Transformer，共 1.4B 可训练参数</span>
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">4B 参数模型</span>**：仅使用开源数据。与 8B 采用相同生成架构，但主干为 Qwen2.5-VL-3B-Instruct。

1. **<span style="color: rgb(36,91,219); background-color: inherit">Diffusion Transformer 架构</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

采&#x7528;**`Lumina-Next`**&#x6A21;型架构，其基于改进&#x7684;**`Next-DiT`**&#x67B6;构，是可扩展、高效的 Diffusion Transformer，专为文本到图像及通用多模态生成设计。<span style="color: rgb(100,37,208); background-color: inherit">引入</span>**`3D RoPE Embedding`**<span style="color: rgb(100,37,208); background-color: inherit">，在不依赖可学习位置 token 的情况下编码时间、高度、宽度上的时空结构</span>。每个 Transformer 块采&#x7528;**`sandwich normalization`**，即注意力/MLP 前后均使用 RMSNorm，以及分组查询注意&#x529B;**`GQA`**，以提升稳定性并降低计算量。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练方案</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">Step 1：图像生成预训练</span>**<span style="color: rgb(222,120,2); background-color: inherit"> </span>&#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">8B 模型</span>**：结合&#x7EA6;**`25M`**&#x5F00;源数据与额&#x5916;**`30M`**&#x79C1;有图像。所有图像描述由 Qwen2.5-VL-7B-Instruct 生成，平均长度 120 tokens。为提升对不同 prompt 长度的泛化能力，额外加入约 10%（**`6M`**）来自 CC12M 的约 20 tokens 短描述。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">4B 模型</span>**：使用同样&#x7684;**`25M`**&#x516C;开图像，配相同详细描述，并混合约 10%（**`3M`**）短描述。

**<span style="color: rgb(222,120,2); background-color: inherit">Step 2：图像生成指令微调</span>** &#x20;

预训练后，模型在如下一些场景还存在一些缺陷：

> * 生成复杂人体姿态，如<span style="color: rgb(220,155,4); background-color: inherit">人射箭</span>
>
> * 生成常见物体，如<span style="color: rgb(220,155,4); background-color: inherit">各类果蔬</span>
>
> * 生成地标，如<span style="color: rgb(220,155,4); background-color: inherit">金门大桥</span>
>
> * 生成简单文字

虽然预训练本应覆盖这些类别，但数据规模有限导致覆盖不足。为此，针对这些领域进行指令微调：对每类，<span style="color: rgb(100,37,208); background-color: inherit">使用 GPT-4o 生成约</span>**`10K`**<span style="color: rgb(100,37,208); background-color: inherit">条 prompt–image</span>**<span style="color: rgb(100,37,208); background-color: inherit"> </span>**<span style="color: rgb(100,37,208); background-color: inherit">对，构建针对性数据集</span>。同时，<span style="color: rgb(100,37,208); background-color: inherit">从 JourneyDB 与 DALL·E 3 补充 prompt 以提升视觉美学质量</span>，最终获得&#x7EA6;**`60k`**&#x6761;高质量 prompt–image 对。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：
>
> 1. 模型能快速适应 GPT-4o 风格，提升 prompt 对齐与视觉质量
>
> 2. 模型从 AI 生成图像中学到的效果优于从真实图像中学习

### 5.2.9 **<span style="color: rgb(36,91,219); background-color: inherit">Janus（b-4）</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">架构  </span>**

![](../../images/视觉多模态讲义（下）-image-193.png)

针对纯文本理解、多模态理解以及视觉生成任务，Janus 采用独立的编码方法将原始输入转换为特征，随后由一个<span style="color: rgb(100,37,208); background-color: inherit">统一的自回归 Transformer 进行处理</span>：

> * **<span style="color: rgb(36,91,219); background-color: inherit">文本理解</span>**：使用 <span style="color: rgb(100,37,208); background-color: inherit">LLM 内置的 Tokenizer 将文本转换为离散的 ID</span>，并获取每个 ID 对应的特征表示
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**：使用 <span style="color: rgb(100,37,208); background-color: inherit">SigLIP Encoder 从图像中提取高维语义特征</span>。这些特征原本呈二维网格结构，被展平为一维序列后，<span style="color: rgb(100,37,208); background-color: inherit">通过</span>**<span style="color: rgb(100,37,208); background-color: inherit">理解适配器 </span>**<span style="color: rgb(100,37,208); background-color: inherit">understanding adaptor 映射到 LLM 的输入空间</span>
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉生成</span>**：<span style="color: rgb(100,37,208); background-color: inherit">使用 VQ Tokenizer 将图像转换为离散 ID</span>。ID 序列被展平为一维后，<span style="color: rgb(100,37,208); background-color: inherit">通过</span>**<span style="color: rgb(100,37,208); background-color: inherit">生成适配器 </span>**<span style="color: rgb(100,37,208); background-color: inherit">generation adaptor 将每个 ID 对应的 Codebook Embedding 映射至 LLM 的输入空间</span> &#x20;

随后将这<span style="color: rgb(100,37,208); background-color: inherit">几类特征序列拼接成一个多模态特征序列，送入 LLM 进行处理</span>。在纯文本理解和多模态理解任务中，使用 LLM 内置的 prediction head 进行文本预测；在视觉生成任务中，使用一个随机初始化的预测头进行图像预测。**<span style="color: rgb(100,37,208); background-color: inherit">整个模型使用自回归框架</span>**，不需要设计特殊的注意力掩码。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**<span style="color: rgb(36,91,219); background-color: inherit">Semantic Tokenizer 的探索</span>**
>
> ![使用预训练的 SigLIP 监督语义信息的重建，同时使用原始图像监督 RGB 像素值的重建](../../images/视觉多模态讲义（下）-image-191.png)
>
> ![Semantic Tokenizer 输出具有高层语义的连续特征，这些特征经过一个 Adaptor 后，作为 LLM 的输入](../../images/视觉多模态讲义（下）-image-192.png)
>
> 在 LlamaGen 的 Tokenizer 架构基础上构建了 Semantic Tokenizer，下采样率为$$16$$。除了原有&#x7684;**`CNN Pixel Decoder`**&#x5916;，在向量量化 **<span style="color: rgb(216,57,49); background-color: inherit">VQ</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ector </span>**<span style="color: rgb(216,57,49); background-color: inherit">Q</span>**<span style="color: rgb(216,57,49); background-color: inherit">uantization）</span>之后额外增加了一&#x4E2A;**`Semantic Decoder`**&#x5206;支，Semantic Decoder 是一个$$12$$层&#x7684;**`ViT`**，包含$$12$$个注意力头，隐藏维度为$$768$$。在将 Semantic Decoder 与 LLM 集成时，对其使用因果注意力掩码，以支持下一 token 的预测。
>
> ![SE. Tokenizer 表示 semantic tokenizer](../../images/视觉多模态讲义（下）-image-190.png)
>
> 这里只是在消融实验中探索了 semantic tokenizer，实际并未真正采用

* **<span style="color: rgb(36,91,219); background-color: inherit">训练流程</span>**

训练分为三个阶段：

![](../../images/视觉多模态讲义（下）-image-189.png)

主要目标是在 Embedding 空间中建立视觉与语言元素之间的概念关联，使 LLM 能够理解图像中呈现的实体，并具备初步的视觉生成能力。其中<span style="color: rgb(100,37,208); background-color: inherit">视觉编码器和 LLM 保持冻结，仅更新理解适配器、生成适配器和图像预测头中的可训练参数</span>。

利用多模态语料进行统一预训练，同时学习理解与生成能力。<span style="color: rgb(100,37,208); background-color: inherit">使用所有训练数据：纯文本、多模态理解和视觉生成数据</span>。首先使用 ImageNet-1k 进行简单的视觉生成训练，掌握像素依赖关系；随后用文本到图像数据增强模型在开放域的视觉生成能力。

使用指令微调数据对预训练模型进行微调，提升其遵循指令和对话能力。<span style="color: rgb(100,37,208); background-color: inherit">除生成编码器 Gen. Encoder 外，其余所有参数均参与微调</span>。重点监督回答部分，同时对系统 prompt 和用户 prompt 进行掩码处理。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：为确保在多模态理解与生成两方面均具备良好性能，Janus <span style="color: rgb(100,37,208); background-color: inherit">并未为特定任务单独微调模型，而是混合使用纯文本对话数据、多模态理解数据和视觉生成数据，从而保证模型在各种场景下的通用性</span>。

训练时直接采&#x7528;**<span style="color: rgb(100,37,208); background-color: inherit">交叉熵损失函数</span>**： &#x20;

$$\mathcal{L} = -\sum_{i=1}^{N} \log P_{\theta}(x_i \mid x_{<i})$$

其中，$$P(\cdot \mid \cdot)$$表示由 Janus 模型参数$$\theta$$所建模的条件概率。 &#x20;

> * 对于纯文本理解和多模态理解，在文本序列上计算损失 &#x20;
>
> * 对于视觉生成，仅在图像序列上计算损失

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：为保持设计简洁，<span style="color: rgb(100,37,208); background-color: inherit">未对不同任务分配不同的损失权重</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理  </span>**

采用下一 token 预测的方法： &#x20;

> * 对于纯文本理解和多模态理解，我们遵循标准做法，从预测分布中顺序采样 token
>
> * 对于图像生成，采用无分类器引导 **<span style="color: rgb(216,57,49); background-color: inherit">CFG</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">C</span>**<span style="color: rgb(216,57,49); background-color: inherit">lassifier-</span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">ree </span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">uidance）</span>，每个 token 的 logits $$l_g$$按如下方式计算： &#x20;
>
> $$l_g = l_u + s(l_c - l_u)$$
>
> 其中，$$l_c$$为条件 logits，$$l_u$$为无条件 logits，$$s$$为 CFG 引导尺度。这里默认设置$$s = 5$$

### 5.2.10 **<span style="color: rgb(36,91,219); background-color: inherit">Janus-Pro（b-4）</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构  </span>**

Janus-Pro 的架构与 Janus 相同。<span style="color: rgb(100,37,208); background-color: inherit">整体架构的核心设计原则是</span>**<span style="color: rgb(100,37,208); background-color: inherit">解耦用于多模态理解与视觉生成的视觉编码过程</span>**。采用独立的编码方法将原始输入转换为特征，随后由一个统一的自回归 Transformer 进行处理。

> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**：使用 SigLIP 编码器从图像中提取高维语义特征。这些特征原本呈二维网格结构，被展平为一维序列后，通过一个**理解适配器&#x20;**&#x75;nderstanding adaptor 映射到 LLM 的输入空间 &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉生成</span>**：采用 VQ Tokenizer 将图像转换为离散 ID。ID 序列被展平为一维后，通过一个**生成适配器&#x20;**&#x67;eneration adaptor 将每个 ID 对应的 codebook embedding 映射至 LLM 的输入空间&#x20;

然后将上述特征序列拼接成一个多模态特征序列，送入 LLM 进行处理。除了 LLM 内置的预测头外，在视觉生成任务中还<span style="color: rgb(100,37,208); background-color: inherit">额外使用一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">随机初始化的预测头</span>**<span style="color: rgb(100,37,208); background-color: inherit">用于图像预测</span>。整个模型严格遵循自回归框架。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练策略  </span>**

Janus 采用三阶段训练流程： &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：训练适配器和图像预测头
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：统一预训练，除理解编码器和生成编码器外，其余组件参数均参与更新
>
>   1. 在 ImageNet 数据上训练，使用图像类别名称作为 prompt，建模像素依赖关系
>
>   2. 在常规文本到图像数据上训练
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3</span>**：监督微调，在阶段 2 的基础上进一步解冻理解编码器的参数进行训练

但这种策略存在若干问题：在 Stage 2 中，Janus 将文本到图像能力的训练分为两部分，**`66.67%`**<span style="color: rgb(216,57,49); background-color: inherit">的文本到图像训练步数分配给了第一部分，但该策略</span>**<span style="color: rgb(216,57,49); background-color: inherit">次优且计算效率低下</span>**。因此 Janus-Pro 进行了几项改进：

> * **<span style="color: rgb(36,91,219); background-color: inherit">延长 Stage 1 的训练</span>**：<span style="color: rgb(100,37,208); background-color: inherit">增加 Stage 1 的训练步数，使其在 ImageNet 数据集上得到充分训练</span>。实验表明即使 LLM 参数冻结，模型也能有效建模像素依赖关系，并基于类别名称生成合理的图像。 &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">聚焦 Stage 2 的训练</span>**：Stage 2 **<span style="color: rgb(100,37,208); background-color: inherit">完全移除 ImageNet 数据</span>**<span style="color: rgb(100,37,208); background-color: inherit">，直接使用常规文本到图像数据训练模型，使其能够基于密集描述生成图像</span>。这使 Stage 2 能更高效地利用文本到图像数据，从而提升训练效率与整体性能。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">调整 Stage 3 的数据</span>**：调整监督微调过程中不同类型数据的比例，将多模态数据、纯文本数据和文本到图像数据的比例从原先的$$7:3:10$$调整为$$5:1:4$$。<span style="color: rgb(46,161,33); background-color: inherit">通过略微降低文本到图像数据的比例，可以在</span>**<span style="color: rgb(46,161,33); background-color: inherit">保持强大视觉生成能力的同时，显著提升了多模态理解性能</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据与模型扩展  </span>**

在多模态理解和视觉生成两个方面对 Janus 的训练数据进行了大规模扩展：

> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**
>
>   * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2 预训练数据</span>**：<span style="color: rgb(100,37,208); background-color: inherit">新增约</span>**`90M`**<span style="color: rgb(100,37,208); background-color: inherit">样本，包括图像描述数据集以及面向表格、图表和文档理解的数据</span>
>
>   * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3 监督微调数据</span>**：引入 DeepSeek-VL2 中的额外数据集，例如 <span style="color: rgb(220,155,4); background-color: inherit">MEME 理解、中文对话数据以及旨在提升对话体验的数据集</span>。<span style="color: rgb(46,161,33); background-color: inherit">显著增强了模型处理多样化任务的能力，并全面改善了对话体验</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉生成</span>**
>
>   先前版本 Janus 使用的真实世界数据质量较低且噪声较大，导致文本到图像生成不稳定，输出图像美学质量较差。在 Janus-Pro 中，<span style="color: rgb(100,37,208); background-color: inherit">引入约</span>**`72M`**<span style="color: rgb(100,37,208); background-color: inherit">条合成的高质量美学数据，使统一预训练阶段中真实数据与合成数据的比例达到</span>$$1:1$$。<span style="color: rgb(46,161,33); background-color: inherit">模型在合成数据上训练时收敛更快，生成的文本到图像结果不仅更稳定，美学质量也显著提升</span>。

模型扩展：&#x20;

先前版本的 Janus 使用 1.5B 参数量的 LLM 验证了视觉编码解耦的有效性。在 Janus-Pro 中，将模型规模扩展至 **7B**。<span style="color: rgb(100,37,208); background-color: inherit">当使用更大规模的 LLM 时，</span>**<span style="color: rgb(100,37,208); background-color: inherit">多模态理解与视觉生成任务的损失收敛速度均显著快于小模型</span>**。这进一步验证 <span style="color: rgb(46,161,33); background-color: inherit">Janus-Pro 具有强大的可扩展性</span>。

![](../../images/视觉多模态讲义（下）-image-186.png)

### 5.2.11 **<span style="color: rgb(36,91,219); background-color: inherit">S</span> <span style="color: rgb(46,161,33); background-color: inherit">h</span> <span style="color: rgb(216,57,49); background-color: inherit">o</span> <span style="color: rgb(220,155,4); background-color: inherit">w</span>-o2<span style="color: rgb(36,91,219); background-color: inherit">（b-5）</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">模型框架</span>**

整体模型结构图如下：

![](../../images/视觉多模态讲义（下）-image-188.png)

具体流程为：

> 1. 给定交错排列的文本、图像或视频，系统<span style="color: rgb(100,37,208); background-color: inherit">分别通过一个带有 Embedding 层的 text tokenizer 和一个 3D causal VAE encoder，将其处理为连续的文本 Embedding 和视觉潜在表征</span>
>
> 2. 这些视觉潜在表征<span style="color: rgb(100,37,208); background-color: inherit">经过双路径提取，进行空间融合，以构建统一的视觉表征</span>；对于视频还需要进行时间融合
>
> 3. 这些表征被组织成序列，并<span style="color: rgb(100,37,208); background-color: inherit">输入到一个带有 </span>**<span style="color: rgb(100,37,208); background-color: inherit">language head</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">flow head</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的语言模型</span>中，分别<span style="color: rgb(100,37,208); background-color: inherit">通过自回归建模和 flow matching 对序列进行建模</span>
>
> 4. 利用 text de-tokenizer 与 3D causal VAE decoder 联合解码出最终输出

1. **<span style="color: rgb(36,91,219); background-color: inherit">统一视觉表征</span>**

为了可扩展地支持图像和视频模态，Show-o2 采用 3D causal VAE encoder 提取图像或视频的潜在表示。由于多模态理解与生成在特征依赖性上存在差异，作者提出一种双路径架构：

> **<span style="color: rgb(222,120,2); background-color: inherit">语义层</span>$$S(\cdot)$$<span style="color: rgb(222,120,2); background-color: inherit">用于提取富含语义上下文信息的高层表征</span>**：$$S(\cdot)$$共&#x4EAB;**`SigLIP`**&#x7684;视觉 Transformer 模块，并引入一个新的$$2 \times 2$$ patch embedding 层。给定在噪声水平$$t$$下的$$n$$个视觉潜在变量$$\mathbf{x}_t = \{x_i\}_{i=1}^n$$：
>
> $$\mathbf{x}_t = t \cdot \mathbf{x}_1 + (1 - t) \cdot \mathbf{x}_0$$
>
> 其中$$\mathbf{x}_0 \sim \mathcal{N}(0, 1)$$且$$t \sim [0, 1]$$，<span style="color: rgb(100,37,208); background-color: inherit">Show-o2 加载 SigLIP 的预训练权重，并对</span>$$S(\cdot)$$<span style="color: rgb(100,37,208); background-color: inherit">进行 pre-distill</span>：
>
> $$\mathcal{L}_{\text{distill}} = -\frac{1}{n} \sum \log \operatorname{sim}\big(S(\mathbf{x}_t), \text{SigLIP}(X)\big)$$
>
> 其中$$X$$为输入图像，$$\text{SigLIP}(\cdot)$$提取图像 patch 特征，$$\operatorname{sim}(\cdot)$$表示余弦相似度计算函数。这样<span style="color: rgb(46,161,33); background-color: inherit">语义层</span>$$S(\cdot)$$<span style="color: rgb(46,161,33); background-color: inherit">能够从干净或含噪的视觉潜在变量</span>$$\mathbf{x}_t$$<span style="color: rgb(46,161,33); background-color: inherit">中模拟提取语义特征</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">投影层</span>$$P(\cdot)$$<span style="color: rgb(222,120,2); background-color: inherit">保留从视觉潜在表示中提取的完整底层信息</span>**：$$P(\cdot)$$ 由一个二维 patch embedding 层构成。

> **<span style="color: rgb(222,120,2); background-color: inherit">Spatial (-Temporal) Fusion</span>**：所提取的<span style="color: rgb(100,37,208); background-color: inherit">高、低层表征通过特征维度拼接，并由 RMSNorm 与两个 MLP 层进行空间融合，对视频还要进行时间融合</span>，得到统一视觉表征$$\mathbf{u}$$：
>
> $$\mathbf{u} = \text{STF}\big(S(\mathbf{x}_t), P(\mathbf{x}_t)\big)$$
>
> 其中$$\text{STF}$$表示空间-时间融合机制。为支持生成建模，<span style="color: rgb(100,37,208); background-color: inherit">在统一视觉表征前添加时间步</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">的 Embedding，对于干净图像，设 </span>$$t = 1.0$$<span style="color: rgb(100,37,208); background-color: inherit"> 以获得对应的 time step embedding</span>。

然后将文本 Embedding 与统一视觉表征按交错图文格式组织成序列：

$$\texttt{[BOS] \{Text\} [BOI / BOV] \{Image / Video\} [EOI / EOV] \{Text\} \dots [EOS]}$$

这种序列格式灵活，可<span style="color: rgb(46,161,33); background-color: inherit">适配多种输入类型</span>。这里依然采用全注意力机&#x5236;**`omni-attention`**，使序列建模保持因果性，同时在统一视觉表征内部允许全注意力交互。

* **<span style="color: rgb(36,91,219); background-color: inherit">Flow Head</span>**

除用于文本 token 预测的 language head 外，Show-o2 还引入一个 flow head，用于通过 flow matching 预测定义的速度场$$\mathbf{v}_t = \frac{d\mathbf{x}_t}{dt}$$。flow head 由若干 Transformer 层构成，并通&#x8FC7;**`adaLN-Zero`**&#x6A21;块实现时间步控制。训练过程中，<span style="color: rgb(100,37,208); background-color: inherit">对 language head 应用下一 token 预测损失</span>$$\mathcal{L}_{\text{NTP}}$$<span style="color: rgb(100,37,208); background-color: inherit">，对 flow head 应用 flow matching 损失</span>$$\mathcal{L}_{\text{FM}}$$<span style="color: rgb(100,37,208); background-color: inherit">以预测速度</span>：

$$\mathcal{L} = \alpha \mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{FM}}$$

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

现有的统一多模态模&#x578B;**`UMM`**，&#x5982;**`Show-o`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`JanusPro`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`Transfusion`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`Chameleon`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`Emu3`**，通常<span style="color: rgb(100,37,208); background-color: inherit">从大语言模型 LLM、大型多模态模型 LMM 或从零开始训练</span>。这些方法都是在提升视觉生成建模能力的同时保留语言建模能力。但<span style="color: rgb(216,57,49); background-color: inherit">这个过程依赖于网络规模的高质量文本语料库，而此类语料的收集成本极高。因此缺乏此类资源可能导致语言知识和建模性能下降</span>。

为了解决这个问题，Show-o2 采用两阶段训练方案，在无需大规模文本语料的前提下，有效保留语言知识并同步发展视觉生成能力。

![](../../images/视觉多模态讲义（下）-image-187.png)

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：由于在之前已对语义层$$S(\cdot)$$进行了预蒸馏。第一阶段<span style="color: rgb(100,37,208); background-color: inherit">仅训练 </span>**<span style="color: rgb(100,37,208); background-color: inherit">projector</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">spatial (-temporal) fusion</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">flow head</span>**。使用&#x7EA6;**`66M`**&#x56FE;像-文本对，通过自回归建模与 flow matching 训练上述模块，并逐步引入交错数据和视频-文本对。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：使&#x7528;**`9M`**&#x6761;高质量多模态理解指令数据、&#x4ECE;**`66M`**&#x4E07;图像-文本对中筛选出&#x7684;**`16M`**&#x6761;高质量视觉生成数据，以&#x53CA;**`1.6M`**&#x6761;视频理解数据，对整个模型进行微调。

在完成 1.5B LLM 参数的小模型训练后，将预训练的 flow head 迁移到 7B LLM 参数的大模型中，并引入一个轻量级 MLP 变换层以对齐隐藏维度，使其能快速适应更大模型并收敛。

---

[Previous](12-UMM-统一理解生成模型--Diffusion-based.md) | [Contents](../../README.md) | [Next](14-UMM-统一理解生成模型--AR--Diffusion.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=13)
