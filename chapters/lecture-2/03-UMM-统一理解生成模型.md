[Previous](02-Diffusion-Model.md) | [Contents](../../README.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=3)

# 5. <span style="color: rgb(36,91,219); background-color: inherit">UMM 统一理解生成模型</span>

视觉理解（Vision Understanding）与视觉生成（Vision Generation）长期被视为多模态智能的一体两面：

> * **<span style="color: rgb(36,91,219); background-color: inherit">理解</span>**&#x4EFB;务旨在通过对视觉输入的语义提炼，实现分类、检测、问答及复杂逻辑推理
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">生成</span>**&#x4EFB;务则试图对视觉分布进行建模，从而根据指令创作出符合物理常识与审美逻辑的像素内容

尽管两者在应用目标上殊途同归，但在架构层面却曾经历了长期的分裂：<span style="color: rgb(100,37,208); background-color: inherit">自回归架构（Autoregressive, AR）因其强大的序列建模能力统治了多模态理解领域，而扩散模型（Diffusion Models, DM）则因其卓越的分布建模保真度成为了图像生成的基石</span>。在学术界与工业界一直致力于打破这种架构鸿沟，试图在统一的 Transformer 框架下融合理解与生成任务。

* **<span style="color: rgb(36,91,219); background-color: inherit">架构分类</span>**

**<span style="color: rgb(216,57,49); background-color: inherit">UMM</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">U</span>**<span style="color: rgb(216,57,49); background-color: inherit">nified </span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ultimodal </span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">odel）</span>统一多模态模型期望<span style="color: rgb(100,37,208); background-color: inherit">构建一种单一架构，能够同时理解和生成多种模态的数据</span>，用于以统一的方式处理多样化的输入形式，例如<span style="color: rgb(220,155,4); background-color: inherit">文本、图像、视频、音频</span>，并生成一种或多种模态的输出。典型的统一多模态框架可抽象为三个核心组成部分：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">特定模态的编码器</span>**：用于将不同输入模态映射到统一的表示空间
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">模态融合主干网络</span>**：用于整合来自多个模态的信息并支持跨模态推理
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">特定模态的解码器</span>**：用于在目标模态中生成输出，例如<span style="color: rgb(220,155,4); background-color: inherit">文本生成或图像合成</span>

![](../../images/视觉多模态讲义（下）-image-133.png)

这一章主要聚焦于<span style="color: rgb(100,37,208); background-color: inherit">支持视觉-语言理解与生成的统一多模态模型</span>，&#x5373;**<span style="color: rgb(100,37,208); background-color: inherit">同时接受图像和文本作为输入，并能生成文本或图像作为输出的模型</span>**。现有的统一模型大致可分为三大类：<span style="color: rgb(216,57,49); background-color: inherit">扩散模型 </span>**<span style="color: rgb(216,57,49); background-color: inherit">diffusion models </span>`(a)`**、<span style="color: rgb(216,57,49); background-color: inherit">自回归模型 </span>**<span style="color: rgb(216,57,49); background-color: inherit">autoregressive models </span>`(b)`**&#x4EE5;及<span style="color: rgb(216,57,49); background-color: inherit">融合自回归+扩散模型 </span>**<span style="color: rgb(216,57,49); background-color: inherit">fused AR + diffusion models </span>`(c)`**。

对&#x4E8E;**<span style="color: rgb(100,37,208); background-color: inherit">自回归模型</span>**，可以进一步根据其模态编码方法将其细分为四种子类型：<span style="color: rgb(100,37,208); background-color: inherit">基于像素的编码（pixel-based encoding）</span>**`(b-1)`**<span style="color: rgb(100,37,208); background-color: inherit">、基于语义的编码（semantic-based encoding）</span>**`(b-2)`**<span style="color: rgb(100,37,208); background-color: inherit">、基于可学习 Query 的编码（learnable query-based encoding）</span>**`(b-3)`**<span style="color: rgb(100,37,208); background-color: inherit">以及混合编码（hybrid encoding）</span>**`(b-4)` `(b-5)`**。这些不同的编码策略代表了处理视觉与文本数据的不同方式，从而在多模态表征的整合程度与灵活性方面表现出差异。

**<span style="color: rgb(100,37,208); background-color: inherit">融合自回归+扩散模型</span>**&#x5219;根据模态编码方式分为两类：<span style="color: rgb(100,37,208); background-color: inherit">基于像素的编码 </span>**`(c-1)`&#x20;**<span style="color: rgb(100,37,208); background-color: inherit">和混合编码 </span>**`(c-2)`**。这类模型结合了自回归与扩散方法的优势，为实现更统一、高效的多模态生成提供了一种有前景的途径。

## 5.1 <span style="color: rgb(36,91,219); background-color: inherit">Diffusion-based</span>

### 5.1.1 <span style="color: rgb(36,91,219); background-color: inherit">D-DiT</span>

**<span style="color: rgb(216,57,49); background-color: inherit">D-DiT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">ual </span>**<span style="color: rgb(216,57,49); background-color: inherit">Di</span>**<span style="color: rgb(216,57,49); background-color: inherit">ffusion </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformer）</span>采用统一的主干网络，联合建模图像与文本的分布。<span style="color: rgb(100,37,208); background-color: inherit">给定图像</span>$$\mathbf{x}^{(\text{img})}$$<span style="color: rgb(100,37,208); background-color: inherit">和文本</span>$$\mathbf{x}^{(\text{txt})}$$<span style="color: rgb(100,37,208); background-color: inherit">，建模条件分布</span>$$p(\mathbf{x}^{(\text{img})} \mid \mathbf{x}^{(\text{txt})})$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$p(\mathbf{x}^{(\text{txt})} \mid \mathbf{x}^{(\text{img})})$$<span style="color: rgb(100,37,208); background-color: inherit">。前者为</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本到图像生成</span>**<span style="color: rgb(100,37,208); background-color: inherit">，后者构成了多种</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像理解</span>**<span style="color: rgb(100,37,208); background-color: inherit">任务的基础</span>，如图像描述生成和视觉问答。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

![](../../images/视觉多模态讲义（下）-image-148.png)

**D-DiT** 其实和 MM-DiT 比较像，是基于 Transformer 的模型，<span style="color: rgb(100,37,208); background-color: inherit">包含两个分支：一个用于</span>**<span style="color: rgb(100,37,208); background-color: inherit">处理图像 token</span>**<span style="color: rgb(100,37,208); background-color: inherit">，另一个用于</span>**<span style="color: rgb(100,37,208); background-color: inherit">处理文本 token</span>**<span style="color: rgb(100,37,208); background-color: inherit">。在每一层注意力机制中，图像 token 与文本 token 相互进行交叉注意力计算</span>。

> * **<span style="color: rgb(36,91,219); background-color: inherit">图像分支</span>**&#x7684;输出是在文本条件下的速度预测：
>
> $$\dot{\mathbf{x}}_t = \mathbf{v}(\mathbf{x}_t, t)$$
>
> 其中速度场$$\mathbf{v}(\mathbf{x}_t, t) = \dot{\alpha}_t \mathbf{x} + \dot{\sigma}_t \boldsymbol{\epsilon}$$。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">文本分支</span>**&#x7684;输出则是在图像条件下的$$\mathbf{x}^{(\text{txt})}$$预测。标量时间步 Embedding 通&#x8FC7;**`AdaLN`**&#x63A7;制每一层的特征图。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：
>
> 1. 在图像生成过程中，<span style="color: rgb(100,37,208); background-color: inherit">仅将时间步信息</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">输入模型，因为</span>$$\mathbf{x}^{(\text{txt})}_t$$<span style="color: rgb(100,37,208); background-color: inherit">已隐式编码了该信息</span>，即序列中被掩码 token 的比例。
>
> 2. <span style="color: rgb(100,37,208); background-color: inherit">在扩散模型的文本分支之上额外添加了一个带有双向注意力机制的文本编码器</span>。虽然图像分支与文本分支之间的不对称性并非严格必需，但在 DiT 模型之上加入文本编码器能够将许多现有的文本到图像模型，如 SD3 和 FLUX，作为预训练主干适配到 D-DiT 模型中。并且这个文本编码器不应使用因果掩码，因为会破坏掩码扩散过程。
>
> 3. 为降低高分辨率图像建模带来的计算开销，和之前的一些工作一样，将原始像素空间的图像通过 VAE 编码至空间压缩的潜在空间，<span style="color: rgb(100,37,208); background-color: inherit">VAE 使用判别器损失与 KL 散度正则化进行训练</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

作者提出了一种用于图像-文本联合建模的组合训练目标，本质上是一个结合了连续扩散与离散扩散的联合去噪目标。形式上<span style="color: rgb(100,37,208); background-color: inherit">使用</span>**<span style="color: rgb(216,57,49); background-color: inherit">流匹配 </span>**<span style="color: rgb(216,57,49); background-color: inherit">flow matching</span> <span style="color: rgb(100,37,208); background-color: inherit">来学习图像的条件分布，使用</span>**<span style="color: rgb(216,57,49); background-color: inherit">掩码扩散 </span>**<span style="color: rgb(216,57,49); background-color: inherit">masked diffusion</span> <span style="color: rgb(100,37,208); background-color: inherit">来学习文本的条件分布</span>。

在训练过程中，从各自前向扰动过程$$q(\mathbf{x}_t \mid \mathbf{x})$$中采样得到被扰动的样本$$\mathbf{x}_t^{(\text{img})}$$和 $$\mathbf{x}_t^{(\text{txt})}$$。随后为每种模态分别计算扩散损失：

$$L_{\text{image}}=\mathbb{E}_{t,q^{(\text{img})}} \left\| \mathbf{v}_{\theta} \left( \mathbf{x}^{(\text{img})}_{t}, t, \mathbf{x}^{(\text{txt})} \right) - (\boldsymbol{\epsilon} - \mathbf{x}^{(\text{img})}) \right\|_2^2$$

$$L_{\text{text}}= \mathbb{E}_{q^{(\text{txt})}} \left[ -\frac{1}{K}\sum_{i=1}^{K} \log \left( \mathbf{x}_{\theta}(\mathbf{x}_{t_i}^{(\text{txt})}, \mathbf{x}^{(\text{img})}) \cdot \mathbf{x} \right) / t_i \right]$$

> * &#x5728;**<span style="color: rgb(36,91,219); background-color: inherit">文本扩散</span>**&#x4E2D;，<span style="color: rgb(100,37,208); background-color: inherit">对时间步</span>$$t_i$$<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**<span style="color: rgb(100,37,208); background-color: inherit">对偶采样</span>**<span style="color: rgb(100,37,208); background-color: inherit">，将区间</span>$$(\delta, 1]$$<span style="color: rgb(100,37,208); background-color: inherit">均匀离散化为</span>$$K$$<span style="color: rgb(100,37,208); background-color: inherit">个点</span>，其中$$\delta$$是一个很小的数，用于避免数值不稳定。
>
> * &#x5728;**<span style="color: rgb(36,91,219); background-color: inherit">图像扩散</span>**&#x4E2D;，从对数正态分布中采样时间步$$t$$。训练过程中，**<span style="color: rgb(100,37,208); background-color: inherit">条件样本不被扰动</span>**<span style="color: rgb(100,37,208); background-color: inherit">，即在预测文本分布时，图像扩散的时间步始终设为零</span>，反之亦然。

综上所述，整体的双模态训练损失是上述单模态扩散损失的简单加权组合：

$$L_{\text{dual}} = L_{\text{image}} + \lambda_{\text{text}} L_{\text{text}}$$

其中$$\lambda_{\text{text}}$$是一个超参数。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理</span>**

引入三种基于采样的推理方式，可用于不同的视觉-语言任务：

1. **<span style="color: rgb(36,91,219); background-color: inherit">文本到图像生成</span>**

为执行文本引导的图像生成，即$$\mathbf{x} \sim p(\mathbf{x}^{(\text{img})} \mid \mathbf{x}^{(\text{txt})})$$，<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**<span style="color: rgb(100,37,208); background-color: inherit">无分类器引导 </span> <span style="color: rgb(216,57,49); background-color: inherit">CFG</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">C</span>**<span style="color: rgb(216,57,49); background-color: inherit">lassifier-</span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">ree </span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">uidance）</span> <span style="color: rgb(100,37,208); background-color: inherit">从条件分布</span>$$p(\mathbf{x}^{(\text{img})}_t \mid \mathbf{x}^{(\text{txt})})$$<span style="color: rgb(100,37,208); background-color: inherit">中采样</span>。这相当于对速度预测进行重加权：

$$\tilde{\mathbf{v}}_t = s \, \mathbf{v}_{\theta} \left( \mathbf{x}^{(\text{img})}_{t}, t, \mathbf{x}^{(\text{txt})} \right) + (1 - s) \, \mathbf{v}_{\theta} \left( \mathbf{x}^{(\text{img})}_{t}, t, \emptyset \right)$$

其中$$s$$是控制引导强度的超参数，$$\emptyset$$是合适的空 Embedding，例如空文本的 Embedding。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像到文本生成</span>**

为从条件分布中采样文本，使&#x7528;**`ancestral sampling`**，通过代入预测$$\mathbf{x} \approx \mathbf{x}_\theta(\mathbf{x}^{(\text{txt})}_t, \mathbf{x}^{(\text{img})}; t = 0)$$来从后验分布$$q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x})$$中采样。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像-文本填空</span>**

在某些任务中，同时提供文本条件信息和图像条件信息，例如在**视觉问答&#x20;**&#x56;QA 任务中，系统会同时给出一张图像和一个相关问题。在此类场景下希望采样$$\mathbf{x} \sim p(\mathbf{x}^{(\text{answer})} \mid \mathbf{x}^{(\text{img})}, \mathbf{x}^{(\text{question})})$$。

为完成此任务，<span style="color: rgb(100,37,208); background-color: inherit">将问题的扩散先验初始化为掩码 token，并利用文本扩散模型强大的文本填空能力，通过对条件分布进行采样来补全文本序列</span>。在整个采样过程中，问题文本 token 保持固定不变，如右图。

![](../../images/视觉多模态讲义（下）-image-147.png)

### 5.1.2 <span style="color: rgb(36,91,219); background-color: inherit">MMaDA</span>

![](../../images/视觉多模态讲义（下）-image-149.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：预训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Tokenization 分词</span>**

为构建一个能够同时处理文本与视觉数据的统一建模范式，**MMaDA&#x20;**<span style="color: rgb(100,37,208); background-color: inherit">在两种模态上采用一致的离散分词策略，使模型能够在单一建模目标下运行，即预测被掩码的离散 token</span>。

> * 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">文本分词</span>**，采&#x7528;**`LLaDA`**&#x7684; tokenizer
>
> * 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">图像分词</span>**，利&#x7528;**`Show-o`**&#x4E2D;采用的预训练图像量化器，其基&#x4E8E;**`MAGVIT-v2`**&#x67B6;构，将原始图像像素转换为离散语义 token 序列：<span style="color: rgb(100,37,208); background-color: inherit">给定一个尺寸为</span>$$H \times W$$<span style="color: rgb(100,37,208); background-color: inherit">的输入图像，编码器生成一个尺寸为</span>$$\frac{H}{f} \times \frac{W}{f}$$<span style="color: rgb(100,37,208); background-color: inherit">的 token 映射，其中</span>$$f$$<span style="color: rgb(100,37,208); background-color: inherit">表示下采样因子</span>。在 MMaDA 中采用下采样因子$$f = 16$$和 Codebook 大&#x5C0F;**`8192`**，可以将一张$$512 \times 512$$像素的图像转换为长度为 $$32 \times 32 = 1024$$的离散 token 序列。这些转换后的离散图像 token 被用于理解和生成两类建模任务。

2. **<span style="color: rgb(36,91,219); background-color: inherit">目标函数</span>**

近期的统一多模态框架一般会将多种建模目标，例如<span style="color: rgb(220,155,4); background-color: inherit">自回归生成和基于扩散模型的去噪整合到单一架构中，以支持联合理解与生成任务</span>。但这些方法通常引入复杂的混合机制，阻碍了模型效率与一致性。**MMaDA&#x20;**&#x63D0;出一种简化的框架，<span style="color: rgb(46,161,33); background-color: inherit">不仅降低了架构复杂度，还引入了一个统一的扩散目标，在共享的概率建模公式下对视觉与文本模态进行联合建模</span>。通过在不同模态间对齐噪声添加与语义恢复过程，在预训练阶段实现了更有效的跨模态交互，从而促进异构数据源的无缝融合。

![](../../images/视觉多模态讲义（下）-image-143.png)

具体来说，**MMaDA&#x20;**&#x53EF;以表述为一&#x4E2A;**`mask token predictor`**，这里的 token 指图像和文本 token，即<span style="color: rgb(100,37,208); background-color: inherit">一个参数化模型</span>$$p_\theta(\cdot \mid x_t)$$<span style="color: rgb(100,37,208); background-color: inherit">，它以</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">为输入，并同时预测所有被掩码的 token。</span>**<span style="color: rgb(100,37,208); background-color: inherit">模型通过仅在被掩码的图像和文本 token 上计算的统一交叉熵损失进行训练</span>**：

$$\mathcal{L}_{\text{unify}}(\theta) = -\mathbb{E}_{t, x_0, x_t} \left[ \frac{1}{t} \sum_{i=1}^{L} \mathbb{I}[x_t^i = \texttt{[MASK]}] \log p_\theta(x_0^i \mid x_t) \right]$$

其中$$x_0$$为真实值，时间步$$t$$在区间$$[0, 1]$$上均匀采样，$$x_t$$是通过对$$x_0$$应用前向扩散过程得到的。$$\mathbb{I}[\cdot]$$表示指示函数，确保损失仅在被掩码的 token 上计算。

* **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：混合长思维链微调</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">冷启动的长链思维数据构建</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">MMaDA </span>**<span style="color: rgb(100,37,208); background-color: inherit">构建了一个涵盖三项核心任务的小型但高质量的长 CoT 轨迹数据集：</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本推理</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">多模态推理</span>**<span style="color: rgb(100,37,208); background-color: inherit">以及</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像生成</span>**，通过以下原则实现对预训练 MMaDA 模型的稳定后训练：

> * **<span style="color: rgb(36,91,219); background-color: inherit">统一的 CoT 格式</span>**：多模态模型面临的一个关键挑战是不同任务输出格式不同，例如<span style="color: rgb(220,155,4); background-color: inherit">文本生成和图像生成，输出格式差异很大</span>。MMaDA 提出一种任务无关的 CoT 格式：
>
> $$\texttt{|<special\_token>| <reasoning\_process> |<special\_token>| <result>}$$
>
> 其&#x4E2D;**`<reasoning_process>`**&#x7F16;码了最终输出前的逐步推理轨迹。这种统一结构填补了模态特定输出之间的鸿沟，并促进任务间知识迁移。例如，**<span style="color: rgb(220,155,4); background-color: inherit">增强的文本推理能力可通过将语义逻辑与视觉合成对齐，直接提升生成图像的真实感</span>**。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多样性、复杂性与准确性</span>**：MMaDA 利用开源的 LLM 和 VLM 在各类任务中生成多样化的推理轨迹。为保证质量，使用 SOTA 模型作为验证器，过滤掉不准确或浅层的推理，仅保留高质量、长形式的 CoT 样本。这里明确面向以下两类任务：
>
>   1. 推理密集型任务，如<span style="color: rgb(220,155,4); background-color: inherit">数学问题求解</span>
>
>   2. 具备世界知识感知的文生图生成，其中事实一致性至关重要

* **<span style="color: rgb(36,91,219); background-color: inherit">混合长链思维微调</span>**

有了上面统一的扩散架构与概率建模，**MMaDA&#x20;**<span style="color: rgb(100,37,208); background-color: inherit">开发了一种</span>**<span style="color: rgb(100,37,208); background-color: inherit">混合任务的长 CoT 微调策略</span>**<span style="color: rgb(100,37,208); background-color: inherit">，以联合优化模型在异构任务上的表现</span>。该方法不仅增强了任务特定能力，还为后续的强化学习（RL）阶段提供了强有力的初始化。训练过程包含以下步骤：



**<span style="color: rgb(222,120,2); background-color: inherit">Prompt 保留与 token 掩码</span>**：保留原始 Prompt $$p_0$$，并对结果部分$$x_0$$中的 token 独立进行掩码，记为$$r_t$$。

![](../../images/视觉多模态讲义（下）-image-146.png)

**<span style="color: rgb(222,120,2); background-color: inherit">输入拼接·与损失计算</span>**：将拼接后的输入$$[p_0, r_t]$$输入到预训练的 mask predictor 中计算损失。这使得模型能够利用 prompt 和加噪结果中的上下文信息，重建被掩码区域$$r_0$$。

目标函数定义如下：

$$\mathcal{L}_{\text{Mixed-SFT}} = -\mathbb{E}_{t, p_0, r_0, r_t} \left[ \frac{1}{t} \sum_{i=1}^{L'} \mathbb{I}[r_t^i = \texttt{[MASK]}] \log p_\theta(r_0^i \mid p_0, r_t) \right]$$

其中$$L'$$表示序列长度。此处，$$[p_0, r_0]$$与$$[p_0, r_t]$$分别对应干净数据$$x_0$$及其含噪版本$$x_t$$。这个公式确保模型在恢复被掩码 token 的同时，保持与原始 prompt 及任务特定推理逻辑的一致性。

* **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：统一强化学习后训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">UniGRPO</span>**

![](../../images/视觉多模态讲义（下）-image-144.png)

通过混合长链思维微调，MMaDA 已展现出在最终输出前生成统一且连贯推理链的能力。<span style="color: rgb(100,37,208); background-color: inherit">为进一步提升其在知识密集型任务及复杂推理/生成场景中的表现，作者提出 </span>**<span style="color: rgb(100,37,208); background-color: inherit">UniGRPO</span>**<span style="color: rgb(100,37,208); background-color: inherit">，专为扩散基础模型量身定制的新型基于策略梯度的强化学习算法</span>。其构建了一个以扩散模型为核心的 RL 训练框架，能够跨不同模态与推理范式统一任务特定目标。该方法包含两个核心组成部分：

> * 基于扩散模型的 RL 的统一数学形式化
>
> * 多样化的奖励建模，以使策略梯度与任务特定奖励对齐

**<span style="color: rgb(222,120,2); background-color: inherit">将自回归 GRPO 适配到扩散模型的难点</span>**

原始 GRPO 依赖于计算 token 级别的对数似然$$\pi_\theta(o_{i,t} \mid q, o_{i,<t})$$以及序列级别的概率$$\pi_\theta$$与$$\pi_{\text{ref}}$$。在自回归大语言模型中，这些指标可通过生成的链式法则高效推导。然而，扩散模型有三个关键的难点：

> * **<span style="color: rgb(36,91,219); background-color: inherit">局部掩码依赖性</span>**：token 级别的对数似然$$\log \pi_\theta(o_{i,t} \mid q, o_{i,<t})$$仅在扩散过程中被掩码的区域有效，而 AR 模型中所有 token 均有效
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">掩码比例敏感性</span>**：为近似策略分布$$\pi_\theta$$，必须对回答段采样一个统一的掩码比例，因为扩散动态依赖于掩码模式
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">非自回归的序列级似然</span>**：由于扩散模型缺乏自回归链式法则，序列级对数似然无法直接由 token 级概率累加得到

现有方法采用次优策略应对上述问题：

> * **`LLaDA`**&#x901A;过对大量掩码比例进行蒙特卡洛采样，在 on-policy RL 中带来高昂计算开销
>
> * **`d1`**&#x56FA;定掩码比例并对问题部分随机掩码，降低了噪声多样性，并忽略了扩散模型多步去噪的本质

**<span style="color: rgb(222,120,2); background-color: inherit">扩散 GRPO 的统一形式化</span>**

为克服上述限制，作者提出 **UniGRPO**，专为扩散架构设计的计算高效近似算法。给定一批针对Query $$q$$的回答$$\{o_i\}_{i=1}^G$$，每个回答在梯度更新期间保持固定，以确保策略评估的稳定性。UniGRPO 的三个关键设计如下：

> * **<span style="color: rgb(36,91,219); background-color: inherit">结构化加噪策略</span>**
>
> 对每个$$o_i$$，从区间$$[0, 1]$$均匀采样一个掩码比例$$p_i$$，并通过将 token 替换&#x4E3A;**`[MASK]`**&#x6784;造扰动版本$$\tilde{o}_{i,p}$$。$$p_i$$的随机种子在不同梯度步之间变化。这个策略<span style="color: rgb(100,37,208); background-color: inherit">在保留随机性的同时，使模型暴露于扩散去噪过程的不同阶段，从几乎完全掩码到几乎完全去噪的答案</span>。由此，UniGRPO 能从多步去噪信息中学习，这与扩散模型的传统训练方法一致，并充分利用其多步生成能力。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">高效的对数似然近似</span>**
>
> 将扰动分布下的期望 token 级对数似然定义为：
>
> $$\pi'_\theta(o_{i,t} \mid q, o, \tilde{o}_i) = \mathbb{E}_{p_i \sim [0,1]} \left[ \mathbb{I}[o_{i,t,p} = \texttt{[MASK]}] \log p_\theta(o_{i,t,p} \mid q) \right]$$
>
> 序列级对数似然则通过对被掩码 token 取平均来近似：
>
> $$\pi'_\theta = \frac{1}{M} \sum_{o_{i,t} \in \mathcal{M}} \log p_\theta(o_{i,t} \mid q)$$
>
> 其中$$M$$表示被掩码 token 的数量。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">策略梯度目标</span>**
>
> token 级奖励计算方法是当前策略与旧策略似然之比：
>
> $$r'_{i,t}(\theta) = \frac{\pi'_\theta(o_{i,t} \mid q, o, \tilde{o}_i)}{\pi'_{\text{old}}(o_{i,t} \mid q, o, \tilde{o}i)}$$
>
> 最终 UniGRPO 目标结合了裁剪的替代奖励与 KL 正则项：
>
> $$\begin{aligned}
> J_{\text{UniGRPO}}(\theta) = \mathbb{E}_{\substack{(q,a)\sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q), \{p_i \in [0,1]\}_{i=1}^G}} \Bigg[
> \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
> \min \Big( & r'_{i,t}(\theta) \hat{A}_{i,t}, \\
> & \text{clip}\big(r'_{i,t}(\theta), 1 - \varepsilon, 1 + \varepsilon\big) \hat{A}_{i,t} \Big) 
> \beta D_{\text{KL}}(\pi'_\theta \,\|\, \pi'_{\text{ref}})
>   \Bigg]
>   \end{aligned}$$
>
> 其中$$\hat{A}_{i,t}$$表示优势估计，$$\varepsilon$$控制裁剪范围，$$\beta$$平衡 KL 散度惩罚项。

通过这个设计，<span style="color: rgb(100,37,208); background-color: inherit">UniGRPO 捕捉了扩散模型本质的多步去噪动态。它允许模型在多样化的掩码条件下预测答案，同时保留输入的自然结构，从而避免了 LLaDA 的计算低效和 d1 的过度简化预测等问题</span>。UniGRPO 的训练流程如下：

![](../../images/视觉多模态讲义（下）-image-142.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">多样化奖励建模</span>**

可以将 UniGRPO 的优化目标进一步简化为：

$$J_{\text{UniGRPO}}(\theta) = \mathbb{E}_{o \sim \pi_\theta(\cdot|q)} \left[ F(R^{\text{Uni}}(o)) - \beta P(o) \right]$$

其中$$R_{\text{Uni}}(o)$$表示模型生成回答$$o$$所获得的奖励，$$P(\cdot)$$为惩罚项，即 KL 散度。<span style="color: rgb(100,37,208); background-color: inherit">这是一个统一的基于规则的奖励系统，其中</span>$$R_{\text{Uni}}(\cdot)$$<span style="color: rgb(100,37,208); background-color: inherit">可根据任务实例化为不同类型的奖励</span>。为满足不同任务的多样化需求，在统一公式下定义了一系列奖励，为各任务分支提供定制化的 RL 优化方向。主要采用以下三类奖励：

![](../../images/视觉多模态讲义（下）-image-145.png)

> * **<span style="color: rgb(36,91,219); background-color: inherit">文本推理奖励 Textual Reasoning Rewards</span>**
>
> 在 GSM8K 数据集的训练集上应用 UniGRPO，并定义复合奖励：若答案正确，给&#x4E88;**<span style="color: rgb(100,37,208); background-color: inherit">正确性奖励 Correctness Reward</span>&#x20;**$$2.0$$分；若回答符合预设格&#x5F0F;**`<think>...</think>`**，额外给&#x4E88;**<span style="color: rgb(100,37,208); background-color: inherit">格式奖励 Format Reward</span>** $$0.5$$分。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态推理奖励 Multimodal Reasoning Rewards</span>**
>
> 对于 GeoQA 和 CLEVR 等数学任务，采用与文本推理相同的正确性与格式奖励。对于图像描述任务，引入 **<span style="color: rgb(100,37,208); background-color: inherit">CLIP Reward</span>**：
>
> $$0.1 \cdot \text{CLIP}(\text{image}, \text{text})$$
>
> 将衡量图文对齐度的原始 CLIP 分数缩放$$0.1$$倍，以平衡其影响。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">文生图生成奖励 Text-to-Image Generation Rewards</span>**
>
> 对于图像生成任务，同样使用 CLIP 奖励评估图文语义对齐，并加入反映人类偏好&#x7684;**<span style="color: rgb(100,37,208); background-color: inherit">图像奖励 Image Reward</span>**。两类奖励均乘以$$0.1$$的缩放因子，以确保在优化过程中贡献均衡。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理阶段</span>**

![](../../images/视觉多模态讲义（下）-image-139.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">文本生成的半自回归采样</span>**

对于文本生成，<span style="color: rgb(100,37,208); background-color: inherit">采用 LLaDA 提出的半自回归去噪策略，将自回归解码与基于扩散的去噪相结合</span>。即输出序列被划分为多个块，并从左至右依次生成。在每个块内，对所有被掩码位置计算 logits，并随机或基于置信度分数选择一部分 token 进行去噪。掩码调度采用线性计划，与 LLaDA 一致。该去噪过程重复若干步。

在评估中，<span style="color: rgb(100,37,208); background-color: inherit">将总序列长度设为</span>$$N = 1024$$<span style="color: rgb(100,37,208); background-color: inherit">，执行</span>$$N/2 = 512$$<span style="color: rgb(100,37,208); background-color: inherit">步去噪。序列被划分为每块</span>$$64$$<span style="color: rgb(100,37,208); background-color: inherit">个 token。每一步中，在当前块内选择置信度最低的 2 个 token 进行去掩码，无论其位置如何</span>。一旦某一块中所有 token 均被去噪，流程即进入下一块。

![](../../images/视觉多模态讲义（下）-image-140.png)

根据这个定性对比可以知道，<span style="color: rgb(100,37,208); background-color: inherit">半自回归去噪策略倾向于生成更复杂、细节更丰富的描述；而非自回归的固定长度生成往往产生极短的回答</span>。这与 LLaDA 的结论是一致的。对于经过指令微调的模型，由于训练过程中包含大&#x91CF;**`|EOS|`** token，若不划分块而直接应用最低置信度重掩码策略，会导致生成句子&#x4E2D;**`|EOS|`**&#x51FA;现频率异常偏高。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像生成的并行非自回归采样</span>**

对于图像生成，<span style="color: rgb(100,37,208); background-color: inherit">采用低置信度重掩码策略，并遵循余弦噪声调度，与 MAGVIT-v2 的设置一致</span>。与文本生成不同，这里不采用半自回归方式，而是<span style="color: rgb(100,37,208); background-color: inherit">将整个输出序列视为单一生成块</span>。评估时，生成长度为$$1024$$的序列，对应$$512 \times 512$$分辨率的图像。去噪过程包含 50 个时间步，并应用无分类器引&#x5BFC;**`classifier-free guidance`**，引导尺度设为$$3.5$$。

### 5.1.3 **<span style="color: rgb(36,91,219); background-color: inherit">FUDOKI</span>**

**FUDOKI&#x20;**&#x901A;过<span style="color: rgb(216,57,49); background-color: inherit">离散流匹配</span>**`discrete flow matching`**&#x8FD9;一视角，统一了视觉与语言，能够在视觉和文本模态之间实现感知与生成的统一方法。

* **<span style="color: rgb(36,91,219); background-color: inherit">数学背景：离散流匹配</span>**

一般来说，离散流匹配的目标是从一个已知的源分布$$p(x)$$出发，逼近目标真实数据分布$$q(x)$$，其中$$x = (x^1, x^2, \dots, x^D)$$属于离散空间$$\mathcal{S} = \mathcal{T}^D$$。这里$$D$$表示离散变量的数量，$$\mathcal{T} = [K] = \{1, 2, \dots, K\}$$表示每个变量可能取值的有限集合。

* **<span style="color: rgb(36,91,219); background-color: inherit">概率路径 Probability Paths</span>**

给定定义在有限状态空间$$\mathcal{S}$$上的源分布$$p(x)$$和目标分布$$q(x)$$，<span style="color: rgb(100,37,208); background-color: inherit">离散流匹配引入一组以时间索引的概率分布</span>$$\{p_t(x)\}_{t \in [0,1]}$$<span style="color: rgb(100,37,208); background-color: inherit">，用以描述从</span>$$p$$<span style="color: rgb(100,37,208); background-color: inherit">到</span>$$q$$<span style="color: rgb(100,37,208); background-color: inherit">的平滑变换过程，称为</span>**<span style="color: rgb(100,37,208); background-color: inherit">概率路径</span>**。每个$$p_t(x)$$构造如下：

$$p_t(x) \coloneqq \sum_{x_1 \in \mathcal{S}} p_t(x \mid x_1) \, q(x_1)$$

其中条件分布按维度分解为：

$$p_t(x \mid x_1) \coloneqq \prod_{i=1}^D p_t(x_i \mid x^i_1)$$

此处，每个$$p_t(x^i \mid x^i_1)$$定义了从基础分布$$p(x^i)$$到点质量分布$$\delta_{x^i_1}(x^i)$$的单变量插值，其中

$$\delta_{x^i_1}(x^i) = 
\begin{cases}
1, & \text{if } x^i = x^i_1 \\
0, & \text{otherwise}
\end{cases}$$

这种插值的一种常见设计是<span style="color: rgb(216,57,49); background-color: inherit">混合路径</span>**`mixture path`**，通过一个依赖于时间的调度函数$$\kappa_t(x_i^1) \in [0, 1]$$定义：

$$p_t(x^i \mid x^i_1) = (1 - \kappa_t(x^i_1)) \, p(x^i) + \kappa_t(x^i_1) \, \delta_{x^i_1}(x^i)$$

其中满足边界条件$$\kappa_0(\cdot) = 0$$、$$\kappa_1(\cdot) = 1$$。<span style="color: rgb(100,37,208); background-color: inherit">当基础分布取为掩码 token 的点质量分布，即</span>$$p(x_i) = \delta_m(x_i)$$<span style="color: rgb(100,37,208); background-color: inherit">，</span>$$m$$<span style="color: rgb(100,37,208); background-color: inherit">表示掩码 token 时，这个路径退化为</span>**<span style="color: rgb(100,37,208); background-color: inherit">掩码数据构造方式</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">概率速度 Probability Velocities</span>**

为了模拟沿预设路径$$\{p_t(x)\}_{t \in [0,1]}$$演化的生成过程，考虑一个定义在离散空间$$\mathcal{S}$$上&#x7684;***连续时间马尔可夫链*`CTMC`**$$\{x_t\}_{t \in [0,1]}$$，使得$$x_t \sim p_t$$。再具体点就是，<span style="color: rgb(100,37,208); background-color: inherit">通过</span>**<span style="color: rgb(100,37,208); background-color: inherit">概率速度</span>**$$u_t^i(\cdot, x_t)$$<span style="color: rgb(100,37,208); background-color: inherit"> 来描述该 CTMC，它刻画了</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">在第</span>$$i$$<span style="color: rgb(100,37,208); background-color: inherit">个 token 位置上的概率变化速率</span>。这与连续流匹&#x914D;**`Flow Matching`**&#x4E2D;的速度场概念类似。离散流匹配中的形式化定义如下：

> **<span style="color: rgb(222,120,2); background-color: inherit">定义</span>**：若对任意$$t \in [0, 1)$$及任意样本$$x_t \sim p_t$$，其更新后的样本$$x_{t+h}$$满足：对每个坐标$$i$$，有 &#x20;
>
> $$x^i_{t+h} \sim \delta_{x^i_t}(\cdot) + h \, u_t^i(\cdot, x_t)$$
>
> 且整体满足$$x_{t+h} \sim p_{t+h} + o(h)$$，即当$$h \to 0$$时，称概率速度$$u_t$$**生成**了概率路径$$p_t$$。

此外，概率速度$$u_t$$还需满足以下**速率条件**：

$$\sum_{x^i \in [K]} u_t^i(x^i, z) = 0, \quad u_t^i(x^i, z) \geq 0 \quad \forall i \in [D], \; x^i \neq z^i$$

以确保更新后的$$x^i_{t+h}$$可从一个合法的概率分布中采样。还有一个知识点是离散流匹配中的<span style="color: rgb(216,57,49); background-color: inherit">连续性方程</span>**`Continuity Equation`**，或者称为 Kolmogorov 前向方程，用于描述状态概率的时间导数$$\dot{p}_t(x)$$，其中$$x \in \mathcal{S}$$：

$$\dot{p}_t(x) + \mathrm{div}_x (p_t u_t) = 0$$

其中散度项定义为：

$$\mathrm{div}_x (p_t u_t) = \sum_{z \in \mathcal{S}} \sum_{i=1}^D \delta_x(z^{i}) \left[ p_t(x) u_t^i(z^i, x) - p_t(z) u_t^i(x^i, z) \right]$$

它<span style="color: rgb(100,37,208); background-color: inherit">衡量了状态</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">的</span>**<span style="color: rgb(100,37,208); background-color: inherit">总流出通量</span>**$$x \to z$$<span style="color: rgb(100,37,208); background-color: inherit">减去</span>**<span style="color: rgb(100,37,208); background-color: inherit">总流入通量</span>**$$z \to x$$。此处，

$$\delta_x(z^i) = \prod_{j \neq i} \delta_{x_j}(z_j)$$

表示<span style="color: rgb(100,37,208); background-color: inherit">仅当</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$z$$**<span style="color: rgb(100,37,208); background-color: inherit">仅在第</span>**$$i$$**<span style="color: rgb(100,37,208); background-color: inherit">个坐标上不同</span>**<span style="color: rgb(100,37,208); background-color: inherit">时，才计入该坐标的通量</span>。

$$\dot{p}_t(x) + \mathrm{div}_x (p_t u_t) = 0$$表明：状态$$x$$处的概率变化率等于该处净剩余的概率通量$$p_t u_t$$。若连续性方程成立，则$$u_t$$即为按上述定义所述生成概率路径$$p_t$$的有效速度场。

* **<span style="color: rgb(36,91,219); background-color: inherit">FUDOKI 中的离散流匹配</span>**

考虑由<span style="color: rgb(216,57,49); background-color: inherit">离散度量</span>**`discrete metrics`**&#x6240;诱导的概率路径：

给定一个距离函数$$d : \mathcal{T} \times \mathcal{T} \to \mathbb{R}_{\geq 0}$$，满足$$d(x^i, x^i_1) = 0$$当且仅当$$x^i = x^i_1$$，可以通过以下方式定义条件分布路径：

$$p_t(x^i \mid x^i_1) = \mathrm{softmax}\left( -\beta_t \cdot d(x^i, x^i_1) \right)$$

其中$$\beta_t : [0, 1] \to \mathbb{R}_{\geq 0}$$是一个单调递增的时间调度函数，边界条件为$$\beta_0 = 0$$、$$\beta_1 = \infty$$。当$$t = 0$$时，该分布为均匀分布；当$$t \to 1$$时，分布收敛至以$$x^i_1$$为中心的狄拉克 delta 函数。与基于掩码的概率路径$$p_t(x^i \mid x^i_1) = (1 - \kappa_t(x^i_1)) \, p(x^i) + \kappa_t(x^i_1) \, \delta_{x^i_1}(x^i)$$相比，<span style="color: rgb(100,37,208); background-color: inherit">这种</span>**<span style="color: rgb(100,37,208); background-color: inherit">度量诱导的概率路径</span>**<span style="color: rgb(100,37,208); background-color: inherit">定义了一种语义上更有意义的变换：当将</span>$$d(\cdot, \cdot)$$<span style="color: rgb(100,37,208); background-color: inherit">设为衡量 token Embedding 距离的函数时，随着</span>$$t \to 1$$<span style="color: rgb(100,37,208); background-color: inherit">，不仅目标 token </span>$$x^i_1$$<span style="color: rgb(100,37,208); background-color: inherit">的概率增加，与其语义相近的 tokens 的概率也会相应提升</span>。

在定义了上述度量诱导的概率路径后，便可以通过最小化动&#x80FD;**`kinetic energy`**&#x6765;获得概率速度。也就是说<span style="color: rgb(100,37,208); background-color: inherit">希望最小化通量</span>$$p_t u_t$$<span style="color: rgb(100,37,208); background-color: inherit">的大小，以确保沿概率路径的变换足够平滑</span>。同时，所得到的速度还需满足若干约束条件，包括连续性方程$$\dot{p}_t(x) + \mathrm{div}_x (p_t u_t) = 0$$、不同状态间通量的非负性$$\sum_{x^i \in [K]} u_t^i(x^i, z) = 0$$，以及$$p$$和$$q$$的边界条件。由此，$$p_t(x^i \mid x^i_1) = \mathrm{softmax}\left( -\beta_t \cdot d(x^i, x^i_1) \right)$$对应的动能最优速度可表示为：

$$u_t^i(x^i, z \mid x_1) = p_t(x^i \mid x^i_1) \, \dot{\beta}_t \, \big[ d(z^i, x^i_1) - d(x^i, x^i_1) \big]_+$$

其中$$[\cdot]_+ = \max\{\cdot, 0\}$$为 ReLU 算子，$$\dot{\beta}_t$$是$$\beta_t$$对$$t$$的导数。<span style="color: rgb(100,37,208); background-color: inherit">对于第</span>$$i$$<span style="color: rgb(100,37,208); background-color: inherit">个坐标</span>$$z^i \in \mathcal{T}$$<span style="color: rgb(100,37,208); background-color: inherit">，该速度确保概率质量仅当 </span>$$x^i$$<span style="color: rgb(100,37,208); background-color: inherit">比</span>$$z^i$$<span style="color: rgb(100,37,208); background-color: inherit">更接近目标</span>$$x^i_1$$<span style="color: rgb(100,37,208); background-color: inherit">，即</span>$$d(x^i, x^i_1) < d(z^i, x^i_1)$$<span style="color: rgb(100,37,208); background-color: inherit">时，才从状态</span>$$z^i$$<span style="color: rgb(100,37,208); background-color: inherit">流向</span>$$x^i$$<span style="color: rgb(100,37,208); background-color: inherit">。因此，整个流过程单调地朝向</span>$$x^i_1$$<span style="color: rgb(100,37,208); background-color: inherit">推进</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

**FUDOKI** 基&#x4E8E;**`Janus-1.5B`**&#x67B6;构，并进行了少量适配以支持统一的视觉-语言离散流建模：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">采用全注意力掩码</span>**&#x800C;非标准因果掩码，使所有 token 可相互关注，从而更好地捕捉全局上下文
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">对输出 logits 应用位置偏移操作</span>**，使模型尽可能继承自回归多模态大语言模型的下一 token 预测能力
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">不引入显式的时间 Embedding 层</span>**&#x6765;指示输入中的噪声水平，这与连续扩散模型不同。在所定义的度量诱导概率路径$$p_t(x^i \mid x^i_1) = \mathrm{softmax}\left( -\beta_t \cdot d(x^i, x^i_1) \right)$$下，模型可<span style="color: rgb(100,37,208); background-color: inherit">隐式地从含噪输入中推断时间步</span>，从而在实验中实现更快的适应

其余架构与 Janus-1.5B 完全一致：

> * **<span style="color: rgb(36,91,219); background-color: inherit">文本模态</span>**：使用词表大小&#x4E3A;**`102,400`**&#x7684; tokenizer
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图像模态</span>**：理解与生成采用解耦的处理路径：
>
>   * **<span style="color: rgb(36,91,219); background-color: inherit">理解路径</span>**：采用语义编码器 SigLIP 提取高维图像特征，并通过一个适配器 adaptor 将其重塑并映射到 LLM 的输入空间
>
>   * **<span style="color: rgb(36,91,219); background-color: inherit">生成路径</span>**：按照 LlamaGen 的方法，使用像素编码器-解码器将图像转换为离散 token，图像 token 词表大小设&#x4E3A;**`16,384`**。每个图像 token 嵌入再经由一个<span style="color: rgb(100,37,208); background-color: inherit">生成适配器</span>**&#x20;**&#x67;eneration adaptor 转换为输入特征后送入 LLM
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">输出端</span>**，使用两个输出头：<span style="color: rgb(100,37,208); background-color: inherit">文本头、图像头</span>，二者将 Transformer 的输出分别转换为离散类别分布。推理时，根据目标模态选择相应的输出头。

与先前基于 AR 或 Diffusion 的多模态对比如下：

![](../../images/视觉多模态讲义（下）-image-141.png)

![](../../images/视觉多模态讲义（下）-image-137.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

模型初始化自预训练的 Janus-1.5B 权重，并在收集的数据集上进一步微调，这个数据集包含<span style="color: rgb(100,37,208); background-color: inherit">文本到图像</span>**`T2I`**<span style="color: rgb(100,37,208); background-color: inherit">的</span>**<span style="color: rgb(100,37,208); background-color: inherit">生成任务</span>**<span style="color: rgb(100,37,208); background-color: inherit">和图像到文本</span>**`I2T`**<span style="color: rgb(100,37,208); background-color: inherit">的</span>**<span style="color: rgb(100,37,208); background-color: inherit">理解任务</span>**&#x4E24;类样本。**FUDOKI&#x20;**&#x7684;训练分为两个阶段：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：主要目标是快速重新学习 AR 式 LLM，使其无缝支持离散流匹配范式。<span style="color: rgb(100,37,208); background-color: inherit">仅微调 Transformer 参数，冻结模型其他部分</span>，包括语义编码器和嵌入适配器，以加速收敛并稳定训练
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：在第一阶段基础上，对整个模型进行端到端微调，以提升其在理解和生成任务上的整体性能

在每个训练阶段中，真实目标$$x_1$$从数据分布$$q(\cdot)$$中采样，条件为：

> * **<span style="color: rgb(36,91,219); background-color: inherit">T2I</span>**：文本 prompt
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">I2T</span>**：图像-问题对

其中，**<span style="color: rgb(100,37,208); background-color: inherit">T2I</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">为图像 token 序列，</span>**<span style="color: rgb(100,37,208); background-color: inherit">I2T</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">为文本 token 序列</span>。每一步训练中，时间$$t \in [0, 1]$$从均匀分布采样，含噪序列$$x_t$$根据$$p_t(x^i \mid x^i_1) = \mathrm{softmax}\left( -\beta_t \cdot d(x^i, x^i_1) \right)$$定义的概率路径$$p_t(\cdot \mid x_1)$$采样。**FUDOKI** 将距离函数$$d(\cdot, \cdot)$$设为<span style="color: rgb(100,37,208); background-color: inherit">归一化 token Embedding 之间的 L2 距离</span>，这有助于提高那些在 Embedding 空间中靠近真实 token $$x^i_1$$的 tokens 的采样概率，从而使加噪过程更具语义意义，促进学习。

<span style="color: rgb(100,37,208); background-color: inherit">模型以</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">为输入，预测</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">，并对每个位置输出 token 级别的 logits。训练损失定义为真实序列</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">与模型预测分布之间的期望交叉熵</span>：

$$\mathcal{L}_{\mathrm{CE}}(\theta) = \mathbb{E}_{\substack{t \sim \mathcal{U}[0,1], x_1 \sim q(\cdot), x_t \sim p_t(\cdot \mid x_1)}} \left[ -\sum_{i=1}^{D} \log p^{\theta}_{1|t}(x^i_1 \mid x_t) \right]$$

其中$$p^{\theta}_{1|t}(\cdot \mid x_t)$$表示模型在给定输入$$x_t$$下对第$$i$$个位置预测的类别分布，参数为$$\theta$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理</span>**

推理阶段采用 **Euler 求解器**进行更鲁棒的采样。其模拟连续时间马尔可夫&#x94FE;**`CTMC`**&#x8FC7;程$$(x_t)_{0 \leq t \leq 1}$$。假设当前$$x_t \sim p_t$$，求解器按如下步骤将第$$i$$个坐标从时间$$t$$更新至$$t + h$$：

> 1. 从模型采样$$x^i_1 \sim p^i_{1|t}(\cdot \mid x_t)$$
>
> 2. 计算总条件转移速率：
>
> $$\lambda^i = \sum_{x^i \neq x^i_t} u_t^i(x^i, x^i_t \mid x^i_1)$$
>
> 3. 从均匀分布采样随机变量$$Z^i_{\text{change}} \sim \mathcal{U}[0, 1]$$
>
> 4. 按如下规则采样$$x^i_{t+h}$$：
>
>    * 若$$Z^i_{\text{change}} \leq 1 - e^{-h \lambda^i}$$，则从分布$$\frac{u_t^i(\cdot, x^i_t \mid x^i_1)}{\lambda^i} (1 - \delta_{x^i_t}(\cdot))$$中采样$$x^i_{t+h}$$
>
>    * 否则，令$$x^i_{t+h} = x^i_t$$，其中$$\delta_{x^i_t}(\cdot)$$为以$$x^i_t$$为中心的 delta 函数

这个推理过程可以这样理解：

> * 第二步中的$$\lambda^i$$可解释为：<span style="color: rgb(100,37,208); background-color: inherit">在当前位置</span>$$x^i_t$$<span style="color: rgb(100,37,208); background-color: inherit">，概率质量流向其他状态</span>$$x^i \neq x^i_t$$<span style="color: rgb(100,37,208); background-color: inherit">的</span>**<span style="color: rgb(100,37,208); background-color: inherit">强度</span>**
>
> * 当前 timestep 发生状态变化的概率由阈值$$1 - e^{-h \lambda^i}$$与随机变量$$Z^i_{\text{change}}$$比较决定：$$\lambda^i$$<span style="color: rgb(100,37,208); background-color: inherit">越大，发生跳变的可能性越高</span>
>
> * 若发生跳变，则$$x^i_{t+h}$$从所有其他可能状态中按与$$u_t^i(\cdot, x^i_t \mid x^i_1)$$成正比的分布采样。这意味着<span style="color: rgb(100,37,208); background-color: inherit">更新倾向于将</span>$$x^i_{t+h}$$<span style="color: rgb(100,37,208); background-color: inherit">移向更接近模型预测</span>$$x^i_1$$<span style="color: rgb(100,37,208); background-color: inherit">的状态</span>

总的来说，采样过程具备两大优势：

> 1. 沿概率路径<span style="color: rgb(100,37,208); background-color: inherit">持续优化预测</span>
>
> 2. 在每个 timestep <span style="color: rgb(100,37,208); background-color: inherit">灵活地将 token 调整为语义相近的替代项</span>

![](../../images/视觉多模态讲义（下）-image-136.png)

如上图，**FUDOKI** 与先前基于掩码的离散扩散模型形成鲜明对比：在那些模型中，<span style="color: rgb(216,57,49); background-color: inherit">一旦某个 token 被 unmasked，通常就无法再被修改，即使它包含错误</span>。

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



## 5.3 <span style="color: rgb(36,91,219); background-color: inherit">AR + Diffusion</span>

### 5.3.1 **<span style="color: rgb(36,91,219); background-color: inherit">Transfusion（c-1）</span>**

Transfusion 主要创新在于证明：<span style="color: rgb(100,37,208); background-color: inherit">可以在共享数据和参数的基础上，对不同模态采用不同的损失函数，文本使用</span>**<span style="color: rgb(100,37,208); background-color: inherit">语言建模损失</span>**<span style="color: rgb(100,37,208); background-color: inherit">，图像使用</span>**<span style="color: rgb(100,37,208); background-color: inherit">扩散损失</span>**。

![单一的 Transformer 能够感知、处理并生成所有模态的数据：离散的文本 token 以自回归方式处理，并通过下一 token 预测目标进行训练；连续的图像向量以并行方式联合处理，并通过扩散目标进行训练。特殊的标记 token BOI（图像开始）和 EOI（图像结束）用于分隔不同模态。](../../images/视觉多模态讲义（下）-image-183.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">数据表示</span>**

数据涵盖两种模态：**<span style="color: rgb(100,37,208); background-color: inherit">离散的文本和连续的图像</span>**。

> * 使用 tokenizer 将文本<span style="color: rgb(100,37,208); background-color: inherit">分割为来自固定词汇表的一系列离散 token</span>，每个 token 用一个整数表示
>
> * 图像<span style="color: rgb(100,37,208); background-color: inherit">通过 VAE 编码为潜在空间中的 patch 序列</span>，其中每个 patch 表示为一个连续向量；这些 patch 按从左到右、从上到下的顺序排列，形成一个图像 patch 向量序列
>
> * 对于混合模态样本，在插入文本序列前，用特殊的<span style="color: rgb(216,57,49); background-color: inherit">图像开始</span>**`BOI`**&#x548C;<span style="color: rgb(216,57,49); background-color: inherit">图像结束</span>**`EOI`&#x20;**&#x74;oken 将图像序列包裹起来；由此得到一个同时包含文本 token 的离散元素和图像 patch 的连续元素的单一序列

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

模型的绝大部分参数属于一个 Transformer，这个<span style="color: rgb(100,37,208); background-color: inherit"> Transformer 处理所有序列，接收一个</span>$$\mathbb{R}^d$$<span style="color: rgb(100,37,208); background-color: inherit">中的高维向量序列作为输入，并输出类似维度的向量序列</span>。为了<span style="color: rgb(100,37,208); background-color: inherit">将数据映射到该空间，使用轻量级的模态特定模块</span>，这些模块的参数不共享。 &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">文本</span>**：映射模块是 Embedding 矩阵，<span style="color: rgb(100,37,208); background-color: inherit">将每个输入整数映射到向量空间，并将每个输出向量转换为词汇表上的离散分布</span>
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图像</span>**：探索了两种方法，将$$k \times k$$的局部 patch 向量窗口压缩为单个 Transformer 向量及其逆过程：
>
>   * 一个简单的线性层
>
>   * U-Net 的上采样与下采样模块，如右图



**<span style="color: rgb(36,91,219); background-color: inherit">Attention 机制</span>**

<span style="color: rgb(100,37,208); background-color: inherit">语言模型通常使用因果掩码 causal mask</span>，以便在一次前向-后向传播中高效计算整个序列的损失和梯度，同时避免未来 token 的信息泄露。虽然文本天然具有顺序性，但图像通常不具备，因此<span style="color: rgb(100,37,208); background-color: inherit">图像建模常采用无限制的双向注意力</span>。Transfusion 结合了这两种注意力模式：**<span style="color: rgb(100,37,208); background-color: inherit">对序列中所有元素施加因果注意力，同时在每个图像内部的 patch 元素之间启用双向注意力</span>**。这使得<span style="color: rgb(100,37,208); background-color: inherit">同一图像内的每个 patch 都能关注其他所有 patch，但只能关注序列中先前出现的文本或其他图像的 patch</span>。

![](../../images/视觉多模态讲义（下）-image-184.png)

![](../../images/视觉多模态讲义（下）-image-185.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

<span style="color: rgb(100,37,208); background-color: inherit">对文本 token 的预测应用</span>**<span style="color: rgb(100,37,208); background-color: inherit">语言建模目标</span>**$$\mathcal{L}_{\text{LM}}$$<span style="color: rgb(100,37,208); background-color: inherit">，对图像 patch 的预测应用</span>**<span style="color: rgb(100,37,208); background-color: inherit"> diffusion 目标</span>**$$\mathcal{L}_{\text{DDPM}}$$。语言建模损失按 token 计算，而扩散损失按图像计算，一张图像可能对应序列中的多个 patch 元素。Transfusion 根据扩散过程向每个输入潜在图像$$x_0$$添加噪声$$\epsilon$$，生成$$x_t$$，然后进行 patch 化，并计算图像级别的扩散损失。然后将两个损失简单相加以平衡系数$$\lambda$$进行组合：

$$\mathcal{L}_{\text{Transfusion}} = \mathcal{L}_{\text{LM}} + \lambda \cdot \mathcal{L}_{\text{DDPM}}$$

将离散分布损失与连续分布损失结合，以优化同一个模型。

**<span style="color: rgb(36,91,219); background-color: inherit">VAE 训练</span>**

VAE 训练目标如下：

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_1 + \mathcal{L}_{\text{LPIPS}} + 0.5\,\mathcal{L}_{\text{GAN}} + 0.2\,\mathcal{L}_{\text{ID}} + 0.000001\,\mathcal{L}_{\text{KL}}$$

其中$$\mathcal{L}_1$$ 是像素空间中的 L1 损失；$$\mathcal{L}_{\text{LPIPS}}$$ 是基于 LPIPS 相似度的感知损失；$$\mathcal{L}_{\text{GAN}}$$ 是基于 patch 的判别器损失；$$\mathcal{L}_{\text{ID}}$$ 是基于 MoCo v2 模型内部特征的感知损失；$$\mathcal{L}_{\text{KL}}$$ 是标准的 KL 正则化项，用于促使编码器输出接近标准正态分布。

Transfusion  <span style="color: rgb(100,37,208); background-color: inherit">将 GAN 训练的起始时间（即在损失函数中引入对抗损失）延迟至第 50,000 步，以确保 VAE 首先达到足够好的重建性能</span>，使用的潜在空间维度为 8。

VQ-GAN 的训练目标与 VAE 基本一致，但有一个显著区别： &#x20;

<span style="color: rgb(100,37,208); background-color: inherit">将</span>$$\mathcal{L}_{\text{KL}}$$<span style="color: rgb(100,37,208); background-color: inherit">替换为标准的 codebook commitment loss </span>$$\mathcal{L}_{\text{codebook}}$$<span style="color: rgb(100,37,208); background-color: inherit"> ，该损失促使编码器输出与码本向量彼此靠近</span>。这里设置$$\beta = 0.25$$，并采用损失权重为$$1.0$$。因此，VQ-VAE 的最终损失函数为：

$$\mathcal{L}_{\text{VQ-VAE}} = \mathcal{L}_1 + \mathcal{L}_{\text{LPIPS}} + 0.5\,\mathcal{L}_{\text{GAN}} + 0.2\,\mathcal{L}_{\text{ID}} + \mathcal{L}_{\text{codebook}}$$

向量量化层在将编码器输出投影到 8 维空间之后应用。

除了损失函数的变更和量化层的引入外，用于 Transfusion 的 VAE 与用于 Chameleon 的 VQ-VAE 在训练设置上完全相同，例如：<span style="color: rgb(220,155,4); background-color: inherit">相同的训练计算量、相同的训练数据，以及相同的编码器解码器架构</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

与训练目标一致，解码也在两种模式间切换：语言建模模式和 diffusion 模式。在 LM 模式下，遵循标准做法，<span style="color: rgb(100,37,208); background-color: inherit">逐 token 从预测分布中采样</span>。当采样&#x5230;**`BOI`** token 时，<span style="color: rgb(100,37,208); background-color: inherit">解码算法切换至扩散模式</span>，并遵循扩散模型的标准解码流程： &#x20;

<span style="color: rgb(100,37,208); background-color: inherit">在输入序列末尾追加一段纯噪声</span>$$x_T$$，形式为$$n$$个图像 patch，取决于目标图像尺寸，并在$$T$$步内逐步去噪。<span style="color: rgb(100,37,208); background-color: inherit">在每一步</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">，利用模型预测的噪声生成</span>$$x_{t-1}$$<span style="color: rgb(100,37,208); background-color: inherit">，并用其覆盖序列中的</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">；即模型始终以当前含噪图像的最新时间步为条件，无法访问之前的时间步</span>。扩散过程结束后，在生成的图像后追&#x52A0;**`EOI`** token，并切换回 LM 模式。Transfusion 支持任意文本与图像模态的混合生成。

### 5.3.2 **<span style="color: rgb(36,91,219); background-color: inherit">S</span> <span style="color: rgb(46,161,33); background-color: inherit">h</span> <span style="color: rgb(216,57,49); background-color: inherit">o</span> <span style="color: rgb(220,155,4); background-color: inherit">w</span>-o<span style="color: rgb(36,91,219); background-color: inherit">（c-1）</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">Tokenizer</span>**

Show-o 基于预训练的 LLM，因此在离散空间中进行统一学习是一个很自然的想法，其维护一个统一的词汇表，同时包含离散的文本和图像 token。

1. **<span style="color: rgb(36,91,219); background-color: inherit">Text Tokenization</span>**

Show-o 基于 LLM，因此<span style="color: rgb(100,37,208); background-color: inherit">直接使用原有的分词器对文本数据进行分词</span>，不做任何修改。

* **<span style="color: rgb(36,91,219); background-color: inherit">Image Tokenization</span>**

按照 MAGVIT-v2 的想法，<span style="color: rgb(100,37,208); background-color: inherit">使用大规模图像数据训练一个 lookup-free 的 Quantizer</span>。这个 Quantizer <span style="color: rgb(100,37,208); background-color: inherit">维护一个大小为</span>$$K = 8192$$<span style="color: rgb(100,37,208); background-color: inherit">的 Codebook，并将</span>$$256 \times 256$$<span style="color: rgb(100,37,208); background-color: inherit">分辨率的图像编码为</span>$$16 \times 16$$<span style="color: rgb(100,37,208); background-color: inherit">的离散 token</span>，如图(a)。

另一种方法是对理解和生成使用不同的 Tokenizer。这里作者也进行了探究，从预训练的 MAGVIT-v2 和 CLIP-ViT 编码器中<span style="color: rgb(100,37,208); background-color: inherit">提取连续图像表示作为输入</span>，探索多模态理解能力的提升，如图(b)和(c)。以下默认的使用图(a)的离散图像 token 作为输入。

![](../../images/视觉多模态讲义（下）-image-181.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

Show-o 继承 LLM 的架构，唯一的改动是<span style="color: rgb(100,37,208); background-color: inherit">在每个注意力层前添加了一个 QK-Norm 操作。使用预训练 LLM 的权重初始化 Show-o，并通过新增</span>$$8192$$<span style="color: rgb(100,37,208); background-color: inherit">个可学习的 Embedding 来扩展 Embedding 层，以支持离散图像 tojken</span>。与当前 SOTA 的扩散模型需要额外文本编码器不一样的是 Show-o 本身就可以编码文本条件信息，用于文生图任务。

1. **<span style="color: rgb(36,91,219); background-color: inherit">统一的 Prompt</span>**

为在多模态理解和生成任务上实现统一学习，Show-o <span style="color: rgb(100,37,208); background-color: inherit">设计了统一的 prompt 策略，用于格式化各类输入数据</span>。给定一个图文对$$(x, y)$$，首先分别通过图像和文本 tokenizer 将其转换为$$M$$个图像 token $$\mathbf{u} = \{u_i\}_{i=1}^M$$和$$N$$个文本token $$\mathbf{v} = \{v_i\}_{i=1}^N$$。然后，根据任务类型将其组织为输入序列。**`[MMU]`**&#x548C;**`[T2I]`**&#x662F;预定义的任务 token，用于指示输入序列对应的学习任务；**`[SOT]`**&#x548C;**`[EOT]`**&#x662F;特殊 token，分别表示文本 token 的开始和结束；**`[SOI]`**&#x548C; **`[EOI]`**&#x662F;预定义的特殊 token，用于标记图像 token 的开始和结束。

![](../../images/视觉多模态讲义（下）-image-182.png)

通过这种 prompt 设计，<span style="color: rgb(46,161,33); background-color: inherit">可以将多模态理解、文生图以及混合模态生成等多种任务的输入数据有效编码为序列数据，从而在不同任务间无缝实现统一学习</span>。训练完成后，即可通过相应 prompt 引导 Show-o 处理多种视觉-语言任务，包括视觉问答和文生图：

![](../../images/视觉多模态讲义（下）-image-180.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Omni-Attention 机制</span>**

与目前使用的比较多的仅对序列进行自回归建模的方法不同的是，Show-o 提&#x51FA;**<span style="color: rgb(216,57,49); background-color: inherit">全向注意力机制</span>`omni-attention mechanism`**，使 Show-o 能以不同方式建模各类信号。这是综合性的注意力机制，结合了<span style="color: rgb(216,57,49); background-color: inherit">因果注意力</span>**`causal attention`**<span style="color: rgb(216,57,49); background-color: inherit">和全注意力</span>**`full attention`**，并能根据输入序列的格式自适应地混合与切换：

![](../../images/视觉多模态讲义（下）-image-208.png)

Show-o 对序列中的文本token $$\mathbf{v}$$使用因果注意力；对图像 token $$\mathbf{u}$$ 使用全注意力，使每个 token 都能与所有其他 token 充分交互。对于格式化的输入序列，<span style="color: rgb(100,37,208); background-color: inherit">在</span>**<span style="color: rgb(100,37,208); background-color: inherit">多模态理解</span>**<span style="color: rgb(100,37,208); background-color: inherit">任务(a)中，序列中的文本 token 可以关注所有先前的图像 token；在</span>**<span style="color: rgb(100,37,208); background-color: inherit">文生图</span>**<span style="color: rgb(100,37,208); background-color: inherit">任务(b)中，图像 token 能够与所有前置的文本 token 交互；当</span>**<span style="color: rgb(100,37,208); background-color: inherit">输入仅为文本</span>**<span style="color: rgb(100,37,208); background-color: inherit"> token 时，退化为标准的因果注意力(c)</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练目标</span>**

为同时支持自回归建模和扩散建模，采用两个学习目标：

> 1. 下一 token 预测 **<span style="color: rgb(216,57,49); background-color: inherit">NTP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">ext </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">oken </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">rediction）</span>
>
> 2. 掩码 token 预测 **<span style="color: rgb(216,57,49); background-color: inherit">MTP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ask </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">oken </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">rediction）</span>

**<span style="color: rgb(222,120,2); background-color: inherit">NTP</span>**：<span style="color: rgb(100,37,208); background-color: inherit">对于多模态理解任务，给定包含</span>$$M$$<span style="color: rgb(100,37,208); background-color: inherit">个图像 token </span>$$\mathbf{u} = \{u_1, u_2, \dots, u_M\}$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$N$$<span style="color: rgb(100,37,208); background-color: inherit">个文本 token </span>$$\mathbf{v} = \{v_1, v_2, \dots, v_N\}$$<span style="color: rgb(100,37,208); background-color: inherit">的序列，通过标准语言建模目标最大化文本 token 的似然</span>：

$$\mathcal{L}_{\text{NTP}} = \sum_{i} \log p_\theta(v_i \mid v_1, \dots, v_{i-1}, u_1, \dots, u_M)$$

其中$$p(\cdot \mid \cdot)$$表示由 Show-o 的参数$$\theta$$建模的条件概率，模型通过随机梯度下降进行训练。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：若输入序列仅包含文本 token ，则条件中不包含图像 token $$\mathbf{u}$$

**<span style="color: rgb(222,120,2); background-color: inherit">MTP</span>**：通过将掩码 token 预测作为学习目标，可以将简化的离散扩散建模无缝集成到 Show-o 中。对于建模输入序列中的图像 token $$\mathbf{u} = \{u_1, u_2, \dots, u_M\}$$，首先<span style="color: rgb(100,37,208); background-color: inherit">以时间步控制的某个随机比例将部分图像 token 替换为</span>**`[MASK]`**<span style="color: rgb(100,37,208); background-color: inherit"> token ，记为</span>$$\mathbf{u}_*$$<span style="color: rgb(100,37,208); background-color: inherit">，从而构造掩码序列</span>$$\mathbf{u}_* = \{u_*, u_2, \dots, u_*, u_M\}$$。然后<span style="color: rgb(100,37,208); background-color: inherit">通过最大化以下似然，从掩码 token 中重建原始图像 token ，条件为未掩码区域和前置文本 token</span>：

$$\mathcal{L}_{\text{MTP}} = \sum_{j} \log p_\theta(u_j \mid u_*, u_2, \dots, u_*, u_M, v_1, \dots, v_N)$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：
>
> 1. 损失仅作用于被掩码的 token
>
> 2. 使用 MaskGIT 采样策略对图像 token 进行掩码，并利用输入序列中所有文本 token 和未掩码图像 token 的信息进行重建
>
> 3. <span style="color: rgb(100,37,208); background-color: inherit">使用无分类器引导 CFG 以一定概率将条件文本 token 随机替换为空文本</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">总损失</span>**：给定一批输入序列，总体训练损失为$$\mathcal{L}_{\text{MTP}}$$与$$\mathcal{L}_{\text{NTP}}$$的加权组合：

$$\mathcal{L} = \mathcal{L}_{\text{MTP}} + \alpha \mathcal{L}_{\text{NTP}}$$

其中$$\alpha$$为超参数，用于调节$$\mathcal{L}_{\text{NTP}}$$项的权重。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

由于图像 token 的 Embedding 是全新初始化的，因此需要大规模预训练，以实现多模态理解与生成之间的对齐。并且 Show-o 移除了用于文生图任务的独立文本编码器，这给在单一 Transformer 内部实现文本与图像内容的有效对齐带来了比较大的困难。Show-o 采用一种三阶段训练策以略渐进且高效地训练：

**<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：图像 Token Embedding 训练</span>**

使用 RefinedWeb 数据集训练 Show-o，以保持其语言建模能力。同时，分别采用 ImageNet-1K 数据集和大规模图文对数据，<span style="color: rgb(100,37,208); background-color: inherit">训练 Show-o 的</span>**<span style="color: rgb(100,37,208); background-color: inherit">类别条件图像生成</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像描述生成</span>**<span style="color: rgb(100,37,208); background-color: inherit">能力</span>。这里直接利用 ImageNet-1K 中的类别名称作为文本输入，以学习类别条件图像生成。这个阶段的主要目标为：<span style="color: rgb(100,37,208); background-color: inherit">学习离散图像 token 的新增可学习 Embedding、建模图像生成中的像素依赖关系，以及在图像描述任务中实现图像与文本的初步对齐</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：多模态理解与生成的图文对齐</span>**

使用通用图文对数据进行文生图训练。<span style="color: rgb(100,37,208); background-color: inherit">核心聚焦于在</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像描述</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">文生图</span>**<span style="color: rgb(100,37,208); background-color: inherit">两项任务</span>中，进一步强化图像与文本之间的语义对齐。

**<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：高质量数据微调</span>**

通过引入经过筛选的高质量图文对数据用于文生图生成，以及指令遵循数据用于多模态理解与混合模态生成，对预训练的 Show-o 进行进一步微调。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理</span>**

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**：给定一张图像及其对应的视觉问题，Show-o 以自回归方式预测文本答案。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">视觉生成</span>**：将所&#x6709;**`[MASK]`** token 作为 Show-o 的初始输入，并在$$T$$步内迭代地&#x5C06;**`[MASK]`** token 替换为预测的图像 token。

### 5.3.3 <span style="color: rgb(36,91,219); background-color: inherit">LMFusion（c-1）</span>

Transfusion 的一个显著特点是，其架构与主流 LLM 相同，却能通过端到端训练$$\mathcal{L}_{\text{Transfusion}} = \mathcal{L}_{\text{LM}} + \lambda \cdot \mathcal{L}_{\text{DDPM}}$$同时实现文本生成、图像理解和图像生成。<span style="color: rgb(216,57,49); background-color: inherit">Transfusion 仅使用纯文本数据和图像-标题对数据训练，然而这种从零开始的训练需要大量计算资源，并且其在纯文本任务上的性能仍落后于经过预训练的 LLM</span>。 &#x20;

LMFusion 的目标是有效地将预训练的纯文本 LLM 适配为能够处理图像理解与生成任务的多模态模型。其<span style="color: rgb(100,37,208); background-color: inherit">基于开源的 Llama-3，并继续以 Transfusion 的目标函数对其进行训练，使其能够处理两种模态</span>。由于 Transfusion 在语言建模和图像扩散目标之间共享参数，关键难点在于：**<span style="color: rgb(100,37,208); background-color: inherit">在优化新引入的图像能力的同时，避免 Llama-3 强大的纯文本性能下降</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

![](../../images/视觉多模态讲义（下）-image-209.png)

LMFusion 的解决办法是：**<span style="color: rgb(100,37,208); background-color: inherit">将一个预训练的纯文本 Llama 模型与一个专用的图像 Transformer 相结合，分别用于视觉生成与理解，使得每种模态可通过独立的参数进行处理</span>**。通过冻结文本模块、仅微调视觉模块，在保留模型原有语言能力的同时，为视觉理解与生成的学习提供了一个良好的起点。

LMFusion 是一个仅含 Decoder 的模型，由$$N$$个 Transformer 层组成，核心设计&#x662F;**<span style="color: rgb(100,37,208); background-color: inherit">模态特定的注意力层</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">前馈网络FFN</span>**<span style="color: rgb(100,37,208); background-color: inherit">，各自仅处理对应模态的数据</span>。模型输入包括文本 token $$\mathbf{x}^{\text{txt}}$$和带噪图像表示：

$$\mathbf{x}^{\text{img}}_{t} = \sqrt{\bar{\alpha}_t}\, \mathbf{x}^{\text{img}} + \sqrt{1 - \bar{\alpha}_t}\, \boldsymbol{\epsilon}$$

1. **<span style="color: rgb(36,91,219); background-color: inherit">输入投影层</span>**

输入<span style="color: rgb(100,37,208); background-color: inherit">文本 token </span>$$\mathbf{x}^{\text{txt}}$$<span style="color: rgb(100,37,208); background-color: inherit">通过线性层投影为文本隐状态序列</span>$$\mathbf{h}^{\text{txt}}_{\text{in}}$$；带噪<span style="color: rgb(100,37,208); background-color: inherit">图像 </span>$$\mathbf{x}^{\text{img}}_{t}$$<span style="color: rgb(100,37,208); background-color: inherit"> 则通过 U-Net downsampler 投影为图像表示序列</span>$$\mathbf{h}^{\text{img}}_{\text{in}}$$：

$$\begin{aligned}
\mathbf{h}^{\text{txt}}_{\text{in}} &= \textcolor{blue}{\mathrm{Proj}_{\text{text}}}(\mathbf{x}^{\text{txt}}) \\
\mathbf{h}^{\text{img}}_{\text{in}} &= \textcolor{red}{\mathrm{UNet\text{-}Down}_{\text{img}}}(\mathbf{x}^{\text{img}}_{t}, t)
\end{aligned}$$

随后，文本隐状态$$\mathbf{h}^{\text{txt}}_{\text{in}}$$和图像隐状态$$\mathbf{h}^{\text{img}}_{\text{in}}$$被送入后续的注意力层。

* **<span style="color: rgb(36,91,219); background-color: inherit">自注意力层</span>**

<span style="color: rgb(100,37,208); background-color: inherit">为每种模态分别构建独立的注意力矩阵</span>：文本隐状态$$\mathbf{h}^{\text{txt}}_{\text{in}}$$和图像隐状态$$\mathbf{h}^{\text{img}}_{\text{in}}$$分别通过各自的$$Q$$、$$K$$、$$V$$矩阵转换为对应的 query、 key 和 value。<span style="color: rgb(100,37,208); background-color: inherit">注意力前的层归一化也是模态特定的</span>，并被融合进$$QKV$$函数中：

$$\begin{aligned}
\mathbf{h}^{\text{txt}}_{Q}, \mathbf{h}^{\text{txt}}_{K}, \mathbf{h}^{\text{txt}}_{V} &= \textcolor{blue}{\mathrm{QKV}_{\text{text}}}(\mathbf{h}^{\text{txt}}_{\text{in}})  \\
\mathbf{h}^{\text{img}}_{Q}, \mathbf{h}^{\text{img}}_{K}, \mathbf{h}^{\text{img}}_{V} &= \textcolor{red}{\mathrm{QKV}_{\text{img}}}(\mathbf{h}^{\text{img}}_{\text{in}}) 
\end{aligned}$$

为实现跨模态注意力，将图像和文本模态的 query、 key 和 value 拼接成统一序列。随后，<span style="color: rgb(100,37,208); background-color: inherit">在文本和图像 token 位置上的注意力加权值分别通过各自模态专属的输出投影矩阵 </span>$$O$$<span style="color: rgb(100,37,208); background-color: inherit"> 映射回隐状态维度</span>：

$$\begin{aligned}
\mathbf{h}^{\text{txt}}_{O} &= \textcolor{blue}{\mathrm{O}_{\text{text}}} \left( \mathrm{softmax}\left( \frac{\mathbf{h}^{\text{txt}}_{Q} [\mathbf{h}^{\text{img}}_{K} \circ \mathbf{h}^{\text{txt}}_{K}]^\top + \mathbf{M}}{\sqrt{d}} \right) [\mathbf{h}^{\text{img}}_{V} \circ \mathbf{h}^{\text{txt}}_{V}] \right)  \\
\mathbf{h}^{\text{img}}_{O} &= \textcolor{red}{\mathrm{O}_{\text{img}}} \left( \mathrm{softmax}\left( \frac{\mathbf{h}^{\text{img}}_{Q} [\mathbf{h}^{\text{txt}}_{K} \circ \mathbf{h}^{\text{img}}_{K}]^\top + \mathbf{M}}{\sqrt{d}} \right) [\mathbf{h}^{\text{txt}}_{V} \circ \mathbf{h}^{\text{img}}_{V}] \right) 
\end{aligned}$$

其中$$\circ$$表示拼接操作，$$\mathbf{M}$$为混合注意力掩码，与 Transfusion 中一致：<span style="color: rgb(100,37,208); background-color: inherit">对文本 token 应用因果掩码，对图像 token 应用双向掩码</span>。这种设计支持模态内及跨模态的自注意力，促进多模态融合。

* **<span style="color: rgb(36,91,219); background-color: inherit">前馈网络 FFN</span>**

在注意力层之后，<span style="color: rgb(100,37,208); background-color: inherit">采用模态特定的 FFN 分别处理文本和图像数据</span>。FFN 前的层归一化同样是模态特定的，并被融合进 FFN 函数中：

$$\begin{aligned}
\mathbf{h}^{\text{txt}}_{\text{FFN}} &= \textcolor{blue}{\mathrm{FFN}_{\text{text}}}(\mathbf{h}^{\text{txt}}_{O})  \\
\mathbf{h}^{\text{img}}_{\text{FFN}} &= \textcolor{red}{\mathrm{FFN}_{\text{img}}}(\mathbf{h}^{\text{img}}_{O}) 
\end{aligned}$$

* **<span style="color: rgb(36,91,219); background-color: inherit">输出投影层</span>**

在经过$$N$$层自注意力和 FFN 后，所得隐状态被分别投影为：通过语言模型输出头得到文本 logits，或通过 U-Net upsampler 得到图像噪声预测：

$$\begin{aligned}
\mathbf{p}_{\text{logits}} &= \textcolor{blue}{\mathrm{LM\text{-}Head}_{\text{text}}}(\mathbf{h}^{\text{txt}}_{\text{FFN}})  \\
\boldsymbol{\epsilon}_{\text{pred}} &= \textcolor{red}{\mathrm{UNet\text{-}Up}_{\text{img}}}(\mathbf{h}^{\text{img}}_{\text{FFN}}, t, \mathbf{h}^{\text{img}}_{\text{in}}) 
\end{aligned}$$

与 Transfusion 相同，<span style="color: rgb(100,37,208); background-color: inherit">输出</span>$$\mathbf{p}_{\text{logits}}$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\boldsymbol{\epsilon}_{\text{pred}}$$<span style="color: rgb(100,37,208); background-color: inherit">分别通过语言建模损失</span>$$\mathcal{L}_{\mathrm{LM}} = \mathbb{E}_{x_i^{txt}} \left[ -\log P_\theta(x_i^{txt} \mid \boldsymbol{x}_{<i}^{txt}, \boldsymbol{x}^{img}) \right]$$<span style="color: rgb(100,37,208); background-color: inherit">和 DDPM 损失</span>$$\mathcal{L}_{\mathrm{DDPM}} = \mathbb{E}_{\boldsymbol{x}^{img}, t, \epsilon} \left[ \left\| \epsilon - \epsilon_\theta(\boldsymbol{x}_t^{img}, t, \boldsymbol{x}^{txt}) \right\|_2^2 \right]$$<span style="color: rgb(100,37,208); background-color: inherit">进行优化</span>。所有文本模块参数，以及图像模块中的自注意力和 FFN 参数，均从预训练的 Llama 模型初始化而来。

在优化过程中，对文本和图像参数组采用解耦的学习率：文本学习率$$\eta_{\text{text}}$$用于$$\{ \textcolor{blue}{\mathrm{Proj}_{\text{text}}, \mathrm{QKV}_{\text{text}}, \mathrm{O}_{\text{text}}, \mathrm{FFN}_{\text{text}}, \mathrm{LM\text{-}Head}_{\text{text}}} \}$$，图像学习率$$\eta_{\text{img}}$$用于$$\{ \textcolor{red}{\mathrm{UNet\text{-}Down}_{\text{img}}, \mathrm{QKV}_{\text{img}}, \mathrm{O}_{\text{img}}, \mathrm{FFN}_{\text{img}}, \mathrm{UNet\text{-}Up}_{\text{img}}} \}$$。为保留模型在纯文本基准上的性能，设置 $$\eta_{\text{text}} = 0$$，即冻结文本模块。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：为方便描述，上述流程将残差连接与层归一化直接融合进自注意力和 FFN 中。

### 5.3.4 **<span style="color: rgb(36,91,219); background-color: inherit">Janusflow（c-2）</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">JanusFlow 用到的技术</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态大模型 Multimodal LLM</span>**

给定一个包含离散 token 序列的数据集$$\mathcal{D}$$，其中每个序列可表示为$$\mathbf{x} = (x_1, \dots, x_\ell)$$，LLM 以自回归方式建模该序列的分布：

$$\log P_{\theta_{\text{LLM}}}(\mathbf{x}) = \sum_{i=0}^{\ell-1} \log P_{\theta_{\text{LLM}}}(x_{i+1} \mid x_1, \dots, x_i)$$

其中$$\theta_{\text{LLM}}$$表示 LLM 的参数，$$\ell$$为序列长度。在大规模数据集上训练后，LLM 展现出跨任务泛化能力。<span style="color: rgb(100,37,208); background-color: inherit">为使这些模型能够处理视觉输入，通常添加一个视觉编码器</span>，例如，<span style="color: rgb(220,155,4); background-color: inherit">LLaVA 通过一个投影层将预训练的 CLIP 图像编码器与 LLM 集成，将提取的图像特征映射到 LLM 可处理的 Word Embedding 联合空间中</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">Rectified Flow</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

对于由连续$$d$$维数据点$$\mathbf{x} = (x_1, \dots, x_d)$$构成的数据集$$\mathcal{D}$$，服从未知数据分布$$\pi_1$$， rectified flow 通过学习一个定义在时间$$t \in [0, 1]$$上的常微分方程 ODE 来建模该分布：

$$\frac{d\mathbf{z}_t}{dt} = v_{\theta_{\text{NN}}}(\mathbf{z}_t, t), \quad \mathbf{z}_0 \sim \pi_0$$

其中$$\theta_{\text{NN}}$$表示速度神经网络的参数，$$\pi_0$$是一个简单分布，通常为标准高斯噪声$$\mathcal{N}(0, I)$$。这个网络<span style="color: rgb(100,37,208); background-color: inherit">通过最小化神经速度与从</span>$$\pi_0$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\pi_1$$<span style="color: rgb(100,37,208); background-color: inherit">中随机采样点之间线性路径方向的欧氏距离进行训练</span>：

$$\min_{\theta} \mathbb{E}_{t \sim P(t), \mathbf{z}_0 \sim \pi_0, \mathbf{x} \sim \pi_1} \left[ \left\| v_{\theta_{\text{NN}}}(\mathbf{z}_t, t) - (\mathbf{x} - \mathbf{z}_0) \right\|^2 \right], \quad \mathbf{z}_t = t\mathbf{x} + (1 - t)\mathbf{z}_0$$

这里$$P(t)$$是时间$$t \in [0, 1]$$上的分布。当网络容量足够且目标函数被完美最小化时，最优速度场$$v_{\theta^*_{\text{NN}}}$$将基本分布$$\pi_0$$映射到真实数据分布$$\pi_1$$。若$$\mathbf{z}_0 \sim \pi_0$$，则$$\mathbf{z}_1 = \int_0^1 v_{\theta^*_{\text{NN}}}(\mathbf{z}_t, t) dt$$的分布服从$$\pi_1$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**

在多模态理解任务中，LLM 处理由交错排列的文本和图像数据组成的输入序列。<span style="color: rgb(100,37,208); background-color: inherit">文本被分词为离散 token，每个 token 被映射为维度为</span>$$D_{\text{emb}}$$<span style="color: rgb(100,37,208); background-color: inherit">的 Embedding 向量。图像编码器</span>$$f_{\text{enc}}$$<span style="color: rgb(100,37,208); background-color: inherit">将每张图像</span>$$\mathbf{x}_{\text{im}}$$<span style="color: rgb(100,37,208); background-color: inherit">编码为形状为</span>$$H_{\text{im}} \times W_{\text{im}} \times D_{\text{enc}}$$<span style="color: rgb(100,37,208); background-color: inherit">的特征图</span>。然后特征图被展平并通过一个线性变换层投影为形状为$$H_{\text{im}} W_{\text{im}} \times D_{\text{emb}}$$的 Embedding 序列，其中$$H_{\text{im}}$$和$$W_{\text{im}}$$由图像编码器决定。<span style="color: rgb(100,37,208); background-color: inherit">文本与图像 Embedding 拼接后形成 LLM 的输入序列，LLM 基于此 Embedding 序列自回归地预测下一个 token</span>。JanusFlow 在图像前添加特殊 token **`BOI`**，图像后添&#x52A0;**`EOI`**，以帮助模型定位序列中的图像 Embedding 位置。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像生成</span>**

LLM 以文本序列$$\mathbf{x}^{\text{con}}$$为条件，利用 rectified flow 生成对应图像。为提升计算效率，生成过程在潜在空间中进行，使用预训练&#x7684;**`SDXL-VAE`**。生成过程如下：

> 1. 首先在潜在空间中采样高斯噪声$$\mathbf{z}_0$$，其形状为$$H_{\text{latent}} \times W_{\text{latent}} \times D_{\text{latent}}$$
>
> 2. 通过生成编码器$$g_{\text{enc}}$$将其处理为形状为$$H_{\text{gen}} W_{\text{gen}} \times D_{\text{emb}}$$的 Embedding 序列
>
> 3. 这个序列与表示当前时间步$$t$$的时间 Embedding 拼接，形成长度为$$H_{\text{gen}} W_{\text{gen}} + 1$$的序列。这里作者发现因果注意力已足够有效，其他掩码方案未带来性能提升。LLM 对应于$$\mathbf{z}_0$$的输出通过生成解码器$$g_{\text{dec}}$$转换回潜在空间，得到形状为$$H_{\text{latent}} \times W_{\text{latent}} \times D_{\text{latent}}$$的速度向量。

状态通过标准欧拉求解器更新：

$$\mathbf{z}_{t+dt} = \mathbf{z}_t + v(\mathbf{z}_t, t) \, dt$$

$$dt$$为用户定义的步长。这里将输入中的$$\mathbf{z}_0$$替换为$$\mathbf{z}_{dt}$$，并迭代此过程直至获得$$\mathbf{z}_1$$，再通过 VAE 解码器将其解码为最终图像。<span style="color: rgb(100,37,208); background-color: inherit">为提升生成质量，在计算速度时采用无分类器引导 CFG</span>：

$$v(\mathbf{z}_t, t) = w \, v(\mathbf{z}_t, t \mid \mathbf{x}^{\text{con}}) + (1 - w) \, v(\mathbf{z}_t, t \mid \emptyset)$$

其中$$v(\mathbf{z}_t, t \mid \emptyset)$$表示无文本条件下的速度，$$w \geq 1$$控制 CFG 的强度。增大$$w$$可提高语义对齐度。与多模态理解类似，在序列开头添加特殊 token **`BOI`**&#x4EE5;指示图像生成的起始位置。

![](../../images/视觉多模态讲义（下）-image-206.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">解耦编码器</span>**

以往将自回归生成与扩散模型统一于联合 LLM 训练框架的方法通常对理解和生成任务使用相同的编码器，即$$f_{\text{enc}} = g_{\text{enc}}$$。例如，<span style="color: rgb(220,155,4); background-color: inherit">Transfusion 在同一 VAE 潜在空间中使用共享的 U-Net 或线性编码器执行两类任务，Show-o 利用 MAGVIT-v2 将图像块编码为离散 token 用于两类任务</span>。

但<span style="color: rgb(216,57,49); background-color: inherit">这种共享编码器设计效果不是很好，尤其在通过向量量化 token 进行图像生成的模型中</span>。JanusFlow 采&#x7528;**<span style="color: rgb(100,37,208); background-color: inherit">解耦编码器设计</span>**：

> 使用预训练&#x7684;**`SigLIP-Large-Patch/16`**&#x6A21;型作为$$f_{\text{enc}}$$以提取用于多模态理解的语义连续特征
>
> 生成任务使用从零初始化的独&#x7ACB;**`ConvNeXt`**&#x6A21;块作为$$g_{\text{enc}}$$和$$g_{\text{dec}}$$，因为在生成任务中表现优异

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：JanusFlow 在$$g_{\text{enc}}$$与$$g_{\text{dec}}$$之间引入长跳跃连接。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

模型训练分为三个连续阶段：

![](../../images/视觉多模态讲义（下）-image-207.png)

仅训练随机初始化的组件：<span style="color: rgb(100,37,208); background-color: inherit">线性层、生成编码器和生成解码器</span>。目的是使这些新模块能与预训练的 LLM 和 SigLIP 编码器有效协同工作，本质上是<span style="color: rgb(100,37,208); background-color: inherit">为新引入模块提供初始化</span>。

训练除视觉编码器外的整个模型，数据包含三类：<span style="color: rgb(100,37,208); background-color: inherit">多模态理解、图像生成和纯文本</span>。初期分配较高比例的<span style="color: rgb(100,37,208); background-color: inherit">多模态理解数据增强理解能力</span>；随后逐步增加图像生成数据，<span style="color: rgb(100,37,208); background-color: inherit">提升生成能力</span>。

指令微调，数据包括<span style="color: rgb(100,37,208); background-color: inherit">对话、任务特定的交互以及高质量的文本条件图像生成样本</span>。同时解冻 SigLIP 编码器的参数。微调过程<span style="color: rgb(100,37,208); background-color: inherit">使模型能有效回答用户在多模态理解与图像生成任务中的指令</span>。

训练 JanusFlow 涉及两类数据：<span style="color: rgb(100,37,208); background-color: inherit">多模态理解数据与图像生成数据。两类数据均包含 condition 与 response 两部分</span>。condition 指任务 prompt，如<span style="color: rgb(220,155,4); background-color: inherit">生成任务中的文本 prompt、理解任务中的图像</span>，response 指对应任务的输出。数据可表示为$$\mathbf{x} = (\mathbf{x}^{\text{con}}, \mathbf{x}^{\text{res}})$$，其中上标$$\text{con}$$表示 condition，$$\text{res}$$表示 response。<span style="color: rgb(100,37,208); background-color: inherit">记整个序列 </span>$$\mathbf{x}$$<span style="color: rgb(100,37,208); background-color: inherit"> 的长度为</span>$$\ell$$<span style="color: rgb(100,37,208); background-color: inherit">，</span>$$\mathbf{x}^{\text{con}}$$<span style="color: rgb(100,37,208); background-color: inherit">的长度为</span>$$\ell_{\text{con}}$$<span style="color: rgb(100,37,208); background-color: inherit">，</span>$$\mathbf{x}^{\text{res}}$$<span style="color: rgb(100,37,208); background-color: inherit">的长度为</span>$$\ell_{\text{res}}$$<span style="color: rgb(100,37,208); background-color: inherit">。用</span>$$\theta$$<span style="color: rgb(100,37,208); background-color: inherit">表示 JanusFlow 中所有可训练参数的集合，包括 LLM、</span>$$f_{\text{enc}}$$<span style="color: rgb(100,37,208); background-color: inherit">、</span>$$g_{\text{enc}}$$<span style="color: rgb(100,37,208); background-color: inherit">、</span>$$g_{\text{dec}}$$<span style="color: rgb(100,37,208); background-color: inherit">以及线性层</span>。



1. **<span style="color: rgb(36,91,219); background-color: inherit">自回归训练目标</span>**

![Data ratio 表示多模态理解数据、图像生成数据和纯文本数据的占比。在第二阶段的前 10,000 步训练中，采用 30 : 50 : 20 的数据比例，以增强模型的理解能力。](../../images/视觉多模态讲义（下）-image-205.png)

对于多模态理解任务，$$\mathbf{x}^{\text{res}}$$仅包含文本 token。JanusFlow 采用最大似然原则进行训练：

$$\mathcal{L}_{\text{AR}}(\theta) = -\mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{und}}} \left[ \sum_{i=\ell_{\text{con}}}^{\ell-1} \log P_{\theta}(x_{i+1} \mid x_1, \dots, x_i) \right]$$

其中<span style="color: rgb(100,37,208); background-color: inherit">期望取自多模态理解数据集</span>$$\mathcal{D}_{\text{und}}$$<span style="color: rgb(100,37,208); background-color: inherit">中的所有</span>$$(\mathbf{x}^{\text{con}}, \mathbf{x}^{\text{res}})$$<span style="color: rgb(100,37,208); background-color: inherit">对，且仅在</span>$$\mathbf{x}^{\text{res}}$$<span style="color: rgb(100,37,208); background-color: inherit">的 token 上计算损失</span>。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Rectified Flow 训练目标</span>**

对于图像生成任务，$$\mathbf{x}^{\text{con}}$$为文本 token，$$\mathbf{x}^{\text{res}}$$为对应图像。JanusFlow 采用 rectified flow 目标进行训练：

$$\mathcal{L}_{\text{RF}}(\theta) = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{gen}},\, t \sim P(t),\, \mathbf{z}_0 \sim \mathcal{N}(0, I)} \left[ \left\| v_{\theta}(\mathbf{z}_t, t \mid \mathbf{x}^{\text{con}}) - (\mathbf{x}^{\text{res}} - \mathbf{z}_0) \right\|^2 \right]$$

其中$$\mathbf{z}_t = t \mathbf{x}^{\text{res}} + (1 - t) \mathbf{z}_0$$。按照 SD3 的做法，<span style="color: rgb(100,37,208); background-color: inherit">将时间分布</span>$$P(t)$$<span style="color: rgb(100,37,208); background-color: inherit">设为 logit-normal 分布</span>。为支持 CFG 推理，在训练中随机丢&#x5F03;**`10%`**&#x7684;文本 prompt。

* **<span style="color: rgb(36,91,219); background-color: inherit">表征对齐正则化</span>**

在 Diffusion Transformer 与语义编码器之间对齐中间表征可增强扩散模型的泛化能力。上述提到的<span style="color: rgb(100,37,208); background-color: inherit">解耦编码器设计使得对齐可高效实现为一个正则项</span>。对于生成任务，<span style="color: rgb(100,37,208); background-color: inherit">将理解编码器</span>$$f_{\text{enc}}$$<span style="color: rgb(100,37,208); background-color: inherit">的特征与 LLM 的中间特征对齐</span>：

$$\mathcal{L}_{\text{REPA}}(\theta, \varphi) = -\mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{gen}}} \left[ \text{sim}\left( \text{stop\_grad}(f_{\text{enc}}(\mathbf{x}^{\text{res}})),\, h_{\varphi}(q_{\theta}(\mathbf{z}_t)) \right) \right]$$

其中$$q_{\theta}(\mathbf{z}_t)$$表示给定输入$$\mathbf{z}_t$$时 LLM 的中间表征，$$h_{\varphi}$$是一个小型可训练 MLP，将$$q_{\theta}(\mathbf{z}_t)$$投影至维度$$D_{\text{enc}}$$。函数$$\text{sim}(\cdot, \cdot)$$计算 Embedding 间逐元素余弦相似度的均值。在计算损失前，将$$h_{\varphi}(q_{\theta}(\mathbf{z}_t))$$重塑为$$H_{\text{gen}} \times W_{\text{gen}} \times D_{\text{enc}}$$。这里<span style="color: rgb(100,37,208); background-color: inherit">特意调整</span>$$g_{\text{enc}}$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$g_{\text{dec}}$$<span style="color: rgb(100,37,208); background-color: inherit">的配置，确保</span>$$H_{\text{gen}} = H_{\text{im}}$$<span style="color: rgb(100,37,208); background-color: inherit">且</span>$$W_{\text{gen}} = W_{\text{im}}$$。$$\mathcal{L}_{\text{REPA}}$$的梯度不反向传播至理解编码器。<span style="color: rgb(46,161,33); background-color: inherit">对齐损失有助于 LLM 的内部特征空间在噪声输入</span>$$\mathbf{z}_t$$<span style="color: rgb(46,161,33); background-color: inherit">下与理解编码器的语义特征空间对齐，从而在推理阶段从新随机噪声和文本条件生成图像时提升生成质量</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">总损失</span>**

上述三个目标在所有训练阶段均被应用：多模态理解任务使用$$\mathcal{L}_{\text{AR}}$$，图像生成任务使用联合损失$$\mathcal{L}_{\text{RF}} + \mathcal{L}_{\text{REPA}}$$。

### 5.3.5 <span style="color: rgb(36,91,219); background-color: inherit">Mogao（c-2）</span>

VLM 通过建模条件分布$$P(x_{\text{txt}} \mid x_{\text{img}}, x_{\text{txt}})$$来实现视觉理解，而文本到图像 **<span style="color: rgb(216,57,49); background-color: inherit">T2I</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ext-</span>**<span style="color: rgb(216,57,49); background-color: inherit">to</span>**<span style="color: rgb(216,57,49); background-color: inherit">-</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage）</span>模型则利用$$P(x_{\text{img}} \mid x_{\text{txt}})$$进行图像生成。<span style="color: rgb(100,37,208); background-color: inherit">Mogao 使用统一的主干网络对图像与文本的联合分布进行建模，天然具备理解和图像生成的双重能力</span>。此外<span style="color: rgb(46,161,33); background-color: inherit">进一步通过因果建模方法扩展生成能力，使 Mogao 能够执行更复杂的任务：</span>**<span style="color: rgb(46,161,33); background-color: inherit">交错多模态生成</span>**<span style="color: rgb(46,161,33); background-color: inherit">，即利用过去任意模态生成的信息作为条件，生成任意模态的新输出</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">基础知识</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**

文本数据可表示为离散 token 序列$$\mathbf{z} = (z_0, \dots, z_L)$$，LLM 以因果方式对其联合分布进行建模。基于 LLM，VLM 通过增加视觉编码器将视觉特征转换为连续 token 序列以处理视觉输入。通过对大规模多模态数据集的训练，VLM 扩展了条件分布：

$$\log P_{\theta_{\text{VLM}}}(\mathbf{z} \mid \mathbf{x}) = \sum_{l=0}^{L} \log P_{\theta_{\text{VLM}}}(z_{l+1} \mid z_0, \dots, z_l, \mathbf{x})$$

其中$$\theta_{\text{VLM}}$$和$$\mathbf{x} = (x_0, \dots, x_N)$$分别表示 VLM 的参数和视觉输入的连续 token 序列，且$$\mathbf{x}$$可为空$$\emptyset$$。<span style="color: rgb(100,37,208); background-color: inherit">通常 LLM 和 VLM 仅输出文本 token </span>$$z_i$$<span style="color: rgb(100,37,208); background-color: inherit">。因此可通过最小化</span>$$P_\theta$$<span style="color: rgb(100,37,208); background-color: inherit">与数据分布之间的交叉熵来优化模型</span>，即下一 token 预&#x6D4B;**&#x20;<span style="color: rgb(216,57,49); background-color: inherit">NTP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">ext </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">oken </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">rediction）</span>损失：

$$\mathcal{L}_{\text{NTP}} = \mathbb{E}_{z_i} \left[ -\log P_{\theta_{\text{VLM}}}(z_{l+1} \mid z_0, \dots, z_l, \mathbf{x}) \right]$$

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉生成</span>**

<span style="color: rgb(100,37,208); background-color: inherit">Mogao 采用 rectified flow matching 对图像分布进行建模，给定图像分布</span>$$x \sim \pi$$<span style="color: rgb(100,37,208); background-color: inherit">和简单先验分布</span>$$\epsilon \sim \mathcal{N}(0, 1)$$<span style="color: rgb(100,37,208); background-color: inherit">，构建一个关于时间</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">的线性轨迹</span>：

$$x_t = t \cdot x + (1 - t) \cdot \epsilon, \quad t \in [0, 1]$$

由上式可导出如下 ODE 常微分方程：

$$\frac{dx_t}{dt} = v(x_t, t)$$

其中速度场$$v = x - \epsilon$$。为生成图像，可从$$x_0 \sim \mathcal{N}(0, 1)$$出发，对上式的 ODE 进行反向时间积分，即：

$$x_1 = \int_0^1 v(x_t, t) \, dt$$

因此，图像生成任务需训练参数为$$\theta$$的网络以逼近$$v$$，通过优化 flow matching 损失：

$$\mathcal{L}_{\text{flow}} = \mathbb{E}_{t, x \sim \pi, \epsilon \sim \mathcal{N}(0,1)} \left[ \| v_\theta(x_t, t) - (x - \epsilon) \|_2^2 \right]$$

* **<span style="color: rgb(36,91,219); background-color: inherit">交错多模态生成</span>**

Mogao <span style="color: rgb(100,37,208); background-color: inherit">以因果方式在</span>**<span style="color: rgb(100,37,208); background-color: inherit">模态级别</span>**<span style="color: rgb(100,37,208); background-color: inherit">上对序列分布进行建模</span>：

$$\log P_\theta(\mathbf{x}) = \sum_{t=0}^{T} \log P_\theta(x_{t+1} \mid x_0, \dots, x_t), \quad x_t \in \{x_{\text{img}}, x_{\text{txt}}\}$$

其中$$\theta$$为模型参数；$$x_t$$表示视觉或文本模态，而非单个 token。一旦$$x_{t+1}$$生成，它将被加入历史信息作为后续生成的条件。因此，<span style="color: rgb(100,37,208); background-color: inherit">Mogao 能够基于任意混合模态的上下文，生成任意模态的输出，即实现</span>**<span style="color: rgb(100,37,208); background-color: inherit">交错多模态生成</span>**。

对于交错多模态生成任务，需要<span style="color: rgb(100,37,208); background-color: inherit">在同一交错图文样本中</span>**<span style="color: rgb(100,37,208); background-color: inherit">同时计算</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本的 NTP 损失和带噪图像的 flow matching 损失</span>。因此，总训练损失为两者的加权组合：

$$\mathcal{L}_{\text{total}} = \lambda \cdot \mathcal{L}_{\text{NTP}} + \mathcal{L}_{\text{flow}}$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：在训练过程中，Mogao 使用两个特殊 token 来区分视觉与文本模态：所有图像序列前后分别插&#x5165;**`<img>`**&#x548C; **`</img>`**。在推理阶段，Mogao 根据上下文决定是否生成图像，当生&#x6210;**`<img>`**&#x65F6;切换至图像生成模式，生&#x6210;**`</img>`**&#x65F6;切换回文本生成模式。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

![](../../images/视觉多模态讲义（下）-image-204.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">双视觉编码器</span>**

当前主流 T2I 模型在扩散过程前使用 VAE 将图像编码至低维潜在空间。但<span style="color: rgb(216,57,49); background-color: inherit">与 VLM 中 ViT 提供的预对齐语义视觉表征相比，VAE 生成的视觉表征在视觉理解任务中表现不足</span>，这在 JanusFlow 等工作中也提到了：<span style="color: rgb(220,155,4); background-color: inherit">JanusFlow 在视觉生成时使用 VAE ，而在视觉理解时使用 ViT</span>。在 Mogao 的交错多模态生成场景中进一步优化了这个策略：**<span style="color: rgb(100,37,208); background-color: inherit">当图像作为条件时，Mogao 同时提取 ViT 和 VAE 的视觉表征，并将其拼接到历史序列中</span>**：

> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**：文本 token 仅关注历史序列中的 ViT token 和文本 token
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态生成</span>**：带噪的 VAE token 会关注历史序列中的所有 token，包括 ViT、VAE 和文本 token

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：这样做是因为 <span style="color: rgb(100,37,208); background-color: inherit">ViT 提供的视觉表征具有强语义性，有助于提升图像生成中的上下文对齐能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">深度融合架构</span>**

Mogao 基于预训练的 Qwen2.5，并借鉴 MMDiT 的设计以增强视觉生成能力，对每个 Transformer 块进行如下修改：

> 1. 使用<span style="color: rgb(100,37,208); background-color: inherit">统一的自注意力层</span>同时处理视觉与文本序列
>
> 2. 考虑到视觉与文本模态的差异，<span style="color: rgb(100,37,208); background-color: inherit">在 FFN 中使用不同的</span>**<span style="color: rgb(100,37,208); background-color: inherit"> </span>**<span style="color: rgb(100,37,208); background-color: inherit">MLP，并在注意力块中使用独立的线性层分别处理两种模态</span>
>
> 3) 由于 ViT 提供的视觉表征具有与文本对齐的高层语义，<span style="color: rgb(100,37,208); background-color: inherit">将 ViT token 路由至文本分支</span>
>
> 4) 由于 rectified flow 需要预测当前时间步的速度，因此<span style="color: rgb(100,37,208); background-color: inherit">通过 AdaLN 层将 timestep Embedding 融合到每个块的视觉特征图上</span>

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：Mogao 有几个优点
>
> 1. <span style="color: rgb(46,161,33); background-color: inherit">无需依赖传统 T2I 模型中常用的 CLIP 或 T5 文本编码器即可实现高质量图像生成</span>，这归功于 LLM 文本表征与图像特征在每一层的深度融合
>
> 2. 模型<span style="color: rgb(46,161,33); background-color: inherit">自然继承了 LLM 的长上下文能力</span>，这对长序列交错多模态生成至关重要

3. **<span style="color: rgb(36,91,219); background-color: inherit">交错旋转位置编码 Interleaved RoPE</span>**

Mogao 的输入与输出均为多模态，因此模型需捕捉三维位置关系：$$H$$和$$W$$表示图像的空间位置信息，$$T$$表示所有图像与文本 token 之间的时间位置关系。Mogao 在 RoPE 的基础上提出<span style="color: rgb(216,57,49); background-color: inherit">交错旋转位置编码</span>**<span style="color: rgb(216,57,49); background-color: inherit"> IL-RoPE</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">nter</span>**<span style="color: rgb(216,57,49); background-color: inherit">l</span>**<span style="color: rgb(216,57,49); background-color: inherit">eaved </span>**<span style="color: rgb(216,57,49); background-color: inherit">RoPE</span>**<span style="color: rgb(216,57,49); background-color: inherit">）</span>，针对同时生成文本与图像的场景，改进了各维度的频率分配与位置 ID 设置。

![](../../images/视觉多模态讲义（下）-image-202.png)

在位置 ID 设计上：

> * 对于图像，<span style="color: rgb(100,37,208); background-color: inherit">每个图像的</span>$$H/W$$<span style="color: rgb(100,37,208); background-color: inherit">空间位置 ID 均从 0 开始计算，确保所有图像在空间上具有一致的位置信息</span>，有利于图像编辑、多图生成等任务，因为这些任务需参考先前图像
>
> * 时间位置 ID 随每个 token 递增，但<span style="color: rgb(100,37,208); background-color: inherit">同一图像内所有 token 具有相同的时间位置 ID</span>

在频率分配方面，M-RoPE 在$$d=128$$维中按如下方式分配：

$$\Theta_T = \left\{ \beta^{-\frac{2i}{d}} \mid i \in [0, 16) \right\}; \quad
\Theta_H = \left\{ \beta^{-\frac{2i}{d}} \mid i \in [16, 40) \right\}; \quad
\Theta_W = \left\{ \beta^{-\frac{2i}{d}} \mid i \in [40, 64) \right\}$$

其中$$\Theta$$表示应用于特定维度的旋转频率，$$\beta$$为基频。这说明<span style="color: rgb(100,37,208); background-color: inherit">时间维度</span>$$T$$<span style="color: rgb(100,37,208); background-color: inherit">侧重捕捉局部关系，空间维度</span>$$H/W$$<span style="color: rgb(100,37,208); background-color: inherit">侧重长程依赖</span>。但<span style="color: rgb(216,57,49); background-color: inherit">图像合成高度依赖局部语义信息，显著损害图像生成性能</span>，例如，$$256$$<span style="color: rgb(220,155,4); background-color: inherit">分辨率图像在</span>$$H/W$$<span style="color: rgb(220,155,4); background-color: inherit">上仅含 16 个 token</span>。因此 IL-RoPE <span style="color: rgb(100,37,208); background-color: inherit">在</span>$$d$$<span style="color: rgb(100,37,208); background-color: inherit">维中</span>**<span style="color: rgb(100,37,208); background-color: inherit">交错分配</span>**$$\{T, H, W\}$$<span style="color: rgb(100,37,208); background-color: inherit">的频率，以平衡各维度对长程与局部语义信息的捕捉能力</span>：

$$\Theta_T = \left\{ \beta^{-\frac{2i}{d}} \mid i = 3j \right\}, \quad
\Theta_H = \left\{ \beta^{-\frac{2i}{d}} \mid i = 3j + 1 \right\}, \quad \\
\Theta_W = \left\{ \beta^{-\frac{2i}{d}} \mid i = 3j + 2 \right\}$$

![](../../images/视觉多模态讲义（下）-image-203.png)

IL-RoPE <span style="color: rgb(100,37,208); background-color: inherit">交替将前 48 个通道分配给</span>$$T/H/W$$<span style="color: rgb(100,37,208); background-color: inherit">，并将剩余 16 个通道分配给</span>$$T$$<span style="color: rgb(100,37,208); background-color: inherit">以建模长序列</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态无分类器引导</span>**

Mogao 采用无分类器引导 CFG 提升生成质量，其通过结合条件预测与无条件预测，使结果更贴合指定条件：

$$\nabla_x \log p(x \mid c) = \gamma \left( \nabla_x \log p(x \mid c) - \nabla_x \log p(x) \right) + \nabla_x \log p(x)$$

其中$$c$$和$$\gamma$$分别表示条件和 CFG 系数。传统 T2I 模型的条件仅包含文本模态$$c_{\text{txt}}$$，但在本文的交错多模态生成场景中，条件$$c$$可分解为$$c_{\text{img}}$$和$$c_{\text{txt}}$$。但<span style="color: rgb(216,57,49); background-color: inherit">条件中的图像会成为模型生成后续图像的</span>`捷径`<span style="color: rgb(216,57,49); background-color: inherit">，导致类似</span> **[STIV](https://arxiv.org/pdf/2412.07730)&#x20;**<span style="color: rgb(216,57,49); background-color: inherit">提到的</span>**<span style="color: rgb(216,57,49); background-color: inherit">时间停滞 </span>**<span style="color: rgb(216,57,49); background-color: inherit">temporal stagnation 的现象</span>。类似 **[InstructPix2Pix](https://arxiv.org/pdf/2211.09800)&#x20;**&#x7684;思想，Mogao <span style="color: rgb(100,37,208); background-color: inherit">引入不同的 CFG 系数</span>$$\gamma$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\gamma_{\text{img}}$$<span style="color: rgb(100,37,208); background-color: inherit">以分别控制不同模态的影响</span>：

$$\begin{aligned}
\nabla_x \log p(x \mid c_{\text{img}}, c_{\text{txt}}) = &\ \gamma \left( \nabla_x \log p(x \mid c_{\text{img}}, c_{\text{txt}}) - \nabla_x \log p(x \mid c_{\text{img}}) \right) \\
&+ \gamma_{\text{img}} \left( \nabla_x \log p(x \mid c_{\text{img}}) - \nabla_x \log p(x) \right) + \nabla_x \log p(x)
\end{aligned}$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：当$$\gamma_{\text{img}} = \gamma$$时，这个公式退化为$$\nabla_x \log p(x \mid c) = \gamma \left( \nabla_x \log p(x \mid c) - \nabla_x \log p(x) \right) + \nabla_x \log p(x)$$。常规跨模态 CFG 系数对模态内条件可能过大。但这样设计后可有效调节不同模态对图像生成的影响，作者使用$$\gamma = 7.5$$和$$\gamma_{\text{img}} = 1.5$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

Mogao 混合使用纯文本、视觉理解、图像生成及交错多模态数据训练模型：

> * 纯文本与视觉理解数据<span style="color: rgb(100,37,208); background-color: inherit">继承自 DouBao LM 和 VLM 数据集</span>
>
> * 图像生成数据<span style="color: rgb(100,37,208); background-color: inherit">采用 SeedDream 的训练数据</span>以提升图像质量与多样性
>
> * 交错多模态数据从公开网站与视频中整理：
>
>   * 对于原生图文数据，<span style="color: rgb(100,37,208); background-color: inherit">训练时保留原始顺序</span>
>
>   * 对于视频片段，<span style="color: rgb(100,37,208); background-color: inherit">训练了一个 VLM 为采样帧生成字幕，训练样本由帧与字幕交错组成</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">原生分辨率</span>**

将图像缩放至固定分辨率显然次优，因其同时降低训练推理效率并严重限制模型的视觉理解能力。因此，<span style="color: rgb(100,37,208); background-color: inherit">整个训练阶段均采用原生分辨率策略：将输入图像转换为可变数量的视觉 token，同时保持其宽高比</span>。此外，为弥补视觉生成与理解之间的表征差异，Mogao 在这两项任务中一致采用该策略。

* **<span style="color: rgb(36,91,219); background-color: inherit">全局损失</span>**

在多 GPU 上训练混合源多模态数据时，需谨慎平衡不同权重 token 的损失以实现任务间的良好权衡。设第$$n$$个 GPU 上第$$i$$个 token 的损失为$$L_i^n$$，权重为$$w_i^n$$，理想做法是将所有 GPU 的 token 视为一个全局批次，计算关于模型参数$$W$$的梯度：

$$\frac{\partial}{\partial W} \left( \frac{\sum_n \sum_i w_i^n L_i^n}{\sum_n \sum_i w_i^n} \right)$$

但这需要在所有 rank 间收集权重与损失，通信开销巨大，严重影响训练效率。常用近似方法是计算每卡损失后平均梯度：

$$\frac{1}{N} \sum_n \frac{\partial}{\partial W} \left( \frac{\sum_i w_i^n L_i^n}{\sum_i w_i^n} \right)$$

但该梯度存在偏差。Mogao 使用以下无偏代理：

$$\frac{1}{N} \sum_n \frac{\partial}{\partial W} \left( \frac{\sum_i w_i^n L_i^n}{\frac{1}{N} \sum_n \sum_i w_i^n} \right)$$

这可推导为无偏梯度。具体实现时<span style="color: rgb(100,37,208); background-color: inherit">先对每卡的权重求和并进行 </span>**<span style="color: rgb(100,37,208); background-color: inherit">all-reduce</span>**<span style="color: rgb(100,37,208); background-color: inherit">，再用归约后的总权重作为归一化项</span>。由于仅需通信$$N$$个标量，开销可忽略，且<span style="color: rgb(100,37,208); background-color: inherit">实际中采用异步归约 async-reduce，不会引入训练效率损失</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">ECTF（Efficient Complete Teacher Forcing）</span>**

在交错多模态生成中，训练与推理存在显著差异：<span style="color: rgb(216,57,49); background-color: inherit">训练时所有视觉元素均需预测，图像经过多级高斯噪声扰动的扩散前向过程，后续文本或图像模态在此噪声条件下训练，导致训练与推理之间存在域偏移</span>。<span style="color: rgb(220,155,4); background-color: inherit">Transfusion 通过约束噪声尺度缓解该问题，但仍保留训练时的迭代去噪与推理时的单次生成之间的不匹配。其他一些方法仅在最终图像上计算损失，虽理论上可行，但因重复计算前置项导致计算复杂度随序列长度平方增长，对长多模态序列不可行</span>。

![方法比较。图像的 VAE 和 ViT 表征分别以蓝色和黄色显示，VAE token 与 ViT token 共享相同的位置 ID。](../../images/视觉多模态讲义（下）-image-200.png)

![ECTF 注意力掩码。VAE token 可以关注所有 token，但文本 token 和 ViT token 无法关注 VAE token。](../../images/视觉多模态讲义（下）-image-198.png)

Mogao 提&#x51FA;**`ECTF`**，在<span style="color: rgb(46,161,33); background-color: inherit">计算效率与条件一致性间取得最优平衡</span>。ECTF 引入 **<span style="color: rgb(100,37,208); background-color: inherit">clean-noise 解耦</span>**<span style="color: rgb(100,37,208); background-color: inherit">：通过右图动态因果注意力掩码，将带交错句子的干净图像与噪声图像分离，避免了最后扰动方法中的冗余计算</span>。

### 5.3.6 <span style="color: rgb(36,91,219); background-color: inherit">BAGEL（c-2）</span>

![](../../images/视觉多模态讲义（下）-image-197.png)

BAGEL 采用 **<span style="color: rgb(216,57,49); background-color: inherit">MoT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ixture-</span>**<span style="color: rgb(216,57,49); background-color: inherit">o</span>**<span style="color: rgb(216,57,49); background-color: inherit">f-</span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformers）</span>架构，包含两个 Transformer 专家模块：<span style="color: rgb(100,37,208); background-color: inherit">一个专用于</span>**<span style="color: rgb(100,37,208); background-color: inherit">多模态理解</span>**<span style="color: rgb(100,37,208); background-color: inherit">，另一个专用于</span>**<span style="color: rgb(100,37,208); background-color: inherit">多模态生成</span>**。相应地，模型采用了两个独立的视觉编码器：<span style="color: rgb(100,37,208); background-color: inherit">一个面向理解的编码器和一个面向生成的编码器</span>。这两个 Transformer 专家<span style="color: rgb(100,37,208); background-color: inherit">在每一层都通过共享的自注意力作用于相同的 token 序列</span>。

> * 预测文本 token 时，BAGEL 使&#x7528;**<span style="color: rgb(100,37,208); background-color: inherit">下一 token 预测</span>**<span style="color: rgb(100,37,208); background-color: inherit">范式</span>，发挥自回归语言模型的优势
>
> * 预测视觉 token 时，BAGEL 采用 **<span style="color: rgb(100,37,208); background-color: inherit">Rectified Flow </span>**<span style="color: rgb(100,37,208); background-color: inherit">方法</span>，使用视觉生成比较成功的方法

统一的多模态生成与理解模型的三种设计范式的优缺点如下：

**<span style="color: rgb(222,120,2); background-color: inherit">自回归方法</span>**

基于离散视觉 tokenizer 的自回归视觉生成方法。这类方法对文本和视觉 token 均采用下一 Token 预测范式，<span style="color: rgb(46,161,33); background-color: inherit">实现简单，可直接复用现有 LLM 基础设施</span>。但自回归的<span style="color: rgb(216,57,49); background-color: inherit">视觉生成质量劣于扩散模型，且由于其顺序生成特性，推理延迟较高</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">外接扩散模型</span>**

将 LLM 与外部扩散模块结合。通过轻量级、可训练的 adapter 连接预训练的 LLM/VLM 与扩散模型。LLM 自回归地生成一组潜在 token 作为语义条件信号，再由扩散模块用于图像生成。这种方法<span style="color: rgb(46,161,33); background-color: inherit">收敛迅速、数据消耗少，在多模态生成与理解能力上表现优异</span>。主要缺点是<span style="color: rgb(216,57,49); background-color: inherit">将 LLM 上下文压缩为少量潜在 token，在理解与生成模块之间引入了显式瓶颈，可能导致信息损失，尤其在长上下文多模态推理中</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">集成 Transformer</span>**

将 LLM 与扩散模型统一集成于单一 Transformer 中。受自回归 Transformer 强大的理解推理能力与扩散 Transformer 强大的视觉生成能力互补优势驱动，利用二者共有的模型架构实现范式间的无缝切换。相比外接扩散模型，它<span style="color: rgb(216,57,49); background-color: inherit">需要更高的训练算力</span>，但关键优势在于<span style="color: rgb(46,161,33); background-color: inherit">在所有 Transformer 块中保持无瓶颈的上下文传递，从而实现生成与理解模块之间的无损交互，更易于扩展</span>。

统一的集成模型有能力从大规模交错多模态数据中学习更丰富的多模态能力，这是其他范式无法捕捉的涌现能力。为此 BAGEL 选择无瓶颈的集&#x6210;**&#x20;Transformer** 方案，其在大规模训练场景中潜力更大，更适合作为长上下文多模态推理及强化学习的基础模型。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

BAGEL 的主模型以 **Qwen2.5 LLM** 初始化，它<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**`RMSNorm`**<span style="color: rgb(100,37,208); background-color: inherit">进行归一化，</span>**`SwiGLU`**<span style="color: rgb(100,37,208); background-color: inherit">作为激活函数，</span>**`RoPE`**<span style="color: rgb(100,37,208); background-color: inherit">用于位置编码，</span>**`GQA`**<span style="color: rgb(100,37,208); background-color: inherit">用于 KV Cache 压缩</span>。在此基础上，BAGEL <span style="color: rgb(100,37,208); background-color: inherit">在每个注意力块中加入</span>**`QK-Norm`**，这也是是图像视频生成模型中的常见做法，可有效稳定训练过程。视觉信息从两个方面表示：

> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉理解</span>**：<span style="color: rgb(100,37,208); background-color: inherit">利用 </span>**<span style="color: rgb(100,37,208); background-color: inherit">ViT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 编码器将原始像素转换为 token</span>。采用固定分辨率$$384$$&#x7684;**`SigLIP2-so400m/14`**&#x4F5C;为 ViT 编码器的初始化。在此基础上，<span style="color: rgb(100,37,208); background-color: inherit">首先对位置编码进行插值，将最大输入尺寸设为</span>$$980 \times 980$$<span style="color: rgb(100,37,208); background-color: inherit">，然后集成</span>**`NaViT`**<span style="color: rgb(100,37,208); background-color: inherit">以支持原生宽高比图像处理</span>。采用两层 MLP 将 ViT token 的特征维度与 LLM 隐藏状态对齐。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉生成</span>**：使&#x7528;**`FLUX`**&#x4E2D;的预训练 VAE 模型，在像素空间与潜在空间之间进行双向转换。<span style="color: rgb(100,37,208); background-color: inherit">潜在表示的下采样率为</span>$$8$$<span style="color: rgb(100,37,208); background-color: inherit">，潜在通道数为</span>$$16$$<span style="color: rgb(100,37,208); background-color: inherit">，随后通过</span>$$2 \times 2$$<span style="color: rgb(100,37,208); background-color: inherit">的 patch embedding 层进一步降低空间尺寸并匹配 LLM 主干的隐藏维度</span>。VAE 模型在训练期间冻结。

在将 ViT 和 VAE tokens 注入 LLM 前，<span style="color: rgb(100,37,208); background-color: inherit">添加 2D 位置编码</span>。对于 timestep，<span style="color: rgb(100,37,208); background-color: inherit">直接将 timestep embedding 加到 VAE token 的初始隐藏状态上，而不是使用常规 DiT 中的 AdaLN</span>。这在保持性能的同时简化了架构。在 LLM 内部，来自理解与生成任务的文本、ViT 和 VAE token 根据输入的模态结构交错排列。对于同一样本的 token，采用<span style="color: rgb(216,57,49); background-color: inherit">广义因果注意力</span>**`generalized causal attention`**：首&#x5148;**<span style="color: rgb(100,37,208); background-color: inherit">将 token 划分为多个连续片段，每个片段仅包含单一模态</span>**，如<span style="color: rgb(220,155,4); background-color: inherit">文本、ViT 或 VAE</span> <span style="color: rgb(100,37,208); background-color: inherit">。某一片段中的 token 可关注所有前序片段中的 token；在片段内部，文本 token 采用</span>**<span style="color: rgb(100,37,208); background-color: inherit">因果注意力</span>**<span style="color: rgb(100,37,208); background-color: inherit">，而视觉 tokens 保留</span>**<span style="color: rgb(100,37,208); background-color: inherit">双向注意力</span>**。

1. **<span style="color: rgb(36,91,219); background-color: inherit">广义因果注意力</span>**

训练期间，一个交错多模态生成样本可能包含多张图像。对每张图像，准备三组视觉 token：

> * **<span style="color: rgb(36,91,219); background-color: inherit">带噪 VAE token</span>**：经扩散噪声污染的 VAE 潜变量，<span style="color: rgb(100,37,208); background-color: inherit">仅用于 Rectified Flow 训练</span>，MSE 损失在此集合上计算
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">干净 VAE token</span>**：原始潜变量，在<span style="color: rgb(100,37,208); background-color: inherit">生成后续图像或文本 token 时作为条件</span>
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">ViT token</span>**：<span style="color: rgb(100,37,208); background-color: inherit">由 SigLIP2 编码器提取</span>，有助于统一交错生成与理解数据的输入格式，并经验性地提升交错生成质量

在交错图像或文本生成中，<span style="color: rgb(100,37,208); background-color: inherit">后续图像或文本 token 可关注前序图像的干净 VAE tokens 和 ViT token，但不可关注其带噪 VAE token</span>。

![BAGEL 在训练期间的因果掩码机制如下：VAE 和 ViT 分别表示 VAE 特征和 ViT 特征，t 为扩散噪声的时间步，t = 0 表示无噪声。对于每一张独立图像，其内部的 VAE 特征与 ViT 特征之间采用 full attention。左图表示在交错图文生成过程中，当前图像仅能关注前序图像；  右图表示在交错多图像或视频片段生成中，采用 diffusion forcing 策略，即当前图像以前序图像的带噪表示作为条件。为增强生成一致性，随机将连续图像分组，并在每个组内应用 full attention。](../../images/视觉多模态讲义（下）-image-201.png)

对于交错多图像生成，<span style="color: rgb(100,37,208); background-color: inherit">采用 </span>**<span style="color: rgb(100,37,208); background-color: inherit">diffusion forcing 策略</span>**<span style="color: rgb(100,37,208); background-color: inherit">，为不同图像添加独立的噪声水平，并以先前图像的带噪表示作为条件</span>。此外，为增强生成一致性，<span style="color: rgb(100,37,208); background-color: inherit">随机将连续图像分组，并在组内应用 full attention</span>，组内噪声水平相同。

这里使用 PyTorch **`FlexAttention`**&#x5B9E;现广义因果注意力，相比常规的缩放点积注意力提速约$$2\times$$。推理时广义因果注意力可以<span style="color: rgb(100,37,208); background-color: inherit">缓存生成的多模态上下文的键值对 ，从而加速多模态解码</span>。仅缓存干净 VAE token 和 ViT token 的 KV 对；<span style="color: rgb(100,37,208); background-color: inherit">一旦图像完全生成，上下文中的对应带噪 VAE token 即被其干净版本替换</span>。为在交错推理中启用无分类器引导 CFG，以概率$$0.1$$、$$0.5$$、$$0.1$$分别随机丢弃文本、ViT 和干净 VAE token。

* **<span style="color: rgb(36,91,219); background-color: inherit">Transformer 设计</span>**

作者比较了三种 Transformer 变体：Dense Transformer、MoE Transformer 和 MoT 架构。

> * **<span style="color: rgb(36,91,219); background-color: inherit">MoE</span>**：仅在每个 Qwen2.5 块中复制 FFN，作为生成专家的初始化
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">MoT</span>**：复制 Qwen2.5 的全部可训练参数，创建一个完整尺寸的生成专家

MoE 与 MoT 均采用硬路由：<span style="color: rgb(100,37,208); background-color: inherit">新复制的生成专家</span>**<span style="color: rgb(100,37,208); background-color: inherit">仅处理 VAE token</span>**<span style="color: rgb(100,37,208); background-color: inherit">，而原始参数的理解专家处理文本和 ViT token</span>。尽管 MoE 与 MoT 使总参数量约为 Dense 的两倍，但<span style="color: rgb(46,161,33); background-color: inherit">这三者结构在训练与推理时的 FLOPs 完全相同</span>。

![](../../images/视觉多模态讲义（下）-image-199.png)

如上图，MoT 变体在所有任务上均优于 Dense 和 MoE 设计，尤其在多模态生成任务上差距最显著。<span style="color: rgb(100,37,208); background-color: inherit">生成任务的 MSE 损失呈现平滑单调下降趋势，</span> <span style="color: rgb(46,161,33); background-color: inherit">MoT 不仅收敛最快，且最终损失最低</span> <span style="color: rgb(100,37,208); background-color: inherit">，理解任务的 CE 损失因交错异构数据而波动较大，但 </span> <span style="color: rgb(46,161,33); background-color: inherit">MoT 仍总体表现最佳</span>。这说&#x660E;**<span style="color: rgb(100,37,208); background-color: inherit">将生成与理解任务的参数解耦</span>**<span style="color: rgb(100,37,208); background-color: inherit">的明显优势，二者可能将模型引向参数空间的不同区域</span>。也就是说为多模态理解与生成分配独立参数，可缓解由模态特定学习目标竞争引发的优化难题。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

BAGEL 的训练数据覆盖语言、图像、视频和网页等多种模态，通过统一的多模态接口支持复杂的推理、上下文预测、物理建模与未来帧生成。除了常规的 VLM、文本到图像 T2I 和 LLM 数据外，还<span style="color: rgb(100,37,208); background-color: inherit">构建了大规模的</span>**<span style="color: rgb(100,37,208); background-color: inherit">视觉-文本交错数据集</span>**<span style="color: rgb(100,37,208); background-color: inherit">，以增强模型在长上下文、多轮交互场景下的多模态理解与生成能力</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">纯文本数据</span>**

为维持底层 LLM 的语言能力，引入高质量纯文本语料，确保模型在通用任务中保持强推理与生成性能。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉-文本配对数据</span>**

![](../../images/视觉多模态讲义（下）-image-195.png)

分为两个子集： &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">VLM 预训练数据</span>**：来源于网络 alt-text 和图像标题，经过 CLIP 相似度过滤、分辨率/宽高比约束、文本长度限制及去重处理，并采用概念感知采样提升稀有类别覆盖。同时加入 OCR、图表和定位标注等结构化监督，强化空间与阅读理解能力。 &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">T2I 生成数据</span>**：包含高质量图文对及少量合成数据，强调 caption 风格多样性，如艺术性、超现实描述和图像质量，清晰度、结构完整性、语义丰富性，以提升生成的视觉保真度与风格表现力。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉-文本交错数据</span>**

配对数据难以建模多图像与中间文本的复杂交互，因此从**视频**和**网页**两大来源构建交错序列：

![](../../images/视觉多模态讲义（下）-image-196.png)

> * **<span style="color: rgb(36,91,219); background-color: inherit">视频数据</span>**：利用公开视频资源（如 Koala36M、MVImgNet2.0）捕获真实世界的时空动态。通过轻量级 captioning 模型（基于 Qwen2.5-VL-7B 微调）生成连续帧间的视觉变化描述（如动作、物体移动、场景切换），每段视频平均采样 4 帧，共构建 **4500 万条时序接地的交错序列**。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">网页数据</span>**：基于 OmniCorpus 等大规模网页语料，提取自然交错的图文文档（如教程、百科条目）。为提升图像生成的可控性，对每张图，先用 VLM 生成简洁描述并插入图前作为语义脚手架，引导模型在生成前形成概念预期；同时对过长的上下文文本进行 LLM 摘要压缩。最终构建 **2000 万条高质量交错网页文档**。

此外引入 **50 万条推理增强样本**，在 T2I 生成、自由图像编辑和概念性变换等任务中显式嵌入语言推理链（如“目标是什么 → 如何实现”），帮助模型将高层意图转化为具体视觉操作，显著提升生成的逻辑性与指令遵循能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

![](../../images/视觉多模态讲义（下）-image-221.png)

采用多阶段训练策略，动态混合上述数据，具体包括：**<span style="color: rgb(100,37,208); background-color: inherit">对齐阶段 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Alignment、</span>**<span style="color: rgb(100,37,208); background-color: inherit">预训练阶段 PT</span>**<span style="color: rgb(100,37,208); background-color: inherit">（Pre-training）、</span>**<span style="color: rgb(100,37,208); background-color: inherit">持续训练阶段 CT</span>**<span style="color: rgb(100,37,208); background-color: inherit">（Continued Training）和</span>**<span style="color: rgb(100,37,208); background-color: inherit">监督微调阶段 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SFT</span>：

> * **<span style="color: rgb(36,91,219); background-color: inherit">对齐</span>**：<span style="color: rgb(100,37,208); background-color: inherit">仅训练 MLP，冻结视觉编码器与语言模型</span>。仅使用图文对数据进行图像 captioning，每张图像调整为$$378 \times 378$$固定分辨率，匹配预训练 SigLIP2 输入尺寸。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**：<span style="color: rgb(100,37,208); background-color: inherit">向 LLM 添加 QK-Norm，除 VAE 外所有参数可训练</span>。训练语料含 2.5T token，包括文本、图文对、多模态对话、网页交错与视频交错数据。对多模态理解与生成均采用原生分辨率策略，限制每张图像的最长边与最短边。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">持续训练</span>**：相比预训练，提升视觉输入分辨率，这对生成与理解性能都比较重要，并<span style="color: rgb(100,37,208); background-color: inherit">提高交错数据采样比例，以强化跨模态推理学习，此时模型核心能力已更稳定</span>。持续训练阶段消耗约 2.6T token。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">监督微调</span>**：对多模态生成，<span style="color: rgb(100,37,208); background-color: inherit">从图文对与交错生成数据集中构建高质量子集</span>；对多模态理解，<span style="color: rgb(100,37,208); background-color: inherit">从 LLaVA-OV与 Mammoth-VL 指令微调数据中筛选子集</span>。总训练 token 为 72.7B。

所有训练阶段均使用 AdamW 优化器，$$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$。设$$\epsilon = 1.0 \times 10^{-15}$$以抑制 loss 尖峰。提升生成分辨率时，同步<span style="color: rgb(100,37,208); background-color: inherit">将扩散 timestep 从</span>$$1.0$$<span style="color: rgb(100,37,208); background-color: inherit">增至</span>$$4.0$$<span style="color: rgb(100,37,208); background-color: inherit">，以确保噪声水平分布合理</span>。PT、CT、SFT 阶段采用恒定学习率，便于在不重启训练的前提下轻松扩展数据规模。为保证各 rank 负载均衡，将每 rank 序列打包至窄长度范围：<span style="color: rgb(100,37,208); background-color: inherit">对齐与 PT 阶段：</span>**`32K–36K`**<span style="color: rgb(100,37,208); background-color: inherit"> token；CT 与 SFT 阶段：</span>**`40K–45K`**<span style="color: rgb(100,37,208); background-color: inherit"> token</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">数据采样比例</span>**

![](../../images/视觉多模态讲义（下）-image-220.png)

通过在 1.5B Qwen2.5 LLM 上调整多模态生成数据与理解数据的比例，进行系列控制实验。如上图，将生成数据采样比例从 50% **`1g1u`**&#x589E;至 80% **`4g1u`**，<span style="color: rgb(100,37,208); background-color: inherit">MSE 损失稳步下降，绝对值降低</span>**`0.4%`**<span style="color: rgb(100,37,208); background-color: inherit">，这对 Rectified Flow 是显著提升</span>。相比之下，<span style="color: rgb(100,37,208); background-color: inherit">交叉熵 CE 损失在不同比例下无一致规律</span>；**`4g1u`**<span style="color: rgb(100,37,208); background-color: inherit">与</span>**`2g1u`**<span style="color: rgb(100,37,208); background-color: inherit">在第 14000 步的最大差距</span>**`0.07`**<span style="color: rgb(100,37,208); background-color: inherit">对下游基准影响可忽略</span>。这些发现表明：**<span style="color: rgb(100,37,208); background-color: inherit">生成样本应远高于理解样本的采样频率</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">学习率</span>**

![](../../images/视觉多模态讲义（下）-image-223.png)

结果如上图，生成和理解数据比例1：1。<span style="color: rgb(100,37,208); background-color: inherit">较大学习率加速 MSE 损失收敛，较小学习率利于 CE 损失。为权衡此矛盾，为两个目标分配独立权重因子</span>。

## 5.4 <span style="color: rgb(36,91,219); background-color: inherit">Any-to-Any（Omni 全模态）</span>

### 5.4.1 <span style="color: rgb(36,91,219); background-color: inherit">AnyGPT</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

AnyGPT 是可统一训练的综合框架，由三个主要模块构成：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态分词器 multimodal tokenizer</span>**：将连续的非文本模态转换为离散的 token，随后这些 token 被组织成多模态交错序列
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">多模态大语言模型</span>**：通过 next token prediction 的目标对序列进行训练
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">多模态反分词器 multimodal de-tokenizer</span>**：在推理阶段，多模态 token 通过对应的反分词器被解码回其原始表示形式

![](../../images/视觉多模态讲义（下）-image-217.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">分词</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">图像分词器 Image Tokenizer</span>**

采用 SEED 分词器进行图像分词。其包含多个组件：<span style="color: rgb(100,37,208); background-color: inherit">ViT Encoder、Causal Q-Former、VQ Codebook、MLP 和 UNet Decoder</span>。

![](../../images/视觉多模态讲义（下）-image-215.png)

> 1. SEED 接收一张$$224 \times 224$$的 RGB 图像作为输入，ViT Encoder 将其编码为$$16 \times 16$$的图像 patch
>
> 2. Causal Q-Former 将这些块特征转换为$$32$$个 Causal Embedding
>
> 3. 一个包含$$8192$$个条目的 Codebook 将这些 Embedding 离散化为一串量化的 code
>
> 4. MLP 将视觉码解码为 Generation Embedding，该 Embedding 与预训练的 Stable Diffusion 的潜在空间对齐
>
> 5. UNet Decoder 将 Generation Embedding 还原为原始图像

**<span style="color: rgb(222,120,2); background-color: inherit">语音分词器 Speech Tokenizer</span>**

AnyGPT 采用 SpeechTokenizer 作为语音分词器，其采用基于<span style="color: rgb(216,57,49); background-color: inherit">残差向量量化 </span>**<span style="color: rgb(216,57,49); background-color: inherit">RVQ</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">R</span>**<span style="color: rgb(216,57,49); background-color: inherit">esidual </span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ector </span>**<span style="color: rgb(216,57,49); background-color: inherit">Q</span>**<span style="color: rgb(216,57,49); background-color: inherit">uantization）</span>的 Encoder-Decoder 架构。SpeechTokenizer <span style="color: rgb(100,37,208); background-color: inherit">使用每层含 1024 个 Codebook 条目的 8 层分层量化器将单通道音频序列压缩为离散矩阵，并达到 50 Hz 的帧率。第一层量化器捕获语义内容，第 2 至第 8 层则编码 paralinguistic 细节</span>。一段 10 秒的音频因此被转换为一个$$500 \times 8$$的矩阵，分为语义 token 和声学 token。AnyGPT 采用在 CommonVoice 和 LibriSpeech 数据集上预训练的 SpeechTokenizer 变体。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：在 AnyGPT 中，LLM 用于建模语义 token，而语音克隆模型则补充其余的副语言信息。因此，LLM 中语音词汇表的大小等于一个 Codebook 的大小，即$$1024$$。

**<span style="color: rgb(222,120,2); background-color: inherit">音乐分词器 Music Tokenizer</span>**

语音与音乐具有相似的数据格式，但内容差异显著，因此 AnyGPT 将它们视为独立模态，并使用专用分词器。对于音乐，采用 Encodec 作为音乐分词器，其是<span style="color: rgb(100,37,208); background-color: inherit">使用 RVQ 对潜在空间进行量化的卷积自编码器</span>。AnyGPT <span style="color: rgb(100,37,208); background-color: inherit">使用一个在 20K 首音乐曲目上预训练的现成 Encodec 变体，处理 32 kHz 单声道音频，帧率为 50 Hz</span>。其生成的 Embedding 通过 4 层 RVQ 进行量化，每层 Codebook 大小为$$2048$$，因此组合后的音乐词汇表大小为$$8192$$。AnyGPT 将 5 秒的音乐编码为$$250$$个潜在帧，最终生成一个$$250 \times 4$$的码矩阵。为使语言模型能够预测整段音乐片段，将这 4 层音乐码按帧逐帧展平为一个因果序列。语言模型首先预测第一帧的前 4 个 token，然后以类似方式继续预测后续帧。

* **<span style="color: rgb(36,91,219); background-color: inherit">LLM</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">扩展词汇表</span>**

为将多模态离散表示融入预训练 LLM，<span style="color: rgb(100,37,208); background-color: inherit">在原有词汇表基础上新增模态专用 token，并相应扩展 Embedding 层和预测层</span>。新引入的参数随机初始化。所有模态的 token 共同构成新的词汇表，<span style="color: rgb(100,37,208); background-color: inherit">各模态在语言模型中被训练以对齐到一个共享的表示空间</span>。该增强词汇表的大小记为$$V$$，其为所有模态词汇表大小之和：

$$V = \sum_{i=1}^{n} V_i$$

其中 $$V_i$$ 表示第 $$i$$ 个模态的词汇表大小。

**<span style="color: rgb(222,120,2); background-color: inherit">统一多模态语言模型</span>**

借助模态专用分词器，可以将多模态数据压缩为离散 token 序列，并通过下一个 token 预测损失对语言模型进行训练。这<span style="color: rgb(100,37,208); background-color: inherit">使得核心 LLM 能以自回归方式统一处理感知、理解、推理和生成等任务</span>。AnyGPT 采&#x7528;**`LLaMA-2 7B`**&#x4F5C;为主干模型，其在 2 TB 文本 token 上进行了预训练。<span style="color: rgb(100,37,208); background-color: inherit">除 Embedding 矩阵和预测层的形状调整外，语言模型的其余部分保持不变</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态生成</span>**

高质量多模态数据的生成是非常困难的，如<span style="color: rgb(220,155,4); background-color: inherit">高清图像和高保真音频</span>。这类数据<span style="color: rgb(216,57,49); background-color: inherit">通常需要大量比特才能准确表示，导致序列长度过长，而语言模型的计算复杂度随序列长度呈指数增长</span>。为了解决这个问题，AnyGPT 采用两阶段框架进行高保真生成：**<span style="color: rgb(100,37,208); background-color: inherit">语义信息建模</span>**<span style="color: rgb(100,37,208); background-color: inherit">与</span>**<span style="color: rgb(100,37,208); background-color: inherit">感知信息建模</span>**<span style="color: rgb(100,37,208); background-color: inherit">。首先，语言模型负责生成在语义层面已完成融合与对齐的内容；随后，非自回归模型将多模态语义 token 转换为感知层面的高保真多模态内容，在性能与效率之间取得平衡</span>。

> * **<span style="color: rgb(36,91,219); background-color: inherit">图像</span>**：使用与扩散模型潜在空间对齐的 <span style="color: rgb(100,37,208); background-color: inherit">SEED token 进行视觉语言建模</span>。语义级 SEED token 由扩散模型解码为高质量图像。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">语音</span>**：<span style="color: rgb(100,37,208); background-color: inherit">采用非自回归的掩码语言模型 SoundStorm 从语义 token 生成 SpeechTokenizer 的声学 token</span>。AnyGPT 训练了一个 SoundStorm 变体，其在 Multilingual LibriSpeech 数据集上使用 SpeechTokenizer 进行训练。随后，SpeechTokenizer 的解码器将所有语音 token 转换为原始音频数据。这种方法可以使 AnyGPT <span style="color: rgb(46,161,33); background-color: inherit">能够仅凭 3 秒语音提示即可复现任意说话人的声音，同时显著缩短 LLM 所需处理的语音序列长度</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音乐</span>**：<span style="color: rgb(100,37,208); background-color: inherit">使用 Encodec token 滤除人耳无法感知的高频细节</span>，再通过 Encodec 解码器将这些 token 重建为高保真音频数据。

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态数据</span>**

为支持任意模态到任意模态的生成，AnyGPT 构建了两类关键数据：**<span style="color: rgb(100,37,208); background-color: inherit">大规模对齐的预训练数据</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">高质量交错式指令微调数据</span>**。

1. **<span style="color: rgb(36,91,219); background-color: inherit">预训练数据：以文本为中心的跨模态对齐</span>**

由于天然对齐的多模态数据稀缺，采用<span style="color: rgb(100,37,208); background-color: inherit">以文本作为桥梁的策略，将所有模态分别与文本对齐，从而间接实现模态间对齐</span>。

> * **<span style="color: rgb(36,91,219); background-color: inherit">图像–文本</span>**：整合 LAION-2B、LAION-COCO、LAION-Aesthetics 和合成数据集 JourneyDB，经严格过滤后得到 3 亿高质量图文对；同时引入 MMC4 的 730 万交错文档，提升模型处理图文混合序列的能力。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">语音–文本</span>**：融合 Gigaspeech、Common Voice 和 MLS，共 57000 小时语音-文本对，覆盖多样口音、场景和录音条件。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音乐–文本</span>**：爬取超百万音乐视频，通过 Spotify API 匹配歌曲并提取元数据；利用 GPT-4 将噪声元数据转化为高质量文本描述，构建大规模音乐-文本对。

所有模态数据<span style="color: rgb(100,37,208); background-color: inherit">按 token 数量统一量化，并对低资源模态进行过采样，确保训练批次中各模态均衡</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">指令微调数据：合成多模态交错对话</span>**

![](../../images/视觉多模态讲义（下）-image-222.png)

现有指令数据极少包含三种以上模态。因此 AnyGPT 提出两阶段合成方法，构建了 10.8 万条高质量多轮多模态对话：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">文本级对话生成</span>**：用 GPT-4 生成包含图像、语音、音乐等模态描述的纯文本对话，共 100 个元主题 → 2 万具体主题 → 多轮对话。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">模态实例化</span>**：将文本描述转换为真实模态：图像使用 DALL·E 3、音乐使用 MusicGen、语音使用 Azure TTS &#x20;

最终数据集包含约 20.5 万图像、50.3 万语音片段和 11.3 万音乐片段。此外，我们还将 10 万条纯文本指令对话通过 TTS 转为语音对话，进一步增强语音交互能力。

> 注：这种合成策略<span style="color: rgb(46,161,33); background-color: inherit">高效解决了多模态指令数据稀缺问题，为模型支持复杂、交错的多模态人机交互奠定基础</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

采用多种模板构建多模态句子，以确保预训练数据的多样性。每种非文本模态内容，如<span style="color: rgb(220,155,4); background-color: inherit">图像、语音或音乐</span>，均<span style="color: rgb(100,37,208); background-color: inherit">通过成对的特殊 token 标识其起始与结束位置</span>。配对数据包含一个非文本模态$$X$$，例如<span style="color: rgb(220,155,4); background-color: inherit">图像、语音或音乐</span>，及其对应的文本$$T$$，如<span style="color: rgb(220,155,4); background-color: inherit">标题或转录文本</span>。这里利用 OpenAI GPT-4 生成数百条双向指令，涵盖$$X$$到文本和文本到$$X$$两类任务。

<span style="color: rgb(100,37,208); background-color: inherit">给定一个模态 token 序列</span>$$S$$<span style="color: rgb(100,37,208); background-color: inherit">和相关文本</span>$$T$$<span style="color: rgb(100,37,208); background-color: inherit">，从预定义的指令池中随机选择一个生成方向及对应指令</span>$$I$$<span style="color: rgb(100,37,208); background-color: inherit">，构成三元组</span>$$(I, S, T)$$。随后，根据生成方向，将该三元组按以下模板之一组织为训练序列：

> * **<span style="color: rgb(36,91,219); background-color: inherit">非文本 → 文本</span>**
>
> $$\texttt{[Human]: } \{I\}.\{S\}\texttt{<eoh>. [AnyGPT]: } \{T\}\texttt{<eos>.}$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">文本 → 非文本</span>**
>
> $$\texttt{[Human]: } \{I\}. \text{This is input:}\{T\}\texttt{<eoh>. [AnyGPT]: } \{S\}\texttt{<eos>.}$$

对于图文交错的多模态数据,如网页文档中穿插的图像与文本，<span style="color: rgb(100,37,208); background-color: inherit">直接将非文本内容替换为其对应的 token 序列,因其天然构成连贯语句，无需额外模板</span>。由于大部分图像和音乐数据源自网络，存在一定噪声，可能影响多模态生成质量。因此<span style="color: rgb(100,37,208); background-color: inherit">在初始预训练完成后，进一步采用高质量子集进行精调</span>：

> * **<span style="color: rgb(36,91,219); background-color: inherit">文本到图像生成</span>**：使用 **JourneyDB** 和 **LAION-Aesthetics**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图像描述生成</span>**：使用 **LAION-COCO**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音乐任务</span>**：引入 **AnyInstruct-108k** 数据集

![](../../images/视觉多模态讲义（下）-image-213.png)

其余数据保持不变，模型在此基础上继续预训练 **4000 步**。详细训练超参数如上表。

* **<span style="color: rgb(36,91,219); background-color: inherit">指令微调</span>**

![一个多模态对话数据：输入是一张图片和一个语音指令用于生成音乐。输出是符合要求的音乐，以及相应的文本和语音回复。所有数据都被处理成离散的标记，并由 LLM 进行自回归处理](../../images/视觉多模态讲义（下）-image-214.png)

### 5.4.2 <span style="color: rgb(36,91,219); background-color: inherit">NExT-GPT</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

<span style="color: rgb(216,57,49); background-color: inherit">现有的 MLLM 大多只能处理多模态输入，而无法以多种模态生成内容，限制了其在实际应用中的灵活性和功能</span>。人类在感知世界和与他人交流时会使用多种模态，因此开发能够接受和输出任何模态内容的 MLLM 对于实现类人水平的人工智能至关重要。尽管有一些工作尝试实现类似人类的任意模态转换，但它们<span style="color: rgb(216,57,49); background-color: inherit">要么缺乏 LLM 的核心推理和决策能力，要么受限于简单的配对内容生成</span>。此外，一些系统由于完全基于管道架构，信息传递完全依赖于 LLM 产生的离散文本，这不可避免地会<span style="color: rgb(216,57,49); background-color: inherit">引入噪声并传播错误，且整个系统仅利用现有的预训练工具进行推理，缺乏整体的端到端训练</span>，限制了内容理解和多模态生成的能力。为了克服上述局限性，需要构建一个能够处理任意模态输入和输出的端到端 MLLM，以实现更自然、更灵活的人机交互。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

NExT-GPT 包含三个主要阶段：**<span style="color: rgb(100,37,208); background-color: inherit">编码</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">LLM 理解与推理</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">解码</span>**。如下图：

![](../../images/视觉多模态讲义（下）-image-219.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态编码</span>**

<span style="color: rgb(100,37,208); background-color: inherit">利用现有成熟模型对多种模态的输入进行编码</span>。针对不同模态，有多种可选的编码器，例如<span style="color: rgb(220,155,4); background-color: inherit">视觉的</span>**`CLIP`**<span style="color: rgb(220,155,4); background-color: inherit">、语音的</span>**`HuBERT`**&#x7B49;。NExT-GPT 采&#x7528;**`ImageBind`**，<span style="color: rgb(100,37,208); background-color: inherit">一个统一的高性能编码器，支持六种模态</span>。借助 ImageBind，可以不需要管理大量异构的模态专用编码器。随后，通过一个投影层将不同模态的输入表示映射为类语言表示，使其能被LLM理解。

* **<span style="color: rgb(36,91,219); background-color: inherit">LLM 理解与推理</span>**

采用 Vicuna 7B，一个开源的 LLM，LLM接收来自不同模态的表示，并对其进行语义理解与推理。输出包括：

> * 直接生成的文本回答
>
> * 各模态的信号 token ，指示解码层是否生成多模态内容，以及在需要时生成何种内容

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态生成</span>**

在接收到 LLM 发出的带有特定指令的多模态信号后，基于 Transformer 的<span style="color: rgb(100,37,208); background-color: inherit">输出投影层将这些信号 token 的表示映射为后续多模态解码器可理解的形式</span>。NExT-GPT 采用现成的、基于潜在空间条件的扩散模型进行不同模态的生成：

> * **<span style="color: rgb(36,91,219); background-color: inherit">图像合成</span>**&#x4F7F;用 Stable Diffusion-v1.5
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视频合成</span>**&#x4F7F;用 Zeroscope-v2
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音频合成</span>**&#x4F7F;用 AudioLDM

经过一个投影层后，信号表示被送入条件扩散模型以生成内容。

![](../../images/视觉多模态讲义（下）-image-212.png)

在整个系统中，仅需在后续学习过程中更新参数量较小的输入和输出投影层，其余所有编码器和解码器均保持冻结。<span style="color: rgb(46,161,33); background-color: inherit">需更新的参数量为</span>**`155M`**<span style="color: rgb(46,161,33); background-color: inherit">（28+33+31+31+32），而总参数量为</span>**`155M + 12.275B`**<span style="color: rgb(46,161,33); background-color: inherit">（1.2+7+1.3+1.8+0.975），即仅有约</span>**`1%`**<span style="color: rgb(46,161,33); background-color: inherit">的参数需要更新</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态对齐学习</span>**

NExT-GPT 设计的是一个松耦合的三层系统，仅需更新编码端和解码端的两个投影层。

1. **<span style="color: rgb(36,91,219); background-color: inherit">编码端对齐</span>**

之前的一些多模态大模型采用基于 Transformer 架构的多模态编码器，生成网格状的块级特征，例如图像、音频或视频的 patch 特征，并通过线性层将其直接投影到文本特征空间，以使其能被 LLM 理解。但<span style="color: rgb(216,57,49); background-color: inherit">基于块的特征单元可能无法很好地契合复杂的文本 token 语义，因为语言 token 通常封装的是离散的概念。可能导致多模态大模型中的信息感知次优</span>。因此 NExT-GPT <span style="color: rgb(100,37,208); background-color: inherit">设计了一种可学习的概念 token，通过分组机制将网格级特征分层聚合为语义概念 token ，再将这种概念表示输入LLM</span>。

为了实现对齐，<span style="color: rgb(100,37,208); background-color: inherit">在现有语料库和基准数据集上的</span>**`X-caption pair`**<span style="color: rgb(100,37,208); background-color: inherit">数据上训练一个</span>**`X-to-text`**<span style="color: rgb(100,37,208); background-color: inherit">生成任务，其中</span>**`X`**<span style="color: rgb(100,37,208); background-color: inherit">代表图像、音频或视频</span>：给定一个 X 的表示，使 LLM 生成对应的文本描述。使用三类 X-caption pair 数据：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">视频-描述对</span>**：WebVid-2M，一个大规模短视频数据集，文本描述来源于素材网站
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">图像-描述对</span>**：CC3M，包含超过300万张图像，配有风格多样的自然语言描述
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">音频-描述对</span>**：AudioCaps，一个包含约4.6万条音频片段的数据集，每条配有人工撰写的文本描述，通过众包收集

训练流程如下：

![](../../images/视觉多模态讲义（下）-image-216.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">解码端对齐</span>**

集成了来自外部的预训练条件扩散模型。主要目标是使扩散模型与 LLM 输出的指令对齐。<span style="color: rgb(216,57,49); background-color: inherit">若对每个扩散模型与 LLM 进行全面对齐，计算开销巨大</span>。NExT-GPT 使用一种更高效的方法：**<span style="color: rgb(100,37,208); background-color: inherit">解码端指令跟随对齐</span>**：

![](../../images/视觉多模态讲义（下）-image-218.png)

这里不直接输出纯文本指令，而是设计三类特殊 token：

> * **<span style="color: rgb(36,91,219); background-color: inherit">图像信号 token</span>**：$$[\text{IMG}_i]$$（$$i = 0, \dots, 4$$）
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音频信号 token</span>**：$$[\text{AUD}_i]$$（$$i = 0, \dots, 8$$）
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视频信号 token</span>**：$$[\text{VID}_i]$$（$$i = 0, \dots, 24$$）

这些 token 隐式地携带丰富灵活的指令，用于指导下游扩散模型。目标是希望 LLM 能同时学习生成文本内容和模态信号 token：<span style="color: rgb(100,37,208); background-color: inherit">当 LLM 判断需生成某模态内容时，输出对应类型的特殊 token 以激活该模态；否则不输出该 token ，表示该模态未被激活</span>。

一般来说扩散模型仅基于文本导向的文本编码器的表示进行条件生成。但这种以文本为中心的条件机制与 NExT-GPT 中 LLM 的模态信号 token 存在显著差异，导致扩散模型难以准确解读 LLM 的指令。因此一方面<span style="color: rgb(100,37,208); background-color: inherit">将 LLM 输出的经 Transformer 投影层后的模态信号 token 表示作为去噪过程中的条件输入，引导扩散模型生成合适的图像、视频或音频</span>；另一方面<span style="color: rgb(100,37,208); background-color: inherit">最小化投影后的信号 token 表示与扩散模型中文本编码器生成的条件文本表示之间的距离，以加速对齐学习</span>。所有 U-Net 扩散主干网络均保持冻结，从而确保训练极为轻量。

在对齐训练阶段，以 CC3M、WebVid 和 AudioCaps 中的描述文本作为输入，将其与信号 token 拼接作为输出。损失函数包含三个关键部分：

> 1. 生成信号 token 的负对数似然损失
>
> 2. 描述对齐损失：LLM 生成的信号 token 隐藏状态与扩散模型文本编码器生成的条件文本表示之间的$$\ell_2$$距离
>
> 3) 条件潜在去噪损失

* **<span style="color: rgb(36,91,219); background-color: inherit">模态切换指令微调</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">指令微调</span>**

要使整个系统忠实理解并遵循用户指令、生成期望的多模态输出，仍存在差距。因此进一步进行<span style="color: rgb(216,57,49); background-color: inherit">指令微调 </span>**<span style="color: rgb(216,57,49); background-color: inherit">IT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（ </span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">nstruction </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">uning）</span>，以增强 LLM 的能力与可控性。IT 通过使&#x7528;**`(input, output) pair`**&#x5BF9;整个模型进行额外训练，<span style="color: rgb(100,37,208); background-color: inherit">采用 LoRA 训练，仅更新 NExT-GPT 中一小部分参数，并与两层投影层联合优化</span>。

![](../../images/视觉多模态讲义（下）-image-211.png)

当一个样本输入系统时，LLM 重构并生成输入的文本内容，同时用多模态信号 token 表示多模态内容。将输出投影编码的模态信号 token 表示与扩散条件编码器编码的多模态标题表示进行对齐。除 LLM 微调外，我们<span style="color: rgb(100,37,208); background-color: inherit">还对 NExT-GPT 的解码端进行微调：将输出投影层编码的模态信号 token 表示与扩散条件编码器编码的标准多模态描述表示对齐</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">指令数据集</span>**

为支持 NExT-GPT 在 any-to-any 场景下的多模态生成能力，构建&#x4E86;**`Text→Text+X`**&#x6307;令数据集。基于现有大规&#x6A21;**`X-caption pair`**，如 CC3M、WebVid、AudioCaps 等，<span style="color: rgb(100,37,208); background-color: inherit">利用模板和 GPT-4 自动生成多样化的用户指令，将原始描述包装为符合交互场景的 Instruction-Output 对</span>。

但现有 IT 数据集多局限于单模态输出且对话简短，难以满足动态多模态交互需求。为此，NExT-GPT 提出 **<span style="color: rgb(216,57,49); background-color: inherit">MosIT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Mo</span>**<span style="color: rgb(216,57,49); background-color: inherit">dality-</span>**<span style="color: rgb(216,57,49); background-color: inherit">s</span>**<span style="color: rgb(216,57,49); background-color: inherit">witching </span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">nstruction </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">uning）</span>数据集：通过人工设&#x8BA1;**`Human–Machine`**&#x5BF9;话模板，<span style="color: rgb(100,37,208); background-color: inherit">引导 GPT-4 生成涵盖100+主题、3–7轮的多轮对话，要求模态在输入输出端交替切换，并包含感知、推理、规划等复杂行为</span>。生成的图像、音频、视频多模态内容通过检索系统或 AIGC 工具（如 Stable-XL、Midjourney）匹配补充。经人工筛选后，最终获得5K条高质量、多模态、多轮交互样本，显著优于现有IT数据集的复杂性与真实性。

![训练超参数，阶段 1：编码端对齐学习，阶段 2：解码端对齐学习，阶段 3：端到端指令调优](../../images/视觉多模态讲义（下）-image-210.png)

**<span style="color: rgb(46,161,33); background-color: inherit">优点</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">端到端的任意模态输入输出能力</span>**：NExT-GPT 是首个端到端的通用 MLLM，<span style="color: rgb(46,161,33); background-color: inherit">能够处理文本、图像、视频和音频等模态的任意组合输入和输出</span>，为实现更自然、更灵活的人机交互提供了可能。

**<span style="color: rgb(36,91,219); background-color: inherit">轻量级对齐学习技术</span>**：通过编码端的 LLM 中心对齐和解码端的指令跟随对齐，<span style="color: rgb(46,161,33); background-color: inherit">仅需调整</span>**`1%`**<span style="color: rgb(46,161,33); background-color: inherit">参数，即可实现有效的语义对齐，降低了训练成本，同时便于未来扩展到更多潜在模态</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">高质量的模态切换指令微调数据集</span>**：人工收集和标注的 MosIT 数据集涵盖了各种模态组合的复杂指令，有助于增强 MLLM 的跨模态语义理解和内容生成能力，使其更接近人类水平的交互能力。

**<span style="color: rgb(216,57,49); background-color: inherit">缺点</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">模态生成质量的局限性</span>**：尽管 NExT-GPT 在多模态生成方面表现出色，但在某些任务上，其<span style="color: rgb(216,57,49); background-color: inherit">生成质量可能受到扩散模型能力的限制，尤其是在文本条件下的模态编辑任务中</span>，可能不如一些专门针对特定模态生成的模型。

**<span style="color: rgb(36,91,219); background-color: inherit">模态扩展的潜力</span>**：目前系统仅支持四种模态：文本、图像、视频和音频，未来需要扩展到更多模态，如网页、3D视觉、热图、表格和图形等，以提高系统的通用性。

**<span style="color: rgb(36,91,219); background-color: inherit">LLM变体的多样性</span>**：目前仅实现&#x4E86;**`7B`** Vicuna 版本的 LLM，未来可以考虑整合更多类型的 LLM 及其不同大小的变体，以满足不同用户的需求。

**<span style="color: rgb(36,91,219); background-color: inherit">多模态生成策略的改进空间</span>**：可以探索将检索式方法与生成式方法相结合，以提高系统在多模态生成任务中的性能。

### 5.4.3 **<span style="color: rgb(36,91,219); background-color: inherit">Unified-IO 2</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">统一的序列建模框架</span>**

Unified-IO 2 使用一个 Encoder-Decoder Transformer 处理文本、图像、音频、视频历史、稀疏结构、稠密视觉预测与具身动作。所有任务都被改写为“多模态输入序列到多模态目标序列”的条件生成：Encoder 接收由不同模态表示拼接而成的输入，Decoder 逐 Token 生成文本 Token、图像码本索引、音频码本索引、坐标 Token 或动作 Token。模型不再为分类、检测、深度估计、音频生成分别设置任务头，任务差异由输入中的模态标记、训练范式标记和自然语言指令共同表达。

核心条件分布写成：

$$p_{\theta}(\mathbf{y}\mid\mathbf{x})=\prod_{t=1}^{T_y}p_{\theta}(y_t\mid y_{<t},\mathbf{x})$$

其中 $$\mathbf{x}$$ 可以同时包含文本、当前图像、音频以及历史图像/音频，$$\mathbf{y}$$ 可以是任何一种离散化目标。<span style="color: rgb(100,37,208); background-color: inherit">统一的不是原始数据形态，而是进入 Transformer 之后的 Token 接口与自回归预测接口。</span>连续视觉和声学信号先经过预训练模态 Encoder，生成任务需要输出的信号再经过离散 Tokenizer 变成有限词表中的索引；稀疏几何结构和动作则直接量化为专用 Token。

* **<span style="color: rgb(36,91,219); background-color: inherit">Encoder-Decoder 主干</span>**

主干由 **`24`** 层 Encoder 和 **`24`** 层 Decoder 组成，提供 L、XL、XXL 三个规模，参数量分别约为 **`1.1B`**、**`3.2B`** 和 **`6.8B`**。对应隐藏维度为 **`1024`**、**`2048`**、**`3072`**，注意力头数为 **`16`**、**`16`**、**`24`**。各模态输入都先投影到主干隐藏维度，Decoder 输出再通过共享输出投影映射到统一词表。

图像和音频历史最多各保留 **`4`** 个片段。每个片段先走与当前输入相同的 ViT 或 AST，再经 Perceiver Resampler 压缩为固定长度：每张历史图像压成 **`32`** 个潜在 Token，每段历史音频压成 **`16`** 个潜在 Token。<span style="color: rgb(46,161,33); background-color: inherit">固定长度重采样把多轮多模态上下文的开销从原始 Patch 数量中解耦，使历史信息可以直接拼入统一上下文。</span>

![Unified-IO 2 的统一 Encoder-Decoder 架构与多模态输入输出路径](../../images/视觉多模态讲义（下）-uio2-architecture.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">文本与结构化目标表示</span>**

文本使用 LLaMA 的 SentencePiece BPE Tokenizer，基础词表为 **`32K`**。在此基础上增加 **`200`** 个 Span Mask Token，用于 UL2 风格的文本破坏与恢复；另设 **`10`** 个特殊 Token，指向输入中的图像、音频和历史片段。单个样本的文本输入与文本输出最长均为 **`512`** 个 Token。

点、边界框、相机位姿、三维包围盒等稀疏结构共享 **`1000`** 个位置 Token。二维点用两个量化坐标表示，二维框用四个坐标表示；三维包围盒使用 **`12`** 个离散量，依次编码投影中心、虚拟深度、对数归一化后的三维尺寸以及连续自我中心旋转。导航动作直接写成离散文本动作，机械臂操作把位置增量、旋转增量和夹爪状态分别量化到同一组位置 Token。<span style="color: rgb(100,37,208); background-color: inherit">这套表示让检测、关键点、三维理解和机器人控制都落入“生成一串结构 Token”的统一接口。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">图像输入与图像输出</span>**

图像输入由在 LAION-2B 上训练的 ViT-B 编码。为同时保留低层纹理和高层语义，取 ViT 第 **`2`** 层与倒数第 **`2`** 层特征，在通道维拼接后线性投影到主干维度。输入分辨率最高为 **`384×384`**，形成 **`24×24=576`** 个视觉 Token。

图像生成目标由 Dense VQ-GAN 离散化。**`256×256`** 图像按 **`8×8`** Patch 压成 **`32×32=1024`** 个索引，码本大小为 **`16512`**。深度、表面法线和分割等稠密预测也被转成 RGB 图像：深度先归一化为灰度图，法线的 $$x/y/z$$ 分量映射到 $$r/g/b$$，目标实例分割则输出由类别和框指定的二值掩码。这样，稠密视觉任务与普通图像生成共用同一个离散解码接口。

* **<span style="color: rgb(36,91,219); background-color: inherit">音频输入与音频输出</span>**

每段音频最长为 **`4.08 s`**，以 **`16 kHz`** 采样。波形一次性变换为 log-Mel 频谱：FFT 窗长 **`1024`** 个采样点，Hop Length 为 **`256`**，频率范围 $$[0,8000]$$ Hz，最终得到 **`128×256`** 的频谱。音频输入由 AST 编码，同样拼接第 2 层和倒数第 2 层特征后投影到主干维度。

音频生成使用 ViT-VQGAN，把 **`256×128`** 频谱按 **`8×8`** Patch 量化为 **`512`** 个 Token，音频码本包含 **`8196`** 个离散码字；计入特殊符号后，音频输出词表为 **`8320`**。生成的码本索引先还原为频谱，再由 HiFi-GAN 声码器转换为波形。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练样本的统一构造</span>**

一个原始样本可能同时带有文本、图像帧、音轨、深度、框和动作，但训练时不会把全部字段机械地塞进输入。样本构造按五步执行：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">选择目标模态</span>**：从当前样本真实存在的模态中随机选一个作为需要恢复或生成的目标。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">选择输入模态</span>**：目标模态可以从输入中移除，也可以保留一个被破坏的版本；其他模态随机保留或屏蔽。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">选择训练范式</span>**：根据目标类型选择 Span Corruption、因果生成、极端 Span Corruption 或模态去噪。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">生成输入 Mask</span>**：为文本、图像或音频构造与当前训练范式匹配的掩码形状和掩码比例。
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">添加前缀</span>**：把目标模态 Token 与范式 Token 写到输入前缀，显式告诉模型要生成什么、按哪种规则生成。

<span style="color: rgb(100,37,208); background-color: inherit">目标模态、条件模态、破坏方式和前缀共同定义任务；自然语言指令负责描述语义要求，而不是独自承担任务路由。</span>例如视频样本可提取连续帧、对应音频频谱与转写文本，把音频设为目标，保留文本和图像历史，选择音频掩码去噪，再以 `[Audio][R]` 前缀组成最终输入。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练数据混合</span>**

预训练采样分布为：自然语言 **`33%`**，图文数据 **`40%`**，视频与音频 **`25%`**，三维与具身数据 **`1%`**，自动构造的增强数据 **`1%`**。增强数据主要补足稠密和稀疏标注：一类使用自动分割结果生成按点或框定位的分割任务，另一类在图像中合成几何形状，要求输出形状边界框或数量。

指令微调阶段覆盖 **`220`** 个任务，来自超过 **`120`** 个数据集。与预训练相比，这一阶段减少随机任务组合，改为使用明确 Prompt，把理解、生成、编辑、定位、三维和具身任务统一成可直接调用的指令格式。

![从视频样本选择目标模态、条件模态、训练范式和输入掩码的构造过程](../../images/视觉多模态讲义（下）-uio2-sample-construction.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Multimodal Mixture of Denoisers</span>**

训练目标扩展 UL2 的 Mixture of Denoisers。文本包含三种范式：`[R]` 表示标准 Span Corruption，`[S]` 表示从左到右的因果语言建模，`[X]` 表示掩码比例更高、跨度更长的 Extreme Span Corruption。图像与音频使用两种对应范式：`[R]` 随机遮挡输入 Patch 并重建完整目标，`[S]` 则完全移除目标模态，只根据其他输入模态从头生成。

所有范式仍最小化同一个自回归负对数似然：

$$\mathcal{L}_{\mathrm{NLL}}=-\sum_{t=1}^{T_y}\log p_{\theta}(y_t\mid y_{<t},\tilde{\mathbf{x}},m,o)$$

其中 $$\tilde{\mathbf{x}}$$ 是经过破坏或删减的输入，$$m$$ 表示目标模态，$$o$$ 表示训练范式。模态前缀和范式前缀使同一组参数能够区分“根据被遮挡图像做重建”与“根据文本从头生成图像”。

* **<span style="color: rgb(36,91,219); background-color: inherit">Decoder 动态 Mask</span>**

图像和音频去噪若直接套用普通自回归 Teacher Forcing，会发生目标泄漏。设当前要预测第 $$t$$ 个目标 Token，Decoder 输入中通常包含右移后的真实序列；当目标模态的一部分同时出现在 Encoder 端时，Decoder 可能从历史输入中读到本应被遮挡的信息。完全屏蔽所有 Decoder 目标 Token 虽能消除泄漏，却会让生成范式与去噪范式相互冲突。

<span style="color: rgb(100,37,208); background-color: inherit">动态 Mask 只在“预测某个 Token 的那一步”开放该 Token 对应的 Decoder 输入，其余时间保持屏蔽。</span>这样，当前位置仍能使用严格因果上下文，后续位置也不会提前看到被遮挡目标。对于二维图像和频谱，Decoder 还组合行形、列形与卷积邻域形状的稀疏 Mask，使可见上下文更贴近二维局部结构。

<span style="color: rgb(46,161,33); background-color: inherit">这套 Mask 在一个 Decoder 内兼容从头生成与掩码去噪，同时阻断由 Teacher Forcing 引入的信息捷径。</span>

![普通自回归、完全掩码与动态掩码三种 Decoder 训练方式](../../images/视觉多模态讲义（下）-uio2-dynamic-masking.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态稳定化设计</span>**

普通一维 RoPE 只按序列位置旋转 Query 与 Key，无法区分二维 Patch 的行列关系。二维 RoPE 把每个注意力头的 Query/Key 通道均分为两半，分别使用行坐标和列坐标旋转：

$$\operatorname{RoPE}_{2D}(q_{h,w})=\left[\operatorname{RoPE}(q^{(1)}_{h,w},h);\operatorname{RoPE}(q^{(2)}_{h,w},w)\right]$$

当前图像和音频频谱使用二维坐标；历史片段把第一维换成时间片段索引，第二维表示 Perceiver 潜在位置。<span style="color: rgb(100,37,208); background-color: inherit">同一个位置编码算子因此同时表达图像空间、频谱时间-频率结构和多轮历史顺序。</span>

不同模态混合后，注意力 Logit 容易出现极端值。主干在点积前对 Query 与 Key 分别做 LayerNorm，Perceiver Resampler 使用 Scaled Cosine Attention，注意力 Logit 统一以 FP32 计算。ViT 与 AST 在预训练阶段冻结，只在指令微调末段解冻。<span style="color: rgb(216,57,49); background-color: inherit">如果一开始同时更新模态 Encoder 与大规模主干，异质 Token 的尺度漂移会进一步放大训练不稳定。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">动态 Packing 与训练日程</span>**

样本经过 ViT、AST 和 Perceiver 后长度差异很大，因此 Packing 放在模态 Encoder 之后、主干 Transformer 之前执行。系统维护一个大小为 **`10`** 的候选池，为新样本寻找满足总长度约束且合并后利用率最高的已有 Pack；Pack 内使用 Block-Diagonal Attention Mask，禁止不同样本相互注意。预训练阶段 Encoder/Decoder 的 Pack 长度分别为 **`864`** 和 **`1280`**，单样本最大输入/目标长度分别为 **`1152`** 和 **`2048`**。

优化器为 Adafactor，前 **`5000`** 步线性 Warmup，之后学习率按 $$1/\sqrt{k}$$ 衰减；动量参数为 $$\beta_1=0.9$$、$$\beta_2=1-k^{-0.8}$$，梯度范数裁剪为 **`1.0`**。总训练 **`3M`** 步，其中预训练和指令微调各 **`1.5M`** 步。

### 5.4.4 <span style="color: rgb(36,91,219); background-color: inherit">M2-omni</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

M2-omni 构建了一个统一框架，同时支持多模态理解与生成任务，通过解耦的架构设计，最小化不同模态任务之间的相互干扰。这里的编码流程受到 Unified-IO2 的启发，其<span style="color: rgb(100,37,208); background-color: inherit">利用一个模态感知编码器，将图像、文本、音频和视频等多种输入映射到一个共享的 token 表示空间中</span>。之前 Janus 说过，多模态理解与生成任务之间可能存在相互干扰，主要源于图像理解与生成所需的信息粒度存在显著差异。M2-Omni 与 Janus 采用独立视觉编码路径不同的是，<span style="color: rgb(100,37,208); background-color: inherit">在图像生成任务中以文本描述作为中间表示，有效避免了在潜在图像特征进行直接对齐</span>。对于语音生成，<span style="color: rgb(100,37,208); background-color: inherit">采用基于离散 token 预测的方法，实现实时流式音频合成，同时最小化对其他模态分支性能的影响</span>。模型结构如下：

![](../../images/视觉多模态讲义（下）-image-229.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉编码器 Vision Encoder</span>**

在 M2-omni 中，视觉编码器从图像或完整视频中提取表示。采&#x7528;**`NaViT`**，能够处理任意分辨率的视频和图像。为减少视觉 token 的长度，<span style="color: rgb(100,37,208); background-color: inherit">将相邻的</span>$$2 \times 2$$<span style="color: rgb(100,37,208); background-color: inherit">个 token 拼接为一个 token，并使用 MLP 将其维度压缩回原始维度，从而对视觉表示进行下采样</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">音频编码器 Audio Encoder</span>**

采&#x7528;**`SAN-M`**&#x7F16;码器提取音频 token。随后<span style="color: rgb(100,37,208); background-color: inherit">对音频编码器输出应用</span>$$1 \times 3$$<span style="color: rgb(100,37,208); background-color: inherit">的平均池化操作，将每三个相邻 token 聚合为一个 token，从而减少音频 token 的总数</span>。为适应音频 token 序列长度的可变性，使用特殊 token **`<audio_pad>`**<span style="color: rgb(100,37,208); background-color: inherit">对压缩后的音频序列进行填充</span>，确保所有序列具有统一长度。

* **<span style="color: rgb(36,91,219); background-color: inherit">LLM</span>**

M2-omni LLM 融合多模态信息，并输出用于统一多模态理解与生成的解码器 Embedding。这里<span style="color: rgb(100,37,208); background-color: inherit">以</span>**`Llama3.1-8B`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`Llama3.3-70B`**<span style="color: rgb(100,37,208); background-color: inherit">的预训练权重初始化</span>。为实现文本、图像、视频和音频模态的统一位置编码，在推理时能泛化至更长序列，<span style="color: rgb(100,37,208); background-color: inherit">将 Llama 中原有的 </span>**<span style="color: rgb(100,37,208); background-color: inherit">1D-RoPE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 替换为</span>**`M-RoPE`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像生成器 Image Generator</span>**

为解耦生成与理解的表示空间，在图像生成任务中使用文本描述作为中间表示。训练阶段，<span style="color: rgb(100,37,208); background-color: inherit">将图像标题用两个特殊 token </span>**`<gen_image>`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`</gen_image>`**<span style="color: rgb(100,37,208); background-color: inherit">包裹，使模型能以灵活且无约束的方式生成用于图像生成的文本描述</span>。推理阶段，M2-omni LLM 生成文本描述，<span style="color: rgb(100,37,208); background-color: inherit">被上述两个特殊 token 包围的生成标题将作为图像生成的文本条件，采用离线的 Stable Diffusion 模型 作为图像生成器</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">音频解码器 Audio Decoder</span>**

采用端到端方式，利用 M2-omni LLM 预测离散音频 token 用于语音生成。<span style="color: rgb(100,37,208); background-color: inherit">预测出的离散音频 token 随后输入预训练的 CosyVoice flow matching 与 vocoder 模型，以生成音频流</span>。由于音频离散 token 与语言 token 在形式上的相似性，可以复用 M2-omni LLM 的模型结构来支持音频生成任务，从而与多模态理解任务兼容。

下表是 M2-omni 的详细模型配置：

![](../../images/视觉多模态讲义（下）-image-230.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">训练与对齐</span>**

给定一个多模态数据集，采用模态感知编码器将包括图像、文本、音频和视频在内的多种模态输入投影到一个统一的 token 表示空间中。形式上，输入的多模态序列记为$$\mathbf{x} = (x_1, \dots, x_\ell)$$，其中$$\ell$$表示序列长度，每个$$x_i$$对应一个模态输入 token，如<span style="color: rgb(220,155,4); background-color: inherit">图像、文本、音频或视频</span>。<span style="color: rgb(100,37,208); background-color: inherit">以自回归方式建模多模态序列的联合概率分布</span>，即每个 token 以先前所有 token 为条件，如下式所示：

$$\log p_\theta(\mathbf{x}) = \sum_{i=s}^{\ell-1} \log p_\theta(x_{i+1} \mid x_0, \dots, x_i)$$

其中$$s$$表示离散输出 token 的起始索引，仅$$x_{>s}$$被视为建模目标，$$\theta$$表示模型参数。M2-omni 提出一种<span style="color: rgb(100,37,208); background-color: inherit">多阶段训练框架，通过逐步引入多模态知识，实现渐进式的模态对齐</span>。整体训练流程包含三个主要阶段：**<span style="color: rgb(100,37,208); background-color: inherit">预训练 pre-training</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">指令微调 instruction tuning</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 和</span>**<span style="color: rgb(100,37,208); background-color: inherit">对齐微调 alignment tuning</span>**。其中，预训练和指令微调阶段又进一步细分为三个子阶段，每个子阶段旨在逐步引入新的模态，每个阶段的详细参数如下：

![](../../images/视觉多模态讲义（下）-image-228.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">预训练 Pre-training</span>**

预训练阶段主要聚焦于将多种模态与 M2-omni LLM 对齐，使其能够捕获多模态概念表示并发展跨模态感知能力。

![](../../images/视觉多模态讲义（下）-image-225.png)

**<span style="color: rgb(222,120,2); background-color: inherit">编码器对齐</span>**

利用<span style="color: rgb(100,37,208); background-color: inherit">图像-文本对、OCR 数据和音频-文本对进行训练，实现视觉和音频编码器与 M2-omni LLM 的对齐</span>。通过将多个图像-文本对拼接为单一交错序列，增强上下文理解能力，并实现$$1.5$$倍的训练效率提升。

**<span style="color: rgb(222,120,2); background-color: inherit">图-文知识增强</span>**

使用 Stage-1 中筛选的<span style="color: rgb(100,37,208); background-color: inherit">高质量图文对和 OCR 数据进行训练</span>，提升图像-文本细粒度理解能力，为 Stage-3 的交错图文和视频理解任务奠定基础。同时<span style="color: rgb(100,37,208); background-color: inherit">引入纯文本数据以防止 M2-omni LLM 语言理解能力退化</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">多模态联合训练</span>**

整合全模态知识，促进全模态对齐与统一表示学习。引入<span style="color: rgb(100,37,208); background-color: inherit">高质量图文对、视频-文本对、交错图文序列、音频-文本对及纯文本数据，进行端到端多模态预训练</span>。采&#x7528;**`step balance strategy`**&#x5E73;衡不同模态的收敛速度

* **<span style="color: rgb(36,91,219); background-color: inherit">指令微调 Instruction Tuning</span>**

指令微调使模型更好地理解用户指令并完成指定任务。

![](../../images/视觉多模态讲义（下）-image-226.png)

**<span style="color: rgb(222,120,2); background-color: inherit">图文指令微调</span>**

提升模型在图像模态上的指令遵循能力，尤其针对科学、OCR、文档和图表等预训练阶段未充分学习的专项任务。

**<span style="color: rgb(222,120,2); background-color: inherit">视觉指令微调</span>**

全面提升模型在视觉模态上的综合能力，包括图像-文本、视频-文本及交错图像-文本理解。



**<span style="color: rgb(222,120,2); background-color: inherit">全模态指令微调</span>**

进一步整合音频模态与生成任务，使模型能够处理混合多模态序列的指令。提&#x51FA;**`dynamic adaptive balance strategy`**&#x4EE5;在所有模态上同时达到最优性能

* **<span style="color: rgb(36,91,219); background-color: inherit">对齐微调 Alignment Tuning</span>**

聚焦于优化对话交互的质量与风格一致性，确保模型在所有模态上保持高水准表现。<span style="color: rgb(100,37,208); background-color: inherit">尽管指令微调阶段已赋予模型通用多模态对话能力，但其回复常存在简短、不流畅、无关、格式不当或幻觉等问题，影响用户体验</span>。为缓解这些问题并进一步提升对话体验，在指令微调之后引入偏好对齐微调阶段。采用统一训练策略，融合 DPO 与指令微调，损失函数为：

$$\mathcal{L}_{\text{at}}(\mathbf{x}) = \mathcal{L}_{\text{dpo}}(\mathbf{x}_{\text{chosen}}, \mathbf{x}_{\text{rejected}}) + \lambda \cdot \mathcal{L}_{\text{it}}(\mathbf{x}_{\text{chosen}}, \mathbf{x}_{\text{it}})$$

其中$$\mathcal{L}_{\text{dpo}}$$和$$\mathcal{L}_{\text{it}}$$分别表示 DPO 损失和指令微调损失；$$\mathbf{x}_{\text{chosen}}$$与$$\mathbf{x}_{\text{rejected}}$$为偏好数据集中的优选与拒选样本；$$\mathbf{x}_{\text{it}}$$为指令数据集中的样本。这里将$$\lambda$$设为 0.3。此外<span style="color: rgb(100,37,208); background-color: inherit">采用 LoRA 微调 LLM 主干网络 5.0% 的权重，以防止灾难性遗忘</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">任务平衡策略</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">步长平衡策略 Step Balance Strategy</span>**

在预训练的多模态联合训练阶段，面临两大难点：**<span style="color: rgb(216,57,49); background-color: inherit">数据样本平衡</span>**<span style="color: rgb(216,57,49); background-color: inherit">与</span>**<span style="color: rgb(216,57,49); background-color: inherit">损失权重平衡</span>**。一方面，<span style="color: rgb(216,57,49); background-color: inherit">各模态数据量差异显著，限制了数据稀缺模态的性能</span>；另一方面，<span style="color: rgb(216,57,49); background-color: inherit">不同模态的损失不在同一量级，导致训练方向偏向损失更大的模态，造成次优收敛</span>。为此，M2-omni 提出步长平衡策略，同时解决这两个难点。

**<span style="color: rgb(222,120,2); background-color: inherit">难点 1：数据样本平衡</span>**

设$$\{D_1, D_2, \dots, D_M\}$$为$$M$$种不同模态的训练数据集合，$$L_i$$为第$$i$$个模态对应的损失函数。预训练阶段，作者<span style="color: rgb(100,37,208); background-color: inherit">探索不同模型更新方法，重点关注其在平衡多模态能力方面的有效性</span>。所有方法均在每个小批量仅包含单一模态数据的约束下进行比较，以确保训练过程的平衡与高效。主要探索三种方法：

**<span style="color: rgb(222,120,2); background-color: inherit">随机采样</span>**

从整个数据集中随机抽取 mini-batch，各模态采样概率与其数据量成正比：

$$\theta_{t+1} = \theta_t - \eta \nabla L_i(B_i)$$

其中$$i$$为随机选择的模态索引，$$B_i$$为来自模态$$i$$的 mini-batch，$$\theta_t$$和$$\eta$$分别表示时刻$$t$$的模型权重与学习率。

**<span style="color: rgb(222,120,2); background-color: inherit">轮询</span>**

轮流使用各数据集的 mini-batch 更新参数，确保各模态迭代步数相等：

$$\theta_{t+1} = \theta_t - \eta \nabla L_{i_t}(B_{i_t})$$

其中$$i_t = (t \bmod M) + 1$$表示时刻$$t$$选择的模态。

**<span style="color: rgb(222,120,2); background-color: inherit">梯度累积</span>**

轮流前向传播各模态的一个 batch，累积梯度后统一更新参数：

$$\theta_{t+1} = \theta_t - \eta \sum_{i=1}^{M} \nabla L_i(B_i)$$

梯度累积法始终优于其他两种方法，因为其更稳定的梯度和对各模态的充分训练。

**<span style="color: rgb(222,120,2); background-color: inherit">难点 2：损失权重平衡</span>**<span style="color: rgb(222,120,2); background-color: inherit">  </span>

采用一种简单而有效的方法确定模态特定的损失权重，步骤如下： &#x20;

> 1. 在子集$$D_i^{\text{sub}} \subset D_i$$上训练模型直至收敛
>
> 2. 记录收敛损失值$$L_i^*$$ &#x20;
>
> 3) 利用下式计算归一化权重$$w_i$$：
>
> $$w_i = \alpha \cdot \frac{1 / L_i^*}{\sum_{j=1}^{M} 1 / L_j^*}$$
>
> 这个权重随后用于参数梯度更新。对于采用梯度累积策略的数据样本平衡，设$$\alpha = 10$$，并按如下方式更新参数：
>
> $$\theta_{t+1} = \theta_t - \eta \sum_{i=1}^{M} w_i \nabla L_i(B_i)$$

完整的步长平衡策略算法流程为：

![](../../images/视觉多模态讲义（下）-image-227.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">动态自适应平衡策略 Dynamic Adaptive Balance Strategy</span>**

在全模态指令微调阶段，采用动态自适应平衡策略调节各模态的收敛速度。将每个模态视为独立训练任务，借鉴多任务学&#x4E60;**`MTL`**&#x539F;则平衡各任务的训练进度：<span style="color: rgb(100,37,208); background-color: inherit">对收敛曲线平缓的模态降低权重以防过拟合，对收敛陡峭的模态提高权重以加强学习</span>。与传统 MTL 方法需在整个数据集上交替训练与验证不同，<span style="color: rgb(100,37,208); background-color: inherit">M2-omni 在训练过程中以固定间隔插入周期性验证片段，利用小型预设验证子集计算各模态的验证损失，从而在历史窗口内通过验证损失及其收敛斜率追踪各模态训练进度</span>。这<span style="color: rgb(46,161,33); background-color: inherit">以极低计算开销实现模态权重的动态调整，提升全模态学习性能</span>。

![](../../images/视觉多模态讲义（下）-image-224.png)

**<span style="color: rgb(222,120,2); background-color: inherit">数据划分</span>**

从各模态训练数据中随机划分验证子集，共包含$$\sum_{i=1}^{M} S_i \cdot B_i$$个样本，其中$$S_i$$为第$$i$$个模态每次验证片段的验证步数，$$B_i$$为其验证 batch size，$$M$$为模态总数。该验证子集不参与模型训练。

**<span style="color: rgb(222,120,2); background-color: inherit">收敛斜率计算</span>**

不同模态训练难度各异，损失数值范围不同。为公平分配权重，从归一化验证损失计算收敛斜率：

$$L^{\text{val}}_{i,t} = \frac{L^{\text{val}}_{i,t} - \min\left\{ \min\{L^{\text{val}}_{i,j}\}_{j=t-H+1}^{t}, \epsilon \right\}}{\max\{L^{\text{val}}_{i,j}\}_{j=t-H+1}^{t}  - \min\left\{ \min\{L^{\text{val}}_{i,j}\}_{j=t-H+1}^{t}, \epsilon \right\}}$$

其中$$L^{\text{val}}_{i,t}$$为第$$i$$个模态在第$$t$$个验证片段的验证损失，$$H$$为历史窗口大小，$$\epsilon = 10^{-6}$$防止除零。随后用线性回归模型$$a_{i,t} x + b_{i,t}$$拟合历史窗口内的验证损失，斜率系数$$a_{i,t}$$即为当前模态的收敛速率。

**<span style="color: rgb(222,120,2); background-color: inherit">权重分配调整</span>**

为确保平衡初始化并缓解初始收敛轨迹的不准确性，前$$H$$个验证片段中所有模态损失权重$$\tilde{w}_{i,0}$$固定为 1。对于第$$t$$个验证片段$$t > H$$，首先计算第$$i$$个模态的归一化斜率$$\tilde{a}_{i,t}$$与收敛得分$$s_{i,t}$$：

$$\tilde{a}_{i,t} = \frac{M \cdot a_{i,t}}{\sum_{j=1}^{M} |a_{j,t}|}, \quad s_{i,t} = \text{softmax}(\tilde{a}_{i,t}) \cdot (-1 \cdot \tilde{a}_{i,t})$$

其中 softmax 沿模态维度进行。接着计算当前片段的模态权重分配：

$$w_{i,t} = M \cdot \text{softmax}(f \cdot s_{i,t})$$

其中$$f$$为调节权重分布的缩放因子，乘以$$M$$确保总权重和为$$M$$。<span style="color: rgb(100,37,208); background-color: inherit">为避免单步更新引起的剧烈波动，采用指数移动平均</span>**`EMA`**<span style="color: rgb(100,37,208); background-color: inherit">机制平滑调整各模态训练权重</span>：

$$\tilde{w}_{i,t} = \alpha \cdot \tilde{w}_{i,t-1} + (1 - \alpha) \cdot w_{i,t}$$

其中平滑因子$$\alpha = 0.9$$。调整后的模态特定损失权重$$\tilde{w}_{i,t}$$用于下一验证片段前的所有训练步。通过该动态自适应平衡策略，所有模态性能均优于对应单模态模型。

* **<span style="color: rgb(36,91,219); background-color: inherit">语言能力维持策略</span>**

全模态 MLLM 应维持强大的语言能力，因为语言本身也是其支持的模态之一。在 M2-omni 的预训练与后训练阶段，<span style="color: rgb(216,57,49); background-color: inherit">若仅使用多模态数据解冻 M2-omni LLM，会导致语言能力显著下降，凸显引入纯文本数据的重要性</span>。这是因为多模态数据中的文本往往缺乏纯文本数据的多样性与复杂性，易导致模型偏差。为此在每次解冻 M2-omni LLM 时，均引入受控比例的纯文本数据。<span style="color: rgb(46,161,33); background-color: inherit">当纯文本数据比例控制在约 25% 时，可有效防止多模态能力退化，同时维持稳健的语言能力</span>。

当然可以。以下是简化后的 **第 4 节“数据配置”** 内容，保留关键信息，语言更简洁：

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">预训练数据</span>**

预训练数据围绕对齐不同模态和学习世界知识两个目标构建：

> * **<span style="color: rgb(36,91,219); background-color: inherit">图像-文本</span>**：约 20 亿对，主要来自公开网络爬取数据，并在第二阶段加入高质量人工标注标题
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">OCR</span>**：包含多个英文 OCR 数据集及大规模中文内部 OCR 数据（涵盖文档与场景文本）
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音频-文本</span>**：3000 万对，覆盖语音识别（ASR）、音频描述（AAC）和音频标签（AAT）三类任务
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">交错图像-文本</span>**：采用公开数据集 MMC4，包含图文交错的多图文档
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视频-文本</span>**：来自 WebVid-10M 和 Youku-mPLUG，并补充了内部高清视频数据
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">纯文本</span>**：使用 Pile、Wudao 及内部语料，用于维持语言能力

预训练分三阶段逐步引入模态：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：图像、OCR、音频 + 文本，共 21.7 亿样本
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：精选高质量图像-文本和 OCR 数据，并加入 15% 纯文本
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3</span>**：全面融合所有模态，新增视频、音频和交错图文数据

* **<span style="color: rgb(36,91,219); background-color: inherit">指令微调数据</span>**

结合开源与自建数据，覆盖图像、视频、音频及纯文本指令任务。为缓解长尾类别（如科学、数学）数据不足的问题，M2-omni 设计了一个自动数据生成流程：<span style="color: rgb(100,37,208); background-color: inherit">利用 GPT-4 生成主题，检索相关图像，再用 GPT-4V 生成图文问答对，从而扩充高质量指令数据</span>。微调也分三阶段：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：聚焦图像理解和基础对话，文本占比约 30%
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：加入视频和交错图文理解
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3</span>**：引入音频问答和图像生成任务，支持全模态混合指令

* **<span style="color: rgb(36,91,219); background-color: inherit">对齐微调数据</span>**

使用两类数据：

> * **<span style="color: rgb(36,91,219); background-color: inherit">偏好数据</span>**：由真实用户提示出发，让 M2-omni 和 GPT-4o 分别生成回复，人工标注更优答案，构建偏好对
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">指令数据</span>**：从全模态指令微调数据中均衡采样，与偏好数据按 1:1 混合

### 5.4.5 **<span style="color: rgb(36,91,219); background-color: inherit">AR-Omni</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">单主干 Any-to-Any 自回归建模</span>**

AR-Omni 使用一个 **`7B`** 参数的 Decoder-only Transformer 统一处理文本、语音和图像。三种模态先被离散化，再并入联合词表：

$$\mathcal{V}=\mathcal{V}_{\mathrm{text}}\cup\mathcal{V}_{\mathrm{speech}}\cup\mathcal{V}_{\mathrm{image}}$$

模型接收交错的多模态 Token 序列，使用完全一致的 Next-Token Prediction 生成后续文本、语音或图像 Token：

$$p_{\theta}(\mathbf{x})=\prod_{t=1}^{T}p_{\theta}(x_t\mid x_{<t})$$

<span style="color: rgb(100,37,208); background-color: inherit">理解与生成共享同一个 Transformer、同一套注意力层和同一条自回归时间线。</span>输出的非文本 Token 直接交给对应 Detokenizer 还原，不再通过额外扩散模型或模态专家生成像素与波形。文本、语音、图像可以出现在输入和输出的任意一侧，因此文本到图像、语音识别、语音合成、图像描述、语音到图像以及多轮语音对话都只是不同的序列边界配置。

* **<span style="color: rgb(36,91,219); background-color: inherit">三类离散 Tokenizer</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">文本</span>**：沿用 Chameleon 初始化所使用的 SentencePiece BPE Tokenizer，使文本 Embedding 与已有语言模型权重直接对齐。

**<span style="color: rgb(222,120,2); background-color: inherit">语音</span>**：使用单码本 WavTokenizer，把波形映射为纯声学 Token。它不先预测语义 Token、再用第二级模型补充声学细节，而是让统一主干一次性预测可解码的声学码。<span style="color: rgb(46,161,33); background-color: inherit">Detokenizer 在拿到少量 Token 后即可开始合成波形，因此同一自回归输出流可以自然支持流式播放。</span>

**<span style="color: rgb(222,120,2); background-color: inherit">图像</span>**：采用 Chameleon 的 Scene-Aware VQ Tokenizer，将图像编码成一维排列的离散视觉 Token。图像生成结束后直接用 VQ Detokenizer 还原像素，不调用扩散 UNet。<span style="color: rgb(216,57,49); background-color: inherit">一维因果顺序要求模型在生成早期就确定全局布局，后续 Token 只能在既有前缀条件下补充细节，不能像扩散过程那样反复全局修正。</span>

![AR-Omni 以联合离散词表和单一自回归 Decoder 连接文本、语音与图像](../../images/视觉多模态讲义（下）-ar-omni-architecture.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">交错序列与模态边界</span>**

每个用户轮次以 `<bos>` 开始，以 `<eoh>` 结束输入。语音 Token 包在 `<boa>...<eoa>` 中，图像 Token 包在 `<boi>...<eoi>` 中。单轮任务的 Assistant 输出以 `<eos>` 结束；多轮对话使用 `<eom>` 结束当前 Assistant 消息，再把后续历史按同一格式继续拼接。

文本在序列中承担模态桥接作用。例如语音到图像任务可以先输出转写文本，再输出图像 Token；图像与语音混合输入也可以先由文本说明当前意图，再进入目标模态。边界 Token 同时提供三类信息：模态起止位置、用户/助手轮次边界以及对话是否继续。

* **<span style="color: rgb(36,91,219); background-color: inherit">Task-Aware 有限状态解码</span>**

解码器在联合词表上输出分布，但不同任务不采用完全相同的采样规则。ASR 和 TTS 的目标较确定，使用 Greedy Decoding；开放式文本生成、文本到图像以及其他创造性任务使用随机采样。有限状态控制器根据 Prompt 模板和已生成的边界 Token 限制下一个合法 Token 集合：进入语音区间后只允许语音词表与结束标记，进入图像区间后只允许图像词表与结束标记。

<span style="color: rgb(100,37,208); background-color: inherit">有限状态约束把“下一个 Token 属于哪种模态”从概率偏好提升为语法约束。</span>这避免联合词表在长序列中混入错误模态 Token，也允许同一个主干在不同阶段切换 Greedy 与 Sampling 策略。

* **<span style="color: rgb(36,91,219); background-color: inherit">加权 Next-Token Prediction</span>**

图像和语音序列通常远长于文本。如果对所有 Token 等权平均，长模态会占据主要梯度，短文本回答的学习信号被稀释。训练时为不同位置设置权重：

$$\mathcal{L}_{\mathrm{wNTP}}=-\frac{1}{T}\sum_{t=1}^{T}w_t\log p_{\theta}(x_t\mid x_{<t})$$

在图像到文本、语音到文本等 X2T 任务中，提高 Assistant 文本 Token 的 $$w_t$$；非响应区域与提示区域保持较低权重。这样，损失规模不再简单等同于每种模态的 Token 数量。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像感知损失</span>**

离散图像 Token 的交叉熵把所有错误码字视为同等错误，但相邻码字可能具有相似纹理或颜色。AR-Omni 使用固定图像码本 Embedding $$E$$ 作为感知目标，将主干隐藏状态 $$h_t$$ 通过可训练投影 $$W_h$$ 映射到码本空间：

$$\mathcal{L}_{\mathrm{perc}}=\frac{1}{|\mathcal{T}_{I}|}\sum_{t\in\mathcal{T}_{I}}\left\|W_hh_t-E[x_t]\right\|_2^2$$

$$\mathcal{L}=\mathcal{L}_{\mathrm{wNTP}}+\lambda_{\mathrm{perc}}\mathcal{L}_{\mathrm{perc}}$$

$$\mathcal{T}_{I}$$ 只包含目标图像位置，$$\lambda_{\mathrm{perc}}$$ 取较小值，使感知梯度与离散分类梯度保持同一量级。<span style="color: rgb(100,37,208); background-color: inherit">交叉熵负责预测正确码字，感知项进一步要求隐藏状态靠近目标码字的连续语义位置。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Residual-Post-Norm</span>**

主干使用 Swin-Norm 风格的 Residual-Post-Norm，在注意力或 FFN 输出进入残差分支前做归一化：

$$h=x+\operatorname{Norm}(\operatorname{Attn}(x)),\qquad x'=h+\operatorname{Norm}(\operatorname{FFN}(h))$$

该结构限制每个残差更新的尺度，减少文本、语音和图像批次交替时激活分布的漂移。

![不同 Any-to-Any 任务使用统一对话模板和显式模态边界 Token](../../images/视觉多模态讲义（下）-ar-omni-prompt-templates.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">两阶段训练</span>**

主干从 Anole 7B 初始化。第一阶段进行多模态预训练，联合优化加权 NTP 与图像感知损失；第二阶段进行指令微调，只在 Assistant 响应 Token 上计算损失，用户输入、历史上下文和系统 Prompt 仅作为条件。

预训练数据由三类子集混合：纯文本、图文和语音文本，采样比例为 **`0.5:1:2`**。各子集不做循环过采样，任一子集耗尽时结束该阶段。纯文本使用 Ultra-FineWeb；图文数据来自 LAION-2B、LAION-Aesthetics 与 JourneyDB；语音文本来自 GigaSpeech、Common Voice 与 MLS。

指令微调以 AnyInstruct 组织交错 Any-to-Any 样本。语音输入用 DEMAND 环境噪声增强；语音助手数据包含 VoiceAssistant-400K，并用 CosyVoice2 合成与上下文说话人音色对齐的 Assistant 语音；纯文本对话使用 UltraChat。训练模板覆盖文本到图像、ASR、TTS、图像描述、语音转写后生成图像以及多轮语音对话，使边界 Token 和有限状态解码在训练阶段就保持一致。

* **<span style="color: rgb(36,91,219); background-color: inherit">优化配置</span>**

训练使用 **`8`** 张 NVIDIA A100，优化器为 Adam，学习率按线性日程变化，Warmup Ratio 为 **`0.05`**，全局梯度裁剪为 **`1.0`**。预训练最大序列长度 **`1300`**，全局 Batch Size **`480`**，训练 **`140K`** 步，峰值学习率 **`6×10^{-5}`**；指令微调最大长度 **`3456`**，Batch Size **`64`**，训练 **`18K`** 步，峰值学习率 **`2×10^{-5}`**。

<span style="color: rgb(216,57,49); background-color: inherit">联合词表并没有消除模态长度失衡、边界合法性与视觉码字距离缺失的问题。</span>加权 NTP、有限状态解码和感知损失分别处理这三类问题，它们是单主干方案能够稳定工作的组成部分。

![AR-Omni 预训练与指令微调的优化配置](../../images/视觉多模态讲义（下）-ar-omni-training-config.png)

### 5.4.6 **<span style="color: rgb(36,91,219); background-color: inherit">NExT-OMNI</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">统一表示、统一主干与统一生成过程</span>**

NExT-OMNI 覆盖文本、图像、视频和音频，使用 Discrete Flow Matching 统一理解、生成与跨模态检索。图像/视频共享视觉 Encoder，音频使用音频 Encoder；同一个 Encoder 同时服务理解特征、生成离散码字和检索表示，不为理解与生成拆成两套视觉或声学分支。主干从 Qwen2.5-7B 初始化，保留文本 LM Head，并为视觉和音频各增加轻量模态 Head，新增 Head 总参数量约为 **`128M`**。

<span style="color: rgb(100,37,208); background-color: inherit">视觉、文本和音频 Token 在每一层双向自注意力中深度融合，而不是通过 Mixture-of-Experts 或 Mixture-of-Transformers 把不同模态分发到独立专家。</span>生成端不外挂扩散模型或连续 Flow Head，图像、视频和音频都由主干直接预测离散码本索引，再由各自 Decoder 重建原始信号。

文本分支仍保留自回归语言模型的一位右移格式，使 Qwen2.5 的 Next-Token Prediction 权重可直接迁移；非文本响应则在 Discrete Flow Matching 中从噪声 Token 迭代校正。理解任务读取统一隐藏表示后由文本 LM Head 输出答案；生成任务由目标模态 Head 输出码本分布；检索任务对中间融合表示进行池化和相似度匹配。

![NExT-OMNI 在统一表示上连接全模态理解、生成与跨模态检索](../../images/视觉多模态讲义（下）-next-omni-overview.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">离散 Flow Matching 主流程</span>**

设目标多模态序列为 $$\mathbf{x}_1=(x_1^1,\ldots,x_1^D)$$。训练时采样时间 $$t\sim\mathcal{U}(0,1)$$，再从条件路径 $$p_t(\mathbf{x}_t\mid\mathbf{x}_1)$$ 采样当前噪声状态 $$\mathbf{x}_t$$。当 $$t=0$$ 时响应区域接近随机 Token，当 $$t=1$$ 时接近真实目标。主干接收指令部分和当前响应状态，在每个响应位置预测最终干净 Token 的条件分布 $$p_{1\mid t}$$。

图像和音频位置不会把整数码本 ID 直接当作无序类别输入。对于码本索引 $$k$$，先取对应连续码字向量 $$c_k$$，再用模态投影映射到文本隐藏维度：

$$e_t^i=W_Mc_{x_t^i},\qquad M\in\{V,A\}$$

<span style="color: rgb(100,37,208); background-color: inherit">码本代表向量把离散 Token 之间的连续邻近关系带入主干，使跨模态注意力看到的是可比较的连续特征，而不是没有几何结构的整数编号。</span>文本位置使用原始 Word Embedding，三类 Embedding 按交错序列拼接后进入同一 Transformer。

训练只对响应区进行去噪和校正，指令区始终保持干净。基本交叉熵为：

$$\mathcal{L}_{\mathrm{CE}}=\mathbb{E}_{t,\mathbf{x}_1,\mathbf{x}_t}\left[-\sum_{i\in\mathcal{T}_{\mathrm{resp}}}\log p_{1\mid t}(x_1^i\mid\mathbf{x}_t,t)\right]$$

推理从随机响应序列开始，在多个时间步上并行更新所有未固定位置；文本、图像、视频和音频只是使用不同 Tokenizer、码本 Embedding 与输出 Head，Flow 迭代规则保持一致。

![统一离散 Flow 主干、模态 Encoder 与轻量输出 Head 的训练和推理路径](../../images/视觉多模态讲义（下）-next-omni-pipeline.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">模态 Encoder Warmup</span>**

视觉和音频 Encoder 在接入 7B 主干前先进行统一表示 Warmup。连续输入 $$X^M$$ 经过 Encoder 得到 $$z^M=E^M(X^M)$$，再在多码本中选择最近码字：

$$z_q^M=\operatorname*{arg\,min}_{c\in\mathcal{C}^M}\left\|z^M-c\right\|_2^2,\qquad M\in\{V,A\}$$

Warmup 同时优化重建与语义对齐：

$$\mathcal{L}_{\mathrm{warm}}^M=\mathcal{L}_{\mathrm{rec}}^M+\mathcal{L}_{\mathrm{sem}}^M$$

视觉重建项由像素重建、VQ 约束、LPIPS 感知损失和 GAN 判别损失组成；语义项使用句子级 CLIP 对比损失，使整图表示与对应文本描述对齐。视觉 Encoder 从 CLIP-ViT-Large 初始化，在约 **`70M`** 图文对上训练，使用 **`4×4096`** 的多码本量化。

音频重建项包含 Mel 频谱重建、VQ 约束、特征匹配的 L2 损失和 GAN 损失；语义项不是句子级对比，而是由文本 Decoder 逐 Token 预测音频描述。音频 Encoder 从 Whisper-Turbo 初始化，Warmup 数据约 **`102K`** 小时，使用 **`2×2048`** 多码本，并用 Qwen2.5-0.5B 作为音频描述 Decoder；单段音频最长 **`15 s`**。

<span style="color: rgb(100,37,208); background-color: inherit">重建让离散码字保留生成所需的细粒度信号，文本对齐让同一 Encoder 的中间表示可以直接服务理解与检索。</span>只做语义对齐会丢失颜色、纹理和声学细节，只做重建则难以与语言主干建立稳定语义接口。

![视觉与音频 Encoder Warmup 同时优化信号重建和文本语义对齐](../../images/视觉多模态讲义（下）-next-omni-encoder-warmup.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">重建增强的 Flow 目标</span>**

模态 Encoder 接入 DFM 主干后，训练仍保留视觉和音频重建项，避免大规模指令训练把码本推向只保留语义、忽略生成细节的方向：

$$\mathcal{L}_{\mathrm{overall}}=\lambda_1\mathcal{L}_{\mathrm{CE}}+\lambda_2\mathcal{L}_{\mathrm{rec}}^V+\lambda_3\mathcal{L}_{\mathrm{rec}}^A$$

三个系数不固定手调，而是用 GradNorm 根据各损失的梯度范数动态平衡。<span style="color: rgb(46,161,33); background-color: inherit">Flow 交叉熵负责把随机离散状态校正到目标 Token，重建项持续约束码本保真度，两部分在同一训练过程中共同更新。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">多码本模态 Head</span>**

多码本量化在每个时空位置产生多个子码本索引。若一个位置包含 $$K$$ 个子码本结果，模态 Head 需要从主干隐藏状态同时恢复 $$(c_1,\ldots,c_K)$$。NExT-OMNI 比较两种结构：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">并行 Multi-Token Head</span>**：为每个子码本设置独立输出 Head，所有子码本同时预测。计算路径短，但各子码本只能共享主干状态，无法显式利用同一位置已预测的其他码字。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">自回归 Head</span>**：把一个位置的子码本索引展开成局部短序列，依次预测后续子码字；Head 内使用注意力读取主干隐藏状态与前面子码本结果。

最终采用自回归 Head。<span style="color: rgb(100,37,208); background-color: inherit">主干仍在所有时空位置上并行执行 Flow 校正，只有单个位置内部的多个子码本按顺序解码。</span>这种设计增加少量 Head 计算，但为子码本之间建立明确条件依赖，训练更稳定。

![多码本位置的自回归 Head 与并行 Multi-Token Head](../../images/视觉多模态讲义（下）-next-omni-modality-heads.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">动态长度生成</span>**

离散 Flow 一次更新整段响应，但生成前通常不知道目标长度。训练时先把响应补齐到 Block Size 的整数倍，并在真实结束位置放置 EOS，剩余位置填 PAD；默认 Block Size 为 **`64`**。推理从一个块开始，每轮根据尾部 EOS 置信度决定是否停止：若 EOS 置信度不足，则再扩展一个随机 Token 块继续去噪，直到置信度达到阈值或长度上限。

<span style="color: rgb(100,37,208); background-color: inherit">长度不由独立回归器一次性猜测，而是在 Flow 迭代中随内容逐块增长。</span>短回答可以尽早结束，长图像/音频/视频序列也能按相同机制扩展，不需要为每种模态设计固定输出长度。

* **<span style="color: rgb(36,91,219); background-color: inherit">Adaptive Cache</span>**

指令区在所有 Flow 时间步都保持不变，因此其 Key/Value Cache 只计算一次并全程复用。响应区会随迭代变化，但并非每个位置都需要每轮重算。系统比较当前 Value 特征与缓存 Value 特征的余弦相似度：

$$s_i^{(t)}=\frac{\left\langle v_i^{(t)},\hat v_i\right\rangle}{\left\|v_i^{(t)}\right\|_2\left\|\hat v_i\right\|_2}$$

相似度高于阈值的位置沿用缓存，变化明显的位置才更新对应 Cache。EOS 与新扩展块始终重新计算，避免缓存阻断长度决策。<span style="color: rgb(46,161,33); background-color: inherit">缓存策略利用 Flow 后期大量位置已趋于稳定这一事实，把计算集中到仍在变化的响应位置。</span>

<span style="color: rgb(216,57,49); background-color: inherit">响应缓存不是永久冻结：若阈值过低，过期特征会阻碍后续迭代修正；若阈值过高，则退化为几乎每轮全量计算。</span>

![按块扩展的动态长度生成与按特征变化更新的自适应缓存](../../images/视觉多模态讲义（下）-next-omni-dynamic-cache.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">三阶段训练流程</span>**

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：Pre-Training</span>**
>
> 图像分辨率 **`256×256`**，视觉下采样率 **`16×`**，音频不超过 **`15 s`**，文本上下文不超过 **`2K`** Token。训练数据约 **`83M`** 对，迭代 **`10K`** 步；模态 Encoder/Decoder 学习率 **`2×10^{-5}`**，其他参数 **`1×10^{-4}`**，每 GPU Batch Size 为 **`16`**。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：Continual Pre-Training</span>**
>
> 图像提高到 **`384×384`**；长音频按不超过 **`15 s`** 的块切分；视频最多 **`8`** 帧；上下文扩展到 **`16K`**。约 **`52M`** 图文、音文和交错样本训练 **`18K`** 步；Encoder/Decoder 学习率降到 **`1×10^{-6}`**，其他参数 **`2×10^{-5}`**，每 GPU Batch Size 为 **`8`**。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：Supervised Fine-Tuning</span>**
>
> 保持 **`384×384`** 图像、最多 **`8`** 视频帧和 **`16K`** 上下文，使用约 **`18M`** 指令样本训练 **`25K`** 步，每 GPU Batch Size 为 **`4`**。指令数据覆盖全部 Any-to-Any 路径，并加入约 **`4M`** 推理指令和 **`5M`** 图像生成推理合成数据。

三个阶段都使用 AdamW 与 Cosine Learning Rate Scheduler，Weight Decay 为 **`0.05`**；Warmup 步数依次为 **`1000`**、**`500`**、**`500`**。生成任务以 **`0.1`** 的概率移除条件，训练 Classifier-Free Guidance。为了保持单批次张量结构整齐，每个 Batch 只放一种目标模态，不同任务在梯度累积窗口内交错。

<span style="color: rgb(100,37,208); background-color: inherit">完整路线是：先把视觉与音频 Encoder 训练成兼顾重建和语义的离散表示，再让 7B 主干学习跨模态 DFM，最后用 Any-to-Any 指令收紧输入输出协议；动态长度和自适应缓存只改变推理过程，不改变训练目标。</span>

![NExT-OMNI 从预训练、持续预训练到监督微调的三阶段配置](../../images/视觉多模态讲义（下）-next-omni-training-recipe.png)

---

[Previous](02-Diffusion-Model.md) | [Contents](../../README.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=3)
