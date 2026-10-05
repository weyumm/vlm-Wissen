[Previous](11-Diffusion-Model--视频生成.md) | [Contents](../../README.md) | [Next](13-UMM-统一理解生成模型--Autoregressive-based.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=12)

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

---

[Previous](11-Diffusion-Model--视频生成.md) | [Contents](../../README.md) | [Next](13-UMM-统一理解生成模型--Autoregressive-based.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=12)
