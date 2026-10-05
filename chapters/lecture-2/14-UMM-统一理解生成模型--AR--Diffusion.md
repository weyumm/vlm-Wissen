[Previous](13-UMM-统一理解生成模型--Autoregressive-based.md) | [Contents](../../README.md) | [Next](15-UMM-统一理解生成模型--Any-to-AnyOmni-全模态.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=14)

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

---

[Previous](13-UMM-统一理解生成模型--Autoregressive-based.md) | [Contents](../../README.md) | [Next](15-UMM-统一理解生成模型--Any-to-AnyOmni-全模态.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=14)
