[Previous](06-Vision-Language-Model--BLIP-家族.md) | [Contents](../../README.md) | [Next](08-Vision-Language-Model--LLaVA-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=7)

## 2.3 <span style="color: rgb(36,91,219); background-color: inherit">结构优化</span>

### 2.3.1 <span style="color: rgb(36,91,219); background-color: inherit">VLMo</span>

**VLMo&#x20;**&#x7684;核心在于其创新性的混合模态专家 **<span style="color: rgb(216,57,49); background-color: inherit">MoME</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ixture-</span>**<span style="color: rgb(216,57,49); background-color: inherit">o</span>**<span style="color: rgb(216,57,49); background-color: inherit">f-</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">odality-</span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">xperts）</span>框架，以及灵活的模型结构设计。相比于传统的单一结构模型，**VLMo&#x20;**&#x80FD;够根据任务需求动态切换双编码器和单塔编码器模式，从而在不同的应用场景中表现出色。总的来说，**VLMo** 的贡献主要有两个：

> * **<span style="color: rgb(36,91,219); background-color: inherit">模型结构的改进</span>**：**MoME**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">训练方式的改进</span>**：分阶段的预训练

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

具体而言，**<span style="color: rgb(46,161,33); background-color: inherit">VLMo </span>**<span style="color: rgb(46,161,33); background-color: inherit">通过引入一种名为 MoME-Transformer 的架构，解决了传统双编码器模型在特征融合上的不足，同时兼顾了高效性和灵活性</span>。这种架构允许模型在训练阶段针对不同模态选择性地更新参数，如图像、文本或混合模态，而在推理阶段则可以根据任务需求选择最优的计算路径。

目前多模态有两种主流的模型结构

> * **<span style="color: rgb(36,91,219); background-color: inherit">Dual Encoder</span>**：例&#x5982;**`CLIP`**、**`ALIGN`**，双塔模型，余弦相似度做交互，适合大规模图文检索，但<span style="color: rgb(216,57,49); background-color: inherit">交互太简单，无法处理比较复杂的场景</span>
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Fusion Encoder</span>**：用 Transformer Encoder 做多模态交互，弥补双塔模型的缺陷，&#x5728;**`VR`**，**`VE`**，**`VQA`**&#x4EFB;务效果较好，但是<span style="color: rgb(216,57,49); background-color: inherit">不适合检索，所有可能的图像文本都要编码</span>

**VLMo&#x20;**&#x7ED3;合两者优点，设计 MoME 结构：

![FFN 层变成了并行的多个 FFN，分别处理图像，文本和多模态。其中 F=2，即十层分开计算，两层融合。在前向传播时需要两次：mask 和不 mask。缺点是训练很慢，和 ViLT 相当，比 ALBEF 慢](../../images/视觉多模态讲义（上）-image-111.png)

给定图文对，VLMo 通过 MoME Transformer 网络获得仅图像、仅文本以及图文对的表示。如上图所示，<span style="color: rgb(46,161,33); background-color: inherit">统一的预训练过程优化了共享的 MoME Transformer</span>。<span style="color: rgb(100,37,208); background-color: inherit">在仅图像和仅文本的表示上进行图文对比学习，在图文对的表示上进行图文匹配和掩码语言建模。由于模型的灵活性，该模型可以用作双编码器以处理检索任务，在微调阶段分别对图像和文本进行编码；也可以作为融合编码器进行微调，以对图像和文本的深度模态交互进行建模，从而完成分类任务</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">输入表示</span>**

<span style="color: rgb(100,37,208); background-color: inherit">给定一个图文对，将该对编码为图像、文本以及图文向量表示。这些表示随后被输入到 MoME Transformer 中，以学习上下文化的表示并对齐图像和文本的特征向量</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像表示</span>**

遵循视觉 Transformer 的做法，二维图像$$v \in \mathbb{R}^{H \times W \times C}$$被分割并重塑为$$N = HW / P^2$$个图像块$$v_p \in \mathbb{R}^{N \times (P^2C)}$$，其中$$C$$是通道数，$$(H, W)$$是输入图像的分辨率，$$(P, P)$$是图像块的分辨率。这些图像块被展平为向量并通过线性投影得到嵌入表示。然后在序列前添加一个可学习的特殊 Token **`[I_CLS]`**。最终，图像输入表示通过将图像块嵌入、可学习的一维位置嵌入$$V_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$$和图像类型嵌入$$V_{\text{type}} \in \mathbb{R}^D$$相加得到： &#x20;

$$H_0^v = [v_{[\text{I\_CLS}]}, V v_i^p, \dots, V v_N^p] + V_{\text{pos}} + V_{\text{type}}$$

其中$$H_0^v \in \mathbb{R}^{(N+1) \times D}$$，线性投影$$V \in \mathbb{R}^{(P^2C) \times D}$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">文本表示</span>**

遵循 BERT 的做法，使用 WordPiece 将文本分词为子词单元。在文本序列中添加序列起始 Token **`[T_CLS]`**&#x548C;特殊边界 Token **`[T_SEP]`**。文本输入表示$$H_0^w \in \mathbb{R}^{(M+2) \times D}$$通过对相应的词嵌入、文本位置嵌入和文本类型嵌入求和得到： &#x20;

$$H_0^w = [w_{[\text{T\_CLS}]}, w_i, \dots, w_M, w_{[T\_SEP]}] + T_{\text{pos}} + T_{\text{type}}$$

其中$$M$$表示分词后子词单元的长度。

* **<span style="color: rgb(36,91,219); background-color: inherit">图文表示</span>**

将图像和文本输入向量拼接起来，形成图文输入表示： &#x20;

$$H^{vl}_0 = [H_0^w; H_0^v]$$

* **<span style="color: rgb(36,91,219); background-color: inherit">MoME Transformer</span>**

受混合专家网络 mixture-of-experts networks 的启发，VLMo 提出了一种通用的多模态 Transformer 模型用于视觉-语言任务，称为MoME Transformer，以编码不同的模态。 &#x20;

MoME Transformer 引入了模态专家混合 mixture of modality experts 来替代标准 Transformer 中的前馈网络 FFN。给定上一层的输出向量$$H_{l-1}$$，其中$$l \in [1, L]$$，每个 MoME Transformer 块通过切换到不同的模态专家捕获特定模态的信息，并使用跨模态共享的多头自注意力机制 MSA 对齐视觉和语言内容。LN 表示层归一化：

$$H_l' = \text{MSA}(\text{LN}(H_{l-1})) + H_{l-1}$$

$$H_l = \text{MoME-FFN}(\text{LN}(H_l')) + H_l'$$

MoME-FFN 根据输入向量$$H_l'$$的模态以及 Transformer 层的索引，从多个模态专家中选择一个专家来处理输入。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">有三种模态专家：视觉专家</span>**`V-FFN`**<span style="color: rgb(100,37,208); background-color: inherit">、语言专家</span>**`L-FFN`**<span style="color: rgb(100,37,208); background-color: inherit">以及视觉-语言专家</span>**`VL-FFN`**。如果输入是仅图像或仅文本向量，则分别使用视觉专家编码图像，语言专家编码文本。如果输入包含多模态向量，例如<span style="color: rgb(220,155,4); background-color: inherit">图文对的向量，则在底层的 Transformer 层中分别使用视觉专家和语言专家编码各自的模态向量，在顶层使用视觉-语言专家捕获更多的模态交互</span>。针对这三种输入向量，可以获得仅图像、仅文本以及图文对的上下文表示。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练任务</span>**

**VLMo&#x20;**&#x901A;过在图像和文本表示上进行图文对比学习，在图文对表示上进行掩码语言建模和图文匹配来进行联合预训练，所有任务共享参数。

1. **<span style="color: rgb(36,91,219); background-color: inherit">图文对比学习</span>**

给定一批大小为$$N$$的图文对，图文对比学习的目标是从$$N \times N$$个可能的图文对中预测匹配的图文对。在一个训练批次中，存在$$N^2 - N$$个负样本图文对。 &#x20;

最终输出&#x7684;**`[I_CLS]`<span style="color: rgb(216,57,49); background-color: inherit"> </span>**&#x54;oken &#x548C;**`[T_CLS]`** Token 的向量分别作为图像和文本的聚合表示。经过线性投影和归一化后，得到训练批次中的图像向量$$\{\hat{h}_i^v\}_{i=1}^N$$和文本向量$$\{\hat{h}_i^w\}_{i=1}^N$$，用于计算图像到文本和文本到图像的相似度： &#x20;

$$s^{i2t}_{i,j} = \hat{h}_i^{v\top} \hat{h}_j^w, \quad s^{t2i}_{i,j} = \hat{h}_i^{w\top} \hat{h}_j^v$$

$$p^{i2t}_i = \frac{\exp(s^{i2t}_{i,i} / \sigma)}{\sum_{j=1}^N \exp(s^{i2t}_{i,j} / \sigma)}, \quad p^{t2i}_i = \frac{\exp(s^{t2i}_{i,i} / \sigma)}{\sum_{j=1}^N \exp(s^{t2i}_{i,j} / \sigma)}$$

其中，$$s^{i2t}_{i,j}$$表示第$$i$$个图像与第$$j$$个文本之间的图像到文本相似度，$$s^{t2i}_{i,j}$$表示文本到图像相似度。$$\hat{h}_i^w \in \mathbb{R}^D$$和$$\hat{h}_j^v \in \mathbb{R}^D$$分别表示第$$i$$个文本和第$$j$$个图像的归一化向量，$$\sigma$$是一个可学习的温度参数。$$p^{i2t}_i$$和$$p^{t2i}_i$$是经过 softmax 归一化的相似度值。 &#x20;

通过图像到文本和文本到图像相似度的交叉熵损失来训练模型。

* **<span style="color: rgb(36,91,219); background-color: inherit">掩码语言建模</span>**

遵循 BERT 的做法，在文本序列中随机选择一些 Token，并将其替换&#x4E3A;**`[MASK]`** Token。模型被训练通过其他未被遮盖的 Token 和视觉线索来预测这些被遮盖的 Token。**VLMo&#x20;**&#x4F7F;用与 BERT 相同&#x7684;**`15%`**&#x906E;盖概率。被遮盖 Token 的最终输出向量通过一个分类器在整个文本词汇表上进行预测，损失函数为交叉熵损失。

* **<span style="color: rgb(36,91,219); background-color: inherit">图文匹配</span>**

图文匹配的目标是预测图像和文本是否匹配。**VLMo&#x20;**&#x4F7F;&#x7528;**`[T_CLS]`** Token 的最终隐藏向量表示图文对，并将该向量输入一个二分类分类器，损失函数为交叉熵损失。&#x53D7;**`ALBEF`**&#x7684;启发，基于对比学习中的图像到文本和文本到图像相似度采样难负样本图文对。与 **ALBEF&#x20;**&#x4E0D;同的是，后者从单个 GPU 的训练样本中采样难负样本，这称之为局部难负样本挖掘。**<span style="color: rgb(100,37,208); background-color: inherit">VLMo </span>**<span style="color: rgb(100,37,208); background-color: inherit">提出了全局难负样本挖掘方法，从所有 GPU 收集的更多训练样本中采样难负样本图文对。全局难负样本挖掘能够找到更具信息量的图文对，从而显著提升模型性能</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">分阶段预训练</span>**

**VLMo&#x20;**&#x63D0;出了一种分阶段预训练策略，该策略利用大规模的仅图像和仅文本语料库来改进视觉-语言模型。如下图所示：

![](../../images/视觉多模态讲义（上）-image-118.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉预训练</span>**：在仅图像数据上进行视觉预训练，训练 MoME Transformer 的注意力模块和视觉专家，方法类似&#x4E8E;**`BEiT`**，使&#x7528;**`Mask Image Modeling`**。**VLMo&#x20;**&#x76F4;接使用 BEiT 的预训练参数来初始化注意力模块和视觉专家。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">语言预训练</span>**：在仅文本数据上进行语言预训练，冻结注意力模块和视觉专家的参数，并使用掩码语言建&#x6A21;**`Mask Language Modeling`**&#x5728;仅文本数据上优化语言专家。以学习通用的图像和文本表示。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">视觉-语言预训练</span>**：上面预训练的模型被用于初始化模型，使用&#x548C;**`ALBEF`**&#x4E00;样&#x7684;**`ITC`**、**`ITM`**、**`MLM`**&#x7684;预训练任务，以学习视觉信息和语言信息的对齐。<span style="color: rgb(46,161,33); background-color: inherit">与图文对相比，仅图像和仅文本数据更容易收集</span>。此外，图文对中的文本通常较短且简单。在仅图像和仅文本语料库上的预训练提高了模型在复杂图文对上的泛化能力。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：多模态的数据比较少，但是单模态的数据很多，进行单一模态预训练可以提供更好的模型初始化，因此使用分阶段预训练的办法
>
> ***先在 Vision 上训练，把自注意力参数冻住训 Language 可以有很好的效果，但反过来不行（有趣）***

* **<span style="color: rgb(36,91,219); background-color: inherit">下游任务微调</span>**

如下图所示，**VLMo** 可以通过微调适应各种视觉-语言检索和分类任务。

![](../../images/视觉多模态讲义（上）-image-112.png)

**<span style="color: rgb(222,120,2); background-color: inherit">视觉-语言检索</span>**

对于检索任务，**VLMo&#x20;**&#x53EF;以作为双编码器分别对图像和文本进行编码。<span style="color: rgb(100,37,208); background-color: inherit">在微调阶段，模型通过优化图文对比损失进行训练。在推理阶段，首先计算所有图像和文本的表示，然后使用点积获得所有可能图文对的图像到文本和文本到图像的相似度分数</span>。相比于基于融合编码器的模型，分离编码显著提升了推理速度。

![](../../images/视觉多模态讲义（上）-image-113.png)

**<span style="color: rgb(222,120,2); background-color: inherit">视觉-语言分类</span>**

对于诸如视觉问答和视觉推理等分类任务，VLMo 被用作融合编码器，以建模图像和文本之间的模态交互。**VLMo&#x20;**&#x4F7F;&#x7528;**`[T_CLS]`**&#x6807;记的最终编码向量作为图文对的表示，并将其输入任务特定的分类器层以预测标签。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**VLMo&#x20;**&#x901A;过引入混合模态专家框架和可切换的模型结构，成功地解决了多模态预训练中的诸多挑战。其灵活的设计和强大的性能使其在图像-文本检索、跨模态分类和 Zero-Shot 学习等多个任务中表现出色。

### 2.3.2 <span style="color: rgb(36,91,219); background-color: inherit">CoCa</span>

随着人工智能的发展，单模态模型已经无法满足复杂场景下的需求。为了实现更自然的人机交互，研究人员开始探索多模态模型，即将图像、文本等多种信息融合起来进行理解和生成。**CoCa&#x20;**&#x6B63;是在这一背景下诞生的，它的设计灵感来源&#x4E8E;**`ALBEF`**&#x6A21;型，但在架构和训练目标上进行了显著改进。

**CoCa&#x20;**&#x7684;名字来源于其核心训练方法：<span style="color: rgb(216,57,49); background-color: inherit">对比学习 </span>**<span style="color: rgb(216,57,49); background-color: inherit">Co</span>**<span style="color: rgb(216,57,49); background-color: inherit">ntrastive Learning</span> 和<span style="color: rgb(100,37,208); background-color: inherit">字幕生成 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Ca</span>**<span style="color: rgb(100,37,208); background-color: inherit">ptioning</span>。通过结合这两个目标函数，**CoCa&#x20;**&#x80FD;够在统一框架下实现视觉-语言的理解与生成能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">自然语言监督</span>**

![](../../images/视觉多模态讲义（上）-image-114.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">单编码器分类</span>**

经典的单编码器方法<span style="color: rgb(100,37,208); background-color: inherit">通过在大规模众包图像标注数据集上进行图像分类来预训练视觉编码器</span>，&#x5982;**`ImageNet`**、**`Instagram`**&#x6216;**`JFT`**，其中标注文本的词汇表通常是固定的。这些<span style="color: rgb(100,37,208); background-color: inherit">图像标注通常被映射为离散的类别向量</span>，并通过交叉熵损失进行学习：

$$\mathcal{L}_{\text{Cls}} = -p(y) \log q_\theta(x)$$

其中$$p(y)$$是来自真实标签$$y$$的 one-hot、multi-hot 或平滑后的标签分布。预训练的图像编码器随后作为通用视觉表示提取器用于下游任务。

* **<span style="color: rgb(36,91,219); background-color: inherit">双编码器对比学习</span>**

与需要人工标注标签和数据清洗的单编码器分类预训练相比，<span style="color: rgb(100,37,208); background-color: inherit">双编码器方法利用噪声较大的网络规模文本描述，并引入可学习的文本塔来编码自由形式的文本</span>。两个编码器通过对样本批次中的配对文本与其他文本进行对比优化：

$$\mathcal{L}_{\text{Con}} = -\frac{1}{N} \left( \sum_{i=1}^N \log \frac{\exp(x_i^\top y_i / \sigma)}{\sum_{j=1}^N \exp(x_i^\top y_j / \sigma)} + \sum_{i=1}^N \log \frac{\exp(y_i^\top x_i / \sigma)}{\sum_{j=1}^N \exp(y_i^\top x_j / \sigma)} \right)$$

其中$$x_i$$和$$y_j$$分别是第$$i$$对图像和第$$j$$对文本的归一化 Embedding，$$N$$是批次大小，$$\sigma$$是缩放 logits 的温度参数。除了图像编码器外，双编码器方法还学习了一个对齐的文本编码器，从而支持跨模态对齐应用，例如图文检索和零样本图像分类。实验证明，<span style="color: rgb(46,161,33); background-color: inherit">零样本分类在损坏或分布外图像上表现更鲁棒</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">编码器-解码器描述生成</span>**

虽然双编码器方法将文本整体编码，但<span style="color: rgb(100,37,208); background-color: inherit">生成式方法 captioner 追求更细粒度的建模，并要求模型自回归地预测</span>$$y$$<span style="color: rgb(100,37,208); background-color: inherit">的精确分词文本</span>。遵循标准的编码器-解码器架构，图像编码器提供潜在编码特征，例如使用 Vision Transformer 或 ConvNets，而文本解码器则学习最大化配对文本$$y$$在前向自回归分解下的条件似然：

$$\mathcal{L}_{\text{Cap}} = -\sum_{t=1}^T \log P_\theta(y_t | y_{<t}, x)$$

编码器-解码器通过 teacher-forcing 进行训练，以并行化计算并提高学习效率。与以往方法不同，<span style="color: rgb(100,37,208); background-color: inherit">描述生成方法产生了一种联合的图像-文本表示，可用于视觉-语言理解，同时也能通过自然语言生成应用于图像描述生成任务</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">CoCa 预训练</span>**

下图展示了所提出的对比描述生成模型 **<span style="color: rgb(216,57,49); background-color: inherit">CoCa</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Co</span>**<span style="color: rgb(216,57,49); background-color: inherit">ntrastive </span>**<span style="color: rgb(216,57,49); background-color: inherit">Ca</span>**<span style="color: rgb(216,57,49); background-color: inherit">ptioners）</span>，这是一种简单的编码器-解码器方法，无缝结合了三种训练范式。与标准的图文编码器-解码器模型类似，**<span style="color: rgb(100,37,208); background-color: inherit">CoCa </span>**<span style="color: rgb(100,37,208); background-color: inherit">通过神经网络编码器将图像编码为潜在表示</span>，例&#x5982;**`ViT`**&#x6216;**`ConvNets`**，并<span style="color: rgb(100,37,208); background-color: inherit">通过因果掩码 Transformer 解码器生成文本</span>。与标准解码器 Transformer 不同的是，**<span style="color: rgb(100,37,208); background-color: inherit">CoCa </span>**<span style="color: rgb(100,37,208); background-color: inherit">在解码器的前半部分省略了交叉注意力机制以编码单模态文本表示，并在后半部分级联剩余的解码器层，通过交叉注意力机制与图像编码器交互以生成多模态图文表示</span>。因此，**CoCa&#x20;**&#x89E3;码器同时生成单模态和多模态文本表示，可以同时应用对比目标和生成目标：

$$\mathcal{L}_{\text{CoCa}} = \lambda_{\text{Con}} \cdot \mathcal{L}_{\text{Con}} + \lambda_{\text{Cap}} \cdot \mathcal{L}_{\text{Cap}}$$

其中$$\lambda_{\text{Con}}$$和$$\lambda_{\text{Cap}}$$是损失权重超参数。单编码器的交叉熵分类目标可以解释为一种生成方法的特例，应用于图像标注数据时，词汇表是所有标签名称的集合。

![](../../images/视觉多模态讲义（上）-image-110.png)

**<span style="color: rgb(36,91,219); background-color: inherit">解耦文本解码器与 CoCa 架构</span>**

描述生成方法优化文本的条件似然，而对比方法则使用无条件文本表示。为了解决这一矛盾并将两种方法结合到单一模型中，**CoCa&#x20;**&#x63D0;出了一种简单的解耦解码器设计，将解码器分为单模态和多模态组件，通过在单模态解码器层中跳过交叉注意力机制实现。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">底部</span>$$n_{\text{uni}}$$<span style="color: rgb(100,37,208); background-color: inherit">层单模态解码器通过因果掩码自注意力机制将输入文本编码为潜在向量，顶部</span>$$n_{\text{multi}}$$<span style="color: rgb(100,37,208); background-color: inherit">层多模态解码器进一步应用因果掩码自注意力机制，并结合交叉注意力机制处理视觉编码器的输出</span>。所有解码器层都禁止对未来的 Token 进行注意力操作，因此可以直接使用多模态文本解码器的输出作为描述生成目标$$\mathcal{L}_{\text{Cap}}$$。对于对比目标$$\mathcal{L}_{\text{Con}}$$，**<span style="color: rgb(100,37,208); background-color: inherit">CoCa </span>**<span style="color: rgb(100,37,208); background-color: inherit">在输入句子末尾添加一个可学习的</span>**`[CLS]`<span style="color: rgb(100,37,208); background-color: inherit"> </span>**<span style="color: rgb(100,37,208); background-color: inherit">Token，并使用单模态解码器对应的输出作为文本 Embedding。</span>**<span style="color: rgb(100,37,208); background-color: inherit">CoCa </span>**<span style="color: rgb(100,37,208); background-color: inherit">将解码器平分为两部分，使得</span>$$n_{\text{uni}} = n_{\text{multi}}$$。按&#x7167;**`ALIGN`**&#x7684;设置，使用分辨率为$$288 \times 288$$、patch 大小为$$18 \times 18$$的图像进行预训练，总共生&#x6210;**`256`**&#x4E2A;图像 Token。最大的 **CoCa&#x20;**&#x6A21;型遵&#x5FAA;**`ViT-giant`**&#x8BBE;置，图像编码器包&#x542B;**`1B`**&#x53C2;数，整个模型总&#x8BA1;**`2.1B`**&#x53C2;数。此外，还有两个较小的变&#x4F53;**`CoCa-Base`**&#x548C;**`CoCa-Large`**。

**<span style="color: rgb(36,91,219); background-color: inherit">注意力池化</span>**

对比损失为每个图像使用单一 Embedding，而解码器通常会关注编码器-解码器描述生成模型中的一系列图像输出 Token。初步实验表明，<span style="color: rgb(100,37,208); background-color: inherit">单一池化的图像 Embedding 作为全局表示有助于视觉识别任务，而更多的视觉 Token 则对需要区域级特征的多模态理解任务有益</span>。因此，**CoCa&#x20;**&#x91C7;用任务特定的注意力池化来定制用于不同类型训练目标和下游任务的视觉表示。在这里，池化器是一个具有$$n_{\text{query}}$$可学习 Query 的单个多头注意力层，编码器输出作为键和值。通过这种方式，模型可以学习为两种训练目标生成不同长度的嵌入，如上图所示。任务特定池化的使用不仅满足了不同任务的需求，还引入了池化器作为自然的任务适配器。**CoCa&#x20;**&#x5728;预训练中为生成损失使用$$n_{\text{query}} = 256$$，为对比损失使用$$n_{\text{query}} = 1$$。

**<span style="color: rgb(36,91,219); background-color: inherit">预训练效率</span>**

解耦自回归解码器设计的一个关键优势在于它能够高效计算两种训练损失。由于单向语言模型在完整句子上使用因果掩码进行训练，<span style="color: rgb(100,37,208); background-color: inherit">解码器可以通过一次前向传播高效生成对比损失和生成损失的输出，相比双向方法需要两次前向传播</span>。因此，<span style="color: rgb(100,37,208); background-color: inherit">这两种损失之间的大部分计算是共享的，</span>**<span style="color: rgb(100,37,208); background-color: inherit">CoCa </span>**<span style="color: rgb(100,37,208); background-color: inherit">相比标准编码器-解码器模型仅引入了最小的开销</span>。另一方面，尽管许多现有方法在多种数据源和/或模态上分多个阶段训练模型组件，**CoCa&#x20;**&#x901A;过将所有标签视为文本，直接从零开始端到端预训练于各种数据源，即标注图像和噪声替代文本图像，同时服务于对比目标和生成目标。

* **<span style="color: rgb(36,91,219); background-color: inherit">下游任务</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Zero-Shot 迁移</span>**

预训练的 CoCa 模型通过结合图像和文本输入，能够以 Zero-Shot 方式执行多种任务，包括 Zero-Shot 图像分类、 Zero-Shot 图文跨模态检索以及 Zero-Shot 视频-文本跨模态检索。遵循以往的做法， Zero-Shot 在这里与经典的 Zero-Shot 学习有所不同，因为在预训练期间，模型可能会接触到相关的监督信息，但在迁移中不会使用任何监督样本。对于预训练数据，<span style="color: rgb(100,37,208); background-color: inherit">CoCa 遵循以往工作引入的严格去重程序，过滤掉所有接近下游任务领域的样本</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">冻结特征评估</span>**

CoCa 采用任务特定的注意力池化来为不同类型的下游任务定制视觉表示，同时共享主干编码器。这<span style="color: rgb(100,37,208); background-color: inherit">使得模型作为冻结编码器时能够获得强大的性能，这里只需学习一个新的池化器来聚合特征即可</span>。这种方法也适用于多任务问题，其中共享相同的冻结图像编码器计算，但使用不同的任务特定头部。正如之前工作所验证的，线性评估难以准确衡量学习到的表示，而注意力池化在实际应用中更具实用性。

* **<span style="color: rgb(36,91,219); background-color: inherit">CoCa 在视频动作识别中的应用</span>**

**CoCa&#x20;**&#x91C7;用一种简单的方法使预训练模型适用于视频动作识别任务。首先，<span style="color: rgb(100,37,208); background-color: inherit">提取视频的多帧图像，并将每一帧单独输入共享的图像编码器</span>，如右图所示。对于冻结特征评估或微调，在空间和时间特征 Token 之上学习一个额外的池化器，并使用 softmax 交叉熵损失进行优化。<span style="color: rgb(100,37,208); background-color: inherit">池化器只有一个 Query Token，因此对所有空间和时间标记进行池化的计算开销并不高</span>。对于 Zero-Shot 视频-文本检索，采用了一种更简单的方法，即通过对视频&#x7684;**`16`**&#x5E27;计算平均 Embedding，<span style="color: rgb(100,37,208); background-color: inherit">帧从视频中均匀采样</span>。在计算检索指标时，还将每个视频的字幕编码为目标 Embedding。

![](../../images/视觉多模态讲义（上）-image-108.png)

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

CoCa 展现了令人印象深刻的能力。<span style="color: rgb(100,37,208); background-color: inherit">无论是在图像分类、跨模态检索还是图文生成任务中，CoCa 都取得了领先的成绩。尤其是在需要兼顾理解与生成的任务中，其 Encoder-Decoder 架构展现出了无可比拟的优势</span>。

然而，挑战依然存在。例如，<span style="color: rgb(220,155,4); background-color: inherit">如何进一步降低模型的计算成本？如何在小样本或低质量数据的情况下保持高性能？</span>这些都是未来研究需要解决的问题。

### 2.3.3 <span style="color: rgb(36,91,219); background-color: inherit">BEiT-3</span>

传统的深度学习模型通常针对特定的任务进行设计，例如图像分类、目标检测或自然语言处理。然而，这种单一任务驱动的设计方法在面对复杂现实场景时显得力不从心。<span style="color: rgb(100,37,208); background-color: inherit">如何构建一个通用的框架，使其既能独立处理不同模态的数据，又能实现跨模态的协同工作，成为了学术界和工业界共同关注的问题</span>。

**BEiT-3** 的出现正是为了应对这一挑战。其核心思想是<span style="color: rgb(100,37,208); background-color: inherit">将不同模态的数据视为同一种</span>`语言`<span style="color: rgb(100,37,208); background-color: inherit">，通过自监督学习的方式，让模型从大量无标签数据中提取出丰富的特征表示</span>。这种方法不仅提高了模型的泛化能力，还使得它能够灵活地应用于各种下游任务。

![](../../images/视觉多模态讲义（上）-image-109.png)

如上图所示，**<span style="color: rgb(100,37,208); background-color: inherit">BEiT-3</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 通过在单模态和多模态数据上进行掩码数据建模预训练，使用共享的多路 Transformer 网络。该模型可以迁移到各种视觉和视觉-语言下游任务中</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：目前无论是视觉、语言还是多模态，都有大一统的趋势，把模型做大做强，**BEiT-3** 采用大一统的思想，使用<span style="color: rgb(100,37,208); background-color: inherit">更大一统的框架，模型结构统一，目标函数统一，模型大小统一，模型大小和数据集大小整体变大</span>。作者把大一统再往前推一步，彻底把 **VLP** 做得更好
>
> 之前的模型结构：
>
> * Dual Encoder 方式&#x7684;**`CLIP`**，做检索
>
> * Encoder-Decoder 做生成任务
>
> * Fusion Encoder，只用 Encoder 且多模态融合，&#x5982;**`ALBEF`**，**`VLMo`**，做图像文本理解
>
> 但是<span style="color: rgb(216,57,49); background-color: inherit">这些都需要根据下游任务做改变</span>，因此提出 BEiT-3

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

**BEiT-3** 使用多路 Transformer 作为骨干模型来编码不同模态的数据。如上图所示，<span style="color: rgb(100,37,208); background-color: inherit">每个多路 Transformer 块包含一个共享的自注意力模块和一组前馈网络，即模态专家，用于处理不同的模态</span>。**BEiT-3** 将每个输入 Token 根据其模态路由到相应的专家。在实现中，<span style="color: rgb(100,37,208); background-color: inherit">每一层包含一个视觉专家和一个语言专家。此外，顶层的三层还设计了用于融合编码器的视觉-语言专家</span>，如下图所示。使用一组模态专家鼓励模型捕获更多模态特定的信息。共享的自注意力模块学习不同模态之间的对齐，并为多模态任务提供深度融合。

![](../../images/视觉多模态讲义（上）-image-107.png)

如上图所示，统一的架构使 **BEiT-3&#x20;**&#x80FD;够支持广泛的下游任务。例如，**<span style="color: rgb(220,155,4); background-color: inherit">BEiT-3</span>**<span style="color: rgb(220,155,4); background-color: inherit"> 可以用作各种视觉任务的图像骨干网络，包括图像分类、目标检测、实例分割和语义分割。它还可以微调为双编码器以实现高效的图文检索，或作为多模态理解和生成任务的融合模型</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：模型使&#x7528;**`VLMo`**&#x63D0;出的 MoME，即 Multway Transformer，结&#x5408;**`BEiT`**，**`BEiT v2`**，**`VLBEiT`**，做大做强

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练任务</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">BEiT-3 </span>**<span style="color: rgb(100,37,208); background-color: inherit">通过统一的掩码数据建模目标，在图像和文本单模态和图文对多模态数据上进行预训练</span>。在预训练过程中，随机遮蔽一定比例的文本 Token 或图像 Patch，并训练模型恢复被遮蔽的 Token。统一的先掩码再预测任务不仅学习表示，还学习不同模态之间的对齐。具体而言，文本数据通&#x8FC7;**`SentencePiece`**&#x5206;词器进行分词。图像数据通&#x8FC7;**`BEiT v2`**&#x7684;分词器获得离散的视觉 Token 作为重建目标。**<span style="color: rgb(100,37,208); background-color: inherit">BEiT-3 </span>**<span style="color: rgb(100,37,208); background-color: inherit">随机遮蔽单模态文本的</span>**`15%`**<span style="color: rgb(100,37,208); background-color: inherit"> Token 和图文对中文本的</span>**`50%`**<span style="color: rgb(100,37,208); background-color: inherit"> Token。对于图像，使用类似于</span>**`BEiT`**<span style="color: rgb(100,37,208); background-color: inherit">的分块掩码策略遮蔽</span>**`40%`**<span style="color: rgb(100,37,208); background-color: inherit">的图像 Patch</span>。

**<span style="color: rgb(46,161,33); background-color: inherit">BEiT-3 </span>**<span style="color: rgb(46,161,33); background-color: inherit">仅使用一个预训练任务，这使得训练过程更容易扩展</span>。相比之下，之前的视觉-语言模型通常采用多个预训练任务，例如<span style="color: rgb(220,155,4); background-color: inherit">图文对比、图文匹配和词-块/区域对齐</span>。<span style="color: rgb(46,161,33); background-color: inherit">使用</span>**<span style="color: rgb(46,161,33); background-color: inherit">先掩码再预测</span>**<span style="color: rgb(46,161,33); background-color: inherit">任务时，可以使用更小的预训练批次大小</span>。相比之下，<span style="color: rgb(216,57,49); background-color: inherit">基于对比的模型通常需要非常大的批次大小进行预训练，这带来了更多的工程挑战</span>，例如 <span style="color: rgb(220,155,4); background-color: inherit">GPU 内存消耗</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：Mask Data Modeling 已应用到各个模态，&#x5982;**`BERT`**，**`BEiT`**，**`VLBEiT`**&#x7B49;，<span style="color: rgb(46,161,33); background-color: inherit">用掩码学习可以学到很好的图像、文本、多模态特征</span>。但是多个目标函数存在训练慢、优化和调参、loss互补和互斥的问题。因此作者只用一个目标函数，把图像看作一种语言：图像&#x662F;**`Imglish`**，文本&#x662F;**`English`**，图像文本对&#x662F;**`Parallel Sentence`**，都&#x7528;**`Mask Modeling Loss`**&#x6765;训练

* **<span style="color: rgb(36,91,219); background-color: inherit">BEiT-3 预训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">骨干网络</span>**

**BEiT-3** 是一个巨大的基础模型，遵&#x5FAA;**`ViT-giant`**&#x7684;设置

![](../../images/视觉多模态讲义（上）-image-106.png)

**BEiT-3** &#x7531;**`40`**&#x5C42;多路 Transformer 组成，具&#x6709;**`1408`**&#x7684;隐藏大小、**`6144`**&#x7684;中间大小&#x548C;**`16`**&#x4E2A;注意力头。所有层都包含视觉专家和语言专家。视觉-语言专家也被部署在顶层的三个多路 Transformer 层中。自注意力模块在不同模态之间共享。**BEiT-3** 总共包&#x542B;**`1.9B`**&#x53C2;数，其中视觉专家&#x5360;**`692M`**&#x53C2;数，语言专家&#x5360;**`692M`**&#x53C2;数，视觉-语言专家&#x5360;**`52M`**&#x53C2;数，共享的自注意力模块&#x5360;**`317M`**&#x53C2;数。需要注意的是，当模型用作视觉编码器时，只有与视觉相关的参数被激活，即与 **ViT-giant** 相媲美的规模，&#x7EA6;**`1B`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练数据</span>**

**BEiT-3** 在单模态和多模态数据上进行预训练，如右表所示。

![](../../images/视觉多模态讲义（上）-image-127.png)

对于多模态数据，从五个公共数据集中收集了&#x7EA6;**`15M`**&#x5F20;图像&#x548C;**`21M`**&#x56FE;文对，这些数据集包括Conceptual 12M **`CC12M`**、Conceptual Captions **`CC3M`**、SBU Captions **`SBU`**、**`COCO`**&#x548C; Visual Genome **`VG`**。对于单模态数据，使用来自 ImageNet-21K &#x7684;**`14M`**&#x5F20;图像和来自英文维基百科、BookCorpus、OpenWebText、CC-News 和 Stories &#x7684;**`160GB`**&#x6587;本语料库。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：模型和数据集全部 Scale Up，把模型大小扩展到B量级，数据集也很大，但&#x548C;**`CoCa`**&#x6BD4;数据少很多，因为数据质量很关键

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练设置</span>**

**BEiT-3** 进行&#x4E86;**`1M`**&#x6B65;的预训练。每个批次总共包&#x542B;**`6144`**&#x4E2A;样本，其中包&#x62EC;**`2048`**&#x5F20;图像、**`2048`**&#x6BB5;文本&#x548C;**`2048`**&#x4E2A;图文对。批次大小比对比模型小得多。**BEiT-3** 使&#x7528;**`14×14`**&#x7684;块大小，并&#x5728;**`224×224`**&#x5206;辨率下进行预训练。这里使用&#x4E0E;**`BEiT`**&#x76F8;同的图像增强技术，包括随机调整裁剪、水平翻转和颜色抖动。文本数据使用词汇量&#x4E3A;**`64k`**&#x7684;**`SentencePiece`**&#x5206;词器进行分词。使&#x7528;**`AdamW`**&#x4F18;化器进行优化，$$β1=0.9，β2=0.98，ε=1e-6$$。采用余弦学习率衰减调度器，峰值学习率&#x4E3A;**`1e-3`**，并进&#x884C;**`10k`**&#x6B65;的线性热启动。权重衰减&#x4E3A;**`0.05`**。，并且使用速率&#x4E3A;**`0.1`**&#x7684;随机深度技术。为了稳定 Transformer 训练，使用 **BEiT&#x20;**&#x521D;始化算法。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**BEiT-3** 有以下几个关键技术要点：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">统一的 Transformer 架构</span>**：通过将不同模态的数据编码为统一的序列形式，BEiT-3 实现了真正的多模态融合
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">掩码数据建模</span>**：作为核心训练方法，掩码数据建模使模型能够从大量无标签数据中学习到丰富的特征表示
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">多任务联合训练</span>**：结合多种辅助任务，进一步增强了模型的表示能力
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">广泛的适用性</span>**：无论是视觉任务还是视觉-语言任务，BEiT-3 都展现出了卓越的迁移性能

---

[Previous](06-Vision-Language-Model--BLIP-家族.md) | [Contents](../../README.md) | [Next](08-Vision-Language-Model--LLaVA-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=7)
