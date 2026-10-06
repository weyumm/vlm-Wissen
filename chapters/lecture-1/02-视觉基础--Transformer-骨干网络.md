[Previous](01-视觉基础--CNN-骨干网络.md) | [Contents](../../README.md) | [Next](03-视觉基础--对比学习.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=2)

## 1.2 <span style="color: rgb(36,91,219); background-color: inherit">Transformer 骨干网络</span>

### 1.2.1 <span style="color: rgb(36,91,219); background-color: inherit">Vision Transformer</span>

卷积神经网络长期主导计算机视觉，其<span style="color: rgb(100,37,208); background-color: inherit">核心假设是</span>**<span style="color: rgb(100,37,208); background-color: inherit">局部性先验</span>**<span style="color: rgb(100,37,208); background-color: inherit"> locality prior 和</span>**<span style="color: rgb(100,37,208); background-color: inherit">平移等变性</span>**<span style="color: rgb(100,37,208); background-color: inherit"> translation equivariance</span>。然而，<span style="color: rgb(216,57,49); background-color: inherit">这种</span>**<span style="color: rgb(216,57,49); background-color: inherit">归纳偏置</span>**<span style="color: rgb(216,57,49); background-color: inherit"> inductive bias 限制了模型对全局上下文关系的建模能力</span>。

近年来，深度学习领域的发展日新月异，尤其是 Transformer 模型的崛起，彻底改变了自然语言处理的研究范式。然而，Transformer 的强大能力并不仅限于文本数据，它同样可以扩展到其他模态的任务中。2020年，Google Brain 团队提出了一种开创性的工作——<span style="color: rgb(100,37,208); background-color: inherit">Vision Transformer (ViT)，将 Transformer 架构成功应用于计算机视觉任务。这一工作不仅标志着 Transformer 在图像分类任务中的首次大规模应用，还为后续的视觉研究奠定了基础</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">核心思想</span>**

> 🥖 **ViT&#x20;**&#x7684;核心命题是：<span style="color: rgb(100,37,208); background-color: inherit">抛弃卷积操作，将图像视为由图像块 patch 组成的序列，通过自注意力机制实现全局交互</span>。其突破性在于证明：<span style="color: rgb(46,161,33); background-color: inherit">当训练数据足够庞大时，如</span>**`JFT-300M`**<span style="color: rgb(46,161,33); background-color: inherit">，无空间先验的纯 Transformer 架构可超越 CNN</span>。

传统的卷积神经网络通过局部感受野和权重共享机制来提取图像特征，而 **ViT&#x20;**&#x5219;另辟蹊径，采用了一种全新的思路：<span style="color: rgb(100,37,208); background-color: inherit">将图像视为</span>**<span style="color: rgb(100,37,208); background-color: inherit">词序列</span>**。具体来说，**<span style="color: rgb(100,37,208); background-color: inherit">ViT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 首先将输入图像分割成固定大小的小块 Patches，每个小块被视为一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">词</span>**<span style="color: rgb(100,37,208); background-color: inherit">或 </span>**<span style="color: rgb(100,37,208); background-color: inherit">token</span>**。例如，<span style="color: rgb(220,155,4); background-color: inherit">对于一张分辨率为</span>**`224×224`**<span style="color: rgb(220,155,4); background-color: inherit">的RGB图像，若每个 Patch 的大小为</span>**`16×16`**<span style="color: rgb(220,155,4); background-color: inherit">，则整张图像会被划分为</span>**`196`**<span style="color: rgb(220,155,4); background-color: inherit">个 Patch</span>。<span style="color: rgb(100,37,208); background-color: inherit">这些 Patch 随后被展平并嵌入到高维空间中，形成一个类似于 NLP 中词嵌入的表示形式</span>。

这种<span style="color: rgb(46,161,33); background-color: inherit">非重叠的 Patch Embedding 方法是 ViT 的核心设计之一，它使得 Transformer 能够直接处理图像数据，而无需依赖复杂的卷积操作</span>。此外，<span style="color: rgb(100,37,208); background-color: inherit">为了保留图像的空间信息，</span>**<span style="color: rgb(100,37,208); background-color: inherit">ViT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 还在每个 patch 的嵌入中添加了位置编码 positional encoding，以显式地引入 Patch 的位置信息</span>。

位置编码的设计有多种实现方式，主要分为两类：

> ⛱️ 1. 基于固定算法生成的<span style="color: rgb(100,37,208); background-color: inherit">正余弦函数编码</span>，类似于原始 Transformer 中的实现
>
> 2. <span style="color: rgb(100,37,208); background-color: inherit">可学习的位置编码</span>

**ViT** 选择了后者，即通过训练优化位置编码参数，从而更好地适应图像数据的特性。

* **<span style="color: rgb(36,91,219); background-color: inherit">架构设计</span>**

**ViT** 的整体架构与标准的 Transformer 模型非常相似，但针对图像数据的特点进行了适当的调整，如下图所示。其主要组成部分包括以下几个模块：

![](../../images/视觉多模态讲义（上）-image-17.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Patch Embedding层</span>**：**<span style="color: rgb(100,37,208); background-color: inherit">ViT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 首先将图像分割为多个 patch，并通过一个线性投影层将其映射到高维空间</span>。这一过程可以看作是对图像的初步特征提取。假设输入图像的尺寸为$$H \times W \times C$$（高度、宽度和通道数），每个Patch 的大小为$$P \times P$$，则经过分割后，图像被转换为一个长度为$$N = \frac{H}{P} \cdot \frac{W}{P}$$的序列，每个 Patch 的维度为$$P^2 \cdot C$$。线性投影层进一步将每个 Patch 映射到一个固定维度的 embedding 向量。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">位置编码</span>**：由于 Transformer 本身不具备对输入顺序的感知能力，**ViT** 引入了可学习的位置编码，用于捕捉 Patch 之间的空间关系。这些位置编码与 Patch embedding 相加后，作为 Transformer 的输入。这里需要注意的是，<span style="color: rgb(100,37,208); background-color: inherit">位置编码的长度与 patch 的数量相同，因此它的维度也受到图像分辨率的影响</span>。
>
> 👍 3) **<span style="color: rgb(36,91,219); background-color: inherit">Transformer Encoder</span>**：**ViT** 的核心是一个标准的 Transformer Encoder，包含多头自注意力机制 **<span style="color: rgb(216,57,49); background-color: inherit">MHSA</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ulti-</span>**<span style="color: rgb(216,57,49); background-color: inherit">H</span>**<span style="color: rgb(216,57,49); background-color: inherit">ead </span>**<span style="color: rgb(216,57,49); background-color: inherit">S</span>**<span style="color: rgb(216,57,49); background-color: inherit">elf-</span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">ttention）</span>和前馈神经网络 **<span style="color: rgb(216,57,49); background-color: inherit">FFN</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">eed-</span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">orward </span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">etwork）</span>。<span style="color: rgb(100,37,208); background-color: inherit">通过多层堆叠，Transformer 能够捕捉图像全局的上下文信息，从而实现对复杂模式的学习</span>。每一层的输出都会通过残差连接 Residual Connection 和层归一化 Layer Normalization 进行优化，以提高训练的稳定性和收敛速度。
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">分类头</span>**：在 Transformer 的最后一层输出中，ViT 引入了一个特殊的分类 Token：**`Class Token`**，用于汇总整个图像的信息。这个 Token 在输入时被插入到 Patch 序列的最前端，并与其他 Patch 一起参与 Transformer 的计算。最终，分类 Token 的输出经过一个全连接层后，生成最终的分类预测结果。

整个网络的公式化表示如下：

$$\begin{aligned}
z_0 &= [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \cdots; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{pos},  \quad 
\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}, \quad \mathbf{E}_{pos} \in \mathbb{R}^{(N+1) \times D}, \\
z_{\ell}' &= \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \quad \ell = 1 \ldots L, \\
z_{\ell} &= \text{MLP}(\text{LN}(z_{\ell}')) + z_{\ell}', \quad \ell = 1 \ldots L, \\
\mathbf{y} &= \text{LN}(\mathbf{z}_L^0).
\end{aligned}$$

其中$$\mathbf{x}_{\text{class}}$$表示类别标记；$$\mathbf{x}_p^i$$是第$$i$$个图像块的标记；$$\mathbf{E}$$表示嵌入矩阵，用于将输入转换为模型可以处理的形式；$$\mathbf{E}_{pos}$$则是位置嵌入，用于提供序列中元素的位置信息；$$D$$表示嵌入维度；$$L$$表示 Transformer 编码器中的层数。 MSA，LN，MLP 分别表示：多头自注意力机制，层归一化以及多层感知机。

* **<span style="color: rgb(36,91,219); background-color: inherit">技术亮点</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">全局建模能力</span>**：传统的 CNN 通过局部感受野逐步扩大视野范围，而 **<span style="color: rgb(46,161,33); background-color: inherit">ViT </span>**<span style="color: rgb(46,161,33); background-color: inherit">通过自注意力机制直接建模图像的全局依赖关系。这种特性使得 </span>**<span style="color: rgb(46,161,33); background-color: inherit">ViT </span>**<span style="color: rgb(46,161,33); background-color: inherit">在处理需要长距离上下文信息的任务时表现出色</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">细粒度图像分类和目标检测</span>。具体来说，自注意力机制允许模型动态地关注图像中不同区域的重要性，从而避免了手工设计特征的问题。

2. **<span style="color: rgb(36,91,219); background-color: inherit">简单而有效的设计</span>**：**ViT&#x20;**&#x7684;架构相对简洁，几乎完全复用了 NLP 领域的 Transformer 模型，仅需少量改动即可适应图像数据。这种设计<span style="color: rgb(46,161,33); background-color: inherit">不仅降低了实现难度，还提高了模型的可扩展性</span>。此外，**<span style="color: rgb(100,37,208); background-color: inherit">ViT </span>**<span style="color: rgb(100,37,208); background-color: inherit">的成功表明，Transformer 作为一种通用的架构，具有跨模态的潜力，可以在不同的任务中展现出强大的性能</span>。

3) **<span style="color: rgb(36,91,219); background-color: inherit">数据驱动的性能提升</span>**：尽管 **ViT&#x20;**&#x5728;小规模数据集上的表现可能不如 CNN，但在大规模数据集上预训练后，&#x5982;**`ImageNet-21k`**&#x6216;**`JFT-300M`**，**ViT&#x20;**&#x80FD;够展现出超越 CNN 的性能。这一现象表明，**<span style="color: rgb(100,37,208); background-color: inherit">ViT </span>**<span style="color: rgb(100,37,208); background-color: inherit">的潜力更多依赖于数据量和计算资源的支持</span>。事实上，**ViT&#x20;**&#x5728; ImageNet 上的实验结果表明，<span style="color: rgb(46,161,33); background-color: inherit">当数据量足够大时，</span>**<span style="color: rgb(46,161,33); background-color: inherit">ViT </span>**<span style="color: rgb(46,161,33); background-color: inherit">的性能甚至可以媲美最先进的 CNN 模型</span>。

4) **<span style="color: rgb(36,91,219); background-color: inherit">可解释性强</span>**：**<span style="color: rgb(100,37,208); background-color: inherit">ViT </span>**<span style="color: rgb(100,37,208); background-color: inherit">的注意力权重可以直观地展示模型在图像中关注的重点区域，这为模型的可解释性提供了新的途径</span>。通过对注意力图的分析，研究人员可以更好地理解模型的决策过程，从而改进模型的设计。

* **<span style="color: rgb(36,91,219); background-color: inherit">局限性</span>**

尽管 **ViT&#x20;**&#x53D6;得了显著的成功，但它也暴露了一些局限性：

> **ViT&#x20;**&#x5BF9;大规模预训练数据的需求较高，在小数据集上容易过拟合
>
> ⛱️ **ViT&#x20;**&#x7684;计算复杂度随着图像分辨率的增加而急剧上升，这限制了其在高分辨率图像任务中的应用

为了解决这些问题，研究者们提出了多种改进方案。例如，**<span style="color: rgb(216,57,49); background-color: inherit">PVT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">yramid </span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ision </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformer）</span>和 <span style="color: rgb(216,57,49); background-color: inherit">Swin Transformer</span> <span style="color: rgb(220,155,4); background-color: inherit">等模型通过引入金字塔结构和局部窗口注意力机制，显著降低了计算开销，同时提升了模型的灵活性</span>。此外，<span style="color: rgb(100,37,208); background-color: inherit">结合 CNN 和 Transformer 的混合架构也被证明是一种有效的策略，能够在保持 </span>**<span style="color: rgb(100,37,208); background-color: inherit">ViT </span>**<span style="color: rgb(100,37,208); background-color: inherit">全局建模能力的同时，利用 CNN 的局部特征提取优势</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

作者一共设计&#x4E86;**`3`**&#x79CD;规格的模&#x578B;**`Base`**，**`Large`**，**`Huge`**，如右图所示。这里展示一个 pytorch 搭建&#x7684;**`ViT-base`**&#x6A21;型：

![](../../images/视觉多模态讲义（上）-image-19.png)

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**ViT&#x20;**&#x7684;提出不仅推动了 Transformer 在计算机视觉领域的广泛应用，还启发了一系列后续研究，包括目标检测、语义分割和视频理解等任务。随着硬件性能的提升和数据规模的扩大，**ViT&#x20;**&#x53CA;其变体有望在更多实际场景中发挥作用。

总的来说，<span style="color: rgb(100,37,208); background-color: inherit">Vision Transformer 在计算机视觉领域是一项很有意义的工作，它打破了传统 CNN 在视觉领域的主导地位，为深度学习研究开辟了新的方向</span>。

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # q, k, v 的线性变换
        self.qkv = nn.Linear(dim, dim * 3)
        self.dropout = nn.Dropout(dropout)
```


**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**


```python
import torch

# 假设你的 ViT 类名是 VisionTransformer，已经定义好
# from your_vit_module import VisionTransformer

# 先创建一个随机图像张量
# 假设输入图像大小是 224x224，3 通道 (RGB)，batch 大小是 1
img = torch.randn(1, 3, 224, 224)  # torch.randn 生成标准正态分布的随机数 :contentReference[oaicite:0]{index=0}

# 创建 ViT 模型实例
```


```python
class VisionTransformer(nn.Module):
    def __init__(self, *, img_size=224, patch_size=16, num_classes=1000,
                 dim=768, depth=12, num_heads=12, mlp_dim=3072, dropout=0.0):
        super().__init__()
        # patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels=3, embed_dim=dim)
        num_patches = (img_size // patch_size) ** 2

        # learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, dim))  # +1 for cls token
        self.pos_dropout = nn.Dropout(dropout)
```


```python
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout)
```


```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
```

### 1.2.2 <span style="color: rgb(36,91,219); background-color: inherit">DeiT</span>

传统的卷积神经网络在图像分类任务中表现出色，但其设计<span style="color: rgb(216,57,49); background-color: inherit">依赖于手工构造的特征提取机制，难以适应多样化的应用场景</span>。而 Transformer，尤其是基于自注意力机制的架构，因其强大的全局建模能力，<span style="color: rgb(46,161,33); background-color: inherit">在自然语言处理领域取得了巨大成功</span>。

**ViT&#x20;**&#x7684;提出首次将 Transformer 引入图像分类任务，并在大规模数据集上展现了卓越的性能。**<span style="color: rgb(216,57,49); background-color: inherit">ViT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Vi</span>**<span style="color: rgb(216,57,49); background-color: inherit">sion </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformer）</span>的出现为图像分类任务提供了一种全新的视角。然而，**<span style="color: rgb(216,57,49); background-color: inherit">ViT</span>**<span style="color: rgb(216,57,49); background-color: inherit"> 的成功依赖于海量的训练数据和强大的计算资源：数百万张标注图像和大量 GPU，这使得其在实际应用中面临诸多限制，对于许多实际场景来说是不现实的</span>。

针对这一问题，**`Facebook AI`**&#x56E2;队提出了 **<span style="color: rgb(216,57,49); background-color: inherit">DeiT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">ata-</span>**<span style="color: rgb(216,57,49); background-color: inherit">e</span>**<span style="color: rgb(216,57,49); background-color: inherit">fficient </span>**<span style="color: rgb(216,57,49); background-color: inherit">i</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformers）</span>，**<span style="color: rgb(100,37,208); background-color: inherit">DeiT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的核心目标是解决 </span>**<span style="color: rgb(100,37,208); background-color: inherit">ViT </span>**<span style="color: rgb(100,37,208); background-color: inherit">的问题，通过改进训练方法和引入知识蒸馏策略，使得 Transformer 能够在较小的数据集上达到与 </span>**<span style="color: rgb(100,37,208); background-color: inherit">ViT </span>**<span style="color: rgb(100,37,208); background-color: inherit">相当甚至更高的性能</span>。这种数据高效的特性使得 **DeiT&#x20;**&#x6210;为资源受限环境下的理想选择。<span style="color: rgb(46,161,33); background-color: inherit">这项工作不仅推动了 Transformer 在计算机视觉领域的普及，还为模型优化和知识蒸馏技术提供了重要的启示</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">核心方法</span>**

**DeiT&#x20;**&#x5047;设可以使用一个强大的图像分类器作为教师模型。这个教师模型可以是一个卷积神经网络，也可以是多个分类器的混合。**DeiT&#x20;**&#x901A;过比较准确率与图像吞吐量之间的权衡，用 Transformer 替换卷积神经网络可能是有益的。这里首先介绍蒸馏的两个维度：硬蒸馏与软蒸馏，以及经典蒸馏与蒸馏 Token。 &#x20;

**<span style="color: rgb(36,91,219); background-color: inherit">软蒸馏</span>**

软蒸馏旨在最小化教师模型的 softmax 输出与学生模型的 softmax 输出之间的 KL 散度。 &#x20;

设 $$Z_t$$ 表示教师模型的 logits，$$Z_s$$表示学生模型的 logits。用$$\tau$$表示蒸馏的温度参数，$$\lambda$$表示平衡 KL 散度损失和基于真实标签 $$y$$ 的交叉熵损失$$\mathcal{L}_{CE}$$的系数，$$\psi$$表示 softmax 函数。蒸馏的目标函数为： &#x20;

$$\mathcal{L}_{\text{global}} = (1 - \lambda) \mathcal{L}_{\text{CE}}(\psi(Z_s), y) + \lambda \tau^2 \text{KL}(\psi(Z_s / \tau), \psi(Z_t / \tau))$$

其中，第一项是学生模型对真实标签的预测误差，通过交叉熵损失计算。第二项则衡量教师模型和学生模型输出分布之间的差异，通过KL散度计算。

**<span style="color: rgb(36,91,219); background-color: inherit">硬蒸馏 </span>**

DeiT 引入了一种蒸馏的变体，在这种方法中，将教师模型的硬决策 hard decision 视为真实标签。设$$y_t = \arg\max_c Z_t(c)$$为教师模型的硬决策，则与这种硬标签蒸馏相关的目标函数为： &#x20;

$$\mathcal{L}_{\text{hardDistill}}^{\text{global}} = \frac{1}{2} \mathcal{L}_{\text{CE}}(\psi(Z_s), y) + \frac{1}{2} \mathcal{L}_{\text{CE}}(\psi(Z_s), y_t)$$

对于给定的图像，与教师模型相关的硬标签可能会因特定的数据增强而发生变化。这种选择优于传统方法，同时它无需额外参数且概念上更简单：教师模型的预测$$y_t$$在此方法中扮演了与真实标签$$y$$相同的角色。还需注意的是，硬标签也可以通过标签平滑 label smoothing 转换为软标签。在标签平滑中，真实标签被认为具有$$1 - \epsilon$$的概率，而剩余的$$\epsilon$$则均匀分配到其他类别上。在所有使用真实标签的实验中，将该参数固定为$$\epsilon = 0.1$$。

**<span style="color: rgb(36,91,219); background-color: inherit">蒸馏 Token</span>**

如右图所示，在包括图像&#x5757;**`patch tokens`**&#x4EE5;及类别 Toke&#x6E;**`class token`**&#x7684;初始嵌入中添加了一个新的标记，即蒸馏 Token **`distillation token`**。蒸馏 Token 与类别 Token 的使用方式类似：<span style="color: rgb(100,37,208); background-color: inherit">它通过自注意力机制与其他嵌入进行交互，并在网络的最后一层输出</span>。它的目标由损失函数中的蒸馏部分决定。蒸馏嵌入使 DeiT 能够像常规蒸馏一样从教师模型的输出中学习，同时仍然与类别嵌入保持互补性。 &#x20;

DeiT 团队观察到学习到的类别标记和蒸馏标记会收敛到不同的向量：<span style="color: rgb(100,37,208); background-color: inherit">它们之间的平均余弦相似度仅为</span>**`0.06`**<span style="color: rgb(100,37,208); background-color: inherit">。由于类别嵌入和蒸馏嵌入在每一层都会被计算，它们在网络中逐渐变得更加相似，直到最后一层时它们的相似度变得非常高，余弦相似度到了</span>**`0.93`**。这是符合预期的，因为它们的目标是生成相似但不完全相同的输出。

DeiT 验证了蒸馏 Token 确实为模型带来了额外的价值：作者尝试了一种带有两个类别 Token 的 Transformer 模型，其中一个类别标记替代了教师伪标签。即使这两个标记在初始化时是随机且独立的，在训练过程中它们会收敛到几乎相同的向量，余弦相似度达&#x5230;**`0.999`**，并且输出嵌入也几乎是完全一致的。这种额外的类别标记对分类性能没有任何提升。相比之下，蒸馏策略相较于普通的蒸馏基线方法有显著改进。

![](../../images/视觉多模态讲义（上）-image-20.png)

**<span style="color: rgb(36,91,219); background-color: inherit">使用蒸馏进行微调</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>&#x20;

<span style="color: rgb(100,37,208); background-color: inherit">在更高分辨率下的微调阶段，同时使用真实标签和教师模型的预测结果，并且使用的教师模型具有相同的目标分辨率</span>。**DeiT&#x20;**&#x4E5F;尝试过仅使用真实标签进行微调，但这减少了教师模型的作用，导致性能下降。

**<span style="color: rgb(36,91,219); background-color: inherit">联合分类器</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

在测试阶段，Transformer 生成的类别嵌入和蒸馏嵌入都可以与线性分类器结合，用于推断图像标签。然而 **DeiT&#x20;**<span style="color: rgb(100,37,208); background-color: inherit">将这两个独立的分类头进行后期融合，具体做法是将两个分类器的 softmax 输出相加以做出最终预测</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">模型规模</span>**

较大的模&#x578B;**`DeiT-B`**&#x5177;有&#x4E0E;**`ViT-B`**&#x76F8;同的架构。不同模型之间唯一变化的参数是嵌入维度和注意力头的数量，DeiT 保持每个头的维度恒定，即等&#x4E8E;**`64`**。

![](../../images/视觉多模态讲义（上）-image-18.png)

较小的模型具有更少的参数量和更快的吞吐量。吞吐量是在分辨率&#x4E3A;**`224×224`**&#x7684;图像上测量的。

* **<span style="color: rgb(36,91,219); background-color: inherit">性能分析</span>**

**DeiT&#x20;**&#x5728;多个基准数据集上的实验结果充分证明了其有效性。在 ImageNet-1k 数据集上，**`DeiT-B`**&#x5728; Top-1 准确率上达到&#x4E86;**`81.8%`**，<span style="color: rgb(46,161,33); background-color: inherit">与 ViT-B 相当，但训练所需的数据量和计算资源却大幅减少</span>。此外，通过蒸馏训练，**`DeiT-Distilled`**&#x7248;本进一步将准确率提升&#x81F3;**`83.4%`**，超越了许多经典的卷积网络模型。这些结果表明，**<span style="color: rgb(46,161,33); background-color: inherit">DeiT </span>**<span style="color: rgb(46,161,33); background-color: inherit">不仅在数据效率上具有优势，还在性能上达到了行业领先水平</span>。

进一步的消融实验显示，<span style="color: rgb(100,37,208); background-color: inherit">基于 Token 的蒸馏方法对模型性能的提升起到了关键作用</span>。<span style="color: rgb(216,57,49); background-color: inherit">在未使用蒸馏 Token 的情况下，</span>**<span style="color: rgb(216,57,49); background-color: inherit">DeiT </span>**<span style="color: rgb(216,57,49); background-color: inherit">的 Top-1 准确率下降了约</span>**`1.5%`**。这表明蒸馏 Token 在捕获教师模型知识方面的重要性。

* **<span style="color: rgb(36,91,219); background-color: inherit">核心创新点</span>**

**DeiT&#x20;**&#x7684;技术亮点主要体现在以下几个方面：

> 🍞 1. **<span style="color: rgb(36,91,219); background-color: inherit">基于 Token 的知识蒸馏</span>**：**DeiT&#x20;**&#x901A;过引入一个特殊的**蒸馏 Token** 来捕获教师模型的知识。**DeiT&#x20;**&#x5728;输入序列中添加了一个额外的 Token，用于表示教师模型的预测分布。这种方法<span style="color: rgb(46,161,33); background-color: inherit">不仅保留了原始 Transformer 的结构，还显著提高了模型的泛化能力</span>。相比于传统的软标签蒸馏方法，基于 Token 的蒸馏更加灵活，能够更好地适应 Transformer 的架构特点。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">混合教师模型</span>**：在蒸馏过程中，**DeiT&#x20;**&#x91C7;用了一种混合教师模型的策略。它结合了卷积神经网络和 Transformer 模型的优点，利用卷积网络的局部特征提取能力和 Transformer 的全局建模能力，生成高质量的指导信号。这种跨模态的蒸馏方式<span style="color: rgb(46,161,33); background-color: inherit">不仅提升了学生模型的性能，还为多模态学习提供了新的思路</span>。
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">数据高效的训练策略</span>**：**DeiT&#x20;**&#x901A;过优化训练流程，大幅减少了对大规模标注数据的依赖。例如，<span style="color: rgb(220,155,4); background-color: inherit">它仅需 ImageNet-1k 数据集即可达到 SOTA 性能，而无需像 </span>**<span style="color: rgb(220,155,4); background-color: inherit">ViT </span>**<span style="color: rgb(220,155,4); background-color: inherit">那样依赖 JFT-300M 等超大规模数据集</span>。此外，DeiT 的训练过程可以&#x5728;**`4`**&#x5757;GPU上完成，耗时仅三天，相较于ViT的训练成本显著降低。这种高效性使其在学术研究和工业应用中都具有广泛的适用性。
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">模型压缩与加速</span>**：为了进一步提升 DeiT 的实用性，研究者还探索了模型压缩和加速技术。例如，<span style="color: rgb(220,155,4); background-color: inherit">通过多准则 Token 融合方法，</span>**<span style="color: rgb(220,155,4); background-color: inherit">DeiT </span>**<span style="color: rgb(220,155,4); background-color: inherit">成功将计算复杂度 FLOPs 减少了</span>**`44%`**<span style="color: rgb(220,155,4); background-color: inherit">，同时保持了较高的分类精度</span>。这种即插即用的优化策略为 Transformer 模型的实际部署提供了重要支持。
>
> 5) **<span style="color: rgb(36,91,219); background-color: inherit">正则化与数据增强</span>**：为了提高模型的泛化能力，DeiT 采用了多种正则化技术，&#x5982;**`Mixup`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`CutMix`**<span style="color: rgb(220,155,4); background-color: inherit">和随机深度</span>**`Stochastic Depth`**。这些技术通过引入噪声或随机性，增强了模型对不同输入分布的适应能力。此外，**<span style="color: rgb(46,161,33); background-color: inherit">DeiT</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 还结合了</span>**`AutoAugment`**<span style="color: rgb(46,161,33); background-color: inherit">和</span>**`RandAugment`**<span style="color: rgb(46,161,33); background-color: inherit">等数据增强方法，进一步提升了模型的鲁棒性和准确性</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**<span style="color: rgb(46,161,33); background-color: inherit">DeiT </span>**<span style="color: rgb(46,161,33); background-color: inherit">的成功为 Transformer 在计算机视觉领域的广泛应用铺平了道路。其数据高效的特性使其特别适合于医疗影像、卫星图像分析等标注数据稀缺的领域</span>。然而，**<span style="color: rgb(216,57,49); background-color: inherit">DeiT </span>**<span style="color: rgb(216,57,49); background-color: inherit">也面临着一些挑战。例如，如何在更复杂的任务中进一步优化 Transformer 的性能，如目标检测和语义分割，仍然是一个开放性问题</span>。此外，随着模型规模的增大，如何平衡计算效率与性能也是一个亟待解决的难题。

**<span style="color: rgb(46,161,33); background-color: inherit">DeiT </span>**<span style="color: rgb(46,161,33); background-color: inherit">作为数据高效图像 Transformer 的代表作，不仅展示了 Transformer 在图像分类任务中的潜力，还为知识蒸馏和模型优化提供了宝贵的实践经验</span>。它的创新之处在于<span style="color: rgb(46,161,33); background-color: inherit">通过改进训练方法和引入蒸馏 Token，显著降低了 Transformer 对大规模数据和计算资源的依赖</span>。

### 1.2.3 <span style="color: rgb(36,91,219); background-color: inherit">Swin Transformer</span>

在深度学习的发展历程中，Transformer 模型最初因其在自然语言处理领域的卓越表现而声名鹊起。然而，微软亚洲研究院的研究团队在 2021 年提出了一种全新的视觉架构——**Swin Transformer**，将 Transformer 的强大能力引入计算机视觉领域。<span style="color: rgb(100,37,208); background-color: inherit">这一工作不仅为视觉任务提供了新的解决方案，还成功地结合了卷积神经网络和 Transformer的优势，成为图像分类、目标检测、语义分割等任务中的标杆模型</span>。

![](../../images/视觉多模态讲义（上）-image-41.png)

传统的卷积神经网络在计算机视觉领域长期占据主导地位，其通过局部感受野和权重共享机制有效地提取图像的局部特征。然而，<span style="color: rgb(216,57,49); background-color: inherit">随着数据规模的增长和任务复杂度的提升，CNN 在全局建模和多尺度特征提取方面的局限性逐渐显现</span>。与此同时，Transformer 凭借自注意力机制在捕捉长距离依赖关系和全局信息方面表现出色，但其<span style="color: rgb(216,57,49); background-color: inherit">直接应用于高分辨率图像时计算成本过高，限制了其在视觉领域的广泛应用</span>。

**Swin Transformer** 的提出正是为了弥合这一鸿沟。<span style="color: rgb(46,161,33); background-color: inherit">它借鉴了 CNN 的分层结构思想，同时引入了 Transformer 的核心特性——自注意力机制，从而实现了对图像的高效建模</span>。通过这些创新，**<span style="color: rgb(46,161,33); background-color: inherit">Swin Transformer </span>**<span style="color: rgb(46,161,33); background-color: inherit">不仅能够处理超分辨率图像，还能在不同尺度上提取丰富的特征表示，为视觉任务带来了革命性的变化</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

![](../../images/视觉多模态讲义（上）-image-29.png)

**Swin Transformer** 架构的总体结构如上图(a)所示，其中展示了其微型版&#x672C;**`Swin-T`**。<span style="color: rgb(100,37,208); background-color: inherit">该架构首先通过一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像块分割</span>**<span style="color: rgb(100,37,208); background-color: inherit">模块将输入的 RGB 图像划分为不重叠的图像块，类似于 ViT</span>。每个图像块被视为一个 Token，其特征设置为原始像素 RGB 值的拼接。在论文中使用&#x4E86;**`4×4`**&#x7684;图像块大小，因此每个图像块的特征维度为$$4×4×3=48$$。然后，通过一个线性嵌入层将这些原始特征投影到任意维度，记作$$C$$。

![](../../images/视觉多模态讲义（上）-image-39.png)

接着，对这些图像块 Token 应用若干经过修改自注意力计算的 Transformer 块，这些块称为 **Swin Transformer** 块。这些 Transformer 块保持 Token 的数量不变，即$$\frac{H}{4} \times \frac{W}{4}$$，与线性嵌入层一起被称&#x4E3A;**<span style="color: rgb(216,57,49); background-color: inherit">阶段1</span>**。

为了生成分层表示，随着网络深度的增加，通过**图像块合并层**减少 Token 的数量。第一个图像块合并层将每&#x7EC4;**`2×2`**&#x76F8;邻图像块的特征拼接起来，并在线性层上对这&#x4E9B;**`4C`**&#x7EF4;的拼接特征进行处理。

![Patch Merging](../../images/视觉多模态讲义（上）-image-38.png)

这使得 Token 数量减少了$$2×2=4$$倍，即分辨率下采&#x6837;**`2`**&#x500D;，输出维度被设置&#x4E3A;**`2C`**。随后，应用 **Swin Transformer** 块进行特征变换，分辨率保持在$$\frac{H}{8} \times \frac{W}{8}$$。这一阶段的图像块合并和特征变换被记&#x4F5C;**<span style="color: rgb(216,57,49); background-color: inherit">阶段2</span>**。

该过程重复两次，分别对&#x5E94;**<span style="color: rgb(216,57,49); background-color: inherit">阶段3</span>**&#x548C;**<span style="color: rgb(216,57,49); background-color: inherit">阶段4</span>**，输出分辨率分别为$$\frac{H}{16} \times \frac{W}{16}$$和$$\frac{H}{32} \times \frac{W}{32}$$。这些阶段共同生成了一个分层表示，其特征图分辨率与典型的卷积网络一致，例&#x5982;**`VGG`**&#x548C;**`ResNet`**。因此，<span style="color: rgb(100,37,208); background-color: inherit">Swin Transformer 可以方便地替代现有方法中的骨干网络，用于各种视觉任务</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">Swin Transformer 块</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>&#x20;

**Swin Transformer** 通过用基于**滑动窗口**的模块替换 Transformer 块中的标准多头自注意力 MSA 模块构建而成，其他层保持不变。如右图所示，一个 Swin Transformer 块由一个基于滑动窗口的 MSA 模块组成，后接一个两层 MLP，中间使&#x7528;**`GELU`**&#x975E;线性激活函数。在每个 **MSA** 模块和每个 **MLP&#x20;**&#x4E4B;前应用 **LN** 层，并在每个模块之后添加残差连接。



![](../../images/视觉多模态讲义（上）-image-34.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">基于滑动窗口的自注意力机制</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

标准的 Transformer 架构及其在图像分类中的改进版本都采用了全局自注意力机制，其中计算了每个 Token 与所有其他 Token 之间的关系。这种全局计算的复杂度相对于 Token 数量呈平方增长，使得它不适合许多需要大量 Token 进行密集预测或表示高分辨率图像的视觉任务。

**<span style="color: rgb(36,91,219); background-color: inherit">非重叠窗口内的自注意力机制</span>**

为了实现高效的建模，**Swin Transformer** 提出在局部窗口内计算自注意力。这些窗口被均匀地划分为非重叠区域，以覆盖整个图像。假设每个窗口包含$$M \times M$$个图像块，则对于一个大小为$$h \times w$$的图像块，全局多头自注意力 MSA 模块和基于窗口的自注意力模块的计算复杂度分别为： &#x20;

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC$$

其中前者的复杂度与图像块数量$$hw$$呈平方关系，而后者在$$M$$固定时呈线性关系，$$M$$默认设置&#x4E3A;**`7`**。对于较大的$$hw$$，全局自注意力计算通常难以承受，而基于窗口的自注意力机制具有良好的可扩展性。

**<span style="color: rgb(36,91,219); background-color: inherit">连续块中的滑动窗口划分</span>** &#x20;

基于窗口的自注意力模块缺乏跨窗口的连接，这限制了其建模能力。为了在保持非重叠窗口高效计算的同时引入跨窗口连接，**Swin Transformer&#x20;**&#x63D0;出了一种滑动窗口划分方法，在连续的 **Swin Transformer** 块中交替使用两种划分配置。

如右图所示，第一个模块采用常规的窗口划分策略，<span style="color: rgb(220,155,4); background-color: inherit">从左上角像素开始，将</span>$$8 \times 8$$<span style="color: rgb(220,155,4); background-color: inherit">的特征图均匀划分为</span>$$2 \times 2$$<span style="color: rgb(220,155,4); background-color: inherit">个大小为</span>$$4 \times 4$$<span style="color: rgb(220,155,4); background-color: inherit">的窗口（</span>$$M = 4$$<span style="color: rgb(220,155,4); background-color: inherit">）。接着，下一个模块采用一种滑动窗口配置，该配置相对于前一层的窗口划分进行了位移，具体是沿两个方向各移动</span>$$(\lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor)$$<span style="color: rgb(220,155,4); background-color: inherit">像素</span>。通过这种滑动窗口划分方法，连续的 **Swin Transformer** 块的计算如下：

![](../../images/视觉多模态讲义（上）-image-37.png)

$$\hat{z}^l = \text{W-MSA}\left(\text{LN}(z^{l-1})\right) + z^{l-1}$$

$$z^l = \text{MLP}\left(\text{LN}(\hat{z}^l)\right) + \hat{z}^l$$

$$\hat{z}^{l+1} = \text{SW-MSA}\left(\text{LN}(z^l)\right) + z^l$$

$$z^{l+1} = \text{MLP}\left(\text{LN}(\hat{z}^{l+1})\right) + \hat{z}^{l+1}$$

其中$$\hat{z}^l$$和$$z^l$$分别表示第$$l$$个块&#x4E2D;**`(S)W-MSA`**&#x6A21;块&#x548C;**`MLP`**&#x6A21;块的输出特征；**`W-MSA`**&#x548C;**`SW-MSA`**&#x5206;别表示使用常规窗口划分和滑动窗口划分配置的基于窗口的多头自注意力机制。滑动窗口划分方法在前一层中引入了相邻非重叠窗口之间的连接，结果表明它在图像分类、目标检测和语义分割任务中都非常有效。

**<span style="color: rgb(36,91,219); background-color: inherit">针对滑动配置的高效批量计算</span>** &#x20;

滑动窗口划分的一个问题是，<span style="color: rgb(216,57,49); background-color: inherit">它会导致窗口数量增加，从常规配置中的</span>$$\lceil \frac{h}{M} \rceil \times \lceil \frac{w}{M} \rceil$$<span style="color: rgb(216,57,49); background-color: inherit">增加到滑动配置中的</span>$$(\lceil \frac{h}{M} \rceil + 1) \times (\lceil \frac{w}{M} \rceil + 1)$$，并且某些窗口会小于$$M \times M$$。一个简单的解决方案是将较小的窗口填充到$$M \times M$$的大小，并在计算注意力时屏蔽掉填充的部分。然而，当常规划分的窗口数量较小时，例如$$2 \times 2$$<span style="color: rgb(220,155,4); background-color: inherit">，这种简单方法带来的计算量增加是显著的，从</span>$$2 \times 2$$<span style="color: rgb(220,155,4); background-color: inherit">增加到</span>$$3 \times 3$$<span style="color: rgb(220,155,4); background-color: inherit">，增加了</span>**`2.25`**<span style="color: rgb(220,155,4); background-color: inherit">倍</span>。 &#x20;

为了解决这一问题，**Swin Transformer&#x20;**&#x63D0;出了一种更高效的批量计算方法，<span style="color: rgb(100,37,208); background-color: inherit">通过对特征图进行向左上角方向的</span>**<span style="color: rgb(100,37,208); background-color: inherit">循环移位</span>**，如右图所示。经过这种移位后，<span style="color: rgb(216,57,49); background-color: inherit">一个批量窗口可能由特征图中不相邻的多个子窗口组成</span>。

![](../../images/视觉多模态讲义（上）-image-36.png)

![](../../images/视觉多模态讲义（上）-image-35.png)

因此<span style="color: rgb(100,37,208); background-color: inherit">需要引入一种屏蔽机制，将自注意力计算限制在每个子窗口内部</span>。通过这种循环移位，<span style="color: rgb(46,161,33); background-color: inherit">批量窗口的数量与常规窗口划分保持一致，从而保证了计算效率</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">相对位置偏置</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

在计算自注意力时，遵循之前工作的做法，<span style="color: rgb(100,37,208); background-color: inherit">在计算相似性时为每个注意力头引入一个相对位置偏置矩阵</span>$$B \in \mathbb{R}^{M^2 \times M^2}$$：

$$\text{Attention}(Q, K, V) = \text{SoftMax}\left(\frac{QK^T}{\sqrt{d}} + B\right)V$$

其中$$Q, K, V \in \mathbb{R}^{M^2 \times d}$$分别是查询、键和值矩阵；$$d$$是 Query / Key 的维度，$$M^2$$是窗口内的图像块数量。<span style="color: rgb(100,37,208); background-color: inherit">由于每个轴上的相对位置范围为</span>$$[-M+1, M-1]$$<span style="color: rgb(100,37,208); background-color: inherit">，作者参数化了一个更小的偏置矩阵</span>$$\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$$<span style="color: rgb(100,37,208); background-color: inherit">，并从中提取值填充到</span>$$B$$<span style="color: rgb(100,37,208); background-color: inherit">中</span>。 &#x20;

与没有使用该偏置项或使用绝对位置嵌入的方法相比，<span style="color: rgb(46,161,33); background-color: inherit">这种方法带来了显著的性能提升</span>。实验还证明在输入中添加绝对位置嵌入会略微降低性能，因此在实现中未采用这种方法。 &#x20;

预训练中学到的相对位置偏置可以通过双三次插值用于初始化不同窗口大小的微调模型。

![](../../images/视觉多模态讲义（上）-image-32.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">模型规格</span>**&#x20;

**Swin Transformer&#x20;**&#x6784;建了基础模型，称&#x4E3A;**`Swin-B`**，其模型规模和计算复杂度&#x4E0E;**`ViT-B`**/**`DeiT-B`**&#x76F8;似。除此之外还引入&#x4E86;**`Swin-T`**、**`Swin-S`**&#x548C;**`Swin-L`**，分别是基础模型规模和计算复杂度的&#x7EA6;**`0.25×`**、**`0.5×`**&#x548C;**`2×`**&#x7248;本。需要注意的是，**Swin-T** 和 **Swin-S** 的复杂度分别&#x4E0E;**`ResNet-50`**/**`DeiT-S`**&#x548C;**`ResNet-101`**&#x7C7B;似。默认情况下，窗口大小设置为$$M = 7$$。每个注意力头的查询维度为$$d = 32$$，每个 MLP 层的扩展系数为$$\alpha = 4$$，所有实验均保持一致。这些模型变体的架构超参数如下： &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">Swin-T</span>**：$$C = 96$$，层数 = {2, 2, 6, 2} &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Swin-S</span>**：$$C = 96$$，层数 = {2, 2, 18, 2} &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Swin-B</span>**：$$C = 128$$，层数 = {2, 2, 18, 2} &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Swin-L</span>**：$$C = 192$$，层数 = {2, 2, 18, 2} &#x20;

其中$$C$$是第一阶段隐藏层的通道数。更加详细的模型规模参数如下表：

![](../../images/视觉多模态讲义（上）-image-30.png)

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**Swin Transformer** 一经推出便在多个视觉任务中取得了突破性进展。<span style="color: rgb(46,161,33); background-color: inherit">在图像分类任务中，它在 ImageNet 数据集上的表现超越了众多经典模型；在目标检测和语义分割任务中，</span>**<span style="color: rgb(46,161,33); background-color: inherit">Swin Transformer</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 同样展现了强大的性能，成为 COCO 和 ADE20K 等数据集上的领先方法</span>。此外，其分层结构和高效的计算设计也为超分辨率图像处理提供了新的可能性。

值得一提的是，**Swin Transformer** 的成功不仅限于学术界，其<span style="color: rgb(46,161,33); background-color: inherit">开源实现和预训练权重文件也为工业界提供了极大的便利</span>。无论是研究者还是开发者，都可以基于 **Swin Transformer** 快速构建高效的视觉应用。

尽管 Swin Transformer 已经在多个领域取得了显著成果，但其潜力远未被完全挖掘。例如，<span style="color: rgb(220,155,4); background-color: inherit">如何进一步优化滑动窗口机制以适应更复杂的场景？如何将 Swin Transformer 与其他模态相结合，如文本或语音，构建多模态学习框架？</span>这些问题都为未来的研究指明了方向。

总之，**Swin Transformer&#x20;**&#x4E0D;仅是 Transformer 在视觉领域的一次成功尝试，更是深度学习发展历程中的一个重要角色。<span style="color: rgb(100,37,208); background-color: inherit">它证明了 Transformer 可以作为一种普适的视觉骨干网络，为后续的研究和应用开阔了思路</span>。对于希望深入了解现代视觉模型的读者来说，**Swin Transformer** 是一个不可忽视的经典案例。

### 1.2.4 <span style="color: rgb(36,91,219); background-color: inherit">BEiT</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">方法总览</span>**

<span style="color: rgb(100,37,208); background-color: inherit">BEiT 同时构造图像块与离散视觉 token 两种表示：图像块作为 Transformer 的输入，原始图像对应的视觉 token 作为被遮挡位置的分类目标。</span>预训练时先对图像块执行块状掩码，再根据未被遮挡的上下文恢复原始视觉 token；完成预训练后，在 Encoder 之上接入任务层，并对整套网络进行端到端微调。

![BEiT 的两条表示路径：图像块进入 Transformer，离散视觉 token 提供被遮挡位置的分类目标](../../images/视觉多模态讲义（上）-beit-pretraining-source.jpg)

* **<span style="color: rgb(36,91,219); background-color: inherit">图像表示</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像块</span>**

为了让标准 Transformer 直接接收二维图像，先把输入图像划分为互不重叠的规则图像块。设输入图像为 $$x\in\mathbb{R}^{H\times W\times C}$$，其中 $$H$$ 和 $$W$$ 分别表示图像的高和宽，$$C$$ 表示通道数；每个图像块的空间分辨率为 $$P\times P$$。图像被重排为 $$N$$ 个图像块：

$$N=\frac{HW}{P^2},\qquad x^p\in\mathbb{R}^{N\times(P^2C)}$$

每个图像块先展平为一个长度为 $$P^2C$$ 的向量，再通过线性投影转换为 Transformer 使用的 Patch Embedding。这一路径保留原始像素信息，充当 BEiT Encoder 的输入表示。在标准设置中，分辨率为 **`224×224`** 的图像被划分为 **`14×14`** 个图像块，每个图像块的大小为 **`16×16`**，因此输入序列包含 **`196`** 个图像块。

2. **<span style="color: rgb(36,91,219); background-color: inherit">离散视觉 token </span>**

第二种表示不是像素向量，而是由图像 tokenizer 产生的离散索引序列。对输入图像 $$x$$，tokenizer 输出：

$$z=[z_1,\ldots,z_N]\in\mathcal{V}^{h\times w},\qquad \mathcal{V}=\{1,\ldots,|\mathcal{V}|\}$$

视觉 token 学习采用离散变分自 Encoder。tokenizer $$q_\phi(z\mid x)$$ 根据视觉码本把图像像素映射为离散潜变量 $$z$$；decoder $$p_\psi(x\mid z)$$ 再根据这些视觉 token 重建输入图像。重建目标为：

$$\mathbb{E}_{z\sim q_\phi(z\mid x)}\left[\log p_\psi(x\mid z)\right]$$

由于潜变量是离散索引，直接采样会阻断梯度，因此训练 tokenizer 与 decoder 时使用 Gumbel-Softmax relaxation，并在 dVAE 训练中对 $$q_\phi$$ 施加均匀先验。BEiT 直接使用已经训练好的 DALL-E dVAE 图像 tokenizer，而不是在 MIM 阶段同时更新 tokenizer。

每幅图像被转换为 **`14×14`** 的视觉 token 网格， token 数量与图像块数量一一对应；视觉词表大小为 **`8192`**。这种一一对齐使第 $$i$$ 个图像块被遮挡后，可以直接把同一位置的离散视觉 token  $$z_i$$ 作为分类标签。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像 Transformer  backbone 网络</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">输入向量</span>**

&#x20;backbone 网络沿用标准 ViT 的 Transformer  Encoder 。第 $$i$$ 个展平图像块 $$x_i^p$$ 先经过线性投影矩阵 $$E$$：

$$E\in\mathbb{R}^{(P^2C)\times D},\qquad x_i^pE\in\mathbb{R}^{D}$$

序列开头加入一个可学习的特殊 token **`[S]`**，并把可学习的一维位置 Embedding 加到序列上。输入第一个 Transformer 层的表示写为：

$$H^0=[e_{[S]},x_1^pE,\ldots,x_N^pE]+E_{\mathrm{pos}}$$

这里 $$e_{[S]}$$ 是特殊 token 的 Embedding，$$E_{\mathrm{pos}}$$ 提供各图像块在展平序列中的位置信息。图像块内容与位置编码相加后共同进入 Transformer。

2. **<span style="color: rgb(36,91,219); background-color: inherit">多层 Transformer 编码</span>**

&#x20;Encoder 包含 $$L$$ 层 Transformer Block。每一层接收前一层的完整序列表示：

$$H^l=\operatorname{Transformer}(H^{l-1}),\qquad l=1,\ldots,L$$

最后一层输出为：

$$H^L=[h_{[S]}^L,h_1^L,\ldots,h_N^L]$$

其中 $$h_i^L$$ 是第 $$i$$ 个图像块经过全局上下文编码后的表示。MIM 预测头只在被遮挡位置读取相应的 $$h_i^L$$，并据此预测原始图像在该位置上的视觉 token。

* **<span style="color: rgb(36,91,219); background-color: inherit">掩码图像建模</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">构造被遮挡的图像块序列</span>**

给定图像 $$x$$，一条路径把它划分为 $$N$$ 个图像块 $$\{x_i^p\}_{i=1}^{N}$$，另一条路径把同一图像编码为 $$N$$ 个视觉 token  $$\{z_i\}_{i=1}^{N}$$。随机选取约 **`40%`** 的位置组成掩码集合：

$$\mathcal{M}\subseteq\{1,\ldots,N\},\qquad |\mathcal{M}|\approx0.4N$$

所有位于 $$\mathcal{M}$$ 中的图像块都被替换为同一个可学习的 Mask Embedding **`[M]`**，其向量维度与隐藏维度 $$D$$ 相同。损坏后的输入可以写为：

$$x^{\mathcal{M}}=\{x_i^p:i\notin\mathcal{M}\}\cup\{e_{[M]}:i\in\mathcal{M}\}$$

<span style="color: rgb(100,37,208); background-color: inherit">掩码只作用于送入 Transformer 的图像块分支；作为监督信号的视觉 token 始终由未被破坏的原始图像产生。</span>因此模型看不到被遮挡位置的像素，却仍然拥有这些位置的离散目标标签。

2. **<span style="color: rgb(36,91,219); background-color: inherit">预测被遮挡位置的视觉 token </span>**

损坏后的序列经过 $$L$$ 层 Transformer 后，在每个被遮挡位置得到上下文化表示 $$h_i^L$$。一个带 Softmax 的线性分类器把该向量映射到整个视觉词表：

$$p_{\mathrm{MIM}}(z'\mid x^{\mathcal{M}})=\operatorname{softmax}_{z'}(W_ch_i^L+b_c)$$

分类矩阵与偏置的形状分别为：

$$W_c\in\mathbb{R}^{|\mathcal{V}|\times D},\qquad b_c\in\mathbb{R}^{|\mathcal{V}|}$$

预训练目标是在训练语料 $$\mathcal{D}$$ 上，对随机掩码集合取期望，并最大化所有被遮挡位置正确视觉 token 的对数似然：

$$\max_{\theta}\sum_{x\in\mathcal{D}}\mathbb{E}_{\mathcal{M}}\left[\sum_{i\in\mathcal{M}}\log p_{\mathrm{MIM}}(z_i\mid x^{\mathcal{M}})\right]$$

实现时等价于只在 $$i\in\mathcal{M}$$ 的位置计算对 **`8192`** 类视觉 token 的交叉熵；未被遮挡位置参与上下文编码，但不计入该项预测损失。

3. **<span style="color: rgb(36,91,219); background-color: inherit">块状掩码</span>**

掩码位置不是逐块独立采样，而是通过多个矩形区域逐步扩充。初始化 $$\mathcal{M}=\varnothing$$。每次先从区间 $$[16,0.4N-|\mathcal{M}|]$$ 中采样目标面积 $$s$$，再从 $$[0.3,1/0.3]$$ 中采样宽高比 $$r$$，矩形的高和宽由下式确定：

$$a=\sqrt{sr},\qquad b=\sqrt{\frac{s}{r}}$$

![](../../images/视觉多模态讲义（上）-image-33.png)

随后在 $$h\times w$$ 的图像块网格上随机采样左上角 $$(t,l)$$，把矩形 $$[t,t+a)\times[l,l+b)$$ 覆盖的位置并入 $$\mathcal{M}$$。这一过程不断重复，直到掩码数量达到约 $$0.4N$$。单个候选矩形最少包含 **`16`** 个图像块；在 **`14×14`** 的网格上，实际最多遮挡 **`75`** 个位置。

<span style="color: rgb(216,57,49); background-color: inherit">如果直接回归被遮挡图像块的原始像素，训练目标容易被局部相关性和高频细节主导。</span>

<span style="color: rgb(46,161,33); background-color: inherit">把目标改为 dVAE 视觉 token 后，预测空间变成离散分类空间，视觉 token 同时形成信息瓶颈，把像素细节压缩为更高层的视觉抽象。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">变分自 Encoder 视角</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">条件生成的 ELBO</span>**

设 $$x$$ 为原始图像，$$\tilde{x}$$ 为被遮挡后的图像，$$z$$ 为离散视觉 token。目标可以理解为根据损坏图像恢复原始图像，即建模条件似然 $$p(x\mid\tilde{x})$$。其证据下界写为：

$$\sum_{(x_i,\tilde{x}_i)\in\mathcal{D}}\log p(x_i\mid\tilde{x}_i)\geq\sum_{(x_i,\tilde{x}_i)\in\mathcal{D}}\left(\mathbb{E}_{z_i\sim q_\phi(z\mid x_i)}[\log p_\psi(x_i\mid z_i)]-D_{\mathrm{KL}}[q_\phi(z\mid x_i)\Vert p_\theta(z\mid\tilde{x}_i)]\right)$$

这个下界中，$$q_\phi(z\mid x)$$ 是把原始图像转换为视觉 token 的 tokenizer；$$p_\psi(x\mid z)$$ 是从视觉 token 重建原始图像的 decoder；$$p_\theta(z\mid\tilde{x})$$ 则根据被遮挡图像恢复视觉 token ，对应 BEiT 的 MIM 模型。

2. **<span style="color: rgb(36,91,219); background-color: inherit">第一阶段：学习视觉 tokenizer</span>**

第一阶段训练离散变分自 Encoder，在均匀先验下最小化重建损失：

$$-\mathbb{E}_{z_i\sim q_\phi(z\mid x_i)}[\log p_\psi(x_i\mid z_i)]$$

完成后得到固定的 tokenizer $$q_\phi$$ 和 decoder $$p_\psi$$。这一阶段解决的是如何把连续像素变成离散视觉 token，以及如何验证这些 token 仍保留足够的图像信息。

3. **<span style="color: rgb(36,91,219); background-color: inherit">第二阶段：学习条件先验</span>**

第二阶段保持 $$q_\phi$$ 与 $$p_\psi$$ 不变，只更新根据被遮挡图像预测视觉 token 的 $$p_\theta$$。tokenizer 的分布被简化为集中在最大概率 token 上的单点分布：

$$\hat{z}_i=\arg\max_z q_\phi(z\mid x_i)$$

在这一近似下，两阶段目标可以写成视觉 token 重建项与掩码图像建模项之和：

$$\sum_{(x_i,\tilde{x}_i)\in\mathcal{D}}\left(\underbrace{\mathbb{E}_{z_i\sim q_\phi(z\mid x_i)}[\log p_\psi(x_i\mid z_i)]}_{\text{Stage 1: Visual Token Reconstruction}}+\underbrace{\log p_\theta(\hat{z}_i\mid\tilde{x}_i)}_{\text{Stage 2: Masked Image Modeling}}\right)$$

<span style="color: rgb(100,37,208); background-color: inherit">因此，BEiT 的 MIM 训练不是重新学习图像重建器，而是在固定离散视觉表示的前提下，学习一个从被遮挡图像到原始视觉 token 的条件先验。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练设置</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">网络配置</span>**

backbone 网络采用 ViT-Base 配置：Transformer 深度为 **`12`** 层，隐藏维度为 **`768`**，每层包含 **`12`** 个注意力头，前馈网络的中间维度为 **`3072`**。输入图像块大小固定为 **`16×16`**，视觉词表大小为 **`8192`**。

2. **<span style="color: rgb(36,91,219); background-color: inherit">数据与增强</span>**

预训练数据使用 ImageNet-1K 训练集，约包含 **`1.2M`** 幅图像，但不使用类别标签。数据增强包括随机尺度裁剪、水平翻转和颜色抖动。输入分辨率为 **`224×224`**，对应 **`14×14`** 个图像块和相同数量的视觉 token；每幅图像最多遮挡 **`75`** 个图像块，约占全部位置的 **`40%`**。

3. **<span style="color: rgb(36,91,219); background-color: inherit">优化设置</span>**

预训练共运行约 **`500k`** 个更新步骤，相当于 **`800`** 个 epoch，全局 batch size 为 **`2k`**。优化器使用 Adam，参数为 $$\beta_1=0.9$$、$$\beta_2=0.999$$；基础学习率为$$1.5×10^{-3}$$，先进行 **`10`** 个 epoch 的 warmup，再使用余弦学习率衰减。weight decay 为 **`0.05`**，stochastic depth rate 为 **`0.1`**，普通 dropout 关闭。

这套配置在 **`16`** 张 Nvidia Tesla V100 32GB GPU 上训练约 **`5`** 天。

4. **<span style="color: rgb(36,91,219); background-color: inherit">参数初始化与深层残差缩放</span>**

所有参数先在较小区间内随机初始化，例如 **`[-0.02,0.02]`**。随后对第 $$l$$ 个 Transformer 层的两类输出矩阵进行缩放：一类是 Self-Attention 子层最后的线性投影，另一类是前馈网络子层最后的线性投影。缩放系数为$$\sqrt{\frac{1}{2l}}$$，层数越深，残差分支末端投影的初始幅度越小，用于稳定大规模 Transformer 预训练。

* **<span style="color: rgb(36,91,219); background-color: inherit">下游视觉任务微调</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像分类</span>**

图像分类只在预训练 Encoder 之后增加一个线性分类器。BEiT 不直接使用特殊 token **`[S]`** 作为全局表示，而是对全部图像块的最后一层表示进行平均池化，再输入 Softmax 分类器：

$$p(y\mid x)=\operatorname{softmax}\left(\operatorname{avg}(\{h_i^L\}_{i=1}^{N})W_c\right)$$

其中 $$W_c\in\mathbb{R}^{D\times C}$$，$$C$$ 是类别数。训练时同时更新 BEiT Encoder 和线性分类器，通过最大化有标签数据的类别似然完成端到端微调。

2. **<span style="color: rgb(36,91,219); background-color: inherit">语义分割</span>**

语义分割使用预训练 BEiT 作为 backbone Encoder，并接入 SETR-PUP 的任务层。Decoder 由若干反卷积层组成，把图像块级表示逐步上采样到像素级分割结果。 Encoder 与 Decoder 共同进行端到端微调。

3. **<span style="color: rgb(36,91,219); background-color: inherit">中间微调</span>**

在自监督预训练与目标任务微调之间，还可以加入一次有监督的中间微调。完整顺序是：先用 MIM 学习 BEiT Encoder，再在数据量较大的 ImageNet-1K 上使用类别标签训练，最后把得到的参数继续微调到目标视觉任务。

### 1.2.5 <span style="color: rgb(36,91,219); background-color: inherit">BEiT v2</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">方法总览</span>**

<span style="color: rgb(100,37,208); background-color: inherit">BEiT v2 保留 BEiT 的掩码图像建模框架，但把监督目标从偏向低层图像细节的离散 Token，替换为经过语义知识蒸馏得到的视觉 Token。</span>训练分成两个阶段：先通过 VQ-KD 把 Teacher 的连续语义特征压缩到离散 Codebook，再冻结视觉 Tokenizer，用这些语义 Token 监督 BEiT v2 Backbone 恢复被遮挡的 image Patch。Patch Aggregation 额外训练 \[CLS] Token，使 Patch 级重建目标同时形成可用于图像级任务的全局表示。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像表示</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Image Patch 序列</span>**

BEiT v2 使用 ViT 作为 Backbone。设输入图像为 $$x\in\mathbb{R}^{H\times W\times C}$$，其中 $$H$$、$$W$$ 和 $$C$$ 分别表示图像高度、宽度和通道数；每个 image Patch 的空间大小为 $$P\times P$$。图像被重排为 $$N$$ 个互不重叠的 image Patch：

$$N=\frac{HW}{P^2},\qquad x^p\in\mathbb{R}^{N\times(P^2C)}$$

标准输入分辨率为 **`224×224`**，Patch Size 为 **`16×16`**，因此得到 **`14×14`** 的 Patch 网格，共 **`196`** 个位置。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Patch Embedding</span>**

每个 image Patch 展平为长度 $$P^2C$$ 的向量，再经过线性投影转换为 Transformer 的输入 Embedding。Backbone 为全部 Patch 产生一一对应的上下文化表示$$\{h_i\}_{i=1}^{N}$$，这组表示既是视觉 Tokenizer 执行向量量化的输入，也是 BEiT v2 MIM Head 预测被遮挡视觉 Token 的依据。

* **<span style="color: rgb(36,91,219); background-color: inherit">VQ-KD 视觉 Tokenizer</span>**

![VQ-KD 将 Teacher 的连续语义特征压缩为离散视觉 Token，并通过 Straight-Through Gradient 训练 Tokenizer Encoder](../../images/视觉多模态讲义（上）-beit-v2-vqkd.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">Tokenizer 的输出空间</span>**

视觉 Tokenizer 把图像映射为离散 Token 序列：

$$z=[z_1,z_2,\ldots,z_N]\in\mathcal{V}^{(H/P)\times(W/P)}$$

视觉词表就是可学习的 Codebook。设 Codebook 包含 $$K$$ 个、每个维度为 $$D$$ 的 Embedding：

$$\mathcal{V}\in\mathbb{R}^{K\times D},\qquad \mathcal{V}=\{v_1,v_2,\ldots,v_K\}$$

Tokenizer 由 ViT Encoder 和向量量化器组成。Encoder 先把输入图像转换为 Patch 表示 $$h_i$$，量化器再为每个位置选择一个 Codebook 索引 $$z_i$$。

2. **<span style="color: rgb(36,91,219); background-color: inherit">最近邻量化</span>**

Encoder 输出和 Codebook Embedding 都先进行 L2 归一化，然后按欧氏距离查找最近的 Code。第 $$i$$ 个 image Patch 的离散索引为：

$$z_i=\arg\min_{j\in\{1,\ldots,K\}}\left\|\ell_2(h_i)-\ell_2(v_j)\right\|_2$$

归一化后的向量模长相同，因此最小化该距离等价于最大化余弦相似度。量化结果不再保留连续特征值，只保留最接近的 Codebook 索引。

3. **<span style="color: rgb(36,91,219); background-color: inherit">语义重建</span>**

量化完成后，Transformer Decoder 接收归一化的 Codebook Embedding 序列：

$$\{\ell_2(v_{z_i})\}_{i=1}^{N}$$

Decoder 为每个位置输出 $$o_i$$。监督信号不是原始像素，而是 Teacher 在同一 Patch 位置产生的语义特征 $$t_i$$；默认 Teacher 为 OpenAI CLIP-B/16。训练直接最大化 $$o_i$$ 与 $$t_i$$ 的余弦相似度。

<span style="color: rgb(216,57,49); background-color: inherit">像素重建会把大量建模能力消耗在颜色、纹理和局部高频细节上，这些信息不一定对应下游识别需要的高层语义。</span>

<span style="color: rgb(46,161,33); background-color: inherit">VQ-KD 改为重建 Teacher 的 Patch 级语义特征，使每个离散 Token 更接近一个压缩后的语义概念，而不是像素模板。</span>

4. **<span style="color: rgb(36,91,219); background-color: inherit">Straight-Through Gradient</span>**

最近邻索引是离散操作，无法直接求导。反向传播时，Decoder 输入端的梯度被直接复制到 Encoder 输出端，绕过最近邻查找；前向过程仍使用真实的离散 Code。这样既保持离散 Token，又能更新 Tokenizer Encoder。

5. **<span style="color: rgb(36,91,219); background-color: inherit">VQ-KD 训练目标</span>**

设 $$\operatorname{sg}[\cdot]$$ 为 stop-gradient：前向传播保持恒等，反向传播梯度为零。VQ-KD 的完整目标为：

$$\max\sum_{x\in\mathcal{D}}\sum_{i=1}^{N}\left(\cos(o_i,t_i)-\left\|\operatorname{sg}[\ell_2(h_i)]-\ell_2(v_{z_i})\right\|_2^2-\left\|\ell_2(h_i)-\operatorname{sg}[\ell_2(v_{z_i})]\right\|_2^2\right)$$

第一项让 Decoder 输出逼近 Teacher 语义特征；第二项只更新被选中的 Codebook Embedding，使它靠近当前 Encoder 输出；第三项只更新 Encoder，使其输出靠近选中的 Codebook Embedding。两个 stop-gradient 把 Codebook 更新与 Encoder 的 commitment 约束分开。

* **<span style="color: rgb(36,91,219); background-color: inherit">Codebook 利用率</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Codebook Collapse</span>**

<span style="color: rgb(216,57,49); background-color: inherit">向量量化训练容易出现 Codebook Collapse：大量输入持续落到少数 Code 上，其余 Code 几乎不被使用。</span>这会缩小离散语义空间的有效容量，也会让不同视觉概念共享过于粗糙的目标。

2. **<span style="color: rgb(36,91,219); background-color: inherit">低维归一化查找</span>**

最近邻查找只在 **`32`** 维的归一化 Codebook 空间中完成。低维 Codebook Embedding 在送入 Transformer Decoder 之前，再映射回较高维的隐藏空间。

<span style="color: rgb(46,161,33); background-color: inherit">低维查找降低了最近邻分配的难度，L2 归一化又消除了向量模长差异，两者共同提高 Codebook 的有效使用率。</span>

3. **<span style="color: rgb(36,91,219); background-color: inherit">EMA 更新</span>**

Codebook Embedding 使用指数移动平均更新，而不是完全依赖普通梯度更新。EMA 根据分配到每个 Code 的 Encoder 输出逐步移动 Codebook 中心，使 VQ-KD 训练更稳定。

* **<span style="color: rgb(36,91,219); background-color: inherit">BEiT v2 掩码图像建模</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">块状掩码与损坏输入</span>**

预训练沿用 BEiT 的块状掩码。每幅图像约有 **`40%`** 的 image Patch 被选中，通常约为 **`75`** 个位置，掩码集合记为 $$\mathcal{M}$$。这些位置共享同一个可学习的 Mask Embedding **`[M]`**：

$$x_i^{\mathcal{M}}=\delta(i\in\mathcal{M})\odot e_{[M]}+\left(1-\delta(i\in\mathcal{M})\right)\odot x_i^p$$

其中 $$\delta(\cdot)$$ 是指示函数。序列开头再加入可学习的 **`[CLS]`** Token，完整输入为：

$$[e_{\mathrm{CLS}},\{x_i^{\mathcal{M}}\}_{i=1}^{N}]$$

2. **<span style="color: rgb(36,91,219); background-color: inherit">MIM Head</span>**

Vision Transformer 输出 $$\{h_i\}_{i=0}^{N}$$，其中 $$h_0$$ 对应 \[CLS] Token。MIM Head 是一个全连接分类层，只在被遮挡位置预测视觉 Token：

$$p(z_i\mid h_i)=\operatorname{softmax}_{z_i}(W_ch_i+b_c),\qquad i\in\mathcal{M}$$

目标 Token $$z_i$$ 由已经训练完成并冻结的 VQ-KD Tokenizer 从原始图像中产生。视觉 Token 与 image Patch 的位置数量相同，因此监督可以逐位置对齐。

3. **<span style="color: rgb(36,91,219); background-color: inherit">MIM 损失</span>**

损失只在被遮挡位置计算，对正确视觉 Token 的负对数似然求和：

$$\mathcal{L}_{\mathrm{MIM}}=-\sum_{x\in\mathcal{D}}\sum_{i\in\mathcal{M}}\log p(z_i\mid x_i^{\mathcal{M}})$$

<span style="color: rgb(100,37,208); background-color: inherit">Backbone 必须利用未被遮挡 Patch 的上下文推断缺失位置对应的语义 Token，而不是直接复制输入像素。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Patch Aggregation</span>**

![Patch Aggregation 把最终层 \[CLS\] Token 与中间层 Patch Token 组合，通过第二个 MIM 目标训练全局表示](../../images/视觉多模态讲义（上）-image-31.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">Patch 目标与全局表示的差距</span>**

<span style="color: rgb(216,57,49); background-color: inherit">标准 MIM 把每个被遮挡 Patch 的恢复放在首位，[CLS] Token 本身没有直接训练目标，因此 Patch 级预训练与图像级表示之间存在差距。</span>Patch Aggregation 通过一个只在预训练阶段存在的信息瓶颈，迫使最终层 \[CLS] Token 汇总全局信息。

2. **<span style="color: rgb(36,91,219); background-color: inherit">聚合序列</span>**

设 Backbone 共包含 $$L$$ 层，第 $$l$$ 层的 Patch Token 为 $$\{h_i^l\}_{i=1}^{N}$$，最终层 \[CLS] Token 为 $$h_{\mathrm{CLS}}^L$$。Patch Aggregation 把两者拼成新的序列：

$$S=[h_{\mathrm{CLS}}^L,h_1^l,h_2^l,\ldots,h_N^l]$$

中间层 Patch Token 提供局部内容，但从第 $$l+1$$ 层到第 $$L$$ 层的信息只能通过最终 \[CLS] Token 进入第二条预测路径，因而形成显式的信息瓶颈。

3. **<span style="color: rgb(36,91,219); background-color: inherit">浅层 Transformer Decoder</span>**

序列 $$S$$ 输入一个浅层 Transformer Decoder，默认深度为 **`2`** 层，并再次预测被遮挡位置的视觉 Token：

$$p(z\mid S)=\operatorname{softmax}_{z}(W_cS+b_c)$$

Patch Aggregation 分支与 Backbone 最后一层的原始 MIM Head 共享分类参数 $$W_c$$ 和 $$b_c$$，第二项损失仍只在掩码位置计算。

4. **<span style="color: rgb(36,91,219); background-color: inherit">总损失与训练后移除</span>**

最终训练目标是两条 MIM 损失之和：

$$\mathcal{L}=\mathcal{L}_{\mathrm{MIM}}+\mathcal{L}_{\mathrm{MIM}}^{c}$$

<span style="color: rgb(100,37,208); background-color: inherit">额外分支让后半段 Backbone 更愿意把跨 Patch 的全局信息压入最终 [CLS] Token。</span>浅层 Transformer Decoder 只用于预训练，完成训练后被丢弃，不增加下游推理开销。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练设置</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">VQ-KD 架构</span>**

| **<span style="color: rgb(36,91,219); background-color: inherit">配置项</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">设置</span>**                       |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Tokenizer Encoder                                                             | ViT-B/16，**`12`** 层，隐藏维度 **`768`**，FFN 维度 **`3072`**，**`12`** 个 Attention Head，每个 Head 维度 **`64`** |
| Transformer Decoder                                                           | 实验使用 **`1`** 层或 **`3`** 层，默认设置采用 **`3`** 层；隐藏维度和 Attention Head 数与 Tokenizer Encoder 相同            |
| Teacher                                                                       | OpenAI CLIP-B/16                                                                                   |
| Codebook                                                                      | **`8192×32`**                                                                                      |
| 输入                                                                            | 分辨率 **`224×224`**，Patch Size **`16×16`**                                                           |

2. **<span style="color: rgb(36,91,219); background-color: inherit">VQ-KD 优化设置</span>**

VQ-KD 在 ImageNet-1K 上训练 `100` 个 epoch，batch size 为 `512`。优化器使用 Adam，β 为 `(0.9, 0.99)`；峰值学习率为 `2e-4`，最低学习率为 `1e-5`，先 warmup `5` 个 epoch，再执行余弦衰减。weight decay 为 `1e-4`，关闭 gradient clipping、dropout 和 stochastic depth，数据增强使用 RandomResizeAndCrop。

3. **<span style="color: rgb(36,91,219); background-color: inherit">BEiT v2 Backbone 配置</span>**

| **<span style="color: rgb(36,91,219); background-color: inherit">配置项</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">ViT-B/16</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">ViT-L/16</span>** |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Transformer 层数                                                                | **`12`**                                                                           | **`24`**                                                                           |
| 隐藏维度                                                                          | **`768`**                                                                          | **`1024`**                                                                         |
| FFN 中间维度                                                                      | **`3072`**                                                                         | **`4096`**                                                                         |
| Attention Head                                                                | **`12`**                                                                           | **`16`**                                                                           |
| Layer Scale                                                                   | **`0.1`**                                                                          | **`1e-5`**                                                                         |
| Patch Aggregation 取层                                                          | 第 **`9`** 层                                                                        | 第 **`21`** 层                                                                       |

两种模型都使用 **`16×16`** Patch Size、Relative Position Embedding 和共享的 Relative Position Embedding。VQ-KD Tokenizer 和 Teacher 始终使用 base-size 配置，即使目标 Backbone 是 ViT-L/16。

4. **<span style="color: rgb(36,91,219); background-color: inherit">BEiT v2 预训练优化设置</span>**

自监督数据为不使用标签的 ImageNet-1K，输入分辨率为 `224×224`。训练周期为 `300` 或 `1600` 个 epoch，batch size 为 `2048`。Adam 的 β₁ 为 `0.9`；β₂ 在 `300` epoch 设置下为 `0.98`，在长训练中为 `0.999`。峰值学习率为 `1.5e-3`，最低学习率为 `1e-5`，warmup `10` 个 epoch 后使用余弦衰减。gradient clipping 为 `3.0`，weight decay 为 `0.05`，关闭 dropout；drop path 在短训练中为 `0`，长训练中为 `0.1`，color jitter 为 `0.4`。

* **<span style="color: rgb(36,91,219); background-color: inherit">下游任务适配</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像分类微调</span>**

图像分类在预训练 Backbone 上接入分类 Head，并端到端更新全部参数。ViT-B/16 和 ViT-L/16 的峰值学习率都为 `5e-4`，batch size 为 `1024`；微调周期分别为 `100` 和 `50` 个 epoch，warmup 分别为 `20` 和 `5` 个 epoch，layer-wise learning rate decay 分别为 `0.65` 和 `0.8`。

优化器使用 Adam，ε 为 `1e-8`，β 为 `(0.9, 0.999)`；最低学习率为 `1e-6`，学习率按余弦衰减。weight decay 为 `0.05`，label smoothing 为 `0.1`，stochastic depth 对 base/large 分别为 `0.1` 和 `0.2`。数据增强包含 RandAugment `9/0.5`、Mixup `0.8`、CutMix `1.0` 和 random erasing `0.25`。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Linear Probing</span>**

Linear Probing 冻结整个 Backbone，只训练顶部的线性分类 Head。启用 Patch Aggregation 时使用 \[CLS] Token 作为全局图像表示；不启用 Patch Aggregation 的基线模型则对所有 Patch Token 做平均池化。

3. **<span style="color: rgb(36,91,219); background-color: inherit">语义分割微调</span>**

语义分割在 ADE20K 上接入 UperNet，并对 Backbone 与任务 Head 进行端到端微调。输入分辨率为 `512×512`，训练 `160K` 个 step，batch size 为 `16`。候选峰值学习率为 `{0.5, 0.8, 1.0}e-4`，layer-wise learning rate decay 为 `{0.75, 0.8, 0.85}`，warmup 为 `1500` 个 step，随后使用线性学习率衰减到 `0`。

Adam 的 ε 为 `1e-8`，β 为 `(0.9, 0.999)`，weight decay 为 `0.05`；ViT-B/16 与 ViT-L/16 的 stochastic depth 分别为 `0.1` 和 `0.2`。微调使用 Relative Position Embedding，但不共享 Relative Position Embedding。

---

[Previous](01-视觉基础--CNN-骨干网络.md) | [Contents](../../README.md) | [Next](03-视觉基础--对比学习.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=2)
