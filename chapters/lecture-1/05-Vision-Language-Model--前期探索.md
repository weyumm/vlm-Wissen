[Previous](04-视觉基础--其他经典模型.md) | [Contents](../../README.md) | [Next](06-Vision-Language-Model--BLIP-家族.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=5)

# 2. <span style="color: rgb(36,91,219); background-color: inherit">Vision-Language Model </span>

## 2.1 <span style="color: rgb(36,91,219); background-color: inherit">前期探索</span>

### 2.1.1 <span style="color: rgb(36,91,219); background-color: inherit">CLIP</span>

在人工智能领域，视觉与语言的结合一直是研究的热点。传统的计算机视觉模型往往专注于单一模态的任务，例如图像分类、目标检测等，而自然语言处理则主要集中在文本生成、语义理解等领域。然而，人类的认知能力天生是多模态的：我们可以通过阅读一段文字想象出相应的画面，也可以通过观察一幅画联想到相关的描述。这种跨模态的理解能力正是多模态学习的核心目标之一。

**<span style="color: rgb(216,57,49); background-color: inherit">CLIP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">C</span>**<span style="color: rgb(216,57,49); background-color: inherit">ontrastive </span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">anguage-</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">re-training）</span>模型，由 OpenAI 于 2021 年提出，是一个划时代的多模态预训练模型。它通过将图像和文本映射到同一个嵌入空间，实现了视觉与语言之间的无缝连接。**<span style="color: rgb(100,37,208); background-color: inherit">CLIP</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的核心思想是利用互联网上大量的弱监督图文对数据，即图片及其对应的描述性文本，通过对比学习的方式训练一个联合的图像-文本嵌入模型</span>。这一设计<span style="color: rgb(46,161,33); background-color: inherit">不仅突破了传统视觉模型对标注数据的依赖，还为下游任务的 Zero-Shot迁移提供了强大的支持</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">自然语言监督</span>**

**CLIP&#x20;**&#x7684;核心思想是利用自然语言中包含的监督信息来学习感知能力。其实<span style="color: rgb(100,37,208); background-color: inherit">这一思想并不新颖，但在这一领域的研究中，使用的术语多种多样，甚至看似矛盾，研究者的动机也各不相同</span>。一些研究提出了从与图像配对的文本中学习视觉表示的方法，但分别将其描述为无监督、自监督、弱监督和有监督的方法。

需要强调的是，这些工作的共同点并不在于具体方法的细节，而在于它们都将自然语言视为一种训练信号。所有这些方法都在从自然语言监督中学习。尽管早期研究在使用主题模型&#x548C;**`n-gram`**&#x8868;示时曾因自然语言的复杂性而遇到困难，但<span style="color: rgb(216,57,49); background-color: inherit">深度上下文表示学习的进步表明，现在已具备有效利用这一丰富监督来源的工具</span>。

与其他训练方法相比，自然语言学习具有若干潜在优势。<span style="color: rgb(100,37,208); background-color: inherit">扩展自然语言监督比扩展标准的众包标注更容易，因为后者需要以经典的</span>**<span style="color: rgb(100,37,208); background-color: inherit">机器学习兼容格式</span>**<span style="color: rgb(100,37,208); background-color: inherit">进行标注</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">典型的</span>**`1-of-N `**`多数投票`<span style="color: rgb(220,155,4); background-color: inherit">的</span>**`gold label`**。相比之下，基于自然语言的方法可以从互联网上大量文本中被动地学习其中包含的监督信息。此外，自然语言学习相较于大多数无监督或自监督学习方法还有一个重要优势：<span style="color: rgb(46,161,33); background-color: inherit">它不仅学习到一种表示，还将这种表示与语言连接起来，从而实现灵活的 Zero-Shot 迁移能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据集创建</span>**

现有的研究主要使用了三个数据集：**`MS-COCO`**、**`Visual Genome`**&#x548C;**`YFCC100M`**。尽管 **MS-COCO** 和 **Visual Genome** 是高质量的众包标注数据集，但按照现代标准来看，它们的<span style="color: rgb(216,57,49); background-color: inherit">规模较小，每个数据集仅有大约</span>**`100k`**<span style="color: rgb(216,57,49); background-color: inherit">张训练照片</span>。相比之下，其他计算机视觉系统可以使用多&#x8FBE;**`3.5B`**&#x5F20; Instagram 照片进行训练。**YFCC100M&#x20;**&#x6570;据集虽然包&#x542B;**`100M`**&#x5F20;照片，是一个可能的替代方案，但<span style="color: rgb(216,57,49); background-color: inherit">每张图像的元数据稀疏且质量参差不齐。许多图像使用自动生成的文件名作为标题，或包含相机曝光设置的描述</span>。经过筛选，仅保留具有自然语言标题和/或英文描述的图像后，数据集规模缩小了六倍，仅&#x5269;**`15M`**&#x5F20;照片，这与 ImageNet 的规模大致相同。

自然语言监督的一个主要动机是互联网上公开可用的大量此类形式的数据。由于现有数据集未能充分反映这种可能性，仅基于这些数据集的结果会低估这一研究方向的潜力。为了解决这个问题，CLIP 构建了一个新的数据集，包含从互联网上各种公开来源收集&#x7684;**`400M`**&#x4E2A;**`(Image，Text)`**&#x5BF9;。为了尽可能覆盖广泛的视觉概念，在构建过程中搜索那些文本包&#x542B;**`500k`**&#x4E2A;查询词之一&#x7684;**`(Image，Text)`**&#x5BF9;。CLIP 通过为每个查询词最多包含 20,000 &#x4E2A;**`(Image，Text)`**&#x5BF9;来近似实现类别平衡。最终得到的数据集在总词数上与用于训练 GPT-2 的 WebText 数据集相当。我们将这个数据集称为 **<span style="color: rgb(216,57,49); background-color: inherit">WIT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">W</span>**<span style="color: rgb(216,57,49); background-color: inherit">eb</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage</span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ext）</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型选择</span>**

最先进的计算机视觉系统需要使用大量的计算资源。考虑到这些系统通常仅针对 1000 个 ImageNet 类别进行训练，从自然语言中学习开放的视觉概念的任务显得尤为艰巨。但是 **<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">发现训练效率是成功扩展自然语言监督的关键，因此基于这一指标选择了最终的预训练方法</span>。

最初的方法类似&#x4E8E;**`VirTex`**，即<span style="color: rgb(100,37,208); background-color: inherit">从零开始联合训练一个图像卷积神经网络和文本 Transformer，以预测图像的标题</span>。然而，**CLIP&#x20;**&#x5728;扩展这种方法时遇到了困难，如右图所示，<span style="color: rgb(216,57,49); background-color: inherit">一个拥有</span>**`63M`**<span style="color: rgb(216,57,49); background-color: inherit">个参数的 Transformer 语言模型，其计算量已经是其 ResNet-50 图像编码器的两倍，但其识别 ImageNet 类别的速度却比一个更简单的基线方法慢三倍</span>，而该基线方法仅预测相同文本的词袋编码。



![](../../images/视觉多模态讲义（上）-image-85.png)

这两种方法有一个关键的相似点：它们都试图预测与每张图像相关联的文本的确切词汇。<span style="color: rgb(216,57,49); background-color: inherit">由于图像伴随的描述、评论和相关文本种类繁多，这是一项非常困难的任务</span>。最近，在图像对比表示学习领域的研究表明，相比等效的预测目标，对比目标可以学习到更好的表示。其他研究也发现，<span style="color: rgb(216,57,49); background-color: inherit">尽管生成式图像模型可以学习高质量的图像表示，但它们所需的计算量比具有相同性能的对比模型高出一个数量级</span>。基于这些发现，**<span style="color: rgb(100,37,208); background-color: inherit">CLIP</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 探索了一种潜在更简单的代理任务，即仅预测哪些文本整体与哪些图像配对，而不是预测文本的确切词汇</span>。从相同的词袋编码基线出发，**<span style="color: rgb(46,161,33); background-color: inherit">CLIP </span>**<span style="color: rgb(46,161,33); background-color: inherit">在上图中将预测目标替换为对比目标，并观察到在 Zero-Shot 迁移到 ImageNet 的速率上进一步提高了 4 倍</span>。

给定一批包&#x542B;**`N`**&#x4E2A;**`(Image，Text)`**&#x5BF9;的数据，**CLIP&#x20;**&#x88AB;训练来预测这批数据中实际发生&#x7684;**`N × N`**&#x79CD;可能&#x7684;**`(Image，Text)`**&#x914D;对中的正确组合。为此，**<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">学习一个多模态嵌入空间，通过联合训练图像编码器和文本编码器，最大化批次中</span>**`N`**<span style="color: rgb(100,37,208); background-color: inherit">对真实配对的图像和文本嵌入之间的余弦相似度，同时最小化</span>**`N² − N`**<span style="color: rgb(100,37,208); background-color: inherit">个错误配对的嵌入之间的余弦相似度</span>。**CLIP&#x20;**&#x5BF9;这些相似度分数优化了一个对称交叉熵损失。右图包含了 **CLIP&#x20;**&#x6838;心实现的伪代码。这种批量构建技术和目标最早在深度度量学习领域被提出，称&#x4E3A;**<span style="color: rgb(216,57,49); background-color: inherit">多类 N-pair 损失</span>**，随后被推广用于对比表示学习，称&#x4E3A;**`InfoNCE`**&#x635F;失，并最近被改编用于医学影像领域的对&#x6BD4;**`(Image，Text)`**&#x8868;示学习。



![](../../images/视觉多模态讲义（上）-image-90.png)

由于预训练数据集规模庞大，过拟合并不是一个主要问题，因此 **CLIP&#x20;**&#x7684;训练细节得到了简化：<span style="color: rgb(100,37,208); background-color: inherit">从零开始训练 CLIP，既没有用 ImageNet 权重初始化图像编码器，也没有用预训练权重初始化文本编码器。</span>**<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">没有使用表示与对比嵌入空间之间的非线性投影，而是仅使用线性投影将每个编码器的表示映射到多模态嵌入空间</span>。**CLIP&#x20;**&#x63A8;测非线性投影可能仅在当前自监督表示学习方法中与图像相关的细节共同适应。此外，**<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">移除了之前工作实现中的文本转换函数</span>$$t_u$$，该函数从文本中均匀采样单个句子，因为 **CLIP&#x20;**&#x9884;训练数据集中的许&#x591A;**`(Image，Text)`**&#x5BF9;仅包含单个句子。**<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">还简化了图像转换函数</span>$$t_v$$。随机方形裁剪是从调整大小后的图像中提取的唯一数据增强方式。最后，控制 softmax 中 logits 范围的温度参数$$τ$$在训练过程中直接作为对数参数化的乘法标量进行优化，以避免将其作为一个超参数进行调整。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型选择与缩放</span>**

**CLIP&#x20;**&#x8003;虑了两种不同的图像编码器架构：

> * 对于第一种架构，选&#x62E9;**`ResNet-50`**&#x4F5C;为图像编码器的基础架构，因为它被广泛采用且性能优异。**CLIP&#x20;**&#x5728;原始版本的基础上进行了几项改进：<span style="color: rgb(100,37,208); background-color: inherit">使用了凯明提出的 ResNet-D 改进方法，以及抗锯齿 rect-2 模糊池化技术</span>。此外，<span style="color: rgb(100,37,208); background-color: inherit">将全局平均池化层替换为注意力池化机制。这种注意力池化通过一个 Transformer 风格的多头 QKV 注意力层实现，其中 query 基于图像的全局平均池化表示进行条件化</span>。
>
> * 对于第二种架构，尝试了最近引&#x5165;**`ViT`**。**CLIP&#x20;**&#x57FA;本遵循其原始实现，仅做了少量修改：<span style="color: rgb(100,37,208); background-color: inherit">在 Transformer 之前对组合的 patch 和位置嵌入添加了一个额外的层归一化，并采用了稍微不同的初始化方案</span>。

![](../../images/视觉多模态讲义（上）-image-87.png)

文本编码器是一个 Transformer，基础模型是一个包&#x542B;**`63M`**&#x4E2A;参数&#x7684;**`12`**&#x5C42;、宽度&#x4E3A;**`512`**&#x7684;模型，具&#x6709;**`8`**&#x4E2A;注意力头。该 Transformer 使用小写字节对编&#x7801;**`BPE`**&#x8868;示，词汇表大小&#x4E3A;**`49,152`**。为了提高计算效率，最大序列长度被限制&#x4E3A;**`76`**。文本序列&#x4EE5;**`[SOS]`**&#x548C;**`[EOS]`**&#x6807;记始末，Transformer 最高层&#x5728;**`[EOS]`**&#x6807;记处的激活被视为文本的特征表示，经过层归一化后线性投影到多模态嵌入空间中。文本编码器中使用了掩码自注意力机制，以保留通过预训练语言模型初始化或添加语言建模作为辅助目标的能力，但对此的进一步探索留作未来工作。

过去计算机视觉研究通常通过单独增加模型的宽度或深度来扩展模型规模。<span style="color: rgb(100,37,208); background-color: inherit">对于 ResNet 图像编码器，</span>**<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">采用了</span>**`EfficientNet`**<span style="color: rgb(100,37,208); background-color: inherit">的思想，因为他们发现将额外的计算资源分配到宽度、深度和分辨率的所有维度上，比仅分配到单一维度上效果更好</span>。虽然 EfficientNet 在架构中调整了每个维度的计算资源分配比例，但是 **<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">采用了一种简单的基线方法：将额外的计算资源平均分配到宽度、深度和分辨率的扩展上</span>。对于文本编码器，仅<span style="color: rgb(100,37,208); background-color: inherit">按比例增加模型的宽度，使其与 ResNet 宽度的增加量成正比，而不改变深度</span>，因为实验表明 **CLIP&#x20;**&#x7684;性能对文本编码器容量的变化不太敏感。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

CLIP 训练了一系&#x5217;**`5`**&#x4E2A; ResNet &#x548C;**`3`**&#x4E2A; ViT：

> * **<span style="color: rgb(36,91,219); background-color: inherit">对于 ResNet</span>**，**CLIP&#x20;**&#x8BAD;练了一个 **`ResNet-50`**、一个 **`ResNet-101`**，以及另外三个遵循 EfficientNet 风格模型扩展的变体，分别使用了大&#x7EA6;**`4`**&#x500D;、**`16`**&#x500D;&#x548C;**`64`**&#x500D;于 ResNet-50 的计算量。它们分别被标记为 **`RN50x4`**、**`RN50x16`**&#x548C;**`RN50x64`**。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">对于 ViT</span>**，CLIP 训练&#x4E86;**`ViT-B/32`**、**`ViT-B/16`**&#x548C;**`ViT-L/14`**。

所有模型都训练&#x4E86;**`32`**&#x4E2A; epoch，使用 Adam 优化器，并对所有非增益或偏置的权重应用解耦权重衰减正则化，学习率通过余弦调度进行衰减。<span style="color: rgb(100,37,208); background-color: inherit">初始超参数是通过对基线 ResNet-50 模型在单个 epoch 训练中的网格搜索、随机搜索和手动调整组合设置的</span>。由于计算资源限制，<span style="color: rgb(100,37,208); background-color: inherit">针对更大的模型，超参数通过启发式方法进行了调整</span>。可学习的温度参数$$τ$$初始化为等效&#x4E8E;**`0.07`**&#x7684;值，并对其进行裁剪以防止 logits 的缩放超&#x8FC7;**`100`**，这可以防止训练不稳定。

CLIP 使用了非常大的批量大小，&#x4E3A;**`32,768`**。<span style="color: rgb(100,37,208); background-color: inherit">为了加速训练并节省内存，采用了混合精度训练技术。为了进一步节省内存，还使用了梯度检查点、半精度 Adam 统计信息以及半精度随机舍入的文本编码器权重</span>。嵌入相似度的计算也被分片处理，每个 GPU 仅计算其局部嵌入批次所需的部分成对相似度。<span style="color: rgb(100,37,208); background-color: inherit">最大的 ResNet 模型</span>**`RN50x64`**<span style="color: rgb(100,37,208); background-color: inherit">在</span>**`592`**<span style="color: rgb(100,37,208); background-color: inherit">个 V100 GPU 上训练了</span>**`18`**<span style="color: rgb(100,37,208); background-color: inherit">天，而最大的</span>**`ViT`**<span style="color: rgb(100,37,208); background-color: inherit">模型在</span>**`256`**<span style="color: rgb(100,37,208); background-color: inherit">个 V100 GPU 上训练了</span>**`12`**<span style="color: rgb(100,37,208); background-color: inherit">天</span>。对&#x4E8E;**`ViT-L/14`**，CLIP 还以更高的 336 像素分辨率额外训练了一个 epoch，以类似于 FixRes 的方式提升性能，将此模型标记&#x4E3A;**`ViT-L/14@336px`**。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**<span style="color: rgb(46,161,33); background-color: inherit">优点</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">强大的 Zero-Shot 迁移能力</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">在未见过的任务上表现出色，能够直接应用于多种下游任务而无需额外微调</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">在图像分类任务中，用户只需提供类别名称，</span>**<span style="color: rgb(220,155,4); background-color: inherit">CLIP </span>**<span style="color: rgb(220,155,4); background-color: inherit">即可通过计算图像与文本嵌入的相似度完成分类</span>。这种灵活性使其成为一种通用的多模态工具，适用于从图像检索到视频分析的广泛场景。

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态学习能力</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">CLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">能够同时处理图像和文本两种模态的数据</span>，捕捉图像中的物体特征，如<span style="color: rgb(220,155,4); background-color: inherit">形状、颜色、空间关系</span>，以及文本中的语义信息。这种多模态的学习方式使得 **CLIP&#x20;**&#x5728;图文匹配、图像生成等任务中展现出卓越性能。

* **<span style="color: rgb(36,91,219); background-color: inherit">灵活和通用性</span>**

由于 **CLIP&#x20;**&#x76F4;接从自然语言中学习广泛的视觉概念，其<span style="color: rgb(100,37,208); background-color: inherit">泛化能力远超传统的 ImageNet 预训练模型</span>。无论是细粒度分类还是开放式任务，**CLIP&#x20;**&#x90FD;能轻松应对。此外，**CLIP&#x20;**&#x53EF;以跨越多个视觉和语言任务，表现出极高的适应性。

* **<span style="color: rgb(36,91,219); background-color: inherit">性能好</span>**

**CLIP&#x20;**&#x4ECE;互联网上采集了超&#x8FC7;**`400M`**&#x4E2A;图文对进行预训练，这种大规模的弱监督学习方式不仅降低了标注成本，还使模型接触到更加多样化的内容。这<span style="color: rgb(46,161,33); background-color: inherit">为 </span>**<span style="color: rgb(46,161,33); background-color: inherit">CLIP </span>**<span style="color: rgb(46,161,33); background-color: inherit">提供了强大的泛化能力和鲁棒性</span>。

**<span style="color: rgb(216,57,49); background-color: inherit">缺点</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">细粒度分类能力不足</span>**

尽管 **CLIP&#x20;**&#x5728;许多任务上表现出色，但在需要高精度区分的任务中，如<span style="color: rgb(220,155,4); background-color: inherit">区分汽车型号、花卉种类或飞机变体，其表现相对较弱</span>。这表明 **CLIP&#x20;**&#x5728;处理复杂或专业领域时可能存在局限性。

* **<span style="color: rgb(36,91,219); background-color: inherit">对未见类别的敏感性</span>**

**CLIP&#x20;**&#x5728;面对未出现在训练数据中的类别时，可能无法输出正确的结果。例如，<span style="color: rgb(220,155,4); background-color: inherit">如果训练集中只有动物图片，而在推理阶段输入一张汽车图片，模型的表现会显著下降</span>。这种<span style="color: rgb(216,57,49); background-color: inherit">局限性反映了 </span>**<span style="color: rgb(216,57,49); background-color: inherit">CLIP </span>**<span style="color: rgb(216,57,49); background-color: inherit">对训练数据分布的高度依赖</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉缺陷</span>**

研究表明，CLIP存在一些视觉理解上的短板，例如<span style="color: rgb(220,155,4); background-color: inherit">难以准确区分方向、数量、颜色和结构等基本视觉属性</span>。<span style="color: rgb(216,57,49); background-color: inherit">这些缺陷限制了其在某些应用场景中的可靠性</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">效率问题</span>**

虽然 CLIP 在 Zero-Shot 和 Few-Shot 任务上表现出色，但其<span style="color: rgb(216,57,49); background-color: inherit">推理效率仍是一个挑战</span>。尤其是在处理大规模数据集或实时任务时，双塔架构的计算开销可能成为一个瓶颈。



总体而言，**<span style="color: rgb(46,161,33); background-color: inherit">CLIP </span>**<span style="color: rgb(46,161,33); background-color: inherit">作为一种创新性的多模态模型，凭借其零样本迁移能力、多模态学习能力和大规模数据驱动的优势，在学术界和工业界都产生了深远的影响</span>。然而，其<span style="color: rgb(216,57,49); background-color: inherit">在细粒度分类、未见类别处理以及视觉缺陷等方面的不足也提醒我们，</span>**<span style="color: rgb(216,57,49); background-color: inherit">CLIP </span>**<span style="color: rgb(216,57,49); background-color: inherit">并非万能的解决方案</span>。未来的研究方向可能包括优化对比学习框架、提升细粒度分类能力，以及探索更多模态的融合方法，如<span style="color: rgb(220,155,4); background-color: inherit">音频、3D数据</span>。CLIP 的成功不仅展示了大规模预训练模型的潜力，也为多模态人工智能的发展奠定了坚实的基础。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

### 2.1.2 <span style="color: rgb(36,91,219); background-color: inherit">ViLT</span>

在多模态学习领域，视觉和语言的融合一直是研究的核心问题之一。传统的多模态模型往往依赖于复杂的视觉特征提取器，如目标检测模型，以及独立的语言处理模块，这不仅增加了计算开销，还限制了模型的灵活性。随着 Transformer 架构的广泛应用，一种全新的多模态融合方式应运而生——**<span style="color: rgb(216,57,49); background-color: inherit">ViLT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Vi</span>**<span style="color: rgb(216,57,49); background-color: inherit">sion-and-</span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">anguage </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformer）</span>，它以其极简的设计理念和高效的性能表现，成为了多模态领域不可忽视的工作。

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

预训练和微调的方案已经扩展到视觉和语言联合的多模态，催生了视觉与语言预训练 VLP 模型的类别。这些模型<span style="color: rgb(100,37,208); background-color: inherit">通过图像文本匹配和掩码语言建模目标在图像及其对齐的描述上进行预训练，并在涉及双模态输入的视觉-语言下游任务上进行微调</span>。

为了被输入到 VLP 模型中，图像像素需要与语言标记一起以密集的形式进行嵌入。以往深度卷积网络一直被认为是这一视觉嵌入步骤的核心。大多数 VLP 模型采用了一种&#x5728;**`Visual Genome`**&#x6570;据集上预训练的检测器，该数据集标注&#x4E86;**`1600`**&#x4E2A;对象类别&#x548C;**`400`**&#x4E2A;属性类别。

迄今为止，大多数 VLP 研究都集中在通过增强视觉嵌入器的能力来提升性能。然而，在学术实验中，<span style="color: rgb(216,57,49); background-color: inherit">由于区域特征通常在训练时提前缓存以减轻特征提取的负担，因此拥有一个庞大的视觉嵌入器的缺点往往被忽视</span>。但在实际应用中，这种局限性仍然显而易见，因为 Query 必须经历一个缓慢的提取过程。为此，**ViLT&#x20;**&#x5C06;注意力转向轻量级且快速的视觉输入嵌入方法。由&#x4E8E;**`ViT`**&#x4EE5;&#x53CA;**`DeiT`**&#x5DE5;作的出现证明了，使用简单的线性投影对图像块进行嵌入足以将像素输入到 Transformer 中。因此<span style="color: rgb(100,37,208); background-color: inherit"> </span>**<span style="color: rgb(100,37,208); background-color: inherit">ViLT </span>**<span style="color: rgb(100,37,208); background-color: inherit">认为 VLP 模型中用于模态交互的 Transformer 模块也可以像处理文本特征一样，直接处理视觉特征，而无需依赖卷积视觉嵌入器</span>。

**ViLT&#x20;**&#x662F;一种以统一方式处理两种模态的模型。<span style="color: rgb(46,161,33); background-color: inherit">它与以往的 VLP 模型的主要区别在于其浅层、无卷积的像素级输入嵌入方法。移除专门为视觉输入设计的深层嵌入器显著减少了模型的大小和运行时间</span>。如右图，**<span style="color: rgb(100,37,208); background-color: inherit">ViLT </span>**<span style="color: rgb(100,37,208); background-color: inherit">比使用区域特征的 VLP 模型快几十倍，比使用网格特征的模型至少快四倍</span>，同时在视觉-语言下游任务上表现出相似甚至更好的性能。

![](../../images/视觉多模态讲义（上）-image-89.png)

**<span style="color: rgb(36,91,219); background-color: inherit">一句话总结</span>**：**<span style="color: rgb(100,37,208); background-color: inherit">将目标检测从视觉端拿掉</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">VLP 模型分类</span>**

VLP 模型可以根据两个关键点对视觉与语言模型进行分类：

> 1. 两种模态在专用参数和/或计算资源上的表达能力是否均衡
>
> 2. 两种模态是否在深度网络中进行交互

这两个维度的组合衍生出了下图中的四种典型架构：

![](../../images/视觉多模态讲义（上）-image-88.png)

1. 视觉语义嵌入 **VSE** 模型，&#x5982;**`VSE++`**&#x548C;**`SCAN`**，属&#x4E8E;**`a`**&#x7C7B;别。它们<span style="color: rgb(100,37,208); background-color: inherit">分别使用独立的嵌入器处理图像和文本，其中图像嵌入器的复杂度远高于文本嵌入器</span>。随后，这些模型<span style="color: rgb(100,37,208); background-color: inherit">通过简单的点积或浅层注意力机制来表示两种模态嵌入特征之间的相似性</span>。

2. **`CLIP`**&#x6A21;型属&#x4E8E;**`b`**&#x7C7B;别，因为<span style="color: rgb(100,37,208); background-color: inherit">它为每种模态使用了独立但同样复杂的 Transformer 嵌入器。虽然图像向量和文本向量的交互仍然较浅，只使用了点积</span>，但是 **CLIP&#x20;**&#x5728;图像到文本检索任务中表现出卓越的零样本性能。然而，<span style="color: rgb(216,57,49); background-color: inherit">在其他视觉-语言下游任务上，其表现并不理想</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">在</span>**`NLVR2`**<span style="color: rgb(220,155,4); background-color: inherit">数据集上微调 </span>**<span style="color: rgb(220,155,4); background-color: inherit">CLIP </span>**<span style="color: rgb(220,155,4); background-color: inherit">的 MLP 头部时，将池化后的视觉和文本向量的点积作为多模态表示，仅能达到</span>**`50.99 ± 0.38`**<span style="color: rgb(220,155,4); background-color: inherit">的开发集准确率</span>。由于随机水平准确率只有0.5，因此这种表示无法胜任该任务。这一结果也印证了所有简单融合多模态表示的模型都无法有效学习 NLVR2 的结论。

   > **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：这些结果表明即使来自高性能单模态嵌入器的输出，其简单融合也可能不足以学习复杂的视觉-语言任务，这进一步凸显了更严格跨模态交互方案的重要性。

3. 近年来的 VLP 模型大多属&#x4E8E;**`c`**&#x7C7B;别，与浅层交互模型不同，它们<span style="color: rgb(100,37,208); background-color: inherit">利用深度 Transformer 建模图像和文本特征的交互</span>。然而，除了交互模块外，<span style="color: rgb(100,37,208); background-color: inherit">卷积神经网络仍被用于提取和嵌入图像特征</span>，之前说过这占据了大部分计算资源。

4. **ViLT&#x20;**&#x662F;首个属&#x4E8E;**`d`**&#x7C7B;别的模型，其原始像素的嵌入层与文本标记一样浅层且计算高效。这种架构将大部分计算集中在模态交互建模上。

**<span style="color: rgb(36,91,219); background-color: inherit">模态交互模式</span>**

VLP 模型的核心在于 Transformer。它们接收视觉和文本嵌入序列作为输入，通过各层建模跨模态交互，有时还包括模态内的交互，并最终输出上下文化特征序列。

领域内的研究学者一般将交互模式分为两类：

> 1. 单流方法：&#x5982;**`VisualBERT`**、**`UNITER`**，其中各层共同操作图像和文本输入的拼接序列
>
> 2. 双流方法：&#x5982;**`ViLBERT`**、**`LXMERT`**，其中两种模态在输入层并未拼接

**ViLT&#x20;**&#x9009;择单流方法作为交互 Transformer 模块的基础，因为<span style="color: rgb(100,37,208); background-color: inherit">双流方法会引入额外参数</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">视觉嵌入模式</span>**

尽管所有高性能 VLP 模型共享相同的文本嵌入器，即来自预训练 BERT 的分词器及其类似 BERT 的词嵌入和位置嵌入，但它们在视觉嵌入器上存在显著差异。然而，在大多数情况下，<span style="color: rgb(100,37,208); background-color: inherit">视觉嵌入仍是现有 VLP 模型的瓶颈。ViLT 通过引入图像块投影来简化这一过程，而非依赖区域或网格特征，因为这些特征通常需要复杂的提取模块</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">区域特征</span>**

![](../../images/视觉多模态讲义（上）-image-86.png)

VLP 模型主要使用区域特征，也称为自下而上特征，这些特征通常由现成的检测器生成，&#x5982;**`Faster R-CNN`**。

生成区域特征的一般流程如下：

> 1. 区域提议网络 RPN 基于从 CNN 主干提取的网格特征提出感兴趣区域 RoI
>
> 2. 非极大值抑制 NMS 将 RoI 数量减少到数千个
>
> 3. 通过 RoI Align 等操作池化后，RoI 经过 RoI 头部处理并转化为区域特征
>
> 4. 对每个类别再次应用NMS，最终将特征数量减少到数百个

上述过程涉及多个影响性能和运行时间的因素：<span style="color: rgb(100,37,208); background-color: inherit">主干网络、NMS 风格、RoI 头部。以往的研究对这些因素的控制较为宽松，导致不同工作做出不同选择</span>。 &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">主干网络</span>**：**`ResNet-101`**&#x548C;**`ResNeXt-152`**&#x662F;两种常用主干网络。&#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">NMS</span>**：NMS 通常按类别进行。当类别数量庞大时，如 VG 数据集中&#x7684;**`1.6K`**&#x4E2A;类别，逐类别应用 NMS 会成为主要运行时间瓶颈。新的一些工作引入的类别无关 NMS 主要在解决这一问题。 &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">RoI头部</span>**：最初使用的&#x662F;**`C4`**&#x5934;部，后来引入&#x4E86;**`FPN-MLP`**&#x5934;部。由于 RoI 头部针对每个 RoI 操作，因此带来了显著的运行负担。 &#x20;

尽管目标检测器本身较为轻量，但其速度通常仍慢于主干网络或单层卷积。冻结视觉主干并在训练前缓存区域特征仅能在训练阶段节省时间，而在推理阶段无效，更不用说这可能限制性能。

2. **<span style="color: rgb(36,91,219); background-color: inherit">网格特征</span>**

除了检测器头部，<span style="color: rgb(100,37,208); background-color: inherit">卷积神经网络的输出特征网格也可用作视觉-语言预训练的视觉特征。直接使用网格特征最早由专用于 VQA 任务的模型提出</span>，主要是为了避免耗时的区域选择操作。

> **<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：
>
> * **`X-LXMERT`**&#x91CD;新审视了网格特征，将区域提议固定为网格而非来自区域提议网络的区域。然而，其特征缓存排除了对主干网络的进一步微调。
>
> * **`Pixel-BERT`**&#x662F;唯一一个用 ImageNet 分类任务预训练的ResNet变体主干替代 VG 预训练对象检测器的 VLP 模型。与基于区域特征的 VLP 模型中冻结的检测器不同，Pixel-BERT 的主干在视觉-语言预训练期间可调。尽管 Pixel-BERT 的下游性能低于基于区域特征的 VLP 模型，但使用更重&#x7684;**`ResNeXt-152`**&#x65F6;，其性能与其他竞争者相当。

然而，**ViLT&#x20;**&#x8BA4;为网格特征并非最佳选择，因为深度 CNN 仍然计算量很大，占整体计算的很大一部分。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像块投影</span>**

为了最小化开销，**<span style="color: rgb(100,37,208); background-color: inherit">ViLT </span>**<span style="color: rgb(100,37,208); background-color: inherit">采用了最简单的视觉嵌入方案：对图像块进行线性投影。图像块投影嵌入最早由 ViT 引入用于图像分类任务</span>。图像块投影极大地简化了视觉嵌入步骤，使其与文本嵌入一样简单。 &#x20;

具体来说，**ViLT&#x20;**&#x4F7F;&#x7528;**`32 × 32`**&#x7684;图像块投影，仅&#x9700;**`2.4M`**&#x53C2;数。这与复杂&#x7684;**`ResNe(X)t`**&#x4E3B;干和检测组件形成鲜明对比。此外，其运行时间可以忽略不计。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

如下图所示，**ViLT&#x20;**&#x4F5C;为一种 VLP 模型，具有简洁的架构，采用最小化的视觉嵌入 pipeline，并遵循单流方法。 &#x20;

与之前工作的不同之处在于，<span style="color: rgb(46,161,33); background-color: inherit">交互 Transformer 权重从预训练的 ViT 初始化，而非 BERT。这种初始化方式利用了交互层处理视觉特征的能力，同时避免了独立的深层视觉嵌入器</span>。

![](../../images/视觉多模态讲义（上）-image-84.png)

公式化描述如下：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">文本嵌入</span>**：$$\bar{t} = [t_{\text{class}}; t_1^T; \cdots; t_L^T] + T_{\text{pos}}$$
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">图像嵌入</span>**：$$\bar{v} = [v_{\text{class}}; v_1^V; \cdots; v_N^V] + V_{\text{pos}}$$
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">模态类型嵌入</span>**：$$z^0 = [\bar{t} + t_{\text{type}}; \bar{v} + v_{\text{type}}]$$
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">Transformer 层更新</span>**：
>
> $$\hat{z}^d = \text{MSA}(\text{LN}(z^{d-1})) + z^{d-1}, \quad d = 1 \ldots D$$
>
> $$z^d = \text{MLP}(\text{LN}(\hat{z}^d)) + \hat{z}^d, \quad d = 1 \ldots D$$
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">多模态池化表示</span>**：$$p = \tanh(z_0^D W_{\text{pool}})$$

ViT由堆叠的模块组成，每个模块包括一个多头自注意力 MSA 层和一个 MLP 层。ViT 与 BERT 的主要区别在于层归一化 LN 的位置：<span style="color: rgb(100,37,208); background-color: inherit">在 BERT 中，LN 位于 MSA 和 MLP 之后，即</span>**<span style="color: rgb(100,37,208); background-color: inherit">后归一化</span>**<span style="color: rgb(100,37,208); background-color: inherit">，而在 ViT 中则位于之前，即</span>**<span style="color: rgb(100,37,208); background-color: inherit">前归一化</span>**。输入文本$$t \in \mathbb{R}^{L \times |V|}$$通过词嵌入矩阵$$T \in \mathbb{R}^{|V| \times H}$$和位置嵌入矩阵$$T_{\text{pos}} \in \mathbb{R}^{(L+1) \times H}$$嵌入为$$\bar{t} \in \mathbb{R}^{L \times H}$$。 &#x20;

输入图像$$I \in \mathbb{R}^{C \times H \times W}$$被分割成图像块并展平为$$v \in \mathbb{R}^{N \times (P^2 \cdot C)}$$，其中$$(P, P)$$是图像块分辨率，$$N = HW/P^2$$。随后通过线性投影$$V \in \mathbb{R}^{(P^2 \cdot C) \times H}$$和位置嵌入$$V_{\text{pos}} \in \mathbb{R}^{(N+1) \times H}$$，将$$v$$嵌入为$$\bar{v} \in \mathbb{R}^{N \times H}$$。 &#x20;

文本和图像嵌入分别与其对应的模态类型嵌入向量$$t_{\text{type}}, v_{\text{type}} \in \mathbb{R}^H$$相加，然后拼接为组合序列$$z^0$$。通过$$D$$层 Transformer 迭代更新上下文化向量$$z$$，最终得到上下文化序列$$z^D$$。多模态输入的池化表示$$p$$通过对$$z^D$$的第一个索引应用线性投影$$W_{\text{pool}} \in \mathbb{R}^{H \times H}$$和双曲正切函数获得。 &#x20;

**ViLT&#x20;**&#x4F7F;用从 ImageNet 预训练&#x7684;**`ViT-B/32`**&#x6743;重，因此命名&#x4E3A;**`ViLT-B/32`**。隐藏层大小$$H=768$$，层数$$D=12$$，图像块大小$$P=32$$，MLP 大小为 3072，注意力头数为 12。

**<span style="color: rgb(36,91,219); background-color: inherit">预训练目标</span>**

**ViLT&#x20;**&#x4F7F;用两个常用于训练 VLP 模型的目标来训练 **ViLT**：<span style="color: rgb(100,37,208); background-color: inherit">图像文本匹配</span>**`ITM`**<span style="color: rgb(100,37,208); background-color: inherit">和掩码语言建模</span>**`MLM`**。

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像文本匹配 ITM</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>

**ViLT&#x20;**&#x4EE5;**`0.5`**&#x7684;概率随机用另一张不相关的图像替换对齐的图像。一个单层线性 **ITM&#x20;**&#x5934;部将池化输出特征$$p$$投影到二分类的 logits 上，并计算负对数似然损失作为 **ITM&#x20;**&#x635F;失。 &#x20;

此外，**ViLT&#x20;**&#x8BBE;计了词-图像块对&#x9F50;**`WPA`**，使用近似近端点&#x6CD5;**`IPOT`**&#x8BA1;算$$z^D$$的两个子集之间的对齐分数：文本子集 $$z^D|t$$和视觉子集$$z^D|v$$。**ViLT&#x20;**&#x6309;照经验值选择 **IPOT&#x20;**&#x7684;超参数$$beta = 0.5, N = 50$$，并将近似 Wasserstein 距离乘&#x4EE5;**`0.1`**&#x52A0;入 ITM 损失。

* **<span style="color: rgb(36,91,219); background-color: inherit">掩码语言建模 MLM</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>

该目标是通过上下文化向量$$z^D|t$$预测被掩码的文本标记$$t_{\text{masked}}$$的真实标签。根据经验，&#x4EE5;**`0.15`**&#x7684;概率随机掩码文本$$t$$。 &#x20;

**ViLT&#x20;**&#x4F7F;用一个两层 MLP MLM 头部，输入$$z^D|t$$并输出词汇表上的 logits，类似于BERT的MLM目标。然后计算被掩码标记的负对数似然损失作为 MLM 损失。

**<span style="color: rgb(36,91,219); background-color: inherit">全词掩码</span>**

全词掩码是一种掩码技术，它会掩码构成整个单词的所有连续子词标记。研究表明，当应用于原始BERT和中文BERT时，这种方法在下游任务中非常有效。 &#x20;

**ViLT&#x20;**&#x8BA4;为全词掩码对于 VLP 尤为重要，因为它可以充分利用来自其他模态的信息。例如，<span style="color: rgb(220,155,4); background-color: inherit">单词“</span>**`giraffe`**<span style="color: rgb(220,155,4); background-color: inherit">”会被预训练的 bert-base-uncased 分词器分解为三个子词标记 [&quot;</span>**`gi`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;, &quot;</span>**`##raf`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;, &quot;</span>**`##fe`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;]。如果未完全掩码这些标记，比如 [&quot;</span>**`gi`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;, &quot;</span>**`[MASK]`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;, &quot;</span>**`##fe`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;]，模型可能会仅依赖附近的两个语言标记 [&quot;</span>**`gi`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;, &quot;</span>**`##fe"] 来预测被掩码的 "##raf`**<span style="color: rgb(220,155,4); background-color: inherit">&quot;，而不是利用来自图像的信息</span>。 &#x20;

**ViLT&#x20;**&#x5728;预训练期间&#x4EE5;**`0.15`**&#x7684;概率对整个单词进行掩码。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**ViLT** 以其极简的设计理念、高效的性能表现和广泛的应用前景，重新定义了视觉与语言融合的可能性。通过摒弃复杂的视觉特征提取器、采用单塔结构和共享 Transformer 编码器，ViLT 不仅大幅降低了模型的计算开销，还开创了一种全新的多模态学习范式。

但是 **ViLT** 仍存在一些缺点：

> * 性能不够好，比不过范式3，<span style="color: rgb(100,37,208); background-color: inherit">应该需要更强的视觉部分，且视觉模型应该要比文本模型更大</span>
>
> * Visual Embedding 是随机初始化，效果很差
>
> * 推理快但训练很慢

尽管仍存在一些改进空间，但 **ViLT** 无疑为未来的多模态研究奠定了坚实的基础。**ViLT** 的成功不仅仅在于其技术上的突破，更在于它为多模态学习领域提供了一个全新的视角：<span style="color: rgb(100,37,208); background-color: inherit">简单的设计未必意味着性能的妥协，反而可能带来意想不到的高效与灵活</span>。

---

[Previous](04-视觉基础--其他经典模型.md) | [Contents](../../README.md) | [Next](06-Vision-Language-Model--BLIP-家族.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=5)
