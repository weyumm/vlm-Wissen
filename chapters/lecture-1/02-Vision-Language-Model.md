[Previous](01-视觉基础.md) | [Contents](../../README.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=2)

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

## 2.2 <span style="color: rgb(36,91,219); background-color: inherit">BLIP 家族</span>

### 2.2.1 <span style="color: rgb(36,91,219); background-color: inherit">ALBEF</span>

在多模态学习领域，如何有效地对齐和融合图像与文本信息一直是研究的核心问题。传统的视觉-语言预训练模型 **<span style="color: rgb(216,57,49); background-color: inherit">VLP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ision-and-</span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">anguage </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">re-training）</span>通常依赖于一个预训练的图像检测器来提取图像特征，并通过多模态编码器将图像和文本特征进行联合建模。然而，这种方法存在两个主要问题：

> 一是对<span style="color: rgb(216,57,49); background-color: inherit">图像检测器的依赖增加了计算复杂度</span>
>
> 二是<span style="color: rgb(216,57,49); background-color: inherit">从互联网上爬取的大规模图文对数据通常包含噪声</span>，导致模型难以有效学习高质量的跨模态表示

为了解决这些问题，**<span style="color: rgb(216,57,49); background-color: inherit">ALBEF</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">AL</span>**<span style="color: rgb(216,57,49); background-color: inherit">ign </span>**<span style="color: rgb(216,57,49); background-color: inherit">BE</span>**<span style="color: rgb(216,57,49); background-color: inherit">fore </span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">use）</span>应运而生。**<span style="color: rgb(100,37,208); background-color: inherit">ALBEF </span>**<span style="color: rgb(100,37,208); background-color: inherit">提出了一种全新的训练框架，通过</span>**<span style="color: rgb(100,37,208); background-color: inherit">先对齐再融合</span>**<span style="color: rgb(100,37,208); background-color: inherit">的策略，显著提升了多模态模型的性能</span>。此外，**ALBEF&#x20;**&#x5F15;入了动量蒸馏 **<span style="color: rgb(216,57,49); background-color: inherit">MoD</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Mo</span>**<span style="color: rgb(216,57,49); background-color: inherit">mentum </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">istillation）</span>技术，使得模型能够更好地利用带噪声的数据集，从而在大规模弱监督数据上取得突破性进展。

**ALBEF** 总结&#x4E86;**`ViLT`**&#x548C;**`CLIP`**：

> * 更大的视觉编码器，要比文本编码器大，因&#x4E3A;**`ViLT`**&#x4E0D;如范&#x5F0F;**`3`**&#x7684;结果，并且模态交互网络要尽可能大
>
> * **`CLIP`**&#x4E2D;&#x7684;**`ITC`<span style="color: rgb(216,57,49); background-color: inherit"> </span>**&#x4C;oss &#x548C;**`ViLT`**&#x4E2D;&#x7684;**`ITM`**、**`MLM`<span style="color: rgb(216,57,49); background-color: inherit"> </span>**&#x4C;oss 的学习任务比较好

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：ALBEF 和 ViLT 都舍弃目标检测器提特征。后者只是因为推理慢，前者认为预训练好的目标检测器抽的特征与文本空间不对齐，没有进行 end to end 的训练，对多模态编码器有挑战
>
> 解决：用对比学习，即 CLIP Loss，在 Fuse 之前就能 Align。这也是本文的**核心贡献**

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

![](../../images/视觉多模态讲义（上）-image-83.png)

如上图所示，**<span style="color: rgb(100,37,208); background-color: inherit">ALBEF </span>**<span style="color: rgb(100,37,208); background-color: inherit">包含一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像编码器</span>**<span style="color: rgb(100,37,208); background-color: inherit">、一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本编码器</span>**<span style="color: rgb(100,37,208); background-color: inherit">和一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">多模态编码器</span>**。**<span style="color: rgb(100,37,208); background-color: inherit">图像编码器</span>**<span style="color: rgb(100,37,208); background-color: inherit">使用</span>**`12`**<span style="color: rgb(100,37,208); background-color: inherit">层的视觉Transformer </span>**`ViT-B/16`**<span style="color: rgb(100,37,208); background-color: inherit">作为图像编码器，并用在 ImageNet-1k 上预训练的 DEiT 权重进行初始化</span>。输入图像$$I$$被编码为一系列嵌入向量：$$\{v_\text{cls}, v_1, \cdots, v_N\}$$，其中$$v_\text{cls}$$&#x662F;**`[CLS]`**&#x6807;记的嵌入向量。**<span style="color: rgb(100,37,208); background-color: inherit">文本编码器</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">多模态编码器</span>**<span style="color: rgb(100,37,208); background-color: inherit">均采用</span>**`6`**<span style="color: rgb(100,37,208); background-color: inherit">层的 Transformer 结构</span>。**文本编码器**通过$$\text{BERT}_\text{base}$$模型的&#x524D;**`6`**&#x5C42;进行初始化，而**多模态编码器**则通过$$\text{BERT}_\text{base}$$的&#x540E;**`6`**&#x5C42;进行初始化。文本编码器将输入文本T转换为嵌入序列$$\{w_\text{cls}, w_1, \cdots, w_N\}$$，然后将其传递到多模态编码器中。在<span style="color: rgb(100,37,208); background-color: inherit">多模态编码器的每一层中，图像特征通过交叉注意力机制与文本特征进行融合</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：一共是两份参数，Momentum Model 有一份，通过左边训练的参数通过 moving average 得到，这&#x4E0E;**`MoCo`**&#x76F8;同。其中加权参数设置很&#x5927;**`0.995`**，保证 Momentum Model 参数更新慢，产生特征稳定，<span style="color: rgb(100,37,208); background-color: inherit">一可以得到更稳定的 negative sample，二可以做 Momentum Distillation</span>

**<span style="color: rgb(36,91,219); background-color: inherit">目标函数</span>**

**ALBEF&#x20;**&#x901A;过三个目标对进行预训练：<span style="color: rgb(100,37,208); background-color: inherit">单模态编码器上的图文对比学习</span>**`ITC`**、<span style="color: rgb(100,37,208); background-color: inherit">多模态编码器上的掩码语言建模</span>**`MLM`**&#x548C;<span style="color: rgb(100,37,208); background-color: inherit">图文匹配</span>**`ITM`**。并且<span style="color: rgb(100,37,208); background-color: inherit">通过在线对比硬负样本挖掘改进了 </span>**<span style="color: rgb(100,37,208); background-color: inherit">ITM</span>**。

1. **<span style="color: rgb(36,91,219); background-color: inherit">图文对比学习 ITC</span>**

**ITC** 旨在融合之前学习到更好的单模态表示。它学习一个相似度函数$$s = g_v(v_{\text{cls}})^\top g_w(w_{\text{cls}})$$，使得平行的图文对具有更高的相似度得分。$$g_v$$和$$  g_w  $$是线性变换，&#x5C06;**`[CLS]`**&#x5D4C;入映射到归一化的低维表示，一般&#x4E3A;**`256`**&#x7EF4;。&#x53D7;**`MoCo`**&#x542F;发，**<span style="color: rgb(100,37,208); background-color: inherit">ALBEF </span>**<span style="color: rgb(100,37,208); background-color: inherit">维护两个队列以存储来自动量单模态编码器的最近</span>**`M`**<span style="color: rgb(100,37,208); background-color: inherit">个图文表示</span>。动量编码器生成的归一化特征记为 $$  g_v'(v_{\text{cls}}')  $$ 和$$g_w'(w_{\text{cls}}')$$。这里定义：

$$s(I, T) = g_v(v_{\text{cls}})^\top g_w'(w_{\text{cls}}'), \quad s(T, I) = g_w(w_{\text{cls}})^\top g_v'(v_{\text{cls}}')$$

对于每个图像和文本，<span style="color: rgb(100,37,208); background-color: inherit">计算 softmax 归一化的</span>**<span style="color: rgb(100,37,208); background-color: inherit">图像到文本</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本到图像</span>**<span style="color: rgb(100,37,208); background-color: inherit">的相似度</span>：

$$p_m^{\text{i2t}}(I) = \frac{\exp(s(I, T_m)/\tau)}{\sum_{m=1}^M \exp(s(I, T_m)/\tau)}, \quad p_m^{\text{t2i}}(T) = \frac{\exp(s(T, I_m)/\tau)}{\sum_{m=1}^M \exp(s(T, I_m)/\tau)}$$

其中$$\tau$$是可学习的温度参数。设$$  y^{\text{i2t}}(I)  $$和$$  y^{\text{t2i}}(T)  $$表示真实的 one-hot 相似度分布，其中负样本的概率&#x4E3A;**`0`**，正样本的概率&#x4E3A;**`1`**。图文对比损失定义为$$  p  $$和$$  y  $$之间的交叉熵$$H$$：

$$\mathcal{L}_{\text{itc}} = \frac{1}{2} \mathbb{E}_{(I,T)\sim D} \left[ H(y^{\text{i2t}}(I), p^{\text{i2t}}(I)) + H(y^{\text{t2i}}(T), p^{\text{t2i}}(T)) \right]$$

* **<span style="color: rgb(36,91,219); background-color: inherit">掩码语言建模 MLM</span>**

**MLM&#x20;**&#x5229;用图像和上下文文本预测被遮掩的单词。这里&#x4EE5;**`15%`**&#x7684;概率随机遮掩输入标记，并将其替换为特殊 Token **`[MASK]`**。设$$\hat{T}$$表示被遮掩的文本，$$p_{\text{msk}}(I, \hat{T})$$表示模型对遮掩标记的预测概率。**MLM&#x20;**&#x6700;小化交叉熵损失：

$$\mathcal{L}_{\text{mlm}} = \mathbb{E}_{(I,\hat{T})\sim D} H(y_{\text{msk}}, p_{\text{msk}}(I, \hat{T}))$$

其中$$  y_{\text{msk}}  $$是词汇表分布的真实 one-hot 向量，真实标记的概率&#x4E3A;**`1`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">图文匹配 ITM</span>**

**ITM&#x20;**&#x9884;测一对图像和文本是正样本还是负样本，即匹配还是不匹配。**ALBEF&#x20;**&#x4F7F;用多模态编码器输出&#x7684;**`[CLS]`**&#x6807;记嵌入作为图文对的联合表示，并附加一个全连接 FC 层和 softmax 来预测二分类概率$$p_{\text{itm}}$$。**ITM&#x20;**&#x635F;失定义为：

$$\mathcal{L}_{\text{itm}} = \mathbb{E}_{(I,T)\sim D} H(y_{\text{itm}}, p_{\text{itm}}(I, T))$$

其中 $$  y_{\text{itm}}  $$ 是表示真实标签的二维 one-hot 向量。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**<span style="color: rgb(100,37,208); background-color: inherit">ALBEF </span>**<span style="color: rgb(100,37,208); background-color: inherit">提出了一种无需额外计算开销的策略，用于为 </span>**<span style="color: rgb(100,37,208); background-color: inherit">ITM </span>**<span style="color: rgb(100,37,208); background-color: inherit">任务采样硬负样本</span>：如果一个负样本它们共享相似的语义但在细粒度细节上不同，那么称这个图文对是 hard 的。**ALBEF&#x20;**&#x5229;用 **ITC&#x20;**&#x4E2D;的对比相似度在批次内寻找硬负样本。对于小批量中的每个图像，根据对比相似度分布从同一批次中采样一个负文本，与图像更相似的文本有更高的采样概率。同样，我们也为每个文本采样一个硬负样本图像。<span style="color: rgb(100,37,208); background-color: inherit">也就是说图文对比相似度最高的是正样本，第二高的是硬负样本</span>。

**ALBEF&#x20;**&#x7684;完整预训练目标为：

$$\mathcal{L} = \mathcal{L}_{\text{itc}} + \mathcal{L}_{\text{mlm}} + \mathcal{L}_{\text{itm}}$$

**<span style="color: rgb(36,91,219); background-color: inherit">动量蒸馏</span>**

用于预训练的图文对大多是从网络上收集的，因此往往带有噪声。正样本对通常是弱相关的：<span style="color: rgb(216,57,49); background-color: inherit">文本可能包含与图像无关的词语，或者图像可能包含未在文本中描述的实体</span>。对于图文对比学习 **ITC**，<span style="color: rgb(216,57,49); background-color: inherit">负样本文本可能也与图像内容匹配</span>。对于掩码语言建模 **MLM**，<span style="color: rgb(216,57,49); background-color: inherit">可能存在其他不同于标注但同样能描述图像的词语，甚至可能那更好</span>。然而，<span style="color: rgb(216,57,49); background-color: inherit">ITC 和 MLM 的 one-hot 标签会惩罚所有负预测，而不论其正确性</span>。

为了解决这一问题，**ALBEF&#x20;**&#x63D0;出从动量模型生成的伪目标中学习。<span style="color: rgb(100,37,208); background-color: inherit">动量模型是一个持续演化的教师模型，由单模态和多模态编码器的指数移动平均版本组成</span>。在训练过程中，目标是<span style="color: rgb(100,37,208); background-color: inherit">训练基础模型，使其预测结果与动量模型的预测结果一致</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">图文对比学习 ITC</span>**

对于 **ITC**，我们首先使用动量单模态编码器的特征计算图像-文本相似度：

$$s'(I, T) = g_v'(v_{\text{cls}}')^\top g_w'(w_{\text{cls}}'), \quad s'(T, I) = g_w'(w_{\text{cls}}')^\top g_v'(v_{\text{cls}}')$$

然后通过将之前 **ITC&#x20;**&#x516C;式中的$$  s  $$替换为$$  s'  $$来计算软伪目标$$  q^{\text{i2t}}  $$和$$q^{\text{t2i}}$$。$$\text{ITC}_\text{MoD}$$损失定义为：

$$\mathcal{L}_{\text{itc}}^{\text{mod}} = (1 - \alpha)\mathcal{L}_{\text{itc}} + \frac{\alpha}{2} \mathbb{E}_{(I,T)\sim D} \left[ \text{KL}(q^{\text{i2t}}(I) \| p^{\text{i2t}}(I)) + \text{KL}(q^{\text{t2i}}(T) \| p^{\text{t2i}}(T)) \right]$$

其中 $$\text{KL}$$ 表示 KL 散度。

* **<span style="color: rgb(36,91,219); background-color: inherit">掩码语言建模 MLM</span>**

对于 **MLM**，设$$  q_{\text{msk}}(I, \hat{T})  $$表示动量模型对遮掩标记的预测概率，$$\text{MLM}_\text{MoD}$$损失定义为：

$$\mathcal{L}_{\text{mlm}}^{\text{mod}} = (1 - \alpha)\mathcal{L}_{\text{mlm}} + \alpha \mathbb{E}_{(I,\hat{T})\sim D} \text{KL}(q_{\text{msk}}(I, \hat{T}) \| p_{\text{msk}}(I, \hat{T}))$$

下图展示了伪目标中&#x524D;**`5`**&#x4E2A;候选词/文本的例子，<span style="color: rgb(100,37,208); background-color: inherit">这些伪目标有效地捕捉了与图像相关的内容</span>：

![](../../images/视觉多模态讲义（上）-image-102.png)

除此之外，**ALBEF&#x20;**&#x8FD8;将 **MoD&#x20;**&#x5E94;用于下游任务。每个任务的最终损失是原始任务损失与模型预测和伪目标之间的 KL 散度的加权组合。为了简化，我们将权重$$\alpha$$设置&#x4E3A;**`0.4`**，适用于所有预训练和下游任务。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**为什么 ITM loss 没有动量版本**
>
> 因为本身基于 ground truth，必须要知道是不是一个 pair，就是一个二分类任务。而且有困难负样本，与动量冲突

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

为了从噪声的数据更有效的学习图像和文本特征，ALBEF 采用额外的监督信号，即伪标签的自训练的方式

* ALBEF 借&#x9274;**`MoCo`**&#x8BBA;文里的 Momentum Encoder 的方式，用 Momentum Model 生成伪标签，即 softmax score，不再是 one-hot label。具体做法是使用 **<span style="color: rgb(216,57,49); background-color: inherit">EMA</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">xponential </span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">oving </span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">verage）</span>

* 最后的结果是达到一个很好的平均，很多的信息可以从 one hot label 里面学习，当它是错误或噪声的时候，稳定的 Momentum Model 可以提供一些改进

* 多个目标函数，&#x5982;**`ITM`**、**`MLM`**、**`Momentum Distillation`**&#x90FD;是为了同一个图像文本对生成不同的视角，变相的数据增强，让训练后的模型具备 semantic preserving 的功能，即只要是语义匹配的图像文本对，应该被当作一对

**<span style="color: rgb(36,91,219); background-color: inherit">数据集</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">版本 1</span>**：**`Conceptual captions 3M`**+**`SBU caption`**+**`COCO`**+**`Visual Genome`**&#x5171;**`4M`**

**<span style="color: rgb(36,91,219); background-color: inherit">版本2</span>**：加&#x5165;**`Conceptual captions 12M`**，&#x5171;**`14.1M`**

![](../../images/视觉多模态讲义（上）-image-105.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">另一个角度的 ALBEF</span>**

形式上，定义两个随机变量$$a$$和$$b$$作为数据点的两种不同视图。<span style="color: rgb(100,37,208); background-color: inherit">在自监督学习中，</span>$$a$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$b$$<span style="color: rgb(100,37,208); background-color: inherit">是同一图像的两种增强版本。在视觉-语言表示学习中，将</span>$$a$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$b$$<span style="color: rgb(100,37,208); background-color: inherit">视为捕捉图文对语义意义的不同变体。目标是学习对视图变化具有不变性的表示</span>。这可以通过最大化$$a$$和$$b$$之间的互信&#x606F;**`MI`**&#x6765;实现。在实践中，通过最小&#x5316;**`InfoNCE`**&#x635F;失来最大化$$\text{MI}(a, b)$$的下界，**InfoNCE&#x20;**&#x635F;失定义为：

$$L_{\text{NCE}} = -\mathbb{E}_{p(a,b)} \left[ \log \frac{\exp(s(a, b))}{\sum_{\hat{b} \in \hat{B}} \exp(s(a, \hat{b}))} \right]$$

其中 $$s(a, b)$$是一个评分函数，例如<span style="color: rgb(220,155,4); background-color: inherit">两个表示之间的点积</span>，$$\hat{B}$$包含正样本$$b$$和$$|\hat{B}| - 1$$个从分布中采样的负样本。

然后使用 one-hot 标签的 **ITC&#x20;**&#x635F;失可以重写为：

$$\mathcal{L}_{\text{itc}} = -\frac{1}{2} \mathbb{E}_{p(I,T)} \left[ \log \frac{\exp(s(I, T)/\tau)}{\sum_{m=1}^M \exp(s(I, T_m)/\tau)} + \log \frac{\exp(s(T, I)/\tau)}{\sum_{m=1}^M \exp(s(T, I_m)/\tau)} \right]$$

<span style="color: rgb(100,37,208); background-color: inherit">最小化 </span>$$\mathcal{L}_{\text{itc}}$$<span style="color: rgb(100,37,208); background-color: inherit"> 可以被视为最大化 </span>**<span style="color: rgb(100,37,208); background-color: inherit">InfoNCE </span>**<span style="color: rgb(100,37,208); background-color: inherit">的对称版本</span>。因此，ITC 将两个单独模态：图像$$I$$和文本$$T$$，视为图文对的两个视图，并训练单模态编码器以最大化正样本对的图像视图和文本视图之间的互信息。

一些文献指出，<span style="color: rgb(100,37,208); background-color: inherit">可以将 </span>**<span style="color: rgb(100,37,208); background-color: inherit">MLM </span>**<span style="color: rgb(100,37,208); background-color: inherit">解释为最大化被 Mask Token 与其 Mask 上下文，即图像+Mask 文本之间的互信息</span>。具体来说，可以将使用 one-hot 标签的 MLM 损失重写为：

$$\mathcal{L}_{\text{mlm}} = -\mathbb{E}_{p(I,\hat{T})} \left[ \log \frac{\exp(\psi(y_{\text{msk}})^\top f(I, \hat{T}))}{\sum_{y \in V} \exp(\psi(y)^\top f(I, \hat{T}))} \right]$$

其中$$\psi(y): V \to \mathbb{R}^d$$是多模态编码器输出层中的查找函数，它将词标记$$y$$映射到一个向量，$$V$$是完整词汇表集合，$$f(I, \hat{T})$$是返回与遮掩上下文对应的多模态编码器最终隐藏状态的函数。因此，MLM 将图文对的两个视图视为：

> 1. 一个随机选择的 Token
>
> 2. 图像+该词被 Mask 的上下文文本

ITC 和 MLM 都通过从图文对中提取部分信息来生成视图，前者通过模态分离，后者通过词遮掩。动量蒸馏可以被视为从整个提议分布中生成替代视图。以$$\text{ITC}_\text{MoD}$$为例，最小化 $$\text{KL}(p^{\text{i2t}}(I), q^{\text{i2t}}(I))$$ 等价于最小化以下目标：

$$-\sum_m q_m^{\text{i2t}}(I) \log p_m^{\text{i2t}}(I) = -\sum_m \frac{\exp(s'(I, T_m)/\tau)}{\sum_{m=1}^M \exp(s'(I, T_m)/\tau)} \log \frac{\exp(s(I, T_m)/\tau)}{\sum_{m=1}^M \exp(s(I, T_m)/\tau)}$$

它最大化了与图像$$I$$具有相似语义的文本的$$\text{MI}(I, T_m)$$，因为这些文本会有更大的$$q_m^{\text{i2t}}(I)$$。类似地，$$\text{ITC}_\text{MoD}$$还最大化了与文本$$T$$相似的图像的$$\text{MI}(I_m, T)$$。可以用同样的方法证明，$$\text{MLM}_\text{MoD}$$为被 Mask 的词$$y_{\text{msk}}$$生成了替代视图$$y' \in V$$，并最大化了$$y'$$与$$(I, \hat{T})$$之间的互信息。因此，<span style="color: rgb(100,37,208); background-color: inherit">动量蒸馏可以被视为对原始视图进行数据增强。动量模型生成了一组多样化的视图，这些视图在原始图文对中并不存在，并促使基础模型学习能够捕捉视图不变语义信息的表示</span>。

总的来说，这里从另一个角度解读了 **ALBEF**，并证明它最大化了图文对不同视图之间的互信息的下界。**`ITC`**、**`MLM`**&#x548C;**`MoD`**&#x53EF;以被解释为生成这些视图的不同方式。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**ALBEF&#x20;**&#x7684;出现标志着多模态学习进入了一个新的阶段。<span style="color: rgb(46,161,33); background-color: inherit">它通过</span>**<span style="color: rgb(46,161,33); background-color: inherit">先对齐再融合</span>**<span style="color: rgb(46,161,33); background-color: inherit">的策略，解决了传统模型在图文对齐和噪声数据处理上的难题。动量蒸馏技术的引入使得 ALBEF 能够充分利用大规模弱监督数据集，从而在多种下游任务中表现出色</span>。

然而，**ALBEF&#x20;**&#x4E5F;存在一些局限性。例如，<span style="color: rgb(222,120,2); background-color: inherit">在处理细粒度分类任务或复杂场景时，其性能可能有所下降</span>。此外，<span style="color: rgb(216,57,49); background-color: inherit">动量蒸馏的计算开销较大，可能限制其在实时任务中的应用</span>。

### 2.2.2 <span style="color: rgb(36,91,219); background-color: inherit">BLIP</span>

多模态学习始终是一个令人兴奋且充满挑战的领域。视觉和语言作为人类感知世界的核心方式，其结合为机器理解和生成跨模态信息提供了新的可能性。2022年，Salesforce 的研究团队提出了一个名为 **<span style="color: rgb(216,57,49); background-color: inherit">BLIP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">B</span>**<span style="color: rgb(216,57,49); background-color: inherit">ootstrapping </span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">anguage-</span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mage </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">re-training）</span>的框架，这一工作不仅统一了视觉语言任务的理解与生成能力，还在广泛的下游任务中展现了卓越的性能。

传统的视觉语言预训练模型通常专注于特定的任务，例如<span style="color: rgb(220,155,4); background-color: inherit">图像描述生成或视觉问答 VQA</span>。然而，<span style="color: rgb(216,57,49); background-color: inherit">这些模型往往缺乏通用性，难以同时处理理解型任务</span>，如<span style="color: rgb(220,155,4); background-color: inherit">分类、检索</span>，和生成型任务，如<span style="color: rgb(220,155,4); background-color: inherit">描述生成</span>。此外，<span style="color: rgb(216,57,49); background-color: inherit">许多模型依赖于高质量的标注数据，而这些数据的获取成本高昂且容易引入偏差</span>。**BLIP&#x20;**&#x7684;提出正是为了应对这些挑战，它<span style="color: rgb(46,161,33); background-color: inherit">通过一种新颖的预训练框架，实现了对多样任务的支持，同时减少了对昂贵标注数据的依赖</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：两个主要贡献
>
> * 统一的框架，同时完成理解 encoder-only 和生成 encoder-decoder 的任务
>
> * 清洗数据集

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

**BLIP&#x20;**&#x4F7F;&#x7528;**`ViT`**&#x4F5C;为图像编码器，它将输入图像划分为若干小块，并将它们编码为嵌入序列，同时添加一&#x4E2A;**`[CLS]`**&#x6807;记以表示全局图像特征。与使用预训练目标检测器进行视觉特征提取相比，**ViT&#x20;**&#x8BA1;算效率更高，已被最近的方法广泛采用。

为了预训练一个兼具理解和生成能力的统一模型，**BLIP&#x20;**&#x63D0;出了多模态编码器-解码器混合架&#x6784;**<span style="color: rgb(216,57,49); background-color: inherit">MED</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ultimodal mixture of </span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">ncoder-</span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">ecoder）</span>，这是一种多任务模型，结&#x5408;**`ALBEF`**&#x548C;**`VLMo`**&#x7684;思想，如下图，可以在以下三种功能之一中运行：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">单模态编码器</span>**：分别对图像和文本进行编码。文本编码器与 **BERT&#x20;**&#x76F8;同，在文本输入开头添加一&#x4E2A;**`[CLS]`**&#x6807;记以总结句子。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">基于图像的文本编码器</span>**：通过在每个 Transformer 块的自注意力层和前馈网络之间插入额外的跨注意力层，注入视觉信息。在文本末尾添加一个特定任务&#x7684;**`[Encode]`<span style="color: rgb(216,57,49); background-color: inherit"> </span>**&#x54;oken，其输出嵌入用作图文对的多模态表示。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">基于图像的文本解码器</span>**：将基于图像的文本编码器中的双向自注意力层替换为因果自注意力层。使&#x7528;**`[Decode]`** Token 表示序列开始，使用结束标记表示序列结束。

![](../../images/视觉多模态讲义（上）-image-100.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练目标</span>**

在预训练过程中联合优化三个目标，包括<span style="color: rgb(100,37,208); background-color: inherit">两个理解目标和一个生成目标。每对图文只需通过计算密集的视觉 Transformer 一次，而文本 Transformer 则需要三次前向传播，每次激活不同功能以计算三种损失</span>。

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">图文对比损失 ITC</span>**：激活单模态编码器，旨在<span style="color: rgb(100,37,208); background-color: inherit">通过对齐视觉 Transformer 和文本 Transformer 的特征空间，使正样本图文对的表示相似，而负样本对的表示不同</span>。BLIP 仍然采用动量编码器和软标签策略来改进对比学习，使用**动量编码器**生成 soft label，防止错过 negative pairs 中潜在的正样本。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">图文匹配损失 ITM</span>**：激活基于图像的文本编码器，旨在<span style="color: rgb(100,37,208); background-color: inherit">学习捕捉视觉和语言细粒度对齐的多模态表示</span>。ITM 是一个二分类任务，模型通过 ITM 头预测图文对是否匹配。这里也采用硬负样本挖掘策略以提高训练效果，即 ITC 中分数除自身最高的，一个批次中具有较高对比相似性的负对更有可能被选择来计算损失。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">语言建模损失 LM</span>**：激活基于图像的文本解码器，旨在根据图像生成文本描述。LM 优化交叉熵损失，以自回归方式最大化文本生成的可能性。BLIP 应用&#x4E86;**`0.1`**&#x7684;标签平滑。<span style="color: rgb(46,161,33); background-color: inherit">与广泛使用的 MLM 损失相比，LM 增强了模型将视觉信息转化为连贯描述的能力</span>。

为了实现高效预训练并利用多任务学习，<span style="color: rgb(100,37,208); background-color: inherit">文本编码器和文本解码器共享除自注意力层外的所有参数</span>。这是因为<span style="color: rgb(100,37,208); background-color: inherit">编码和解码任务的差异主要体现在自注意力层上</span>，特别是，编码器使用双向自注意来构建当前输入 Token 的表示，而解码器使用因果自注意来预测下一个 Token。另一方面，<span style="color: rgb(46,161,33); background-color: inherit">其他层在两种任务中功能相似，因此共享这些层可以提高训练效率，同时受益于多任务学习</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">CapFlit</span>**

由于高昂的标注成本，高质量的人工标注图文对数量有限，例&#x5982;**`COCO`**<span style="color: rgb(220,155,4); background-color: inherit">数据集</span>。有一些工作工作利用从网页自动收集的大规模图像和替代文本对，但<span style="color: rgb(216,57,49); background-color: inherit">这些替代文本往往不能准确描述图像内容，导致信号嘈杂，不利于视觉-语言对齐学习</span>。

BLIP 提出了 **<span style="color: rgb(216,57,49); background-color: inherit">CapFilt</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Cap</span>**<span style="color: rgb(216,57,49); background-color: inherit">tioning and </span>**<span style="color: rgb(216,57,49); background-color: inherit">Filt</span>**<span style="color: rgb(216,57,49); background-color: inherit">ering）</span>，一种提升文本语料质量的新方法。CapFilt引入两个模块：

> * 一&#x4E2A;**<span style="color: rgb(36,91,219); background-color: inherit">生成器</span>**&#x7528;于为网页图像生成描述
>
> * 一&#x4E2A;**<span style="color: rgb(36,91,219); background-color: inherit">过滤器</span>**&#x7528;于去除噪声图文对

<span style="color: rgb(216,57,49); background-color: inherit">生成器和过滤器均从相同的预训练 </span>**<span style="color: rgb(216,57,49); background-color: inherit">MED </span>**<span style="color: rgb(216,57,49); background-color: inherit">模型初始化，并在 </span>**<span style="color: rgb(216,57,49); background-color: inherit">COCO </span>**<span style="color: rgb(216,57,49); background-color: inherit">数据集上分别微调</span>，微调过程轻量化。

![](../../images/视觉多模态讲义（上）-image-104.png)

> <span style="color: rgb(216,57,49); background-color: inherit">Tw</span>: `CC12M` <span style="color: rgb(46,161,33); background-color: inherit">Th</span>: `COCO` <span style="color: rgb(46,161,33); background-color: inherit">Tw: </span>`Filtered CC12M` <span style="color: rgb(216,57,49); background-color: inherit">Ts</span>: `synthetic CC12M` <span style="color: rgb(46,161,33); background-color: inherit">Ts</span>: `filtered synthetic CC12M`&#x20;

具体而言，<span style="color: rgb(100,37,208); background-color: inherit">生成器是一个基于图像的文本解码器，通过 LM 目标微调以生成图像描述。给定网页图像</span>$$I_w$$<span style="color: rgb(100,37,208); background-color: inherit">，生成器为每张图像生成一条合成描述</span>$$T_s$$。<span style="color: rgb(100,37,208); background-color: inherit">过滤器是一个基于图像的文本编码器，通过 ITC 和 ITM 目标微调以判断文本是否匹配图像</span>。过滤器会移除原始网页文本$$T_w$$和合成文本$$T_s$$中的噪声文本，<span style="color: rgb(100,37,208); background-color: inherit">若 ITM 头预测某文本与图像不匹配，则认为该文本为噪声</span>。最后将过滤后的图文对与人工标注对结合形成新数据集，用于预训练新模型。

下图展示了一些示例描述及其对应的图像，定性地展示了生成器生成新文本描述的效果，以及过滤器从原始网页文本和合成文本中去除噪声描述的作用。

![](../../images/视觉多模态讲义（上）-image-103.png)

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**BLIP&#x20;**&#x5728;多个视觉语言任务上取得了显著的成果，包括但不限于<span style="color: rgb(220,155,4); background-color: inherit">图像描述生成、视觉问答、图像-文本检索等</span>。实验结果表明，**BLIP&#x20;**&#x5728;这些任务上的性能优于当时的大多数现有方法，尤其是在生成型任务中表现出色。此外，**BLIP**的灵活性使其能够轻松迁移到新的任务和领域，为实际应用提供了广泛的可能性。

**BLIP&#x20;**&#x7684;提出标志着多模态学习领域的一个重要事件。它不仅展示了视觉语言预训练的强大潜力，还为未来的研究提供了丰富的思路和方法。无论是从模型架构还是数据处理的角度来看，**BLIP&#x20;**&#x90FD;为我们提供了一个值得深入研究和借鉴的范例。

### 2.2.3 <span style="color: rgb(36,91,219); background-color: inherit">BLIP-2</span>

在人工智能领域，视觉与语言的结合一直是研究的热点。从早期的图文匹配任务到如今的多模态推理、视觉对话等复杂场景，技术的演进不断推动着我们对多模态智能的理解。**<span style="color: rgb(46,161,33); background-color: inherit">BLIP-2</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 作为这一领域的最新成果，以其创新的设计和高效的性能，成为连接视觉与语言的新一代桥梁</span>。

随着深度学习的发展，预训练模型已经成为解决多模态任务的核心工具。然而，<span style="color: rgb(216,57,49); background-color: inherit">传统的多模态模型通常需要从头开始联合训练视觉编码器和语言模型，这不仅计算成本高昂，还难以充分利用现有的高质量单模态预训练模型</span>。为了解决这一问题，**<span style="color: rgb(100,37,208); background-color: inherit">BLIP-2</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 提出了一种全新的范式：通过冻结预训练的图像编码器和大型语言模型，并引入一个轻量级的模块</span>**`Q-Former`**<span style="color: rgb(100,37,208); background-color: inherit">来实现高效且高性能的视觉-语言预训练</span>，如右图。

![](../../images/视觉多模态讲义（上）-image-96.png)

这种方法的<span style="color: rgb(100,37,208); background-color: inherit">核心思想是</span>**<span style="color: rgb(100,37,208); background-color: inherit">解耦</span>**<span style="color: rgb(100,37,208); background-color: inherit">——将视觉编码器和语言模型的训练过程分开，从而避免了重复训练的冗余性</span>。同时，通过引入 **Q-Former**，**BLIP-2** 能够在保持高效率的同时，灵活地融合视觉和语言信息。这借鉴了目前常见的做法：<span style="color: rgb(220,155,4); background-color: inherit">冻结一些预训练好的模型，如冻结 Image Encoder，另一些冻结 LLM，但挑战是需要把视觉特征对齐到文本空间</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：总的来说，BLIP-2 提出从现成的冻结预训练图像编码器和冻结的大型语言模型中引导视觉语言预训练。使用轻量级 **Q-Former** 弥补了模式上的差距，该转换器分两个阶段进行预训练：
>
> 1. 第一阶段从固定图像编码器中引导视觉语言表示学习
>
> 2. 第二阶段从一个固定的语言模型中引导视觉到语言的生成学习

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

**BLIP-2** 提出 **Q-Former** 作为连接冻结图像编码器和冻结 LLM 之间差距的可训练模块。它从图像编码器中提取固定数量的输出特征，与输入图像分辨率无关。如右图所示，<span style="color: rgb(100,37,208); background-color: inherit">Q-Former 由两个共享相同自注意力层的转换器子模块组成</span>：

![](../../images/视觉多模态讲义（上）-image-101.png)

> 1. 一个与冻结图像编码器交互以提取视觉特征&#x7684;**<span style="color: rgb(36,91,219); background-color: inherit">图像转换器</span>**
>
> 2. 一个可以同时作为文本编码器和文本解码器&#x7684;**<span style="color: rgb(36,91,219); background-color: inherit">文本转换器</span>**

**BLIP-2&#x20;**&#x521B;建了一组可学习的 Query Embedding 作为图像转换器的输入。<span style="color: rgb(100,37,208); background-color: inherit">这些 Query 通过自注意力层相互作用，并通过交叉注意力层与冻结的图像特征交互</span>。此外，<span style="color: rgb(100,37,208); background-color: inherit">Query 还可以通过相同的自注意力层与文本交互。根据预训练任务的不同，应用不同的自注意力掩码来控制 Query 与文本的交互</span>。Q-Former 使用$$\text{BERT}_\text{base}$$的预训练权重初始化，而交叉注意力层则随机初始化。Q-Former 总计包&#x542B;**`188M`**&#x4E2A;参数。需要注意的是，Query 被视为模型参数。在实验中，**BLIP-2&#x20;**&#x4F7F;用&#x4E86;**`32`**&#x4E2A; Query，每个 Query 的维度&#x4E3A;**`768`**，这与 Q-Former 的隐藏维度保持一致。这里用$$Z$$表示输出的 Query 表示。$$Z$$的大&#x5C0F;**`32 × 768`**&#x8FDC;小于冻结图像特征的大小，例&#x5982;**`ViT-L/14`**&#x7684;**`257 × 1024`**。这种瓶颈架构与我们的预训练目标相结合，迫使查询提取与文本最相关的视觉信息。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉-语言表示学习</span>**

在表示学习阶段，将 **Q-Former** 连接到冻结的图像编码器，并使用图文对进行预训练。目标是<span style="color: rgb(100,37,208); background-color: inherit">训练 Q-Former，使 Query 能够学习提取对文本最具信息量的视觉表示，即挖掘视觉编码器的潜力</span>。受 BLIP 启发，**<span style="color: rgb(100,37,208); background-color: inherit">BLIP-2 </span>**<span style="color: rgb(100,37,208); background-color: inherit">联合优化三个共享相同输入格式和模型参数的预训练目标</span>。每个目标采用不同的注意力掩码策略来控制 Query 与文本的交互，如右图。



1. **<span style="color: rgb(36,91,219); background-color: inherit">图文对比学习 ITC</span>**

![](../../images/视觉多模态讲义（上）-image-98.png)

**<span style="color: rgb(100,37,208); background-color: inherit">ITC</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 学习对齐图像表示和文本表示，以最大化它们的互信息</span>。通过对正样本对的图文相似度与负样本对的相似度进行对比实现。**BLIP-2&#x20;**&#x5C06;图像转换器输出的查询表示$$Z$$与文本转换器输出的文本表示$$t$$对齐，其中$$t$$&#x662F;**`[CLS]`** Token 的输出 Embedding。由于$$Z$$包含多个输出 Embedding，每个 Query 一个，因此<span style="color: rgb(100,37,208); background-color: inherit">首先计算每个 Query 输出与</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">之间的成对相似度，然后选择最高的一个作为图文相似度</span>。为了避免信息泄露，使用单模态自注意力掩码，Query 和文本不允许看到彼此。由于使用了冻结的图像编码器，相比端到端方法，可以在每块 GPU 上容纳更多样本。因此，**BLIP-2&#x20;**&#x4F7F;用批量内的负样本，而不是 **BLIP&#x20;**&#x4E2D;的动量队列。

2. **<span style="color: rgb(36,91,219); background-color: inherit">基于图像的文本生成 ITG 损失</span>**

**ITG&#x20;**&#x635F;失训练 Q-Former 根据输入图像生成文本。由于 Q-Former 的架构不允许冻结的图像编码器与文本标记直接交互，<span style="color: rgb(100,37,208); background-color: inherit">生成文本所需的信息必须首先由 Query 提取，然后通过自注意力层传递给文本标记。因此，Query 被迫提取包含所有文本相关信息的视觉特征</span>。**BLIP-2&#x20;**&#x91C7;用多模态因果自注意力掩码来控制 Query 与文本的交互。Query 可以相互关注，但不能关注文本 Token。每个文本 Token 可以关注所有查询及其之前的文本标记。**BLIP-2&#x20;**&#x8FD8;&#x5C06;**`[CLS]`** Token 替换为新&#x7684;**`[DEC]`** Token 作为第一个文本标记，以指示解码任务。

* **<span style="color: rgb(36,91,219); background-color: inherit">图文匹配 ITM</span>**

ITM 旨在学习图像与文本表示之间的细粒度对齐。这是一个二分类任务，<span style="color: rgb(100,37,208); background-color: inherit">要求模型预测图文对是否为正样本或负样本，即匹配或不匹配</span>。这里使用<span style="color: rgb(100,37,208); background-color: inherit">双向自注意力掩码，所有 Query 和文本都可以相互关注</span>。输出的 Query Embedding$$Z$$因此捕获了多模态信息。**BLIP-2&#x20;**&#x5C06;每个输出 Query Embedding 输入到两类线性分类器中以获得logits，并将所有 Query 的 logits 平均作为输出匹配分数。除此之外还采用困难负样本挖掘策略来生成信息丰富的负样本对。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉到语言生成学习</span>**

在生成式预训练阶段，<span style="color: rgb(100,37,208); background-color: inherit">将连接冻结图像编码器的 Q-Former 与一个冻结的 LLM 相连，以利用 LLM 的生成语言能力</span>。如下图所示，<span style="color: rgb(100,37,208); background-color: inherit">使用一个全连接 FC 层将输出的 Query Embedding </span>$$Z$$<span style="color: rgb(100,37,208); background-color: inherit">线性投影到与 LLM 文本嵌入相同的维度</span>。投影后的 Query Embedding 被添加到输入文本嵌入的前面。它们充当软视觉提示，使 LLM 基于 Q-Former 提取的视觉表示进行条件化处理。<span style="color: rgb(46,161,33); background-color: inherit">由于 Q-Forme r已经过预训练，能够提取对语言有信息量的视觉表示，因此它有效地充当了一个信息瓶颈，在向 LLM 传递最有用的信息的同时过滤掉无关的视觉信息</span>。这<span style="color: rgb(46,161,33); background-color: inherit">减轻了 LLM 学习视觉-语言对齐的负担，从而缓解了灾难性遗忘问题</span>。

![](../../images/视觉多模态讲义（上）-image-97.png)

**BLIP-2&#x20;**&#x5C1D;试了两种类型的 LLM：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">基于解码器的 LLM</span>**：使用语言建模损失进行预训练，其中冻结的 LLM 根据 Q-Former 提供的视觉表示生成文本。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">基于编码器-解码器的 LLM</span>**：我们使用前缀语言建模损失进行预训练，其中将文本分为两部分。前缀文本与视觉表示拼接后作为 LLM 编码器的输入，而后缀文本则用作 LLM 解码器的生成目标。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型训练</span>**

在第一阶段预训&#x7EC3;**`250k`**&#x6B65;，在第二阶段预训&#x7EC3;**`80k`**&#x6B65;。

第一阶段中，对&#x4E8E;**`ViT-L/ViT-g`**，分别使&#x7528;**`2320/1680`**&#x7684;批量大小

第二阶段中，对&#x4E8E;**`OPT/FlanT5`**，分别使&#x7528;**`1920/1520`**&#x7684;批量大小

在预训练期间，将冻结的 ViT 和 LLM 的参数转换&#x4E3A;**`FP16`**&#x683C;式，其&#x4E2D;**`FlanT5`**&#x9664;外，它使&#x7528;**`BFloat16`**。作者发现<span style="color: rgb(46,161,33); background-color: inherit">与使用32位模型相比，没有性能下降。由于使用了冻结模型，预训练比现有的大规模视觉-语言预训练 </span>**<span style="color: rgb(46,161,33); background-color: inherit">VLP</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 方法更加计算友好</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">使用一台配备</span>**`16`**<span style="color: rgb(220,155,4); background-color: inherit">块</span>**`40G A100`**<span style="color: rgb(220,155,4); background-color: inherit">显卡的机器，最大模型</span>**`ViT-g`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`FlanT5-XXL`**<span style="color: rgb(220,155,4); background-color: inherit">在第一阶段不到</span>**`6`**<span style="color: rgb(220,155,4); background-color: inherit">天完成，在第二阶段不到</span>**`3`**<span style="color: rgb(220,155,4); background-color: inherit">天完成</span>。

所有模型均使用相同的预训练超参数。优化器使&#x7528;**`AdamW`**，参数设置为$$β1 = 0.9$$、$$β2 = 0.98$$，权重衰减&#x4E3A;**`0.05`**。学习率采用余弦衰减策略，峰值学习率&#x4E3A;**`1e-4`**，并使&#x7528;**`2000`**&#x6B65;的线性预热。第二阶段的最低学习率&#x4E3A;**`5e-5`**。图像使&#x7528;**`224×224`**&#x7684;大小，并通过随机调整裁剪和水平翻转进行数据增强。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**BLIP-2** 是一种高效且灵活的视觉-语言预训练模型，通过冻结的预训练视觉编码器和语言模型，结合轻量级 **Q-Former&#x20;**&#x5B9E;现跨模态融合，在图生文任务，如<span style="color: rgb(220,155,4); background-color: inherit">字幕生成、视觉问答</span>中表现出色。其模块化设计支持灵活替换视觉编码器和语言模型，适应不同任务需求。除此之外，**BLIP-2&#x20;**&#x53EF;进一步扩展： &#x20;

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">支持更多模态</span>**：如音频、视频等，迈向通用多模态框架 &#x20;
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">提升细粒度识别</span>**：优化查询机制，减少信息丢失，增强细节捕捉能力
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">拓展任务范围</span>**：尝试图像分类、检测、分割及生成等计算机视觉任务 &#x20;

**BLIP-2** 不仅是一项技术突破，更是一种连接视觉与语言的新理念，为多模态的发展提供了一种新的思路。

### 2.2.4 <span style="color: rgb(36,91,219); background-color: inherit">InstructBLIP</span>

在多模态的大背景下，视觉与语言的融合一直是研究者们不断追求的目标。从早期的简单图文匹配任务，到如今复杂的跨模态推理与生成任务，可谓是技术的飞速进步。作为 BLIP 系列工作的延续，**InstructBLIP&#x20;**&#x4E0D;仅继承了前作的技术积淀，还通过引入<span style="color: rgb(216,57,49); background-color: inherit">指令微调 Instruction Tuning </span>的方法，开创了一个全新的研究方向。

在多模态模型的研究中，如何让模型能够更好地理解图像与文本之间的关联，并在此基础上完成多样化的任务，是一个核心问题。<span style="color: rgb(216,57,49); background-color: inherit">尽管早期的模型如 CLIP 和 BLIP 已经展示了强大的能力，但它们在面对复杂指令或未见过的任务时，往往表现得力不从心</span>。这是因为<span style="color: rgb(216,57,49); background-color: inherit">这些模型大多依赖于预训练数据中的隐式对齐，缺乏显式的任务导向性训练</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：目前通用语言模型使用预训练加指令微调的范式。但是通用的视觉-语言模型具有挑战，因为不同任务之间差距过大。虽然 VLP 被广泛探索，但目&#x524D;**&#x20;Vision-Language Instruction Tuning** 探索的比较少。

正是基于这样的背景，InstructBLIP 应运而生。<span style="color: rgb(100,37,208); background-color: inherit">它的主要目标是解决视觉-语言模型在指令微调中的挑战，并显著提升模型在未见数据和任务上的泛化能力</span>。通过<span style="color: rgb(46,161,33); background-color: inherit">引入指令微调机制，</span>**<span style="color: rgb(46,161,33); background-color: inherit">InstructBLIP </span>**<span style="color: rgb(46,161,33); background-color: inherit">不仅能够看懂图像，还能根据指令进行推理并生成相应的回答，从而实现了更深层次的多模态理解</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据集</span>**

为了确保指令调优数据的多样性并兼顾其可获取性，**InstructBLIP&#x20;**&#x6536;集了全面的公开视觉-语言数据集，并将其转换为指令调优格式。如右图所示，最终的数据集涵盖&#x4E86;**`11`**&#x4E2A;任务类别&#x548C;**`26`**&#x4E2A;数据集

![](../../images/视觉多模态讲义（上）-image-95.png)

包括`图像描述生成`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`结合阅读理解的图像描述生成`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`视觉推理`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`图像问答`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`基于知识的图像问答`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`结合阅读理解的图像问答`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`图像问题生成`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`视频问答`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`视觉对话式问答`<span style="color: rgb(216,57,49); background-color: inherit">、</span>`图像分类`<span style="color: rgb(216,57,49); background-color: inherit">和</span>**`LLaVA-Instruct-150K`**。

对于每个任务，**InstructBLIP&#x20;**&#x7CBE;心设计&#x4E86;**`10`**&#x5230;**`15`**&#x4E2A;不同的自然语言指令模板。这些模板构成了指令调优数据的基础，明确了任务及其目标。对于那些天然倾向于简短回答的公共数据集，在部分对应的指令模板中加入了诸如`简短`或`简洁`等词汇，以降低模型过度拟合生成简短输出的风险。对于 LLaVA-Instruct-150K 数据集，由于其本身已经是指令格式，因此未添加额外的指令模板。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练评估策略</span>**

为了确保训练和零样本评估时有充足的数据和任务，**<span style="color: rgb(100,37,208); background-color: inherit">InstructBLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">将</span>**`26`**<span style="color: rgb(100,37,208); background-color: inherit">个数据集分为</span>**`13`**<span style="color: rgb(100,37,208); background-color: inherit">个</span>**`held-in`**<span style="color: rgb(100,37,208); background-color: inherit">数据集和</span>**`13`**<span style="color: rgb(100,37,208); background-color: inherit">个</span>**`held-out`**<span style="color: rgb(100,37,208); background-color: inherit">数据集，在上图中分别用黄色和白色表示</span>。**InstructBLIP&#x20;**&#x4F7F;用 **held-in** 数据集的训练集进行指令调优，并使用其验证集或测试集进行 **held-in** 评估。

对于 **held-out** 评估，**InstructBLIP&#x20;**&#x7684;目标是了解指令调优如何提升模型在未见数据上的 Zero-Shot 性能。这里定义了两种类型的 **held-out** 数据：

> 1. 训练期间未暴露给模型，但其任务类型存在于 **held-in** 集群中的数据集（数据集在训练时没见过但任务见过，用于测试数据分布的泛化性）
>
> 2. 训练期间完全未见过的数据集及其相关任务（任务也没有见过，用于测试模型通用性）

<span style="color: rgb(100,37,208); background-color: inherit">处理第一种 </span>**<span style="color: rgb(100,37,208); background-color: inherit">held-out</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 评估由于 </span>**<span style="color: rgb(100,37,208); background-color: inherit">held-in</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">held-out</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 数据集之间的数据分布差异而变得复杂</span>。对于第二种类型，完全保留了多个任务，包括<span style="color: rgb(220,155,4); background-color: inherit">视觉推理、视频问答、视觉对话式问答和图像分类</span>。

为避免数据污染，**<span style="color: rgb(100,37,208); background-color: inherit">InstructBLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">仔细选择了数据集，以确保没有任何评估数据出现在不同数据集的 </span>**<span style="color: rgb(100,37,208); background-color: inherit">held-in</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 训练集群中</span>。在指令调优过程中，<span style="color: rgb(100,37,208); background-color: inherit">将所有 </span>**<span style="color: rgb(100,37,208); background-color: inherit">held-in</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 训练集混合，并对每个数据集均匀采样指令模板</span>。模型通过标准的语言建模损失进行训练，直接根据指令生成回答。此外，<span style="color: rgb(100,37,208); background-color: inherit">对于涉及场景文本的数据集，在指令中添加 OCR 标记作为补充信息</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉特征提取</span>**

现有的 Zero-Shot 图像到文本生成方法在提取视觉特征时采用了一种与指令无关的方法，包&#x62EC;**`BLIP-2`**&#x7B49;。这导致<span style="color: rgb(216,57,49); background-color: inherit">无论任务如何，都有一组静态的视觉表示被输入到 LLM 中</span>。相比之下，<span style="color: rgb(46,161,33); background-color: inherit">一个指令感知的视觉模型可以适应任务指令，并生成最有利于当前任务的视觉表示</span>。如果期望同一输入图像的任务指令存在显著差异，这种方法显然具有优势。

![](../../images/视觉多模态讲义（上）-image-93.png)

上图展示了 **InstructBLIP&#x20;**&#x7684;架构。与 BLIP-2 类似，**<span style="color: rgb(100,37,208); background-color: inherit">InstructBLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">使用 Q-Former 从冻结的图像编码器中提取视觉特征</span>。Q-Former 的输入包含一组 $$K$$ 个可学习的 Query Embedding，它们通过交叉注意力机制与图像编码器的输出交互。<span style="color: rgb(100,37,208); background-color: inherit">Q-Former 的输出由 </span>$$K$$<span style="color: rgb(100,37,208); background-color: inherit"> 个编码后的视觉向量组成，每个 Query Embedding 对应一个向量，这些向量经过线性投影后被输入到冻结的 LLM 中</span>。与 BLIP-2 类似，Q-Former 在指令调优之前分两个阶段进行预训练：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">第一阶段</span>**&#x5C06; Q-Former 与冻结的图像编码器一起预训练，用于视觉-语言表示学习
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">第二阶段</span>**&#x5C06; Q-Former 的输出适配为软视觉提示，以支持冻结 LLM 的文本生成

预训练完成后，使用指令调优对 Q-Former 进行微调，其中 LLM 接收来自 Q-Former 的视觉编码和任务指令作为输入。

在 BLIP-2 的基础上，**<span style="color: rgb(100,37,208); background-color: inherit">InstructBLIP </span>**<span style="color: rgb(100,37,208); background-color: inherit">提出了一个指令感知的 Q-Former 模块，该模块将指令文本 Token 作为额外输入。指令通过 Q-Former 的自注意力层与 Query Embedding 交互，促进任务相关图像特征的提取</span>。通过如此操作，LLM 接收到更有利于遵循指令的视觉信息。当然实验也证明，指令感知的视觉特征提取在 **held-in** 和 **held-out** 评估中均带来了显著的性能提升。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：以下几点需要注意
>
> 1. 与 BLIP-2 不同的是在 Q-Former 和 LLM 中加入 Instruction
>
> 2. 在 Instruction Tuning 过程中冻结图像编码器和 LLM
>
> 3. Instruction Tuning 使用标准 Language Modeling Loss

* **<span style="color: rgb(36,91,219); background-color: inherit">平衡数据集</span>**

由于训练数据集<span style="color: rgb(216,57,49); background-color: inherit">数量众多且各数据集规模差异较大，均匀混合这些数据可能导致模型对较小数据集过拟合，而对较大数据集欠拟合</span>。为了解决这一问题，**InstructBLIP&#x20;**&#x63D0;出<span style="color: rgb(100,37,208); background-color: inherit">按数据集大小的平方根比例采样数据集</span>。具体来说，给定$$D$$个数据集，其大小分别为$$\{S_1, S_2, \dots, S_D\}$$，从数据集$$d$$中选择一个样本的概率为：

$$p_d = \frac{\sqrt{S_d}}{\sum_{i=1}^D \sqrt{S_i}}$$

在此公式的基础上，**InstructBLIP&#x20;**&#x5BF9;手动调整某些数据集的权重以优化训练效果，因为不同数据集和任务即使规模相似，也可能需要不同的训练强度。例如<span style="color: rgb(220,155,4); background-color: inherit">降低多选题数据集</span>**`A-OKVQA`**<span style="color: rgb(220,155,4); background-color: inherit">的权重，增加需要开放式文本生成的数据集</span>**`OKVQA`**<span style="color: rgb(220,155,4); background-color: inherit">的权重</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理</span>**

在推理阶段，**InstructBLIP&#x20;**&#x9488;对不同数据集采用了两种略有不同的生成方法：

1. 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">大多数数据集</span>**，如图像描述生成和开放式视觉问答，指令调优模型直接根据 prompt 生成回答，随后将其与真实值进行比较以计算指标。

2. 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">分类和多选视觉问答任务</span>**，采用词汇排名方法。具体来说，仍然提示模型生成答案，但将其词汇限制在一个候选列表中。然后计算每个候选的对数似然 log-likelihood，并选择值最高的候选作为最终预测。该排名方法应用于 **`ScienceQA`**、**`IconQA`**、**`A-OKVQA`**、**`HatefulMemes`**、**`Visual Dialog`**、**`MSVD`**&#x548C;**`MSRVTT`**&#x6570;据集。此外，对于二分类任务，将正负标签扩展为一组稍宽泛的表达词，以利用自然文本中的词频，例如<span style="color: rgb(220,155,4); background-color: inherit">正类用</span>**`yes`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`true`**<span style="color: rgb(220,155,4); background-color: inherit">，负类用</span>**`no`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`false`**。对于视频问答任务，每段视频使用四个均匀采样的帧。每个帧单独通过图像编码器和 Q-Former 处理，提取的视觉特征在输入 LLM 之前被拼接。

* **<span style="color: rgb(36,91,219); background-color: inherit">实现细节</span>**

得益于 BLIP-2 模块化架构设计的灵活性，可以快速将模型适配到各种 LLM。在实验中采用了四种 BLIP-2 变体，使用相同的图像编码&#x5668;**`ViT-g/14`**&#x4F46;不同的冻结 LLM，包&#x62EC;**`FlanT5-XL-3B`**、**`FlanT5-XXL-11B`**、**`Vicuna-7B`**&#x548C;**`Vicuna-13B`**。**FlanT5** 是基于编码器-解码器 Transformer T5 的指令调优模型。**Vicuna** 则是一个最近发布的仅解码器 Transformer 模型，基于 LLaMA 指令调优而来。<span style="color: rgb(100,37,208); background-color: inherit">在视觉-语言指令调优过程中，从预训练的 BLIP-2 Checkpoint 初始化模型，仅微调 Q-Former 的参数，同时保持图像编码器和 LLM 冻结</span>。由于原始 BLIP-2 模型未提供 Vicuna 的 Checkpoint，**InstructBLIP&#x20;**&#x4F7F;用与 BLIP-2 相同的流程对 Vicuna 进行预训练。

所有模型均通过最&#x591A;**`60,000`**&#x6B65;指令调优，&#x6BCF;**`3,000`**&#x6B65;验证一次性能。对于每个模型，选择单个最佳 Checkpoint 用于所有数据集的评估。这里分别&#x4E3A;**`3B`**、**`7B`**&#x548C;**`11/13B`**&#x6A21;型设置了批量大小&#x4E3A;**`192`**、**`128`**&#x548C;**`64`**。优化器采&#x7528;**`AdamW`**，设置$$\beta_1 = 0.9$$、$$\beta_2 = 0.999$$和权重衰减&#x4E3A;**`0.05`**。此外，在&#x524D;**`1,000`**&#x6B65;中应用学习率的线性预热，从$$10^{-8}$$增加到$$10^{-5}$$，随后使用余弦衰减，最小学习率&#x4E3A;**`0`**。所有模型均使&#x7528;**`16`**&#x5757;**`40G A100`** GPU 进行训练，并&#x5728;**`1.5`**&#x5929;内完成训练。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

为了验证 InstructBLIP 的有效性，作者团队在多个公开数据集上进行了广泛的实验。结果显示，<span style="color: rgb(100,37,208); background-color: inherit">InstructBLIP 在几乎所有任务上的表现都优于现有的多模态模型，包括 BLIP-2、CLIP 等。尤其是在零样本场景下，InstructBLIP 展现出了卓越的泛化能力，证明了指令微调机制的强大潜力</span>。

此外，InstructBLIP 还在一些新兴任务上取得了突破性的进展。例如，<span style="color: rgb(220,155,4); background-color: inherit">在视觉推理任务中，它能够准确地根据图像内容回答复杂的逻辑问题；在跨模态翻译任务中，它能够生成流畅且语义准确的目标语言文本</span>。这些成果充分体现了 InstructBLIP 在多模态领域的领先地位。

InstructBLIP 的成功不仅在于其技术上的创新，更在于它为多模态模型的研究指明了一个新的方向。通过引入指令微调机制，InstructBLIP 成功地将视觉与语言的理解能力提升到了一个新的高度。

### 2.2.5 <span style="color: rgb(36,91,219); background-color: inherit">BLIP-3</span>

在人工智能领域，多模态模型正成为连接视觉与语言的重要桥梁。**BLIP-3**，作&#x4E3A;**`xGen-MM`**&#x6846;架的核心成果之一，不仅延续了 **BLIP&#x20;**&#x7CFB;列在多模态领域的深厚积累，更在开放性、轻量化和性能上取得了显著突破。

多模态模型的目标是让机器能够同时理解图像、视频和文本等多源信息，并在此基础上完成复杂的任务，例如视觉问&#x7B54;**`VQA`**、图像描述生&#x6210;**`Captioning`**&#x4EE5;及指令跟&#x8E2A;**`Instruction Following`**。然而，这一领域仍面临诸多挑战：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">资源获取的差异</span>**：<span style="color: rgb(216,57,49); background-color: inherit">开源模型与专有模型之间的差距显著</span>，尤其是在权重、训练配方和数据集方面
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">高质量数据的需求</span>**：多模态模型需要大量高质量的图像-文本对数据，而<span style="color: rgb(216,57,49); background-color: inherit">这些数据的获取和标注成本高昂</span>
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">计算资源的限制</span>**：训练大规模多模态模型需要强大的硬件支持，<span style="color: rgb(216,57,49); background-color: inherit">如何在有限资源下实现高性能</span>成为一个难题

虽然过去 BLIP 系列对多模态大模型发展起到至关重要的作用，但从效果上来说，已经远落后于当下的 SOTA 模型，**BLIP-3** 正是在这样的背景下诞生，它通过一系列创新设计，试图解决上述问题，并推动多模态模型的发展。主要贡献有以三点：

![](../../images/视觉多模态讲义（上）-image-99.png)

**<span style="color: rgb(222,120,2); background-color: inherit">数据上</span>**

**<span style="color: rgb(216,57,49); background-color: inherit">改进前</span>**

之前训练数据数量少、质量不高、多样性不强

**<span style="color: rgb(46,161,33); background-color: inherit">改进后</span>**

构造了更大的、质量更高、多样性更强的数据集

**<span style="color: rgb(222,120,2); background-color: inherit">训练策略上</span>**

**<span style="color: rgb(216,57,49); background-color: inherit">改进前</span>**

多个 stage，包&#x62EC;**`ITM`**，**`ITC`**， **`ITG`**，训练流程冗长，而且 up scale 的训练开销大

**<span style="color: rgb(46,161,33); background-color: inherit">改进后</span>**

提出 3 stage 的训练范式，并统一用 next token prediction作为训练目标目标，提升训练效率和模型效果

**<span style="color: rgb(222,120,2); background-color: inherit">模型结构上</span>**

**<span style="color: rgb(216,57,49); background-color: inherit">改进前</span>**

BLIP系列仅支持单图输入，应用范围相对较窄

**<span style="color: rgb(46,161,33); background-color: inherit">改进后</span>**

支持交错图文输入

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

架构概述。如右图所示，BLIP-3 框架采用了一种由 **ViT** **`ViT-SO400M-14-SigLIP-384`**、**`Vision Token Sampler`**&#x548C;预训练的 **LLM** **`phi3-mini`**&#x7EC4;成的架构。**<span style="color: rgb(100,37,208); background-color: inherit">ViT </span>**<span style="color: rgb(100,37,208); background-color: inherit">用于提取图像特征，</span>**<span style="color: rgb(100,37,208); background-color: inherit">Vision Token Sampler </span>**<span style="color: rgb(100,37,208); background-color: inherit">用于对图像 Embedding 进行下采样，而 </span>**<span style="color: rgb(100,37,208); background-color: inherit">LLM </span>**<span style="color: rgb(100,37,208); background-color: inherit">则负责处理多模态输入</span>。模型的输入可以是来自不同多模态数据源的自由形式的交错文本和视觉 Token。在 BLIP-3 中，ViT 不参与训练，但 LLM 参与训练。



![](../../images/视觉多模态讲义（上）-image-94.png)

**<span style="color: rgb(36,91,219); background-color: inherit">任意分辨率视觉 Token 采样</span>**

正如在 LLaVA-NeXT 等工作中证明的，动态高分辨率图像编码策略在微调和后训练阶段非常有效。BLIP-3 通过基于图像块的编码实现了更高分辨率的图像理解。这种块级编码通过将单张图像分割为多个小块并分别编码，尽可能保留原始图像的分辨率。遵循以往的经验，BLIP-3 将编码后的图像块与缩小尺寸的原始图像连接起来，后者提供了全局信息。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**`Any-Resolution Vision Token Sampling`**&#x7B56;略可以强化模型对细粒度信息的捕获能力

具体流程如下：

1. **<span style="color: rgb(36,91,219); background-color: inherit">找到最优分辨率</span>**

<span style="color: rgb(100,37,208); background-color: inherit">预设了一些模版，通过下面的目标找到输入图片最适合的分辨率</span>

$$\begin{aligned} &\mathrm{Objection:} \mathrm{Arg} \min_{t} (\mathrm{wasted\_resolution}), t=1,2, \cdots N \\ \end{aligned}$$

其中$$t$$为模版的索引，一共有$$N$$个预设模版

$$\begin{aligned} r_{t} &= \min( \frac{w_t} {w_{\rm ori}}, \frac{h_t} {h_{\rm ori}}) \\ \mathrm{wasted\,resolution} &= w_t * h_t - \min (w{\rm ori} * h_{\rm ori}, \mathrm{INT}(w_{\rm ori} * r_{t}) * \mathrm{INT} (h_{\rm ori} * r_{t}) ) \\\end{aligned}$$

示例代码如下：

* **<span style="color: rgb(36,91,219); background-color: inherit">切 Patch</span>**

每个 Patch 的 Size 为 ViT 的输入Size，**`BLIP-3`**&#x6240;用&#x7684;**`ViT-SO400M-14-SigLIP-384`** 的输入 Size &#x4E3A;**`384x384`**。假设<span style="color: rgb(220,155,4); background-color: inherit">输入图片所映射的模板 Size 为</span>**`768x768`**<span style="color: rgb(220,155,4); background-color: inherit">，将图片 Resize 到</span>**`768x768`**<span style="color: rgb(220,155,4); background-color: inherit">后进行切分，得到</span>**`5`**<span style="color: rgb(220,155,4); background-color: inherit">个Patch：</span>**`4`**<span style="color: rgb(220,155,4); background-color: inherit">个切分 Patch 加上一个包含全局信息的 Patch</span>，如下图所示。

![](../../images/视觉多模态讲义（上）-image-91.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">计算每个 Patch 的 Image Embedding</span>**

一&#x4E2A;**`384x384`**&#x7684;图片经&#x8FC7;**`ViT-SO400M-14-SigLIP-384`**&#x540E;得&#x5230;**`576`**&#x4E2A;token：$$(\frac{384}{16})^2 = 576$$

![](../../images/视觉多模态讲义（上）-image-92.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">分别提取每个 Patch 的 Vision Token 再拼接</span>**

如下图，不同模版分辨率的 Image Token是不同的。通常来说，Token 数越多，包含的细粒度信息就越多。对于下游感知密集型的推理任务会有更大优势，&#x5982;**`DocQA`**

![](../../images/视觉多模态讲义（上）-image-115.png)

**<span style="color: rgb(36,91,219); background-color: inherit">视觉语言连接器 VL-connector</span>**

在视觉语言连接器中，BLIP-3 使&#x7528;**`perceiver resampler`**&#x5BF9;视觉 Token 进行下采样。通过任意分辨率图像编码，对每个图像 Patch，包括缩小的原始图像，独立执行下采样操作。随后，这些下采样的视觉 Token 被合并并传递给 LLM。通过视觉语言连接器中的下采样，可以根据 **perceiver resampler&#x20;**&#x4E2D; Query Token 的数量，将视觉 Token 的序列长度减少五倍或更多。

BLIP-3 中没有沿用 BLIP-2 &#x7684;**`Q-Former`**，而是用&#x4E86;**`Flamingo`**&#x7684;**`Perceiver Resampler`**，如右图所示。二者核心思路其实都差不多——以 Learnable Queries 的方式，将Image Encoder 提取的 Image Embedding 转为固定长度的 Image Token。

![](../../images/视觉多模态讲义（上）-image-120.png)

![](../../images/视觉多模态讲义（上）-image-117.png)

![](../../images/视觉多模态讲义（上）-image-116.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

预训练的目标是预测在混合数据集上训练时的下一个文本 Token。总体而言，最终的基础模&#x578B;**`xGen-MM-Phi3-mini-base-r`**&#x5728;组合数据集上预训练了大&#x7EA6;**`100B`**&#x4E2A;多模态 Token，预训练的图像分辨率&#x4E3A;**`384x384`**&#x50CF;素，&#x4E0E;**`SigLIP`**&#x4FDD;持一致。

![](../../images/视觉多模态讲义（上）-image-119.png)

三个自建数据集：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">BLIP3-KALE</span>**：大规模高质量描述数据集
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">BLIP3-OCR-200M</span>**：&#x4ECE;**`Datacomp-1B`**&#x4E2D;搜集&#x4E86;**`200M`**&#x5F20;高分辨率图片。&#x7528;**`PaddleOCR`**&#x63D0;取里面的文字信息。虽&#x7136;**`BLIP-3`**&#x4E2D;并没有用到文本的 **bounding box**，但是使用的话会提升 OCR QA 类任务的效果
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">BLIP3-GROUNDING-50M</span>**：&#x4ECE;**`Datacomp-1B`**&#x4E2D;搜&#x96C6;**`50M`**&#x56FE;片，图文信息都要。用开源的开集目标检测模&#x578B;**`Grounding-DINO`**&#x548C;**`Recognize Anything`**&#x8FDB;行识别。将 caption 中的对应的 object 替换为包含位置信息的 object

**<span style="color: rgb(36,91,219); background-color: inherit">监督微调 SFT</span>**

进一步在遵循指令的示例数据集上对预训练模型进行微调，使其能够更好地理解和回答用户问题。在微调阶段，<span style="color: rgb(100,37,208); background-color: inherit">使用了一系列公开可用的指令跟随数据集</span>。这里<span style="color: rgb(100,37,208); background-color: inherit">采用了任意分辨率视觉 Token 采样策略，以便更好地理解高分辨率图像</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">包含大量文本的文档数据</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：微调&#x4E86;**`1`**&#x4E2A;epoch
>
> 开源数据集包括：多模态会话，image caption，VQA，document QA，science and math understand

**<span style="color: rgb(36,91,219); background-color: inherit">交错多图像监督微调</span>**

在已经过指令微调的模型上进行第二阶段的微调，使用混合了多图像和单图像指令跟随样本的数据集。这一阶段的<span style="color: rgb(46,161,33); background-color: inherit">目标是增强模型对交错图文输入的理解能力，这对多模态上下文学习、多图像问答以及许多实际应用场景非常有帮助</span>。在多图像微调中，<span style="color: rgb(100,37,208); background-color: inherit">同样采用了与之前 SFT 阶段相同的任意分辨率视觉 Token 采样策略</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：用多图交错图文数据&#x96C6;**`MANTIS`**，**`Mmdu`**&#x7ED3;合之前微调的单图交错图片数据对模型进行进一步微调

**<span style="color: rgb(36,91,219); background-color: inherit">后训练 Post-training</span>**

最后通过两个阶段的后训练来提升模型的实用性，同时减少幻觉和毒性等有害特性。

1. **<span style="color: rgb(36,91,219); background-color: inherit">通过</span>`DPO`<span style="color: rgb(36,91,219); background-color: inherit">提升模型的</span>`Truthfulness`**

> * **<span style="color: rgb(36,91,219); background-color: inherit">训练数据</span>**：该阶段利用了开源&#x7684;**`VLFeedback`**&#x6570;据集的指令，**VLFeedback** 是一个用 GPT4-v 构造多模态偏好数据集，总&#x8BA1;**`80K`**。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">构造方式</span>**：给定指令，让多个 VLM 模型做生成，随后 GPT4-v &#x4ECE;**`helpfulness`**, **`visual faithfulness`**,  **`ethics`**&#x5BF9;生成的结果进行打分。分值高的输出作为 **preferred responses**，分值低的输出作为 **dispreferred responses**。**BLIP-3&#x20;**&#x8FDB;一步过滤掉首选响应得分较低的sample，最终得&#x5230;**`62.6K`**&#x6570;据。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">训练方式</span>**：**BLIP-3&#x20;**&#x91C7;&#x7528;**`DPO`**&#x4F5C;为训练目标，&#x7528;**`LoRA`**&#x5FAE;调 LLM **`2.5%`**&#x53C2;数，总计训&#x7EC3;**`1`**&#x4E2A;epoch。

* **<span style="color: rgb(36,91,219); background-color: inherit">通过</span>`Safty-SFT`<span style="color: rgb(36,91,219); background-color: inherit">提升模型的</span>`Harmlessness`**

> - **<span style="color: rgb(36,91,219); background-color: inherit">训练数据</span>**：&#x7528;**`VLGuard`**&#x6570;据集+随&#x673A;**`5K`** SFT数据集对 BLIP-3 再次进行微调，在保&#x7559;**`helpfulness`**&#x7684;同时提&#x5347;**`harmlessness`**。
>
> - **<span style="color: rgb(36,91,219); background-color: inherit">训练方式</span>**：**`LoRA`**&#x5FAE;调 LLM **`2.5%`**&#x53C2;数

最后，模型在训练和推理方面的流程如下：

**<span style="color: rgb(222,120,2); background-color: inherit">训练阶段</span>**

图片用 visual tokenizer 转为 visual token，文本用 text tokenizer 转为 text token，最后按序拼接起来送入 LLM，用 causal mask 做并行，仅对文本 token 处计算自回归损失。

**<span style="color: rgb(222,120,2); background-color: inherit">推理阶段</span>**

图片用 visual tokenizer 转为 visual token，文本用 text tokenizer 转为 text token，最后按序拼接起来送入 LLM，按照 next token prediction 的范式做生成。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

BLIP-3 在多个公开基准测试中表现出色，其性能可与当前最先进的专有模型相媲美：

> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉问答 VQA</span>**：&#x5728;**`COCO-VQA`**&#x6570;据集上，BLIP-3 的准确率达到&#x4E86;**`92.5%`**，比前代模型提升了&#x7EA6;**`5%`**。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图像描述生成 Image Captioning</span>**：&#x5728;**`Flickr30K`**&#x6570;据集上，BLIP-3 的BLEU-4得分达&#x5230;**`71.3%`**，远超同类开源模型。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">指令跟踪 Instruction Following</span>**：在 **<span style="color: rgb(216,57,49); background-color: inherit">MME</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ulti-</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">odal </span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">valuation）</span>基准测试中，BLIP-3 在多项任务中均取得了领先成绩。

这些结果表明，BLIP-3 不仅在学术研究中具有重要意义，也为实际应用提供了坚实的技术支撑。

但 BLIP-3 也还存在一些缺点，就是只能解决多模态图文交错输入，单模态文本输出。

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

## 2.4 <span style="color: rgb(36,91,219); background-color: inherit">LLaVA 系列</span>

### 2.4.1 <span style="color: rgb(36,91,219); background-color: inherit">LLaVA</span>

**<span style="color: rgb(216,57,49); background-color: inherit">LLaVA</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">arge </span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">anguage </span>**<span style="color: rgb(216,57,49); background-color: inherit">a</span>**<span style="color: rgb(216,57,49); background-color: inherit">nd </span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ision </span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">ssistant）</span>是一个多模态大模型，它<span style="color: rgb(100,37,208); background-color: inherit">通过结合视觉编码器和语言模型，实现了对图像和文本数据的联合理解与生成</span>。作为开源领域的新兴力量，**LLaVA&#x20;**&#x4E0D;仅展示了强大的多模态能力，还以其模块化架构和高效的训练方法吸引了广泛的关注。

现在有很多图像文本配对数据，但是<span style="color: rgb(216,57,49); background-color: inherit">基于图像和文本间的联系相对较浅</span>，因此目前研究者希望更进一步让模型去图像中挖掘详细信息。但是这种期望基于现有数据集很难达成。因此 **<span style="color: rgb(100,37,208); background-color: inherit">LLaVA </span>**<span style="color: rgb(100,37,208); background-color: inherit">使用 GPT 等语言模型内部存在的常识去强化数据集中文字部分。相当于把语言模型的各种信息灌注到 Image-Text数据集，从而取得好的数据集，做更多详细的下游任务</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据生成</span>**

目前公共多模态数据数量的激增，&#x4ECE;**`CC`**&#x5230;**`LAION`**。然而，<span style="color: rgb(216,57,49); background-color: inherit">当涉及到多模态指令跟随数据时，可用的数据量有限，部分原因是创建此类数据的过程耗时且在考虑人工众包时定义不够明确</span>。受最近 GPT 模型在文本标注任务中成功的启发，**LLaVA&#x20;**&#x5229;用 GPT 基于广泛存在的图像对数据进行多模态指令跟随数据收集。对于一张图像 $$X_v$$ 及其相关标题$$X_c$$，很自然地可以创建一组问题$$X_q$$，指示助手描述图像内容。**LLaVA&#x20;**&#x63D0;示 GPT 整理出这样的问题列表。因此，将图文对扩展为其指令跟随版本的一种简单方法是：

$$\text{Human}: X_q X_v\langle\text{STOP}\rangle \quad\text{Assistant}: X_c\langle\text{STOP}\rangle$$

尽管这种简单的扩展版本构建成本低廉，但<span style="color: rgb(216,57,49); background-color: inherit">在指令和回答方面缺乏多样性和深入推理</span>。

为缓解这一问题，**LLaVA&#x20;**&#x5229;用仅支持文本输入的 GPT 作为强大的教师模型，以创建涉及视觉内容的指令跟随数据。具体来说，为了将图像编码为其视觉特征以提示纯文本的 GPT，使用两种符号表示：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">标题</span>**&#x901A;常从不同角度描述视觉场景
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">边界框</span>**&#x901A;常定位场景中的对象，每个框编码了对象概念及其空间位置。

![](../../images/视觉多模态讲义（上）-image-126.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：
>
> * 拿到图像数据集后，可以拿到图像的标注，语言模型可以以这些为基础扩充数据集，生成想要的数据格式
>
> * Caption & Bounding Box + 人工设计的几个例子，让语言模型根据这几个样本，不断地生成数据集，从而训练自己的模型

这种符号表示能够将图像编码为大语言模型可识别的序列。**LLaVA&#x20;**&#x4F7F;&#x7528;**`COCO`**&#x56FE;像并生成三种类型的指令跟随数据。对于每种类型，首先手动设计一些示例。这些是数据收集中唯一的人工标注，并用作上下文学习中的种子示例来查询 GPT-4。

> * **<span style="color: rgb(36,91,219); background-color: inherit">对话</span>**：**<span style="color: rgb(100,37,208); background-color: inherit">LLaVA </span>**<span style="color: rgb(100,37,208); background-color: inherit">设计了一个助理与一个人之间关于这张照片提问的对话</span>。回答的语气按照助理正在看图像并回答问题。提出的问题涵盖了图像视觉内容的多样性，<span style="color: rgb(100,37,208); background-color: inherit">包括物体类型、物体计数、物体动作、物置、物体之间的相对位置等</span>。只有具有明确答案的问题被考虑。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">详细描述</span>**：为了包含丰富而全面的图像描述，**<span style="color: rgb(100,37,208); background-color: inherit">LLaVA </span>**<span style="color: rgb(100,37,208); background-color: inherit">创建了一个带有此意图的问题列表。并提示 GPT-4 扩展该列表</span>。对于每张图像，随机从列表中抽取一个问题，要求 GPT-4 生成详细描述。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">复杂推理</span>**：上述两种类型关注于视觉内容本身，在此基础上，进一步创建了深度推理问题。回答通常需要通过严格的逻辑逐步推理。

![Prompt 设计](../../images/视觉多模态讲义（上）-image-128.png)

![示例](../../images/视觉多模态讲义（上）-image-134.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：数据格式
>
> 1. 根据 Caption 和 Box 信息生成一些对话，一问一答的形式
>
> 2. 生成更详细的对图像的总结，问题列表如下
>
> 3. 结合前两种问题，提出一些需要思考的问题

**<span style="color: rgb(36,91,219); background-color: inherit">图像问题列表</span>**

![](../../images/视觉多模态讲义（上）-image-125.png)

![](../../images/视觉多模态讲义（上）-image-133.png)

**LLaVA&#x20;**&#x603B;共收集&#x4E86;**`158K`**&#x4E2A;独特的语言-图像指令跟随样本，其中包&#x62EC;**`58K`**&#x4E2A;对话样本、**`23K`**&#x4E2A;详细描述样本&#x548C;**`77K`**&#x4E2A;复杂推理样本。实验中对比了 ChatGPT 和 GPT-4 的使用，发现 GPT-4 始终提供更高质量的指令跟随数据，例如空间推理。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

主要目标是有效利用预训练 LLM 和视觉模型的能力。网络架构如右图所示。**LLaVA&#x20;**&#x9009;&#x62E9;**`Vicuna`**&#x4F5C;为参数化为$$\phi$$的 LLM $$f_\phi(\cdot)$$，因为它在公开可用的 Checkpoint 中具有最佳的指令跟随能力。

![](../../images/视觉多模态讲义（上）-image-131.png)

对于输入图像$$X_v$$，使用预训练&#x7684;**`CLIP`**&#x89C6;觉编码&#x5668;**`ViT-L/14`**，它提供视觉特征$$Z_v = g(X_v)$$。**<span style="color: rgb(100,37,208); background-color: inherit">LLaVA </span>**<span style="color: rgb(100,37,208); background-color: inherit">使用一个简单的线性层将图像特征连接到词嵌入空间</span>。具体来说，应用一个可训练的投影矩阵$$W$$将$$Z_v$$转换为语言Embedding Token $$H_v$$，其维度与语言模型中的词嵌入空间相同：

$$H_v = W \cdot Z_v$$

$$Z_v = g(X_v)$$

因此得到了一系列视觉Token $$H_v$$。需要注意的是，这里简单投影方案非常轻量级，这能够快速进行以数据为中心的实验。当然也可以考虑更复杂的方案来连接图像和语言表示，例&#x5982;**`Flamingo`**&#x4E2D;的门控交叉注意力&#x548C;**`BLIP-2`**&#x4E2D;&#x7684;**`Q-former`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

对于每个图像$$X_v$$，生成多轮对话数据$$(X_q^1, X_a^1, \dots, X_q^T, X_a^T)$$，其中$$T$$是总轮数。**LLaVA&#x20;**&#x5C06;它们组织为一个序列，将所有答案视为助手的回答，并将第$$t$$轮的指令$$X_{\text{instruct}}^t$$定义为：

$$X_{\text{instruct}}^t =
\begin{cases}
\text{Randomly choose}[X_q^1, X_v] \text{or} [X_v, X_q^1], & \text{ first turn} \; t = 1 \\
X_q^t, & \text{remaining turns} \; t > 1
\end{cases}$$

这生成了右表中所示的多模态指令跟随序列的统一格式。在预测 Token 上对 LLM 进行指令微调，使用其原始的自回归训练目标。

![具体的，\<STOP> 设置为 ###。仅绿色的 Token 计算损失（训练 Assistant 回答）](../../images/视觉多模态讲义（上）-image-130.png)

具体来说，对于长度为$$L$$的序列，通过以下公式计算目标答案$$X_a$$的概率：

$$p(X_a | X_v, X_{\text{instruct}}) = \prod_{i=1}^L p_\theta({\color{green}x_i} | X_v, X_{\text{instruct},<i}, X_{a,<i})$$

其中$$\theta$$是可训练参数，$$X_{\text{instruct},<i}$$和$$X_{a,<i}$$分别是当前预测 Token $$x_i$$之前所有轮次的指令和答案 Token。对于公式中的条件分布，显式添加$$X_v$$以强调图像在所有答案中都被对齐。对于 **LLaVA** 模型训练，采用<span style="color: rgb(100,37,208); background-color: inherit">两阶段指令微调过程</span>：

1. **<span style="color: rgb(36,91,219); background-color: inherit">特征对齐的预训练</span>**

&#x4ECE;**`CC3M`**&#x4E2D;筛选&#x51FA;**`595K`**&#x4E2A;图像-文本对。这些数据使用上面生成的指令跟随数据。每个样本可以被视为单轮对话。为了构建输入$$X_{\text{instruct}}$$，对于图像$$X_v$$，随机采样一个问题$$X_q$$，这是一个语言指令，要求助手简要描述图像。真实预测答案$$X_a$$是原始标题。在训练过程中，<span style="color: rgb(100,37,208); background-color: inherit">冻结视觉编码器和 LLM 的权重，仅通过最大化上述公式的似然来更新可训练参数</span>$$\theta = W$$<span style="color: rgb(100,37,208); background-color: inherit">，即投影矩阵</span>。通过这种方式，图像特征$$H_v$$可以与预训练 LLM 的词嵌入对齐。这一阶段可以理解为为冻结的 LLM 训练一个兼容的视觉分词器。

* **<span style="color: rgb(36,91,219); background-color: inherit">端到端微调</span>**

始终<span style="color: rgb(100,37,208); background-color: inherit">冻结视觉编码器的权重，并继续更新 </span>**<span style="color: rgb(100,37,208); background-color: inherit">LLaVA </span>**<span style="color: rgb(100,37,208); background-color: inherit">中投影层和 LLM 的预训练权重</span>；即上述公式中的可训练参数为 $$\theta = \{W, \phi\}$$。这里使用两种特定的使用场景：

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态聊天机器人</span>**：通过构造&#x7684;**`158K`**&#x4E2A;语言-图像指令跟随数据进行微调来开发聊天机器人。在三种类型的回答中，对话是多轮的，而其他两种是单轮的。它们在训练中均匀采样。

* **<span style="color: rgb(36,91,219); background-color: inherit">科学问答</span>**：&#x5728;**`ScienceQA`**&#x57FA;准测试上测试 LLaVA，这是第一个大规模多模态科学问题数据集，其答案标注包含详细的讲解和解释。每个问题都提供自然语言或图像形式的上下文。助手以自然语言提供推理过程，并从多个选项中选择答案。这里将数据组织为单轮对话，问题和上下文作为$$X_{\text{instruct}}$$，推理和答案作为$$X_a$$。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**LLaVA&#x20;**&#x662F;一种创新的多模态大模型，结合了视觉编码器 ViT-L/14 和语言模型 Vicuna，能够高效处理图像与文本数据的联合理解与生成任务。通过端到端训练和指令微调技术，<span style="color: rgb(46,161,33); background-color: inherit">LLaVA 在图像描述、视觉问答及多模态对话等任务中表现出色，同时具备高效性与可扩展性</span>。其轻量化版&#x672C;**`LLaVA-Mini`**&#x8FDB;一步优化了性能，适用于高分辨率图像、视频理解及医学影像分析等多样化应用场景。

### 2.4.2 <span style="color: rgb(36,91,219); background-color: inherit">LLaVA-1.5</span>

LLaVA 系列模型由微软研究团队提出，旨在将视觉与语言的理解和生成能力结合在一起。最初的 <span style="color: rgb(46,161,33); background-color: inherit">LLaVA 通过整合视觉编码器和语言模型，展现了在图像描述、视觉问答等任务上的卓越性能</span>。然而，随着多模态任务需求的不断增长，研究人员意识到需要进一步提升模型的效率与性能。于是，在 2023 年 10 月，**LLaVA-1.5** 被提出，核心目标是通过简单的架构调整和更高效的数据利用，达到甚至超越现有最先进模型的效果。它不仅在多个基准测试中刷新了记录，还以极小的数据量实现了令人瞩目的性能提升。

作为视觉指令调优的开创性工作，<span style="color: rgb(46,161,33); background-color: inherit">LLaVA 展现了在视觉推理能力上的卓越表现，甚至在多样化的基准测试中超越了许多近期模型，特别是在现实生活中涉及视觉指令跟随的任务上</span>。<span style="color: rgb(100,37,208); background-color: inherit">LLaVA 使用一个线性层将视觉特征映射到语言空间，并对整个 LLM 进行优化以实现视觉指令调优</span>。然而，<span style="color: rgb(216,57,49); background-color: inherit">LLaVA 在学术基准测试中表现欠佳，这些测试通常需要简短的回答</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">单个词</span>，并且由于训练数据分布中缺乏此类数据，它倾向于在是非问题中回答`是`。

另一方面，**`InstructBLIP`**&#x662F;首个结合面向学术任务的数据&#x96C6;**`VQA-v2`**&#x548C;**`LLaVA`**&#x7684;模型，并&#x5728;**`VQA`**&#x57FA;准测试中表现出性能提升。它&#x5728;**`129M`**&#x5F20;图像-文本对上预训练&#x4E86;**`Q-Former`**，并仅对指令感知型 **Q-Former** 进行微调以实现视觉指令调优。然而，最近的研究表明，<span style="color: rgb(216,57,49); background-color: inherit">它在参与现实生活中的视觉对话任务时不如 LLaVA 表现良好。具体来说，即使面对需要详细回答的请求，它也可能过度拟合到具有简短回答的 VQA 训练集</span>。如右图

![](../../images/视觉多模态讲义（上）-image-124.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Prompt 格式</span>**

像 InstructBLIP 这样的方法无法在长篇和短篇视觉问答 VQA 之间取得平衡，主要是由于以下原因：

> 1. 回答格式的模糊提示。例如，**`Q: {Question} A: {Answer}`**。<span style="color: rgb(216,57,49); background-color: inherit">这样的提示没有明确指出所需的输出格式，可能导致大语言模型行为上过度适配为短篇回答，即使是在自然的视觉对话中也是如此</span>。
>
> 2. 未对大语言模型进行微调。这一问题因 InstructBLIP 仅对 Q-Forme&#x72;**&#x20;**&#x8FDB;行指令调优而加剧。它要求 Q-Forme&#x72;**&#x20;**&#x7684;视觉输出 Token 控制大语言模型输出的长度，使其可以是长篇或短篇，类似于前缀调优，但 <span style="color: rgb(216,57,49); background-color: inherit">Q-Former</span>**<span style="color: rgb(216,57,49); background-color: inherit"> </span>**<span style="color: rgb(216,57,49); background-color: inherit">可能由于其容量有限而无法正确执行此操作，尤其是在与 LLaMA 等大语言模型相比时</span>。

为了使 LLaVA 能够更好地处理短篇回答，同时解决 InstructBLIP 的问题，**LLaVA-1.5** 使用单一的回答格式提示，该提示清楚地表明输出格式。<span style="color: rgb(100,37,208); background-color: inherit">当鼓励短篇回答时，该提示会附加在 VQA 问题的末尾：用一个单词或短语回答问题</span>。当大语言模型使用此类提示进行微调时，LLaVA 能够根据用户的指令适当调整输出格式，如右图，并且<span style="color: rgb(46,161,33); background-color: inherit">不需要使用 ChatGPT 对 VQA 回答进行额外处理，这进一步使得扩展到各种数据源成为可能</span>。这个改进仅通过在训练中加入 VQAv2，<span style="color: rgb(46,161,33); background-color: inherit">LLaVA 在 MME 上的表现显著提高，</span>**`1323.8`**<span style="color: rgb(46,161,33); background-color: inherit">对</span>**`809.6`**<span style="color: rgb(46,161,33); background-color: inherit">，并以</span>**`111`**<span style="color: rgb(46,161,33); background-color: inherit">分的优势超越了 InstructBLIP</span>。

![](../../images/视觉多模态讲义（上）-image-123.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">数据模型扩展</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">MLP vision-language connector</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

受到自监督学&#x4E60;**`SimCLR v2`**&#x4E2D;通过将线性投影替换为多层感知机 MLP 而提升性能的启发，作者发现使用两层 MLP 增强视觉-语言连接器的表征能力，相较于原始的线性投影，可以显著提升 LLaVA 的多模态能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">面向学术任务的数据</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

![](../../images/视觉多模态讲义（上）-image-132.png)

进一步<span style="color: rgb(100,37,208); background-color: inherit">引入了更多面向学术任务的 VQA 数据集，包括视觉问答 VQA、光学字符识别 OCR 以及区域级感知任务</span>，以多种方式增强模型的能力。首先，加入了 InstructBLIP 中使用的四个额外数据集：

> 开放知识型 **<span style="color: rgb(36,91,219); background-color: inherit">VQA</span>**：**`OKVQA`**&#x548C;**`A-OKVQA`**
>
> **<span style="color: rgb(36,91,219); background-color: inherit">OCR</span>**：**`OCRVQA`**&#x548C;**`TextCaps`**

A-OKVQA 被转换为多项选择题，并使用特定的回答格式提示：直接从给定选项中回答选项字母。即使仅使用 InstructBLIP 使用的部分数据集，<span style="color: rgb(46,161,33); background-color: inherit">LLaVA 在三项任务中均超越了 InstructBLIP</span>，这表明 LLaVA 设计的有效性。此外，<span style="color: rgb(46,161,33); background-color: inherit">加入区域级 VQA 数据集能够提升模型对细粒度视觉细节的定位能力</span>，&#x5982;**`Visual Genome`**&#x548C;**`RefCOCO`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">进一步扩展</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

进一步将输入图像分辨率提升&#x81F3;**`336²`**，使大语言模型能够更清晰地看到图像的细节。为此，将视觉编码器替换为 **`CLIP-ViT-L-336px`**。此外，添加了 GQA 数据集作为额外的视觉知识来源，并引入 ShareGPT 数据，同时将大语言模型扩展&#x81F3;**`13B`**&#x53C2;数规模。**`MM-Vet`**&#x7684;结果显示，在将大语言模型扩展至 **13B** 时，性能提升最为显著，这表明基础大语言模型的能力对于视觉对话的重要性。由于输入图像分辨率提升至 **336²**，LLaVA-1.5 的训练时间约为 LLaVA 的两倍：大约需&#x8981;**`6`**&#x5C0F;时进行预训练，**`20`**&#x5C0F;时进行视觉指令调优，使&#x7528;**`8`**&#x5757;**`A100`** GPU。

* **<span style="color: rgb(36,91,219); background-color: inherit">扩展到更高分辨率</span>**

由上一节可知提升输入图像分辨率能够显著增强模型的能力。然而，<span style="color: rgb(216,57,49); background-color: inherit">现有的开源 CLIP 视觉编码器的图像分辨率限制在</span>**`336²`**<span style="color: rgb(216,57,49); background-color: inherit">，无法通过简单替换视觉编码器支持更高分辨率的图像</span>。

当使用 ViT 作为视觉编码器时，为了扩展分辨率，以往的方法通常选择进行位置嵌入插值，并在微调过程中调整 ViT 主干网络以适应新的分辨率。然而，<span style="color: rgb(216,57,49); background-color: inherit">这种方法通常需要模型在大规模图像-文本配对数据集上进行微调，并且会将推理时模型可接受的图像分辨率限制为固定大小</span>。

![](../../images/视觉多模态讲义（上）-image-122.png)

相比之下，如上图所示，<span style="color: rgb(100,37,208); background-color: inherit">LLaVA-1.5 通过将图像分割为更小的图像块来克服这一限制，这些图像块的分辨率与视觉编码器最初训练的分辨率一致，并独立对其进行编码</span>。在获得各个图像块的特征图后，将它们合并为一个目标分辨率的大特征图，并将其输入大语言模型。<span style="color: rgb(100,37,208); background-color: inherit">为了向大语言模型提供全局上下文信息并减少分割-编码-合并操作带来的伪影，还额外将下采样图像的特征与合并后的特征图拼接在一起</span>。这种方法可以将输入扩展到任意分辨率，同时保持 LLaVA-1.5 的数据效率。这一新模型命名&#x4E3A;**`LLaVA-1.5-HD`**。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

LLaVA 在视觉推理任务中表现出色，超越了许多现有方法，但<span style="color: rgb(216,57,49); background-color: inherit">在需要简短回答（如单个词）的学术基准测试中表现欠佳</span>。这主要是因为 LLaVA 没有在大规模数据上进行预训练，而其他方法通常通过大量数据预训练来提升性能。为了解决这一问题，<span style="color: rgb(46,161,33); background-color: inherit">LLaVA-1.5 通过改进 Vision-Language 连接器、增加多层感知机 MLP 以及优化回答格式等技术调整，显著提升了性能</span>。它以极简的架构和高效的数据利用，<span style="color: rgb(46,161,33); background-color: inherit">在 11 个基准测试中达到最先进的水平，支持高分辨率图像输入，并在视觉问答、图像描述等任务中表现出色。此外，LLaVA-1.5 开源且训练效率高，仅需约一天即可完成训练</span>。

### 2.4.3 <span style="color: rgb(36,91,219); background-color: inherit">LLaVA-NeXT</span>

LLaVA 系列模型自问世以来，便以其强大的语言与视觉跨模态理解能力吸引了广泛关注。然而，早期版本在处理高分辨率图像、视频序列以及大规模多模态数据时，仍存在一定的局限性。为了解决这些问题，2024 年 1 月，**LLaVA-NeXT** 应运而生，旨在通过技术创新进一步提升模型的性能和适用性。**<span style="color: rgb(46,161,33); background-color: inherit">LLaVA-NeXT</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 的推理、OCR 和世界知识功能均有所改进</span>，并在多个基准测试中超越了 Gemini Pro。

与 LLLaVA-1.5相比，**LLaVA-NeXT** 有几个改进：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">将输入图像分辨率提高</span>**&#x5230; 4 倍像素。这使其能够掌握更多视觉细节。它支持三种宽高比，最高分辨率&#x4E3A;**`672x672`**、**`336x1344`**、**`1344x336`**。
>
> 2. 通过改进的视觉指令调整数据混合，**<span style="color: rgb(36,91,219); background-color: inherit">实现更好的视觉推理和 OCR 能力</span>。**
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">更好的视觉对话，适用于更多场景</span>**，涵盖不同的应用。更好的世界知识和逻辑推理。
>
> 4. 使用SGLang进&#x884C;**<span style="color: rgb(36,91,219); background-color: inherit">高效部署和推理</span>**。

除了性能改进外，<span style="color: rgb(100,37,208); background-color: inherit">LLaVA-NeXT 还保留了 LLaVA-1.5 的简约设计和数据效率</span>。它重新使用了 LLaVA-1.5 的预训练连接器，并且仍然使用&#x4E0D;**`1M`** 个视觉指令微调样本。最大&#x7684;**`34B`**&#x53D8;体在&#x7EA6;**`1`**&#x5929;内使&#x7528;**`32`**&#x4E2A;**`A100`**&#x5B8C;成训练。

**LLaVA-NeXT** 有以下几个亮点:

> * **<span style="color: rgb(36,91,219); background-color: inherit">SoTA 性能</span>**：**LLaVA-NeXT 与`CogVLM`**&#x6216;**`Yi-VL`**&#x7B49;开源多模态大模型相比实现了最佳性能。与商业产品相比，它在选定的基准测试中赶上&#x4E86;**`Gemini Pro`**&#x5E76;优&#x4E8E;**`Qwen-VL-Plus`**。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Zero-Shot 中文能力</span>**：**LLaVA-NeXT** 的 Zero-Shot 中文能力显著增强。其在中文多模态场景上的表现很好，例&#x5982;**`MMBench-CN`**<span style="color: rgb(220,155,4); background-color: inherit">上的 SoTA</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">训练成本低</span>**：**LLaVA-NeXT** 使用 **32** 个 GPU 进行训练，耗时约 **1** 天，总共&#x6709;**`1.3M`**&#x4E2A;数据样本。训练成本比其他方法&#x4F4E;**`100-1000`**&#x500D;。

* **<span style="color: rgb(36,91,219); background-color: inherit">技术改进</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">动态高分辨率</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">LLaVA-NeXT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 以高分辨率设计模型，旨在</span>**<span style="color: rgb(100,37,208); background-color: inherit">保持其数据效率</span>**。当提供高分辨率图像和保留这些细节的表示时，模型感知图像中复杂细节的能力会显著提高。它减少了模型在面对低分辨率图像时猜测想象的视觉内容的幻觉。其设计&#x7684;**`AnyRes`**&#x6280;术旨在适应各种高分辨率的图像。并且采用网格配&#x7F6E;**`{2×2,1×{2,3,4},{2,3,4}×1}`**&#x5E73;衡性能效率和成本。

![](../../images/视觉多模态讲义（上）-COgPbtsRZo86b8xIAj5c5H3SnBe.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">数据混合</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">高质量用户指令数据</span>**：对高质量视觉指令遵循数据的定义取决于两个主要标准：

> 1. 任务指令的多样性，确保充分代表在现实场景中可能遇到的广泛用户意图，特别是在模型的部署阶段。
>
> 2. 响应的优越性至关重要，目的是征求有利的用户反馈。

为此，**LLaVA-NeXT** 使用了两个数据源：

> 1. 现有的 GPT-V 数据。**`LAION-GPT-V`**&#x548C;**`ShareGPT-4V`**
>
> 2. 为了进一步促进更多场景的更好的视觉对话，收集了一个涵盖不同应用程序的小&#x578B;**`15K`**&#x89C6;觉指令调整数据集。指令和图像来自 LLaVA 收集的真实用户的请求。并且仔细过滤可能存在隐私问题或潜在有害的样本，并使用 GPT-4V 生成回答。

**<span style="color: rgb(36,91,219); background-color: inherit">多模态文档/图表数据</span>**：

> 1. 从训练数据中删除&#x4E86;**`TextCaps`** ，因为 **TextCaps&#x20;**&#x4F7F;用&#x4E0E;**`TextVQA`**&#x76F8;同的训练图像集。这可以在评估 **TextVQA&#x20;**&#x65F6;更好地了解 Zero-Shot OCR 能力。为了保持并进一步提高模型的 OCR 能力，&#x7528;**`DocVQA`**&#x548C;**`SynDog-EN`**&#x66FF;换了 **TextCaps**。
>
> 2. &#x53D7;**`Qwen-VL-7B-Chat`**&#x7684;启发，进一步添加&#x4E86;**`ChartQA`**、**`DVQA`**&#x548C;**`AI2D`**，以便更好地理解图表和示意图。

* **<span style="color: rgb(36,91,219); background-color: inherit">扩展 LLM 主干</span>**

除&#x4E86;**`Vicuna-7B-1.5`**&#x548C;**`Vicuna-13B-1.5`**，使用了更多 LLM，包&#x62EC;**`Mistral-7B`**&#x548C;**`Nous-Hermes-2-Yi-34B`**。这些 LLM 具有良好的性能、灵活的商业使用条件、强大的双语支持和更大的语言模型容量。它使 LLaVA 能够支持社区中更广泛的用户和更多场景。LLaVA 可与各种 LLM 配合良好，并可顺利扩展&#x5230;**`34B`**&#x7684; LLM。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**LLaVA-NeXT** 是先进的大型多模态模型，采用经济高效的训练方法利用开放资源开发而成。它使用当时领先的 LLM Yi-34B 增强了多模态能力中的推理、OCR 和世界知识。**<span style="color: rgb(46,161,33); background-color: inherit">LLaVA-NeXT</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 在各种多模态理解任务中表现出色，甚至在 MMMU 和 MathVista 等基准测试中超越了 Gemini-Pro</span>。

***

在 LLaVA-NeXT 发布 4 个月的时间内，具有更强大语言能力的开源 LLM 陆续的出现，例&#x5982;**`LLaMA3`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`Qwen-1.5`**<span style="color: rgb(220,155,4); background-color: inherit">系列</span>。同时，OpenAI GPT-V 等专有 LMM 得到了 GPT-4 等更强大的 LLM 的支持。因此，在 2024 年 5 月，**LLaVA-NeXT** 原作者利用最近更强大的开放式 LLM 扩展了 LLaVA-NeXT，并有以下规律：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">通过更强大、更大的语言模型增强多模态能力</span>**：这使得 LMM 能够呈现从 LLM 继承的更好的视觉世界知识和逻辑推理。这包&#x62EC;**`LLaMA3-8B`**&#x548C;**`Qwen-1.5-72B`**&#x548C;**`Qwen-1.5-110B`**
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">更好的视觉聊天适用于更多现实场景，涵盖不同的应用</span>**：为了在野外评估改进的多模态能力，**LLaVA-NeXT** 收集并开发了新的评估数据&#x96C6;**`LLaVA-Bench (Wilder)`**，用于研究日常生活中的视觉聊天并扩大数据量以进行全面评估。

为了清楚地突出 LLM 在增强多模态性能改进方面的影响，LLaVA-NeXT 重复使用相同的训练方法，从而保持了 LLaVA 系列的简约设计和数据效率。最大&#x7684;**`110B`**&#x7248;本&#x5728;**`18`**&#x5C0F;时内使&#x7528;**`128`**&#x4E2A;**`H800`**&#x5B8C;成训练。

新版 **LLaVA-NeXT&#x20;**&#x7684;亮点：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">SoTA 性能</span>**：**LLaVA-NeXT&#x20;**<span style="color: rgb(100,37,208); background-color: inherit">只需增加 LLM 功能，就能与之前的开源 LMM 相比实现持续更好的性能</span>。它在选定的基准上赶上了 GPT4-V。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">低训练成本</span>**：<span style="color: rgb(100,37,208); background-color: inherit">保持了与之前的 LLaVA 模型一样高效的训练策略</span>。在与旧版 **LLaVA-NeXT `7B`**/**`13B`**/**`34B`** 模型相同的数据上进行了监督微调。<span style="color: rgb(46,161,33); background-color: inherit">目前最大的模型</span>**`LLaVA-NeXT-110B`**<span style="color: rgb(46,161,33); background-color: inherit">在 128 个 H800-80G 上训练了 </span>**<span style="color: rgb(46,161,33); background-color: inherit">18 </span>**<span style="color: rgb(46,161,33); background-color: inherit">个小时</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">能力探索</span>**

在对 **LLaVA-NeXT** 的探索中，作者见证了将 LLM 从 13B 扩展到 34B 时显著的性能飞跃。随着更强大的 LLM 的出现，一个自然的想法就是突破多模态性能的界限，为了衡量 LLM 的语言能力，作者采用了大规模多任务语言理&#x89E3;**`MMLU`**&#x57FA;准的评估分数。为了衡量应用相同 **LLaVA-NeXT** 训练方案后的多模态能力，检查了四个关键基准：用于多学科理解&#x7684;**`MMMU`**、用于视觉数学推理&#x7684;**`Mathvista`**、用于科学图表理解&#x7684;**`AI2D`**&#x548C;用于日常视觉聊天场景&#x7684;**`LLaVA-W`**。这些基准涵盖了 LMM 在现实世界中的各种应用。

右图描绘了多模态和语言能力之间的相关性，利用回归线来说明每个基准的趋势。

1. **<span style="color: rgb(36,91,219); background-color: inherit">提高语言能力</span>**：在规模相当的 LLM 中，&#x5982;**`7B Mistral`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`7B Vicuna`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`7B Qwen`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`8B LLaMA3`**，存在一个一致的模式：<span style="color: rgb(100,37,208); background-color: inherit">以 </span>**<span style="color: rgb(100,37,208); background-color: inherit">MMMU </span>**<span style="color: rgb(100,37,208); background-color: inherit">分数衡量的更高语言能力对应于改进的多模式能力</span>。

2. **<span style="color: rgb(36,91,219); background-color: inherit">模型大小的影响</span>**：在同一个 LLM 系列中，如 **<span style="color: rgb(220,155,4); background-color: inherit">Qwen LLM</span>**<span style="color: rgb(220,155,4); background-color: inherit">：</span>**`7B`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`72B`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`110B`**，较大的模型在多模态基准上始终表现出优异的性能。这说明<span style="color: rgb(100,37,208); background-color: inherit">较大的模型往往具有增强的语言能力，从而提高了多模态任务的性能</span>。

![](../../images/视觉多模态讲义（上）-image-129.png)

在上述两种分析中，更强大的 LLM 可能会产生更出色的多模态能力。这种现象可以归因于<span style="color: rgb(100,37,208); background-color: inherit">更强大的 LLM 通常具有的更广泛的世界知识、强大的逻辑推理和对话能力</span>。通过应用 **LLaVA-NeXT** 的轻量级训练，这些语言能力得到了很好的维持，并可以在视觉语言领域中迁移，这可能是因为跨模态概念的一致性以及在视觉指令微调中与人类意图的一致性。

* **<span style="color: rgb(36,91,219); background-color: inherit">LLaVA-Bench (Wilder)</span>**

开发 LLM 的最终目标之一是构建通用助手，帮助人类完成日常生活中的各种多模式任务。因此，拥有强大的基准来精确衡量相关进展非常重要。**`LLaVA-Bench (In-the-Wild)`**，也称&#x4E3A;**`LLaVA-W`**，用于衡量 LMM 的日常生活视觉聊天能力。然而，<span style="color: rgb(216,57,49); background-color: inherit">由于只有</span>**`60`**<span style="color: rgb(216,57,49); background-color: inherit">个示例可用，因此需要更广泛的数据集</span>。为了改进这一缺点，作者推出了 **`LLaVA-Bench (Wilder)`**，它包含两个版本：

> 一个较小的版本，包&#x542B;**`120`**&#x4E2A;示例，用于快速评估
>
> 一个中等大小的版本，包&#x542B;**`1020`**&#x4E2A;示例，用于全面测量

这些数据集涵盖了各种场景，例如<span style="color: rgb(220,155,4); background-color: inherit">数学问题解决、图像理解、代码生成、视觉 AI 辅助和基于图像的推理</span>。为了构建这些数据集，从在线服务中收集了反映真实用户请求的指令和图像。随后，通过精心筛选样本，以解决隐私问题并减轻潜在危害，这些提示的回答是使用 GPT4-V 生成的。

1. **<span style="color: rgb(36,91,219); background-color: inherit">与其他基准的比较</span>**

右图展示了 **LLaVA-Bench (Wider)** 与现有 LMM 评估基准的视觉比较。<span style="color: rgb(100,37,208); background-color: inherit">许多当前基准都采用固定形式的问答 QA 格式，因为它易于评估指标和呈现模型比较</span>。为了反映这一趋势，**MMMU**、**Mathvista&#x20;**&#x548C; **AI2D&#x20;**&#x7B49;基准经过量身定制，以评估 LMM 在特定知识密集型领域的性能。相比之下，**`RealWorldQA`**&#x4E13;注于日常场景，但仅限于简答格式。然而，作为助手模型，拥有让用户参与自由形式对话的能力至关重要。

![](../../images/视觉多模态讲义（上）-image-121.png)

因此，在日常生活中的视觉聊天场景中加入自由形式的对话变得至关重要。 **<span style="color: rgb(100,37,208); background-color: inherit">LLaVA-W</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 率先引入了这样的基准原型，而 </span>**<span style="color: rgb(100,37,208); background-color: inherit">LLaVA-Bench-Wilder</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 则致力于在此基准的基础上，纳入更多的日常生活场景，涵盖不同的应用</span>。

2. **<span style="color: rgb(36,91,219); background-color: inherit">构建和评估指标</span>**

对于来自在线服务的大量查询，**LLaVA-NeXT&#x20;**&#x4F7F;&#x7528;**`ONE-PEACE`**&#x5D4C;入模型来生成 Embedding。接下来应用加&#x6743;**`K-Means`**&#x805A;类，<span style="color: rgb(100,37,208); background-color: inherit">使用图像的最小-最大归一化总像素值作为权重，确保像素值较高的图像更有可能包含在测试集中</span>。删除重复项后，得到了一个包含 **120&#x20;**&#x4E2A;问题的小版本和一个包含 **1020&#x20;**&#x4E2A;问题的中等版本。此外还<span style="color: rgb(100,37,208); background-color: inherit">进行了净化检查，以确保数据集干净且未受污染</span>。两个版本的图像重叠率均小&#x4E8E;**`2%`**。相比之下，原始 **LLaVA-W** 的图像重叠率&#x4E3A;**`5%`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">参考答案构建</span>**

对于每个筛选的问题，首先使用 GPT4V 生成参考答案，并让人工注释者手动验证问题和参考答案的准确性。<span style="color: rgb(216,57,49); background-color: inherit">相当一部分用户的询问比较模糊，涉及图像分辨率、语法错误或与上传的图像无关的问题</span>。在这些情况下，GPT4V 可能会拒绝回应，或者提供的参考答案可能不正确。为了保持评估数据的质量，<span style="color: rgb(46,161,33); background-color: inherit">手动审查和修改了有问题的答案，确保准确性和可靠性</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">评分方法</span>**

**LLaVA-Bench (Wilder)&#x20;**&#x91C7;用了与 LLaVA-W 相同的评估流程，但用 GPT4-V 代替了 GPT-4。而且没有像 LLaVA-W 那样使用多个类别，而是简单地计算了 GPT4-V 参考答案与模型响应之间的总体得分比率。在作者的评估中，注意到<span style="color: rgb(216,57,49); background-color: inherit">分数不能很好地显示不同模型中的问题，并且可能会不公平地降低参考答案的分数，导致模型的坏情况无法正确反映在总体得分中</span>。为了解决这个问题，作者<span style="color: rgb(46,161,33); background-color: inherit">让 GPT4-V 总是认为正确的答案是完美的，并给它们十分。这意味着其他模型的分数更低，并因其错误受到更多的惩罚。这有助于更好地评估模型在现实生活中的能力</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

LLaVA-NeXT 是一个大型多模态模型，展示了在不同视觉模态上的强大联合训练能力。它<span style="color: rgb(46,161,33); background-color: inherit">引入了动态高分辨率技术支持和高效的特征提取方法，显著提升了对高清及复杂数据的处理性能</span>。此外，LLaVA-NeXT 在指令微调阶段采用了自动化生成的多模态样本，增强了上下文学习能力，并在多项基准测试中表现出接近甚至超越商业模型的水平。

### 2.4.4 <span style="color: rgb(36,91,219); background-color: inherit">LLaVA-OneVision</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">极简视觉—语言主干</span>**

LLaVA-OneVision 延续 LLaVA 的三段式结构：大语言模型、视觉编码器与视觉投影器。具体实例使用 Qwen-2 作为语言模型 $$f_\phi(\cdot)$$，使用 SigLIP 作为视觉编码器 $$g_\psi(\cdot)$$，再用两层 MLP 投影器 $$p_\theta(\cdot)$$ 把视觉特征映射到语言模型的词嵌入空间。

对输入视觉信号 $$X_v$$，视觉编码器产生网格特征：

$$Z_v=g_\psi(X_v)$$

投影器把网格特征变成视觉 token 序列：

$$H_v=p_\theta(Z_v)$$

实现中同时考虑 SigLIP 最后一层 Transformer 之前与之后的网格特征。映射后的视觉 token 与语言指令 token 共同进入 Qwen-2，自回归预测目标答案。对长度为 $$L$$ 的答案序列 $$X_a$$：

$$p(X_a\mid X_v,X_q)=\prod_{i=1}^{L}p\!\left(x_i\mid X_v,X_{q,<i},X_{a,<i}\right)$$

视觉信号在每个答案 token 的条件中持续存在。单图场景的基本视觉单元是图像 crop，多图场景是序列中的单张图像，视频场景则是单帧；三者共用同一视觉编码器、投影器和语言模型。

下面给出当前模型实例及其可扩展到单图、多图和视频信号的通用结构。

![LLaVA-OneVision 统一处理单图、多图和视频的三段式架构](../../images/视觉多模态讲义（上）-llava-onevision-fig1-architecture.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Higher AnyRes：先保留分辨率，再控制 token</span>**

视觉表示由两个互相独立的尺度决定：像素空间分辨率和特征空间 token 数。高分辨率能保留 OCR、图表、细粒度物体等局部信息，但如果每个 crop 都完整保留 token，长宽比复杂或分辨率很高的图像会产生过长序列。LLaVA-OneVision 因此采用 <span style="color: rgb(100,37,208); background-color: inherit">Higher AnyRes 加双线性插值</span>：图像仍以较高分辨率切块编码，编码后的二维特征网格再插值到受控尺寸，最后展平为视觉 token。

对空间配置 $$(a,b)$$，图像被划分为 $$a\times b$$ 个 crop，同时保留一张缩放后的 base image。若每个视觉输入原本产生 $$T$$ 个 token，则总视觉 token 数为：

$$L=(a\times b+1)T$$

给定 token 阈值 $$\tau$$，当总长度超出预算时，每个 crop 的 token 数调整为：

$$T_{\mathrm{new}}=\begin{cases}\dfrac{\tau}{a\times b+1},&L>\tau\\T,&L\le\tau\end{cases}$$

系统预先定义一组可选空间配置 $$(a,b)$$，根据输入分辨率和长宽比选择满足要求且 crop 数最少的配置。<span style="color: rgb(100,37,208); background-color: inherit">这里不是在像素空间先强制压小整张图，而是在视觉编码之后压缩特征网格，因此能够以更少 token 保留更高分辨率的视觉细节</span>。

下图对比 Higher AnyRes 与原始 AnyRes：上半部分把高分辨率图像切块编码后再对特征做双线性插值；下半部分先缩放图像，细节在进入视觉编码器之前就已经丢失。

![Higher AnyRes 在视觉编码后压缩特征网格并保留高分辨率细节](../../images/视觉多模态讲义（上）-llava-onevision-fig2-higher-anyres.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">按场景分配视觉 token</span>**

三类视觉输入需要不同的 token 分配方式，但总预算被设计在相近数量级，使能力可以在单图、多图与视频之间迁移。SigLIP 对 **`384×384`** 输入产生 **`729`** 个视觉 token。

1. **<span style="color: rgb(36,91,219); background-color: inherit">单图</span>**

单图允许较大的 AnyRes 空间配置，尽量保持原始分辨率。base image 与多个高分辨率 crop 都进入视觉编码器，单张图可以使用较长序列。示例预算为 $$(1+9)\times729=7290$$ 个 token。单图训练数据数量多、质量高，长单图序列还可以模拟多帧序列的长度，为视频理解提供更平滑的能力迁移。

2. **<span style="color: rgb(36,91,219); background-color: inherit">多图</span>**

每张图只使用 base image 分辨率，不再做高分辨率多 crop，从而把计算预算留给图像数量。示例最多处理 **`12`** 张图，对应 $$12\times729=8748$$ 个 token。

3. **<span style="color: rgb(36,91,219); background-color: inherit">视频</span>**

每帧缩放到 base image 分辨率并独立编码，再对每帧特征网格做双线性插值以减少 token。示例把每帧压缩为 **`196`** 个 token，最多使用 **`32`** 帧，总计 $$32\times196=6272$$ 个 token。<span style="color: rgb(46,161,33); background-color: inherit">减少每帧 token 可以在固定预算下覆盖更长时间范围</span>。

三种场景的完整 token 配额如下。

![LLaVA-OneVision 为单图、多图和视频分配相近规模的视觉 token 预算](../../images/视觉多模态讲义（上）-llava-onevision-fig3-token-strategy.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">高质量知识数据</span>**

高质量知识学习阶段不继续盲目扩大低质量网页图文对，而是把预训练 LLM 与 ViT 已有知识作为起点，用经过生成和筛选的数据补充细节描述、文档 OCR、中文与纯语言能力。数据分为三类：

1. **<span style="color: rgb(36,91,219); background-color: inherit">重描述的细粒度图像数据</span>**

用 LLaVA-NeXT-34B 为 COCO118K、BLIP558K 和 CC3M 图像重新生成详细描述，合并得到约 **`350 万`** 样本。它把较短或噪声较多的网页文本替换为面向视觉细节的长描述。

2. **<span style="color: rgb(36,91,219); background-color: inherit">文档与 OCR 数据</span>**

使用 UReader 的 Text Reading 子集约 **`10 万`** 样本，并与 SynDOG 英文/中文合并成约 **`110 万`** 文档 OCR 数据。PDF 渲染提供了可扩展的文本—页面配对来源。

3. **<span style="color: rgb(36,91,219); background-color: inherit">中文与语言数据</span>**

使用 GPT-4V 为 ShareGPT4V 原始图像生成约 **`9.2 万`** 条中文详细描述，并加入约 **`14.3 万`** Evo-Instruct 纯语言样本，防止大量图像描述训练削弱通用语言理解。高质量知识数据中约 **`99.8%`** 是合成数据。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉指令数据的三级组织</span>**

视觉指令数据按“视觉输入—语言指令—语言Response”三级结构整理，而不是只按数据集名称拼接。

1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉输入</span>**

明确区分单图、多图和视频，保证采样比例与 token 配置能够和实际场景对应。

2. **<span style="color: rgb(36,91,219); background-color: inherit">语言指令</span>**

按 General QA、General OCR、Doc/Chart/Screen、Math Reasoning、Language 五类能力组织，用任务类别维持技能分布的平衡。

3. **<span style="color: rgb(36,91,219); background-color: inherit">语言Response</span>**

Response分为自由格式和固定格式。自由格式回答保留 GPT-4V、GPT-4o、Gemini 等模型的原始标注；固定格式学术数据需要统一检查问题和答案格式，多选、短答案、OCR 等任务沿用 LLaVA-1.5 的提示模板，避免不同数据源对回答行为施加冲突约束。

单图阶段汇总约 **`320 万`** 样本，覆盖 General、Doc/Chart/Screen、Math/Reasoning、General OCR 和 Language 五类能力。完整分布如下。

![LLaVA-OneVision 的 320 万单图指令数据类别与子集分布](../../images/视觉多模态讲义（上）-llava-onevision-fig4-single-image-data.png)

OneVision 阶段使用约 **`160 万`** 混合样本，其中约 **`56 万`** 为多图数据、**`35 万`** 为视频数据、**`80 万`** 为从单图集合中重新采样的高质量平衡子集。这个阶段不再引入新的单图数据，而是让已有单图能力向多图和视频场景迁移。

![OneVision 阶段 160 万单图、多图和视频混合数据分布](../../images/视觉多模态讲义（上）-llava-onevision-fig5-onevision-data.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">课程式四段训练</span>**

训练按难度和序列长度逐步增加，前一阶段的检查点直接作为下一阶段起点。

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：Language-Image Alignment</span>**
>
> 使用约 **`55.8 万`** LCS 图文样本，只训练投影器，把视觉特征对齐到语言模型词嵌入空间。输入分辨率为 **`384`**，视觉长度为 **`729`** token；投影器学习率为 $$10^{-3}$$。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1.5：High-Quality Knowledge Learning</span>**
>
> 使用约 **`400 万`** 高质量图像知识样本，开始更新完整模型。AnyRes 最大视觉长度提高到 $$729\times5$$。视觉编码器学习率为 $$2\times10^{-6}$$，投影器和 LLM 学习率为 $$10^{-5}$$。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2A：Single-Image Visual Instruction Tuning</span>**
>
> 先在 **`320 万`** 单图指令上训练完整模型，最大视觉长度提高到 $$729\times10$$。这一阶段优先建立强单图指令跟随、OCR、图表和视觉推理能力。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2B：OneVision Training</span>**
>
> 再用 **`160 万`** 单图、多图和视频混合数据继续训练完整模型。每类场景使用前述独立 token 策略，但最大视觉预算维持在相近范围。视觉编码器学习率仍比 LLM 与投影器小 **`5`** 倍，以较慢速度调整视觉表征，同时让语言模型学习跨场景的指令格式与推理行为。

**<span style="color: rgb(36,91,219); background-color: inherit">训练逻辑</span>**

<span style="color: rgb(100,37,208); background-color: inherit">先完成模态对齐，再注入高质量知识；先把单图能力做强，再用统一接口迁移到多图与视频。</span>分辨率、视觉 token 数、可训练模块和任务复杂度都随阶段递增，因此长序列训练不会在最早阶段一次性压到模型上。

## 2.5 <span style="color: rgb(36,91,219); background-color: inherit">MiniGPT 系列</span>

### 2.5.1 <span style="color: rgb(36,91,219); background-color: inherit">MiniGPT-4</span>

传统的语言模型擅长处理纯文本数据，但当面对图像等多模态输入时，其局限性便显现出来。如何让模型同时理解视觉信息并生成连贯的语言描述，是跨模态研究的核心问题之一。**MiniGPT-4** 的出现正是为了解决这一难题。它基于两个关键组件构建：一个超大规模的语言模&#x578B;**`Vicuna`**&#x548C;一个高效&#x7684;**`BLIP-2`**&#x89C6;觉编码器。这种结合<span style="color: rgb(46,161,33); background-color: inherit">不仅充分利用了现有大语言模型的知识库，还通过视觉编码器提取丰富的图像特征，从而实现多模态任务的高效执行</span>。

> **`注`**：目标是**图像理解 + 文本生成**，要利用大模型的**涌现能力**，即最&#x5C11;**`50B`**&#x53C2;数

**MiniGPT-4** 的目标是将来自预训练视觉编码器的视觉信息与先进的 LLM 对齐。具体来说，使&#x7528;**`Vicuna`**&#x4F5C;为语言解码器，它基&#x4E8E;**`LLaMA`**&#x6784;建，能够执行多种复杂的语言任务。对于视觉感知，采用了&#x4E0E;**`BLIP-2`**&#x76F8;同的视觉编码器，&#x5373;**`ViT`**&#x4E3B;干网络，结合了其预训练&#x7684;**`Q-Former`**。语言和视觉模型都是开源的。**<span style="color: rgb(100,37,208); background-color: inherit">MiniGPT-4</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 通过一个线性投影层弥合视觉编码器与 LLM 之间的差距</span>，如右图所示。为了实现有效的 MiniGPT-4，作者提出了一种两阶段训练方法：



![](../../images/视觉多模态讲义（上）-image-144.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">第一阶段</span>**&#x6D89;及在大量对齐的图文对上预训练模型以获取视觉-语言知识。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">第二阶段</span>**&#x5219;使用更小但高质量的图文数据集，结合设计好的对话模板对预训练模型进行微调，以提高生成的可靠性和可用性。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

在初始预训练阶段，模型从大量的对齐图文对中学习视觉-语言知识。**<span style="color: rgb(100,37,208); background-color: inherit">MiniGPT-4</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 将注入的投影层输出视为 LLM 的软提示</span>，促使它生成相应的标准文本。<span style="color: rgb(100,37,208); background-color: inherit">在整个预训练过程中，预训练的视觉编码器和 LLM 始终保持冻结状态，仅对线性投影层进行预训练</span>。预训练使&#x7528;**`Conceptual Caption`**、**`SBU`**&#x548C;**`LAION`**&#x6570;据集来训练模型。模型经&#x8FC7;**`20,000`**&#x6B65;训练，批量大小&#x4E3A;**`256`**，覆盖了大&#x7EA6;**`5M`**&#x5F20;图文对。整个过程约&#x9700;**`10`**&#x5C0F;时，使&#x7528;**`4`**&#x5F20;**`80GB A100`** GPU。

尽管 **MiniGPT-4** 在第一阶段后展现出丰富的知识储备并能对人类提问给出合理回应，但它<span style="color: rgb(216,57,49); background-color: inherit">有时会产生不连贯的语言输出</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">重复的单词或句子、片段化的句子或无关内容</span>。这些问题阻碍了 **MiniGPT-4** 与人类进行流畅的视觉对话的能力。类似的问题也出现在 GPT-3 中。<span style="color: rgb(216,57,49); background-color: inherit">尽管 GPT-3 在大规模语言数据集上进行了预训练，但它仍然难以生成与用户意图完全一致的语言输出</span>。通过指令微调和基于人类反馈的强化学习，GPT-3 演化为 GPT-3.5，能够生成更符合人类需求的输出。这一现象&#x4E0E;**&#x20;MiniGPT-4&#x20;**&#x5F53;前的状态相似，因此在现阶段可能难以生成流畅自然的人类语言输出并不令人意外。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据集构建</span>**

为了提高生成语言的自然度并增强模型的可用性，第二阶段的对齐过程至关重要。在 NLP 领域，指令微调数据集和对话数据集较易获得，但<span style="color: rgb(216,57,49); background-color: inherit">在视觉-语言领域却没有类似的高质量数据集</span>。为了解决这一问题，<span style="color: rgb(100,37,208); background-color: inherit">作者精心策划了一个专门用于视觉-语言对齐的详细图像描述数据集，并在第二阶段对 MiniGPT-4 进行微调</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">初始对齐图文生成</span>**

在初始阶段，<span style="color: rgb(100,37,208); background-color: inherit">利用第一阶段预训练得到的模型生成输入图像的详细描述</span>。为了使模型生成更详细的图像描述，作者设计了一个符合 **Vicuna&#x20;**&#x5BF9;话格式的提示，如下所示。其&#x4E2D;**`<ImageFeature>`**&#x8868;示线性投影层生成的视觉特征：

**`###Human:`**` `**`<Img><ImageFeature></Img> `**`Describe this image in detail. Give as many details as possible. Say everything you see. `**`###Assistant:`**

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：设计了一个遵循 **Vicuna&#x20;**&#x8BED;言模型会话格式的 prompt，让模型生成尽可能多的描述

为了识别不完整的句子，检查生成的句子是否超&#x8FC7;**`80`**&#x4E2A; Token。如果未达到，会加入额外的提&#x793A;**`###Human: `**`Continue`**` ###Assistant:`**，促使 MiniGPT-4 扩展生成过程。通过合并两步的输出，可以生成更全面的图像描述。这种方法能够生成具有详细且信息丰富的图像描述的图文对。作者从 **Conceptual Caption** 数据集中随机选择&#x4E86;**`5,000`**&#x5F20;图像，并使用预训练模型为每张图像生成对应的描述。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据后处理</span>**

上述自动生成的图像描述中包含噪声或不连贯的内容，例如<span style="color: rgb(220,155,4); background-color: inherit">重复的单词或句子、片段化的句子或无关内容</span>。为了解决这些问题，<span style="color: rgb(100,37,208); background-color: inherit">使用 ChatGPT 修复这些描述</span>，采用以下提示：

`Fix the error in the given paragraph. Remove any repeating sentences, meaningless characters, not English sentences, and so on. Remove unnecessary repetition. Rewrite any incomplete sentences. Return directly the results without explanation. Return directly the input paragraph if it is already correct without explanation.`

完成后处理阶段后，<span style="color: rgb(100,37,208); background-color: inherit">手动验证每个图像描述的准确性，以确保其高质量</span>。具体来说，首先识别一些频繁出现的错误，如`I’m sorry I made a mistake...`<span style="color: rgb(220,155,4); background-color: inherit">或</span>`I apologize for that ...`，然后<span style="color: rgb(100,37,208); background-color: inherit">编写硬规则自动过滤掉它们</span>。除此之外还<span style="color: rgb(100,37,208); background-color: inherit">手动优化生成的标题，删除 ChatGPT 未能检测到的冗余单词或句子</span>。最终，仅有&#x7EA6;**`3,500`**&#x5BF9;图文满足要求，这些数据随后被用于第二阶段对齐过程。

* **<span style="color: rgb(36,91,219); background-color: inherit">微调</span>**

在第二阶段，使用精选的高质量图文对对预训练模型进行微调。在微调过程中，使用以下模板中的预定义提示：

**`###Human: <Img><ImageFeature></Img> `**`<Instruction> `**`###Assistant:`**

其&#x4E2D;**`<Instruction>`**&#x8868;示从预定义指令集中随机采样的指令，例如`Describe this image in detail`<span style="color: rgb(220,155,4); background-color: inherit">或</span>`Could you describe the contents of this image for me`。这里不对特定文本-图像提示计算回归损失。

经过微调，**MiniGPT-4** 能够生成更自然、更可靠的语言输出。此外，这一微调过程效率极高，仅&#x9700;**`400`**&#x6B65;训练，批量大小&#x4E3A;**`12`**，使用单&#x5F20;**`A100`** GPU 大约需&#x8981;**`7`**&#x5206;钟。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**MiniGPT-4** 是一种先进的多模态模型，<span style="color: rgb(100,37,208); background-color: inherit">结合了预训练的视觉编码器和大型语言模型，通过轻量化的线性投影层实现视觉与语言的对齐</span>。它能够生成高质量的图像描述、识别复杂视觉内容，并完成从手写文本生成网站等创造性任务 。相比以往的视觉-语言模型，**MiniGPT-4** 展现了更强的理解和生成能力。

### 2.5.2 <span style="color: rgb(36,91,219); background-color: inherit">MiniGPT-v2</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构：EVA ViT、线性投影与 LLaMA-2-chat</span>**

MiniGPT-v2 由冻结的 EVA 视觉主干、可训练线性投影层和 LLaMA-2-chat 7B 组成。图像以 **`448×448`** 分辨率输入，视觉主干的位置编码通过插值适配更高分辨率；视觉主干在全部训练阶段保持冻结。语言模型作为统一输出接口，视觉问答、描述、指代表达和定位结果都由 LLaMA-2 token 序列生成。

右图给出完整架构：视觉 token 经相邻合并和线性投影后，与任务标识和文本指令一起进入 LLaMA-2。

* **<span style="color: rgb(36,91,219); background-color: inherit">四邻视觉 token 合并</span>**

直接把 **`448×448`** 图像的全部视觉 token 投影到语言模型空间，会产生约 **`1024`** 个输入 token。为缩短序列，MiniGPT-v2 在嵌入空间中把每 **`4`** 个空间相邻视觉 token 沿特征维拼接，再通过一个线性层把拼接向量映射为一个 LLaMA-2 维度的视觉 token。

![MiniGPT-v2 的视觉 token 合并、线性投影与任务标识接口](../../images/视觉多模态讲义（上）-image-142.png)

若相邻 token 为 $$v_{i,1},v_{i,2},v_{i,3},v_{i,4}\in\mathbb{R}^{d_v}$$，拼接与投影可以写为：

$$h_i=W_p[v_{i,1};v_{i,2};v_{i,3};v_{i,4}]+b_p$$

输出视觉 token 数减少为原来的 **`1/4`**。<span style="color: rgb(100,37,208); background-color: inherit">这里压缩的是局部相邻 token，而不是先把输入图像整体降采样，因此仍可利用 448×448 输入中的细粒度目标和文字信息</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">多任务指令模板</span>**

仅把视觉 token 与自然语言问题对齐时，不同任务可能共享同一问法却要求不同输出格式。例如“红色外套的人在哪里”既可以回答自然语言方位，也可以输出边界框。MiniGPT-v2 在指令中加入显式任务标识，把“要解决什么任务”作为条件 token 输入。

通用输入格式为：

**`[INST]`** 表示用户角色，**`[/INST]`** 表示助手角色。用户输入由图像特征、任务标识和自然语言指令三部分组成。视觉无关的纯语言指令不添加任务标识。

六个任务标识与输出职责如下：

| 任务         | 标识            | 期望输出          |
| ---------- | ------------- | ------------- |
| 视觉问答 VQA   | `[vqa]`       | 问题答案          |
| 图像描述       | `[caption]`   | 普通图像描述        |
| 带定位描述      | `[grounding]` | 描述文本与对应边界框    |
| 指代表达理解 REC | `[refer]`     | 给定短语对应的边界框    |
| 指代表达生成 REG | `[identify]`  | 给定边界框对应的短语    |
| 目标解析与定位    | `[detection]` | 从文本解析目标并给出边界框 |

* **<span style="color: rgb(36,91,219); background-color: inherit">边界框的文本表示</span>**

定位任务不引入独立检测头，而是让 LLaMA-2 直接生成边界框的文本 token：

$$\{\langle X_{\mathrm{left}}\rangle\langle Y_{\mathrm{top}}\rangle\langle X_{\mathrm{right}}\rangle\langle Y_{\mathrm{bottom}}\rangle\}$$

$$X$$、$$Y$$ 坐标被归一化并离散到整数区间 **`[0,100]`**。前两个 token 表示左上角，后两个表示右下角。这样，描述、问答与视觉定位都可在同一自回归语言建模接口中训练和解码。

* **<span style="color: rgb(36,91,219); background-color: inherit">三阶段多任务训练</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：预训练</span>**

第一阶段混合弱标注和细粒度数据，并提高弱标注数据的采样率以扩大视觉—语言知识覆盖。弱标注来源包括 LAION、CC3M、SBU 以及用于 REC、REG 和带定位描述的 GRIT-20M。细粒度数据包括 COCO Caption、TextCaps、RefCOCO/RefCOCO+/RefCOCOg、Visual Genome，以及 GQA、VQAv2、OCR-VQA、OK-VQA、AOK-VQA。

REG 数据由 REC 数据反向构造：把“短语 → 边界框”改为“边界框 → 短语”，从而让同一数据同时训练理解与生成两个方向。

**<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：多任务训练</span>**

第二阶段移除 GRIT-20M、LAION 等弱监督数据，只保留对齐质量较高的图像描述、REC、REG 和 VQA 数据，并根据任务出现频率重新设置采样比例。目标是从广覆盖转向各任务的精细对齐。

**<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：多模态指令微调</span>**

第三阶段继续使用第二阶段的细粒度任务数据，但降低其采样比例，同时提高新加入的多模态指令与语言数据比例。新增数据包括 LLaVA 详细描述约 **`2.3 万`** 条、复杂推理约 **`5.8 万`** 条，以及 Flickr30k、多任务多轮对话和 Unnatural Instructions。

Flickr30k 被重组为两类监督：一类选择至少含 **`5`** 个 grounded phrase 的描述，训练带定位图像描述；另一类让模型从输入描述中解析全部目标，再用 `[detection]` 输出每个目标的边界框。多轮混合数据把不同单轮任务拼入同一对话，缓解模型只会在孤立轮次完成单一任务的问题；纯语言 Unnatural Instructions 则用于恢复长期视觉语言训练后可能下降的语言生成能力。

> **<span style="color: rgb(36,91,219); background-color: inherit">参数更新</span>**
>
> 视觉主干始终冻结；训练集中在视觉投影层和语言模型。语言模型通过 LoRA 调整注意力中的 $$W_q$$ 与 $$W_v$$，同时保留统一的任务模板和边界框 token 接口。

## 2.6 <span style="color: rgb(36,91,219); background-color: inherit">Qwen 系列</span>

### 2.6.1 <span style="color: rgb(36,91,219); background-color: inherit">Qwen-VL</span>

传统的单模态模型在处理复杂任务时存在局限性，尤其是在需要同时理解图像和文本的场景中。为了突破这一瓶颈，**Qwen-VL** 应运而生。作为 Qwe&#x6E;**&#x20;**&#x7CFB;列大模型的分支，**<span style="color: rgb(100,37,208); background-color: inherit">Qwen-VL</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 旨在通过融合视觉与语言能力，解决图像描述、OCR、文档理解和视觉问答等任务中的挑战</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：总共包&#x62EC;**`Qwen-VL`**&#x548C;**`Qwen-VL-Chat`**&#x4E24;个模型。
>
> 1. **Qwen-VL** 是预训练版本，通过连接 Vision Encoder 扩展了 LLM 的视觉能力。
>
> 2. **Qwen-VL-Chat&#x20;**&#x662F;交互式视觉语言模型，支持图像描述，视觉问答，视觉定位和灵活的交互能力。可以感知和理解多层次的视觉信号

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

Qwen-VL 的整体网络架构由三个组件组成，模型参数的详细信息如右表所示：

![](../../images/视觉多模态讲义（上）-image-139.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">LLM</span>**：Qwen-VL 使用 Qwen-7B 作为大模型核心组件，并用其预训练权重进行初始化。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">Vision Encoder</span>**：Qwen-VL 的视觉编码器基于 ViT 架构，并使用 OpenCLIP &#x7684;**`ViT-bigG`**&#x9884;训练权重初始化。在训练和推理过程中，<span style="color: rgb(100,37,208); background-color: inherit">输入图像会被调整为特定分辨率</span>。视觉编码器通过将图像分割为步长&#x4E3A;**`14`**&#x7684;小块，生成一组图像特征。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">位置感知的 VL Adapter</span>**：为了缓解长图像特征序列带来的效率问题，Qwen-VL 引入了一&#x4E2A;**`VL Adapter`**&#x6765;压缩图像特征。Adapter 包含一个<span style="color: rgb(100,37,208); background-color: inherit">随机初始化的单层交叉注意力模块，使用一组可训练向量作为 Query，并使用来自视觉编码器的图像特征作为 Key 和 Value 进行交叉注意力操作</span>。这种机制<span style="color: rgb(100,37,208); background-color: inherit">将视觉特征序列压缩到固定长度</span>**`256`**。由于位置信息对细粒度图像理解十分重要，因此<span style="color: rgb(100,37,208); background-color: inherit">在交叉注意力机制的 Query-Key 对中引入了</span>**`2D`<span style="color: rgb(100,37,208); background-color: inherit">绝对位置编码</span>**，以减少压缩过程中可能的位置细节损失。压缩后的长度为 **256&#x20;**&#x7684;图像特征序列随后被送入 LLM。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**<span style="color: rgb(36,91,219); background-color: inherit">VL Adapter </span>**&#x53EA;使用了<span style="color: rgb(100,37,208); background-color: inherit">单层交叉注意力模块</span>，&#x548C;**`InstructBLIP`**&#x76F8;比说明模型结构不是越复杂越好

* **<span style="color: rgb(36,91,219); background-color: inherit">模型输入输出</span>**

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">图像输入</span>**：图像通过 Visual Encoder 和 Adapter 处理后，生成固定长度的图像特征序列。<span style="color: rgb(100,37,208); background-color: inherit">为了区分图像特征输入和文本特征输入，在图像特征序列的开头和结尾分别添加两个特殊 Token：</span>**`<img>`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`</img>`**，表示图像内容的起始和结束。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">边界框输入输出</span>**：为了增强模型在细粒度视觉理解和定位方面的能力，**Qwen-VL** 的训练数据包括区域描述、问题和检测结果。与传统的图文描述或问答任务不同，**Qwen-VL** 要求模型能够准确理解和生成指定格式的区域描述，即 Box 坐标。对于任意给定的边界框，<span style="color: rgb(100,37,208); background-color: inherit">首先进行标准化处理，调整到</span>**`[0, 1000)`**<span style="color: rgb(100,37,208); background-color: inherit">的范围</span>，然后转换为指定的字符串格式：$$(X_\text{topleft}, Y_\text{topleft}),(X_\text{bottomright}, Y_\text{bottomright})$$。<span style="color: rgb(100,37,208); background-color: inherit">该字符串以文本形式进行分词，无需额外的位置词汇表。为了区分 Box 字符串和普通文本字符串，在 Box 字符串的开头和结尾分别添加两个特殊 Token：</span>**`<box>`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`</box>`**。此外，为了将边界框与其对应的描述性词语或句子正确关联，还<span style="color: rgb(100,37,208); background-color: inherit">引入了另一组特殊 Token：</span>**`<ref>`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`</ref>`**<span style="color: rgb(100,37,208); background-color: inherit">，用于标记边界框所引用的内容</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

Qwen-VL 模型的训练过程分为三个阶段：<span style="color: rgb(100,37,208); background-color: inherit">两个预训练阶段和一个最终的指令微调阶段</span>。

![](../../images/视觉多模态讲义（上）-image-137.png)

**<span style="color: rgb(36,91,219); background-color: inherit">Stage1：预训练</span>**

在第一阶段的预训练中，主要利用了一个<span style="color: rgb(100,37,208); background-color: inherit">大规模、弱标注、从网络爬取的图文对数据集</span>。**Qwen-VL** 的预训练数据集由多个公开来源和部分内部数据组成。并且对数据集中某些模式进行了清洗。如下表左所示，原始数据集包含总&#x8BA1;**`5B`**&#x4E2A;图文对，清洗后剩&#x4F59;**`1.4B`**&#x6761;数据，其&#x4E2D;**`77.3%`**&#x4E3A;英文数据，**`22.7%`**&#x4E3A;中文数据。

在此阶&#x6BB5;**<span style="color: rgb(100,37,208); background-color: inherit">冻结 LLM，仅优化 ViT 和 VL Adapter</span>**。输入图像被调整&#x4E3A;**`224 × 224`**&#x7684;分辨率。训练目标是最小化文本 Token 的交叉熵损失。优化器&#x4E3A;**`AdamW`**，余弦学习率，最大学习率&#x4E3A;**`2e−4`**，**`500`**&#x6B65;**`warm-up`**，训练过程使用了批量大小&#x4E3A;**`30720`**&#x7684;图文对数据。整个第一阶段的预训练持&#x7EED;**`50,000`**&#x6B65;，消耗了&#x7EA6;**`1.5B`**&#x4E2A;图文样本。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：此阶段将 Vision Encoder 和冻结的 LLM 对齐，使用大量低分辨率图像文本对，因为<span style="color: rgb(100,37,208); background-color: inherit">数据规模比较重要</span>
>
> 冻住 LLM 是为了<span style="color: rgb(100,37,208); background-color: inherit">保证 LLM 的理解和生成能力</span>，避免数据对 LLM 造成不利的影响

![Stage1预训练数据集](../../images/视觉多模态讲义（上）-image-141.png)

![Stage2多任务预训练数据集](../../images/视觉多模态讲义（上）-image-143.png)

**<span style="color: rgb(36,91,219); background-color: inherit">Stage2：多任务预训练</span>**

在第二阶段的多任务预训练中，<span style="color: rgb(100,37,208); background-color: inherit">引入了高质量且细粒度的视觉-语言标注数据，并使用更大的输入分辨率以及交错排列的图文数据</span>。如上表右所示，&#x5728;**`7`**&#x4E2A;任务上同时训练 **Qwen-VL**。

> * 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">文本生成任务</span>**，使用内部收集的语料库以保持大语言模型的能力。描述生成数据与预训练相同，但样本数量更少，且不包&#x62EC;**`LAION-COCO`**。
>
> * 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">视觉问答 VQA 任务</span>**，我们使用公开可用的数据集混合，包&#x62EC;**`GQA`**、**`VGQA`**、**`VQAv2`**、**`DVQA`**、**`OCRVQA`**&#x548C;**`DocVQA`**。
>
> * 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">定位任务</span>**，基&#x4E8E;**`GRIT`**&#x6570;据集进行轻微修改。
>
> * 对&#x4E8E;**<span style="color: rgb(36,91,219); background-color: inherit">参考定位和定位描述的双重任务</span>**，我们&#x4ECE;**`GRIT`**、**`Visual Genome`**、**`RefCOCO`**、**`RefCOCO+`**&#x548C;**`RefCOCOg`**&#x4E2D;构建训练样本。
>
> * 为了改&#x8FDB;**<span style="color: rgb(36,91,219); background-color: inherit">面向文本的任务</span>**，&#x4ECE;**`Common Crawl`**&#x4E2D;收集 PDF 和 HTML 格式数据，并生成具有自然风景背景的中英双语文本合成 OCR 数据。

最后，通过将相同任务的数据打包成长度&#x4E3A;**`2048`**&#x7684;序列，简单构造交错排列的图文数据。这一阶段使&#x7528;**`AdamW`**&#x4F18;化器，使用 <span style="color: rgb(100,37,208); background-color: inherit">ViT 和 LLM 的模型并行技巧</span>。共训&#x7EC3;**`19k`**&#x6B65;，余弦学习率，**`400`**&#x6B65;**`warm-up`**。

在此阶段 **Qwen-VL&#x20;**&#x5C06; ViT 的输入分辨率&#x4ECE;**`224 × 224`**&#x63D0;高&#x5230;**`448 × 448`**，以<span style="color: rgb(100,37,208); background-color: inherit">减少因图像下采样导致的信息损失</span>。这个阶段 **<span style="color: rgb(100,37,208); background-color: inherit">LLM 也加入训练，即对整个模型进行训练</span>**，训练目标与预训练阶段相同。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：此阶段赋予 **Qwen-VL** 多种下游任务的能力，包括视觉问答，图像描述等。<span style="color: rgb(100,37,208); background-color: inherit">由于需要细粒度的位置信息，图像分辨率变为 448x448，减少图像下采样带来的信息损失</span>，引入高质量细粒度图文标注数据。

**<span style="color: rgb(36,91,219); background-color: inherit">Stage3：监督微调</span>**

此阶段将 Qwen-VL 与人类偏好对齐，对 Qwen-VL 预训练模型进行指令微调，增强其指令跟随和对话能力，从而得到交互式的 Qwen-VL-Chat 模型。多模态指令微调<span style="color: rgb(100,37,208); background-color: inherit">数据主要来源于 Caption 数据或通过大语言模型自指令生成的对话数据</span>（LLM Self-Instruction方式生成），<span style="color: rgb(100,37,208); background-color: inherit">这些数据通常仅针对单图像对话和推理，并局限于图像内容理解</span>。作者通过<span style="color: rgb(100,37,208); background-color: inherit">人工标注、模型生成和策略拼接构建了一组额外的对话数据，以将定位能力和多图像理解能力融入 Qwen-VL 模型</span>。此外在训练过程中<span style="color: rgb(100,37,208); background-color: inherit">混合了多模态和纯文本对话数据，以确保模型在对话能力上的通用性</span>。指令微调数据量总计&#x4E3A;**`350k`**&#x6761;。

在此阶&#x6BB5;**<span style="color: rgb(100,37,208); background-color: inherit">冻结 ViT 并优化 LLM 和 VL Adapter</span>**。输入图像同样采用高分辨率图像，批量大小&#x4E3A;**`128`**，**`3000`**&#x6B65;**`warm-up`**。指令微调格式为 OpenAI 的 ChatML 格式，如右图

![](../../images/视觉多模态讲义（上）-image-138.png)

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

Qwen-VL 通过整合大型语言模型、视觉编码器和位置感知适配器，<span style="color: rgb(46,161,33); background-color: inherit">在多模态任务中展现了卓越性能</span>。其三阶段训练流程<span style="color: rgb(46,161,33); background-color: inherit">确保了模型的强大泛化能力，使其在高分辨率识别、文本分析和图像推理等方面取得重大突破</span>。目前，Qwen-VL 已在智能客服、医疗影像分析、自动化办公和教育辅助等领域展现出广泛应用潜力。

### 2.6.2 <span style="color: rgb(36,91,219); background-color: inherit">Qwen2-VL</span>

目前多模态模型逐渐成为连接视觉与语言的重要工具。传统的视觉语言模型在处理不同分辨率图像和视频时面临效率与精度的挑战。为了解决这一问题，Qwen2-VL 被提出。<span style="color: rgb(46,161,33); background-color: inherit">该模型引入了</span>**<span style="color: rgb(46,161,33); background-color: inherit">动态分辨率适应</span>**<span style="color: rgb(46,161,33); background-color: inherit">技术和</span>**<span style="color: rgb(46,161,33); background-color: inherit">多模态旋转位置嵌入</span>`M-RoPE`**<span style="color: rgb(46,161,33); background-color: inherit">，显著提升了对复杂视觉信息的理解能力</span>。此外，Qwen2-VL 在多模态任务中的高效表现，使其在医学影像分析、智能客服等领域展现了巨大潜力。Qwen2-VL 的能力如右图

![](../../images/视觉多模态讲义（上）-image-140.png)

Qwen2-VL 系列包含三种不同规模的模型，分别&#x662F;**`Qwen2-VL-2B`**、**`Qwen2-VL-7B`**&#x548C;**`Qwen2-VL-72B`**。下表列出了超参数和重要信息。<span style="color: rgb(100,37,208); background-color: inherit">Qwen2-VL 在各种规模的 LLM 中均采用了一个</span>**`675M`**<span style="color: rgb(100,37,208); background-color: inherit">参数的 ViT，确保 ViT 的计算负载在 LLM 不同规模下保持恒定</span>。

![](../../images/视觉多模态讲义（上）-image-135.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

![](../../images/视觉多模态讲义（上）-image-136.png)

上图展示了 Qwen2-VL 的整体结构。模型<span style="color: rgb(100,37,208); background-color: inherit">保留了 Qwen-VL 的框架，集成视觉编码器和语言模型</span>。为了适应不同规模的需求，实现了一个 **`675M`**&#x53C2;数的 ViT，能够高效处理图像和视频输入。在语言处理方面，选择了更强大&#x7684;**`Qwen2`**&#x7CFB;列语言模型。为进一步增强模型对视频中视觉信息的感知和理解能力，引入了几项关键升级：

1. **<span style="color: rgb(36,91,219); background-color: inherit">动态分辨率</span>**

Qwen2-VL 的一项关键架构改进是<span style="color: rgb(100,37,208); background-color: inherit">引入了动态分辨率</span>。与 Qwen-VL 不同，Qwen2-VL 可以处理任意分辨率的图像，并将其动态转换为可变数量的视觉 Token。为了支持这一功能，Qwen 团队修改了 ViT，<span style="color: rgb(100,37,208); background-color: inherit">移除了原始的绝对位置嵌入，并引入了</span>**`2D-RoPE`**<span style="color: rgb(100,37,208); background-color: inherit">来捕捉图像的二维位置信息</span>。在推理阶段，不同分辨率的图像被打包成单个序列，通过控制打包长度来限制 GPU 内存使用。此外，<span style="color: rgb(100,37,208); background-color: inherit">为了减少每张图像的视觉 Token 数量，在 ViT 后添加了一层简单的 MLP，将相邻的</span>**`2 × 2`**<span style="color: rgb(100,37,208); background-color: inherit"> Token 压缩为一个 Token，并在压缩后的视觉 Token 序列的开头和结尾分别插入特殊 Token：</span>**`<|vision_start|>`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`<|visition_end|>`**。因此，一张分辨率&#x4E3A;**`224 × 224`**&#x7684;图像，使&#x7528;**`patch_size=14`**&#x7684; ViT 编码后，会被压缩&#x4E3A;**`66`**&#x4E2A; Token，随后进入大语言模型。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：这一版本的 Qwen2-VL 并没有采用当下流行的大图切分方式，比&#x5982;**`LLaVA-NeXT`**<span style="color: rgb(220,155,4); background-color: inherit">，</span>**`InternVL 2.5`**<span style="color: rgb(220,155,4); background-color: inherit">，以及</span>**`MiniCPM-V`**，而是直接对图像进行 patch 化，然后直接过 Image Encoder 进行特征提取，最后对齐到 LLM 之前，使&#x7528;**`PatchMerger`**&#x5C42;进行视觉 token 数的压缩与进一步提取特征

* **<span style="color: rgb(36,91,219); background-color: inherit">多模态旋转位置嵌入 M-RoPE</span>**

另一项关键架构改进&#x662F;**<span style="color: rgb(100,37,208); background-color: inherit">多模态旋转位置嵌入</span>`M-RoPE`**。与传统大语言模型中&#x7684;**`1D-RoPE`**&#x53EA;能编码一维位置信息不同，**<span style="color: rgb(100,37,208); background-color: inherit">M-RoPE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 能够有效建模多模态输入的位置信息</span>。这是<span style="color: rgb(100,37,208); background-color: inherit">通过将原始旋转嵌入分解为三个组成部分实现的：</span>`时间`<span style="color: rgb(100,37,208); background-color: inherit">、</span>`高度`<span style="color: rgb(100,37,208); background-color: inherit">和</span>`宽度`。

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">文本输入</span>**：这三个部分使用相同的位置 ID，使得 <span style="color: rgb(100,37,208); background-color: inherit">M-RoPE 在功能上等同于 1D-RoPE</span>
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">图像输入</span>**：每个视觉 Token 的时间 ID 保持不变，而高度和宽度组件则根据标记在图像中的位置分配不同的 ID
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">视频输入</span>**：视频被视为帧序列，每帧的时间 ID 递增，而高度和宽度组件遵循与图像相同的 ID 分配模式

<span style="color: rgb(100,37,208); background-color: inherit">在模型输入包含多种模态的情况下，每种模态的位置编号通过将前一模态的最大位置 ID 加一进行初始化</span>。M-RoPE 的示意图如下图所示。<span style="color: rgb(46,161,33); background-color: inherit">M-RoPE 不仅增强了位置信息的建模能力，还减少了图像和视频的位置 ID 值，使模型能够在推理阶段外推到更长的序列</span>。

![](../../images/视觉多模态讲义（上）-image-156.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">统一的图像与视频理解</span>**

Qwen2-VL 采用混合训练方案，同时包含图像和视频数据，从而确保在图像理解和视频理解方面的熟练度。为了<span style="color: rgb(100,37,208); background-color: inherit">尽可能完整地保留视频信息，以每秒两帧的频率对视频进行采样</span>。此外<span style="color: rgb(46,161,33); background-color: inherit">引入深度为 2 的 3D 卷积来处理视频输入，使模型能够处理 3D tube 而非 2D 图像 patch，从而在不增加序列长度的情况下处理更多视频帧</span>。为了保持一致性，每张图像被视为两个相同的帧。为了<span style="color: rgb(100,37,208); background-color: inherit">平衡长视频处理的计算需求与整体训练效率，动态调整每帧视频的分辨率，将每段视频的 Token 总数限制为</span>**`16,384`**。这种训练方法<span style="color: rgb(46,161,33); background-color: inherit">在模型理解长视频的能力和训练效率之间取得了平衡</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：Qwen2-VL 对视觉编码器及其处理部分做了较大的改变：
>
> * 第一&#x5C42;**`patch_embed`**&#x5C42;，使用了一&#x4E2A;**`3D`**&#x5377;积层，其中卷积核大小&#x4E3A;**`(2, 14, 14)`**，步长同样&#x4E3A;**`(2, 14, 14)`**，表示卷积核在时间维度上的大小&#x4E3A;**`2`**，在空间维度上的大小&#x4E3A;**`14x14`**
>
> * 定制化设计&#x4E86;**`rotary_pos_emb`**&#x5C42;，用于<span style="color: rgb(100,37,208); background-color: inherit">对视觉输入做时间和空间上的旋转位置编码</span>
>
> * 对齐&#x5C42;**`PatchMerger`**&#x4F7F;用了普通的 MLP 层，包含两层 Linear，与 Qwen-VL 使用&#x7684;**`Cross-attention`**&#x4E0D;同，这里并不是通过可学习的 Query 来减少视觉 token 数，而是<span style="color: rgb(100,37,208); background-color: inherit">在 </span>**<span style="color: rgb(100,37,208); background-color: inherit">PatchMerger </span>**<span style="color: rgb(100,37,208); background-color: inherit">层中，对相邻的视觉 token 进行合并来实现的，即减少 token 数，同时会增加每个 token 的特征维度</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

沿用 Qwen-VL 的方法，采用三阶段训练策略：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">第一阶段</span>**，专注于训练 ViT 组件，利用海量图文对数据增强 LLM 的语义理解能力
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">第二阶段</span>**，解冻所有参数，并使用更广泛的数据进行训练以实现更全面的学习
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">最后阶段</span>**，锁定 ViT 参数，仅对 LLM 使用指令数据集进行微调

模型在多样化的数据集上进行预训练，包括<span style="color: rgb(100,37,208); background-color: inherit">图文对、OCR 数据、交错排列的图文文章、视觉问答数据集、视频对话和图像知识数据集</span>。数据来源主要包括<span style="color: rgb(100,37,208); background-color: inherit">清洗后的网页、开源数据集和合成数据</span>。这种多样化的数据构成对于开发强大的多模态理解能力至关重要。

1. **<span style="color: rgb(36,91,219); background-color: inherit">初始预训练阶段</span>**

Qwen2-VL 使用&#x4E86;**`600B`**&#x4E2A;标记。Qwen2-VL 的 LLM 组件使用 Qwen2 的参数初始化，而其<span style="color: rgb(100,37,208); background-color: inherit">视觉编码器则使用 DFN 中的 ViT 初始化</span>。然而，原始 DFN 的 ViT 中的固定位置嵌入被替换为 RoPE-2D。<span style="color: rgb(100,37,208); background-color: inherit">这一预训练阶段主要集中在学习图文关系、通过 OCR 进行图像中的文本内容识别以及图像分类任务</span>。这种<span style="color: rgb(46,161,33); background-color: inherit">基础训练有助于模型建立对核心视觉-文本关联和对齐的理解</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">第二阶段预训练</span>**

这一阶段额外引入&#x4E86;**`800B`**&#x4E2A;与图像相关的 Token。此阶段引入了<span style="color: rgb(46,161,33); background-color: inherit">更高比例的混合图文内容，促进了对视觉与文本信息交互的理解。视觉问答数据集的加入提升了模型对图像相关问题的回答能力</span>。此外，<span style="color: rgb(46,161,33); background-color: inherit">多任务数据集的引入对于提升模型处理多样化任务的能力至关重要</span>，尤其是处理复杂数据时。与此同时，<span style="color: rgb(46,161,33); background-color: inherit">纯文本数据在提升模型语言能力方面也有重要作用</span>。

在整个预训练阶段，Qwen2-VL 累计处理&#x4E86;**`1.4T`**&#x4E2A; Token。不仅包括文本 Token，还包括图像 Token。然而，在训练过程中，仅对文本 Token 提供监督信号。这<span style="color: rgb(46,161,33); background-color: inherit">确保了模型能够深入理解视觉与文本信息之间的复杂关系，从而为各种多模态任务奠定了坚实的基础</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">指令微调阶段</span>**

采用 ChatML 格式构建指令跟随数据。该数据集不仅包含纯文本对话数据，还包括多模态对话数据。<span style="color: rgb(100,37,208); background-color: inherit">多模态部分涵盖图像问答、文档解析、多图像比较、视频理解、视频流对话以及基于代理的交互</span>。全面的数据构建方法增强了模型在多种模态下理解和执行指令的能力。通过引入多样化的数据类型，作者开发出一种<span style="color: rgb(46,161,33); background-color: inherit">更加通用且强大的语言模型，能够处理复杂的多模态任务以及传统的文本交互</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">数据格式</span>**

与 Qwen-VL 一致，Qwen2-VL 也使用特殊 Token 来区分视觉和文本输入。**`<|vision_start|>`**&#x548C;**`<|vision_end|>`**&#x88AB;插入到图像特征序列的开头和结尾，以标识图像内容。

1. **<span style="color: rgb(36,91,219); background-color: inherit">对话数据</span>**

使用 ChatML 格式构建指令微调数据集，其中每个交互的语句用两个特殊 Token：**`<|im_start|>`**&#x548C;**`<|im_end|>`**，以便于对话终止。蓝色标记的部分表示监督部分。

![](../../images/视觉多模态讲义（上）-image-159.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉定位</span>**

为了赋予模型视觉定位能力，边界框坐标被标准化&#x5230;**`[0, 1000)`**&#x8303;围内，并表示为

![](../../images/视觉多模态讲义（上）-image-157.png)

$$(X_\text{top left}, Y_\text{top left}),(X_\text{bottom right}, Y_\text{bottom right})$$。**`<|box_start|>`**&#x548C;**`<|box_end|>`**&#x7528;于标识边界框文本。为了将边界框与其文本描述准确关联，引入&#x4E86;**`<|object_ref_start|>`**&#x548C;**`<|object_ref_end|>`**&#x4EE5;指示边界框引用的内容，从而使模型能够有效解释并生成特定区域的精确描述。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉 Agent</span>**

为了将 Qwen2-VL 开发为通用的视觉-语言智能体 VL-Agent，作者将各种代理任务视为序列决策问题，如 <span style="color: rgb(220,155,4); background-color: inherit">UI 操作、机器人控制、游戏和导航</span>，使 Qwen2-VL 能够通过多步动作执行完成任务。对于每个任务，<span style="color: rgb(100,37,208); background-color: inherit">首先定义一组允许的动作和函数调用的关键字模式</span>，即带下划线的部分。Qwen2-VL <span style="color: rgb(100,37,208); background-color: inherit">随后分析观察结果，进行推理和规划，执行选定的动作，并与环境交互以获取新的观察结果</span>。这一循环迭代进行，直到任务成功完成。通过集成各种工具并利用大型视觉-语言模型的视觉感知能力，Qwen2-VL 能够逐步执行涉及真实世界视觉交互的越来越复杂的任务。

![](../../images/视觉多模态讲义（上）-image-158.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">并行训练</span>**

采用了数据并&#x884C;**`DP`**、张量并&#x884C;**`TP`**&#x548C;流水线并&#x884C;**`PP`**&#x7684;三维并行策略来扩展 Qwen2-VL 模型的训练规模。同时利用 DeepSpeed &#x7684;**`ZeRO-1`**&#x5BF9;状态进行分片以节省内存。除此之外还结合了序列并&#x884C;**`SP`**&#x548C;选择性激活检查点技术来进一步减少内存使用。<span style="color: rgb(100,37,208); background-color: inherit">在启用 </span>**<span style="color: rgb(100,37,208); background-color: inherit">TP</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 训练时，将视觉编码器和大语言模型一起分片，但不对视觉合并模块进行分片，因为其参数较少</span>。由于卷积操作的非确定性行为，**TP&#x20;**&#x8BAD;练会导致共享权重的不同。<span style="color: rgb(46,161,33); background-color: inherit">通过离线归约共享权重解决了这一问题，从而避免了额外的全归约通信步骤，且对性能的影响微乎其微</span>。对于 Qwen2-VL 72B 的训练，采用&#x4E86;**`1F1B`**&#x6D41;水线并行策略。<span style="color: rgb(100,37,208); background-color: inherit">将视觉编码器、视觉适配器和若干大语言模型的解码层组合为一个阶段，并将剩余的解码层均匀划分</span>。由于视觉和文本序列长度对于每个数据点都是动态的，因此在启动 **1F1B&#x20;**&#x6D41;程之前广播动态序列长度，并通过批量索引访问形状信息。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**Qwen2-VL** 通过创新的技术架构和训练方法，在视觉与语言融合领域取得了重要突破。<span style="color: rgb(46,161,33); background-color: inherit">其</span>**<span style="color: rgb(46,161,33); background-color: inherit">动态分辨率</span>**<span style="color: rgb(46,161,33); background-color: inherit">适应和 </span>**<span style="color: rgb(46,161,33); background-color: inherit">M-RoPE</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 技术大幅提升了模型的性能，使其能够高效处理不同分辨率的图像和视频</span>。目前，**Qwen2-VL** 已在多个实际场景中展现出广泛应用价值，包括自动化办公、医疗影像分析和教育辅助等。

### 2.6.3 <span style="color: rgb(36,91,219); background-color: inherit">Qwen2.5-VL</span>

随着大型视觉语言模&#x578B;**`LVLM`**&#x7684;快速发展，现有模型在细粒度感知、长视频理解以及多分辨率处理等方面仍存在显著不足。例如，<span style="color: rgb(216,57,49); background-color: inherit">传统模型在处理高分辨率图像或长时间视频时，往往面临计算复杂度高、性能不稳定等问题</span>。为应对这些挑战，Qwen2.5-VL 应运而生。该模型<span style="color: rgb(46,161,33); background-color: inherit">通过引入动态分辨率适应和时间感知编码等创新技术，提升了对复杂视觉信息的理解能力，并扩展其在文档解析、物体定位和事件检测等任务中的应用</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

Qwen2.5-VL 的整体模型架构由三个主要组件构成：

![Qwen2.5-VL 模型参数设置](../../images/视觉多模态讲义（上）-image-155.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">大语言模型</span>**：Qwen2.5-VL 系列采&#x7528;**`Qwen2.5`** LLM 作为其基础组件，并使用其预训练权重初始化。为了更好地满足多模态理解的需求，将<span style="color: rgb(100,37,208); background-color: inherit">一维 RoPE 修改为对齐绝对时间的多模态旋转位置嵌入</span>。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">视觉编码器</span>**：Qwen2.5-VL 的视觉编码器采用了重新设计的 ViT 架构。在结构上引&#x5165;**`2D-RoPE`**&#x548C;窗口注意力机制，以支持原生输入分辨率并加速整个视觉编码器的计算。在训练和推理过程中，<span style="color: rgb(100,37,208); background-color: inherit">输入图像的高度和宽度会被调整为</span>**`28`**<span style="color: rgb(100,37,208); background-color: inherit">的倍数，然后送入 ViT。视觉编码器通过步幅为</span>**`14`**<span style="color: rgb(100,37,208); background-color: inherit">的方式将图像分割成小块，生成一组图像特征</span>。我们在第2.1.1节中详细介绍了视觉编码器。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">基于 MLP 的视觉-语言融合模块</span>**：为了解决图像特征长序列带来的效率挑战，采用了一种简单而有效的方法：<span style="color: rgb(100,37,208); background-color: inherit">在将特征序列送入 LLM 之前对其进行压缩</span>。具体来说，并不是直接使用 ViT 提取的原始小块特征，而是<span style="color: rgb(100,37,208); background-color: inherit">首先将空间相邻的四组小块特征分组。这些分组后的特征被连接并通过一个两层多层感知机 MLP 进行投影，使其维度与 LLM 中使用的文本 Embedding 对齐</span>。这种方法<span style="color: rgb(46,161,33); background-color: inherit">不仅降低了计算成本，还提供了一种灵活的方式来动态压缩不同长度的图像特征序列</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">快速高效的视觉编码器</span>**

视觉编码器在多模态大模型中起着关键作用。为了解决由于原生分辨率输入导致的训练和推理过程中计算负载不平衡的问题，Qwen2.5-VL 重新设计了 ViT 架构。一个关键问题是<span style="color: rgb(216,57,49); background-color: inherit">处理不同尺寸图像时带来的二次计算复杂度</span>。为缓解这一问题，<span style="color: rgb(100,37,208); background-color: inherit">在大多数层中引入了窗口注意力机制，确保计算成本随小块数量线性增长，而非二次增长</span>。在 ViT 架构中，只有四层使用全自注意力，其余层则使用最大窗口大小&#x4E3A;**`112×112`**（对&#x5E94;**`8×8`**&#x5C0F;块）的窗口注意力。小&#x4E8E;**`112×112`**&#x7684;区域无需填充即可保留其原始分辨率。这种设计使模型能够在输入分辨率下原生运行，避免不必要的缩放或失真。

对于位置编码，采&#x7528;**`2D RoPE`**&#x6765;有效捕捉二维空间中的空间关系。此外，为了更好地处理视频输入，将方法扩展&#x5230;**`3D`**&#x5C0F;块。具体来说，使&#x7528;**`14×14`**&#x56FE;像小块作为基本单位，与传统的静态图像 ViT 一致。对于<span style="color: rgb(100,37,208); background-color: inherit">视频数据，将两帧连续帧组合在一起，显著减少了送入语言模型的 token 数量</span>。这种设计不仅保持了与现有架构的兼容性，还提高了处理顺序视频数据的效率。

为了简化整体网络结构，将 ViT 架构更加紧密地与 LLM 对齐，<span style="color: rgb(46,161,33); background-color: inherit">采用</span>**`RMSNorm`**<span style="color: rgb(46,161,33); background-color: inherit">进行归一化，并使用</span>**`SwiGLU`**<span style="color: rgb(46,161,33); background-color: inherit">作为激活函数。这些选择增强了计算效率以及视觉和语言组件之间的兼容性</span>。

在训练方面，wen2.5-VL 从头开始训练重新设计的 ViT。训练过程包括多个阶段，例&#x5982;**`CLIP`**<span style="color: rgb(220,155,4); background-color: inherit">预训练、视觉-语言对齐和端到端微调</span>。为了确保在不同输入分辨率下的鲁棒性，<span style="color: rgb(100,37,208); background-color: inherit">在训练期间使用动态采样以适应原生分辨率</span>。图像根据其原始宽高比随机采样，这使模型能够有效地泛化到不同分辨率的输入。这种方法<span style="color: rgb(46,161,33); background-color: inherit">不仅提高了模型的适应性，还确保了在不同大小的视觉数据上的稳定高效训练</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">动态分辨率和帧率</span>**

Qwen2.5-VL 在空间和时间维度上都进行了改进：

在空间域中，Qwen2.5-VL 动态地将不同大小的图像转换为相应长度的 token 序列。与传统方法对坐标进行归一化不同，<span style="color: rgb(46,161,33); background-color: inherit">Qwen2.5-VL 直接使用输入图像的实际尺寸来表示边界框、点和其他空间特征。这使模型能够学习固有的尺度信息，从而提高其在不同分辨率下处理图像的能力</span>。

对于视频输入，<span style="color: rgb(100,37,208); background-color: inherit">Qwen2.5-VL 结合了动态帧率训练和绝对时间编码</span>。通过适应可变帧率，模型可以更好地捕捉视频内容的时间动态。与其他方法通过引入文本时间戳或使用额外头来实现时间对齐不同，作者提出了一种新颖且高效的策略，<span style="color: rgb(46,161,33); background-color: inherit">将</span>**`MRoPE`**<span style="color: rgb(46,161,33); background-color: inherit">的</span>**`ID`**<span style="color: rgb(46,161,33); background-color: inherit">直接与时间戳对齐。这种方法使模型能够通过时间维度 ID 之间的时间间隔理解时间节奏，而无需增加额外的计算开销</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">对齐绝对时间的多模态 RoPE</span>**

基于 Qwen2-VL 中引入的多模态旋转位置嵌&#x5165;**`MRoPE`**，Qwen2.5-VL 扩展了其能力，以更好地处理视频中的时间信息。Qwen2-VL 中的 **MRoPE** 将位置嵌入分解为三个不同的组成部分：`时间`、`高度`和`宽度`，以有效建模多模态输入：

> **<span style="color: rgb(36,91,219); background-color: inherit">文本输入</span>**：这三个组成部分使用相同的位置 ID，使 MRoPE 在功能上等同于传统的 1D RoPE。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">图像输入</span>**：时间 ID 在所有视觉标记中保持不变，而高度和宽度部分则根据每个标记在图像中的空间位置分配唯一 ID。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">视频输入</span>**：在处理被视为帧序列的视频时，时间 ID 会随着每一帧递增，而高度和宽度部分遵循与静态图像相同的分配模式。

然而，在 Qwen2-VL 中，**<span style="color: rgb(216,57,49); background-color: inherit">MRoPE </span>**<span style="color: rgb(216,57,49); background-color: inherit">中的时间位置 ID 与输入帧的数量绑定，未考虑内容变化速度或视频中事件的绝对时间</span>。为解决这一限制，Qwen2.5-VL 引入了一项关键改进：**<span style="color: rgb(100,37,208); background-color: inherit">将 MRoPE 的时间部分与绝对时间对齐</span>**。如下图所示，<span style="color: rgb(100,37,208); background-color: inherit">通过利用时间 ID 之间的时间间隔，模型能够跨不同 FPS 采样率的视频学习一致的时间对齐</span>。

![](../../images/视觉多模态讲义（上）-image-150.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

与 Qwen2-VL 相比，Qwen2.5-VL 显著扩展了预训练数据的规模，&#x4ECE;**`1.2T`**&#x4E2A; token 增加到&#x7EA6;**`4T`**&#x4E2A; token。预训练数据集通过多种方法构建，包括<span style="color: rgb(220,155,4); background-color: inherit">清理原始网络数据、合成数据</span>等，并且涵盖了广泛的多模态数据类型，例如<span style="color: rgb(220,155,4); background-color: inherit">图像描述、图文交错数据、OCR 数据、视觉知识，如名人、地标、动植物识别、多模态学术问题、定位数据、文档解析数据、视频描述、视频定位以及基于代理的交互数据</span>。在整个训练过程中，根据不同阶段的需求调整这些数据类型的组成和比例，以优化学习效果。

1. **<span style="color: rgb(36,91,219); background-color: inherit">图文交错数据</span>**

图文交错数据对于多模态学习至关重要，具有三个主要优势：

> 1. 支持结合视觉和文本线索进行上下文学习
>
> 2. 在图像缺失时仍能保持强大的纯文本能力
>
> 3. 包含广泛的一般信息

然而，许多<span style="color: rgb(216,57,49); background-color: inherit">现有的图文交错数据存在噪声大、图文关联性弱的问题，限制了其在复杂推理和创造性生成中的实用性</span>。为了解决这些问题，Qwen2.5-VL 设计了一套评分和清理流程，确保只使用高质量且相关的图文交错数据。该流程包括<span style="color: rgb(100,37,208); background-color: inherit">标准数据清理和四阶段评分，评分标准涵盖：纯文本质量、图文相关性、信息互补性和信息密度平衡</span>。通过这种方法，模型能够更高效地执行复杂推理任务，并生成连贯的多模态内容。

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">图文相关性</span>**：衡量图像与文本之间的联系强度，高分表示图像能有意义地补充或解释文本内容
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">信息互补性</span>**：强调图像和文本各自提供独特信息的能力，避免冗余
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">信息密度平衡</span>**：则确保图像和文本之间的信息分布均衡，避免某一方信息过载

这种细致的数据评估方法显著提升了模型的多模态理解能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">定位数据（绝对位置坐标）</span>**

为了实现对世界的更准确感知，<span style="color: rgb(100,37,208); background-color: inherit">Qwen2.5-VL 采用基于输入图像实际尺寸的绝对坐标来表示边界框和点的位置</span>。与相对坐标相比，这种方法<span style="color: rgb(46,161,33); background-color: inherit">能更好地捕捉对象的真实世界尺度和空间关系，从而提升目标检测和定位任务的性能</span>。为增强定位能力的泛化性，Qwen2.5-VL 构建了一个全面的数据集，涵盖带有引用表达式的边界框和点数据。这些数据来源于公开可用数据集和专有数据，并通过多种格式进行合成，&#x5982;**`XML`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`JSON`**<span style="color: rgb(220,155,4); background-color: inherit">等</span>。

为了进一步提升模型在开放词汇检测中的表现，<span style="color: rgb(100,37,208); background-color: inherit">将训练数据扩展到超过</span>**`10,000`**<span style="color: rgb(100,37,208); background-color: inherit">个对象类别，并合成了不存在的对象类别以及包含多个实例的图像数据</span>。此外，为了提高基于点的对象定位能力，构建了一个指代数据集，包括公开数据（&#x5982;**`PixMo`**&#x63D0;供的指代和计数数据）和自动化流水线生成的精确指向数据。通过这些方法，模型在极端场景下的目标检测和定位能力得到了显著提升。

3. **<span style="color: rgb(36,91,219); background-color: inherit">文档全能解析数据</span>**

为了赋予 Qwen2.5-VL 全面解析和理解文档的能力，作者<span style="color: rgb(100,37,208); background-color: inherit">合成了大量包含多样化元素的文档数据</span>。传统方法通常依赖单独的模型处理布局分析、文本提取和图表解释，而 <span style="color: rgb(100,37,208); background-color: inherit">Qwen2.5-VL 通过统一的</span>**`HTML`**<span style="color: rgb(100,37,208); background-color: inherit">格式实现了多模态文档元素的无缝集成</span>。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">文档中的</span>`表格`<span style="color: rgb(100,37,208); background-color: inherit">、</span>`图表`<span style="color: rgb(100,37,208); background-color: inherit">、</span>`公式`<span style="color: rgb(100,37,208); background-color: inherit">、</span>`自然或合成图像`<span style="color: rgb(100,37,208); background-color: inherit">、</span>`乐谱`<span style="color: rgb(100,37,208); background-color: inherit">和</span>`化学公式`<span style="color: rgb(100,37,208); background-color: inherit">等元素被标准化为 </span>**<span style="color: rgb(100,37,208); background-color: inherit">HTML </span>**<span style="color: rgb(100,37,208); background-color: inherit">结构</span>，并将布局框信息和插图描述嵌入到 **HTML&#x20;**&#x6807;签中。

![](../../images/视觉多模态讲义（上）-image-151.png)

除此之外还根据典型阅读顺序优化了文档布局，并在 HTML 标注中加入了每个模块（如<span style="color: rgb(220,155,4); background-color: inherit">段落和图表</span>）的坐标信息。这种创新方法<span style="color: rgb(46,161,33); background-color: inherit">使得任何文档的完整信息（包括</span> <span style="color: rgb(220,155,4); background-color: inherit">布局、文本、图表和插图</span> <span style="color: rgb(46,161,33); background-color: inherit">）都能以标准化和统一的方式表示，从而显著提升了模型在文档理解和转换任务中的效率和准确性</span>。

4. **<span style="color: rgb(36,91,219); background-color: inherit">OCR 数据</span>**

为了增强 OCR 性能，<span style="color: rgb(100,37,208); background-color: inherit">Qwen2.5-VL 整合了来自不同来源的数据，包括合成数据、开源数据和内部收集数据</span>。合成数据通过视觉文本生成引擎生成高质量的野外文本图像，确保模型能适应多样化的文本环境。同时引入了一个大规模多语言 OCR 数据集，支持<span style="color: rgb(220,155,4); background-color: inherit">法语、德语、意大利语、西班牙语、葡萄牙语、阿拉伯语、俄语、日语、韩语和越南语等多种语言</span>。数据集经过精心策划，结合了高质量的合成图像和真实场景图像，以确保多样性和质量。

此外，作者通过可视化库合成&#x4E86;**`1M`**&#x56FE;表样本，&#x5982;**`matplotlib`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`seaborn`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`plotly`**，<span style="color: rgb(100,37,208); background-color: inherit">涵盖条形图、关系图和热力图等类型</span>。对于表格数据，利用<span style="color: rgb(100,37,208); background-color: inherit">离线端到端表格识别模型处理了</span>**`6M`**<span style="color: rgb(100,37,208); background-color: inherit">真实样本，并过滤掉低置信度表格、重叠表格和单元格密度过低的表格</span>。这些方法确保了模型在各种语言环境和复杂文档场景下的强大性能。

* **<span style="color: rgb(36,91,219); background-color: inherit">视频数据</span>**

为了提升模型在处理不同帧率视频数据时的鲁棒性，Qwen2.5-VL 在训练期间动态采样 FPS，以实现训练数据集中 FPS 的均匀分布。<span style="color: rgb(100,37,208); background-color: inherit">对于长度超过半小时的长视频，通过定向合成流水线生成多帧描述，专门构建了一组长视频描述数据</span>。时间戳以秒 second 或 **<span style="color: rgb(216,57,49); background-color: inherit">hmsf</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">h</span>**<span style="color: rgb(216,57,49); background-color: inherit">our-</span>**<span style="color: rgb(216,57,49); background-color: inherit">m</span>**<span style="color: rgb(216,57,49); background-color: inherit">inute-</span>**<span style="color: rgb(216,57,49); background-color: inherit">s</span>**<span style="color: rgb(216,57,49); background-color: inherit">econd-</span>**<span style="color: rgb(216,57,49); background-color: inherit">f</span>**<span style="color: rgb(216,57,49); background-color: inherit">rame）</span>格式标注，确保模型能够准确理解和输出各种格式的时间信息。

这种方法不仅提高了模型对不同帧率视频的适应能力，还增强了其在长视频场景中的表现。通过动态采样和时间戳标注，模型能够更精准地理解和处理复杂的视频内容。

* **<span style="color: rgb(36,91,219); background-color: inherit">智能体数据</span>**

为了构建 Qwen2.5-VL 的智能体能力，从感知和决策两个方面入手：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">感知方面</span>**：收集了移动设备、网页和桌面平台的截图，并通过合成数据引擎生成描述和 UI 元素定位标注。<span style="color: rgb(100,37,208); background-color: inherit">描述任务帮助模型理解图形界面，而定位任务则使其能够对齐 UI 元素的外观和功能</span>。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">决策方面</span>**：将移动设备、网页和桌面平台上的操作统一为函数调用格式，共享动作空间。一组从开源数据收集并由 Agent 框架在虚拟环境中合成的多步轨迹被重新格式化为函数形式。作者进一步通过人工和模型标注者为每一步生成推理内容，解释操作背后的意图。例如，<span style="color: rgb(220,155,4); background-color: inherit">给定一个真实操作，将其在截图上高亮显示，并将全局查询及操作前后的截图提供给标注者，要求他们撰写推理内容</span>。基于模型的过滤器用于筛选低质量的推理内容，防止模型过度拟合真实操作，从而提升其在现实场景中的稳健性。

**<span style="color: rgb(36,91,219); background-color: inherit">训练阶段</span>**

Qwen2.5-VL 从零开始训练了一个 ViT，使&#x7528;**`DataComp`**&#x548C;一些内部数据集作为视觉编码器的初始化，同时利用预训练&#x7684;**`Qwen2.5`**&#x4F5C;为 LLM 组件的初始化。

![](../../images/视觉多模态讲义（上）-image-148.png)

如右表，预训练过程分为三个不同的阶段，每个阶段采用不同的数据配置和训练策略，逐步增强模型的能力。

**<span style="color: rgb(222,120,2); background-color: inherit">第一阶段</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">仅训练 ViT</span>**<span style="color: rgb(100,37,208); background-color: inherit">，以提高其与语言模型的对齐能力</span>。此阶段的主要数据来源包括<span style="color: rgb(220,155,4); background-color: inherit">图像描述、视觉知识和 OCR 数据</span>。这些数据集经过精心挑选，旨在培养 ViT 提取有意义视觉表示的能力，使其能够有效与文本信息集成。

**<span style="color: rgb(222,120,2); background-color: inherit">第二阶段</span>**

所有模型参数被解冻，并在多样化的多模态图像数据上进行训练，以<span style="color: rgb(100,37,208); background-color: inherit">增强其处理复杂视觉信息的能力</span>。此阶段引入了更复杂且需要推理的数据集，例如<span style="color: rgb(220,155,4); background-color: inherit">交错数据、多任务学习数据集、视觉问答 VQA、多模态数学、基于代理的任务、视频理解和纯文本数据集</span>。这些数据集增强了模型在视觉和语言模态之间建立更深层次联系的能力，使其能够处理日益复杂的任务。

**<span style="color: rgb(222,120,2); background-color: inherit">第三阶段</span>**

为了<span style="color: rgb(100,37,208); background-color: inherit">进一步提升模型在更长序列上的推理能力</span>，加入了<span style="color: rgb(220,155,4); background-color: inherit">视频和基于 Agent 的数据</span>，并增加了序列长度。这使得模型能够以更高的精度处理更高级和复杂的多模态任务。通过扩展序列长度，模型获得了处理扩展上下文的能力，这对需要长距离依赖性和复杂推理的任务尤其有益。

为了解决图像尺寸和文本长度变化带来的挑战，即可能导致训练期间计算负载不平衡的问题，<span style="color: rgb(100,37,208); background-color: inherit">采用了优化训练效率的策略</span>。训练时主要的计算成本来自 LLM 和视觉编码器。鉴于<span style="color: rgb(100,37,208); background-color: inherit">视觉编码器的参数相对较少，并且引入了窗口注意力机制以进一步降低其计算需求，因此作者更专注于在不同 GPU 之间平衡 LLM 的计算负载</span>。具体来说，根据输入序列长度动态打包数据样本，确保计算负载的一致性。在第一和第二阶段，数据被统一打包&#x5230;**`8,192`**&#x7684;序列长度，而在第三阶段，序列长度增加&#x5230;**`32,768`**，以适应模型处理更长序列的增强能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

Qwen2.5-VL 的后训练对齐框架采用双阶段优化范式，包括<span style="color: rgb(100,37,208); background-color: inherit">监督微调 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SFT </span>**<span style="color: rgb(100,37,208); background-color: inherit">和直接偏好优化 </span>**<span style="color: rgb(100,37,208); background-color: inherit">DPO</span>**。这种分层对齐策略结合了参数高效的领域适应和人类偏好蒸馏，通过不同的优化目标解决表征基础和行为精细化问题。监督微调是通过有针对性的指令优化弥合预训练表征与下游任务需求之间的差距。在此阶段，采&#x7528;**`ChatML`**&#x683C;式来构建指令跟随数据，并且尽量与预训练数据模式不同，同时保持&#x4E0E;**`Qwen2-VL`**&#x67B6;构的一致性。这种格式转换实现了三项关键调整：

> 1. 明确标记多模态对话中的角色
>
> 2. 将视觉 Embedding 与文本指令结构化注入
>
> 3. 通过格式感知打包的方式保留跨模态的位置关系

通过在这一增强模式下向模型输入精心策划的多模态指令-响应对，SFT 实现了高效的知识迁移，同时保持了预训练特征的完整性。

**<span style="color: rgb(36,91,219); background-color: inherit">指令数据</span>**

SFT 阶段使用精心策划的数据集，旨在增强模型在不同模态下的指令跟随能力。该数据集包含&#x7EA6;**`2M`**&#x6570;据，均匀分布在纯文本数&#x636E;**`50%`**&#x548C;多模态数&#x636E;**`50%`**&#x4E4B;间，后者包括图像-文本和视频-文本组合。<span style="color: rgb(100,37,208); background-color: inherit">多模态数据的引入使模型能够有效处理复杂输入</span>。值得注意的是，<span style="color: rgb(100,37,208); background-color: inherit">尽管纯文本和多模态条目数量相等，但由于嵌入的视觉和时间信息，多模态条目在训练期间消耗更多的标记和计算资源</span>。该数据集主要由中文和英文数据组成，并补充了多语言条目以支持更广泛的语言多样性。

数据集的设计反映了不同层次的对话复杂性，包括单轮和多轮交互。这些交互进一步通过从单张图像输入到多图像序列的情景进行上下文化，从而模拟真实的对话动态。<span style="color: rgb(100,37,208); background-color: inherit">Query 主要来自开源存储库，辅以精选的数据集和在线 Query 数据。这种组合确保了广泛的覆盖范围，增强了数据集的代表性</span>。

为了应对广泛的应用场景，数据集包括专门的子集，用于<span style="color: rgb(220,155,4); background-color: inherit">通用视觉问答 VQA、图像描述、数学问题求解、编码任务和安全相关的 Query</span>。此外，还构建了专门针对<span style="color: rgb(220,155,4); background-color: inherit">文档和 OCR、定位、视频分析和 Agent 交互</span>的数据集，以增强特定领域的专业能力。这种<span style="color: rgb(100,37,208); background-color: inherit">结构化和多样化的组成确保了 SFT 阶段有效地将预训练表征与下游多模态任务的细微需求对齐，促进模型在鲁棒性和情境感知方面的性能提升</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">数据过滤</span>**

训练数据的质量是影响视觉-语言模型性能的关键因素。<span style="color: rgb(216,57,49); background-color: inherit">开源和合成数据集通常表现出显著的变异性，往往包含噪声、冗余或低质量样本</span>。因此，严格的清理和过滤过程对于解决这些问题至关重要。低质量数据可能导致预训练表征与下游任务需求之间的对齐不理想，从而削弱模型有效处理复杂多模态任务的能力。因此，<span style="color: rgb(100,37,208); background-color: inherit">确保高质量数据对于实现稳健可靠的模型性能至关重要</span>。

为了解决这些挑战，Qwen2.5-VL 构造了一个两阶段的数据过滤流程，旨在系统性地提高监督微调 SFT 数据集的质量。该流程包括以下阶段：

1. **<span style="color: rgb(36,91,219); background-color: inherit">第一阶段：领域特定分类</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

在初始阶段，使用&#x4ECE;**`Qwen2-VL-72B`**&#x884D;生的专用分类模&#x578B;**`Qwen2-VL-Instag`**&#x5BF9;问答 QA 对进&#x884C;**<span style="color: rgb(100,37,208); background-color: inherit">分层分类</span>**。该模型将 QA 对<span style="color: rgb(100,37,208); background-color: inherit">组织为八个主要领域</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">编码和规划</span>，<span style="color: rgb(100,37,208); background-color: inherit">这些领域进一步细分为</span>**`30`**<span style="color: rgb(100,37,208); background-color: inherit">个细粒度子类别</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">编码领域被细分为代码调试、代码生成、代码翻译和代码理解等子类别</span>。这种分层结构促进了针对领域和子领域的过滤策略，使这个流程能够根据每个类别的特性优化数据清理过程，从而提高 SFT 数据集的质量和相关性。

* **<span style="color: rgb(36,91,219); background-color: inherit">第二阶段：领域定制过滤</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

第二阶段涉及领域定制过滤，<span style="color: rgb(100,37,208); background-color: inherit">结合基于规则和基于模型的方法，全面提高数据质量</span>。鉴于文档处理、光学字符识别 OCR 和视觉定位等领域具有多样性，每个领域可能需要独特的过滤策略。以下是这些领域中应用的一般过滤策略：

**<span style="color: rgb(222,120,2); background-color: inherit">基于规则的过滤</span>**

基于规则的过滤采用预定义的启发式方法消除低质量或有问题的条目。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">对于与文档处理、OCR 和视觉定位任务相关的数据集，重复模式被识别并移除，以防止扭曲模型的学习过程并确保最佳性能</span>。此外，<span style="color: rgb(100,37,208); background-color: inherit">包含不完整、截断或格式不正确的回答的条目被排除</span>，这在合成数据集和多模态环境中常见。为了保持相关性并遵守道德标准，<span style="color: rgb(100,37,208); background-color: inherit">无关或可能导致有害输出的 Query 和答案也被丢弃</span>。这种结构化方法确保数据集符合道德准则并满足任务特定要求。

**<span style="color: rgb(222,120,2); background-color: inherit">基于模型的过滤</span>**

基于模型的过滤进一步通过利用在 **Qwen2.5-VL** 系列上训练的奖励模型来精炼数据集。这些模型从多个维度评估多模态问答对。<span style="color: rgb(100,37,208); background-color: inherit">Query 根据复杂性和相关性进行评估，仅保留那些适当具有挑战性且上下文相关性强的示例。回答则根据正确性、完整性、清晰度、与查询的相关性和帮助性进行评估</span>。在视觉定位任务中，特别关注验证视觉信息的准确解释和使用。这种多维评分确保只有高质量的数据进入 SFT 阶段。

**<span style="color: rgb(36,91,219); background-color: inherit">拒绝采样</span>**

为了补充结构化数据过滤流程，作者<span style="color: rgb(100,37,208); background-color: inherit">采用拒绝采样作为优化数据集并增强视觉-语言模型推理能力的策略</span>。这种方法对需要复杂推理的任务尤为重要，例如<span style="color: rgb(220,155,4); background-color: inherit">数学问题求解、代码生成和领域特定的视觉问答 VQA</span>。先前的研究表明，引入 CoT 推理显著提高了模型的推理性能，突显了结构化推理过程对于实现高质量结果的重要性。

拒绝采样过程从包含真实标注的数据集开始。这些<span style="color: rgb(100,37,208); background-color: inherit">数据集经过精心策划，包含需要多步推理的任务</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">数学问题求解、代码生成和领域特定的VQA</span>。使用 Qwen2.5-VL 模型的中间版本，<span style="color: rgb(100,37,208); background-color: inherit">将生成的回答与真实标注进行对比。只有模型输出与预期答案匹配的样本被保留，从而确保数据集仅由高质量、准确的示例组成</span>。

为进一步提高数据质量，应用额外的约束条件过滤掉不良输出。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">排除表现出代码切换、过长或重复模式的回答</span>。这些标准确保了链式思维推理过程中的清晰性和连贯性，这对下游应用至关重要。

将链式思维推理应用于视觉-语言模型的一个关键挑战是其对文本和视觉模态的依赖。<span style="color: rgb(216,57,49); background-color: inherit">中间推理步骤可能未能充分整合视觉信息，要么忽略相关视觉线索，要么误解它们</span>。为了解决这个问题，作者<span style="color: rgb(100,37,208); background-color: inherit">设计了基于规则和模型驱动的过滤策略，以验证中间推理步骤的准确性</span>。这些机制确保链式思维过程中的每一步都能有效整合视觉和文本模态。尽管如此，实现最佳模态对齐仍然是一个持续的挑战，需要进一步改进。

<span style="color: rgb(46,161,33); background-color: inherit">通过拒绝采样生成的数据显著增强了模型的推理能力。通过迭代优化数据集并移除低质量或错误样本，使模型能够从强调准确和连贯推理的高质量示例中学习</span>。这种方法不仅增强了模型处理复杂任务的能力，还为未来视觉-语言建模的改进奠定了基础。

**<span style="color: rgb(36,91,219); background-color: inherit">训练方案</span>**

Qwen2.5-VL 的后训练过程包括两个阶段：**`SFT`**&#x548C;**`DPO`**，两个阶段均冻结 ViT 参数。

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">SFT 阶段</span>**：模型<span style="color: rgb(100,37,208); background-color: inherit">在多样化的多模态数据上进行微调</span>，包括图像-文本对、视频和纯文本数据，来源于通用 VQA、拒绝采样以及文档、OCR、定位、视频和 Agent 相关任务等专门数据集。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">DPO 阶段</span>**：<span style="color: rgb(100,37,208); background-color: inherit">专注于图像-文本和纯文本数据，利用偏好数据将模型与人类偏好对齐</span>，每个样本仅处理一次以确保高效优化。这一简化的流程增强了模型的跨模态推理和任务特定性能，同时保持与用户意图的一致性。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

Qwen2.5-VL 凭借其动态分辨率处理、时间感知编码和高效架构设计，在视觉语言领域实现了重要突破。它<span style="color: rgb(46,161,33); background-color: inherit">不仅能够精准理解图像和视频中的细节，还能处理超长上下文内容，适用于从 OCR 到事件总结的多种场景</span>。作为一款全能型视觉语言模型，Qwen2.5-VL在提升性能的同时降低了计算成本，为多模态技术的实际应用铺平了道路。

### 2.6.4 <span style="color: rgb(36,91,219); background-color: inherit">Qwen2.5-Omni</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

人类能够同时处理文字、图像、声音等多模态信息，并自然地通过语言和语音表达，但<span style="color: rgb(216,57,49); background-color: inherit">现有的大模型大多只擅长单一模态，或者在多模态融合时存在时序不同步、输出互相干扰、延迟过高等问题</span>。

基于此，作者提出 **Qwen 2.5-Omni**，目标是打造一个统一的全能模型，<span style="color: rgb(46,161,33); background-color: inherit">既能理解文本、图像、视频和音频输入，又能实时生成文本和语音输出</span>。为此，论文<span style="color: rgb(100,37,208); background-color: inherit">引入了 </span>**<span style="color: rgb(100,37,208); background-color: inherit">TMRoPE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 以解决音视频时序对齐问题，并设计了 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Thinker-Talker 架构</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 来分离文本理解与语音生成，从而减少干扰并支持低延迟流式交互</span>。推动多模态大模型向更自然、更实时、更通用的方向发展。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

![](../../images/视觉多模态讲义（上）-image-152.png)

Qwen2.5-Omni 采&#x7528;**`Thinker-Talker`**&#x67B6;构。**`Thinker`**&#x7C7B;似于大脑，负责<span style="color: rgb(100,37,208); background-color: inherit">处理和理解来自文本、音频和视频模态的输入，生成高层表示及相应的文本</span>。**`Talker`**&#x5219;<span style="color: rgb(100,37,208); background-color: inherit">类似于人类的嘴巴，以流式方式接收 Thinker 生成的高层表示和文本，并流畅输出离散的语音 token</span>。<span style="color: rgb(100,37,208); background-color: inherit">Thinker 是一个 Transformer 解码器，辅以音频和图像编码器以提取信息</span>。相比之下，<span style="color: rgb(100,37,208); background-color: inherit">Talker 采用双轨自回归 Transformer 解码器架构</span>。在训练和推理过程中，<span style="color: rgb(100,37,208); background-color: inherit">Talker 直接接收来自 Thinker 的高维表示，并共享 Thinker 的全部历史上下文信息</span>。因此，整个架构作为一个紧密耦合的单一模型运行，支持端到端训练与推理。

**<span style="color: rgb(36,91,219); background-color: inherit">多模态理解</span>**

Thinker 通过将文本、音频、图像和视频转换为一系列中间表示进行处理。

> **<span style="color: rgb(36,91,219); background-color: inherit">文本分词</span>**：采用 Qwen 的 tokenizer，使&#x7528;**`BBPE`**&#x7F16;码，词表包&#x542B;**`151,643`**&#x4E2A;常规 token。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">纯音频 &amp; 视频中的音频</span>**：采样&#x81F3;**`16kHz`**，并<span style="color: rgb(100,37,208); background-color: inherit">将原始波形转换为</span>**`128`**<span style="color: rgb(100,37,208); background-color: inherit">通道的梅尔频谱图，窗口大小为</span>**`25ms`**<span style="color: rgb(100,37,208); background-color: inherit">，步长为</span>**`10ms`**。音频编码器使每帧音频表示大致对应原始音频信号&#x7684;**`40ms`**&#x7247;段。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">视觉编码器</span>**：基于 ViT，参数量&#x7EA6;**`675M`**，可有效处理图像和视频输入。视觉编码器采用混合训练策略，<span style="color: rgb(100,37,208); background-color: inherit">同时使用图像和视频数据，确保在图像理解与视频理解方面的性能</span>。为尽可能完整保留视频信息并适配音频采样率，<span style="color: rgb(100,37,208); background-color: inherit">采用动态帧率对视频采样</span>。为保持一致性，<span style="color: rgb(100,37,208); background-color: inherit">每张图像被视作两帧完全相同的画面</span>。

Qwen 团队提出一种新的位置编码方法。如下图，**`TMRoPE`**&#x5BF9;多模态输入的三维位置信息进行编码，<span style="color: rgb(100,37,208); background-color: inherit">在多模态旋转位置编码</span>**`M-RoPE`**<span style="color: rgb(100,37,208); background-color: inherit">基础上引入绝对时间位置</span>。即<span style="color: rgb(100,37,208); background-color: inherit">将原始旋转嵌入分解为三个分量：</span>**<span style="color: rgb(100,37,208); background-color: inherit">时间</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">高度</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">宽度</span>**。

![](../../images/视觉多模态讲义（上）-image-153.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">文本输入</span>**：三个分量<span style="color: rgb(100,37,208); background-color: inherit">使用相同的位置 ID</span>，此时 M-RoPE 在功能上等价&#x4E8E;**`1D-RoPE`**
>
> **<span style="color: rgb(36,91,219); background-color: inherit">音频输入</span>**：同样<span style="color: rgb(100,37,208); background-color: inherit">使用相同的位置 ID</span>，并<span style="color: rgb(100,37,208); background-color: inherit">引入绝对时间位置编码</span>，每个时间 ID 对&#x5E94;**`40ms`**
>
> **<span style="color: rgb(36,91,219); background-color: inherit">图像输入</span>**：每个视觉 token 的<span style="color: rgb(100,37,208); background-color: inherit">时间 ID 保持恒定</span>，高度和宽度<span style="color: rgb(100,37,208); background-color: inherit">根据 token 在图像中的位置分配不同 ID</span>
>
> **<span style="color: rgb(36,91,219); background-color: inherit">含音频的视频输入</span>**：
>
> 音频仍以&#x6BCF;**`40ms`**&#x4E00;帧的方式<span style="color: rgb(100,37,208); background-color: inherit">使用相同位置 ID 编码</span>
>
> 视频被视为一系列图像，<span style="color: rgb(100,37,208); background-color: inherit">每帧的时间 ID 递增，高度和宽度分量沿用图像的 ID 分配方式</span>。由于视频帧率不固定，<span style="color: rgb(100,37,208); background-color: inherit">根据每帧实际对应的时间动态调整帧间时间 ID，确保每个时间 ID 对应</span>**`40ms`**
>
> 根据实际时间，<span style="color: rgb(100,37,208); background-color: inherit">每</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">秒将视频表示划分为一个块</span>，在每&#x4E2A;**`2`**&#x79D2;块内，<span style="color: rgb(100,37,208); background-color: inherit">将视觉表示置于前部，音频表示置于后部，实现音视频表示的交错排列</span>
>
> 当模型输入包含多个模态时，每个模态的位置编号<span style="color: rgb(100,37,208); background-color: inherit">从上一个模态的最大位置</span>**`ID +1`**<span style="color: rgb(100,37,208); background-color: inherit">开始</span>

> 在为各模态嵌入位置信息后，按顺序排列其表示

**<span style="color: rgb(36,91,219); background-color: inherit">多模态生成</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">文本</span>**：由 Thinker 直接生成。生成逻辑与主流 LLM 基本一致，<span style="color: rgb(100,37,208); background-color: inherit">基于词表概率分布进行自回归采样</span>。生成过程结合重复惩罚、top-p 采样等技术以增强多样性。

**<span style="color: rgb(36,91,219); background-color: inherit">语音</span>**：<span style="color: rgb(100,37,208); background-color: inherit">Talker 接收 Thinker 采样得到的文本 token 嵌入及其高层表示</span>。高维表示与离散采样 token 的融合在此至关重要。作为流式算法，<span style="color: rgb(100,37,208); background-color: inherit">语音生成必须在完整文本尚未生成前预判内容的语气与态度</span>。<span style="color: rgb(100,37,208); background-color: inherit">Thinker 提供的高维表示隐式传递了这些信息，从而实现更自然的流式生成</span>。此外，Thinker 的表示主要在语义空间体现相似性，而非语音相似性。因此，<span style="color: rgb(100,37,208); background-color: inherit">即使语音差异显著的词也可能具有高度相似的高层表示，必须输入离散采样 token 以消除此类歧义</span>。

> Qwen 团队设计了一种高效语音编解码&#x5668;**`qwen-tts-tokenizer`**，可以高效表征语音关键信息，并通过因果音频解码器流式解码为语音。Talker 接收信息后，自回归地生成音频 token 与文本 token。<span style="color: rgb(46,161,33); background-color: inherit">语音生成无需与文本在词级别或时间戳级别对齐，简化了训练数据要求与推理流程</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">流式设计</span>**

在流式音视频交互场景中，首包延迟是衡量系统流式性能的关键指标。该延迟受以下因素影响：

> <span style="color: rgb(216,57,49); background-color: inherit">多模态信息输入</span>处理造成的延迟
>
> 从接收<span style="color: rgb(216,57,49); background-color: inherit">首个文本输入到输出首个语音 token</span> 的延迟
>
> 将<span style="color: rgb(216,57,49); background-color: inherit">首段语音转换为音频</span>的延迟
>
> <span style="color: rgb(216,57,49); background-color: inherit">架构本身固有延迟</span>，与模型规模、计算量 FLOPs 等因素相关

为在多模态交互中支持推理框架广泛采用&#x7684;**`Chunked-Prefills`**，Qwen 团队修改了音频与视觉编码器，使其支持沿时间维度的块状注意力。音频编码器<span style="color: rgb(100,37,208); background-color: inherit">从对整段音频的全注意力改为每</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">秒一个块的块内注意力</span>。视觉编码器<span style="color: rgb(100,37,208); background-color: inherit">采用 Flash Attention 以提升训练与推理效率，并通过一个简单的 MLP 层将相邻</span>**`2×2`**<span style="color: rgb(100,37,208); background-color: inherit">的 token 合并为一个 token</span>。图像块大小设&#x4E3A;**`14`**，使得不同分辨率的图像均可被压缩为序列。

![](../../images/视觉多模态讲义（上）-image-146.png)

为支持音频流式生成，尤其长序列，Qwen 团队<span style="color: rgb(100,37,208); background-color: inherit">使用滑动窗口块注意力机制，限制当前 token 仅能访问有限上下文</span>，即采&#x7528;**`Flow-Matching DiT`**&#x6A21;型。<span style="color: rgb(100,37,208); background-color: inherit">输入编码通过</span>**<span style="color: rgb(216,57,49); background-color: inherit">Flow-Matching</span>**<span style="color: rgb(100,37,208); background-color: inherit">转换为梅尔频谱图，再经修改过的</span>**`BigVGAN`**<span style="color: rgb(100,37,208); background-color: inherit">重建为波形</span>。

为从编码生成波形，将相邻编码分组为块，并以此构建注意力掩码。**`DiT`**<span style="color: rgb(100,37,208); background-color: inherit">的感受野限制为</span>**`4`**<span style="color: rgb(100,37,208); background-color: inherit">个块，包括回看</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">个块与前看</span>**`1`**<span style="color: rgb(100,37,208); background-color: inherit">个块</span>。解码时，<span style="color: rgb(100,37,208); background-color: inherit">使用</span>**`Flow Matching`**<span style="color: rgb(100,37,208); background-color: inherit">以块为单位生成梅尔频谱，确保每个编码块均可访问必要的上下文块</span>。这通过保留上下文信息提升流式输出质量。Qwen 团队也采用这种逐块方法适配 BigVGAN 的固定感受野，以支持流式波形生成。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

模型在多样化的数据集上进行预训练，涵盖<span style="color: rgb(100,37,208); background-color: inherit">图像-文本、视频-文本、视频-音频、音频-文本及纯文本语料等类型</span>。这里作者参照 Qwen2-Audio 的做法，<span style="color: rgb(100,37,208); background-color: inherit">将层级化标签替换为自然语言 prompt</span>，以提升模型的泛化能力与指令遵循能力。

Qwen2.5-Omni 的训练分为三个阶段：

> **<span style="color: rgb(36,91,219); background-color: inherit">Step 1</span>**：<span style="color: rgb(100,37,208); background-color: inherit">冻结 LLM 参数，专注于训练视觉编码器和音频编码器</span>，利用海量的音频-文本与图像-文本配对数据，增强 LLM 对语义的理解能力
>
> Qwen2.5-Omni 的<span style="color: rgb(100,37,208); background-color: inherit"> LLM 使用</span>**`Qwen2.5`**<span style="color: rgb(100,37,208); background-color: inherit">的参数进行初始化，视觉编码器与</span>**`Qwen2.5-VL`**<span style="color: rgb(100,37,208); background-color: inherit">一致，音频编码器使用</span>**`Whisper-large-v3`**<span style="color: rgb(100,37,208); background-color: inherit">初始化</span>。两个编码器在固定 LLM 上独立训练，<span style="color: rgb(100,37,208); background-color: inherit">初期均先训练各自的适配器，再训练编码器本身</span>。这一基础训练对模型掌握核心的视觉-文本与音频-文本关联与对齐能力至关重要。

> **<span style="color: rgb(36,91,219); background-color: inherit">Step 2</span>**：<span style="color: rgb(100,37,208); background-color: inherit">解冻所有参数</span>，使用更广泛的多模态数据进行训练，实现更全面的学习
>
> <span style="color: rgb(100,37,208); background-color: inherit">引入约</span>**`800B`**<span style="color: rgb(100,37,208); background-color: inherit"> token 的图像与视频相关数据、</span>**`300B`**<span style="color: rgb(100,37,208); background-color: inherit"> token 的音频相关数据，以及</span>**`100B`**<span style="color: rgb(100,37,208); background-color: inherit"> token 的含音频视频相关数据</span>。本阶段<span style="color: rgb(46,161,33); background-color: inherit">引入更大规模的混合多模态数据与更丰富的任务类型，强化了听觉、视觉与文本信息之间的交互与深层理解</span>。
>
> > **<span style="color: rgb(36,91,219); background-color: inherit">多模态、多任务数据集</span>**&#x5BF9;培养模型同时处理多种任务与模态的能力至关重要，这是应对复杂现实世界数据的关键能力
> >
> > **<span style="color: rgb(36,91,219); background-color: inherit">纯文本数据</span>**&#x5728;维持并提升语言能力方面发挥着不可替代的作用

> **<span style="color: rgb(36,91,219); background-color: inherit">Step 3</span>**：<span style="color: rgb(100,37,208); background-color: inherit">使用序列长度为</span>**`32k`**<span style="color: rgb(100,37,208); background-color: inherit">的数据</span>，提升模型对复杂长序列数据的理解能力
>
> 前两个阶段中最大 token 长度限制&#x4E3A;**`8192`**，可以提高训练效率。之后<span style="color: rgb(100,37,208); background-color: inherit">引入长音频与长视频数据，将原始文本、音频、图像与视频数据扩展至</span>**`32,768`**<span style="color: rgb(100,37,208); background-color: inherit"> token 进行训练</span>。这些数据在支持长序列建模方面表现出显著提升。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

![](../../images/视觉多模态讲义（上）-image-154.png)

**<span style="color: rgb(36,91,219); background-color: inherit">Thinker</span>**

使&#x7528;**`ChatML`**&#x683C;式的指令跟随数据进行指令微调。数据集包含<span style="color: rgb(100,37,208); background-color: inherit">纯文本对话数据、视觉模态对话数据、音频模态对话数据以及混合模态对话数据</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">Talker</span>**

Qwen 团队为 Talker 设计了三阶段训练流程，使 Qwen2.5-Omni 能同时生成文本与语音回答。

> **<span style="color: rgb(36,91,219); background-color: inherit">Step 1</span>**：训练 Talker 学习上下文延续能力
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Step 2</span>**：采用 DPO提升语音生成的稳定性
>
> $$\mathcal{L}_{\text{DPO}}(P_\theta; P_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{P_\theta(y_w | x)}{P_{\text{ref}}(y_w | x)} - \beta \log \frac{P_\theta(y_l | x)}{P_{\text{ref}}(y_l | x)} \right) \right]$$
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Step 3</span>**：实施多说话人指令微调，以增强语音回答的自然度与可控性

在上下文学习 ICL 训练阶段，除使用与 Thinker 类似的文本监督信号外，还<span style="color: rgb(100,37,208); background-color: inherit">通过下一 token 预测执行语音延续任务</span>，利用大量包含多模态上下文与口语回答的对话数据。Talker <span style="color: rgb(100,37,208); background-color: inherit">学习建立从语义表示到语音的单调映射，同时掌握根据上下文生成具有适当韵律、情感与口音等多样属性语音的能力</span>。此外还引入音色解耦技术，防止模型将特定声音与低频文本模式错误关联。

为扩大说话人与场景的覆盖范围，预训练数据不可避免地包含标签噪声与发音错误，易导致模型产生幻觉。为此<span style="color: rgb(100,37,208); background-color: inherit">引入强化学习阶段以提升语音生成的稳定性</span>。对于每个 request 及其对应的参考语音，构建数据集$$D$$，包含三元组 $$(x, y_w, y_l)$$，其中$$x$$为输入文本序列，$$y_w$$与$$y_l$$分别为生成质量好与差的语音序列。这里<span style="color: rgb(100,37,208); background-color: inherit">根据词错误率</span>**`WER`**<span style="color: rgb(100,37,208); background-color: inherit">与标点停顿错误率计算的奖励分数对样本进行排序</span>。

最后在上述基础模型上进行 speaker fine-tuning，使 Talker 能够模仿特定声音并进一步提升语音自然度。

### 2.6.5 <span style="color: rgb(36,91,219); background-color: inherit">Qwen3-VL</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

Qwen3-VL 采用&#x7531;**<span style="color: rgb(100,37,208); background-color: inherit">视觉编码器</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">视觉–语言融合模块</span>**<span style="color: rgb(100,37,208); background-color: inherit">和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">LLM</span>** 组成的三模块架构：

![](../../images/视觉多模态讲义（上）-image-149.png)



**<span style="color: rgb(222,120,2); background-color: inherit">LLM</span>**：Qwen3-VL 均基于 Qwen3，在多模态任务中超越大多数VLM，并在多数语言基准上优于纯文本版本。

| **<span style="color: rgb(36,91,219); background-color: inherit">Dense</span>** | **Qwen3-VL-2B、Qwen3-VL-4B、Qwen3-VL-8B、Qwen3-VL-32B** |
| ------------------------------------------------------------------------------- | ---------------------------------------------------- |
| **<span style="color: rgb(36,91,219); background-color: inherit">MoE</span>**   | **Qwen3-VL-30B-A3B、Qwen3-VL-235B-A22B**              |

**<span style="color: rgb(222,120,2); background-color: inherit">视觉编码器</span>**：使&#x7528;**`SigLIP-2`**&#x67B6;构，从其权重开始以动态输入分辨率进行训练。使&#x7528;**`CoMP`**&#x65B9;法处理动态分辨率，并使&#x7528;**`2D-RoPE`**&#x5E76;根据输入尺寸插值绝对位置编码。对&#x4E8E;**`2B`** &#x548C;**`4B`**&#x5C0F;规模 LLM 使&#x7528;**`SigLIP2-Large-300M`**。其他使&#x7528;**`SigLIP2-SO-400M`**

**<span style="color: rgb(222,120,2); background-color: inherit">视觉–语言融合模块</span>**：与 Qwen2.5-VL 一致，使用一个两层 MLP 将视觉编码器输出的 $$2 \times 2$$ 视觉特征压缩为一个视觉 token，维度与 LLM 的隐藏维度对齐。

1. **<span style="color: rgb(36,91,219); background-color: inherit">交错式 MRoPE</span>**

Qwen2-VL 引入了 MRoPE 建模多模态输入的位置信息。原始方案<span style="color: rgb(216,57,49); background-color: inherit">将向量维度划分为时间</span>$$t$$<span style="color: rgb(216,57,49); background-color: inherit">、水平</span>$$h$$<span style="color: rgb(216,57,49); background-color: inherit">和垂直</span>$$w$$<span style="color: rgb(216,57,49); background-color: inherit">三个子空间，并为每个子空间分配不同的旋转频率，这导致频谱不平衡，会损害长视频理解基准上的性能</span>。Qwen3-VL 重新设计频率分配方式，<span style="color: rgb(100,37,208); background-color: inherit">将</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">、</span>$$h$$<span style="color: rgb(100,37,208); background-color: inherit">、</span>$$w$$<span style="color: rgb(100,37,208); background-color: inherit">分量在向量维度上交错排列，这确保每个时空轴在低频和高频段均被均匀表示</span>。所得的<span style="color: rgb(46,161,33); background-color: inherit">均衡频谱缓解了原始的频谱偏差，显著提升了视频的长程位置建模能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">DeepStack</span>**

与DeepStack 类似的，将视觉 token 注入 LLM 的多个层中：<span style="color: rgb(100,37,208); background-color: inherit">将 DeepStack 扩展至从 ViT 的中间层提取视觉 token，保留了从低层到高层的丰富视觉信息</span>。从视觉编码器中选取三个不同层级的特征。随后<span style="color: rgb(100,37,208); background-color: inherit">视觉–语言融合模块将这些多层级特征投影为视觉 token</span>，并直接加到 LLM 前三层对应的隐藏状态上。

![](../../images/视觉多模态讲义（上）-image-145.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">视频时间戳</span>**

在 Qwen2.5-VL 中，采用时间同步的 MRoPE 让模型模型可以感知时间。但这存在两个关键缺点：

> 1. 将时间位置 ID 直接绑定到绝对时间，导致长视频产生过大且稀疏的时间位置 ID，削弱模型对长时上下文的理解能力
>
> 2. 有效学习需在各种帧率下进行大量且均匀分布的采样，显著增加训练数据构建成本

为解决这些问题，Qwen3-VL 采用基于文本 token 的时间编码策略：<span style="color: rgb(100,37,208); background-color: inherit">每个视频时间片段前加一个格式化的时间戳文本字符串，例如</span>**`<3.0 seconds>`**。此外<span style="color: rgb(100,37,208); background-color: inherit">在训练中同时生成秒制与</span>**`HMS(hours:minutes:seconds)`**<span style="color: rgb(100,37,208); background-color: inherit">格式的时间戳，以确保模型能理解多样化的时间表示</span>。这略微增加上下文长度，但使模型能更有效、更精确地感知时间信息，从而支持如<span style="color: rgb(220,155,4); background-color: inherit">视频定位和密集描述</span>等时间感知任务。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

系统性地分为四个阶段，逐步构建从基础对齐到长上下文理解的能力：

![](../../images/视觉多模态讲义（上）-image-147.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">Step 0：视觉–语言对齐</span>**。缝合视觉编码器与 LLM 之间的模态差异。此阶段<span style="color: rgb(100,37,208); background-color: inherit">仅训练 MLP 融合模块的参数，视觉编码器和 LLM 主干保持冻结</span>。使用&#x7EA6;**`67B`** token 的精选数据集，包括高质量图像–文本对、视觉知识集合和 OCR 数据。所有训练使&#x7528;**`8,192`**&#x7684;上下文长度。这种<span style="color: rgb(46,161,33); background-color: inherit">先对齐策略为跨模态理解奠定了基础</span>。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Step 1：多模态预训练</span>**。全参数多模态预训练，<span style="color: rgb(100,37,208); background-color: inherit">解冻所有模型组件：视觉编码器、融合模块、LLM，进行端到端联合训练</span>。模型在&#x7EA6;**`1T`** token 的大规模多样化数据集上训练。为了保持 LLM 的语言能力，数据混合包含视觉–语言数据和纯文本数据。VL 部分包括交错图文文档、视觉定位任务、视觉问答、STEM 领域数据及少量视频数据以引入时间理解。上下文长度仍&#x4E3A;**`8,192`**。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Step 2：长上下文预训练</span>**。扩展模型的上下文处理能力，将上下文长度增&#x81F3;**`4`**&#x500D;变&#x4E3A;**`32,768`**，同时<span style="color: rgb(100,37,208); background-color: inherit">保持所有参数可训练</span>。训练使用&#x7EA6;**`1T`** token 的数据集，并<span style="color: rgb(100,37,208); background-color: inherit">调整数据配比以支持长上下文任务：增加纯文本数据比例以增强长文本理解</span>，同时<span style="color: rgb(100,37,208); background-color: inherit">视觉–语言数据中大幅增加视频数据和面向智能体的指令跟随数据</span>。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Step 3：超长上下文适配</span>**。将模型上下文窗口推至操作极限，上下文长度大幅增&#x81F3;**`262,144`**。模型在精心构建&#x7684;**`100B`** token 数据集上训练，数据包含纯文本和视觉–语言数据，重点<span style="color: rgb(100,37,208); background-color: inherit">聚焦于长视频和长文档理解任务</span>。这巩固了 Qwen3-VL 处理和分析极长序列输入的能力，<span style="color: rgb(46,161,33); background-color: inherit">对综合文档分析和长视频摘要等应用至关重要</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">训练方案</span>**

分为三个阶段，主要是优化模型的指令跟随能力、强化推理能力并实现与人类偏好的对齐：

> **<span style="color: rgb(36,91,219); background-color: inherit">监督微调 SFT</span>**：赋予模型指令跟随能力并激活潜在推理技能。分两步进行：<span style="color: rgb(100,37,208); background-color: inherit">初始阶段使用</span>**`32k`**<span style="color: rgb(100,37,208); background-color: inherit">上下文长度，随后扩展至</span>**`256k`**<span style="color: rgb(100,37,208); background-color: inherit">上下文窗口，聚焦长文档和长视频数据</span>。将训练数据分为两类：**<span style="color: rgb(100,37,208); background-color: inherit">标准格式</span>**<span style="color: rgb(100,37,208); background-color: inherit">用于非思考模型，</span>**<span style="color: rgb(100,37,208); background-color: inherit">思维链格式</span>**<span style="color: rgb(100,37,208); background-color: inherit">用于思考模型</span>，显式建模推理过程。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">强到弱蒸馏 Strong-to-Weak Distillation</span>**：由教师模型向学生模型传递能力。这里<span style="color: rgb(100,37,208); background-color: inherit">使用纯文本数据对 LLM 进行微调，在文本中心和多模态任务中均显著提升推理能力</span>。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">强化学习 RL</span>**：进一步提升模型性能与对齐度，<span style="color: rgb(100,37,208); background-color: inherit">分为</span>**<span style="color: rgb(100,37,208); background-color: inherit">推理 RL</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 与</span>**<span style="color: rgb(100,37,208); background-color: inherit">通用 RL</span>**。在数学、OCR、定位、指令跟随等多种文本与多模态领域上进行大规模强化学习，以提升细粒度能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">强到弱蒸馏</span>**

使用 Qwen3 中的强到弱蒸馏流程进一步提升轻量模型性能，包含两个主要阶段：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Off-policy Distillation</span>**：组合教师模型生成的输出来蒸馏，帮助学生模型获得基础推理能力，为后续在线策略训练奠定基础。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">On-policy Distillation</span>**：学生模型基于给定 prompt 生成 response。On-policy 序列随后用于微调学生模型，这里通过最小化 KL 散度对齐学生与教师的预测 logits。

* **<span style="color: rgb(36,91,219); background-color: inherit">强化学习</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">推理强化学习</span>**

在多样化的文本与多模态任务上训练模型，包括数学、编程、逻辑推理、视觉定位和视觉谜题。

**<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**：从开源和专有来源整理训练数据，并进行严格预处理和人工标注，以确保高质量 RL query。对于多模态 query，<span style="color: rgb(100,37,208); background-color: inherit">使用 Qwen3-VL-235B-A22B 的权重，为每个 query 采样</span>**`16`**<span style="color: rgb(100,37,208); background-color: inherit">个 response ；若所有 response 均错误，则丢弃该 query</span>。随后，<span style="color: rgb(100,37,208); background-color: inherit">对每项任务进行初步 RL 实验，剔除改进潜力有限的数据源</span>。最终得到&#x7EA6;**`30K`**&#x4E2A;覆盖各类文本与多模态任务的 RL query 。在训练每个模型时，<span style="color: rgb(100,37,208); background-color: inherit">为所有 query 采样</span>**`16`**<span style="color: rgb(100,37,208); background-color: inherit">个 response ，并过滤掉通过率超过</span>**`90%`**<span style="color: rgb(100,37,208); background-color: inherit">的简单 query</span>。将任务特定数据集打乱并混合，构建多任务批次，确保每项任务按预设比例采样——该比例通过大量预实验确定。

**<span style="color: rgb(36,91,219); background-color: inherit">奖励系统</span>**：系统提供共享基础设施，包括<span style="color: rgb(100,37,208); background-color: inherit">数据预处理、实用函数、奖励管理器</span>以集成多种奖励类型，核心奖励逻辑按任务实现。这里<span style="color: rgb(100,37,208); background-color: inherit">使用任务特定的 prompt 格式引导模型输出符合要求的格式</span>，因此无需显式格式奖励，当 response 语言与 prompt 语言不一致时施加惩罚。

**<span style="color: rgb(36,91,219); background-color: inherit">RL 算法</span>**：采用平滑自适应的策略梯度方&#x6CD5;**`SAPO`**&#x8FDB;行 RL 训练。

**<span style="color: rgb(222,120,2); background-color: inherit">通用强化学习</span>**

提升模型的泛化能力与运行鲁棒性，采用多任务 RL 范式，<span style="color: rgb(100,37,208); background-color: inherit">奖励函数基于 SFT 阶段的综合任务集构建</span>，包括 VQA、图像描述、OCR、文档解析、定位和钟表识别等。奖励机制包括两个核心维度：

> * **<span style="color: rgb(36,91,219); background-color: inherit">指令跟随</span>**：评估模型对用户明确指令的遵循程度，包括处理内容、格式、长度和结构化输出等复杂约束的能力，确保生成 response 精确匹配用户需求。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">偏好对齐</span>**：针对开放式或主观性 query，通过优化有用性、事实准确性和风格适当性，使模型输出与人类偏好一致，从而促进更自然、更吸引人的交互。

这个阶段也可以<span style="color: rgb(100,37,208); background-color: inherit">进行纠错，用于去除 SFT 阶段中错误的知识先验</span>。通过引入专门设计的可验证任务来触发这些特定错误，如<span style="color: rgb(220,155,4); background-color: inherit">反直觉物体计数和复杂钟表时间识别</span>，从而以针对性干预用事实知识替代错误先验。

另一关键目标是缓解<span style="color: rgb(220,155,4); background-color: inherit">不当语言混用、过度重复和格式错误</span>等行为。但这些问题发生率低，通用 RL 作为纠正策略样本效率不高。为此<span style="color: rgb(100,37,208); background-color: inherit">构建专用数据集，专门收集已知会引发此类行为的 prompt </span>。这种聚焦训练使我们能施加高频、定向惩罚，有效抑制这些残余错误。

RL 过程的反馈通过混合奖励系统提供，结合两种互补方法：

> * **<span style="color: rgb(36,91,219); background-color: inherit">基于规则的奖励</span>**：对具有可验证真值的任务，如<span style="color: rgb(220,155,4); background-color: inherit">格式遵循、指令跟随</span>，提供明确、高精度反馈。通过明确定义的启发式规则，提供稳健的正确性评估机制，并<span style="color: rgb(100,37,208); background-color: inherit">有效缓解奖励黑客行为：模型利用学习型奖励函数的模糊性进行投机</span>
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">基于模型的奖励</span>**：<span style="color: rgb(100,37,208); background-color: inherit">采用 Qwen2.5-VL-72B-Instruct 或 Qwen3 作为高级评判模型，评判模型将每个生成 response 与真值参考对比，从多个维度打分</span>。在评估细微或开放式任务时更具灵活性，尤其擅长减少假阴性：对格式或措辞非常规但有效的 response 进行不当惩罚

* **<span style="color: rgb(36,91,219); background-color: inherit">Thinking 模式</span>**

通过两阶段训练范式赋予 Qwen3-VL 带图思考能力。

> **<span style="color: rgb(36,91,219); background-color: inherit">Step 1</span>**：合成一个包含&#x7EA6;**`10k`**&#x4E2A;定位样本的冷启动智能体数据集，主要为简单的两轮视觉问答任务。随后在 Qwen2.5-VL-32B 上进行监督微调，<span style="color: rgb(100,37,208); background-color: inherit">模拟视觉智能体行为：</span>**<span style="color: rgb(100,37,208); background-color: inherit">思考 → 行动 → 分析反馈 → 回答</span>**。为进一步增强推理能力，应用多轮、工具集成的强化学习。
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Step 2</span>**：将第一阶段训练好的 Qwen2.5-VL-32B 视觉智能体进行蒸馏，生成一个更大、更多样化的约 **`120k`**&#x591A;轮智能体交互数据集，覆盖更广泛的视觉任务。随后对 Qwen3-VL 应用类似的冷启动 SFT 与工具集成 RL 流程（现同时使用蒸馏数据与合成数据）。

两个阶段的多轮、工具集成 RL 流程几乎相同，仅底层数据不同。在 RL 过程中采用三种互补的奖励信号：

> * **<span style="color: rgb(36,91,219); background-color: inherit">答案准确度奖励</span>**：利&#x7528;**`Qwen3-32B`**&#x5224;断最终答案是否正确
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多轮推理奖励</span>**：利&#x7528;**`Qwen2.5-VL-72B`**&#x8BC4;估是否正确解读工具或环境反馈，并通过连贯的逐步推理得出答案
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">工具调用奖励</span>**：通过比较实际工具调用次数与专家估计目标，鼓励适当工具使用。&#x7531;**`Qwen2.5-VL-72B`**&#x6839;据任务复杂度离线确定

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：模型倾向于仅进行单次工具调用以投机前两项奖励，无论任务需求如何。为缓解此问题，Qwen3-VL <span style="color: rgb(100,37,208); background-color: inherit">显式引入工具调用奖励，以促进与任务复杂度对齐的自适应工具探索</span>。

### 2.6.6 <span style="color: rgb(36,91,219); background-color: inherit">Qwen3-Omni</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

**Qwen3-Omni** &#x548C;**`Qwen2.5-Omni`**&#x4E00;样采&#x7528;**`Thinker-Talker`**&#x67B6;构，如下图：

![](../../images/视觉多模态讲义（上）-image-170.png)

但是和 Qwen2.5-Omni 相比，Qwen3-Omni 主要有以下几个改进： &#x20;

> 1. **`Thinker`**&#x548C;**`Talker`**&#x90FD;&#x662F;**`MoE`**&#x67B6;构，从而支持高并发和快速推理 &#x20;
>
> 2. **<span style="color: rgb(100,37,208); background-color: inherit">Talker </span>**<span style="color: rgb(100,37,208); background-color: inherit">不再依赖 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Thinker</span> <span style="color: rgb(216,57,49); background-color: inherit"> </span>**<span style="color: rgb(100,37,208); background-color: inherit">的高层文本表征，仅使用音频和视觉多模态特征为条件</span>。这么做的原因是：
>
>    1. 对于文本内容，离散 <span style="color: rgb(100,37,208); background-color: inherit">token 和 Embedding 在信息上是等价的</span>
>
>    2. 多模态条件对音视频同步语音生成是必需的，如语音翻译中保持韵律/音色。而且这种解耦的做法可以让外部模块，&#x5982;**`RAG`**、`函数调用`、`安全过滤器`，干预 **Thinker&#x20;**&#x7684;文本输出，并可根据需要通过受控预处理向 **Talker&#x20;**&#x63D0;供文本，实现流式合成
>
> 3. 由于文本表征是解耦的，**<span style="color: rgb(100,37,208); background-color: inherit">Thinker </span>**<span style="color: rgb(100,37,208); background-color: inherit">和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Talker </span>**<span style="color: rgb(100,37,208); background-color: inherit">可使用不同的 system prompt</span>，分别独立控制 **Thinker&#x20;**&#x7684;回复风格和 **Talker&#x20;**&#x7684;音频风格 &#x20;
>
> 4. **<span style="color: rgb(100,37,208); background-color: inherit">Talker</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 采用</span>**`multi-codebook`**<span style="color: rgb(100,37,208); background-color: inherit">的自回归方案</span>：每步生成一个 codec frame，其余 residual codebook &#x7531;**`MTP`**&#x6A21;块生成 &#x20;
>
> 5. **`Code2Wav`**<span style="color: rgb(100,37,208); background-color: inherit">是轻量级的 causal CNN</span>，简化了音频合成的最终阶段

训练和推理过程中，<span style="color: rgb(100,37,208); background-color: inherit">Talker 的输入来自 Thinker 的高维多模态特征，并共享完整的对话历史，因此可以作为统一的单体模型运行，支持端到端的训练和统一推理</span>。&#x20;

Qwen3-Omni-30B-A3B 各个组件的参数量如右表：

![](../../images/视觉多模态讲义（上）-image-173.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">音频编码器：</span> <span style="color: rgb(216,57,49); background-color: inherit">AuT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Au</span>**<span style="color: rgb(216,57,49); background-color: inherit">dio </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformer）  </span>

**`AuT`**<span style="color: rgb(100,37,208); background-color: inherit">是 Encoder-Decoder 结构的模型</span>，从头开始&#x5728;**`20M`**&#x5C0F;时的有监督音频数据上训练。训练时<span style="color: rgb(100,37,208); background-color: inherit">音频的滤波器组特征在进入注意力层前通过二维卷积下采样</span>**`8`**<span style="color: rgb(100,37,208); background-color: inherit">倍，将 token 速率降至</span>**`12.5 Hz`**。为了学到更好的音频表征，**<span style="color: rgb(100,37,208); background-color: inherit">AuT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 在大规模音频数据集上同时进行语音识别和音频理解任务训练</span>。具体而言，训练数据包&#x542B;**`80%`**&#x4E2D;英文伪标注 ASR 数据、**`10%`**&#x5176;他语言 ASR 数据，以&#x53CA;**`10%`**&#x97F3;频理解数据。为了平衡实时预填充缓存效率和离线音频任务性能，AuT 采用<span style="color: rgb(100,37,208); background-color: inherit">动态注意力窗口大小的 </span>**`Flash Attention`**，覆&#x76D6;**`1`**&#x81F3;**`8`**&#x79D2;范围内的注意力 Query 模式。这里采&#x7528;**`0.6B`**&#x53C2;数的 AuT 作为音频编码器。 &#x20;

![](../../images/视觉多模态讲义（上）-image-172.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">理解模块</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">文本、音频、图像和不带音频的视频</span>**：**Thinker** 将文本、音频、图像和视频转换为一系列的输入表征：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">文本</span>**：使&#x7528;**`Qwen`**&#x7CFB;列大模型的 tokenizer，**`BPE`**&#x7F16;码，词表大小&#x4E3A;**`151,643`**
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">音频及从视频中提取的音频</span>**：重采样&#x81F3;**`16 kHz`**，将原始波形转换&#x4E3A;**`128`**&#x901A;道梅尔谱图，&#x5373;**`25 ms`**&#x7A97;长，**`10 ms`**&#x6B65;长。采用上述 AuT 编码器，每个音频表征帧对应&#x7EA6;**`80 ms`**&#x539F;始音频信号
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">图像及视频</span>**：采&#x7528;**`Qwen3-VL`**&#x7684;视觉编码器，初始化&#x81EA;**`SigLIP2-So400m`**，参&#x6570;**`543M`**，视觉编码器<span style="color: rgb(100,37,208); background-color: inherit">在混合图像和视频的数据上训练，确保强大的图像理解和视频理解能力</span>。为在保持和音频采样率对齐的同时尽可能完整保留视频信息，以<span style="color: rgb(100,37,208); background-color: inherit">动态帧率采样视频帧</span>。 &#x20;

**<span style="color: rgb(222,120,2); background-color: inherit">多模态位置编码：</span> <span style="color: rgb(216,57,49); background-color: inherit">TM-RoPE</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ime-aligned </span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ultimodal </span>**<span style="color: rgb(216,57,49); background-color: inherit">Ro</span>**<span style="color: rgb(216,57,49); background-color: inherit">tary </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">osition </span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">mbedding）</span>

**Qwen3-Omni** 采用时间对齐的多模态旋转位置编&#x7801;**`TM-RoPE`**，是<span style="color: rgb(100,37,208); background-color: inherit">在多模态旋转位置编码</span>**`M-RoPE`**<span style="color: rgb(100,37,208); background-color: inherit">的基础上引入绝对时间信息。</span>**<span style="color: rgb(100,37,208); background-color: inherit">TM-RoPE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 将传统 </span>**<span style="color: rgb(100,37,208); background-color: inherit">RoPE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 分解为三个独立维度：</span>**`时间`**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**`高度`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`宽度`**。在原始 M-RoPE 中，时间维度使用&#x524D;**`16`**&#x4E2A;旋转角度建模，对应高频、强振荡模式。这种方法<span style="color: rgb(216,57,49); background-color: inherit">虽能有效捕捉细粒度局部时间变化，但可能阻碍模型在长序列上的外推能力</span>。为了解决这个问题，修改了旋转角度分配：**<span style="color: rgb(100,37,208); background-color: inherit">时间、高度和宽度分别交错分配</span>`24`<span style="color: rgb(100,37,208); background-color: inherit">、</span>`20`<span style="color: rgb(100,37,208); background-color: inherit">和</span>`20`<span style="color: rgb(100,37,208); background-color: inherit">个旋转角度</span>**。这<span style="color: rgb(46,161,33); background-color: inherit">促进了局部语义和长程依赖的更均衡表征，从而提升模型整体性能</span>。 &#x20;

TM-RoPE 可以根据输入模态定制： &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">文本</span>**：三个分量<span style="color: rgb(100,37,208); background-color: inherit">共享相同位置标识符</span>，功能上等价于一维 RoPE
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音频</span>**：使用<span style="color: rgb(100,37,208); background-color: inherit">共享的位置 ID 并附加绝对时间编码</span>，每个时间 ID 对&#x5E94;**`80 ms`**&#x65F6;长
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图像</span>**：<span style="color: rgb(100,37,208); background-color: inherit">所有视觉 token 分配恒定时间 ID</span>，行/列位置决定高度和宽度 ID
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音视频流</span>**：音频&#x6BCF;**`80 ms`**&#x7F16;码一个时间 ID。视频被视为帧序列，时间 ID 单调递增，并根据实际时间戳动态调整，确保每个 ID 对&#x5E94;**`80 ms`**&#x65F6;间分辨率。<span style="color: rgb(100,37,208); background-color: inherit">视频帧的高度和宽度 ID 分配方式和静态图像相同</span>。为避免多模态处理时的位置冲突，位置编号连续化：<span style="color: rgb(100,37,208); background-color: inherit">每个后续模态从上一模态最大位置 ID 加一开始编号</span>。&#x20;

> **<span style="color: rgb(100,37,208); background-color: inherit">Qwen3-Omni</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 使用时间 ID 直接对齐这些表征，这些时间 ID 锚定在绝对时间上</span>，让模型能够支持任意时长流式输入的灵活性。 这和 Qwen2.5-Omni 将音视频表征分割为固定 2 秒块不同。&#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">语音生成  </span>**

在多轮对话的语音合成中，**<span style="color: rgb(100,37,208); background-color: inherit">Talker</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 依赖于来自 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Thinker </span>**<span style="color: rgb(100,37,208); background-color: inherit">组件的丰富上下文，包括历史文本 token、多模态表征和当前轮次的流式文本</span>。这种对长上下文信息的依赖至关重要，因为<span style="color: rgb(100,37,208); background-color: inherit">高保真语音合成必须根据持续对话调整韵律、响度和情感等声学属性</span>，这也是上下文感知生成模型的基本原则。 &#x20;

**Qwen3-Omni&#x20;**&#x76F4;接在 RVQ token，Talker 采用分层预测方案：<span style="color: rgb(100,37,208); background-color: inherit">主干网络输入当前帧聚合的 codebook 特征，使用线性头预测第零个 codebook，随后由多 token 预测</span>**`MTP`**<span style="color: rgb(100,37,208); background-color: inherit">模块生成所有 residual codebook</span>，这使模型能学习完整的声学细节表征，增强语音表现力。相应地，波形重建简化为轻量级 casual CNN，&#x5373;**`Code2Wav`**，显著降低推理延迟和计算成本，同时相比更复杂&#x7684;**`DiT`**&#x58F0;码器实现更优音频保真度。 &#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">流式设计和并发设计  </span>**

![](../../images/视觉多模态讲义（上）-image-169.png)

首 token 生成时间是影响用户体验的关键因素，模型的并发能力对降低服务成本和提升响应速度至关重要。 &#x20;

**<span style="color: rgb(222,120,2); background-color: inherit">分块预填充</span>**：<span style="color: rgb(100,37,208); background-color: inherit">Qwen3-Omni 保留 Qwen2.5-Omni 的分块预填充机制，音频和视觉编码器沿时间维度输出分块</span>。在实时交互中，Thinker 和 Talker 模块异步执行预填充：<span style="color: rgb(100,37,208); background-color: inherit">当 Thinker 完成当前块预填充后，输出的高层表征立即用于异步预填充 Talker 的当前块，同时 Thinker 开始预填充下一数据块</span>。这显著降低 Thinker 和 Talker 的首 token 生成时间。 &#x20;

**<span style="color: rgb(222,120,2); background-color: inherit">MoE 架构</span>**：**Qwen3-Omni** 的 Thinker 和 Talker 都采用 MoE 设计，提升了服务吞吐量。<span style="color: rgb(46,161,33); background-color: inherit">相比 Dense 结构，MoE 架构显著减少长序列处理中由 KV 缓存引起的 IO 消耗，从而提升生成阶段每秒 token 数并增强并发能力</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">流式多 Codebook 的 Codec 生成</span>**：为最小化用户接收首个生成包的等待时间，**<span style="color: rgb(100,37,208); background-color: inherit">Qwen3-Omni </span>**<span style="color: rgb(100,37,208); background-color: inherit">仅依赖左侧上下文的多 codebook 生成机制，一旦 Talker 生成首个 token，MTP 模块即预测当前帧的其余 token</span>。这些 token 随后由流式多 codebook 的 codec 解码为波形，且解码器仅关注左侧上下文。

> Qwen3-Omni <span style="color: rgb(100,37,208); background-color: inherit">可在 Talker 生成每个 token 后立即输出波形，大幅降低首包延迟</span>。这是不同于 Qwen2.5-Omni 的，其需等待 Talker 提供足够块上下文才开始合成 &#x20;

**<span style="color: rgb(222,120,2); background-color: inherit">轻量级 MTP 模块</span>**：<span style="color: rgb(100,37,208); background-color: inherit">MTP 模块和 codec 解码器都是轻量级模块，计算 FLOPs 低且支持批处理推理，非常适合高并发场景</span>。MTP 是轻量的固定步长自回归稠密 Transformer，对推理硬件内存带宽需求低，支持高吞吐请求的高效批处理。其固定步长自回归推理机制可有效利用固定 KV 缓存内存空间加速，实现低推理延迟。

**<span style="color: rgb(222,120,2); background-color: inherit">卷积网络</span>**：同时，基于 CNN 的 codec decoder 在常见的推理平台都有硬件加速支持，可以实现高吞吐和低延迟，并支持高效批处理。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练  </span>**

Qwen3-Omni 在一个涵盖多种语言、方言及模态的多样化数据集上进行预训练，包&#x62EC;**<span style="color: rgb(100,37,208); background-color: inherit">图文</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">视频-文本</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">音频-文本</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">视频-音频</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">视频-音频-文本</span>**<span style="color: rgb(100,37,208); background-color: inherit">以及</span>**<span style="color: rgb(100,37,208); background-color: inherit">纯文本</span>**<span style="color: rgb(100,37,208); background-color: inherit">语料</span>。和 Qwen2.5-Omni 为每个任务使用单一 prompt 不同的是，<span style="color: rgb(100,37,208); background-color: inherit">Qwen3-Omni 采用更广泛的自然语言 prompt，以增强模型的泛化能力和指令遵循能力</span>。为实现所有模态上的稳健性能，训练策略从预训练早期阶段即融合单模态和跨模态数据。

![](../../images/视觉多模态讲义（上）-image-171.png)

Qwen3-Omni 的预训练分为三个阶段：

> **<span style="color: rgb(36,91,219); background-color: inherit">第一阶段（编码器对齐）</span>**：<span style="color: rgb(100,37,208); background-color: inherit">锁住 LLM 参数，训练视觉和音频编码器，利用海量音频-文本和图像-文本对提升 LLM 的语义理解能力</span>。在初始预训练阶段，Qwen3-Omni 的 LLM 使用 Qwen3 的参数初始化，视觉编码器取自 Qwen3-VL，音频编码器使用 AuT 初始化。<span style="color: rgb(100,37,208); background-color: inherit">两个编码器在固定 LLM 上分别训练，初期先训练各自的适配器，再训练编码器本身</span>。
>
> > Qwen3-Omni 放弃了之前工作中编码器和适配器在冻结 LLM 下联合训练的做法，因为其可能<span style="color: rgb(216,57,49); background-color: inherit">导致编码器补偿冻结 LLM 的局限性，从而削弱感知能力</span>。

> **<span style="color: rgb(36,91,219); background-color: inherit">第二阶段（通用阶段）</span>**：<span style="color: rgb(100,37,208); background-color: inherit">解冻所有参数</span>，使用更广泛的多模态数据进行更全面的学习。<span style="color: rgb(100,37,208); background-color: inherit">使用</span>**`2T`**<span style="color: rgb(100,37,208); background-color: inherit"> token 的大规模数据集，包括</span>**`0.57T`**<span style="color: rgb(100,37,208); background-color: inherit">的文本、</span>**`0.77T`**<span style="color: rgb(100,37,208); background-color: inherit">的音频、</span>**`0.82T`**<span style="color: rgb(100,37,208); background-color: inherit">的图像、</span>**`0.05T`**<span style="color: rgb(100,37,208); background-color: inherit">的视频、</span>**`0.05T`**<span style="color: rgb(100,37,208); background-color: inherit">视频-音频</span>。此阶段引入更多样化的多模态数据和任务，增强模型在听觉、视觉、文本及音视频信息中的理解和交互能力。

> **<span style="color: rgb(36,91,219); background-color: inherit">第三阶段（长上下文）</span>**：<span style="color: rgb(100,37,208); background-color: inherit">将最大 token 长度从</span>**`8,192`**<span style="color: rgb(100,37,208); background-color: inherit">扩展至</span>**`32,768`**<span style="color: rgb(100,37,208); background-color: inherit">，并提高训练数据中长音频和长视频的比例</span>，增强了模型对复杂长序列数据的理解能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练  </span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Thinker </span>**&#x20;

包含三阶段训练流程，使 Qwen3-Omni 具备指令遵循能力。数据集采&#x7528;**`ChatML`**&#x683C;式构建，包括<span style="color: rgb(100,37,208); background-color: inherit">纯文本对话数据、视觉模态对话数据、音频模态对话数据及混合模态对话数据</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">第一阶段 SFT</span>**：<span style="color: rgb(100,37,208); background-color: inherit">通过针对性指令优化减小预训练表征和下游任务需求之间的差距</span>。这里 SFT 其实是偏离预训练数据模式的，同时保持和预训练模型架构一致，实现高效知识迁移并保留预训练特征的完整性。

**<span style="color: rgb(222,120,2); background-color: inherit">第二阶段 Strong-to-Weak Distillation</span>**：<span style="color: rgb(100,37,208); background-color: inherit">采用 Qwen3 中的</span>**<span style="color: rgb(100,37,208); background-color: inherit">强到弱蒸馏</span>**<span style="color: rgb(100,37,208); background-color: inherit">，进一步提升模型性能</span>，包含两个阶段： &#x20;

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Off-policy Distillation</span>**：<span style="color: rgb(100,37,208); background-color: inherit">组合教师模型生成的输出后用于 response 蒸馏，帮助轻量级学生模型获得基础推理能力</span>，为后续在线策略训练奠定坚实基础&#x20;
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">On-policy Distillation</span>**：学生模型基于采样 prompt 生成 response，这些在线策略序列用于微调，<span style="color: rgb(100,37,208); background-color: inherit">通过最小化 KL 散度使学生模型的预测 logits 和教师模型对齐</span>，教师模型&#x4E3A;**`Qwen3-32B`**&#x6216;**`Qwen3-235B-A22B`**

**<span style="color: rgb(222,120,2); background-color: inherit">第三阶段 RL</span>**：利用 GSPO 全面增强模型在文本、图像、视频和音频等模态上的能力和稳定性。为上述模态提供反馈，这里采用两种 Reward： &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">Rule-based Reward</span>**：<span style="color: rgb(100,37,208); background-color: inherit">针对可验证的多模态任务，如数学、编程、指令遵循，奖励信号来自预定义规则集</span>。精心设计的规则奖励能高精度评估模型输出正确性，避免奖励作弊等问题。 &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Model-based Reward</span>**：<span style="color: rgb(100,37,208); background-color: inherit">针对缺乏客观预定义评估指标的多模态任务，使用 LLM 作为裁判。通用任务由</span>**`Qwen3`**<span style="color: rgb(100,37,208); background-color: inherit">评估，视觉相关任务由</span>**`Qwen2.5-VL`**<span style="color: rgb(100,37,208); background-color: inherit">评估</span>。为确保评估更稳健可靠，LLM 在适用情况下会获得对应的真实答案或参考答案。

* **<span style="color: rgb(36,91,219); background-color: inherit">Talker  </span>**

四阶段训练流程，使 Qwen3-Omni 能在生成文本的同时生成语音 response。所有训练数据均采&#x7528;**`ChatML`**&#x683C;式，确保和 Thinker 一致：

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：<span style="color: rgb(100,37,208); background-color: inherit">利用数亿条带有多模态上下文的语音数据训练 Talker</span>，建立从多模态表征到语音的单调映射
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：<span style="color: rgb(100,37,208); background-color: inherit">使用高质量数据进行持续预训练</span>，缓解第一阶段噪声数据引起的幻觉问题，显著提升生成语音质量。同时<span style="color: rgb(100,37,208); background-color: inherit">进行长上下文训练</span>，增强 Talker 处理长而复杂输入并生成上下文适配语音 response 的能力
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3</span>**：为提升多语言语音生成的泛化能力和系统稳定性，<span style="color: rgb(100,37,208); background-color: inherit">从多样化的多语言语音样本构建偏好对，并使用 DPO 优化模型</span>
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 4</span>**：<span style="color: rgb(100,37,208); background-color: inherit">进行 speaker fine-tuning，使 Talker 能采用特定音色</span>，同时优化语音响应的自然度、表现力和可控性

* **<span style="color: rgb(36,91,219); background-color: inherit">Captioner</span>**

Captioning 是多模态理解的一个基础任务，对大规模多模态模型的训练和评估至关重要。但<span style="color: rgb(216,57,49); background-color: inherit">现有研究绝大多数集中于视觉字幕，基本忽略了音频模态</span>。这一缺失意义重大，因为<span style="color: rgb(216,57,49); background-color: inherit">听觉感知是人类感官体验和世界交互的关键组成部分</span>。为填补这一空白并促进更全面的多模态感知研究，<span style="color: rgb(100,37,208); background-color: inherit">Qwen3-Omni 推出</span>**`Qwen3-Omni-30B-A3B-Captioner`**<span style="color: rgb(100,37,208); background-color: inherit">，通过对 Qwen3-Omni-30B-A3B 在大规模精细音频描述数据集上微调而得，可为任意音频输入生成细节丰富、低幻觉的字幕</span>。

### 2.6.7 <span style="color: rgb(36,91,219); background-color: inherit">Qwen3.5-Omni</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">架构</span>**

![](../../images/视觉多模态讲义（上）-image-174.png)

Qwen3.5-Omni 继续采用 **<span style="color: rgb(100,37,208); background-color: inherit">Thinker-Talker</span>&#x20;**&#x67B6;构。与 Qwen3-Omni 相比，Qwen3.5-Omni 在<span style="color: rgb(100,37,208); background-color: inherit">可扩展性、对齐性和实时交互</span>方面有几个创新点：

> * <span style="color: rgb(100,37,208); background-color: inherit">骨干网络采用 MoE 设计</span>，在提升可扩展性的同时，更好地平衡了多模态理解与生成之间的能力和效率。
>
> * Thinker 通过视觉编码器和音频 Transformer 分别接收视觉和音频信号。音频和视频输入交错排列以实现统一的多模态建模，并<span style="color: rgb(100,37,208); background-color: inherit">插入显式时间戳来改善时间感知</span>，特别是对于长视频或音视频上下文。这使 Thinker 能够处理扩展输入，支持高达 256k 个token、10 小时的音频或 400 秒 1FPS 的 720P 视频。
>
> * Talker 负责在结合 Thinker 文本输出的多模态输入条件下，进行上下文感知的语音生成。Qwen3.5-Omni <span style="color: rgb(100,37,208); background-color: inherit">采用 Qwen3-Omni 中引入的基于 RVQ 的语音表示，显著提升了推理效率</span>。
>
> * Qwen3.5-Omni <span style="color: rgb(100,37,208); background-color: inherit">在 Thinker 中采用分块流式输入处理，并采用流式 Talker 设计</span>，实现低延迟的端到端多模态对话。
>
> * Qwen3.5-Omni 中的 Talker <span style="color: rgb(100,37,208); background-color: inherit">采用 ARIA 在交错文本和语音单元之前动态对齐它们</span>。这缓解了由文本和语音分词器编码效率不匹配导致的不稳定性，从而减少了跳词、错误发音和模糊渲染等问题。

1. **<span style="color: rgb(36,91,219); background-color: inherit">AuT（Audio Transformer）</span>**

Qwen3.5-Omni 使用从头训练的基于 Transformer 的音频编码器，即注意力编码器-解码器模型 AuT，如右图。<span style="color: rgb(100,37,208); background-color: inherit">Qwen3.5-Omni 编码器的训练消耗了40M 小时的音频-文本对数据</span>，这些数据由 Qwen3-ASR 生成。音频的滤波器组特征<span style="color: rgb(100,37,208); background-color: inherit">通过 4 个 Conv2D 块进行 16 倍下采样，然后输入自注意力层，以 6.25Hz 的 token 速率获得音频 token</span>。与 Qwen3-Omni 编码器的训练过程相比，Qwen3.5-Omni 的编码器适应了更多超过 20 种语言的多语言数据，中文、英文和多语言数据的比例3.5:3.5:3。<span style="color: rgb(100,37,208); background-color: inherit">采用动态注意力窗口大小训练机制</span>，以保证在实时预填充缓存推理和离线音频理解下的平衡性能。

![](../../images/视觉多模态讲义（上）-image-164.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">感知模块</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">文本、音频、图像和视频</span>**

Thinker 将文本、音频、图像和无声视频输入转换为统一的表示序列：

> * **<span style="color: rgb(36,91,219); background-color: inherit">文本</span>**：使用 Qwen3.5 分词器，<span style="color: rgb(100,37,208); background-color: inherit">采用字节级字节对编码，词表大小为 250k</span>，在大多数语言中将编码和解码效率提高了10-60%。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">音频</span>**：<span style="color: rgb(100,37,208); background-color: inherit">将波形重采样至 16kHz，并使用 25ms 窗口和 10ms 跳步将其转换为 128 通道的梅尔频谱图</span>。使用 AuT 作为音频编码器，在 40M 小时的音频数据上从头训练，每个输出帧对应原始信号约 160ms。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉</span>**：<span style="color: rgb(100,37,208); background-color: inherit">采用 Qwen3.5 的视觉编码器来处理图像和视频</span>。其在图像和视频数据的混合上训练，在图像理解和视频理解方面能力比较强。为了在尽可能保留视频信息的同时保持与音频流的对齐，以动态帧率采样视频帧。

**<span style="color: rgb(222,120,2); background-color: inherit">音视频时间戳</span>**

按照 Qwen3-Omni 的模式，应用 TM-RoPE 赋予模型音视频同步的时间感知能力。但<span style="color: rgb(216,57,49); background-color: inherit">直接通过时间位置 ID 编码绝对时间会导致长视频或音视频输入的时间索引过于稀疏，削弱了长程时间建模能力</span>。此外，<span style="color: rgb(216,57,49); background-color: inherit">这种设计通常需要大规模且在不同帧率上均匀分布的训练样本，增加了数据构建成本</span>。为解决这些问题，Qwen3.5-Omni <span style="color: rgb(100,37,208); background-color: inherit">在每个视频或音视频时间块前添加一个以秒为单位的格式化文本字符串形式的显式时间戳，使模型能够更自然地学习时间码表示</span>。对于音频序列，<span style="color: rgb(100,37,208); background-color: inherit">在随机间隔插入时间戳以改善跨模态的时间对齐</span>。尽管这一策略略微增加了上下文长度，但它提供了更精确和鲁棒的时间感知，特别是对于长上下文多模态。

在多模态音视频流的上下文中，<span style="color: rgb(100,37,208); background-color: inherit">音频组件每 160ms 编码一个时间 ID</span>。视频被视为一系列帧，其时间 ID 单调递增，并根据实际时间戳动态调整，以确保每个 ID 具有一致的 160ms 时间分辨率。<span style="color: rgb(100,37,208); background-color: inherit">视频帧的高度和宽度 ID 的分配方式与静态图像相同</span>。为防止处理多种模态时出现位置冲突，位置编号是连续的，<span style="color: rgb(100,37,208); background-color: inherit">每个后续模态从前一模态的最大位置 ID 加一开始</span>。这种改进的位置编码方法使模型能够有效整合和联合建模来自不同模态的信息。Qwen3.5-Omni 使用它们的明确锚定到绝对时间的时间 ID来对齐这些表示。这一设计选择赋予模型支持任意流式输入的灵活性。

* **<span style="color: rgb(36,91,219); background-color: inherit">语音生成</span>**

Talker 直接操作由 Qwen3.5-Omni-Audio-Tokenizer 生成的 RVQ token。为了建模残差码本，<span style="color: rgb(100,37,208); background-color: inherit">采用 MTP 模块，实现对声学细节的细粒度建模和控制</span>。结合用于波形重建的因果 ConvNet，Talker 以低推理延迟和适度的计算开销提供高保真语音合成。在多轮口语对话中，Talker 以 Thinker 组件提供的丰富上下文信息为条件，包括历史文本token、多模态表示和当前轮次的流式文本。<span style="color: rgb(100,37,208); background-color: inherit">这种条件使 Talker 能够根据不断变化的对话上下文动态调节声学属性</span>，如<span style="color: rgb(220,155,4); background-color: inherit">韵律、响度和情感</span>。在架构设计上主要有以下几点：

> * <span style="color: rgb(100,37,208); background-color: inherit">为 Talker 引入了一个专门的 system prompt，用于指定目标语音特征，从而实现 zero-shot 语音克隆和可控语音生成</span>。与传统的说话人 Embedding 相比，这个 prompt 可以编码更丰富的多模态线索，包括文本描述和码本序列，对声学实现提供更细粒度的控制。
>
> * 提出 **<span style="color: rgb(216,57,49); background-color: inherit">ARIA</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">daptive </span>**<span style="color: rgb(216,57,49); background-color: inherit">R</span>**<span style="color: rgb(216,57,49); background-color: inherit">ate </span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">nterleave </span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">lignment）</span>，将传统的双通道生成范式统一为单通道形式。A<span style="color: rgb(100,37,208); background-color: inherit">RIA 不依赖 MFA 强制对齐导出的对齐或固定的交错速率，而是强制执行自适应速率约束：对于生成序列的任何前缀，累积的语音-文本 token 比率不得超过相应的项目级全局比率</span>。尽管设计简单，但这一设计在包括编码效率相对较低的语言在内的各种语言中提供了灵活的文本-语音对齐，并自然支持任意文本 token 前缀后跟连贯语音 token 后缀。

* **<span style="color: rgb(36,91,219); background-color: inherit">流式与并发</span>**

在流式音视频交互场景中，首包延迟是影响用户体验的关键因素，模型的并发能力是降低服务成本和提高回答速度的关键。

![Qwen3.5-Omni 的架构及音频/视频设置下的端到端首包延迟](../../images/视觉多模态讲义（上）-image-166.png)

**<span style="color: rgb(222,120,2); background-color: inherit">分块预填充与混合 MoE 架构</span>**

Qwen3.5-Omni 中<span style="color: rgb(100,37,208); background-color: inherit">保留了 Qwen3-Omni 和 Qwen2.5-Omni 中实现的分块预填充机制，音频和视觉编码器能够沿时间维度输出块</span>。这种方法显著降低了 Thinker 和 Talker 的首 token 时间 TTFT。在架构上，Qwen3.5-Omni 中的 <span style="color: rgb(100,37,208); background-color: inherit">Thinker 和 Talker 均基于 Qwen3.5 中引入的混合 MoE 架构构建</span>。除了混合 MoE 的一般效率优势外，还包括<span style="color: rgb(216,57,49); background-color: inherit"> </span>**<span style="color: rgb(216,57,49); background-color: inherit">GDN</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">ated </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">elta </span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">et）</span>模块，对于加速长音视频序列的建模特别有效。因此，它显著降低了长上下文推理中的 KV 缓存 I/O 开销，提高了生成吞吐量并实现更高的服务并发。

**<span style="color: rgb(222,120,2); background-color: inherit">基于 ARIA 的流式生成</span>**

对于流式语音生成和高并发服务，Qwen3.5-Omni 在很大程度上继承了 Qwen3-Omni 的高效设计：<span style="color: rgb(100,37,208); background-color: inherit">Talker 使用轻量级 MTP 模块预测 RVQ codec token，生成的 multi-codebook token 由因果流式 ConvNet codec decoder 转换为波形</span>。这些组件计算轻量、适合批处理，且非常适合低延迟部署。在这一共享基础上，ARIA 进一步将 Qwen3-Omni 中的双通道生成模式重构为文本和语音 token 上的统一交错单流形式。<span style="color: rgb(46,161,33); background-color: inherit">通过将文本和语音生成组织在单调交错约束下，ARIA 减少了独立生成轨道之间的同步开销，在解码期间实现更高效的 token 调度，并更好地匹配流式服务的自然增量机制</span>。

![不同并发水平下 Qwen3.5-Omni 的理论首包延迟。A表示音频，V表示视频](../../images/视觉多模态讲义（上）-image-168.png)

这个表是不同并发水平下 Qwen3.5-Omni 在音频和视频输入时的理论首包延迟，在 vLLM 上评估，MTP 模块和 codec decoder 启用了 torch.compile 和 CUDA Graph 加速。其中：

> * Thinker TTFT 表示从接收输入流到 Thinker 生成第一个文本 token 的时间
>
> * Talker TTFC 表示 Talker产生第一个音频块的时间
>
> * TPOP 表示稳态解码期间的每输出 token 延迟，其中 Talker TPOP 包括 Talker 骨干网络和MTP模块的联合延迟
>
> * TPS 表示生成吞吐量

由于 ARIA 将文本和语音生成组织在统一的交错流中，总体延迟不能简单地通过将若干行值相加获得，而是反映到第一个可播放音频包的端到端关键路径。由于 Qwen3.5-Omni-Flash 和 Qwen3.5-Omni-Plus 之间的规模差异显著，两个变体采用不同的部署时资源分配和并行化策略，因此它们的延迟和吞吐量数字不能严格的横向比较。<span style="color: rgb(46,161,33); background-color: inherit">Qwen3.5-Omni 在并发增加时保持稳定的延迟和解码效率，同时低生成 RTF 为平滑流式音频提供了充足的余量</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

Qwen3.5-Omni 在多样化的数据集上进行预训练，涵盖多种语言和方言以及多种模态，包括<span style="color: rgb(100,37,208); background-color: inherit">图像-文本、视频-文本、音频-文本、视频-音频、视频-音频-文本和纯文本语料库</span>。按照 Qwen3-Omni 的做法，Qwen3.5-Omni 采用更广泛的自然语言提示来增强泛化能力和指令遵循能力。为了在所有模态上实现鲁棒性能，<span style="color: rgb(100,37,208); background-color: inherit">训练策略从早期预训练阶段就纳入了单模态和跨模态数据</span>。Qwen3-Omni 采用 TM-RoPE 引入时间感知能力。但这种方法的两个关键局限性：

> * 通过直接将时间位置 ID 绑定到绝对时间，<span style="color: rgb(216,57,49); background-color: inherit">对于长音视频或视频输入会产生过大且稀疏的时间位置 ID</span>，削弱了模型捕获长程时间上下文的能力。
>
> * 有效学习通常需要大规模且在不同帧率上均匀分布的采样，<span style="color: rgb(216,57,49); background-color: inherit">显著增加了训练数据构建的成本</span>。

为解决这些问题，Qwen3.5-Omni <span style="color: rgb(100,37,208); background-color: inherit">在每个视频或音视频时间块前添加一个以秒为单位的格式化文本字符串形式的时间戳，使模型能够更好地学习和解释时间码表示</span>。对于音频序列，<span style="color: rgb(100,37,208); background-color: inherit">在随机间隔插入时间戳以更好地对齐不同模态的训练</span>。尽管这种方法适度增加了上下文长度，但它使模型能够更有效地感知时间信息。

Qwen3.5-Omni 的预训练分为三个不同的阶段：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage1：编码器对齐</span>**：锁定 LLM 参数，<span style="color: rgb(100,37,208); background-color: inherit">专注于训练视觉和音频编码器，利用大量的音频-文本和图像-文本对语料库来增强 LLM 内的语义理解</span>。Qwen3.5-Omni 的 LLM 组件使用 Qwen3.5 的参数初始化，视觉编码器采用 Qwen3.5 的编码器，音频编码器使用 AuT 初始化。两个编码器在固定的 LLM 上分别训练，最初都专注于训练各自的适配器，然后再训练编码器。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：通用阶段</span>**：解冻所有参数，<span style="color: rgb(100,37,208); background-color: inherit">使用更多的多模态数据进行更全面的学习</span>，序列长度为3276&#x38;**。**&#x9884;训练的第二阶段使用包含约 4T token的大规模数据集，分布为：<span style="color: rgb(100,37,208); background-color: inherit">文本 0.92T、音频 1.99T、图像 0.95T、视频 0.14T 和视频-音频 0.29T</span>。这个阶段引入更多样化的多模态数据和任务，增强模型在听觉、视觉、文本和视听信息方面的理解和交互能力。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：长上下文</span>：** <span style="color: rgb(100,37,208); background-color: inherit">将最大 token 长度从 32768 增加到 262144，并提高了训练数据中长音频和长视频的比例</span>。这些调整显著提升了模型理解长序列的能力。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Thinker</span>**

后训练阶段对 Thinker 采用三阶段策略，目的是保留模型在所有模态上的能力而不退化，确保在音频 Query 下的高回答质量，并优化整体交互体验。训练语料库按照 ChatML 格式组织，涵盖纯文本、视觉、音频和混合模态对话数据。包括以下阶段：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：专家蒸馏</span>。**&#x4E3A;了为全模态能力奠定坚实基础，首先<span style="color: rgb(100,37,208); background-color: inherit">通过独立的 SFT 和 RL 训练一套领域专门的教师模型，所有教师模型均从预训练的 Qwen-3.5 checkpoint 微调而来</span>。除了与文本相关的任务外，如<span style="color: rgb(220,155,4); background-color: inherit">智能体、编码和基础推理任务</span>，还为视觉和音频训练专门的教师模型。这些教师模型用于生成领域特定数据，使每个领域中学习的专门能力能够蒸馏到单一统一模型中。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：在线策略蒸馏</span>。**&#x901A;过上述训练，以音频 Query 为条件的回答质量与以文本 Query 为条件的回答质量之间仍存在显著差距。因此引入在线策略蒸馏，目标是将模型在文本输入下更强的回答能力蒸馏到音频输入设置中。<span style="color: rgb(100,37,208); background-color: inherit">对于每个音频-文本配对 Query ，首先获取在文本条件下生成的回答，这个回答通常在流畅性、推理和任务完成度方面表现出更高质量。然后将这个回答用作相应音频条件 Query 的蒸馏目标</span>。通过在此类在线策略目标上训练，模型逐渐将其音频条件输出与其文本条件行为对齐，从而提高音频输入下的回答质量并促进模态一致的生成。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：交互对齐强化学习</span>。**&#x524D;两个阶段显著提升了模型的领域能力和跨模态回答质量，但不足以充分优化模型用于现实世界交互使用。在多轮对话中，还有一些交互特定的问题，包括<span style="color: rgb(216,57,49); background-color: inherit">意外的语言代码切换、人格不一致以及扩展上下文中的指令遵循退化</span>。为缓解这些问题，引入交互对齐 RL，目标是优化交互质量。Qwen3.5-Omni 构建多轮交互轨迹，并围绕这些用户体验目标设计奖励信号，使模型能够学习在长时间交互中更稳定、一致和对齐的行为。通过明确优化交互质量，提高了模型在实际对话场景中的整体可用性。

* **<span style="color: rgb(36,91,219); background-color: inherit">Talker</span>**

Qwen3.5-Omni 对 Talker 采用四阶段训练流程，从而能够生成自然且上下文适当的口语回答，同时与文本联合生成。所有训练数据以 ChatML 格式组织，以与 Thinker 保持一致并便于语音控制。

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：通用阶段</span>**。在初始预训练阶段，<span style="color: rgb(100,37,208); background-color: inherit">在超过 20M 小时的多语言语音数据上训练 Qwen3.5-Omni，这些数据与多模态上下文配对</span>。引入更多样化的任务，如指令遵循语音生成，大大增强了上下文推理和副语言对齐，超越了从多模态表示到语音的简单单调映射。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：长上下文阶段</span>**。通过专门的数据筛选流程进行数据质量分层，并在高质量子集上进行持续预训练。<span style="color: rgb(100,37,208); background-color: inherit">借助 Qwen3-Omni-Captioner 的增强，缓解了初始预训练阶段由噪声数据引入的幻觉，并显著提高了生成语音的自然度和质量</span>。此外将最大上下文长度扩展到 64K token，使模型能够更好地处理长且复杂的用户输入，并产生更具上下文基础的语音回答。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：强化学习阶段</span>**。<span style="color: rgb(100,37,208); background-color: inherit">通过 DPO 进一步将模型行为与人类偏好对齐</span>。基于人工标注构建多语言偏好对，并使用 DPO 优化模型。这里纳入基于规则的奖励，并采用 GSPO 以进一步提高整体能力和跨多样任务的训练稳定性。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 4：说话人微调阶段</span>**。在基础模型之上<span style="color: rgb(100,37,208); background-color: inherit">进行轻量级说话人微调</span>，使 Qwen3.5-Omni 能够捕捉目标说话人特征，同时进一步提高其语音的自然度、表现力和可控性。

## 2.7 <span style="color: rgb(36,91,219); background-color: inherit">Intern 系列</span>

### 2.7.1 <span style="color: rgb(36,91,219); background-color: inherit">InternVL</span>

InternVL 是由上海 AI Lab 和相关研究机构开发的一系列开源多模态大模型，旨在结合视觉和语言能力以解决复杂的多模态任务。该系列模型在多个视觉-语言基准测试中展现了卓越性能，逐步成为 GPT-4V 等闭源模型的有力竞争者。

![](../../images/视觉多模态讲义（上）-image-167.png)

近年来，随着多模态任务需求的快速增长，VLM 的研究取得了显著进展。然而，<span style="color: rgb(216,57,49); background-color: inherit">与 LLM 相比，视觉编码器的进展相对缓慢，成为多模态模型性能提升的主要瓶颈之一</span>。为了解决这一问题，作者提出了一种新的方法，<span style="color: rgb(46,161,33); background-color: inherit">设计了大型的视觉语言基础模型 VL Foundation Model，通过将视觉编码器的参数规模扩展至</span>**`6B`**<span style="color: rgb(46,161,33); background-color: inherit">，显著增强了其表示能力</span>。在此基础上，模型通过逐步与大型语言模型对齐，提升了在多种视觉和语言任务中的表现，为多模态学习提供了更强大的技术支持。

* **<span style="color: rgb(36,91,219); background-color: inherit">总体架构</span>**

如下图所示，与传统的仅视觉主干网络和双编码器模型不同，**InternVL** 设计了一个视觉编码&#x5668;**`InternViT-6B`**&#x548C;一个语言中间&#x4EF6;**`QLLaMA`**。

> * **<span style="color: rgb(36,91,219); background-color: inherit">InternViT-6B</span>** 是一个具&#x6709;**`6B`**&#x53C2;数的视觉 Transformer，经过定制以在性能和效率之间实现良好的权衡。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">QLLaMA</span>&#x20;**&#x662F;一种具&#x6709;**`8B`**&#x53C2;数的语言中间件，初始化时使用了多语言增强版的 **LLaMA**。它可以为图像-文本对比学习提供强大的多语言表示，或者作为连接视觉编码器和现成的 LLM 解码器的桥梁。

为了弥合两个大规模组件在模态和结构上的显著差距，**<span style="color: rgb(100,37,208); background-color: inherit">InternVL</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 引入了一种渐进对齐训练策略</span>。该训练策略逐步进行，<span style="color: rgb(100,37,208); background-color: inherit">从大规模噪声数据上的对比学习开始，逐渐转向精致高质量数据上的生成式学习</span>。通过这种方式，确保了来自各种来源的网络规模图像-文本数据的有效组织和充分利用。然后，借助对齐的视觉编码器和语言中间件，**InternVL** 就像一把瑞士军刀。它具有灵活的组合能力，可以适应各种通用的视觉-语言任务。这些任务包括视觉感知、图像/视频-文本检索、图像描述生成、视觉问答以及多模态对话等。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型设计</span>**

![](../../images/视觉多模态讲义（上）-image-163.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：之前方法的问题 1）参数规模不匹配，导致 LLM 的能力没有被充分利用 2）表示的不一致 3）连接效率低下，连接层轻量且随机初始化
>
> * 参数平衡的视觉和语言组件：包&#x62EC;**`6B`**&#x7684;视觉编码器 **InternViT-6B**，**`8B`**&#x7684; LLM 中间件 QLLaMA
>
> * 渐进式的图像文本对齐策略：1）Q-Former 2）线性投影层 MLP

1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉编码器 InternViT-6B</span>**

作者使用标准的视觉 Transformer 实现了 **InternVL** 的视觉编码器。为了匹配 LLM 的规模，将视觉编码器扩展&#x5230;**`6B`**&#x53C2;数，从而得到&#x4E86;**`InternViT-6B`**&#x6A21;型。<span style="color: rgb(100,37,208); background-color: inherit">为了在准确性、速度和稳定性之间取得良好的权衡，作者对 </span>**<span style="color: rgb(100,37,208); background-color: inherit">InternViT-6B</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 进行了超参数搜索</span>，在以下范围内调整了模型深度$$\{32, 48, 64, 80\}$$、头维度$$\{64, 128\}$$和 MLP 比率$$\{4, 8\}$$。模型宽度和头数量根据给定的模型规模和其他超参数计算得出。

**InternVL** &#x5728;**`LAION-en`**&#x6570;据集&#x7684;**`100M`**&#x5B50;集上进行了对比学习实验，以评估不同配置下 **InternViT-6B** 变体的准确性、速度和稳定性。以下是主要发现。基于这些发现，确定最终模型的最稳定配置如右表所示：

![](../../images/视觉多模态讲义（上）-image-165.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">速度</span>**：对于不同的模型设置，当计算资源未饱和时，较浅的模型在每张图像上的处理速度更快。然而，当 GPU 计算资源被充分利用时，速度差异变得微不足道。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">准确性</span>**：在相同参数量的情况下，模型深度、头维度和 MLP 比率对性能的影响较小。

2. **<span style="color: rgb(36,91,219); background-color: inherit">语言中间件 QLLaMA</span>**

<span style="color: rgb(100,37,208); background-color: inherit">语言中间件 QLLaMA 旨在对齐视觉和语言特征</span>。如上图所示，QLLaMA 基于预训练的多语言 LLaMA 开发，并<span style="color: rgb(100,37,208); background-color: inherit">新增了</span>**`96`**<span style="color: rgb(100,37,208); background-color: inherit">个可学习 Query 和交叉注意力层，总计</span>**`1B`**<span style="color: rgb(100,37,208); background-color: inherit">参数</span>，这些参数是随机初始化的。这种方式<span style="color: rgb(46,161,33); background-color: inherit">使得 QLLaMA 能够平滑地将视觉元素集成到语言模型中，从而增强组合特征的连贯性和有效性</span>。

与流行的使用轻量级连接层的方法相比，QLLaMA 具有三个优势：

> 1. 通过使&#x7528;**`Chinese LLaMA`**&#x9884;训练权重初始化，QLLaMA 可以将 InternViT-6B 生成的图像 token 转换为与 LLM 对齐的表示
>
> 2. QLLaMA 拥&#x6709;**`8B`**&#x53C2;数用于视觉-语言对齐，其规模是 Q-Former &#x7684;**`42`**&#x500D;。因此，<span style="color: rgb(100,37,208); background-color: inherit">即使 LLM 解码器被冻结，InternVL 仍能在多模态对话任务中表现出色</span>
>
> 3) 可以应用于对比学习，为图像-文本对齐任务提供强大的文本表示，例如 <span style="color: rgb(220,155,4); background-color: inherit">Zero-Shot 图像分类和图像-文本检索</span>。

3. **<span style="color: rgb(36,91,219); background-color: inherit">瑞士军刀 InternVL</span>**

通过灵活组合视觉编码器和语言中间件，InternVL 可以支持多种视觉或视觉-语言任务：

![](../../images/视觉多模态讲义（上）-image-161.png)

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉感知任务</span>**：InternVL 的视觉编码器 InternViT-6B 可以用作视觉任务的主干网络。给定输入图像$$I \in \mathbb{R}^{H \times W \times 3}$$，模型可以<span style="color: rgb(100,37,208); background-color: inherit">生成特征图</span>$$F \in \mathbb{R}^{\frac{H}{14} \times \frac{W}{14} \times D}$$<span style="color: rgb(100,37,208); background-color: inherit">用于密集预测任务，或者结合全局平均池化和线性投影进行图像分类</span>。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">对比任务</span>**：如上图(a)(b)所示，可以组合出两种推理模式：**`InternVL-C`**&#x548C;**`InternVL-G`**，分别<span style="color: rgb(100,37,208); background-color: inherit">使用视觉编码器或将 InternViT 与 QLLaMA 结合来编码视觉特征</span>。具体来说，对 InternViT 的视觉特征或 QLLaMA 的 Query 特征应用注意力池化，以计算全局视觉特征$$I_f$$。此外，通过对 QLLaMA &#x7684;**`[EOS]`**&#x6807;记提取特征，将文本编码为$$T_f$$。通过计算$$I_f$$和$$T_f$$之间的相似性得分，支持各种对比任务，例如<span style="color: rgb(220,155,4); background-color: inherit">图像-文本检索</span>。
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">生成任务</span>**：与 Q-Former 不同，得益于参数规模增大，QLLaMA 本身具备出色的图像描述生成能力。<span style="color: rgb(100,37,208); background-color: inherit">QLLaMA 的 Query 重新组织来自 InternViT-6B 的视觉表示，并作为 QLLaMA 的前缀文本</span>。随后的文本 Token 则逐个顺序生成。
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">多模态对话</span>**：另一种模式称&#x4E3A;**`InternVL-Chat`**，利用 InternVL 作为视觉组件与 LLM 连接。为此，有两种配置方案：
>
>    1. 独立使用 InternViT-6B，如上图(c)所示
>
>    2. 同时使用完整的 InternVL 模型，如上图(d)所示

* **<span style="color: rgb(36,91,219); background-color: inherit">对齐策略</span>**

如上所述，**<span style="color: rgb(100,37,208); background-color: inherit">InternVL</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的训练分为三个渐进阶段，包括</span>**<span style="color: rgb(100,37,208); background-color: inherit">视觉-语言对比训练</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">视觉-语言生成训练</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">监督微调</span>**。这些阶段有效利用了来自不同来源的公开数据，从网络上的噪声图像-文本对到高质量的字幕、视觉问答 VQA 和多模态对话数据集。

1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉-语言对比训练</span>**

在第一阶段进行对比学习，以在大规模、噪声较大的图像-文本对上&#x5C06;**`InternViT-6B`**&#x4E0E;多语&#x8A00;**`LLaMA-7B`**&#x5BF9;齐。所有数据均为公开可用，并包含多语言内容，例&#x5982;**`LAION-en`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`LAION-multi`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`LAION-COCO`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`COYO`**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**`Wukong`**<span style="color: rgb(220,155,4); background-color: inherit">等</span>。作者使用这些数据集的组合，并过滤掉一些极低质量的数据来训练模型。如下表所示，原始数据集包&#x542B;**`6.03B`**&#x4E2A;图像-文本对，清洗后剩&#x4F59;**`4.98B`**&#x5BF9;。

在训练过程中，使用 LLaMA-7B 对文本进行编码为$$T_f$$，并用 InternViT-6B 提取视觉特征$$I_f$$。这里使用<span style="color: rgb(100,37,208); background-color: inherit"> CLIP 的目标函数，在批次中的图像-文本对相似性得分上最小化对称交叉熵损失</span>。<span style="color: rgb(46,161,33); background-color: inherit">这一阶段使 InternVL 在 Zero-Shot 图像分类和图像-文本检索等对比任务中表现出色，同时该阶段的视觉编码器在语义分割等视觉感知任务中也表现良好</span>。

![](../../images/视觉多模态讲义（上）-image-160.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉-语言生成训练</span>**

第二阶段将 InternViT-6B 与 QLLaMA 连接，并采用生成式训练策略。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">QLLaMA 使用第一阶段 LLaMA-7B 的权重，并保持 InternViT-6B 和 QLLaMA 冻结状态，仅使用经过筛选的高质量数据训练新增的可学习 Query 和交叉注意力层</span>。从上表可以看出，这一阶段进一步过滤掉了低质量 caption 数据，将数据量从第一阶段&#x7684;**`4.98B`**&#x51CF;少&#x5230;**`1.03B`**。

这一阶段使用 BLIP-2 的损失函数，计算为三个组成部分的总和：图像-文本对&#x6BD4;**`ITC`**&#x635F;失、图像-文本匹&#x914D;**`ITM`**&#x635F;失和基于图像的文本生&#x6210;**`ITG`**&#x635F;失。这使得 <span style="color: rgb(46,161,33); background-color: inherit">Query 能够提取强大的视觉表示，并进一步与 LLM 对齐，得益于有效的训练目标以及大规模、基于 LLM 初始化的 QLLaMA 的使用</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">监督微调</span>**

为了展示 InternVL 在构建多模态对话系统中的优势，作者通过一&#x4E2A;**`MLP`**&#x5C42;将其与现成的大语言模型解码器连接，例&#x5982;**`Vicuna`**<span style="color: rgb(220,155,4); background-color: inherit">或</span>**`InternLM`**，并进行监督微调 SFT。如右表所示，作者收集了广泛的高质量指令数据，总计&#x7EA6;**`4M`**&#x6837;本。

![](../../images/视觉多模态讲义（上）-image-162.png)

由于 QLLaMA 和 LLM 的特征空间相似，<span style="color: rgb(46,161,33); background-color: inherit">即使冻结 LLM 解码器，也可以通过仅训练 MLP 层或同时训练 MLP 层和 QLLaMA 来实现稳健的性能。这种方法不仅加速了 SFT 过程，还保留了 LLM 的原始语言能力</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

InternVL 是由上海 AI Lab 开发的开源多模态大模型系列，参数规模达6B，具备强大的视觉和语言理解能力。它在OCR、文档理解、视觉问答 VQA 等任务中表现卓越，并支持 Zero-Shot 学习和高分辨率图像处理。通过持续优化和与 LLM 对齐，InternVL 在性能上逐步比肩 GPT-4V 等闭源模型，成为多模态领域的领先开源解决方案。

### 2.7.2 <span style="color: rgb(36,91,219); background-color: inherit">InternVL1.5</span>

InternVL 1.5 是由上海人工智能实验室发布的一款开源多模态大语言模型，旨在缩小开源模型与专有商业模型（如 GPT-4V）在多模态理解能力上的性能差距 。该模型通过整合强大的视觉编码器 InternViT-6B 和持续学习能力，结合动态高分辨率策略以及高质量的双语数据集，显著提升了其在跨模态任务中的表现 。此外，InternVL 1.5 还采用了渐进式的图像-文本对齐策略，利用海量带噪声的图文数据进行对比学习预训练，并在第二阶段使用过滤后的高质量数据完成生成式预训练。总的来说，InternVL 1.5 相比 InternVL 有以下三个改进：

> 1. 更强的视觉编码器
>
> 2. 动态高分辨率，支持最高4K
>
> 3. 高质量双语数据集，增强了 OCR 能力和中文能力

* **<span style="color: rgb(36,91,219); background-color: inherit">整体架构</span>**

如右图所示，InternVL1.5 采用了一种类似于广泛使用的多模态模型架构，具体来说&#x662F;**`ViT-MLP-LLM`**&#x67B6;构。具体实现是将预训练&#x7684;**`InternViT-6B`**&#x4E0E;预训练&#x7684;**`InternLM2-20B`**&#x901A;过一个随机初始化&#x7684;**`MLP`**&#x6295;影层集成在一起。

在训练过程中，实施了一种动态分辨率策略，根据输入图像的宽高比和分辨率，将图像划分为大小&#x4E3A;**`448×448`**&#x50CF;素的块，数量&#x4ECE;**`1`**&#x5230;**`12`**&#x4E0D;等。在测试阶段，这种方法可以扩展&#x5230;**`40`**&#x4E2A;块，&#x5373;**`4K`**&#x5206;辨率。

![](../../images/视觉多模态讲义（上）-image-189.png)

为了增强对高分辨率的支持，这里使用了像素重排操作，将视觉 token 的数量减少到原来的四分之一。因此一&#x4E2A;**`448×448`**&#x7684;图像&#x7531;**`256`**&#x4E2A;视觉 token 表示。

* **<span style="color: rgb(36,91,219); background-color: inherit">更强的视觉编码器</span>**

在现有的多模态大语言模型中，最常用的视觉基础模型通常是对比预训练的 ViT。然而，<span style="color: rgb(216,57,49); background-color: inherit">这些 ViT 通常是在从互联网抓取的固定低分辨率图像-文本对上进行训练的，例如</span>**`224×224`**<span style="color: rgb(216,57,49); background-color: inherit">，因此在处理高分辨率图像或来自非互联网来源的图像时，如</span> <span style="color: rgb(220,155,4); background-color: inherit">文档图像</span> <span style="color: rgb(216,57,49); background-color: inherit">，其性能会下降</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">InternViT-6B-448px-V1.2</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

为了解决这一问题，**`InternVL 1.2`**&#x5BF9;**`InternViT-6B`**&#x6301;续预训练。首先，作者发现倒数第四层的特征在多模态任务中表现最佳，因此直接丢弃了最后三层的权重，将 InternViT-6B &#x4ECE;**`48`**&#x5C42;减少&#x5230;**`45`**&#x5C42;。然后将 InternViT-6B 的分辨率&#x4ECE;**`224`**&#x63D0;升&#x5230;**`448`**，并将其&#x4E0E;**`Nous-Hermes-2-Yi-34B`**&#x96C6;成。为了使模型具备高分辨率处理和 OCR 能力，在训练中同时激活了视觉编码器和 MLP，利用了图像描述和 OCR 特定数据集。此过程生成的新 InternViT 权重被发布&#x4E3A;**`InternViT-6B-448px-V1.2`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">InternViT-6B-448px-V1.5</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

**`InternVL 1.5`**&#x57FA;&#x4E8E;**`InternViT-6B-448px-V1.2`**&#x8FDB;行预训练。训练图像的分辨率从<span style="color: rgb(100,37,208); background-color: inherit">固定的</span>**`448×448`**&#x6269;展为<span style="color: rgb(100,37,208); background-color: inherit">动态的</span>**`448×448`**，其中<span style="color: rgb(100,37,208); background-color: inherit">基本块大小为 448×448，块的数量范围为 1 到 12</span>。除此之外还增强了预训练数据集的规模、质量和多样性，从而使得 1.5 版本模型具有强大的鲁棒性、OCR 能力和高分辨率处理能力。

InternVL 1.5 中的 LLM &#x4ECE;**`Nous-Hermes-2-Yi-34B`**&#x66F4;换&#x4E3A;**`InternLM2-20B`**，并与其保持了兼容性和可移植性。这说明 <span style="color: rgb(46,161,33); background-color: inherit">InternViT-6B 在多模态大语言模型预训练阶段学习到的视觉特征具有广泛的适用性，而不仅仅局限于特定的 LLM</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">动态高分辨率</span>**

InternVL 1.5 采用了一种动态高分辨率训练方法，能够有效适应输入图像的不同分辨率和宽高比。该方法<span style="color: rgb(46,161,33); background-color: inherit">利用了将图像分割为块的灵活性，增强了模型处理细节视觉信息的能力，同时适应多样化的图像分辨率</span>。它主要包含以下步骤：

1. **<span style="color: rgb(36,91,219); background-color: inherit">动态宽高比匹配</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

如右图所示，为了在处理过程中保持自然的宽高比，作者<span style="color: rgb(100,37,208); background-color: inherit">动态匹配一组预定义宽高比中的最佳值</span>。由于计算资源有限，<span style="color: rgb(100,37,208); background-color: inherit">在训练期间最多允许</span>**`12`**<span style="color: rgb(100,37,208); background-color: inherit">个块</span>。因此，<span style="color: rgb(100,37,208); background-color: inherit">这组宽高比包括由</span>**`1`**<span style="color: rgb(100,37,208); background-color: inherit">到</span>**`12`**<span style="color: rgb(100,37,208); background-color: inherit">个块形成的所有</span>**`35`**<span style="color: rgb(100,37,208); background-color: inherit">种可能的组合</span>，例&#x5982;**`{1:1, 1:2, 2:1, 3:1, ..., 2:6}`**。在匹配过程中，对于<span style="color: rgb(100,37,208); background-color: inherit">每个输入图像，计算其宽高比，并通过测量绝对差值与35个预定义宽高比进行比较</span>。如果有多个预定义宽高比匹配，例&#x5982;**`1:1`**&#x548C;**`2:2`**，会<span style="color: rgb(100,37,208); background-color: inherit">优先选择不超过输入图像面积两倍的宽高比，从而避免低分辨率图像过度放大</span>。

![](../../images/视觉多模态讲义（上）-image-184.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">图像分割与缩略图</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

一旦确定了合适的宽高比，图像将被调整为相应的分辨率。例如<span style="color: rgb(220,155,4); background-color: inherit">一张</span>**`800×1300`**<span style="color: rgb(220,155,4); background-color: inherit">的图像将被调整为</span>**`896×1344`**。调整后的图像随后被划分&#x4E3A;**`448×448`**&#x50CF;素的块。除了这些块外，输入还包含整个图像的缩略图以捕捉全局上下文。该缩略图被缩小&#x5230;**`448×448`**，帮助模型理解整体场景。因此，在训练期间，视觉 token 的数量范围&#x4E3A;**`256`**&#x5230;**`3,328`**。在测试阶段，块的数量可以增加到最&#x591A;**`40`**&#x4E2A;，从而产&#x751F;**`10,496`**&#x4E2A;视觉 token。

* **<span style="color: rgb(36,91,219); background-color: inherit">高质量双语数据集</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">预训练数据集</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

InternVL 1.5 的预训练数据集来自多种公开资源，涵盖多个任务领域。

> * **<span style="color: rgb(36,91,219); background-color: inherit">图像描述任务</span>**：&#x5360;**`53.9%`**，主要使用 Laion-EN、Laion-ZH、COYO 和 GRIT 等数据集
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">检测与定位任务</span>**：&#x5360;**`5.2%`**，包括 Objects365、GRIT 和 All-Seeing 等数据集
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">OCR 任务</span>**：&#x5360;**`40.9%`**，通过 Wukong-OCR、LaionCOCO-OCR 和 Common Crawl PDFs 等大规模数据集，以及 MMC-Inst、LSVT、ST-VQA 等小规模数据集进行训练

这种多样化的数据组合<span style="color: rgb(46,161,33); background-color: inherit">确保了模型在多模态任务中的鲁棒性</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">微调数据集</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

在微调阶段，作者精心挑选数据集以提升模型性能。以下是主要任务及其对应的数据集：

> **<span style="color: rgb(36,91,219); background-color: inherit">图像描述</span>**：TextCaps、双语 ShareGPT4V

> **<span style="color: rgb(36,91,219); background-color: inherit">通用问答</span>**：VQAv2、GQA、VisualDialog

> **<span style="color: rgb(36,91,219); background-color: inherit">科学图像理解</span>**：AI2D、ScienceQA、TQA

> **<span style="color: rgb(36,91,219); background-color: inherit">图表理解</span>**：ChartQA、MMC-Inst、PlotQA

> **<span style="color: rgb(36,91,219); background-color: inherit">数学问题</span>**：GeoQA+、TabMWP、MathQA&#x20;

> **<span style="color: rgb(36,91,219); background-color: inherit">知识问答</span>**：KVQA、双语维基百科

> **<span style="color: rgb(36,91,219); background-color: inherit">OCR 任务</span>**：OCRVQA、TextVQA、SynthDoG &#x20;

> **<span style="color: rgb(36,91,219); background-color: inherit">文档理解</span>**：DocVQA、Common Crawl PDFs

> **<span style="color: rgb(36,91,219); background-color: inherit">视觉定位</span>**：RefCOCO、Visual Genome

> **<span style="color: rgb(36,91,219); background-color: inherit">多模态对话</span>**：LLaVA-150K、ALLaVA

> **<span style="color: rgb(36,91,219); background-color: inherit">纯文本任务</span>**：OpenHermes2.5、AlpacaGPT4 等用于保留语言能力

这些数据集为模型<span style="color: rgb(46,161,33); background-color: inherit">提供了丰富的多模态训练基础，使其能够应对多样化任务并适应实际应用需求</span>。

3. **<span style="color: rgb(36,91,219); background-color: inherit">数据翻译流程</span>**

为了增强模型的多语言能力，作者设计了一个自动化翻译流程。该流程利用开源大语言模型或 GPT-3.5 将英文数据集翻译为中文，确保双语标注的一致性和准确性。通过调整语言提示，这一流程还可以扩展到其他语言，无需人工干预。 &#x20;

例如，在上面提到的数据集种，<span style="color: rgb(220,155,4); background-color: inherit">有一部分原本为英文，如</span>**`COYO`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`GRIT`**<span style="color: rgb(220,155,4); background-color: inherit">，会通过翻译流程转换为中文</span>。这显著提升了InternVL 1.5在中文任务中的表现。

![](../../images/视觉多模态讲义（上）-image-185.png)

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

InternVL 1.5 在多模态理解领域展现了卓越的能力，特别是在文本与图像结合的任务中表现出色，如阅读理解和文档理解。其灵活的分辨率支持、动态图块划分技术以及针对中文优化的理解能力，使其成为一款性能比肩商业模型的开源黑马。通过多项改进，InternVL 1.5 不仅提升了任务完成的准确度和效率，还为开源社区提供了探索多模态大模型成长的重要参考。

### 2.7.3 <span style="color: rgb(36,91,219); background-color: inherit">InternVL2</span>

InternVL 2 是 OpenGVLab 推出的多模态大型语言模型系列中的一个重要版本，标志着在多模态处理能力上的显著进步。该模型通过结合视觉编码器和语言模型，能够高效处理图像、文本等多种模态数据。其设计目标是为学术研究和产业应用提供一个强大的开源多模态基座模型。InternVL 2.0 的推出奠定了后续版本的基础，特别是在动态高分辨率训练方法的引入上，显著提升了模型对复杂多模态任务的适应能力。

![](../../images/视觉多模态讲义（上）-image-186.png)

InternVL2 系列包括从适用于边缘设备&#x7684;**`1B`**&#x6A21;型到功能更强大&#x7684;**`108B`**&#x6A21;型。 凭借更大规模的语言模型，InternVL2-Pro 展现出出色的多模态理解能力，在各种基准测试中与商业闭源模型的性能相当。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

InternVL2 系列基于以下设计：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">渐进式地与大型语言模型对齐</span>**：引入了渐进式对齐训练策略，从而实现了第一个与大型语言模型原生对齐的视觉基础模型。通过采用渐进式训练策略，模型从小到大，数据从粗到细，<span style="color: rgb(46,161,33); background-color: inherit">以相对较低的成本完成了大型模型的训练。这种方法在有限的资源下表现出色</span>。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">多模式输入</span>**：通过一组参数，InternVL2 <span style="color: rgb(46,161,33); background-color: inherit">支持多种输入模式，包括文本、图像、视频和医疗数据</span>。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">多任务输出</span>**：InternVL2 支持各种输出格式，例如<span style="color: rgb(220,155,4); background-color: inherit">图像、边界框和蒙版</span>，展现出广泛的多功能性。通过将 MLLM 与多个下游任务解码器连接起来，<span style="color: rgb(46,161,33); background-color: inherit">InternVL2 可以推广到数百个视觉语言任务，同时实现与专家模型相当的性能</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">性能</span>**

InternVL2 在处理复杂多模态数据方面表现出色，在数学、科学图表、通用图表、文档、信息图表和 OCR 等任务中表现出色。例如，<span style="color: rgb(46,161,33); background-color: inherit">InternVL2 在 MathVista 基准上实现了</span>**`66.3%`**<span style="color: rgb(46,161,33); background-color: inherit">的准确率，大幅超越其他闭源商业模型和开源模型</span>。此外，<span style="color: rgb(46,161,33); background-color: inherit">InternVL2 在通用图表基准 ChartQA、文档基准 DocVQA、信息图表基准 InfographicVQA 和通用视觉问答基准 MMBench 等一系列基准测试中均取得了最佳性能</span>。

AI2D 基准测试中有两种评估设置：

> 1. 设置一，将图片中矩形框内的内容替换为选项的字母。InternVL2 获得&#x4E86;**`87.3`**&#x7684;性能
>
> 2. 设置二，将矩形框内的内容替换为选项的字母和选项的值。InternVL2 获得&#x4E86;**`96.0`**&#x7684;性能

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

InternVL 2 是一个多模态大型语言模型的承上启下的工作，通过优化视觉与语言的融合能力，为后续版本的迭代提供了坚实的技术基础。它的开源特性也推动了多模态领域的研究与发展。

### 2.7.4 <span style="color: rgb(36,91,219); background-color: inherit">InternVL2.5</span>

InternVL 2.5 是 InternVL 系列的进一步升级版本，代表了当前开源多模态模型的顶尖水平。<span style="color: rgb(46,161,33); background-color: inherit">相比于 InternVL 2.0，2.5 版本在训练和测试策略、数据质量以及模型性能上进行了显著增强</span>。InternVL 2.5 在处理多图像和视频数据集方面表现尤为突出，得益于扩展的动态高分辨率训练方法。此外，它是<span style="color: rgb(46,161,33); background-color: inherit">首个在 MMMU 多模态理解基准上得分超过</span>**`70%`**<span style="color: rgb(46,161,33); background-color: inherit">的开源模型，展现了其在多模态任务中的卓越性能</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">整体架构</span>**

如下图所示，InternVL 2.5 保留了与其前代模&#x578B;**`InternVL 1.5`**&#x548C;**`InternVL 2.0`**&#x76F8;同的模型架构，遵循在多种多模态研究中广泛采用&#x7684;**`ViT-MLP-LLM`**&#x8303;式。 &#x20;

![](../../images/视觉多模态讲义（上）-image-188.png)

在这一新版本中，作者<span style="color: rgb(100,37,208); background-color: inherit">对该架构的实现整合了一个新近增量预训练的</span>**`InternViT-6B`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`InternViT-300M`**<span style="color: rgb(100,37,208); background-color: inherit">，并结合了各种不同大小和类型的预训练 LLM，包括</span>**`InternLM 2.5`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`Qwen 2.5`**<span style="color: rgb(100,37,208); background-color: inherit">，通过一个随机初始化的两层</span>**`MLP`**<span style="color: rgb(100,37,208); background-color: inherit">投影层进行连接</span>。与之前的版本一样，为了增强高分辨率处理的可扩展性，简单地应用了像素重排操作，将视觉 token 的数量减少到原始数量的四分之一。因此一&#x4E2A;**`448×448`**&#x7684;图像块&#x7531;**`256`**&#x4E2A;视觉 token 表示。 &#x20;

在输入数据预处理方面，采用了与 InternVL 1.5 类似的动态分辨率策略，根据输入图像的宽高比和分辨率将图像划分&#x4E3A;**`448×448`**&#x50CF;素的块。关键的区别在于，<span style="color: rgb(100,37,208); background-color: inherit">从 InternVL 2.0 开始，额外引入了对多图像和视频数据的支持</span>，如图上图所示。

**<span style="color: rgb(36,91,219); background-color: inherit">视觉编码器</span>**

InternVL 使用 InternViT 作为视觉编码器。目前，InternViT 有两种不同的模型规模，分别&#x662F;**`InternViT-6B`**&#x548C;**`InternViT-300M`**。&#x20;

![](../../images/视觉多模态讲义（上）-image-182.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">InternViT-6B</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

**`InternViT-6B-224px`**&#x7ED3;构基于标准的 ViT，仅做了一些小调整，包括引&#x5165;**`QK-Norm`**&#x548C;**`RMSNorm`**。该模型拥&#x6709;**`5.9B`**&#x53C2;数、**`48`**&#x5C42;、隐藏层大小&#x4E3A;**`3200`**、**`25`**&#x4E2A;注意力头，并使用对比损失进行训练。由于当时收益有限，因此采用了增量预训练策略以持续优化其权重。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">将 InternViT-6B 通过一个 MLP 投影层连接到一个 LLM，通过 </span>**<span style="color: rgb(216,57,49); background-color: inherit">NTP（N</span>**<span style="color: rgb(216,57,49); background-color: inherit">ext Token Prediction）</span> <span style="color: rgb(100,37,208); background-color: inherit">损失联合训练 InternViT-6B</span>，以增强其视觉特征提取能力。&#x5728;**`V1.0`**&#x548C;**`V1.2`**&#x7248;本中，作者使用了固定&#x7684;**`448×448`**&#x5206;辨率进行训练，但<span style="color: rgb(100,37,208); background-color: inherit">在后续版本中切换为动态分辨率训练</span>，以提升高分辨率处理能力。这里<span style="color: rgb(100,37,208); background-color: inherit">延续 InternVL 1.5 的设定，移除了</span>**`InternViT-6B-448px-V1.2`**<span style="color: rgb(100,37,208); background-color: inherit">的最后三层，将其深度从</span>**`48`**<span style="color: rgb(100,37,208); background-color: inherit">层减少到</span>**`45`**<span style="color: rgb(100,37,208); background-color: inherit">层，因为这些层更倾向于适应 CLIP 损失目标，优先考虑全局对齐而非局部信息</span>。因此，所有后续版本都具&#x6709;**`45`**&#x5C42;&#x548C;**`5.5B`**&#x53C2;数，包括最新&#x7684;**`InternViT-6B-448px-V2.5`**。 &#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">InternViT-300M</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

**`InternViT-300M-448px-Distill`**&#x662F;教师模&#x578B;**`InternViT-6B-448px-V1.5`**&#x7684;蒸馏变体，使用余弦蒸馏损失进行训练。该模型包&#x542B;**`0.3B`**&#x53C2;数、**`24`**&#x5C42;、隐藏层大小&#x4E3A;**`1024`**&#x4EE5;&#x53CA;**`16`**&#x4E2A;注意力头。<span style="color: rgb(100,37,208); background-color: inherit">与 6B 版本不同，</span>**`0.3B`**<span style="color: rgb(100,37,208); background-color: inherit">版本使用标准的</span>**`LayerNorm`**<span style="color: rgb(100,37,208); background-color: inherit">而未采用QK-Norm。为了降低蒸馏成本，使用</span>**`CLIP-ViT-Large-336px`**<span style="color: rgb(100,37,208); background-color: inherit">初始化该模型，尽管存在一些架构差异</span>。蒸馏完成后，将该模型与一个 LLM 集成，并按照类似上述的流程，<span style="color: rgb(100,37,208); background-color: inherit">通过动态高分辨率和 NTP 损失训练视觉编码器</span>。随后，提取了视觉编码器并将其发布&#x4E3A;**`InternViT-300M-448px`**。在这个基础上，<span style="color: rgb(100,37,208); background-color: inherit">通过在更多样化的数据上进一步增量预训练之前的权重，使用 NTP 损失优化了 InternViT-300M，从而生成了增强版</span>**`InternViT-300M-448px-V2.5`**。 &#x20;

**<span style="color: rgb(36,91,219); background-color: inherit">大语言模型</span>**

在 InternVL 2.5 系列中，将语言模型骨干网络更新为最先进模型，包&#x62EC;**`InternLM 2.5`**&#x548C;**`Qwen 2.5`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">多模态数据的动态高分辨率处理</span>**

在 InternVL 2.0 和 2.5 中，作者扩展了 InternVL 1.5 中引入的动态高分辨率训练方法，增强了其处理多图像和视频数据集的能力。该过程主要包括以下步骤： &#x20;

1. **<span style="color: rgb(36,91,219); background-color: inherit">匹配最接近的宽高比</span>** &#x20;

给定输入图像$$I$$的尺寸为$$W \times H$$，宽高比计算为$$r = \frac{W}{H}$$。目标是将图像调整为大小为$$S \times S$$（其中$$S = 448$$）的块，同时选择使失真最小化的最接近的宽高比。块的数量$$n_{\text{tiles}}$$被限制在预定义范围$$[n_{\text{min}}, n_{\text{max}}]$$内。 &#x20;

为了找到最佳的宽高比进行调整，将目标宽高比集合$$R$$定义为： &#x20;

$$R = \left\{ \frac{i}{j} \mid 1 \leq i, j \leq n, \, i \times j \in [n_{\text{min}}, n_{\text{max}}] \right\}$$

通过最小化原始宽高比$$r$$与每个目标宽高比$$r_{\text{target}}$$的差异，选择最接近的宽高比$$r_{\text{best}}$$： &#x20;

$$r_{\text{best}} = \arg\min_{r_{\text{target}} \in R} |r - r_{\text{target}}|$$

如果多个宽高比产生相同的差异，例如，$$1:2$$和$$2:4$$，则优先选择面积小于或等于原图像两倍的宽高比。这在一定程度上防止了低分辨率图像的过度放大。 &#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">图像调整和分割</span>** &#x20;

一旦确定了最佳宽高比，图像将被调整为新的尺寸$$W_{\text{new}} \times H_{\text{new}}$$，其中$$i_{\text{best}}$$和$$j_{\text{best}}$$是对应于$$r_{\text{best}}$$的因子： &#x20;

$$W_{\text{new}} = S \times i_{\text{best}}, \quad H_{\text{new}} = S \times j_{\text{best}}$$

然后将图像分割为大小为$$S \times S$$的块，块的数量计算为$$n_{\text{tiles}} = i_{\text{best}} \times j_{\text{best}}$$。每个块从调整后的图像中裁剪出来以确保尺寸一致。 &#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">缩略图生成</span>**

如果块的数量$$n_{\text{tiles}} > 1$$，则将原始图像$$I$$调整为$$S \times S$$的正方形以生成额外的缩略图$$I_{\text{thumb}}$$。该缩略图附加到块列表中，提供全局视图以补充局部块。如果$$n_{\text{tiles}} = 1$$，则无需生成缩略图，此步骤自然跳过。 &#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">不同数据类型的数据格式</span>** &#x20;

![](../../images/视觉多模态讲义（上）-image-181.png)

如上图所示，InternVL 2.0 和 2.5 中的动态高分辨率方法不仅支持单图像数据集，还扩展到了多图像和视频数据集。 &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">单图像数据集</span>**：最大块数$$n_{\text{max}}$$分配给单个图像，确保其以尽可能高的分辨率处理。在此场景中，视觉标记被包裹&#x5728;**`<img>`**&#x548C;**`</img>`**&#x6807;签内，未使用其他辅助标签。 &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多图像数据集</span>**：总块数$$n_{\text{max}}$$分布在样本内的所有图像中。每张图像由辅助标签标识，&#x5982;**`Image-1`**，以清楚地标记各个图像。图像本身被包裹&#x5728;**`<img>`**&#x548C;**`</img>`**&#x6807;签中，表示图像数据的开始和结束。分配给每张图像$$I_i$$的块数$$n_{\text{max},i}$$与总图像数$$N_{\text{image}}$$成比例，遵循公式： &#x20;
>
> $$n_{\text{max},i} = \max \left( 1, \left\lfloor \frac{n_{\text{max}}}{N_{\text{image}}} \right\rfloor \right)$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视频数据</span>**：此方法通过设置$$n_{\text{max}} = 1$$进行简化。每个视频帧被调整为固定分辨率$$448 \times 448$$，无需分块。这是因为在训练过程中，通常会从单个视频中提取大量帧，例&#x5982;**`32`**<span style="color: rgb(220,155,4); background-color: inherit">帧或</span>**`64`**<span style="color: rgb(220,155,4); background-color: inherit">帧</span>。即使没有高分辨率输入，也会产&#x751F;**`8192`**&#x6216;**`16384`**&#x4E2A;视觉 token。每个视频帧（标记&#x4E3A;**`Frame-1`**&#x7B49;）被包裹&#x5728;**`<img>`**&#x548C;**`</img>`**&#x6807;签中，类似于图像数据。 &#x20;

**<span style="color: rgb(36,91,219); background-color: inherit">单模型训练流程</span>**

![](../../images/视觉多模态讲义（上）-image-178.png)

InternVL 2.5 中单模型的训练流程分为三个阶段，旨在增强模型的视觉感知能力和多模态能力。每个阶段逐步整合视觉和语言模态，在性能优化和训练效率之间取得平衡。 &#x20;

1. **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：MLP热身</span>**

如上图(a)所示，训练从 MLP 的热身开始，MLP 是视觉和语言表示之间的初始桥梁。在此阶段，<span style="color: rgb(100,37,208); background-color: inherit">仅训练 MLP，而视觉编码器 InternViT 和 LLM 保持冻结状态</span>。为了达到最佳性能，从这一阶段开始使用动态高分辨率训练策略。 &#x20;

在此阶段，<span style="color: rgb(100,37,208); background-color: inherit">数据以结构化的</span>**`ChatML`**<span style="color: rgb(100,37,208); background-color: inherit">格式组织，并使用 NTP 损失进行优化。此外，应用较高的学习率以加速收敛，使 MLP 能够快速适应 LLM 的输入空间并建立强大的跨模态对齐</span>。MLP 热身阶段确保模型在解锁后续可训练组件之前已准备好处理多模态任务，从而提高训练稳定性。 &#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1.5：ViT增量学习</span>** &#x20;

阶段 1.5 引入了视觉编码器的增量学习。在此阶段，视觉编码器和 MLP 均可训练，并<span style="color: rgb(100,37,208); background-color: inherit">使用与阶段 1 相同的预训练数据混合和 NTP 损失进行训练</span>。<span style="color: rgb(100,37,208); background-color: inherit">此阶段的目标是增强视觉编码器提取视觉特征的能力，使其能够捕捉更全面的信息，尤其是针对网络规模数据集中相对稀有的领域</span>，例如LAION-5B，如多语言 OCR 数据和数学图表等。 &#x20;

如下表所示，此阶段<span style="color: rgb(100,37,208); background-color: inherit">使用较低的学习率以防止灾难性遗忘，确保编码器不会丢失先前学到的能力</span>。此外，<span style="color: rgb(100,37,208); background-color: inherit">视觉编码器只需训练一次，除非引入新的领域需求或数据</span>。训练完成后，它可以与不同的 LLM 结合使用而无需重新训练，因此阶段 1.5 是可选的。当编码器已经针对某些特定任务进行了优化时，这一点尤其有益，使其能够以较低的额外成本与不同大小的LLM集成。 &#x20;

![](../../images/视觉多模态讲义（上）-image-179.png)

![](../../images/视觉多模态讲义（上）-image-177.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：全模型指令微调</span>** &#x20;

在最后阶段，如上图(a)所示，整个模型，包括 **ViT**、**MLP** 和 **LLM** 在高质量的多模态指令数据集上进行训练。数据质量在此阶段尤为重要，因为负责生成最终用户输出的 LLM 现在是可训练的。<span style="color: rgb(216,57,49); background-color: inherit">即使是少量噪声数据也可能导致模型行为异常</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">几千个样本</span>，如重复输出或特定错误结果。<span style="color: rgb(100,37,208); background-color: inherit">为了减轻 LLM 的退化，在这一阶段实施严格的数据质量控制</span>。 &#x20;

此外，此阶段的训练超参数保持简单，整个模型使用统一的学习率，而不是针对不同组件使用不同的学习率。完成此阶段后，InternVL 2.5 的完整训练过程结束。

**<span style="color: rgb(36,91,219); background-color: inherit">渐进式扩展策略</span>**

如上图所示，作者提出了一种渐进式扩展策略，以高效地将视觉编码器 InternViT 与 LLM 对齐。之前在 InternVL 1.5 和 2.0 的训练中曾采用过类似的策略，这里沿用这个方法。此策略采用分阶段训练方法，<span style="color: rgb(100,37,208); background-color: inherit">从较小且资源高效的 LLM 开始，逐步扩展到更大的 LLM</span>。使用这一方法是因为，<span style="color: rgb(100,37,208); background-color: inherit">即使 ViT 和 LLM 通过 NTP 损失联合训练，生成的视觉特征仍然是通用表示，可以被其他 LLM 轻松理解</span>。 &#x20;

具体来说，在阶段 1.5 中，InternViT 与一个较小的 LLM 一起训练，例&#x5982;**`20B`**<span style="color: rgb(220,155,4); background-color: inherit">参数规模</span>，重点优化基础视觉能力和跨模态对齐。<span style="color: rgb(100,37,208); background-color: inherit">这一阶段避免了直接使用大型 LLM 进行训练所带来的高计算成本</span>。通过共享权重机制，训练好的 InternViT 可以轻松迁移到更大的 LLM，例&#x5982;**`72B`**<span style="color: rgb(220,155,4); background-color: inherit">参数规模</span>，而无需重新训练。因此，在训练更大模型时，可以跳过阶段 1.5，因为之前优化的 InternViT 模块会被复用。这不仅加速了训练过程，还确保了视觉编码器学到的表示得以保留并有效集成到更大的模型中。 &#x20;

通过采用这种渐进式扩展策略，可以以大规模模型训练成本的一小部分实现可扩展的模型更新。例如，**`Qwen2-VL`**<span style="color: rgb(220,155,4); background-color: inherit">处理的累积标记总数达到</span>**`1.4T`**<span style="color: rgb(220,155,4); background-color: inherit">个，而</span>**`InternVL2.5-78B`**<span style="color: rgb(220,155,4); background-color: inherit">仅在约</span>**`120B`**<span style="color: rgb(220,155,4); background-color: inherit">个 token 上训练——不到</span>**`Qwen2-VL`**<span style="color: rgb(220,155,4); background-color: inherit">的十分之一</span>。这种方法在资源受限的情况下特别有利，它<span style="color: rgb(46,161,33); background-color: inherit">通过最大化预训练组件的复用、最小化冗余计算，实现了能够应对复杂视觉-语言任务的模型高效训练</span>。 &#x20;

**<span style="color: rgb(36,91,219); background-color: inherit">训练增强</span>**

为了增强模型在真实场景中的适应性和整体性能，作者引入了两项关键技术。这些优化对于提升用户体验和模型基准性能至关重要。 &#x20;

1. **<span style="color: rgb(36,91,219); background-color: inherit">随机 JPEG 压缩</span>** &#x20;

为了避免训练过程中过拟合并增强模型在现实世界中的表现，作者应用了一种保留空间信息的数据增强技术：**`JPEG `**`压缩`。具体来说，应用质量等级&#x5728;**`75`**&#x5230;**`100`**&#x4E4B;间的随机 JPEG 压缩，以模拟互联网图像中常见的退化现象。这种数据增强提高了模型对噪声和压缩图像的鲁棒性，并通过确保在不同图像质量下的更一致表现，提升了用户体验。&#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">损失重加权</span>** &#x20;

Token 平均和 Sample 平均是两种广泛应用于 NTP 损失加权的策略。

> **<span style="color: rgb(36,91,219); background-color: inherit">Token 平均</span>**&#x8BA1;算所有 token 上的 NTP 损失均值，
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Sample 平均</span>**&#x9996;先计算每个样本内的 NTP 损失均值，然后在样本数量上取平均

这两种策略可以用统一的形式表达为：

$$L = \sum \frac{w_i}{\sum w_j} \cdot L_i, \quad w_i =
\begin{cases} 
\frac{1}{x^0}, & \text{for token averaging} \\ 
\frac{1}{x^1}, & \text{for sample averaging}
\end{cases}$$

其中$$L_i$$和$$w_i$$分别表示 token $$i$$的损失和权重，$$x$$表示 token $$i$$所属回答中的 token 数量。 &#x20;

在使用 <span style="color: rgb(100,37,208); background-color: inherit">Token 平均时，每个 token 对最终损失的贡献相等，这可能导致梯度偏向于包含更多 token 的回答</span>，从而导致基准性能下降。相比之下，<span style="color: rgb(100,37,208); background-color: inherit">Sample 平均确保每个样本的贡献相等，但它可能使模型偏好较短的回答，对用户体验产生负面影响</span>。为了在训练期间减轻对较长或较短回答的偏向，作者采用了一种<span style="color: rgb(100,37,208); background-color: inherit">重加权策略，其中</span>$$w_i = \frac{1}{x^{0.5}}$$<span style="color: rgb(100,37,208); background-color: inherit">。这种方法被称为平方平均，平衡了不同长度回答的贡献</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

InternVL 2.5 通过一系列技术改进和性能优化，成为目前最强大的开源多模态大型语言模型之一。它不仅在学术界引发了广泛关注，还为实际应用场景提供了更高效、更精准的解决方案。

### 2.7.5 <span style="color: rgb(36,91,219); background-color: inherit">InternVL3</span>

![](../../images/视觉多模态讲义（上）-image-187.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

InternVL3 的架构延续了前代模型的框架，遵&#x5FAA;**`ViT-MLP-LLM`**&#x8303;式。并且使用预训练模型权重来初始化 ViT 和 LLM 部分，以降低计算成本。

![](../../images/视觉多模态讲义（上）-image-183.png)

> * ViT 有两种配置：**`InternViT-300M`**&#x548C;**`InternViT-6B`**
>
> * 采用预训练的 LLM，包&#x62EC;**`Qwen2.5`**&#x7CFB;列&#x548C;**`InternLM3-8B`**。LLM 仅从预训练的基础模型初始化，不使用经过指令微调的变体
>
> * MLP 是一个两层网络，采用随机初始化

> &#x4E0E;**`InternVL2.5`**&#x4E00;致，**`InternVL3`**&#x5F15;入了像素逆重&#x6392;**`pixel unshuffle`**，以提升处理高分辨率图像的可扩展性。逆重排将视觉 token 数量减少为原始值的四分之一，每&#x4E2A;**`448×448`**&#x7684;图像块&#x7528;**`256`**&#x4E2A;视觉 token 表示。

**<span style="color: rgb(36,91,219); background-color: inherit">可变视觉位置编码 V2PE</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>

**InternVL3** 集成了<span style="color: rgb(216,57,49); background-color: inherit">可变视觉位置编码 </span>**<span style="color: rgb(216,57,49); background-color: inherit">V2PE</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ariable </span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">isual </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">osition </span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">ncoding）</span>，使视觉 token 使用更小且更灵活的位置增量。这一改进<span style="color: rgb(46,161,33); background-color: inherit">有助于在不显著扩展位置窗口的前提下处理更长的多模态上下文</span>。

具体而言，多模态模型每个训练样本可以表示为：

$$x = (x_1, x_2, \cdots, x_L)$$

其中每个 token $$  x_i  $$ 可以是文本 token embedding、visual enbedding，或其他模态表示，例如<span style="color: rgb(220,155,4); background-color: inherit">视频图像块 embedding</span>。任意 token $$  x_i  $$ 的位置索引 $$  p_i  $$ 可按如下顺序计算：

$$p_i = 
\begin{cases}
0, & \text{if } i = 1 \\
f_{\text{pos}}(p_{i-1}, x_i), & \text{for } i = 2, 3, \cdots, N
\end{cases}$$

传统多模态大模型中无论模态如何，每个 token 的位置索引均以 1 为固定步长递增不同，而 **<span style="color: rgb(100,37,208); background-color: inherit">V2PE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 采用一种模态特定的递归函数来计算位置索引</span>。这导致文本 token 和视觉 token 具有不同的位置索引分配方式：

$$p_i = p_{i-1} + 
\begin{cases}
1, & \text{if } x_i \text{ is textual token}, \\
\delta, & \text{if } x_i \text{ is visual token},
\end{cases}$$

其中$$  \delta  $$是一个较小的增量，即$$\delta < 1$$，用于<span style="color: rgb(100,37,208); background-color: inherit">降低视觉 token 的位置索引增长速率</span>。<span style="color: rgb(100,37,208); background-color: inherit">文本 token 仍保持单位增量 1，用来保留其位置区分性</span>。按照原始 **V2PE** 的设计，<span style="color: rgb(100,37,208); background-color: inherit">需要在单张图像内保持</span>$$  \delta  $$<span style="color: rgb(100,37,208); background-color: inherit">恒定，以维护相对位置关系</span>。

在训练过程中，每张图像对应的$$  \delta  $$从一组预定义的分数值中随机选取：

$$\delta \in \Delta = \left\{1, \frac{1}{2}, \frac{1}{4}, \frac{1}{8}, \frac{1}{16}, \frac{1}{32}, \frac{1}{64}, \frac{1}{128}, \frac{1}{256}\right\}$$

在推理阶段，根据输入序列长度灵活选择$$\delta$$，从而在任务性能与位置索引不超出模型有效上下文范围之间取得平衡。

> 当$$  \delta = 1  $$时，V2PE 退化&#x4E3A;**`InternVL2.5`**&#x4E2D;使用的传统位置编码

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

**InternVL3&#x20;**&#x63D0;出一种原生多模态预训练方法，将语言预训练和多模态对齐训练统一到单一预训练阶段中。也就是说在预训练过程中<span style="color: rgb(100,37,208); background-color: inherit">将多模态数据</span>，如<span style="color: rgb(220,155,4); background-color: inherit">图像-文本、视频-文本或交错的图文序列</span>，<span style="color: rgb(100,37,208); background-color: inherit">与大规模文本语料交织使用，进行端到端的联合优化</span>。

> 传统范式：先训练一个纯 LLM，包括预训练和后训练，再引入其他模态进行适配——我们的方法

这种统一的训练方案使预训练模型能够<span style="color: rgb(46,161,33); background-color: inherit">同时学习语言能力和多模态理解能力，从而在无需引入额外桥接模块或后续跨模型对齐步骤的情况下，显著提升其在视觉-语言任务上的表现</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">多模态自回归建模</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>&#x20;

设$$  M  $$是一个基于 Transformer 的模型，参数为$$\theta$$，能够同时处理文本、图像和视频。对于任意训练样本：

$$x = (x_1, x_2, \cdots, x_L)$$

其中每个$$  x_i  $$表示第$$  i  $$个输入元素，$$  L  $$是序列长度。然后采用标准的从左到右自回归目标函数：

$$\mathcal{L}_{\text{full}}(\theta) = -\sum_{i=2}^{L} w_i \cdot \log p_\theta(x_i \mid x_{<i})$$

其中 $$  w_i  $$ 表示第 $$  i  $$ 个 token 的损失权重。这会在所有模态的 token 上传播梯度，但是 **InternVL3&#x20;**&#x4EC5;对文本 token 计算损失，得到：

$$\mathcal{L}_{\text{text-only}}(\theta) = -\sum_{\substack{i=2 \\ x_i \in \text{Text}}}^{L} w_i \cdot \log p_\theta(x_i \mid x_{<i})$$

在此选择性目标下，<span style="color: rgb(100,37,208); background-color: inherit">视觉 token 仅作为文本生成的条件上下文，而不被直接预测</span>。因此，<span style="color: rgb(100,37,208); background-color: inherit">模型学习以有利于下游语言解码任务的方式嵌入多模态信息</span>。

和之&#x524D;**`InternVL2.5`**&#x8BF4;的一样，常用的 token 平均和样本平均策略分别倾向于偏向长响应和短响应。为缓解此问题，**InternVL3&#x20;**&#x91C7;用平方平均策略：

$$w_i = 
\begin{cases}
\frac{1}{l^0}, & \text{token average} \\
\frac{1}{l^{0.5}}, & \text{square average} \\
\frac{1}{l^1}, & \text{sample average}
\end{cases}$$

其中$$  l  $$表示当前训练样本中需要计算损失的 token 数量。

* **<span style="color: rgb(36,91,219); background-color: inherit">联合参数优化</span>** &#x20;

**InternVL3&#x20;**&#x5728;多模态预训练过程中联合更新所有模型参数，令：

$$\theta^* = \arg\min_\theta \mathbb{E}_{x \in \mathcal{D}_{\text{multi}}} \left[ \mathcal{L}_{\text{text-only}}(\theta) \right]$$

其中 $$  \mathcal{D}_{\text{multi}}  $$ 是大规模纯文本数据与多模态语料的并集，如<span style="color: rgb(220,155,4); background-color: inherit">图像-文本或视频-文本对</span>。由此优化单一模型以处理这些混合数据源。这种多任务联合优化可以使<span style="color: rgb(100,37,208); background-color: inherit">文本表征与视觉特征协同学习，增强跨模态对齐</span>。

这种一体化优化方式联合训练每一层，<span style="color: rgb(100,37,208); background-color: inherit">所有参数都能在大规模多模态语料上共同优化，确保语言和视觉特征同步演化。最终参数因此在纯语言任务和多模态任务上均具备高性能，无需额外的调优步骤</span>。

> 传统流程：在适配多模态大模型时常冻结或部分微调 LLM 甚至 ViT 中的某些层。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

**InternVL3&#x20;**&#x4F7F;用的预训练数据主要分为两类：多模态数据和纯语言数据。

> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态数据</span>**：由现有数据集整合而成，并加入了新采集的真实世界数据。**InternVL3&#x20;**&#x6CBF;用&#x4E86;**`InternVL2.5`**&#x7684;预训练语料，涵盖图像描述生成、通用问答、数学、图表理解、OCR、知识接地、文档理解、多轮对话以及医学数据等多个领域。其实这里整体数据规模未增加，但是<span style="color: rgb(100,37,208); background-color: inherit">通过同时更新 MLP、ViT 和 LLM，提升了数据利用率</span>。为了增强模型在真实应用场景中的泛化能力，还引入了与 GUI、工具使用、3D 场景理解和视频理解相关的额外数据。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">纯语言数据</span>**：由于多模态数据中的文本内容通常较短且多样性不足，因此引入纯语言数据，增强模型的语言理解与生成能力。语言语料主要基&#x4E8E;**`InternLM2.5`**&#x7684;预训练数据，并进一步融合多种开源文本数据集进行扩充。这一增强可以<span style="color: rgb(100,37,208); background-color: inherit">提升模型在知识密集型任务、数学及推理任务上的表现</span>。

由于这些异构数据源之间很难平衡，确定合适的采样策略比较困难。在 InternVL3 中，采用两阶段策略来确定多模态数据与语言数据之间的最优采样比例：

> 1. 在多模态和语言数据集上<span style="color: rgb(100,37,208); background-color: inherit">分别训练独立模型</span>，并在相应基准上评估其性能，从而确定各模态内部的最优采样比例
>
> 2. 在固定总训练预算的前提下，我们将两种模态合并，并<span style="color: rgb(100,37,208); background-color: inherit">确定它们之间的相对采样比例</span>

论文实验展示了语言数据与多模态数据&#x4EE5;**`1:3`**&#x7684;比例进行采样时，在单模态和多模态基准测试中均能取得最佳整体性能。此时总训练 token 数约&#x4E3A;**`200B`**，语言数&#x636E;**`50B`** token，多模态数&#x636E;**`150B`** token。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

**InternVL3&#x20;**&#x91C7;用两阶段后训练策略，进一步提升模型的多模态对话与推理能力。包括 SFT 和<span style="color: rgb(216,57,49); background-color: inherit">混合偏好优化 </span>**<span style="color: rgb(216,57,49); background-color: inherit">MPO</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">ixed </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">reference </span>**<span style="color: rgb(216,57,49); background-color: inherit">O</span>**<span style="color: rgb(216,57,49); background-color: inherit">ptimization）</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">监督微调 SFT</span>**

**InternVL3** 使用了 InternVL2.5 中的<span style="color: rgb(100,37,208); background-color: inherit">随机 JPEG 压缩、平方损失重加权和多模态数据打包</span>等。相较于 InternVL2.5，InternVL3 在 SFT 阶段的主要改进在于<span style="color: rgb(100,37,208); background-color: inherit">使用了质量更高、多样性更强的训练数据</span>。SFT 进一步扩展了工具使用、3D 场景理解、GUI 操作、长上下文任务、视频理解、科学图表、创意写作以及多模态推理等方面的训练样本。

* **<span style="color: rgb(36,91,219); background-color: inherit">混合偏好优化 MPO</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

> 在预训练和 SFT 阶段，模型基于前序真实标签 token 进行下一 token 预测。但在推理过程中，模型依赖自身生成的历史 token 进行预测。这种<span style="color: rgb(216,57,49); background-color: inherit">真实标签 token与模型生成 token之间的差异会导致分布偏移，削弱模型的 CoT 推理能力</span>。

为了缓解这个问题，**InternVL3&#x20;**&#x4F7F;&#x7528;**`MPO`**，<span style="color: rgb(100,37,208); background-color: inherit">通过引入正负样本的联合监督，使模型输出分布更贴近真实分布，从而提升推理性能</span>。

MPO 的训练目标由偏好损失$$\mathcal{L}_p$$、质量损失$$  \mathcal{L}_q  $$和生成损失$$  \mathcal{L}_g  $$组合而成：

$$\mathcal{L} = w_p \mathcal{L}_p + w_q \mathcal{L}_q + w_g \mathcal{L}_g$$

其中 $$  w_*  $$ 表示各损失项的权重。

> * **<span style="color: rgb(36,91,219); background-color: inherit">偏好损失</span>**$$\mathcal{L}_p$$：<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**`DPO`**<span style="color: rgb(100,37,208); background-color: inherit">损失作为偏好学习目标，使模型学会区分优选回答与被拒回答的相对偏好</span>：
>
>   $$\mathcal{L}_p = -\log \sigma\left( \beta \log \frac{\pi_\theta(y_c \mid x)}{\pi_0(y_c \mid x)} - \beta \log \frac{\pi_\theta(y_r \mid x)}{\pi_0(y_r \mid x)} \right)$$
>
>   其中 $$  \beta  $$ 为 KL 正则系数，$$x$$为用户 Query，$$y_c$$和$$  y_r  $$分别为好回答和差回答，策略模型$$  \pi_\theta  $$由初始模型$$  \pi_0  $$初始化
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">质量损失</span>**$$\mathcal{L}_q$$：<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**`BCO`**<span style="color: rgb(100,37,208); background-color: inherit">损失作为质量评估目标，帮助模型判断单个回答的绝对质量</span>：
>
>   $$\mathcal{L}_q = \mathcal{L}_q^+ + \mathcal{L}_q^-$$
>
>   其中 $$  \mathcal{L}_q^+  $$ 和 $$  \mathcal{L}_q^-  $$ 分别对应好回答和差回答的损失，独立计算，促使模型区分单个回答的绝对质量水平：
>
>   $$\mathcal{L}_q^+ = -\log \sigma\left( \beta \log \frac{\pi_\theta(y_c \mid x)}{\pi_0(y_c \mid x)} - \delta \right)$$
>
>   $$\mathcal{L}_q^- = -\log \sigma\left( -\left( \beta \log \frac{\pi_\theta(y_r \mid x)}{\pi_0(y_r \mid x)} - \delta \right) \right)$$
>
>   其中 $$  \delta  $$ 为奖励偏移量，取历史奖励的滑动平均值，用于稳定训练过程。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">生成损失</span>**$$\mathcal{L}_g$$：<span style="color: rgb(100,37,208); background-color: inherit">帮助模型学习生成优选回答的过程</span>：
>
> $$\mathcal{L}_g = -\sum_{\substack{i=2 \\ x_i \in \text{Text}}}^{L} w_i \cdot \log p_\theta(x_i \mid x_{<i})$$

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

对于 SFT 数据，在 InternVL2.5 使用的数据基础上，<span style="color: rgb(100,37,208); background-color: inherit">新增了工具使用、3D 场景理解、GUI 操作、科学图表、创意写作和多模态推理等样本</span>。最终训练样本数量从 InternVL2.5 &#x7684;**`16.3M`**&#x589E;长至 InternVL3 &#x7684;**`21.7M`**。

对于 MPO 数据，<span style="color: rgb(100,37,208); background-color: inherit">基于 MMPR v1.2 提出的数据流程和样本构建偏好对，包括 VQA、科学、图表、数学、OCR 和文档理解等。使用 InternVL3-8B、38B 和 78B 的 SFT 版本生成推理轨迹</span>。在 MPO 阶段，所有模型均在&#x7EA6;**`300K`**&#x6837;本的统一数据集上进行训练。

* **<span style="color: rgb(36,91,219); background-color: inherit">Test-Time Scaling</span>**

**InternVL3&#x20;**&#x91C7;&#x7528;**`Best-of-N`**&#x63A8;理策略，使用 VisualPRM-8B 作为 critic model，在推理和数学任务中选择最优回答。

1. **<span style="color: rgb(36,91,219); background-color: inherit">VisualPRM</span>**&#x20;

VisualPRM <span style="color: rgb(100,37,208); background-color: inherit">首先为给定解法的每一步分配一个质量得分，然后对这些得分取平均，得到该解法的总体评分</span>。这个过程被建模为一个多轮对话任务，以便有效利用多模态大模型的生成能力。<span style="color: rgb(100,37,208); background-color: inherit">在第一轮中输入图像</span>$$I$$<span style="color: rgb(100,37,208); background-color: inherit">、问题</span>$$  q  $$<span style="color: rgb(100,37,208); background-color: inherit">以及逐步解法</span>$$  s = \{s_0, s_1, \cdots, s_n\} \in S  $$<span style="color: rgb(100,37,208); background-color: inherit">的第一步</span>$$s_0$$<span style="color: rgb(100,37,208); background-color: inherit">，后续每一轮依次展示新的步骤</span>。

在训练阶段，模型需在每一轮预测当前步骤的正确性：

$$c_i \sim M(y_i \mid I, q, s_{\leq i})$$

其中$$  c_i \in \{+, -\}  $$表示第$$  i  $$步的正确与否。在推理阶段，每一步的得分定义为模型生&#x6210;**`+`**&#x7684;概率。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

使&#x7528;**`VisualPRM400K`**&#x6570;据集训练 **VisualPRM**，这个数据集基于 MMPR v1.2 收集的多模态问题构建。作者按照 VisualPRM400K 的数据流程，进一步通过 InternVL3 的 8B 和 38B 版本生成推理轨迹，对 VisualPRM400K 进行了扩展。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练框架</span>**

&#x5BF9;**`InternEVO`**&#x6846;架进行了扩展。InternEVO 最初用于优化大规模 LLM 训练中的 ZeRO，现在支持 InternVL 系列模型的训练。这个扩展使模型能够在数千张 GPU 上高效扩展至数千亿参数规模。

增强后的框架为 ViT、MLP 和 LLM <span style="color: rgb(46,161,33); background-color: inherit">引入了灵活且解耦的分片策略，提升了训练效率，实现了通信与计算的重叠</span>。而且也<span style="color: rgb(46,161,33); background-color: inherit">全面支持数据并行、张量并行、序列并行和流水线并行等多种并行策略及其任意组合</span>。

多模态大模型训练中的一个难点是视觉 token 与文本 token 比例不均导致计算负载失衡，这种失衡可能使 ViT 或 LLM 模块过载。为了解决这个难题，作者<span style="color: rgb(100,37,208); background-color: inherit">引入一系列技术，动态平衡各模块间的计算负载，确保资源的高效与均衡利用</span>。

对于不同规模的 InternVL ，扩展后的 InternEVO 框架构建了一个优化目标，可以在不同模块维度上寻找最优配置，以最小化内存占用和通信开销。<span style="color: rgb(46,161,33); background-color: inherit">为了支持最长 32K token 的序列，结合了头并行 head-parallel 和序列并行 sequence-parallel，在保持计算效率的同时有效克服了可扩展性瓶颈</span>。相比 InternVL2.5 的训练过程，在相同计算预算下，<span style="color: rgb(46,161,33); background-color: inherit">InternEVO 在 InternVL3 上的应用使同等规模模型的训练速度提升了</span>**`50%`**<span style="color: rgb(46,161,33); background-color: inherit">至</span>**`200%`**

### 2.7.6 <span style="color: rgb(36,91,219); background-color: inherit">InternVL3.5</span>

![](../../images/视觉多模态讲义（上）-image-176.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

多模态大模型正在从基础的图文理解转变为更具挑战性的任务，如<span style="color: rgb(220,155,4); background-color: inherit">复杂推理、文本生成和具身智能</span>等。然而，<span style="color: rgb(216,57,49); background-color: inherit">当前开源模型在这些方面与领先商业模型之间仍存在显著差距，尤其在强化学习框架的稳定性、可扩展性以及高分辨率多模态处理带来的高昂计算成本等方面都比不了</span>。

为了解决这些问题，商汤推出 InternVL3.5，在通用性、推理能力和系统效率方面实现了全面升级。核心在于提出了一种新颖&#x7684;**`Cascade RL`**&#x6846;架，结合离线与在线强化学习两个互补阶段，并引入了两项关键技术——视觉分辨率路由器 **<span style="color: rgb(216,57,49); background-color: inherit">ViR</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Vi</span>**<span style="color: rgb(216,57,49); background-color: inherit">sual </span>**<span style="color: rgb(216,57,49); background-color: inherit">R</span>**<span style="color: rgb(216,57,49); background-color: inherit">esolution Router）</span>和解耦的视觉-语言部署 **<span style="color: rgb(216,57,49); background-color: inherit">DvD</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">ecoupled </span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ision-Language </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">eployment）</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">InternVL3.5</span>**

沿用之前 InternVL 系列所采用&#x7684;**`ViT–MLP–LLM`**&#x67B6;构范式，配置如下表。LLM 基&#x4E8E;**`Qwen3`**&#x7CFB;列&#x548C;**`GPT-OSS`**&#x521D;始化，视觉编码器采&#x7528;**`InternViT-300M`**&#x548C;**`InternViT-6B`**。此外使用了 InternVL1.5 中提出的<span style="color: rgb(100,37,208); background-color: inherit">动态高分辨率策略，以支持灵活的多尺度视觉理解</span>。

![](../../images/视觉多模态讲义（上）-image-175.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">InternVL3.5-Flash </span>**

在 InternVL3.5 的基础上进一步集成了视觉分辨率路由器 ViR 模块，构建出一系列适用于资源受限场景的高效变体。<span style="color: rgb(100,37,208); background-color: inherit">每个图像块最初被表示为</span>**`1024`**<span style="color: rgb(100,37,208); background-color: inherit">个视觉 token，随后通过像素打乱 pixel shuffle 模块压缩至</span>**`256`**<span style="color: rgb(100,37,208); background-color: inherit">个 token，再传入 LLM</span>。在 InternVL3.5-Flash 中，引入了更高压缩率的额外像素打乱模块，可将视觉 token 进一步压缩&#x81F3;**`64`**&#x4E2A;，如下图。<span style="color: rgb(100,37,208); background-color: inherit">对于每一个图像块，路由模块会根据其语义丰富程度判断合适的压缩级别，并将其分配至相应的压缩路径</span>。得益于这种基于图像块语义感知的压缩机制，<span style="color: rgb(46,161,33); background-color: inherit">InternVL3.5-Flash 能在几乎不损失性能的前提下将视觉 token 数量减少</span>**`50%`**。

![](../../images/视觉多模态讲义（上）-image-180.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

<span style="color: rgb(100,37,208); background-color: inherit">使用大规模文本与多模态语料进行训练，联合更新所有模型参数</span>。给定一个包含多模态 token 序列 $$  x = (x_1, x_2, \ldots, x_L)  $$ 的训练样本，对每个文本 token 计算下一 token 预&#x6D4B;**`NTP`**&#x635F;失：

$$L_i = -\log p_\theta (x_i \mid x_1, \ldots, x_{i-1})$$

其中$$  x_i  $$是待预测的 token，前缀$$  \{x_1, x_2, \ldots, x_{i-1}\}  $$可包含文本或图像 token。对于对话类样本，仅回答部分的 token 参与损失计算。<span style="color: rgb(100,37,208); background-color: inherit">为缓解训练过程中对长或短回答的偏好偏差，采用平方平均法对 NTP 损失进行重加权</span>：

$$L'_i = \frac{w_i}{\sum_j w_j} \cdot L_i, \quad w_i = \frac{1}{N^{0.5}}$$

其中$$  N  $$表示当前样本中参与损失计算的 token 总数。此外还<span style="color: rgb(100,37,208); background-color: inherit">引入随机 JPEG 压缩增强，以提升模型在真实场景中的鲁棒性</span>。

预训练语料分为两类：

> * **<span style="color: rgb(36,91,219); background-color: inherit">多模态数据</span>**：主要<span style="color: rgb(100,37,208); background-color: inherit">来源于 InternVL3 的训练语料</span>，涵盖图像描述、通用问答、数学、科学、图表理解、OCR、知识 grounding、文档理解、多轮对话及医学等多个领域
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">纯文本数据</span>**：<span style="color: rgb(100,37,208); background-color: inherit">基于 InternLM 系列的训练语料，并进一步融合多个开源文本数据集</span>。整体预训练语料包含&#x7EA6;**`116M`**&#x6837;本，总计&#x7EA6;**`250B`** token，文本与多模态数据的比例约&#x4E3A;**`1:2.5`**。最大序列长度设&#x4E3A;**`32K`** token，以支持长上下文的理解与推理。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

采用三阶段后训练策略：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">SFT</span>**：保持与预训练相同的训练目标，但使用更高品质的对话数据进一步提升模型能力
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">级联强化学习 Cascade RL</span>**：结合离线与在线 RL 的优势，提升模型的推理能力
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">视觉一致性学习 ViCO</span>**：用于将 ViR 模块集成至 InternVL3.5，构建高效版本 InternVL3.5-Flash，目标是最小化不同视觉压缩率下的输出差异

1. **<span style="color: rgb(36,91,219); background-color: inherit">SFT</span>**

<span style="color: rgb(100,37,208); background-color: inherit">采用与预训练相同的损失目标，并使用平方平均策略计算最终损失</span>。上下文窗口仍设&#x4E3A;**`32K`** token。InternVL3.5 的 SFT 数据在质量与多样性上均有提升，主要来自三类来源：

> 1. 沿用 InternVL3 的指令跟随数据，确保对视觉-语言任务的广泛覆盖
>
> 2. 思考模式下的多模态推理数据，通过大规模推理模型生成包含详细推理过程的回答样本，经过严格筛选（包括评估推理清晰度、去除冗余、统一格式）确保推理质量，问题覆盖数学、科学等专业领域
>
> 3. 能力扩展数据集，赋予模型新技能，如基于 GUI 的交互、具身智能交互、SVG 的理解与生成

SFT 阶段使用&#x7EA6;**`56M`**&#x6837;本，&#x5373;**`130B`** token，文本与多模态数据比例约&#x4E3A;**`1:3.5`**

* **<span style="color: rgb(36,91,219); background-color: inherit">级联强化学习 Cascade RL</span>**

强化学习的核心优势在于引入负样本，抑制低质量输出区域，从而提升整体质量。DPO 等方法作&#x4E3A;**`offline RL`**<span style="color: rgb(100,37,208); background-color: inherit">，训练效率高，但性能上限较低；而</span>**`online RL`**<span style="color: rgb(100,37,208); background-color: inherit">虽有效，但计算开销大、耗时长</span>。结合这两者的优点，作者提&#x51FA;**`Cascade RL`**，<span style="color: rgb(46,161,33); background-color: inherit">融合两者优势，实现高效、渐进的 MLLM 后训练</span>。具体流程为：

> 1. 使用 **<span style="color: rgb(36,91,219); background-color: inherit">offline RL</span>** 进行高效微调，作为热身阶段，获得高质量的 rollout 输出
>
> 2. 采用 **<span style="color: rgb(36,91,219); background-color: inherit">online RL</span>**，基于模型自身生成的 rollout 进一步优化输出分布

相比单一 RL 阶段，<span style="color: rgb(46,161,33); background-color: inherit">Cascade RL 在显著提升性能的同时，大幅降低 GPU 训练成本</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">offline RL</span>**：<span style="color: rgb(100,37,208); background-color: inherit">采用混合偏好优化 MPO 进行微调</span>。其训练目标为偏好损失$$L_p$$、质量损失$$  L_q  $$和生成损失$$  L_g  $$的加权组合：

$$L_{\text{MPO}} = w_p L_p + w_q L_q + w_g L_g$$

其中$$  w_*  $$为各损失项的权重。<span style="color: rgb(100,37,208); background-color: inherit">DPO 损失作为偏好损失，BCO 损失作为质量损失，LM loss 作为生成损失</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">online RL</span>**：<span style="color: rgb(100,37,208); background-color: inherit">采用无参考模型约束的 GSPO 作为 online RL 算法，因为在训练 Dense 模型和 MoE 模型时更为有效</span>。类似 GRPO，优势函数定义为同一 query 下多个 response 奖励的标准化结果：

$$\hat{A}_{i} = \frac{r(x, y_i) - \text{mean}\left(\{r(x, y_i)\}_{i=1}^G\right)}{\text{std}\left(\{r(x, y_i)\}_{i=1}^G\right)}$$

其中$$  y_i  $$是针对 Query $$  x  $$生成的第$$  i  $$个回答，$$G$$为生成的总回答数，$$r(x, y_i)$$表示该回答的奖励值。GSPO 的训练目标为：

$$L_{\text{GSPO}}(\theta) = \mathbb{E}_{x \sim D, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^G \min\left( s_i(\theta) \hat{A}_{i}, \text{clip}(s_i(\theta), 1 - \varepsilon, 1 + \varepsilon) \hat{A}_{i} \right) \right]$$

其中<span style="color: rgb(100,37,208); background-color: inherit">重要性采样比</span>$$  s_i(\theta)  $$<span style="color: rgb(100,37,208); background-color: inherit">定义为逐 token 比值的几何平均</span>：

$$s_i(\theta) = \left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)} \right)^{1/|y_i|} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})} \right)$$

其中$$  \pi_\theta(y_i \mid x, y_{i,<t})  $$和$$  \pi_\theta(y_{i,t} \mid x, y_{i,<t})  $$分别表示策略模型在参数$$  \theta  $$下生成完整回答$$  y_i  $$和第$$  t  $$个 token 的概率。

Cascade RL 具有三大优势：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">训练更稳定</span>**：离线阶段解耦 rollout 收集与参数更新，<span style="color: rgb(46,161,33); background-color: inherit">有效缓解奖励作弊等问题</span>；在线阶段中，更强的初始模型表现出<span style="color: rgb(46,161,33); background-color: inherit">更稳定的训练动态</span>，MPO 阶段的性能增益进一步提升了 GSPO 阶段的鲁棒性
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">训练更高效</span>**：MPO 阶段的 rollout 可跨模型共享，<span style="color: rgb(46,161,33); background-color: inherit">摊薄在线 RL 的采样成本</span>
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">性能上限更高</span>**：经 MPO 微调的模型在后续在线 RL 阶段以<span style="color: rgb(46,161,33); background-color: inherit">更少训练步数达到更高性能，显著降低总体训练开销</span>

Cascade RL 阶段 offline RL 使&#x7528;**`MMPR-v1.2`**&#x6570;据集，&#x7EA6;**`200K`**&#x6837;本对，online RL 从中筛选模型准确率&#x5728;**`0.2`**&#x81F3;**`0.8`**&#x4E4B;间的 query ，并融合多个新近多模态数据集，构建&#x7EA6;**`70K`** query &#x7684;**`MMPR-Tiny`**&#x6570;据集。rollout 数据直接复用 MMPR-v1.2，避免额外采样开销。

![](../../images/视觉多模态讲义（上）-image-195.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉一致性学习 ViCO</span>**

ViCO 用于将 ViR 模块集成至 InternVL3.5，构建高效推理版本 InternVL3.5-Flash。ViCO 包含两个子阶段

**<span style="color: rgb(222,120,2); background-color: inherit">一致性训练</span>**：整个模型<span style="color: rgb(100,37,208); background-color: inherit">训练目标为最小化不同压缩率下回答分布的差异</span>。引入冻结的参考模型 InternVL3.5，对同一输入图像分别&#x4EE5;**`256`**&#x74;oken（压缩率$$\xi = \frac{1}{4}$$）&#x548C;**`64`**&#x74;oken（$$\xi = \frac{1}{16}$$）表示，训练目标为：

$$L_{\text{ViCO}} = \mathbb{E}_{\xi \sim R} \left[ \frac{1}{N} \sum_{i=1}^N \text{KL}\left( \pi_{\theta_{\text{ref}}}(y_i \mid y_{<i}, I) \parallel \pi_{\theta_{\text{policy}}}(y_i \mid y_{<i}, I_\xi) \right) \right]$$

其中$$\xi$$从$$  \left\{ \frac{1}{4}, \frac{1}{16} \right\}  $$中均匀采样，参考模型始终使用$$  \xi = \frac{1}{4}  $$推理。

**<span style="color: rgb(222,120,2); background-color: inherit">路由训练</span>**：<span style="color: rgb(100,37,208); background-color: inherit">训练 ViR 模块为不同输入选择合适的压缩分辨率。ViR 是一个二分类器，使用标准交叉熵损失进行训练</span>。路由标签通过计算压缩前后输出的 KL 散度比值得到。令：

$$r_i = \frac{L_{\text{ViCO}}(y_i \mid I_{1/16})}{L_{\text{ViCO}}(y_i \mid I_{1/4})}$$

表示压缩带来的相对损失增长。路由标签定义为：

$$y^{\text{router}}_i = 
\begin{cases}
0, & r_i < \tau\\
1, & r_i \geq \tau
\end{cases}$$

其中$$r_i < \tau$$表示压缩影响小，反之压缩影响大；$$  y^{\text{router}}_i = 0  $$ 对应$$\xi = \frac{1}{16}$$，$$y^{\text{router}}_i = 1$$对应$$\xi = \frac{1}{4}$$。阈值$$  \tau  $$为滑动窗口内历史$$  r_i  $$值的第$$  k  $$百分位数，确保标签分布均衡。一致性训练阶段，同一图像的所有 patch 使用随机压缩率，以保留无压缩下的模型能力。<span style="color: rgb(46,161,33); background-color: inherit">InternVL3.5-Flash 在减少</span>**`50%`**<span style="color: rgb(46,161,33); background-color: inherit">视觉 token 的同时，保持了接近</span>**`100%`**<span style="color: rgb(46,161,33); background-color: inherit">的原始性能</span>。

ViCO 阶段的一致性训练使用与 SFT 相同的数据，路由训练则选用 OCR 和 VQA 等视觉信息密集的子集，帮助 ViR 学习基于内容动态决策压缩策略。

* **<span style="color: rgb(36,91,219); background-color: inherit">Test-Time Scaling</span>**

InternVL3.5 实现了一套综合性的测试时扩展策略，<span style="color: rgb(100,37,208); background-color: inherit">同时提升模型的推理深度（即深度思考）和推理广度（即并行思考）</span>：

> * **<span style="color: rgb(36,91,219); background-color: inherit">深度思考 Deep Thinking</span>**：通过激活思考模式，<span style="color: rgb(100,37,208); background-color: inherit">引导模型在生成最终答案前进行有意识的逐步推理，将复杂问题分解为逻辑步骤，并验证中间结论</span>。<span style="color: rgb(46,161,33); background-color: inherit">改善了解决复杂问题时的逻辑结构，尤其对依赖多步推理的任务效果显著，从而增强了推理的深度</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">并行思考 Parallel Thinking</span>**：延续 InternVL3 的做法，在推理任务中采用 **<span style="color: rgb(216,57,49); background-color: inherit">BoN</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">B</span>**<span style="color: rgb(216,57,49); background-color: inherit">est-</span>**<span style="color: rgb(216,57,49); background-color: inherit">o</span>**<span style="color: rgb(216,57,49); background-color: inherit">f-</span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">）</span>策略，使&#x7528;**`VisualPRM-v1.1`**&#x4F5C;为 critic model，从多个推理候选中选择最优回答。该方法<span style="color: rgb(46,161,33); background-color: inherit">通过生成多样化的推理路径并筛选最佳结果，有效提升了推理的广度</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练硬件</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">训练框架</span>**

模型训练主要基于 XTuner 框架，针对 Dense 和 MoE 架构集成了一系列高效训练优化技术。采用 **`FSDP`** 对模型参数进行跨 GPU 分片，降低显存占用；通过<span style="color: rgb(100,37,208); background-color: inherit">数据打包减少填充 token</span>，均衡各设备负载，提升训练效率。引入 <span style="color: rgb(100,37,208); background-color: inherit">FP8 精度训练</span>，结&#x5408;**`DeepGEMM`**&#x548C;**`liger-kernel`**&#x7684;融合交叉熵算子，显著加快训练速度。注意力计算采用 **`FlashAttention-3`**，支持打包序列输入，进一步提升计算效率。针对 MoE 模型，使&#x7528;**`TMA-Adaptive FP8`**&#x5206;组 GEMM 内核进行优化。<span style="color: rgb(100,37,208); background-color: inherit">在强化学习阶段，基于 verl 框架实现高效策略更新</span>。对于特定模型如 InternVL3.5-20B-A4B，还在 GPT-OSS-20B 中通&#x8FC7;**`Triton`**&#x5B9E;现了带 sink 机制的窗口注意力加速，以支持长上下文高效推理。

* **<span style="color: rgb(36,91,219); background-color: inherit">推理架构</span>**

解耦视觉-语言部署 DvD：考虑到视觉编码器和语言模型在计算特性上的显著差异，即<span style="color: rgb(100,37,208); background-color: inherit">前者高度并行、无状态依赖，后者自回归、依赖历史状态</span>，作者提出 **<span style="color: rgb(216,57,49); background-color: inherit">DvD</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">ecoupled </span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ision-Language </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">eployment）</span>架构，将视觉与语言处理解耦部署。具体地，<span style="color: rgb(100,37,208); background-color: inherit">视觉模块（包括 ViT、MLP，以及 InternVL3.5-Flash 中的 ViR）部署在独立的视觉服务器上，负责批量处理图像并生成紧凑的视觉特征；语言模型则单独运行在语言服务器上，接收来自视觉侧的 BF16 格式特征，与文本输入融合后进行自回归解码。通信通过 TCP 单向传输，支持可选 RDMA 以提升传输带宽</span>。

![](../../images/视觉多模态讲义（上）-image-196.png)

该架构<span style="color: rgb(46,161,33); background-color: inherit">将视觉处理、特征传输和语言推理组织为异步三阶段流水线，实现各阶段重叠执行，有效减少等待和阻塞，尤其在处理高分辨率图像或大规模批量请求时优势明显</span>。DvD 不仅提升了视觉侧的 GPU 利用率，也<span style="color: rgb(46,161,33); background-color: inherit">使语言服务器能专注 LLM 推理，显著提高整体吞吐量和响应速度</span>。此外，该设计支持对视觉和语言模块分别进行硬件资源配置与成本优化，并便于未来新模块的接入，如更高分辨率 ViT 或新路由策略，无需改动语言模型部署结构，具备良好的可扩展性。

## 2.8 <span style="color: rgb(36,91,219); background-color: inherit">DeepSeek 系列</span>

### 2.8.1 <span style="color: rgb(36,91,219); background-color: inherit">DeepSeek-VL</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Data Construction</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">两类训练数据与阶段分工</span>**

DeepSeek-VL 把训练数据分成 Vision-Language pretraining data 和 Vision-Language supervised fine-tuning data。前者规模大、来源复杂，用来建立基础的跨模态理解能力；后者规模较小，用 instruction-response 形式教会 Model 完成具体任务并进行多轮对话。

<span style="color: rgb(100,37,208); background-color: inherit">这两类数据与三阶段训练一一对应：pretraining data 先在 Stage 1 中 warm up Vision-Language Adaptor，再在 Stage 2 中与纯文本数据共同训练；SFT data 则用于 Stage 3，把预训练能力转化为可交互的 DeepSeek-VL-Chat。</span>

2. **<span style="color: rgb(36,91,219); background-color: inherit">Vision-Language Pretraining Data 的整体构成</span>**

Stage 2 的 data mixture 同时包含 interleaved image-text、image caption、table/chart、Web Code、scene text OCR、document OCR 和 text-only corpus。各类别占比分别为 **`13.1%`**、**`11.1%`**、**`2.1%`**、**`0.4%`**、**`1.2%`**、**`2.1%`** 和 **`70.0%`**。

<span style="color: rgb(46,161,33); background-color: inherit">纯文本占比保持在 70%，不是为了补充视觉知识，而是为了在学习图像输入时保住原有语言能力。这个数据设计直接服务于 Stage 2 的 language-multimodal joint training。</span>

**<span style="color: rgb(222,120,2); background-color: inherit">Interleaved Image-Text 与 Image Caption Data</span>**

Interleaved image-text data 来自 MMC4、英文和中文 Wikipedia、Wikihow，以及内部整理的 PDF 和 Epub textbook。它保留一段文本中多张图像与上下文交错出现的结构，使 Model 能在多模态上下文中建立跨段落引用关系，而不只是学习单张图对应一句 caption。

Image caption data 由 CapsFusion、TaiSu 和 Detailed Caption 组成。与 interleaved data 相比，这部分样本的图文对应关系更直接，主要强化 object、attribute、action、scene 等视觉实体与语言描述之间的对齐。

**<span style="color: rgb(222,120,2); background-color: inherit">Table、Chart 与 Web Code Data</span>**

Table/chart data 汇集 Chart2Text、Geo170K、UReader、UniChart、M-Paper、ScienceQA、ScreenQA、SciGraphQA-295K、Paper2Figure100K、Widget Captioning、Screen2Words 和 RefExp。它覆盖 chart reading、table QA、scientific figure、screen understanding 与 referring expression，使 Model 能处理结构化视觉内容，而不局限于自然图像。

Web Code data 面向 UI inverse rendering 和 plot inverse rendering。UI 部分使用 WebSight；plot 部分从 Stack corpus 中处理约 **`146 万`**&#x4E2A; Jupyter notebook，收集图表及其前置代码，先形成约 **`200 万`**&#x7EC4; image-code pair，再过滤为 **`110 万`**&#x4E2A;主要训练样本。每个保留样本只含一张图，并要求对应代码不少于 **`5 行`**。

<span style="color: rgb(100,37,208); background-color: inherit">这类数据把视觉理解扩展成视觉到程序的逆向映射：输入是网页截图或 plot，输出不再是自然语言描述，而是能够重建界面的 HTML、CSS 或绘图代码。</span>

**<span style="color: rgb(222,120,2); background-color: inherit">Document OCR Data</span>**

Document OCR 同时覆盖英文和中文文档。第一部分收集约 **`140 万`**&#x7BC7; arXiv article 的 source code 与编译 PDF，再沿用 Nougat 的 preprocessing，把 PDF page 渲染为图像并与结构化文本配对。这些样本保留公式、段落、标题和文档版式，比普通 scene text 更接近真实学术文档解析。

第二部分处理约 **`86 万`**&#x672C;英文 e-book、**`18 万`**&#x672C;中文 e-book，以及数百万道 K-12 educational question。清洗后的 HTML 使用多种 template 重新渲染成 image-text pair，从而扩大 layout、font、language 和页面结构的覆盖范围。

**<span style="color: rgb(222,120,2); background-color: inherit">Scene Text OCR 与 Text-Only Corpus</span>**

Scene text OCR 负责识别融入自然环境的文字，数据源包括 ArT、MLT-17、LSVT、UberText、COCO-Text、RCTW-17、ReCTS、TextOCR、OpenVINO 和 HierText。它与 document OCR 的区别在于：前者处理招牌、道路、商品和自然场景中的文字，后者处理具有明确 layout 的页面级文档。

Text-only corpus 与 DeepSeek-LLM 使用的预训练语料保持一致，规模约为 **`2T text tokens`**。在 joint pretraining 中，这部分数据持续向 LLM 提供原有语言分布，降低 multimodal corpus 过于简单或分布偏移造成的 language forgetting。

* **<span style="color: rgb(36,91,219); background-color: inherit">Supervised Fine-Tuning Data Mixture</span>**

SFT mixture 由五部分组成：内部多模态数据占 **`10.5%`**；ShareGPT4V、LAION-GPTV、LVIS-Instruct4V、TextOCR-GPT4V、LLaVA-1.6-GPT4V 和 IconQA 等 general multimodality data 占 **`35.5%`**；table/chart data 占 **`4.1%`**；Screen-to-Code 与 ScreenQA 构成的 Web Code data 占 **`2.0%`**；DeepSeek-LLM 的 text-only SFT data 占 **`47.9%`**。

<span style="color: rgb(46,161,33); background-color: inherit">SFT 阶段仍然保留接近一半的纯文本对话数据，使 visual instruction following 的增强不会把 Chat Model 退化成只能回答图像问题的专用系统。</span>

**<span style="color: rgb(222,120,2); background-color: inherit">In-House SFT Taxonomy</span>**

内部 SFT data 先从 GPT-4V 和 Gemini 的真实使用案例中整理任务，再建立 taxonomy。一级类别包括 Recognition、Conversion、Analysis、Commonsense Reasoning、Logical Reasoning、Evaluation、Multi-Image 和 Safety。

Recognition 进一步覆盖 global/local description、OCR、position、attribute、counting 与 currency recognition；Conversion 包含 UI-to-Code、Chart-to-Code、Formula-to-Code、Flowchart-to-Code、image-to-prompt 和 text interpretation；Analysis 覆盖 table/chart、circuit、map、music score、medical image、sensor image 与 encyclopedia knowledge。

Reasoning 类任务从 relationship、function、environment、anomaly、humor 一直延伸到 algebra、geometry、physics、chemistry、biology、code 和 IQ question。Multi-Image 关注 temporal sequence 和 cross-image comparison；Safety 则包含 suggestive question、counterfactual question 与 prompt injection。

<span style="color: rgb(100,37,208); background-color: inherit">同一套 taxonomy 同时约束 SFT prompt 采样和后续 human evaluation，使训练覆盖范围与人工测试维度保持一致。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Approach</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">整体架构</span>**

DeepSeek-VL 由 Hybrid Vision Encoder、Vision-Language Adaptor 和 Language Model 三个模块组成。图像先同时进入高分辨率与低分辨率视觉分支，两路 feature 在 Adaptor 中对齐并融合，最终转换到 DeepSeek LLM 的 input Embedding space，与文本 token 一起交给 Language Model。

模型配置并不完全相同：论文表格中的 1B 配置使用 SigLIP，7B 配置使用 SigLIP+SAM。下面的双分支 Hybrid Vision Encoder 对应 7B Model，也是 DeepSeek-VL 处理 1024×1024 输入的核心结构。

<span style="color: rgb(100,37,208); background-color: inherit">核心设计不是单纯增加图像分辨率，而是在固定为 576 个 visual tokens 的预算下，把 SigLIP-L 的高层语义和 SAM-B 的低层细节压进同一组 token。</span>

**<span style="color: rgb(222,120,2); background-color: inherit">Hybrid Vision Encoder 的设计动机</span>**

SigLIP 属于 CLIP 系列视觉表征，擅长学习与文本对齐的高层语义。但这类 Encoder 容易出现 CLIP-blind pair：视觉差异明显的图像可能被映射到相近 representation。常见输入分辨率通常是 224×224、336×336、384×384 或 512×512，对 dense OCR、small object 和 visual grounding 所需的局部细节也不够稳定。

<span style="color: rgb(216,57,49); background-color: inherit">只依靠 text-aligned vision encoder，会同时受到 semantic ambiguity 和低分辨率细节丢失的限制，无法用一个分支兼顾全局语义与精细定位。</span>

因此，DeepSeek-VL 在 SigLIP-L 之外增加 vision-only 的 SAM-B Encoder。SAM-B 本质上使用预训练 ViTDet image encoder，能够接收 **`1024×1024`** 输入，补充文字、边缘、局部结构和小目标所需的低层 feature。

**<span style="color: rgb(222,120,2); background-color: inherit">SAM-B 高分辨率分支</span>**

输入图像先 resize 到 **`1024×1024`**，SAM-B 输出 **`64×64×256`** feature map。VL Adaptor 先把空间尺寸插值到 **`96×96×256`**，再经过两个 stride 为 **`2`** 的 convolution layer，将 feature map 压缩为 **`24×24×1024`**。

随后把二维空间展开为 sequence，得到 **`576×1024`** 的高分辨率视觉 feature。这个过程把 1024×1024 输入中的细节压缩到 576 个位置，避免高分辨率图像直接造成 visual token 数量失控。

![1024×1024 高分辨率分支保留远处 cyclist 等小目标细节，Model 再结合场景关系生成位置判断。](../../images/视觉多模态讲义（上）-tiny-object-example.png)

**<span style="color: rgb(222,120,2); background-color: inherit">SigLIP-L 语义分支与 Feature Fusion</span>**

SigLIP-L 使用 **`384×384`** 输入，直接产生 **`576×1024`** feature。它与 SAM-B 分支在 sequence length 上完全对齐，但语义侧重不同：SigLIP-L 保留 text-aligned semantic representation，SAM-B 提供高分辨率局部结构。

两路 feature 沿 Embedding dimension concatenate，形成 **`576×2048`** representation。经过 GeLU 和后续 projection 后，仍然只向 LLM 提供 **`576`** 个 visual tokens。

<span style="color: rgb(46,161,33); background-color: inherit">这种 embedding concatenation 不增加 sequence length，因此 attention 的 token 开销保持不变，同时每个 visual token 同时携带 semantic feature 与 detail feature。</span>

**<span style="color: rgb(222,120,2); background-color: inherit">Vision-Language Adaptor</span>**

Adaptor 使用 two-layer hybrid MLP。第一层不是共享参数：高分辨率 SAM-B feature 与低分辨率 SigLIP-L feature 分别经过各自的 single-layer MLP，以适配两种 Encoder 不同的数值范围和 feature distribution。处理后的两路 feature 沿 Embedding dimension concatenate，再通过第二层 MLP 映射到 LLM input space。

<span style="color: rgb(100,37,208); background-color: inherit">Hybrid MLP 同时保留 separate projection 的分布适配能力和 shared fusion 的跨分支交互，后续 ablation 也围绕这两个目标比较不同 Adaptor 结构。</span>

**<span style="color: rgb(222,120,2); background-color: inherit">Language Model</span>**

Language Model 建立在 DeepSeek LLM 上，micro architecture 大体沿用 LLaMA：Transformer block 使用 Pre-Norm 与 RMSNorm，FFN 使用 SwiGLU，intermediate dimension 为：

$$\frac{8}{3}d_{\text{model}}$$

位置表示使用 Rotary Embedding，Tokenizer 与 DeepSeek-LLM 保持一致。DeepSeek-VL 不是从随机初始化重新训练 LLM，而是从 DeepSeek-LLM 的 intermediate checkpoint 继续 multimodal pretraining。

DeepSeek-VL-1.3B 以经历约 **`500B text tokens`** 训练的 DeepSeek-LLM-1B 为基础；DeepSeek-VL-7B 以经历约 **`2T text tokens`** 训练的 DeepSeek-LLM-7B 为基础。

* **<span style="color: rgb(36,91,219); background-color: inherit">三阶段 Training Pipeline</span>**

Training pipeline 依次执行 Vision-Language Adaptor warmup、joint vision-language pretraining 和 supervised fine-tuning。当前目标集中在 visual understanding，因此 loss 只计算语言部分的 next-token prediction；visual tokens 作为条件输入，不单独施加 image reconstruction objective。

<span style="color: rgb(100,37,208); background-color: inherit">三个阶段逐步扩大可训练参数范围：先只训练连接层，再让 LLM 学习 multimodal distribution，最后用 instruction data 联合调整对话行为与视觉理解。</span>

![Stage 1 只训练 VL Adaptor，Stage 2 联合训练 Adaptor 与 LLM，Stage 3 使用视觉和纯文本 chat data 完成 instruction tuning。](../../images/视觉多模态讲义（上）-three-stage-pipeline.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：Training Vision-Language Adaptor</span>**
>
> Stage 1 冻结 Hybrid Vision Encoder 和 LLM，只更新 Vision-Language Adaptor。训练数据包括 ShareGPT4V 提供的 **`125 万`**&#x7EC4; image-text caption，以及 **`250 万`**&#x7EC4; document OCR rendering pair。
>
> 这一阶段只负责在 Embedding space 中建立视觉 feature 与语言 token 的初始连接，让 LLM 能够把 Adaptor 输出解释为图像实体。它没有足够参数容量承载完整的多模态知识。
>
> > **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：Stage 1 的 Data Scaling 上限
> >
> > Two-layer MLP 的参数量远小于 LLM。把训练 step 从 **`2K`** 增加到 **`8K`**、**`20K`** 和 **`80K`**，并没有带来稳定收益，平均指标反而从 **`57.5`** 降到 55 左右。
> >
> > <span style="color: rgb(216,57,49); background-color: inherit">Projector warmup 不服从简单的数据规模扩展规律。只增加 caption 数量，无法突破 Adaptor 的容量上限，也不能替代让 LLM 直接参与 multimodal pretraining。</span>

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：Joint Vision-Language Pretraining</span>**
>
> Stage 2 冻结 Vision Encoder，更新 LLM 与 VL Adaptor。若训练数据全部由 multimodal sample 构成，MMBench、SeedBench 等视觉指标会提高，但 MMLU、HellaSwag 和 Pile-test 会明显下降。
>
> 原因来自两个方面：multimodal corpus 通常比纯文本语料简单，分布也与 LLM pretraining corpus 存在明显偏移；视觉与语言数据还会竞争有限的 model capacity，持续只训练 multimodal sample 会造成 language catastrophic forgetting。
>
> **<span style="color: rgb(222,120,2); background-color: inherit">Language-Multimodal Joint Training Ratio</span>**
>
> Stage 2 在每个训练周期同时采样 language data 与 multimodal data。增加 language data 能显著缓解 MMLU、HellaSwag 和 Pile-test 的退化，同时不会明显损害 MMBench、MMBench-CN 和 SeedBench。
>
> 最终 data mixture 采用约 **`language:multimodal=7:3`**，也就是约 70% language data 与 30% multimodal data。
>
> <span style="color: rgb(216,57,49); background-color: inherit">比例方向不能写反：language:multimodal=7:3 对应 multimodal:language=3:7；若写成 multimodal:language=7:3，就会把训练中占主导的数据类型颠倒。</span>
>
> **<span style="color: rgb(222,120,2); background-color: inherit">从 1.3B Scaling 到 7B</span>**
>
> 大量 architecture 与 data experiment 先在 1.3B Model 上完成，再把有效设计迁移到 7B。问题在于，小 Model 的 instruction following 较弱，generation-based metric 会发生尖锐波动：Model 可能知道正确选项，却不能稳定生成符合 parser 要求的答案。
>
> 训练监控改用 Multi-Choice PPL。对于 A、B、C、D 等候选项，把问题、图像和所有选项送入 Model，分别计算候选答案位置的 PPL，并选择概率最高的选项。Stage 2 还混入少量 SFT data，让 1.3B Model 先获得最低限度的指令遵循能力。
>
> <span style="color: rgb(46,161,33); background-color: inherit">Multi-Choice PPL 把生成格式噪声与知识学习进度分开，使小 Model 的训练曲线能够用于筛选可迁移到 7B 的设计。</span>

![不同 modality ratio 同时影响视觉与语言指标；纯 multimodal training 会显著抬高 Pile-test PPL 并损害 language capability。](../../images/视觉多模态讲义（上）-modality-ratio-curves.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：Supervised Fine-Tuning</span>**
>
> Stage 3 使用 Vision-Language SFT data 和纯文本对话数据，把 pretrained checkpoint 转换为 DeepSeek-VL-Chat。7B 配置中的可训练模块包括 LLM、VL Adaptor 和 SigLIP-L，但 SAM-B 因 GPU memory 限制继续冻结；1B 配置没有 SAM-B 分支。
>
> Loss 只监督 assistant answer 与 special tokens，system prompt 和 user prompt 被 mask。这样可以避免把输入提示本身当作预测目标，同时保留多轮 chat template 中的角色边界。

* **<span style="color: rgb(36,91,219); background-color: inherit">Training Hyperparameters</span>**

1B 与 7B 在 Stage 1 都使用 learning rate **`1.0×10⁻³`**、Cosine scheduler、**`15,000`** steps、batch size **`256`** 和 sequence length **`512`**。

Stage 2 改用 Step scheduler。1B 的 learning rate、training steps 和 batch size 分别为 **`3×10⁻⁵`**、**`96,000`** 和 **`1024`**；7B 分别为 **`4.2×10⁻⁵`**、**`42,000`** 和 **`2304`**。两者 sequence length 都是 **`4096`**，并启用 sequence packing。

Stage 3 的 learning rate 为 **`2.0×10⁻⁵`**，训练 **`10,000`** steps，batch size 为 **`256`**，sequence length 为 **`4096`**。所有阶段使用 AdamW，β₁=0.9、β₂=0.95，weight decay 为 0，gradient clipping 为 1.0。

![1B 与 7B 在三个训练阶段使用不同 learning rate、step、batch size 和 parallelism 配置。](../../images/视觉多模态讲义（上）-training-hyperparameters-table.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Distributed Training Infrastructure</span>**

训练使用 HAI-LLM distributed framework。Visual Encoder 输出 Embedding 后与 text Embedding 统一处理，因此在 pipeline parallelism 中，可以把 Visual Encoder 与 text Embedding 合并成第一层，后续层继续沿用 DeepSeek-LLM 的 Transformer partition。

第一层结构复杂，无法直接套用标准 tensor parallelism，但它的计算量相对上层 Transformer 较小。实现中让所有 tensor-parallel rank 重算 Vision Encoder forward，再根据各层执行时间重新分配 pipeline stage，以改善 load balance。上层 Transformer 仍可使用常规 3D parallelism，并重叠 computation 与 communication。

DeepSeek-VL-7B 在 **`64 个 node`**&#x4E0A;训练，每个 node 配置 **`8 张 NVIDIA A100`**，耗时约 **`5 天`**；1B 使用 **`16 个 node`**，耗时约 **`7 天`**。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

DeepSeek-VL 提供 1.3B 与 6.7B 参数规模，对应常用的 DeepSeek-VL-1.3B 和 DeepSeek-VL-7B 命名。它把传统 projector warmup 扩展为完整的 joint vision-language pretraining，并在训练中持续加入 language data，以控制 multimodal learning 对 LLM language capability 的破坏。

<span style="color: rgb(100,37,208); background-color: inherit">方法主线可以归纳为两点：Hybrid Vision Encoder 在 576-token budget 内融合 1024×1024 细节与 384×384 语义；三阶段 training pipeline 让 Adaptor 对齐、joint pretraining 和 instruction tuning 分别承担不同学习任务。</span>

### 2.8.2 <span style="color: rgb(36,91,219); background-color: inherit">DeepSeek-VL2</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">整体结构</span>**

DeepSeek-VL2 延续 decoder-only 的 LLaVA 风格，由 Vision Encoder、VL adaptor 和基于 MoE 的 LLM 三部分组成。图像先经过动态切图，再由共享的 SigLIP-SO400M-384 提取 visual Embedding；VL adaptor 把这些 Embedding 压缩并投影到语言模型的 Embedding 空间，最后与文本 token 一起送入 DeepSeekMoE LLM。

![DeepSeek-VL2 由 Vision Encoder、VL adaptor 和基于 MoE 的 LLM 组成，图像经过 Dynamic Tiling 后转为 visual tokens，再与文本 token 一同送入语言模型。](../../images/视觉多模态讲义（上）-figure-2.png)

<span style="color: rgb(100,37,208); background-color: inherit">整条计算链保持“视觉特征抽取—模态投影—自回归生成”的结构，DeepSeek-VL2 的变化集中在动态高分辨率输入和稀疏 MoE 语言模型两端。</span>

DeepSeek-VL 使用 SigLIP 处理 **`384×384`** 图像，用 SAM-B 处理 **`1024×1024`** 图像，再融合粗粒度与细粒度特征。固定分辨率对普通图像有效，但超长信息图、密集 OCR、文档和精细 visual grounding 往往需要更大的有效输入尺寸。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Dynamic Tiling</span>**

Dynamic Tiling 只保留一个 SigLIP-SO400M-384 Vision Encoder。它先枚举候选分辨率，候选宽高都是 **`384`** 的整数倍，局部 tile 总数最多为 **`9`**：

$$C_R=\{(384m,384n)\mid m,n\in\mathbb{N},\ 1\le m,n,\ mn\le 9\}$$

对尺寸为 $$(H,W)$$ 的输入图像，系统依次尝试每个候选分辨率。图像按原始宽高比缩放，使长边与候选尺寸对齐，再对短边补 padding；最终选择 padding 面积最小的候选项 $$(384m_i,384n_i)$$。缩放后的图像被切成 $$m_i\times n_i$$ 个 **`384×384`** local tiles，同时额外生成一个全局 thumbnail。

每个 tile 都经过同一个 SigLIP-SO400M-384。单个 tile 输出 **`27×27=729`** 个 visual Embedding，每个 Embedding 的维度是 **`1152`**。全局 thumbnail 提供整体布局，local tiles 保存局部文字和细节，两者使用同一套视觉参数。

<span style="color: rgb(46,161,33); background-color: inherit">这种切法可以处理不同宽高比的高分辨率图像，又避免直接把整张大图送入全局 Attention 所产生的二次复杂度。</span>

<span style="color: rgb(216,57,49); background-color: inherit">当一次输入包含超过两张图像时，动态切图会被关闭，以控制 visual token 数量和上下文长度。</span>

3. **<span style="color: rgb(36,91,219); background-color: inherit">Vision-Language Adaptor</span>**

Vision Encoder 输出之后，VL adaptor 先执行 **`2×2`** pixel shuffle，把每个 tile 的 visual tokens 从 **`27×27`** 压缩为 **`14×14=196`**。这里不是简单删除 token，而是把局部空间信息重新排列到更少的位置中，用较短序列承载同一个 tile 的特征。

全局 thumbnail 的每一行末尾追加一个 **`<tile_newline>`**，因此全局视图由 **`14×15=210`** 个 token 表示。local tiles 先按照 $$(14m_i,14n_i)$$ 的二维网格拼接，再在每一整行末尾加入 **`<tile_newline>`**。全局 thumbnail 与 local tiles 之间插入 **`<view_separator>`**。

<span style="color: rgb(100,37,208); background-color: inherit">这些边界 token 把二维 tile 布局显式编码进一维序列，使 LLM 能区分全局视图、局部视图和每一行的结束位置。</span>

![Dynamic Tiling 同时保留全局 thumbnail 与 local tiles，并用 view separator token 和 tile newline token 标记视图与行边界。](../../images/视觉多模态讲义（上）-figure-3.png)

一张图像最终产生的 visual token 数量为：

$$N_{\text{vis}}=210+1+14m_i(14n_i+1)$$

压缩后的序列再经过两层 MLP，维度被投影到 LLM 的 Embedding 空间。这样，visual tokens 可以直接占据语言模型输入序列中的位置。

4. **<span style="color: rgb(36,91,219); background-color: inherit">DeepSeekMoE LLM 与三种模型规模</span>**

语言端使用 DeepSeekMoE。DeepSeek-VL2-Small 和 DeepSeek-VL2 同时采用 Multi-head Latent Attention，MLA rank 为 **`512`**。MLA 把 Key-Value cache 压缩到低维 latent vector，减少长序列推理时的缓存量；MoE 则让每个 token 只激活一部分 expert，降低实际计算量。

MoE 训练为每个 expert 引入一个全局 bias，用较低成本改善不同 expert 之间的 load balancing。

系列包含 DeepSeek-VL2-Tiny、DeepSeek-VL2-Small 和 DeepSeek-VL2。三者面向推理时的总 activated parameters 分别为 **`1.0B`**、**`2.8B`** 和 **`4.5B`**；仅看 LLM，三种基座的总参数量为 **`3B`**、**`16B`** 和 **`27B`**，对应激活的 LLM 参数量为 **`0.57B`**、**`2.4B`** 和 **`4.1B`**。

Tiny 使用 Multi-Head Attention，Small 和完整版使用 MLA。三者的 routed experts 数量依次为 **`64`**、**`64`**、**`72`**，都包含 **`2`** 个 shared experts，每个 token 选择 Top-**`6`** routed experts。Tiny 和 Small 使用 Softmax routing，完整版改用 Sigmoid routing，并启用 expert correction bias。

<span style="color: rgb(46,161,33); background-color: inherit">参数总量负责模型容量，activated parameters 决定单次前向的主要计算量；DeepSeek-VL2 用这两个尺度分离容量与推理成本。</span>

![DeepSeek-VL2-Tiny、Small 与完整版的 Vocabulary、Embedding、Attention、层数和 MoE routing 配置。](../../images/视觉多模态讲义（上）-table-1-architecture.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">数据构建</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">三阶段数据组织</span>**

数据与训练阶段一一对应：VL alignment 负责打通视觉与语言 Embedding，VL pretraining 学习大规模视觉—语言知识，SFT 再把能力整理成可交互的问答、推理和 grounding 行为。每一阶段使用独立的数据集合，不把所有样本混成一套统一 recipe。

<span style="color: rgb(100,37,208); background-color: inherit">数据构建的主线是先解决模态连接，再扩大任务覆盖，最后通过高质量指令数据校正回答形式。</span>

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：Vision-Language Alignment Data</span>**
>
> Alignment 阶段使用 ShareGPT4V 中约 **`1.2M`** 个 caption 与对话样本。这个阶段主要训练 MLP connector，同时让固定分辨率预训练得到的 SigLIP 适应动态高分辨率输入。样本规模相对小，目标不是补充知识，而是让视觉特征能够被 LLM 正确接收。

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：Vision-Language Pretraining Data</span>**
>
> VL pretraining 同时保留多模态数据和 text-only 数据，两者比例约为 **`70%`** 对 **`30%`**。text-only 部分直接来自基座 LLM 的预训练语料，用来减轻多模态训练对原有语言能力的侵蚀。
>
> 交错图文数据由 WIT、WikiHow、OBELICS 的 **`30%`** 随机样本、Wanjuan 中文内容和内部数据组成。OBELICS 的抽样比例先在 DeepSeek-VL2-Tiny 上做过混合实验；Wanjuan 用于补足以英文为主的公开数据，内部数据扩充真实世界知识。
>
> **<span style="color: rgb(222,120,2); background-color: inherit">Pretraining 数据类型与质量控制</span>**
>
> Image captioning 数据先聚合多套公开数据，再统一重写 caption。原始集合的质量差异很大：有的 caption 信息密集、图文一致，有的过短、错配或带有明显幻觉。重写时把 OCR 提示、拍摄地点与相机参数等 metadata、原始 caption 一起放进 prompt，由内部 captioner 重新生成描述。
>
> 大规模自动标注容易产生重复句式，因此 DeepSeek Chat 会按写作质量为 caption 打分，用这个分数过滤低质量和重复内容。<span style="color: rgb(46,161,33); background-color: inherit">这条 pipeline 同时利用图像文字、metadata 和旧 caption，比只看图像重新描述保留了更多可验证线索。</span>
>
> OCR 数据包括 LaTeX OCR、**`12M`** RenderedText 和覆盖多类文档的内部集合。内部 OCR 主要面向英文与中文，其他语言尚未获得同等覆盖。
>
> Visual QA 数据分为四类。General VQA 沿用 DeepSeek-VL 的数据；表格、图表与文档理解使用 PubTabNet、FinTabNet 和 Docmatix；web-to-code 使用 Websight，plot-to-Python 使用公开 Jupyter Notebook 中的图表代码，并用 DeepSeek-V2.5 复刻部分 Websight 样本、生成 Python plot code 来降低噪声；visual prompt QA 会在图像上叠加箭头、方框、圆和涂鸦，再围绕被标记对象构造问题。
>
> **<span style="color: rgb(222,120,2); background-color: inherit">Visual Grounding 与 Grounded Conversation 数据</span>**
>
> Visual grounding 把 object detection 标注转换成可生成的文本序列。一个典型 prompt 使用 **`<|ref|>query<|/ref|>`** 标出查询短语，response 在同一短语后接 **`<|det|>[[x1,y1,x2,y2],…]<|/det|>`**。查询可以是类别名，也可以是“最左边的人”这类描述。
>
> 每个 bounding box 由左上角和右下角坐标表示，四个坐标都按当前图像尺寸归一化到 **`0–999`**。训练数据还包含目标不存在的 negative samples，避免模型在任何查询下都强行返回一个位置。
>
> Grounded conversation 在普通描述前加 **`<|grounding|>`**，回答中的对象名称仍由 ref token 包裹，紧接着输出 det token 与位置。<span style="color: rgb(100,37,208); background-color: inherit">同一套序列同时表达自然语言、指代对象和空间坐标，grounding 因而可以嵌入多轮对话，而不是作为独立检测头运行。</span>

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：Supervised Fine-Tuning Data</span>**
>
> General VQA 的公开数据常见三个问题：回答过短、OCR 质量差、回答内容与图像不一致。SFT 数据会联合原问题、原图像和 OCR 信息重新生成 response，使答案更完整，也减少由错误文字识别带来的幻觉。
>
> 早期 DeepSeek-VL2-Tiny 在中文回答中偶尔无理由插入英文词，较大模型没有同样现象。原因与模型容量以及 VL pretraining 中英文数据不平衡有关。为此，Tiny 增加了内部中文 QA，覆盖多样化图像描述、单轮与多轮对话，同时补充动漫、梗图、饮食和艺术等真实世界与文化视觉知识。
>
> <span style="color: rgb(216,57,49); background-color: inherit">更长、更详细的 reasoning response 对小模型并不总是更好；DeepSeek-VL2-Tiny 使用更简洁的答案时表现更强。</span>面向 reasoning、逻辑和数学的数据仍会补充详细推理过程，并统一把最终答案放在 response 末尾，但不同规模采用的答案长度需要区分。
>
> OCR 与文档理解部分主要清洗已有公开数据，删除 OCR 质量差的样本；内部文档页被转换成多轮 QA。表格 QA 除 Cauldron 外都会基于原问题重写 response，Cauldron 已有较高质量。图表理解在 pretraining 阶段已经形成较强能力，SFT 没有再单独堆叠额外数据。
>
> 其余数据覆盖大学层级教材与学术问题、扩充后的 web-to-code 和 plot-to-Python、visual grounding、grounded conversation 与 text-only instruction tuning。Grounding 数据会把查询短语翻译成中文，加入更多 negative samples，并构造 in-context grounding：第一张图用矩形或椭圆标出参考对象，模型需要在第二张图中找到同类对象。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练方法</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">三阶段 Training Pipeline</span>**

训练分为 Vision-Language Alignment、Vision-Language Pretraining 和 SFT 三个阶段。Stage 1 固定 LLM，只更新 Vision Encoder 与 VL adaptor；Stage 2 和 Stage 3 解冻全部参数，Vision Encoder、VL adaptor 与 DeepSeekMoE LLM 同时训练。

<span style="color: rgb(100,37,208); background-color: inherit">三个阶段都只在文本 token 上计算 next-token prediction loss，图像对应的 visual tokens 作为条件输入，不直接承担语言建模标签。</span>

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：Vision-Language Alignment</span>**
>
> Stage 1 从 DeepSeekMoE **`3B`**、**`16B`** 和 **`27B`** 基座出发。LLM 参数保持冻结，Vision Encoder 与两层 MLP adaptor 参与优化。与固定 Vision Encoder 的常见 alignment 方法不同，这一步会让 SigLIP 适应动态切图后的高分辨率分布。
>
> 三种规模在 Stage 1 都训练约 **`2.0B`** tokens，batch size 为 **`256`**。Vision Encoder 的 learning-rate multiplier 固定为 **`0.1`**，使视觉 Backbone 的更新速度低于新加入的 adaptor。

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：Vision-Language Pretraining</span>**
>
> Stage 2 解冻所有模块，把大部分训练计算投入联合视觉—语言预训练。Tiny、Small 和完整版分别使用约 **`798.5B`**、**`808.9B`**、**`796.5B`** tokens，量级都接近 **`800B`**。
>
> Stage 2 的 batch size 在 Tiny 与 Small 上是 **`2304`**，完整版提高到 **`3360`**。三者的 sequence length 都是 **`4096`**，并启用 sequence packing 和 pipeline parallelism。
>
> <span style="color: rgb(46,161,33); background-color: inherit">联合解冻使视觉表示、模态投影和语言生成可以围绕同一批大规模任务共同适配，同时用 text-only 数据保留基座语言能力。</span>

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3：Supervised Fine-Tuning</span>**
>
> Stage 3 继续更新全部参数，但只监督 answer token 与 special token，system prompt 和 user prompt 被 mask。多模态 SFT 数据与 DeepSeek-V2 的纯文本对话数据混合，以同时保持对话理解和视觉任务能力。
>
> Tiny、Small 和完整版的 SFT tokens 分别为 **`19.5B`**、**`20.0B`** 和 **`19.5B`**，batch size 都是 **`64`**。任务覆盖 dense caption、General VQA、OCR、表格/图表/文档/figure 理解、visual-to-code、visual reasoning、visual grounding 和纯语言理解。

* **<span style="color: rgb(36,91,219); background-color: inherit">Optimization Recipe 与关键超参数</span>**

Tiny 在三个阶段的 learning rate 依次为 **`5.4×10⁻⁴`**、**`5.4×10⁻⁴`**、**`3.0×10⁻⁵`**；Small 为 **`4.2×10⁻⁴`**、**`4.2×10⁻⁴`**、**`1.4×10⁻⁵`**；完整版为 **`4.5×10⁻⁴`**、**`4.5×10⁻⁴`**、**`2.0×10⁻⁵`**。

三个阶段依次采用 Cosine、Step、Constant learning-rate scheduler。Step scheduler 在总训练步数的 **`50%`** 和 **`75%`** 处把 learning rate 除以 $$\sqrt{10}$$。所有模型都使用 AdamW，$$\beta_1=0.9$$、$$\beta_2=0.95$$，weight decay 为 **`0.1`**，gradient clipping 为 **`1.0`**。

Tiny 与 Small 的 MoE auxiliary loss weight 为 **`0.001`**，完整版降到 **`0.0001`**。只有完整版在三个阶段都使用 BF16 optimizer，并在 Stage 2 以 **`0.001`** 的 correction step 更新 expert bias。

![三个模型在 Alignment、Pretraining 和 SFT 阶段的 learning rate、训练 tokens、batch、sequence length 与并行配置。](../../images/视觉多模态讲义（上）-table-2-training-hyperparameters.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Training Infrastructure</span>**

训练与评测运行在 HAI-LLM 平台上。Vision Encoder 位于 pipeline 首端，单个模块的计算形态又不同于后续 LLM blocks，直接切分容易形成 pipeline bubble。处理方式是把 Vision Encoder 进一步按层细分，让它的计算可以更均匀地分配到多张 GPU。

Dynamic Tiling 会让不同样本包含不同数量的 tiles，data parallel ranks 之间的前向与反向负载因此不一致。训练系统会按 tile 数量重新做 load balancing，并同时使用 tensor parallelism 与 expert parallelism。纯文本 batch 和图文 batch 的执行路径不同，因此分别配置两套 pipeline strategy，运行时按 batch 类型切换。

Tiny、Small 和完整版分别训练 **`7`**、**`10`**、**`14`** 天，对应使用 **`16`**、**`33`**、**`42`** 个节点；每个节点配置 **`8`** 张 NVIDIA A100。

### 2.8.3 <span style="color: rgb(36,91,219); background-color: inherit">DeepSeek-OCR</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

当前 LLM 在处理长文本内容时面临巨大的计算挑战，因为<span style="color: rgb(216,57,49); background-color: inherit">计算复杂度随序列长度呈二次方增长</span>。&#x4F46;**<span style="color: rgb(46,161,33); background-color: inherit">一张包含文档文本的图像可以用比数字文本少得多的 token 来表示丰富的信息</span>**<span style="color: rgb(46,161,33); background-color: inherit">，这可以通过视觉 token 进行光学压缩可能实现更高的压缩比</span>。

作者从以 LLM 为中心的视角出发，关注视觉编码器如何提高 LLM 处理文本信息的效率，而非仅仅关注 VQA 任务。<span style="color: rgb(100,37,208); background-color: inherit">OCR 任务作为视觉和语言之间的中间模态，为视觉-文本压缩范式提供了理想的测试平台，因为它在视觉和文本表示之间建立了自然的压缩-解压缩映射，并提供了定量评估指标</span>。因此提出 DeepSeek-OCR 作为高效视觉-文本压缩的可行性初步证明，<span style="color: rgb(100,37,208); background-color: inherit">探索使用视觉模态作为 LLM 文本信息处理的高效压缩介质</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

<span style="color: rgb(100,37,208); background-color: inherit">DeepSeek-OCR 采用统一的端到端视觉语言模型 VLM 架构，由编码器和解码器组成</span>。

> 1. 编码&#x5668;**`DeepEncoder`**&#x8D1F;责提取图像特征，并对视觉表征进行分词与压缩，DeepEncoder 的参数量约&#x4E3A;**`0.38B`**，主要由一&#x4E2A;**`0.08B`**&#x53C2;数&#x7684;**`SAM-base`**&#x548C;一&#x4E2A;**`0.3B`**&#x53C2;数&#x7684;**`CLIP-large`**&#x4E32;联构成。
>
> 2. 解码器基于图像分词和 prompt 生成所需结果，采&#x7528;**`3B`**&#x53C2;数的 MoE 架构，激活参数约&#x4E3A;**`0.57B`**。

![](../../images/视觉多模态讲义（上）-image-194.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">DeepEncoder  </span>**

为了能进行上下文光学压缩，需要满足下面几个特性的视觉编码器： &#x20;

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">能处理高分辨率图像</span>**
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">在高分辨率下激活量低</span>**
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">视觉 token 数量少</span>**
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">支持多分辨率输入 </span>**
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">参数量适中  </span>**

但<span style="color: rgb(216,57,49); background-color: inherit">当前开源的编码器无法完全满足上述所有条件</span>。因此 DeepSeek 自行设计了一个新型视觉编码器 DeepEncoder。DeepEncoder 由三部分组成：

> * 以窗口注意&#x529B;**`window attention`**&#x4E3A;主的视觉感知特征提取模块：采用$$\text{patch\_size}=16$$&#x7684;**`SAM-base`**
>
> * 两层卷积模块对视觉 token 进&#x884C;**`16`**&#x500D;下采样。每层卷积核大小&#x4E3A;**`3`**，步长&#x4E3A;**`2`**，填充&#x4E3A;**`1`**，通道数&#x4ECE;**`256`**&#x589E;加&#x5230;**`1024`**
>
> * 使用密集全局注意&#x529B;**`dense global attention`**&#x7684;视觉知识特征提取模块，使&#x7528;**`CLIP-large`**，由于<span style="color: rgb(100,37,208); background-color: inherit">输入不再是原始图像而是卷积模块输出的 token，因此移除了首个 patch Embedding 层</span>

> **<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：输入一张$$1024 \times 1024$$的图像，<span style="color: rgb(100,37,208); background-color: inherit">DeepEncoder 会将其划分为</span>$$1024/16 \times 1024/16 = 4096$$<span style="color: rgb(100,37,208); background-color: inherit">个 patch</span>。由于编码器前半部分以窗口注意力为主且仅&#x542B;**`80M`**&#x53C2;数，计算量和参数量还是比较可以接受的。在<span style="color: rgb(100,37,208); background-color: inherit">进入全局注意力之前，这</span>**`4096`**<span style="color: rgb(100,37,208); background-color: inherit">个 token 会通过压缩模块，数量减少为</span>$$4096/16 = 256$$，从而使整体内存比较小。

再看这样一个场景：假设有一张包&#x542B;**`1000`**&#x4E2A;光学字符的图像，并希望测试解码所需视觉 token 的数量。这就要求模型支持可变数量的视觉 token，这就<span style="color: rgb(100,37,208); background-color: inherit">要求 DeepEncoder 需支持多分辨率输入</span>。

<span style="color: rgb(100,37,208); background-color: inherit">DeepSeek 通过位置编码的动态插值满足这一点，并设计了多种分辨率用于联合训练</span>，使 DeepSeek-OCR 具备多分辨率支持能力。<span style="color: rgb(100,37,208); background-color: inherit">DeepEncoder 主要支持两种输入模式：原生分辨率</span>**`native resolution`**<span style="color: rgb(100,37,208); background-color: inherit">和动态分辨率</span>**`dynamic resolution`**<span style="color: rgb(100,37,208); background-color: inherit">，每种模式下包含若干子模式</span>。

![](../../images/视觉多模态讲义（上）-image-192.png)

**<span style="color: rgb(222,120,2); background-color: inherit">原生分辨率</span>**&#x5305;含四种子模式：

<table><colgroup><col width="274"><col width="274"><col width="274"></colgroup>
<thead>
<tr>
<th>模式</th>
<th>分辨率</th>
<th>Token 数量</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong><code>Tiny</code></strong></td>
<td>512 \times 512</td>
<td>64</td>
</tr>
<tr>
<td><strong><code>Small</code></strong></td>
<td>640 \times 640</td>
<td>100 </td>
</tr>
<tr>
<td><strong><code>Base</code></strong></td>
<td>1024 \times 1024</td>
<td>256</td>
</tr>
<tr>
<td><strong><code>Large</code></strong></td>
<td>1280 \times 1280</td>
<td>400</td>
</tr>
</tbody>
</table>

> * 对&#x4E8E;**`Tiny`**&#x548C;**`Small`**&#x6A21;式，由于分辨率较低，为避免浪费视觉 token ，<span style="color: rgb(100,37,208); background-color: inherit">图像直接缩放至目标尺寸</span>
>
> * 对&#x4E8E;**`Base`**&#x548C;**`Large`**&#x6A21;式，为<span style="color: rgb(100,37,208); background-color: inherit">保留原始图像的宽高比，图像会被填充至对应尺寸</span>。填充后，有效视觉 token 数少于实际分词数，其计算公式为：
>
> $$N_{\text{valid}} = \left\lceil N_{\text{actual}} \times \left[1 - \frac{\max(w, h) - \min(w, h)}{\max(w, h)}\right] \right\rceil$$
>
> 其中$$w$$和$$h$$分别表示原始输入图像的宽度和高度。

**<span style="color: rgb(222,120,2); background-color: inherit">动态分辨率</span>**&#x53EF;由两种原生分辨率组合而成。

> * **<span style="color: rgb(100,37,208); background-color: inherit">Gundam 模式</span>**<span style="color: rgb(100,37,208); background-color: inherit">由</span>$$n$$<span style="color: rgb(100,37,208); background-color: inherit">个</span>$$640 \times 640$$<span style="color: rgb(100,37,208); background-color: inherit">的局部视图和一个</span>$$1024 \times 1024$$<span style="color: rgb(100,37,208); background-color: inherit">的全局视图组成</span>，分块方法&#x4E0E;**`InternVL2`**&#x4E00;致。<span style="color: rgb(100,37,208); background-color: inherit">支持动态分辨率主要是出于应用考虑，尤其适用于超高分辨率输入</span>，如<span style="color: rgb(220,155,4); background-color: inherit">报纸图像</span>。分块本质上是一种二次窗口注意力机制，可进一步有效降低激活内存。由于原生分辨率已相对较大，在<span style="color: rgb(100,37,208); background-color: inherit">动态分辨率下图像不会被过度碎片化，分块数量一般控制在</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">到</span>**`9`**<span style="color: rgb(100,37,208); background-color: inherit">之间</span>。在 Gundam 模式下，<span style="color: rgb(100,37,208); background-color: inherit">DeepEncoder 输出的视觉 token 总数为</span>$$n \times 100 + 256$$<span style="color: rgb(100,37,208); background-color: inherit">，其中</span>$$n$$<span style="color: rgb(100,37,208); background-color: inherit">为分块数量</span>。若图像的宽高均小于$$640$$，则$$n=0$$，即 Gundam 模式退化为 Base 模式。<span style="color: rgb(100,37,208); background-color: inherit">Gundam 模式与四种原生分辨率模式一同训练，实现单一模型支持多分辨率的目标</span>。
>
> * **<span style="color: rgb(100,37,208); background-color: inherit">Gundam-master 模式</span>**<span style="color: rgb(100,37,208); background-color: inherit">，即</span>$$1024 \times 1024$$<span style="color: rgb(100,37,208); background-color: inherit">局部视图 +</span>$$1280 \times 1280$$<span style="color: rgb(100,37,208); background-color: inherit">全局视图，是在已训练好的 DeepSeek-OCR 模型基础上继续训练得到的</span>。这主要是出于负载均衡考虑，因为Gundam-master 模式的分辨率过大，与其他模式联合训练会显著拖慢整体训练速度。

原生分辨率和动态分辨率下各个模式的总的参数如右图：





![](../../images/视觉多模态讲义（上）-image-193.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">MoE 解码器  </span>**

解码器采&#x7528;**`DeepSeek-3B-MoE`**。推理时模型&#x4ECE;**`64`**&#x4E2A;路由专家中激&#x6D3B;**`6`**&#x4E2A;，并额外激&#x6D3B;**`2`**&#x4E2A;共享专家，激活参数约&#x4E3A;**`0.57B`**。**`3B`**<span style="color: rgb(46,161,33); background-color: inherit">参数的 DeepSeekMoE 非常适合领域专用的 VLM 研究，比如 OCR，因为在保持</span>**`3B`**<span style="color: rgb(46,161,33); background-color: inherit">模型表达能力的同时，有</span>**`0.5B`**<span style="color: rgb(46,161,33); background-color: inherit">小模型的推理效率</span>。解码器从 DeepEncoder 压缩后的潜在视觉 token 中重建原始文本表示：

$$f_{\text{dec}} : \mathbb{R}^{n \times d_{\text{latent}}} \rightarrow \mathbb{R}^{N \times d_{\text{text}}};\quad \hat{X} = f_{\text{dec}}(Z),\quad \text{其中 } n \leq N$$

其中$$Z \in \mathbb{R}^{n \times d_{\text{latent}}}$$是 DeepEncoder 输出的压缩潜在视觉 token，$$\hat{X} \in \mathbb{R}^{N \times d_{\text{text}}}$$ 是重建的文本表示。<span style="color: rgb(100,37,208); background-color: inherit">函数</span>$$f_{\text{dec}}$$<span style="color: rgb(100,37,208); background-color: inherit"> 表示非线性映射，可通过 OCR 风格的训练被紧凑语言模型有效学习</span>。<span style="color: rgb(46,161,33); background-color: inherit">LLM 通过专门的预训练优化，将更自然地融合此类能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据</span>**

训练数据包含这几个方面： &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">OCR 1.0 数据</span>**：涵盖<span style="color: rgb(100,37,208); background-color: inherit">传统 OCR 任务</span>，如场景图像 OCR 和文档 OCR
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">OCR 2.0 数据</span>**：包括<span style="color: rgb(100,37,208); background-color: inherit">复杂人工图像的解析任务</span>，如常见图表、化学式和平面几何解析数据
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">通用视觉数据</span>**：主要用于<span style="color: rgb(100,37,208); background-color: inherit">向 DeepSeek-OCR 注入一定的通用图像理解能力</span>，保留通用视觉接口

1. **<span style="color: rgb(36,91,219); background-color: inherit">OCR 1.0 数据  </span>**

文档数据是 DeepSeek-OCR 的首要任务。<span style="color: rgb(100,37,208); background-color: inherit">从互联网收集了约</span>**`30M`**<span style="color: rgb(100,37,208); background-color: inherit">页涵盖约</span>**`100`**<span style="color: rgb(100,37,208); background-color: inherit">种语言的多样化 PDF 数据，其中中文和英文约占</span>**`25M`**<span style="color: rgb(100,37,208); background-color: inherit">页，其他语言约占</span>**`5M`**<span style="color: rgb(100,37,208); background-color: inherit">页</span>。根据这些数据构建了两类标注：粗粒度标注和细粒度标注。

![](../../images/视觉多模态讲义（上）-image-191.png)

> * **<span style="color: rgb(36,91,219); background-color: inherit">粗粒度标注</span>**&#x76F4;接使用 fitz 从完整数据集中提取，教会模型识别光学文本，尤其是小语种文本。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">细粒度标注</span>**&#x5305;括中英文&#x5404;**`2M`**&#x9875;，使用比较强的布局模型&#x5982;**`PP-DocLayout`**&#x548C; OCR 模型&#x5982;**`MinuerU`**&#x548C;**`GOT-OCR2.0`**&#x6784;建检测与识别交错的数据。

对于小语种，布局模型具有一定泛化能力；<span style="color: rgb(100,37,208); background-color: inherit">在识别部分，使用</span>**`fitz`**<span style="color: rgb(100,37,208); background-color: inherit">生成小块数据训练</span>**`GOT-OCR2.0`**<span style="color: rgb(100,37,208); background-color: inherit">，再用训练好的模型对版面处理后的小块进行标注，通过模型飞轮机制构建了</span>**`600K`**<span style="color: rgb(100,37,208); background-color: inherit">条数据样本</span>。在 DeepSeek-OCR 训练过程中，粗标注和细标注通过不同 prompt 加以区分。细标注图像-文本对的示例如上图。此外还<span style="color: rgb(100,37,208); background-color: inherit">收集了</span>**`3M`**<span style="color: rgb(100,37,208); background-color: inherit">份 Word 文档数据，通过直接提取内容构建高质量的无版面图像-文本对</span>。这类数据主要对公式和 HTML 格式表格的识别有益，选取部分开源数据作为补充。对于自然场景 OCR，模型主要支持中英文。<span style="color: rgb(100,37,208); background-color: inherit">图像数据来源于</span>**`LAION`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`Wukong`**<span style="color: rgb(100,37,208); background-color: inherit">，使用 PaddleOCR 进行标注，中英文各</span>**`100M`**<span style="color: rgb(100,37,208); background-color: inherit">条样本</span>。与文档 OCR 类似，自然场景 OCR 也可通过 prompt 控制是否输出检测框。

* **<span style="color: rgb(36,91,219); background-color: inherit">OCR 2.0 数据  </span>**

将图表、化学式和平面几何解析数据统称为 OCR 2.0 数据。 &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">图表数据</span>**：使用 pyecharts 和 matplotlib 渲染 1000 万张图像，主要包括<span style="color: rgb(100,37,208); background-color: inherit">常用的折线图、柱状图、饼图和复合图表</span>。将图表解析定义为图像到 HTML 表格的转换任务，如下图(a)&#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">化学式</span>**：以 PubChem 中的 SMILES 格式为数据源，使用 RDKit 渲染为图像，构建了 500 万对图像-文本数据
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">平面几何图像</span>**：遵循 Slow Perception 的方法生成，使用perception-ruler大小为 4 来建模每条线段。为增加渲染数据的多样性，引入了几何平移不变性数据增强：将同一几何图形在原始图像中平移，对应在坐标系中心位置绘制的相同标注。共构建了 100 万条平面几何解析数据，如下图(b)所示。

![](../../images/视觉多模态讲义（上）-image-190.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">通用视觉数据  </span>**

DeepEncoder 受益于 CLIP 的预训练优势，并具备足够参数以融合通用视觉知识。根据这一点为 DeepSeek-OCR 准备了相应数据。<span style="color: rgb(100,37,208); background-color: inherit">根据 DeepSeek-VL2 生成了用于图像描述 caption、目标检测和定位 grounding 等任务的相关数据</span>。由于 DeepSeek-OCR 不是通用的 VLM 模型，这类数据仅占总数据的 20%。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：<span style="color: rgb(100,37,208); background-color: inherit">为确保模型的语言能力，引入了 10% 的内部纯文本预训练数据</span>，所有数据均处理&#x4E3A;**`8192`**&#x4E2A; token 的长度，这也是 DeepSeek-OCR 的序列长度。综上，DeepSeek-OCR 的训练数据构成如下：
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">OCR 数据</span>**：**`70%`**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">通用视觉数据</span>**：**`20%`**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">纯文本数据</span>**：**`10%`**

* **<span style="color: rgb(36,91,219); background-color: inherit">训练 </span>**

包括两个阶段： &#x20;

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">独立训练 DeepEncoder</span>**：<span style="color: rgb(100,37,208); background-color: inherit">使用紧凑语言模型以 next token prediction 方式训练 DeepEncoder</span>。使用所有 OCR 1.0 和 OCR 2.0 数据，以及从 LAION 数据集中采样&#x7684;**`100M`**&#x6761;通用数据。所有数据训&#x7EC3;**`2`**&#x4E2A; epoch，批大小&#x4E3A;**`1280`**，优化器&#x4E3A;**`AdamW`**，学习率调度采用余弦退火策略，初始学习率为$$5 \times 10^{-5}$$，训练序列长度&#x4E3A;**`4096`**。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">训练完整的 DeepSeek-OCR</span>**：使用所有数据训练 DeepSeek-OCR，整个训练过程&#x5728;**`HAI-LLM`**&#x5E73;台上进行。<span style="color: rgb(100,37,208); background-color: inherit">整个模型采用</span> <span style="color: rgb(216,57,49); background-color: inherit">流水线并行</span>**`PP`**<span style="color: rgb(100,37,208); background-color: inherit">，划分为</span>**`4`**<span style="color: rgb(100,37,208); background-color: inherit">个部分：DeepEncoder 占</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">部分，解码器占</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">部分</span>。
>
>    1. **<span style="color: rgb(36,91,219); background-color: inherit">DeepEncoder</span>**
>
>       1. 将 SAM 和压缩模块视为视觉 tokenizer，<span style="color: rgb(100,37,208); background-color: inherit">置于</span>**`PP0`**<span style="color: rgb(100,37,208); background-color: inherit">并冻结其参数</span>
>
>       2. 将 CLIP 部分视为输入嵌入层，<span style="color: rgb(100,37,208); background-color: inherit">置</span>&#x4E8E;**`PP1`**<span style="color: rgb(100,37,208); background-color: inherit">并解冻权重进行训练</span>
>
>    2. **<span style="color: rgb(36,91,219); background-color: inherit">语言模型</span>**：由于 DeepSeek-3B-MoE 共&#x6709;**`12`**&#x5C42;，因此&#x5728;**`PP2`**&#x548C;**`PP3`**&#x4E0A;各放&#x7F6E;**`6`**&#x5C42;
>
>    训练使&#x7528;**`20`**&#x4E2A;节点，每个节点&#x542B;**`8`**&#x5757;**`A100-40G GPU`**，数据并行&#x5EA6;**`DP`**&#x4E3A;**`40`**，全局批大小&#x4E3A;**`640`**。优化器为 **`AdamW`**，采用基于步数的学习率调度，初始学习率为$$3 \times 10^{-5}$$。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span> <span style="color: rgb(100,37,208); background-color: inherit">：</span>Gundam-master 模式**在预训练好的 DeepSeek-OCR 上，使&#x7528;**`6M`**&#x6761;采样数据继续训练

### 2.8.4 <span style="color: rgb(36,91,219); background-color: inherit">DeepSeek-OCR 2</span>

整体架构上 DeepSeek-OCR 2 和 DeepSeek-OCR 是类似的，<span style="color: rgb(100,37,208); background-color: inherit">由一个编码器和一个解码器组成，编码器将图像离散化为视觉 token，解码器基于这些视觉 token 和文本 prompt 生成输出</span>。主要的区别在于编码器 DeepEncoder，**<span style="color: rgb(100,37,208); background-color: inherit">DeepSeek-OCR 2 将其升级为 DeepEncoder V2，在保留 DeepSeek-OCR 能力的同时，通过新的架构设计引入因果推理能力</span>**。

![](../../images/视觉多模态讲义（上）-image-206.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">DeepEncoder V2</span>**

DeepEncoder 通过注意力机制提取并压缩图像特征，其中每个 token 都关注所有其他 token，从而实现全图像感受野。但是<span style="color: rgb(216,57,49); background-color: inherit">将二维图像块展平为一维序列会通过文本的位置编码引入刚性的顺序偏置，这与自然的视觉阅读模式矛盾，尤其在光学文本、表单和表格等非线性布局中更为明显</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">视觉分词</span>**

DeepEncoder V2 的第一个组件是视觉 tokenizer。<span style="color: rgb(100,37,208); background-color: inherit">沿用 DeepEncoder 的设计，采用了一个</span>**`80M`**<span style="color: rgb(100,37,208); background-color: inherit">的</span>**`SAM-base`**<span style="color: rgb(100,37,208); background-color: inherit">和两个卷积层的架构。最终卷积层的输出维度从 DeepEncoder 中的</span>**`1024`**<span style="color: rgb(100,37,208); background-color: inherit">降低至</span>**`896`**<span style="color: rgb(100,37,208); background-color: inherit">，与后续流程对齐</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：这种基于压缩的 tokenizer 并非必需，也可以使用简单的 patch embedding。DeepSeek-OCR 2 保留它是因为<span style="color: rgb(100,37,208); background-color: inherit">通过窗口注意力实现了</span>$$16\times$$<span style="color: rgb(100,37,208); background-color: inherit">的 token 压缩，且参数量极小，降低了后续全局注意力模块的计算开销和激活内存</span>。此外，**`80M`**<span style="color: rgb(100,37,208); background-color: inherit">的参数量与 LLM 中的文本 Embedding 参数量（约</span>**`100M`**<span style="color: rgb(100,37,208); background-color: inherit">）相当</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉编码器</span>**

在 DeepEncoder 的视觉 tokenizer 之后使用 CLIP ViT ，用于压缩视觉知识。DeepEncoder V2 <span style="color: rgb(100,37,208); background-color: inherit">将视觉编码器设计为</span>**`dual-stream attention`**<span style="color: rgb(100,37,208); background-color: inherit">的 LLM 架构。视觉 token 使用双向注意力保留 CLIP 的全局建模能力，而新引入的 Causal Query 采用</span>**`Causal Attention`**。这些可学习的 Query 被附加在视觉 token 之后作为后缀，每个 Query 关注所有视觉 token 及其前面的 Query。

![](../../images/视觉多模态讲义（上）-image-203.png)

DeepEncoder V2 <span style="color: rgb(100,37,208); background-color: inherit">保持 Query 与视觉 token 数量相等，在不改变 token 总数的前提下对视觉特征施加语义排序与蒸馏</span>。最终，只把 Causal Query 的输出送入 LLM Decoder。这里的 LLM 使用 Qwen2-0.5B，**`500M`**&#x7684;参数量&#x4E0E;**`300M`**&#x53C2;数量的 CLIP ViT 相当，未引入过多计算开销。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：这里 <span style="color: rgb(100,37,208); background-color: inherit">Decoder-only 的架构配合视觉 token 的前缀拼接至关重要</span>，因为作者实验发现&#x5728;**`mBART`**&#x98CE;格的 Encoder-Decoder 结构中使用交叉注意力无法收敛。因为<span style="color: rgb(216,57,49); background-color: inherit">当视觉 token 被隔离在独立编码器中时，缺乏充分的交互</span>。相比之下，<span style="color: rgb(46,161,33); background-color: inherit">前缀设计使视觉 token 活跃在在所有层中，从而促进其与 Causal Query 之间的有效信息交换</span>。

这个架构实际上建立了两阶段级联的因果推理：**<span style="color: rgb(100,37,208); background-color: inherit">编码器通过可学习的 Query 对视觉 token 进行语义重排序，而 LLM Decoder 则在重排序后的序列上执行自回归推理</span>**。这与通过位置编码强加刚性空间顺序的原始编码器不同，因果排序 Query 能够适应平滑的视觉语义，并自然契合 LLM 的单向注意力模式。这可以填补二维空间结构与一维因果语言建模之间的鸿沟。

* **<span style="color: rgb(36,91,219); background-color: inherit">Causal Query</span>**

Casual Query 的数量等于视觉 token 的数量，计算公式为$$\frac{W \times H}{16^2 \times 16}$$，其中$$W$$和$$H$$分别表示输入到编码器的图像的宽度和高度。<span style="color: rgb(100,37,208); background-color: inherit">为避免为不同分辨率使用多种 Query 集合，DeepEncoder V2 采用</span>**`multi-crop`**<span style="color: rgb(100,37,208); background-color: inherit">策略，在预定义分辨率下使用固定的 Query 配置</span>。

![](../../images/视觉多模态讲义（上）-image-207.png)

如上图，<span style="color: rgb(100,37,208); background-color: inherit">全局视图使用</span>$$1024 \times 1024$$<span style="color: rgb(100,37,208); background-color: inherit">的分辨率，对应</span>$$256$$<span style="color: rgb(100,37,208); background-color: inherit">个 Query Embedding，记为</span>$$\text{query}_{\text{global}}$$<span style="color: rgb(100,37,208); background-color: inherit">。局部裁剪采用</span>$$768 \times 768$$<span style="color: rgb(100,37,208); background-color: inherit">的分辨率，裁剪数量</span>$$k$$<span style="color: rgb(100,37,208); background-color: inherit">范围为</span>$$0$$<span style="color: rgb(100,37,208); background-color: inherit">到</span>$$6$$，当图像两个维度均小于$$768$$时不进行裁剪。所有<span style="color: rgb(100,37,208); background-color: inherit">局部视图共享一组</span>$$144$$<span style="color: rgb(100,37,208); background-color: inherit">个 Query Embedding，记为</span>$$\text{query}_{\text{local}}$$。因此，<span style="color: rgb(100,37,208); background-color: inherit">送入 LLM 的重排序视觉 token 总数为</span>$$k \times 144 + 256$$<span style="color: rgb(100,37,208); background-color: inherit">，范围为</span>$$[256, 1120]$$。最大 token 数$$1120$$低于 DeepSeek-OCR Gundam 模式下的$$1156$$，与 Gemini-3-Pro 的最大视觉 token 数相当。

* **<span style="color: rgb(36,91,219); background-color: inherit">Attention Mask</span>**

注意力掩码由两个区域组成：<span style="color: rgb(100,37,208); background-color: inherit">左侧区域的原始视觉 token 使用</span>**<span style="color: rgb(100,37,208); background-color: inherit">双向注意力</span>**，与 ViT 相同，所有 token 之间完全可见；<span style="color: rgb(100,37,208); background-color: inherit">右侧区域的 Casual token 使用</span>**<span style="color: rgb(100,37,208); background-color: inherit">因果注意力</span>**，与 Decoder-only 的 LLM 相同，每个 token 仅关注前面的 token。这<span style="color: rgb(100,37,208); background-color: inherit">两个部分沿序列维度拼接</span>，构成 DeepEncoder V2 的注意力掩码$$\mathbf{M}$$：

![](../../images/视觉多模态讲义（上）-image-205.png)

$$\mathbf{M} =
\begin{bmatrix}
\mathbf{1}_{m \times m} & \mathbf{0}_{m \times n} \\
\mathbf{1}_{n \times m} & \text{LowerTri}(n)
\end{bmatrix},
\quad \text{其中 } n = m$$

其中$$n$$为 Casual Query 的数量，$$m$$为原始视觉 token 的数量，$$\text{LowerTri}(n)$$表示下三角矩阵。

* **<span style="color: rgb(36,91,219); background-color: inherit">DeepSeek-MoE Decoder</span>**

DeepSeek-OCR 2 <span style="color: rgb(100,37,208); background-color: inherit">保留了 DeepSeek-OCR 中的</span>**`3B`**<span style="color: rgb(100,37,208); background-color: inherit">的 MoE 解码器，约有</span>**`500M`**<span style="color: rgb(100,37,208); background-color: inherit">个激活参数</span>。DeepSeek-OCR 2 的核心前向传播过程可表示为：

$$\mathbf{O} = D\left( \pi_{\mathbf{Q}} \left( T_L \left( E(\mathbf{I}) \oplus \mathbf{Q}_0; \mathbf{M} \right) \right) \right)$$

其中$$\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$$为输入图像，$$E$$为视觉 tokenizer，将图像映射为$$m$$个视觉 token $$\mathbf{V} \in \mathbb{R}^{m \times d}$$，$$\mathbf{Q}_0 \in \mathbb{R}^{n \times d}$$为可学习的 Casual Query Embedding，$$\oplus$$表示序列拼接，$$T_L$$表示带掩码注意力的$$L$$层 Transformer，$$\mathbf{M} \in \{0,1\}^{2n \times 2n}$$为分块注意力掩码，$$\pi_{\mathbf{Q}}$$为投影算子，用于提取最后$$n$$个 token，即$$\mathbf{Z} = \mathbf{X}_{m+1:m+n}$$，$$D$$为 LLM Decoder，$$\mathbf{O} \in \mathbb{R}^{n \times |\mathcal{V}|}$$为 LLM 词表上的输出 logits。

## 2.9 <span style="color: rgb(36,91,219); background-color: inherit">其他</span>

### 2.9.1 <span style="color: rgb(36,91,219); background-color: inherit">Kimi K2.5</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">联合多模态训练</span>**

Kimi K2.5 是原生多模态模型。以 Kimi K2 为基础，通过对大&#x7EA6;**`15T`**&#x4E2A;图文混合 Token 进行大规模的联合预训练打造而来。K2.5 不是为了适配视觉而不得不割裂或牺牲语言及视觉能力的模型，而是采用联合预训练模式，能够同时增强这两种模态的表现。

1. **<span style="color: rgb(36,91,219); background-color: inherit">原生多模态预训练</span>**

设计多模态预训练时的一个关键问题是：<span style="color: rgb(100,37,208); background-color: inherit">在固定的视觉-文本 token 总预算下，什么是最佳的联合训练策略</span>？传统的做法倾向于在语言模型训练的后期，以高比例引入视觉 token，例&#x5982;**`50%`**<span style="color: rgb(220,155,4); background-color: inherit">或更高</span>，认为这样能更快获得多模态能力，把多模态能力视为语言能力的一种附加品。

![各种视觉-文本联合训练策略的效果对比。在限制了图文总 Token 的前提下，减少视觉数据占比并采用早期融合的方式，能取得更出色的表现。](../../images/视觉多模态讲义（上）-image-200.png)

![基于固定的图文总 token 数，各种视觉与文本比例在视觉和语言任务中的学习曲线对比。从中可以看出，降低视觉数据的占比并采用早期融合的方式，能取得更出色的表现。](../../images/视觉多模态讲义（上）-image-202.png)

但作者的实验揭示了不同的结论。他们在固定总 token 预算下，改变视觉数据比例和引入时机进行消融实验。结果发现，<span style="color: rgb(100,37,208); background-color: inherit">视觉比例对最终多模态性能的影响其实很小</span>。并且<span style="color: rgb(100,37,208); background-color: inherit">在固定总预算下，更早地融合视觉数据并采用较低的视觉比例，效果反而更好</span>。这个结论促使他们采用了原生多模态预训练策略：<span style="color: rgb(100,37,208); background-color: inherit">不是在训练尾声集中进行高比例的视觉训练，而是在训练早期就引入中等比例的视觉数据</span>。这让模型能在长期的文本-视觉协同优化中，自然发展出更平衡的多模态表征。

* **<span style="color: rgb(36,91,219); background-color: inherit">无需视觉的监督微调</span>**

预训练后的视觉-语言模型并不会天然具备基于视觉的工具调用能力，这对多模态强化学习来说是个冷启动难题。<span style="color: rgb(216,57,49); background-color: inherit">传统方法是用人工标注或提示工程生成的思维链数据来引导，但这种数据多样性有限</span>，常局限于简单图表和基础工具操作，如<span style="color: rgb(220,155,4); background-color: inherit">裁剪、旋转、翻转</span>。

作者团队的一个发现是：高质量的纯文本 SFT 数据相对丰富且多样。因此，他们提出了一种新方&#x6CD5;**`zero-vision SFT`**，即完全只用文本 SFT 数据来激活模型的视觉和智能体能力。在这个过程中，所有的图像操作都通过 IPython 中的编程方式来代理执行，这实际上是传统视觉工具使用的一种泛化。这&#x79CD;**`zero-vision`**&#x6FC0;活方式催生了多样的推理行为，包括像素级的操作，如<span style="color: rgb(220,155,4); background-color: inherit">通过二值化估算物体大小、计数</span>，并泛化到了视觉定位、计数、OCR等任务。

> **<span style="color: rgb(222,120,2); background-color: inherit">核心结论</span>**：zero-vision SFT 足以激活视觉能力，同时确保跨模态的泛化性。这归功于文本与视觉数据的联合预训练。实验表明，与 zero-vision SFT 相比，<span style="color: rgb(216,57,49); background-color: inherit">使用包含视觉数据的 SFT 在视觉和智能体任务上表现差很多，可能是因为缺乏高质量视觉数据</span>。
>
> RL 训练曲线如右图，从 zero-vision SFT 的起点开始，模型在视觉基准上的性能持续提升，证明了 <span style="color: rgb(100,37,208); background-color: inherit">zero-vision 激活配合长周期 RL，就足以获得鲁棒的视觉能力</span>。

![只有极少纯文本 SFT 作为起点的情况下，Vision RL 在各大视觉测试集上的训练曲线。随着 Vision RL 算力的不断投入，模型表现持续走高。这表明仅靠纯文本唤醒视觉潜力，再配合长周期的强化学习训练，就足以让模型掌握扎实、过硬的视觉本领。](../../images/视觉多模态讲义（上）-image-198.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">联合多模态强化学习</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">基于结果的视觉 RL</span>**

在 zero-vision SFT 之后，模型需要进一步优化，才能可靠地将视觉输入融入推理。<span style="color: rgb(216,57,49); background-color: inherit">仅靠文本激活存在明显的失败模式：有时会忽略视觉输入，或在需要时未关注图像</span>。作者在明确需要视觉理解才能得出正确答案的任务上，采用基于结果的 RL。这些任务分三类：

> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉定位与计数</span>**：准确定位和清点图中物体
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">图表与文档理解</span>**：解读结构化视觉信息并提取文本
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉关键的 STEMM问题</span>**：筛选出的必须依赖视觉输入的数学和科学问题

在这些任务上进行基于结果的 RL，不仅提升了基础视觉能力，还催生了更复杂的智能体行为。将这些轨迹提取出来用于拒绝采样微调，形成了一个自我改进的数据闭环，让后续的联合 RL 阶段能利用更丰富的多模态推理轨迹。

**<span style="color: rgb(222,120,2); background-color: inherit">视觉 RL 提升文本性能</span>**

为研究视觉与文本性能间是否存在此消彼长的关系，作者团队评估了视觉 RL 前后模型在纯文本基准上的表现。结果非常好：基于结果的视觉 RL 在文本任务上带来了巨大提升，例如：

> * MMLU-Pro: $$84.7\% \rightarrow 86.4\%$$
>
> * GPQA-Diamond: $$84.3\% \rightarrow 86.4\%$$
>
> * LongBench v2: $$56.7\% \rightarrow 58.9\%$$

这是因为，视觉 RL 增强了模型在结构化信息提取方面的校准能力，降低了在处理与视觉推理（如计数、OCR）相似的查询时的不确定性。这表明<span style="color: rgb(100,37,208); background-color: inherit">视觉 RL 能促进跨模态泛化，在提升文本推理能力的同时，并未观察到语言能力的退化</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">联合多模态 RL</span>**

由上可知，从 zero-vision SFT 配合视觉 RL 可以涌现鲁棒的视觉能力，并且视觉 RL 还能反过来增强通用文本能力，因此作者团队<span style="color: rgb(100,37,208); background-color: inherit">在 Kimi K2.5 的后训练中采用了一种联合多模态 RL 范式</span>。他们并没有按输入模态（<span style="color: rgb(220,155,4); background-color: inherit">文本</span>、<span style="color: rgb(220,155,4); background-color: inherit">图像</span>）来划分不同的专家模型，而是按能力领域（<span style="color: rgb(220,155,4); background-color: inherit">知识、推理、编程、智能体</span>等）来组织 RL。这些领域专家模型同时从纯文本和多模态 Query 中学习，而生成式奖励模型 GRM 也同样在无模态障碍的异构轨迹上进行优化。这个范式确保了通过文本或视觉输入获得的任何能力提升，都能内在的泛化到另一模态的相关能力上，从而最大化跨模态能力的迁移。

> **<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**：这套方法不仅包含基于结果导向的视觉强化学习，还引出了能够反哺并提升文本能力的涌现式跨模态迁移现象。

* **<span style="color: rgb(36,91,219); background-color: inherit">智能体系统</span>**

<span style="color: rgb(216,57,49); background-color: inherit">现有智能体系统面临的主要挑战在于，它们依赖顺序执行推理和工具调用的步骤。虽然这种结构对简单的短期任务可能有效，但随着任务复杂性的增加和累积上下文的增长，它就变得力不从心</span>。当任务演变为需要广泛的信息收集和复杂、多分支的推理时，顺序系统往往会遇到显著的瓶颈。<span style="color: rgb(216,57,49); background-color: inherit">单个智能体逐步处理每个步骤的能力有限，可能导致实际推理深度和工具调用预算的耗尽，最终阻碍系统处理更复杂的场景</span>。

![在并行智能体强化学习环境中，随着训练推进，训练准确率平稳提升，同时训练期间的并行化水平也逐渐增长。](../../images/视觉多模态讲义（上）-image-197.png)

为了解决这个问题，K2.5 引入了智能体集&#x7FA4;**`Agent Swarm`**&#x548C;并行智能体强化学&#x4E60;**&#x20;<span style="color: rgb(216,57,49); background-color: inherit">PARL</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">arallel </span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">gent </span>**<span style="color: rgb(216,57,49); background-color: inherit">R</span>**<span style="color: rgb(216,57,49); background-color: inherit">einforcement </span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">earning）</span>。K2.5 并不是将任务作为一条推理链来执行，也不是依赖预先指定的并行化启发式规则，而是<span style="color: rgb(100,37,208); background-color: inherit">通过动态任务分解、子智能体实例化和并行子任务调度来启动一个智能体集群。这里的并行性并未被预设为固有优势；关于是否、何时以及如何并行化的决策，是通过环境反馈和强化学习驱动的探索显式学习到的</span>。如上图，性能的提升证明了这种自适应能力，随着训练过程中编排器不断优化其并行化策略，累积奖励平稳增长。

1. **<span style="color: rgb(36,91,219); background-color: inherit">架构设计</span>**

![一个智能体集群，其中可训练的编排器动态创建专门的冻结子智能体，并将复杂任务分解为可并行的子任务，以实现高效的分布式执行。](../../images/视觉多模态讲义（上）-image-204.png)

<span style="color: rgb(100,37,208); background-color: inherit">PARL 采用了一种解耦架构，包含一个可训练的编排器和多个从固定的中间策略检查点实例化而来的冻结子智能体</span>。这种设计有意避免了端到端的协同优化，从而<span style="color: rgb(100,37,208); background-color: inherit">规避了两个难题：功劳分配模糊和训练不稳定性</span>。在这种多智能体设定中，基于结果的奖励本质上是稀疏且有噪声的；一个正确的最终答案并不能保证子智能体的执行完美无缺，而一次失败也并不意味着所有子智能体都有错。<span style="color: rgb(100,37,208); background-color: inherit">通过冻结子智能体并将其输出视为环境观察而非可微分的决策点，将高层的协调逻辑与底层的执行能力分离开，从而实现更稳健的收敛</span>。为提高效率，K2.5 先使用小规模的子智能体来训练编排器，然后再过渡到更大的模型。此外，这个强化学习框架还支持动态调整子智能体与编排器之间的推理实例比例，从而最大化整个集群的资源利用率。

* **<span style="color: rgb(36,91,219); background-color: inherit">PARL 奖励函数</span>**

由于独立子智能体执行过程中固有的延迟、稀疏和非平稳反馈，训练一个可靠的并行编排器具有挑战性。为了解决这个问题，PARL 奖励函数为：

$$r_{\text{PARL}}(x,y)=\lambda_{1} \cdot r_{\text{parallel}} + \lambda_{2} \cdot r_{\text{finish}} + r_{\text{perf}}(x,y)$$

其中，性能奖励$$r_{\text{perf}}$$评估了给定任务$$x$$的解决方案$$y$$的整体成功度和质量。这个奖励由两个辅助奖励来增强，每个都旨在解决学习并行编排时的一个特定难题：

> * **$$r_{\text{parallel}}$$**&#x662F;为了<span style="color: rgb(100,37,208); background-color: inherit">减轻串行坍缩</span>，一种编排器退化到默认单智能体执行的局部最优解。通过激励子智能体的实例化，鼓励对并发调度空间的探索。
>
> * **$$r_{\text{finish}}$$**&#x4E13;注于成功完成所分配的子任务。它被用来<span style="color: rgb(100,37,208); background-color: inherit">防止伪并行</span>，这是一种奖励黑客行为，即编排器通过生成大量子智能体而不进行有意义的任务分解，来大幅提高并行性指标。通过奖励已完成的子任务，$$r_{\text{finish}}$$ 确保了可行性，并引导策略走向有效且合理的分解。

为确保最终策略优化的是主要目标，超参数$$\lambda_{1}$$和$$\lambda_{2}$$会在训练过程中逐步退火至零。

* **<span style="color: rgb(36,91,219); background-color: inherit">关键步骤</span>**

为了衡量并行智能体设置下的计算时间成本，类比计算图中的<span style="color: rgb(216,57,49); background-color: inherit">关键路径</span>**`Critical Path`**&#x6765;定义<span style="color: rgb(216,57,49); background-color: inherit">关键步骤</span>**`Critical Steps`**。这里将一个回合建模为一系列执行阶段，索引为$$t=1,\ldots,T$$。<span style="color: rgb(100,37,208); background-color: inherit">在每个阶段，主智能体执行一个动作，这对应着直接的工具调用，或是并行运行的一组子智能体的实例化</span>。设$$S^{(t)}_{\text{main}}$$表示主智能体在阶段$$t$$采取的步骤数，通常$$S^{(t)}_{\text{main}}=1$$，$$S^{(t)}_{\text{sub},i}$$表示该并行组中第$$i$$个子智能体采取的步骤数。<span style="color: rgb(100,37,208); background-color: inherit">阶段</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">的持续时间由该组中运行时间最长的子智能体决定</span>。因此，一个回合的总关键步骤定义为：

$$\text{CriticalSteps}=\sum_{t=1}^{T}\left(S^{(t)}_{\text{main}}+\max_i S^{(t)}_{\text{sub},i}\right)$$

通过使用关键步骤而非总步骤来约束训练和评估，这个框架<span style="color: rgb(46,161,33); background-color: inherit">显式地激励了有效的并行化</span>。不能缩短并行组最大执行时间的过度子任务创建，在此度量下几乎没什么好处，而<span style="color: rgb(46,161,33); background-color: inherit">能缩短最长并行分支的良好均衡的任务分解，则直接减少了关键步骤</span>。如此一来，就鼓励了编排器以最小化端到端延迟的方式来分配工作给子智能体，而非仅仅最大化并发数或总工作量。

* **<span style="color: rgb(36,91,219); background-color: inherit">Prompt 构建</span>**

为了激励编排器利用并行化的优势，K2.5 构建了一套合成 Prompt，旨在压迫顺序智能体执行的极限。这些 Prompt 强调<span style="color: rgb(216,57,49); background-color: inherit">广度搜索</span>**`wide search`**&#x6216;<span style="color: rgb(216,57,49); background-color: inherit">深度搜索</span>**`deep search`**，<span style="color: rgb(100,37,208); background-color: inherit">前者需要同时探索许多独立信息源，后者则需要多个推理分支并进行延迟聚合</span>。此外还额外加入了受现实世界工作负载启发的任务，如<span style="color: rgb(220,155,4); background-color: inherit">长文档分析和大规模文件下载</span>。当顺序执行时，这些任务很难在固定的推理步骤和工具调用预算内完成。通过设计，它们鼓励编排器并行分配子任务，从而以比单个顺序智能体更少的关键步骤来完成。这些 Prompt 并没有明确指示模型去并行化。相反，它们塑造了任务分布，使得并行分解和调度策略自然受到青睐。

* **<span style="color: rgb(36,91,219); background-color: inherit">方法概述</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">基座</span>**

Kimi K2.5 基于 Kimi K2，一个万亿参数的 MoE Transformer 模型，&#x5728;**`15T`**&#x9AD8;质量文本 Token 上预训练而成。<span style="color: rgb(100,37,208); background-color: inherit">Kimi K2 采用了 Token 高效的 MuonClip 优化器，并结合 QK-Clip 来保证训练稳定性</span>。该模型总参数量&#x4E3A;**`1.04T`**，激活参数为 **`32B`**，使用了 **`384`** 个专家，每个 Token 激活其&#x4E2D;**`8`**&#x4E2A;，即稀疏度&#x4E3A;**`48`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

Kimi K2.5 的多模态架构由三个组件构成：<span style="color: rgb(100,37,208); background-color: inherit">一个三维原生分辨率视觉编码器</span>**`MoonViT-3D`**<span style="color: rgb(100,37,208); background-color: inherit">、一个 MLP 投影器，以及 Kimi K2 MoE 语言模型</span>，按照 Kimi-VL 中确立的设计原则。

> **<span style="color: rgb(222,120,2); background-color: inherit">MoonViT-3D：图像和视频的统一嵌入空间</span>**
>
> 在 Kimi-VL 中，使用 MoonViT 以原始分辨率原生处理图像，消除了复杂的子图分割和拼接操作。MoonViT &#x7531;**`SigLIP-SO-400M`**&#x521D;始化，并采用&#x4E86;**`NaViT`**&#x7684; patch 打包策略，将单张图像分割成 patch，展平后按顺序连接成一维序列，从而能高效地同时在各种分辨率的图像上进行训练。

为了将图像理解能力最大限度地迁移到视频上，K2.5 引入了 MoonViT-3D，它具有统一的架构、完全共享的参数和一致的嵌入空间。通过将 patch 打包的理念推广到时序维度，将最多四个连续帧视为一个时空体：<span style="color: rgb(100,37,208); background-color: inherit">这些帧的二维 patch 被联合展平并打包成一个单一的一维序列，使得相同的注意力机制可以在空间和时间上无缝运作</span>。额外的时序注意力增强了对高速运动和视觉效果的理解，而参数共享则最大化了从静态图像到动态视频的知识泛化，<span style="color: rgb(46,161,33); background-color: inherit">无需专门的视频模块或架构分叉就实现了强大的视频理解性能</span>。在进入 MLP 投影器之前，轻量级的时序池化会聚合每个时序块内的 patch ，实现$$4\times$$的时序压缩，显著扩展了可处理的视频长度。其结果是一个统一的流程，图像预训练中获得的知识和能力，通过一个共享的参数空间和特征表示，整体迁移到视频领域。

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

![](../../images/视觉多模态讲义（上）-image-199.png)

如上表，Kimi K2.5 的预训练建立在 Kimi K2 语言模型 checkpoint 之上，分三个阶段处理&#x7EA6;**`15T`** Token：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">独立 ViT 训练</span>**：建立一个鲁棒的原生分辨率视觉编码器
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">联合预训练</span>**：同时增强语言和多模态能力
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">中期训练</span>**：使用高质量数据和长上下文激活，以优化能力并扩展上下文窗口

**<span style="color: rgb(222,120,2); background-color: inherit">ViT 训练阶段</span>**

MoonViT-3D &#x4ECE;**`SigLIP`**&#x5F00;始，在图像-文本和视频-文本对上进行持续预训练，<span style="color: rgb(100,37,208); background-color: inherit">文本部分由多种目标组成：图像替代文本、图像和视频的合成描述、边界框定位及 OCR 文本</span>。与 Kimi-VL 的实现不同，这次持续预训练不包含对比损失，仅采用交叉熵损失$$L_{caption}$$，用于根据输入图像和视频生成描述。这里采用两阶段对齐策略：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：更新 MoonViT-3D，通过描述损失使其&#x4E0E;**`Moonlight-16B-A3B`**&#x5BF9;齐，消耗&#x7EA6;**`1T`** Token 和极少的训练 FLOPs。此阶段<span style="color: rgb(100,37,208); background-color: inherit">让 MoonViT-3D 初步理解高分辨率图像和视频</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：仅更新 MLP 投影器，在 ViT &#x548C;**`1T`**&#x53C2;数的 LLM 之间架桥，以实现更平滑的联合预训练。

**<span style="color: rgb(222,120,2); background-color: inherit">联合训练阶段</span>**

联合预训练阶段<span style="color: rgb(100,37,208); background-color: inherit">从一个接近训练结束的 Kimi K2 检查点开始，在</span>**`4K`**<span style="color: rgb(100,37,208); background-color: inherit">序列长度下，额外处理</span>**`15T`**<span style="color: rgb(100,37,208); background-color: inherit">视觉-文本 Token</span>。数据方面扩展了 Kimi K2 的预训练分布，引入了独特的 Token，调整了数据比例并增加了编程相关内容的权重，同时控制了每个数据源的最大训练轮数。

第三阶段进行长上下文激活，并整合了更高质量的中期训练数据，通&#x8FC7;**`YaRN`**&#x63D2;值法逐步扩展上下文长度。这在长文本理解和长视频理解方面都带来了显著的泛化提升。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">监督微调</span>**

遵循 Kimi K2 的 SFT 流程，通过从 K2、K2 Thinking 以及一系列自研内部专家模型中合成高质量的候选回答，来开发 K2.5。<span style="color: rgb(100,37,208); background-color: inherit">数据生成策略采用特定领域定制的专门流程，将人工标注与高级提示工程和多阶段验证相结合</span>。这产出了一个大规模指令微调数据集，包含多样的 Prompt 和复杂的推理轨迹，最终训练模型在复杂的现实世界应用中优先进行交互式推理和精确的工具调用。

**<span style="color: rgb(222,120,2); background-color: inherit">强化学习</span>**

为了促进文本与视觉模态的联合优化，并实现用于智能体集群的 PARL，K2.5 开发了一个统一智能体强化学习环境，并对 RL 算法进行了优化。文本-视觉联合 RL 和 PARL 都建立在下面所述算法的基础上。

**策略优化**：对于从数据集$$\mathcal{D}$$中采样的每个问题$$x$$，使用旧策略$$\pi_{\text{old}}$$生成$$K$$个回答$$\{y_1,\ldots,y_K\}$$。根据以下目标优化模型$$\pi_{\bm{\theta}}$$：

$$\begin{split}
L_{\text{RL}}(\bm{\theta}) = \mathbb{E}_{x\sim\mathcal{D}} \left[ \frac{1}{N}\sum_{j=1}^{K}\sum_{i=1}^{|y_j|} 
\text{Clip}\left( \frac{\pi_{\bm{\theta}}(y_j^i|x, y_j^{0:i})}{\pi_{\text{old}}(y_j^i|x, y_j^{0:i})}, \alpha, \beta \right) (r(x, y_j) - \bar{r}(x)) 
\tau \left( \log \frac{\pi_{\bm{\theta}}(y_j^i|x, y_j^{0:i})}{\pi_{\text{old}}(y_j^i|x, y_j^{0:i})} \right)^2 \right]
\end{split}$$

这里$$\alpha, \beta, \tau > 0$$是超参数，$$y_j^{0:i}$$是第$$j$$个回答直到第$$i$$个 Token 的前缀，$$N = \sum_{i=1}^{K}|y_i|$$是一个批次中生成的 Token 总数，$$\bar{r}(x) = \frac{1}{K}\sum_{j=1}^{K} r(x, y_j)$$是所有生成回答的平均奖励。

这个损失函数与 K1.5 中使用的策略优化算法不同，<span style="color: rgb(100,37,208); background-color: inherit">它引入了一个 Token 级别的裁剪机制，旨在减轻因训练和推理框架间差异而被放大的离线策略分歧</span>。这相当于一个简单的梯度掩码方案：

> 对于对数概率比在区间$$[\alpha, \beta]$$内的 Token，正常计算策略梯度
>
> 在此区间外的 Token 的梯度则被置零

与标准 PPO 裁剪的一个关键区别是，<span style="color: rgb(100,37,208); background-color: inherit">这个方法严格基于对数比来显式约束离线策略漂移，而不管优势项的符号正负</span>。这种方法与近期为稳定大规模 RL 训练而提出的策略一致。在需要长周期、多步骤工具使用推理的复杂领域中，这种机制对于维持训练稳定性至关重要。K2.5 使&#x7528;**`MuonClip`**&#x4F18;化器来最小化这个目标。

**奖励函数**：对于有可验证解决方案的任务，如<span style="color: rgb(220,155,4); background-color: inherit">推理和智能体任务</span>，应用<span style="color: rgb(100,37,208); background-color: inherit">基于规则的结局奖励。为优化资源消耗，还加入了预算控制奖励，旨在提高 Token 效率</span>。对于通用任务，采用生成式奖励模型 GRM，它们能提供与 Kimi 内部价值标准对齐的细粒度评估。对于视觉任务，作者团队<span style="color: rgb(100,37,208); background-color: inherit">设计了特定任务的奖励函数以提供细粒度监督</span>：

> * **<span style="color: rgb(36,91,219); background-color: inherit">视觉定位和点位任务</span>**：采用基于 F1 的奖励和软匹配。定位任务通过交并比 IoU 获得软匹配分数，点任务则在最优匹配下通过高斯加权距离获得软匹配分数。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">多边形分割任务</span>**：将预测多边形光栅化为二值掩码，计算其与真实掩码的分割 IoU 来分配奖励。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">OCR 任务</span>**：采用归一化编辑距离来量化预测与真实文本间的字符级对齐度。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">计数任务</span>**：基于预测与真实值之间的绝对差分配奖励。

作者团队还合成了复杂的视觉谜题，并利用一个 Kimi K2 的 LLM 验证器提供反馈。

**生成式奖励模型**：Kimi K2 对开放式生成采用了自我批评的评分标准奖励 ，而 <span style="color: rgb(100,37,208); background-color: inherit">K2.5 扩展了这一工作，系统性地在广泛的智能体行为和多模态轨迹上部署了生成式奖励模型</span>**<span style="color: rgb(100,37,208); background-color: inherit"> </span>**<span style="color: rgb(100,37,208); background-color: inherit">GRM</span>。这里 GRM 的应用范围不仅限于对话输出，而是<span style="color: rgb(100,37,208); background-color: inherit">在已验证的奖励信号之上，将其应用于多样化的环境</span>，包括聊天助手、编程智能体、搜索智能体和生成人工制品的智能体。GRM 并不是充当二元的裁判，而是<span style="color: rgb(100,37,208); background-color: inherit">作为与 Kimi 价值观对齐的细粒度评估器，这些价值观对用户体验至关重要</span>，例如：<span style="color: rgb(220,155,4); background-color: inherit">有用性、回答就绪度、上下文相关性、适当的详细程度、生成人工制品的美学质量，以及严格的指令遵循</span>。这种设计使得奖励信号能够捕捉到微妙的偏好梯度，而这些梯度很难用纯规则或特定任务验证器来编码。为减轻奖励黑客行为和过拟合单一偏好信号，我们采用了多种针对不同任务上下文定制的备选 GRM 评分标准。

**Token 高效的强化学习**：Token 效率对于具备测试时扩展能力的 LLM 至关重要。虽然测试时扩展本质上是用计算换取推理质量，但实际的收益需要能主动平衡这种权衡的算法创新。之前研究表明，<span style="color: rgb(100,37,208); background-color: inherit">施加一个问题相关预算可以有效约束推理时的计算量，激励模型生成更简洁的思维链推理模式，避免不必要的 Token 膨胀</span>。但会<span style="color: rgb(216,57,49); background-color: inherit">存在长度过拟合现象：在严格预算约束下训练的模型，通常无法泛化到更高的计算规模</span>。结果，它们不能有效利用额外的推理时 Token 来解决复杂问题，反而退化到截断的推理模式。为此作者团队提出&#x4E86;**`Toggle`**，一种在推理时扩展和预算约束优化之间交替的训练启发式方法。对于学习迭代$$t$$，奖励函数定义为：

$$\tilde{r}(x, y) = 
\begin{cases}
r(x, y) \cdot \mathbb{I} \left\{ \frac{1}{K}\sum_{i=1}^{K} r(x, y_i) < \lambda \text{ or } |y_i| \leq \text{budget}(x) \right\}, & \text{if } \lfloor t/m \rfloor \pmod{2} = 0 \text{ (Phase0)} \\
r(x, y), & \text{if } \lfloor t/m \rfloor \pmod{2} = 1 \text{ (Phase1)}
\end{cases}$$

其中$$\lambda$$和$$m$$是算法超参数，$$K$$是每个问题的 rollout 次数。算法每$$m$$次迭代在两个优化阶段之间交替：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Phase0（预算限制阶段）</span>**：训练模型在任务相关的 Token 预算内解决问题。为了防止过早地为追求效率而牺牲质量，该约束是有条件应用的：仅当模型在某个问题上的平均准确率超过阈值$$\lambda$$时才执行。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Phase1（标准扩展阶段）</span>**：模型生成回答直到达到最大 Token 限制，鼓励模型利用更多计算来获得更好的推理时扩展效果。

问题相关预算由正确回答子集的 Token 长度的第$$\rho$$百分位数估算得出：

$$\text{budget}(x) = \text{Percentile}\left( \{|y_j| \mid r(x, y_i) = 1, i = 1, \ldots, K\}, \rho \right)$$

这个预算在训练开始时估算一次，之后保持不变。<span style="color: rgb(100,37,208); background-color: inherit">Toggle 作为一个双目标问题的随机交替优化算法，其专门设计就是为了调和推理能力与计算效率</span>。

![](../../images/视觉多模态讲义（上）-image-201.png)

作者团队在 K2 Thinking 上评估了 Toggle 的有效性。如上图，有几个结论：

> 1. 几乎所有基准上的输出长度都持续减少，平均而言，Toggle 减少&#x4E86;**`25~30%`**&#x7684;输出 Token，同时对性能的影响微乎其微
>
> 2. 思维链中的冗余模式，如重复验证和机械计算，大幅减少
>
> 3. Toggle 表现出很强的领域泛化能力，例如，<span style="color: rgb(220,155,4); background-color: inherit">仅在数学和编程任务上训练，模型在 GPQA 和 MMLU-Pro 上同样实现了稳定的 Token 减少，且性能仅有边际下降</span>

### 2.9.2 <span style="color: rgb(36,91,219); background-color: inherit">Kimi K3</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">技术总览</span>**

<span style="color: rgb(100,37,208); background-color: inherit">Kimi K3 是一个面向百万上下文与长程 Agent 训练的原生多模态稀疏模型系统。</span>主干共有 **`93`** 层，包含 **`2.78T`** 总参数和 **`104.2B`** 激活参数；训练上下文扩展到 **`1M`** tokens。它不是把 K2 单纯放大，而是同时重构 sequence mixing、depth mixing、sparse channel mixing、multimodal input、post-training 与 distributed runtime。

<span style="color: rgb(46,161,33); background-color: inherit">整套技术路线围绕同一个目标展开：让超长轨迹在训练时可并行、可恢复，在推理时可缓存、可调度。</span>KDA 与 Gated MLA 处理 token 维信息流，AttnRes 处理 layer 维信息流，Stable LatentMoE 处理 channel 维容量；Native Vision、SFT/RL/MOPD、MXFP4/MXFP8 QAT 和基础设施共同把这些结构变成可训练、可部署的系统。

| **<span style="color: rgb(36,91,219); background-color: inherit">技术维度</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">核心实现</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">解决的问题</span>** |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| Token mixing                                                                   | **`69 KDA + 24 Gated MLA`**                                                    | KDA 用固定 recurrent state 承担高频长序列计算，周期性 MLA 保留全局内容交互。                             |
| Depth mixing                                                                   | **`8×12`** Block AttnRes，含 embedding 共 **`9`** 个 source                        | 把深度残差从顺序累加改成对历史 block 表示的选择性检索。                                                 |
| Channel mixing                                                                 | **`896`** routed experts，激活 **`16`**，另有 **`2`** 个 shared experts               | 以 3584 维 latent expert path 扩大容量，并用 RMSNorm、SiTU-GLU、QB 控制稳定性与负载。               |
| Native Vision                                                                  | **`401M`** MoonViT-V2，27 层，patch 14                                            | 图像、视频与文本从训练起点共享 next-token prediction Backbone，避免后接视觉 Encoder 的优化尖峰。            |
| Post-Training                                                                  | SFT → 9 个 RL experts → MOPD，贯穿 MXFP4/MXFP8 QAT                                 | 把 domain 与 reasoning effort 专家合并为单一 policy，并让训练数值路径与部署量化路径一致。                   |
| Infrastructure                                                                 | KCP、MoonEP、Pipeline ZeRO-2、AgentENV、KDA-aware prefix cache                     | 支撑 3T 级稀疏多模态训练、1M Agentic RL 和百万上下文在线服务。                                        |

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">整体结构</span>**

<span style="color: rgb(100,37,208); background-color: inherit">Kimi K3 的架构目标是同时扩展 sequence length、network depth 和 model width 三条信息流。</span>模型共有 **`93`** 层、**`2.78T`** 总参数和 **`104.2B`** 激活参数。Hidden Dimension 保持 **`7168`**，但 attention heads 从 K2 的 **`64`** 增加到 **`96`**，模型宽度主要通过 Stable LatentMoE 的 expert pool 扩展，而不是继续放大主干 hidden size。

每个主干 block 由 **`3`** 层 KDA 和 **`1`** 层 Gated MLA 组成，每个 attention layer 后接一层 Stable LatentMoE。整网包含 **`69`** 层 KDA 与 **`24`** 层 MLA，主干末尾再放置 Gated MLA，保证最终层执行 global attention。AttnRes 在深度方向选择 embedding、当前 block partial sum 和历史 block 表示；MoonViT-V2 则把图像与视频特征投到共享 Embedding 空间。

<span style="color: rgb(46,161,33); background-color: inherit">这套设计把长序列 mixing、跨层检索、稀疏 channel mixing 和视觉输入放进同一个 next-token prediction Backbone，整体 scaling efficiency 相比 K2 提升约 </span>**`2.5×`**<span style="color: rgb(46,161,33); background-color: inherit">。</span>这张总览图把 token、channel 与 layer 三条信息流放在同一张图里。

![K3 总体架构：Hybrid Attention、Stable LatentMoE、AttnRes 与 MoonViT-V2 的连接关系。](../../images/视觉多模态讲义（上）-figure-02-architecture.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">Hybrid Attention</span>**

<span style="color: rgb(100,37,208); background-color: inherit">Hybrid Attention 用 KDA 负责高频、低成本的长序列 mixing，用周期性的 Gated MLA 提供不受局部递推限制的全局内容交互。</span>KDA 的单头 recurrent state 为固定大小矩阵，状态更新遵循 delta rule，并在写入前对 key channel 施加逐通道 retention：

$$\mathbf{S}_t=(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^{\top})\operatorname{Diag}(\boldsymbol{\alpha}_t)\mathbf{S}_{t-1}+\beta_t\mathbf{k}_t\mathbf{v}_t^{\top},\quad \widetilde{\mathbf{o}}_t=\mathbf{S}_t^{\top}\mathbf{q}_t$$

其中 $$\boldsymbol{\alpha}_t$$ 是逐 key channel 的 retention factor，$$\beta_t$$ 控制写入强度。query、key 与 value 先经过 ShortConv 和 Swish，query 与 key 再做 L2Norm；decay logit 由共享 low-rank projection 加 head-specific bias 生成。训练和 prefill 采用 chunkwise 形式：chunk 内并行，chunk 间只传递固定大小 state。输出由 inter-chunk state 读取与 intra-chunk causal interaction 两部分组成。

<span style="color: rgb(100,37,208); background-color: inherit">KDA 的各 head 先显式生成 query、key、value、write strength 与逐通道 decay logit。</span>$$\mathbf{W}_{\alpha}^{\downarrow}$$ 和 $$\mathbf{W}_{\alpha}^{\uparrow}$$ 在 head 间共享低秩通路，$$\mathbf{b}_{\alpha}^{h}$$ 保留 head-specific 偏置；$$A_h$$ 是可学习的 per-head log-scale，初始化为 **`0`**：

$$\begin{aligned}\mathbf{q}_t^h,\mathbf{k}_t^h &=\operatorname{L_2Norm}(\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}_{q/k}^h\mathbf{x}_t))),\\ \mathbf{v}_t^h &=\operatorname{Swish}(\operatorname{ShortConv}(\mathbf{W}_v^h\mathbf{x}_t)),\\ \beta_t^h &=\operatorname{Sigmoid}(\mathbf{W}_{\beta}^h\mathbf{x}_t),\qquad \mathbf{z}_t^h=\mathbf{W}_{\alpha}^{\uparrow}\mathbf{W}_{\alpha}^{\downarrow}\mathbf{x}_t+\mathbf{b}_{\alpha}^h.\end{aligned}$$

设 chunk size 为 C，$$\boldsymbol{\Gamma}_{[t]}^{1\rightarrow C}$$ 按行堆叠累计 decay，UT transform 产生 $$\mathbf{U}_{[t]}$$ 与 $$\mathbf{W}_{[t]}$$，pseudo-value 为 $$\widetilde{\mathbf{V}}_{[t]}=\mathbf{U}_{[t]}-\mathbf{W}_{[t]}\mathbf{S}_{[t]}$$。chunk 内所有位置可以一次并行计算：

$$\begin{aligned}\boldsymbol{\gamma}_{[t]}^{i\rightarrow j}&=\prod_{r=i}^{j}\boldsymbol{\alpha}_{[t]}^r,\\ \mathbf{A}_{[t]}&=\operatorname{Tril}[(\mathbf{Q}_{[t]}\odot\boldsymbol{\Gamma}_{[t]})(\mathbf{K}_{[t]}/\boldsymbol{\Gamma}_{[t]})^{\top}],\\ \mathbf{O}_{[t]}&=(\boldsymbol{\Gamma}_{[t]}\odot\mathbf{Q}_{[t]})\mathbf{S}_{[t]}+\mathbf{A}_{[t]}\widetilde{\mathbf{V}}_{[t]}.\end{aligned}$$

$$\operatorname{Tril}$$ 保留对角线，因为当前位置读取的是完成当前 token update 后的 state。KDA 与 MLA 最后都使用 input-dependent full-rank channel gate；区别是 KDA 在 gate 前先对 recurrent output 做 head-wise RMSNorm，MLA 直接门控 global-attention output：

$$\mathbf{y}_t^{\mathrm{KDA}}=\mathbf{W}_o[\operatorname{Sigmoid}(\mathbf{W}_g\mathbf{x}_t)\odot\operatorname{RMSNorm}(\widetilde{\mathbf{o}}_t)],\qquad \mathbf{y}_t^{\mathrm{MLA}}=\mathbf{W}_o[\operatorname{Sigmoid}(\mathbf{W}_g\mathbf{x}_t)\odot\widetilde{\mathbf{o}}_t]$$

<span style="color: rgb(216,57,49); background-color: inherit">普通负 Softplus decay 会让 chunk 内累计 retention 的倒数无限增大，对角 tile 因而需要显式 position-pair 计算，既慢又容易在低精度下溢出或溢出。</span>K3 把 log-decay 改成有下界的 scaled sigmoid：

$$\mathbf{g}_t^h=g_{\min}\operatorname{Sigmoid}(e^{A_h}\mathbf{z}_t^h),\quad \boldsymbol{\alpha}_t^h=\exp(\mathbf{g}_t^h),\quad g_{\min}=-5$$

每一步 retention 因而大于 **`e^-5≈6.7×10^-3`**。在 **`16`**-token tile 内，累计 log-decay 被限制在 **`(-80,0)`**，倒数小于 **`e^80`**，仍处于 BF16 动态范围。<span style="color: rgb(46,161,33); background-color: inherit">对角与非对角 causal tile 都可以直接走 Tensor Core dense GEMM，原来的 position-pair 对角路径被移除。</span>下界 decay 对数值范围和 tile kernel 的影响如下。

![下界 decay 把对角 tile 从显式 position-pair 计算改成可直接使用 Tensor Core 的 dense GEMM。](../../images/视觉多模态讲义（上）-figure-03-kda-lower-bound.png)

KDA 输出先做 head-wise RMSNorm，再使用 input-dependent full-rank gate；Gated MLA 也使用同构的 channel-wise full-rank output gate。MLA 只缓存低维 latent KV，并在计算时上投影恢复各 head 的内容 key/value。所有 MLA layer 使用 NoPE，位置敏感性由中间 KDA 的递推 gate 与 decay 提供，因此扩展 context 不需要重设 RoPE base 或引入 YaRN。训练时 attention output 保持 FP32，以修正 flash attention 的偏置舍入误差；kernel 把较大的 output tile 与 KV staging buffer 重叠，释放 shared memory 给更深的 KV pipeline。

3. **<span style="color: rgb(36,91,219); background-color: inherit">Attention Residuals</span>**

<span style="color: rgb(100,37,208); background-color: inherit">AttnRes 把“沿深度累加残差”改成“沿深度做内容选择”。</span>Full AttnRes 为每一层学习 pseudo-query，用 embedding 与所有前序层输出作为 key/value，RMSNorm 防止大模长层输出凭幅值垄断权重：

$$\alpha_{i\rightarrow l}=\frac{\exp(\mathbf{w}_l^{\top}\operatorname{RMSNorm}(\mathbf{k}_i))}{\sum_{j=0}^{l-1}\exp(\mathbf{w}_l^{\top}\operatorname{RMSNorm}(\mathbf{k}_j))},\quad \mathbf{h}_l=\sum_{i=0}^{l-1}\alpha_{i\rightarrow l}\mathbf{v}_i$$

Full 形式的算术量为 $$O(L^2d)$$，在深度小于 **`100`** 时可以接受，真正昂贵的是保存全部 layer output 带来的 $$O(Ld)$$ 显存和跨 PP stage 通信。Block AttnRes 把层划为 N 个 block，block 内输出先做 partial sum，block 间只对 N 个表示执行 full attention，显存和通信降到 $$O(Nd)$$。

K3 采用 **`8`** 个完整 block、每个 **`12`** 层，末尾保留 partial block；把 embedding 也作为独立源后，一共形成 **`9`** 个可检索 block 表示。推理时 inter-block 结果可与 block 内 sequential partial sum 用 online softmax 合并，避免把跨层检索变成明显的 decode 延迟。

Full AttnRes 的 source 定义也包含 token embedding：layer-specific pseudo-query 为 $$\mathbf{q}_l=\mathbf{w}_l$$，第 **`0`** 个 key/value 是 $$\mathbf{h}_1$$，其余 key/value 是各前序层的模块输出：

$$\mathbf{k}_i=\mathbf{v}_i=\begin{cases}\mathbf{h}_1,&i=0,\\ f_i(\mathbf{h}_i),&1\le i\le l-1.\end{cases}$$

Block AttnRes 在 block 内先求 $$\mathbf{b}_n=\sum_{j\in\mathcal{B}_n}f_j(\mathbf{h}_j)$$。<span style="color: rgb(100,37,208); background-color: inherit">block 首层只能访问 embedding 与已完成的历史 block，后续层再加入当前 block 的 partial sum；最终输出层聚合全部 N 个 block：</span>

$$\mathbf{V}=\begin{cases}[\mathbf{b}_0,\mathbf{b}_1,\ldots,\mathbf{b}_{n-1}]^{\top},&i=1,\\ [\mathbf{b}_0,\mathbf{b}_1,\ldots,\mathbf{b}_{n-1},\mathbf{b}_n^{i-1}]^{\top},&i\ge2.\end{cases}$$

4. **<span style="color: rgb(36,91,219); background-color: inherit">Stable LatentMoE</span>**

Stable LatentMoE 把 full model width 与 routed-expert width 解耦。共享 expert 在 **`7168`** 维主空间处理通用变换，routed expert 在 **`3584`** 维 latent space 工作；每层有 **`2`** 个共享 expert、**`896`** 个 routed expert，每个 token 激活 **`16`** 个 routed expert，对应 sparsity **`56`**。

$$\mathbf{u}=\sum_{i\in\mathcal{T}_k(\mathbf{x})}p_iE_i^{\mathrm{routed}}(\mathbf{W}^{\downarrow}\mathbf{x}),\quad \mathbf{y}=\sum_{j=1}^{2}E_j^{\mathrm{shared}}(\mathbf{x})+\mathbf{W}^{\uparrow}\operatorname{RMSNorm}(\mathbf{u})$$

<span style="color: rgb(216,57,49); background-color: inherit">极端稀疏会放大两类故障：routed path 连续经过近四次矩阵乘，内部 activation 容易爆炸；近千个 expert 的 load balancing 也会让固定步长 bias update 出现适应过慢或振荡。</span>Normalized LatentMoE 在 expert aggregate 与 up-projection 之间插入 RMSNorm，先消除不同 expert 组合造成的 scale 漂移。

SiTU-GLU 对 gate branch 与 up branch 分别施加 smooth cap。K3 设置 **`β1=4`**、**`β2=25`**，近原点保持接近 SwiGLU 的线性响应，大输入区间则把乘积上界控制在 **`100`**：

$$\operatorname{SiTU\text{-}GLU}(\mathbf{x})=\left[\beta_1\tanh\left(\frac{\mathbf{W}_g\mathbf{x}}{\beta_1}\right)\odot\operatorname{Sigmoid}(\mathbf{W}_g\mathbf{x})\right]\odot\left[\beta_2\tanh\left(\frac{\mathbf{W}_u\mathbf{x}}{\beta_2}\right)\right]$$

GLU、SwiGLU 与 SiTU-GLU 的分支定义及标量响应如下。

![GLU、SwiGLU 与 SiTU-GLU 的分支定义及标量响应，SiTU-GLU 在大输入区间保持有界。](../../images/视觉多模态讲义（上）-figure-04-situ-glu.png)

<span style="color: rgb(100,37,208); background-color: inherit">Quantile Balancing 不把 balance bias 写入 mixture weight，只用它改变 Top-k dispatch。</span>对 batch 中 m 个 token、n 个 routed expert、每 token 选择 k 个 expert，目标 load 为 $$q=mk/n$$。先对 biased score 取 Top-(k+1)，第 k+1 个值形成 token cutoff；再从每个 expert 的 score margin 中取 $$1-k/n$$ 分位数，直接求下一步 bias：

auxiliary-loss-free router 先计算 $$\mathbf{s}_i=\operatorname{Sigmoid}(\mathbf{W}_r\mathbf{x}_i)$$。<span style="color: rgb(46,161,33); background-color: inherit">balance bias 只参与集合选择，真正的 mixture weight 仍由原始 score 归一化，因此 load control 不会直接篡改 expert output 的加权比例：</span>

$$\mathcal{T}_i=\operatorname{argtop}_k(\mathbf{s}_i+\mathbf{b}),\qquad p_{i,j}=\frac{s_{i,j}}{\sum_{r\in\mathcal{T}_i}s_{i,r}},\quad j\in\mathcal{T}_i$$

固定步长方案使用 $$b_j^{(t+1)}=b_j^{(t)}+\gamma\operatorname{sign}(\bar{\ell}-\ell_j^{(t)})$$：$$\gamma$$ 小时适应慢，大时会围绕目标 load 振荡。QB 把这个手工步长换成目标 quantile。图中的例子取 **`m=8、n=4、k=1`**，初始 load **`(4,3,1,0)`** 经列级 bias 调整后变为 **`(2,2,2,2)`**。

$$\widehat{b}_j^{(t+1)}=-\operatorname{quantile}_{1-k/n}(\mathbf{s}_{:,j}-\boldsymbol{\alpha}^{(t)}),\quad \mathbf{b}^{(t+1)}=\widehat{\mathbf{b}}^{(t+1)}-\operatorname{mean}(\widehat{\mathbf{b}}^{(t+1)})\mathbf{1}$$

bias 只在下一 training step 生效，避免当前 batch 使用由自身统计得到的 bias。真实训练不能 gather 数百万 margin，系统为每个 expert 建 histogram，通过一次 all-reduce 汇总 bin count，再从全局 histogram 估计 quantile；通信量只有数百 bins/expert，inference 时冻结最终 bias。Quantile Balancing 从失衡 dispatch 恢复目标 expert load 的过程如下。

![Quantile Balancing 从不均衡 Top-k routing 推导列级 bias 调整，并恢复目标 expert load。](../../images/视觉多模态讲义（上）-figure-05-quantile-balancing.png)

5. **<span style="color: rgb(36,91,219); background-color: inherit">Native Vision</span>**

K3 从训练起点就把文本、图像和视频放入同一 Backbone 和同一 next-token prediction 目标，没有先训练纯语言模型、再做 modality alignment 的阶段。MoonViT-V2 也不再从 SigLIP 初始化，而是用语言建模目标从头训练。<span style="color: rgb(46,161,33); background-color: inherit">这样既避免预训练视觉 Encoder 接入超大语言 Backbone 后的梯度尖峰，也让视觉表示直接适配 OCR、布局、代码渲染和细粒度结构。</span>梯度范数对比如下。

![MoonViT-V2 从头训练时的梯度范数更低、尖峰更少。](../../images/视觉多模态讲义（上）-figure-06-vision-gradient.png)

MoonViT-V2 有 **`27`** 层、约 **`401M`** 参数、patch size **`14`** 和 **`12`** 个 attention heads，使用 RMSNorm 并移除 linear 与 attention projection 的 bias。图像与视频共享参数；attention 分为空间 intra-frame 与时间 inter-frame 两次 pass，并用 temporal pooling 压缩视频 token。进入 projector 前执行 **`2×2`** pixel shuffle，视觉 token 数降为四分之一，使最高 **`3584×3584`** 像素输入仍能进入 1M context。

<span style="color: rgb(46,161,33); background-color: inherit">从头训练并没有牺牲视觉能力：MoonViT-V2 在视觉评测上匹配 SigLIP 初始化基线。</span>区别在于 next-token prediction 会让视觉表示直接服务语言建模，保留文本、布局和局部结构线索；contrastive pre-training 更偏向全局语义。视觉路径先经 MoonViT-V2，再由轻量 MLP projector 映射到 LLM hidden space。

6. **<span style="color: rgb(36,91,219); background-color: inherit">Per-Head Muon</span>**

K3 延续 Muon 优化矩阵参数，但 Q/K/V projection 不再对完整 momentum matrix 统一做 Newton-Schulz orthogonalization，而是按 attention head 切块后分别正交化。完整矩阵方案会让 gradient 或 momentum 较大的 head 主导共同更新方向，小尺度 head 得不到充分归一化。<span style="color: rgb(46,161,33); background-color: inherit">Per-Head Muon 把各 head 的 update scale 拉回同一数量级，并降低 tall per-head block 上 Newton-Schulz iteration 的开销。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">预训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">预训练数据</span>**

<span style="color: rgb(100,37,208); background-color: inherit">文本 corpus 覆盖 Web Text、Code、Mathematics 和 Knowledge 四个主域；视觉 corpus 包含 caption、interleaved image-text document、OCR、perception、video 与 visual coding。</span>各文本域分别执行规则过滤、classifier quality scoring 和去重，domain sampling rate 由小模型 ablation 决定。Knowledge 与 Mathematics 数据沿用 rephrasing 流程：多风格、多视角 prompt，chunk-wise autoregressive generation，再与源文档做 fidelity verification。

视觉数据结合开源集合和内部过滤、合成、去重 pipeline。定位监督同时使用绝对坐标和归一化 **`[0,1]`** 坐标，兼顾精确定位与分辨率泛化。programmatic multimodal data 把代码与渲染结果绑定，覆盖 SVG、3D asset、Webpage、Game 和 CAD schematic。<span style="color: rgb(100,37,208); background-color: inherit">这部分数据直接服务“写代码→看渲染→继续修改”的视觉闭环，而不是只扩充普通图文问答。</span>

2. **<span style="color: rgb(36,91,219); background-color: inherit">Scaling Law</span>**

架构、数据与 training recipe 同时变化后，旧模型的最优超参数不能直接平移。K3 重新搜索 batch size、peak learning rate、tokens-per-parameter ratio 和 model shape，并在 held-out OOD validation 上拟合 scaling law。<span style="color: rgb(46,161,33); background-color: inherit">曲线相对 K2 整体下移，对应约 </span>**`2.5×`**<span style="color: rgb(46,161,33); background-color: inherit"> scaling efficiency。</span>拟合结果如下。

learning-rate schedule 也单独比较 cosine decay 与 WSD。两者即使模型规模和 token budget 相同，最优 peak learning rate 与 batch size 仍明显不同，因此不能共用一组超参数做横向比较。<span style="color: rgb(46,161,33); background-color: inherit">分别做 scaling-law search 后，cosine decay 的 final loss 更低，最终采用 cosine schedule。</span>

![K3 相对 K2 的 scaling-law 曲线整体下移，对应约 2.5 倍 scaling efficiency。](../../images/视觉多模态讲义（上）-figure-07-scaling-law.png)

| **<span style="color: rgb(36,91,219); background-color: inherit">配置</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Kimi K2</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Kimi K3</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">变化</span>** |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Layers                                                                       | 61                                                                                | 93                                                                                | +52%                                                                         |
| Total Parameters                                                             | 1.04T                                                                             | 2.78T                                                                             | +167%                                                                        |
| Activated Parameters                                                         | 32.6B                                                                             | 104.2B                                                                            | +220%                                                                        |
| Hidden Dimension                                                             | 7168                                                                              | 7168                                                                              | 不变                                                                           |
| Latent MoE Dimension                                                         | 无                                                                                 | 3584                                                                              | 主宽度的 0.5×                                                                    |
| Expert Hidden Dimension                                                      | 2048                                                                              | 3072                                                                              | +50%                                                                         |
| Routed / Active / Shared Experts                                             | 384 / 8 / 1                                                                       | 896 / 16 / 2                                                                      | expert pool 与激活数同步扩大                                                         |
| Attention Heads                                                              | 64                                                                                | 96                                                                                | +50%                                                                         |
| Vocabulary / Dense / MTP                                                     | 160K / 1 / 1                                                                      | 160K / 1 / 1                                                                      | 不变                                                                           |
| Training Context                                                             | 128K                                                                              | 1M                                                                                | 8×                                                                           |
| Attention                                                                    | 61 MLA                                                                            | 69 KDA + 24 MLA                                                                   | Hybrid KDA-MLA                                                               |
| Activation                                                                   | SwiGLU                                                                            | SiTU-GLU                                                                          | 大值区间有界                                                                       |
| ViT                                                                          | 无                                                                                 | 401M，27 层，patch 14，12 heads                                                       | Native Vision                                                                |

3. **<span style="color: rgb(36,91,219); background-color: inherit">Training Recipe</span>**

语言与视觉从训练开始就共同优化，visual token 与 text token 交错进入统一 next-token prediction objective。优化器使用 Per-Head Muon，配合 K2 的 weight clipping 与 Quantile Balancing。learning rate 采用 cosine decay，前 **`1%`** 线性 warmup，weight decay 全程为 **`0.1`**。预训练 context 从 **`8K`** 起步，随后扩到 **`64K`**。

4. **<span style="color: rgb(36,91,219); background-color: inherit">长上下文扩展</span>**

<span style="color: rgb(46,161,33); background-color: inherit">NoPE 让位置关系隐式进入 KDA 的 recurrent gating 与 decay，扩到 1M 时不需要修改 positional encoding。</span>长文档和长视频先经过 exact/fuzzy dedup、视频 frame perceptual hash、规则与 classifier 质量过滤、结构校验；真正连贯的长样本数量不足，因此在 cooldown 阶段上采样，防止被短样本淹没。

<span style="color: rgb(216,57,49); background-color: inherit">只把短文拼长不会自动产生 long-range capability。</span>训练数据还会对多模态文档和子任务做有约束的 permutation 与 concatenation，把解题证据分散在整个 1M context，使任务必须跨远距离检索才能完成。context curriculum 分四阶段：预训练从 **`8K→64K`**，cooldown 再从 **`256K→1M`**；昂贵的长序列计算只占总 token budget 的小部分，KCP 负责把 sequence dimension 分摊到多设备。

* **<span style="color: rgb(36,91,219); background-color: inherit">后训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">三阶段流程</span>**

<span style="color: rgb(100,37,208); background-color: inherit">后训练按 SFT 冷启动、domain RL 专家化、MOPD 统一蒸馏三阶段推进。</span>SFT 建立可用的 Agent policy；RL 在 general tasks、general agents、coding agents 三个大域分别训练 low、high、max 三档 reasoning effort；最后用 Multi-Teacher On-Policy Distillation 把 **`9`** 个 expert policy 合并为同一个可控模型。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Supervised Fine-Tuning</span>**

<span style="color: rgb(100,37,208); background-color: inherit">SFT trajectory 由上一代 Kimi 的 domain-specialized model 合成，再经过多阶段 verification 与 human-in-the-loop annotation。</span>所有复杂 Agent 轨迹用 XTML chat template 序列化，统一表达 reasoning、tool call、tool result 与多轮环境状态。SFT 目标不是只提高 instruction following，而是为后续 RL 准备 adaptive reasoning、精确 tool calling 和 long-horizon execution 的 cold-start policy。

<span style="color: rgb(100,37,208); background-color: inherit">QAT 从 SFT 就开始：MoE expert weight 目标格式为 MXFP4，expert input activation 使用 MXFP8；attention projection、latent MoE projection、shared expert 和 router 保持更高精度。</span> <span style="color: rgb(46,161,33); background-color: inherit">低精度误差在整个后训练阶段被 policy 直接适应，而不是部署前临时量化。</span>

3. **<span style="color: rgb(36,91,219); background-color: inherit">Reinforcement Learning</span>**

三个 RL 大域分别覆盖：general tasks 包含通用体验、视觉、推理、faithfulness、search 与 knowledge work；general agents 包含长程 assistant、deep research 与 paragraph-level writing；coding agents 包含 SWE、coding experience、kernel 与 web development。每个域训练 low、high、max 三档 effort，共形成 **`9`** 个 teacher expert。随着 RL FLOPs 增长，多类任务的 score 与平均 assistant steps 同时上升。

![RL FLOPs 增长时，多类任务得分与平均 assistant steps 同时上升。](../../images/视觉多模态讲义（上）-figure-08-rl-step-scaling.png)

<span style="color: rgb(100,37,208); background-color: inherit">长程 rollout 的尾延迟由 partial rollout 处理。</span>每轮对 N 个 prompt 各采样 K 条轨迹，active workload 为 $$N\times K$$；当完成比例达到 $$\lambda$$ 时立即暂停 generation 并开始 policy optimization，未完成轨迹进入队列，在下一 iteration 优先 resume。单条轨迹可能跨越多轮更新，产生极端 off-policy staleness，<span style="color: rgb(46,161,33); background-color: inherit">训练用 per-token regularization 把更新限制在局部 policy neighborhood 内。</span>

reasoning effort 用 per-problem budget 控制。每个问题先从 cold-start model 估计初始 budget $$b_0(x)$$，若轨迹 token 消耗 $$T(y)$$ 超过 $$\tau b_0(x)$$，task reward 直接改为 **`-1`**。普通任务只统计 thinking token，Agent task 统计 reasoning trace 与 tool-call argument 的累计输出。训练先使用较大的 τ 得到 max expert，再逐步 anneal τ 形成 high 与 low expert，并按 domain 由人工调整。

不可程序验证的 general task 使用 Agentic GRM。judge 必须执行固定协议：读取最终产物、生成 rubric、逐项评分、把 rubric score 写入 scorepad。<span style="color: rgb(216,57,49); background-color: inherit">为抑制“越长越高分”的 reward hacking，输出长度超过 cold-start verbosity</span> $$\ell_0$$ 的 $$\sigma\ell_0$$ 倍时，在 binary comparison 中自动判负。

4. **<span style="color: rgb(36,91,219); background-color: inherit">Multi-Teacher Distillation</span>**

<span style="color: rgb(100,37,208); background-color: inherit">MOPD 每次采样 domain d 与 reasoning effort e，由对应 teacher 指导 student 的 on-policy token。</span>per-token reward 是 teacher/student log-prob ratio 的 stop-gradient，并截断到固定范围：

$$r_{\mathrm{opd}}^d(y_t\mid e,x,y_{<t})=\operatorname{clip}\left(\operatorname{sg}\left(\log\frac{\pi_{\mathrm{teacher}}^{(d,e)}(y_t\mid x,y_{<t})}{\pi_\theta(y_t\mid e,x,y_{<t})}\right),-R_{\max},R_{\max}\right)$$

<span style="color: rgb(46,161,33); background-color: inherit">dense token reward 可直接复用 partial rollout 基础设施。</span> <span style="color: rgb(216,57,49); background-color: inherit">更细粒度的 top-k distillation 在收敛速度和最终性能上都没有清晰优势，因此没有进入最终方案。</span>

5. **<span style="color: rgb(36,91,219); background-color: inherit">部署感知训练</span>**

整个 SFT 与 RL 都使用一致的 MXFP4/MXFP8 scheme，rollout 与 training 不存在 quantization mismatch。speculative decoding 则把预训练 MTP layer 微调成 EAGLE-3 draft model：target model 冻结，只更新单层 draft decoder 与 feature-fusion projection；训练展开 **`7`** 步，第一步之后 draft 使用自己的历史输出，模拟 inference recurrent drafting。

draft 输入融合第 **`1`**、第 **`4`** 和最终 AttnRes block 的低、中、高层特征。projection 初始化为 $$[\mathbf{0}\ \mathbf{0}\ \mathbf{I}]$$，起点等价于 MTP 预训练时使用的高层表示，再逐渐学习低中层信息。优化目标不是普通 KL surrogate，而是直接最大化 lossless speculative sampling acceptance rate：

$$\mathcal{L}_{\mathrm{LK}}=-\log\sum_{x\in\mathcal{V}}\min(p(x),q(x))$$

p 与 q 在 temperature **`1`** 下计算，不附加 ground-truth cross-entropy；draft 也沿用 expert MXFP4、activation MXFP8 的 QAT 设置。

6. **<span style="color: rgb(36,91,219); background-color: inherit">统一 Agent 环境</span>**

<span style="color: rgb(216,57,49); background-color: inherit">固定单一 Agent harness 会让 policy 过拟合某套 tool schema、system prompt、context management 或 interaction protocol。</span> <span style="color: rgb(100,37,208); background-color: inherit">white-box RL environment 把 harness 表示成可组合模块：tool interface、system prompt、context strategy、skill、memory、subagent 等都由配置选择。</span>训练时动态组合出 Kimi Code、Claude Code、Codex、OpenClaw、Hermes 等主流形态，也能生成全新 harness，减少跨 scaffold 迁移时的脆弱性。

7. **<span style="color: rgb(36,91,219); background-color: inherit">任务合成</span>**

<span style="color: rgb(100,37,208); background-color: inherit">task synthesis 由自演化层级知识图谱控制粒度与覆盖。</span>图谱从 coarse seed node 开始，每个 node 分配 Agent 做多轮 web search；新增概念前先查已有 DAG，复用等价或相关 node，edge 始终从 coarse concept 指向 fine-grained concept；当节点足够 atomic 时停止扩展。采样时可选单节点或相关节点组合，并把 ancestor context 一起转成 query keyword，检索真实材料，再按目标 task type 合成训练任务。概念扩展、材料检索与任务合成的关系如下。

![知识图谱驱动的任务合成：概念扩展、材料检索与任务类型采样形成闭环。](../../images/视觉多模态讲义（上）-figure-09-task-synthesis.png)

8. **<span style="color: rgb(36,91,219); background-color: inherit">可验证任务</span>**

> **<span style="color: rgb(36,91,219); background-color: inherit">Agentic environment 中的主要任务族</span>**
>
> 1. **<span style="color: rgb(36,91,219); background-color: inherit">信息与专业工作</span>**：多步 web research、投行、data analysis、legal practice 等任务要求数十到数百步操作，并交付可核验产物。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">视觉推理</span>**：STEM、visual puzzle 与 chart understanding 在隔离 Python sandbox 中迭代 crop、zoom、transform、compute，再把新图像作为 observation 返回。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">Kernel 优化</span>**：覆盖 CUDA、Triton、CuTe DSL、Gluon、ThunderKittens、TileLang 与 BF16/FP8/FP4。数值误差越界 reward 为 0，匹配 expert implementation 得 0.5，逼近 roofline 逐渐趋近 1，并检测 CUDA graph replay、input caching、precision reduction 等作弊。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">Personal Assistant</span>**：Gmail、Notion、Slack、Canvas 等 mock app 组成持续多日环境，单次 rollout 可达数千 tool calls 与数百万 context tokens，每个事件由 deterministic rule 或 LLM evaluator 评分。

Kernel task 从单算子扩展到 fused mega-kernel，材料来自 Flash Linear Attention 等高质量 GitHub 仓库；Personal Assistant 的初始 workspace 也不是固定模板，而是由 Agent 搜索公开材料并构造成一致的任务环境。RL runtime 还要持续建模跨应用 event stream 与 world-state transition，使多日任务在 Gmail、Notion、Slack、Canvas 之间保持状态连续。

AET 的 verifier 不只覆盖黑盒系统复现，也覆盖 quantitative factor discovery 与 tax auditing。Camera Repair Management System 案例要求 Agent 通过 oracle query 反推隐藏的 3D-camera repair system，并重建成 Web 应用；训练循环反复执行 hypothesis、action、verifier feedback 与 adaptation。

Autonomous Execution Task 只给 initial state、约束目标、tool action space、execution budget 和 independent verifier，不提供 reference trajectory。Agent 自己做 decomposition、tool selection、planning、error recovery 与 termination；reward 取决于最终 environment state，不接受自报完成。public verifier 提供诊断，hidden verifier 检查 held-out case；limited submission budget 与 penalty reward 共同压制 reward hacking。黑盒 Camera Repair Management System 的完成曲线如下。

![AET 黑盒系统复现曲线：完成度由独立 verifier 按执行状态评估。](../../images/视觉多模态讲义（上）-figure-10-aet-completion.png)

Web development task 从一句 scene description 到多段 specification，产物覆盖 website、interactive game、3D/WebGL、data visualization、SVG 和 full-stack app。每个任务在 container sandbox 中运行，并随机化 Agent scaffold。reward 同时包含 deterministic functional test、结构与 pixel similarity、source inspection 和可交互 model judging；<span style="color: rgb(216,57,49); background-color: inherit">build 失败、runtime error 或伪造结果时总 reward 直接归零。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">基础设施</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">KDA 协同设计</span>**

<span style="color: rgb(100,37,208); background-color: inherit">KDA 用固定大小 recurrent state 替代随序列增长的 softmax KV cache，但 state update 有串行依赖。</span> <span style="color: rgb(46,161,33); background-color: inherit">FlashKDA 把 chunk 内 token-parallel 计算与跨 chunk head-parallel state propagation 重叠，分别调度和 autotune，服务 training 与 prefill，并作为 flash-linear-attention backend 自动分派。</span>

FlashKDA 是基于 CUTLASS 的 chunkwise kernel，token-parallel stage 与 head-parallel recurrence 各自调度和 autotune，性能明显高于 Triton reference。training、prefill 与 decode 的瓶颈不同，因此 decode 不复用同一条 kernel 路径，而是在 serving 侧单独处理 in-place state 与 speculative verification。

KCP 的跨 rank payload 是两块固定大小 tensor：累计 transition 与 zero-state local contribution。<span style="color: rgb(46,161,33); background-color: inherit">它们通过一次 all-gather 交换，再按 document order 执行 prefix composition；通信量不随 sequence length 增长。</span>对应 KDA context-parallel implementation 已进入 Flash Linear Attention 的公开实现路径。

单卡 ultra-long prefill 下，<span style="color: rgb(216,57,49); background-color: inherit">纯 TP 只切 attention head，不能缩短 recurrence；当每个 rank 只剩少数 head 时，大量 SM 空闲。</span>intra-device CP planner 把 sequence segment 分到同一 rank 的多个 SM，<span style="color: rgb(46,161,33); background-color: inherit">各 segment 独立计算 transition，再精确 compose 初始 state，全程不做跨设备通信。</span>

跨设备 KCP 不能照搬 vanilla linear attention 的 state summation，因为 KDA 的 token-dependent transition matrix 会作用在 incoming state 上。每个 rank 先独立计算累计 transition $$\mathbf{M}_{[i]}^{T_i\leftarrow1}$$ 与 zero-state local contribution $$\widetilde{\mathbf{S}}_{[i]}^{T_i}$$，rank update 具有结合律，可用 prefix scan 复合。所有 rank 只做一次固定大小 all-gather，再按文档顺序恢复 incoming state，compute 随 CP size 线性扩展。局部 transition 与 state composition 的矩阵关系如下。

![KCP 把局部 transition 与 zero-state contribution 组合为固定大小的跨 rank state 同步。](../../images/视觉多模态讲义（上）-figure-kcp-context-parallelism.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">3T 预训练拓扑</span>**

<span style="color: rgb(100,37,208); background-color: inherit">3T 级训练同时使用带 virtual stage 的 PP、EP、ZeRO-1 DP、Pipeline ZeRO-2 gradient sharding 和 CP。</span>shared expert 在 EP rank 间复制，routed expert dispatch/combine 的 all-to-all 与计算重叠。执行时还要同时隐藏 DataLoader、ViT forward/backward、activation offload/onload、gradient reduce、parameter gather 与 EP communication。整体 PP phase overlap 如下。

![PP 各阶段同时重叠计算、EP/NCCL 通信、activation offload/onload 与 gradient movement。](../../images/视觉多模态讲义（上）-figure-11-training-pipeline.png)

3. **<span style="color: rgb(36,91,219); background-color: inherit">MoonEP</span>**

<span style="color: rgb(100,37,208); background-color: inherit">MoonEP 用动态 redundant expert 把每个 EP rank 的 token load 强制变成完全相同的 </span>$$S\times K$$<span style="color: rgb(100,37,208); background-color: inherit">。</span>forward 根据当前 micro-batch 与 layer 的 router output 在线规划并预取 redundant expert；backward 先把其 gradient 写入本地 reduce buffer，计算完成后再归并到 home rank。

理论上每个 rank 预留不超过 $$E/R$$ 个 redundant-expert slot 就总能找到平衡方案，这个上界近似 tight。系统离线用 ILP 生成代表性最优解，在线 GPU planner 追求 near-optimal 且始终满足上界。planning kernel 直接预计算每个 token 的远端 expert-grouped 目标位置，实现 fused permute/unpermute zero-copy；最坏失衡时 DeepEP 需要 $$S\times K\times R$$ 通信 buffer，MoonEP 只需要固定 $$S\times K$$。

<span style="color: rgb(46,161,33); background-color: inherit">完全平衡使所有 layer 的 compute shape 静态已知，host 不再逐层同步 token count。</span>rank 内 expert token count 仍有偏斜，因此 routed-expert GEMM 使用 workload-aware scheduler 和离线校准的硬件 cost model；shared expert GEMM 放到独立 stream，与其他 kernel 重叠。

MoonEP 的可行性保证与预设冗余上限的方法不同。<span style="color: rgb(216,57,49); background-color: inherit">ECHO、UltraEP 一类方案预先固定 redundant expert 数量或 per-rank token cap，遇到 cap 内无解时训练会停，而且 cap 仍需手工调节；</span>MoonEP 的 $$E/R$$ <span style="color: rgb(46,161,33); background-color: inherit">上界保证每一步都存在完全平衡方案。</span>online planner 不追求每步精确 ILP 最优，而是在 GPU 上以可忽略开销给出 near-optimal 解。

4. **<span style="color: rgb(36,91,219); background-color: inherit">内存优化</span>**

> **<span style="color: rgb(36,91,219); background-color: inherit">训练内存的六层处理</span>**
>
> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Activation manager</span>**：recomputation、block-wise FP8 quantization、CPU offload 与 remote offload 作为 tensor-level storage policy 组合，统一 memory pool 避免 multi-stream fragmentation。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">MoE activation</span>**：重写 permuted probability gradient，使其不依赖 forward output；backward 通过重新 dispatch 恢复 group-GEMM input，并把通信藏进 GEMM backward。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">Block AttnRes</span>**：boundary representation 只生成一次，layer 内 checkpointing 保持与普通 residual 相同的 saved activation；PP stage 只增量发送新 block。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">PP rank 平衡</span>**：interleaved 1F1B warmup 导致不同 rank activation 驻留量不均，Mooncake Transfer Engine 把 activation 远端 offload 到空闲 PP rank。
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">Pipeline ZeRO-2</span>**：gradient 在 DP rank 间分片并存入 CPU，GPU 只保留 double grad buffer。
>
> 6. **<span style="color: rgb(36,91,219); background-color: inherit">Muon 参数收集</span>**：每个 rank 只用 P2P 拉取自己拥有参数对应的 shard，按 model-chunk buffer pipeline 通信与 orthogonalization，避免全量 all-gather。

Activation manager 的 recomputation 以 function 为粒度，因此可以跨 layer；存储策略通过 tensor annotation 声明，与模型代码解耦。activation 按 layer prefetch 回 GPU 并与 compute 重叠，element-wise operator 主要走 recomputation，其他大张量通常组合 block-wise FP8 与 CPU/remote offload。Block AttnRes 的 cache-based PP communication 只传新增 block，micro-batch 结束立即释放，达到该结构的理论最低通信驻留内存。

5. **<span style="color: rgb(36,91,219); background-color: inherit">视觉 Encoder 优化</span>**

<span style="color: rgb(216,57,49); background-color: inherit">大图和长视频让 ViT compute 极不均衡。</span>dynamic CP 按 patch dimension 把单个大样本切到多设备，通过 gather-KV 完成 attention；一个 CP group 还可拆成多个 sub-CP group，把多张大图按负载分配，<span style="color: rgb(46,161,33); background-color: inherit">避免 communication fraction 随 scale 一起放大。</span>

这一方案是在 K2.5 的 Decoupled Encoder Process 基础上继续拆分：DEP 已把 ViT 与 text training 分成独立阶段并在 PP stage 间平衡视觉前后向；K3 再把剩余 ViT work 填进 interleaved 1F1B bubble。dynamic CP 先降低单个大样本的 encoder latency 和跨设备失衡，bubble scheduling 才能真正把余下视觉计算从 critical path 隐藏掉。

interleaved 1F1B 中，最早 micro-batch 的 text forward 集中在 pipeline 开头，最后 micro-batch 的 text backward 集中在结尾。系统把 ViT forward/backward 进一步拆分：首批 forward 同步执行，其余 ViT compute 填入 PP bubble，backward 同理，绝大部分视觉 Encoder 开销不再落在 critical path。

6. **<span style="color: rgb(36,91,219); background-color: inherit">1M Agentic RL</span>**

co-located RL training 把单次 1M-context 实验控制在几百张 GPU 内，partial rollout 减少 ultra-long trajectory 的尾延迟，但未完成轨迹的 KV cache 要跨 iteration 保存，与训练显存竞争。external KV cache pool 采用 write-back：active decode block 保留在 GPU，只有被 GPU 驱逐但未来可复用的 idle prefix 才写入 CPU DRAM；KDA state 与 MLA KV block 同步 offload/prefetch。training iteration 结束后，model weight 与 optimizer state 转存 NVMe 给 external pool 腾出 DRAM；rollout 结束后释放 pool。

partial rollout 会让上一 iteration 的大量未完成 prefill 在下一轮同时返回，speculative decoding 又会加快固定 tool-call 间隔内的 request turnover，<span style="color: rgb(216,57,49); background-color: inherit">两者共同增加 prefix-block churn 与 preemption 风险。</span> <span style="color: rgb(46,161,33); background-color: inherit">write-back 相比 write-through 只搬运已被 GPU 驱逐且仍可复用的 idle prefix，不为仍在 active decode path 的 block 制造冗余 CPU 副本和带宽开销。</span>

non-policy weight materialize 到 policy FP32 gradient buffer 是安全的，因为真正计算 gradient 时该 buffer 会被覆盖；<span style="color: rgb(46,161,33); background-color: inherit">双 slot pipeline 让当前 VPP chunk 做 forward 的同时预取下一 chunk，不增加新的 GPU allocation，也避免额外 fragmentation。</span>

<span style="color: rgb(46,161,33); background-color: inherit">rollout auto-throttling scheduler 根据 active request、queue length 和 KV cache utilization 动态控制发往 inference engine 的并发数：</span>早期 context 短时提高利用率，后期 cache pressure 上升时自动降并发。reference model 等 forward-only non-policy model 常驻 CPU，需要时把 weight materialize 到 policy FP32 gradient buffer；K3 每卡只保留两个 VPP chunk 的 gradient slot，一个计算当前 chunk，另一个预取下一 chunk。

7. **<span style="color: rgb(36,91,219); background-color: inherit">Sandbox</span>**

<span style="color: rgb(100,37,208); background-color: inherit">运行时同时包含传统 container、GPU sandbox 与基于 Firecracker microVM 的 AgentENV。</span>microVM 允许 mount disk、run container 甚至 launch VM，同时把 Agent 的高风险探索隔离在虚拟机边界内；<span style="color: rgb(216,57,49); background-color: inherit">传统 container 在早期实验中曾被意外操作触发 kernel panic 与 deadlock。</span>

AgentENV 只保存 checkpoint 后被写脏的 memory page，checkpoint latency 低至 **`133 ms`**，resume 低至 **`49 ms`**。Pause 后 sandbox 不占 CPU/内存，而等待模型 inference 可占生命周期的 **`98%`**；Fork 复制精确状态供 reward judging 且不影响原环境；Snapshot 周期保存用于 error recovery。

OverlayBD image、custom ublk driver、storage sharing 与 P2P transport 支撑数万 sandbox 秒级批量创建，单 sandbox launch 达到 sub-second。copy-on-write memory 与 page-cache optimization 在真实 workload 中达到 **`6.5×`** memory overcommit。整个训练和评测共创建 **`51,219,741`** 个 sandbox，覆盖 **`1,505,678`** 个 image。

8. **<span style="color: rgb(36,91,219); background-color: inherit">Prefix Cache</span>**

Hybrid KDA-MLA 同时维护两种生命周期不同的 cache：MLA KV cache 随 sequence length 增长并按 token 分页；KDA recurrent state 大小固定，每个 request 只有一份。系统把 KDA state pack 进 MLA KV 使用的同一 paged block pool，并统一 page byte size、allocation、reference count 与 eviction。每个 head 的 state byte stream 连续存放；prefill/decode disaggregation 使用不同 TP degree 时，在 transfer path 完成 re-layout，不做 GPU 端二次 shuffle。

<span style="color: rgb(100,37,208); background-color: inherit">每个 KDA head 的 state byte stream 是独立的最小跨节点传输单元。</span>统一 pool 还带来一个零开销调试特性：KDA page 与 MLA page 的 payload 形态高度不对称，<span style="color: rgb(216,57,49); background-color: inherit">若发生 type-confused access，结果会直接变成无意义数据，而不会悄悄产生“看似合理”的错误值。</span>

<span style="color: rgb(216,57,49); background-color: inherit">如果 hash granularity 被 KDA checkpoint granularity 绑定，physical block 会被迫放大到 </span>**`1024–6144`**<span style="color: rgb(216,57,49); background-color: inherit"> tokens，短请求无法命中，chunked prefill 也要等整块填满才可复用。</span>K3 把两种粒度拆开：physical block 为 **`6144`** tokens，内部再切 **`512`**-token hash block；KDA checkpoint 只在 hash endpoint 的稀疏子集保存，通常保留 conversation-turn boundary。

部分填充 MLA page 会按最后一个完整 hash block 的 chained hash 注册。每次 prefill 后，KDA kernel 保存最后一个 hash-aligned position 的 state；被后续 checkpoint 覆盖的中间 snapshot 回收，公开 snapshot 只读，cache hit 时先 copy 到 request-private running state，再写 fresh slot。lookup 先匹配完整 physical block，再在首个 miss block 内匹配 hash endpoint；随后要求每个 KDA cache group 在同一 boundary 都有 checkpoint，最终选择两阶段共同满足的最长 prefix。**`2800`**-token 请求可以在 **`B=2560=5×512`** 处命中，不必重算 **`[0,B)`**。细粒度 prefix reuse 如下。

![6144-token physical block 内使用 512-token hash block，并在 B=2560 处同时恢复 MLA KV 与 KDA checkpoint。](../../images/视觉多模态讲义（上）-figure-12-prefix-cache.png)

<span style="color: rgb(100,37,208); background-color: inherit">并发一致性靠三条约束维持：</span>所有 cache group 先 pin hit block，再做任何 private allocation；当前 scheduling step 内刚分配或刚注册的 block 在 GPU copy 完成前不能参与 match；任一 KDA group 的 checkpoint 被 evict 时，其他 group 的 sibling checkpoint 同步失效，保证 boundary 要么对全部 group 可恢复，要么完全不可见。

9. **<span style="color: rgb(36,91,219); background-color: inherit">高性能 Kernel</span>**

KDA decode 的问题是 recurrent state 每步原地更新，speculative verification 拒绝部分 draft token 后无法直接回滚。系统不保存每个 draft position 的巨型 state snapshot，只缓存小得多的 projected input；verification 后在片上 replay accepted prefix，写回 verified token 与 bonus token state。replay token、bonus token 和下一 draft window 共用一个 fused recurrent loop，覆盖 ShortConv、input normalization、gating、KDA recurrence 与 output normalization。

<span style="color: rgb(46,161,33); background-color: inherit">replay 只缓存 projected input，这些张量远小于 recurrent state。verification latency 随验证 token 数次线性增长，并低于逐位置 state snapshot baseline；</span>projection cache 不离开 decode stage，因此 prefix cache 与 prefill/decode disaggregation 的传输 payload 与非 speculative serving 保持一致。

<span style="color: rgb(100,37,208); background-color: inherit">Block AttnRes prefill 把 TP all-reduce 拆成 reduce-scatter 与 all-gather，在中间 sequence-sharded activation 上执行 intra-block kernel，使每个 token 的 block representation 只 materialize 在一个 rank。</span>decode 时 inter-block kernel 放 side stream，与 main stream 独立计算重叠；intra-block merge、partial-sum update 与 RMSNorm 融进前序 TP all-reduce。

Stable LatentMoE 把 latent down-projection 与 router 融成一个 GEMM，latent weight 按 rank sharding，并用 multimem store 把 output all-gather 融进 GEMM epilogue，再与 shared-expert compute 重叠。小 batch decode 时 routed-expert GEMM 变成 memory-bound weight streaming，kernel 采用 WarpDecode 的 token-centric 设计：每个 warp 负责一个 output neuron，lane team 分担不同 expert，最后 warp-wide reduction；weight layout 离线 permute，减少 runtime dequantization。

10. **<span style="color: rgb(36,91,219); background-color: inherit">集群调度</span>**

1M coding request 的典型输入带有 **`400K`** prefix，但每轮新增 prefill 可能只有 **`4K`**。cache-aware affinity 把 session 路由到持有 prefix cache 的 cluster；consistent hashing 为每个 session 固定 primary 与 secondary，正常流量只走 primary，故障后 secondary 重新 prefill。secondary assignment 在全 fleet 均匀分布，单 cluster failure 的重算压力不会集中到一个接管节点。

production request 从小于 **`2K`** 到 **`1M`**，单请求成本跨度约三个数量级。按平均请求做 capacity planning、queueing 或 rate limit 会在长请求突发时失效，并拖垮短请求 TTFT。budget-based admission control 为不同 request class 分配独立 resource budget，长 context burst 只能消耗自己的 capacity share，不能越过预算破坏其他流量的 SLO。

**<span style="color: rgb(222,120,2); background-color: inherit">技术总结</span>**

<span style="color: rgb(100,37,208); background-color: inherit">K3 的核心不是某一个新算子，而是三条信息流的协同扩展。</span>sequence 维用 Hybrid KDA-MLA 在固定 recurrent state 与 global attention 之间分工；depth 维用 Block AttnRes 从 embedding、历史 block 和当前 partial sum 中选择信息；channel 维用 Stable LatentMoE 在 full-width shared path 与 latent routed path 之间分工。三者分别控制长上下文、深层网络和超大稀疏容量。

训练侧形成了完整闭环：文本与视觉从头联合预训练，context 按 **`8K→64K→256K→1M`** 渐进扩展；SFT 建立 Agent cold start，RL 在 domain × reasoning effort 上训练 **`9`** 个 expert policy，MOPD 再把这些策略蒸馏回单一模型。MXFP4 expert weight、MXFP8 activation 与 EAGLE-3 draft model 都在后训练阶段直接适配，避免训练精度路径和上线执行路径脱节。

<span style="color: rgb(100,37,208); background-color: inherit">系统侧同样围绕模型结构定制：</span>FlashKDA 与 KCP 解决 recurrent state 的设备内和跨设备并行；MoonEP 把每个 EP rank 的 token load 固定为 $$S\times K$$；activation manager、Pipeline ZeRO-2 与 P2P Muon 控制训练内存；external KV pool、AgentENV checkpoint、细粒度 KDA prefix cache 和 fleet-level admission control 负责保存并恢复超长 Agent 状态。

<span style="color: rgb(100,37,208); background-color: inherit">编译器、Web、EDA、科研和个人助理任务在机制上不是彼此独立的能力模块，而是同一套 Agent learning loop 的不同环境实例：</span>white-box harness 组织工具与上下文，知识图谱生成任务，verifier 提供可学习 reward，sandbox 保存环境状态，partial rollout 与 1M runtime 支撑跨 iteration 执行。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>：`2.78T`** 总参数和 **`1M`** context 只是容量上限。真正决定系统能否训练和服务的，是每 token **`104.2B`** 激活参数、KDA/MLA cache 的联合恢复、专家负载是否完全平衡，以及长请求能否在 GPU、CPU DRAM、NVMe 和集群之间稳定迁移。

---

[Previous](01-视觉基础.md) | [Contents](../../README.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=2)
