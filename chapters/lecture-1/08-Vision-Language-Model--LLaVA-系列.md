[Previous](07-Vision-Language-Model--结构优化.md) | [Contents](../../README.md) | [Next](09-Vision-Language-Model--MiniGPT-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=8)

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

---

[Previous](07-Vision-Language-Model--结构优化.md) | [Contents](../../README.md) | [Next](09-Vision-Language-Model--MiniGPT-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=8)
