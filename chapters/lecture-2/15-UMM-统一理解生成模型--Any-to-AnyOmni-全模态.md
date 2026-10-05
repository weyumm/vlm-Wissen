[Previous](14-UMM-统一理解生成模型--AR--Diffusion.md) | [Contents](../../README.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=15)

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

[Previous](14-UMM-统一理解生成模型--AR--Diffusion.md) | [Contents](../../README.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=15)
