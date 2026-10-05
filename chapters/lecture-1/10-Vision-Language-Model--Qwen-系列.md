[Previous](09-Vision-Language-Model--MiniGPT-系列.md) | [Contents](../../README.md) | [Next](11-Vision-Language-Model--Intern-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=10)

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

---

[Previous](09-Vision-Language-Model--MiniGPT-系列.md) | [Contents](../../README.md) | [Next](11-Vision-Language-Model--Intern-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=10)
