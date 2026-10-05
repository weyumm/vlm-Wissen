[Previous](11-Vision-Language-Model--Intern-系列.md) | [Contents](../../README.md) | [Next](13-Vision-Language-Model--其他.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=12)

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

---

[Previous](11-Vision-Language-Model--Intern-系列.md) | [Contents](../../README.md) | [Next](13-Vision-Language-Model--其他.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=12)
