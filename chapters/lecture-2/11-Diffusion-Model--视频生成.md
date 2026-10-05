[Previous](10-Diffusion-Model--其他工作.md) | [Contents](../../README.md) | [Next](12-UMM-统一理解生成模型--Diffusion-based.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=11)

## 4.6 <span style="color: rgb(36,91,219); background-color: inherit">视频生成</span>

### 4.6.1 <span style="color: rgb(36,91,219); background-color: inherit">CogVideo</span>

CogVideo 引&#x5165;**<span style="color: rgb(100,37,208); background-color: inherit">多帧率分层训练</span>**<span style="color: rgb(100,37,208); background-color: inherit">，以更好地对齐文本与视频语义</span>，以&#x53CA;**<span style="color: rgb(100,37,208); background-color: inherit">双通道注意力机制</span>**<span style="color: rgb(100,37,208); background-color: inherit">，用于从预训练的文本-图像模型中继承知识以进行视频生成</span>。为克服大模型和长序列带来的巨大内存与时间开销，参&#x8003;**`Swin Attention`**&#x7684;思想，将其扩展至自回归视频生成场景。

* **<span style="color: rgb(36,91,219); background-color: inherit">多帧率分层训练</span>**

多帧率分层训练与生成遵循 VQVAE 框架，首先将每一帧图像转化为图像 token。每个训练样本包&#x542B;**`5`**&#x5E27;的 token 序列，但这里 CogVideo 的训练方法在训练序列构建和生成过程上有所不同：

1. **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

核心设计是<span style="color: rgb(100,37,208); background-color: inherit">在文本前添加一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">帧率 token</span>**<span style="color: rgb(100,37,208); background-color: inherit">，并按此帧率采样视频帧，以构成固定长度的训练序列</span>。这个设计的动机有两点：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">语义不匹配问题</span>**：直接以固定帧率将长视频切分为片段，往往导致语义不完整。虽然仍使用全文本描述，但截取的片段可能仅包含不完整的动作。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">相邻帧高度相似性</span>**：相邻帧通常非常相似，若模型在预测时发生剧烈变化，将导致较大的损失。这会使得模型倾向于“复制”前一帧（作为捷径），从而削弱其探索长程依赖的能力。

因此在每个训练样本中，作者希望文本与所选帧尽可能语义一致。为此<span style="color: rgb(100,37,208); background-color: inherit">预定义了一系列帧率，并为每对文本-视频选择</span>**<span style="color: rgb(100,37,208); background-color: inherit">最低可行帧率</span>**<span style="color: rgb(100,37,208); background-color: inherit">，只要能在该视频中以该帧率采样到至少</span>**`5`**<span style="color: rgb(100,37,208); background-color: inherit">帧即可</span>。虽然这种方法提升了文本-视频对齐度，但<span style="color: rgb(216,57,49); background-color: inherit">低帧率生成的视频可能缺乏连贯性</span>。为此，<span style="color: rgb(100,37,208); background-color: inherit">额外训练一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">帧插值模型</span>**<span style="color: rgb(100,37,208); background-color: inherit">，用于在序列生成模型输出的关键帧之间插入过渡帧</span>。得益&#x4E8E;**`CogLM`**&#x7684;通用性，这两个模型可共享相同结构，仅通过不同的注意力掩码实现不同功能。

2. **<span style="color: rgb(36,91,219); background-color: inherit">生成</span>**

![](../../images/视觉多模态讲义（下）-image-105.png)

多帧率分层生成是一个递归过程，如上图，包含两个阶段：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1：序列生成</span>**。基于低帧率和文本，顺序生成$$T_s$$个关键帧。输入序列为
>
> $$[\text{\{Frame Rate\}}\ \text{\{Text\}}\ [\text{B}]\ \text{\{Frame}_1\} \dots \text{\{Frame}_{T_s}\}]$$
>
> 实际中设$$T_s = 5$$，最小采样帧率为 1 FPS。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2：递归插值</span>**。基于文本、帧率及已知帧，递归地进行帧插值。每轮插值中，<span style="color: rgb(100,37,208); background-color: inherit">将当前帧序列划分为多个长度为</span>$$\left\lceil \frac{T_s}{2} \right\rceil$$<span style="color: rgb(100,37,208); background-color: inherit">的重叠块，保证首尾重叠，并在每个块内相邻帧之间插入新帧</span>。输入序列形式同上，其中偶数位置的帧，即$$\text{Frame} \ 2i \ (i=1,2,\dots,\left\lfloor \frac{T_s}{2} \right\rfloor)$$，需通过自回归方式生成。<span style="color: rgb(100,37,208); background-color: inherit">通过不断将帧率减半，即</span>$$\text{\{Frame Rate\}} \leftarrow \frac{1}{2} \times \text{\{Frame Rate\}}$$<span style="color: rgb(100,37,208); background-color: inherit">，可实现越来越精细的插值，最终生成高帧率视频</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：CogLM 的作用
>
> 帧插值等任务严重依赖**双向上下文信息**。但<span style="color: rgb(216,57,49); background-color: inherit">以往多数工作采用 GPT 这类单向模型</span>。为利用双向上下文，这里<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**<span style="color: rgb(100,37,208); background-color: inherit">跨模态通用语言模型 </span>**<span style="color: rgb(100,37,208); background-color: inherit">CogLM，它通过将 token 划分为</span>**<span style="color: rgb(100,37,208); background-color: inherit">单向区域</span>**<span style="color: rgb(100,37,208); background-color: inherit">与</span>**<span style="color: rgb(100,37,208); background-color: inherit">双向区域</span>**<span style="color: rgb(100,37,208); background-color: inherit">，统一了双向上下文感知的掩码预测与自回归生成</span>：
>
> * 双向区域内的 token 可相互关注
>
> * 单向区域内的 token 可关注所有双向区域及先前的单向区域
>
> 在上图中：
>
> * Stage 1 的所有帧，以及 Stage 2 中的第 2、4 帧属于**单向区域**；
>
> * $$\text{\{Frame Rate\}}$$、$$\text{\{Text\}}$$及其他帧属于**双向区域**。
>
> 如此设计<span style="color: rgb(100,37,208); background-color: inherit">可在不干扰自回归帧预测的前提下，充分挖掘文本与已知帧中的双向上下文信息</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">双通道注意力</span>**

对于开放域文本到视频生成，理想情况下需覆盖足够多的文本-视频对，以学习视频与文本之间的空间与时间关联。然而<span style="color: rgb(216,57,49); background-color: inherit">高质量文本-视频对的收集往往困难、昂贵且耗时</span>。因此一个自然的想法是利用图像数据辅助学习**空间语义**。**`Video Diffusion Model`**&#x548C;**`NÜWA`**&#x5C1D;试在文本-视频训练中加入文本-图像对，在多项指标上取得了更好效果。但<span style="color: rgb(216,57,49); background-color: inherit">对于纯视频生成模型而言，引入图像数据会显著增加训练成本，尤其在大规模预训练场景下</span>。

CogVideo 提&#x51FA;**<span style="color: rgb(100,37,208); background-color: inherit">不使用图像数据，而是利用预训练的图像生成模型</span>**。例如，<span style="color: rgb(220,155,4); background-color: inherit">CogView2 等模型已充分掌握文本-图像关系，且其训练数据覆盖范围远大于视频数据</span>。作者<span style="color: rgb(100,37,208); background-color: inherit">提出</span>**<span style="color: rgb(100,37,208); background-color: inherit">双通道注意力机制</span>**<span style="color: rgb(100,37,208); background-color: inherit">：在预训练的 CogView2 每一层Transformer中，新增一个</span>**<span style="color: rgb(100,37,208); background-color: inherit">时空注意力通道</span>**。CogView2 的<span style="color: rgb(100,37,208); background-color: inherit">所有原始参数在训练中</span>**<span style="color: rgb(100,37,208); background-color: inherit">冻结</span>**<span style="color: rgb(100,37,208); background-color: inherit">，仅新增注意力层</span>**`Attention-plus`**<span style="color: rgb(100,37,208); background-color: inherit">的参数可训练</span>。这里将 CogView2 原有的注意力块记&#x4E3A;**`Attention-base`**。

![](../../images/视觉多模态讲义（下）-image-104.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：若直接微调 CogView2 用于文本到视频生成，难以有效继承其知识，因为时间注意力具有不同的模式，且在训练初期的大梯度会迅速破坏预训练权重。

结&#x5408;**`Sandwich-LN`**&#x7684;双通道注意力块计算如下：

$$\tilde{x} = \alpha \cdot \text{attention-base}(\text{LayerNorm}(\mathbf{x}_{\text{in}})) + (1 - \alpha) \cdot \text{attention-plus}(\text{LayerNorm}(\mathbf{x}_{\text{in}}))$$

$$\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}} + \text{LayerNorm}(\tilde{x})$$

其中混合因子$$\alpha \in (0,1)^d$$，$$d$$为输入特征$$\mathbf{x}_{\text{in}}$$的隐藏维度。为约束$$\alpha$$在$$(0,1)$$范围内，将其重参数化为：

$$\alpha = \sigma(\mathbf{a}), \quad \mathbf{a} \in \mathbb{R}^d$$

其中$$\sigma$$为 sigmoid 函数，$$\mathbf{a}$$为可学习参数。

**`Attention-plus`**&#x6A21;块与标准多头注意&#x529B;**`Attention-base`**&#x5177;有相同的参数形状，但在计算过程上有所不同。在训练中，作者尝试了两种 attention-plus 结构：**<span style="color: rgb(100,37,208); background-color: inherit">3D 局部注意力</span>**<span style="color: rgb(100,37,208); background-color: inherit">与</span>**<span style="color: rgb(100,37,208); background-color: inherit">3D 移位窗口注意力</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">3D 局部注意力</span>**

位于时空坐标$$(t, x, y)$$（分别对应时间、高度、宽度）的 token 的感受&#x91CE;**`RF`**&#x4E3A;一个三维块，其时空范围由$$l_t, l_x, l_y \in \mathbb{N}^+$$决定：

$$\text{RF}(t,x,y) = \left\{ (k, i, j) \,\middle|\, |x - i| < l_x,\ |y - j| < l_y,\ |t - k| < l_t,\ (k, i, j) \notin \text{Mask}(t,x,y) \right\}$$

其中$$\text{Mask}(t,x,y)$$为 token $$(t,x,y)$$的注意力掩码：

> * 在 **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>**<span style="color: rgb(36,91,219); background-color: inherit">序列生成模型</span>**&#x4E2D;，掩码确保自回归顺序
>
> * 在 **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**<span style="color: rgb(36,91,219); background-color: inherit"> </span>**<span style="color: rgb(36,91,219); background-color: inherit">插值模型</span>**&#x4E2D;，掩码按 CogLM 设计，使所有已知帧对其他帧可见

两个通道在<span style="color: rgb(100,37,208); background-color: inherit">每层融合后</span>**<span style="color: rgb(100,37,208); background-color: inherit">共享同一个 FFN</span>**。这是因为 <span style="color: rgb(100,37,208); background-color: inherit">FFN 包含大量视觉知识且参数量大</span>。鉴于<span style="color: rgb(100,37,208); background-color: inherit">图像与视频的相似性，将图像知识迁移到时序通道有助于视频建模</span>；同时，共享 FFN 可<span style="color: rgb(100,37,208); background-color: inherit">减少参数量，加速训练并降低内存开销</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">3D 移位窗口注意力（Swin Attention）</span>**

为进一步缓解训练与推理中时序通道带来的时间与内存开销，借&#x9274;**`Swin Attention`**&#x7684;思想。原始 Swin Attention 仅适用于非自回归场景，CogVideo <span style="color: rgb(100,37,208); background-color: inherit">通过在移位窗口内施加</span>**<span style="color: rgb(100,37,208); background-color: inherit">自回归注意力掩码</span>**<span style="color: rgb(100,37,208); background-color: inherit">，将其扩展至自回归及时序场景</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：Swin Attention <span style="color: rgb(100,37,208); background-color: inherit">允许</span>**<span style="color: rgb(100,37,208); background-color: inherit">不同帧中相距较远的区域并行生成</span>**<span style="color: rgb(100,37,208); background-color: inherit">，从而进一步加速自回归生成</span>。token 的生成依赖于：
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">自回归掩码</span>**：token 仅能关注当前帧中其之前的 token，或之前帧的所有 token
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">移位窗口</span>**：仅能直接关注宽度和高度维度上距离不超过窗口大小的 token

如右图，<span style="color: rgb(100,37,208); background-color: inherit">可以在未完成前一帧全部生成的情况下，提前开始后续帧部分 token 的生成，也就是实现并行</span>。设每帧高宽为$$X, Y$$，移位窗口高宽为$$A_x, A_y$$。对于两个token $$(t_1, x_1, y_1)$$与$$(t_2, x_2, y_2)$$，其中$$t_1 < t_2$$，若满足

$$(x_1 - x_2)Y + (y_1 - y_2) \geq (t_2 - t_1 + 1)(A_x Y + A_y)$$

则后者无法直接或间接关注前者。

![](../../images/视觉多模态讲义（下）-image-102.png)

这意味着：第$$t$$帧的第$$i$$个 token 可与第$$t+1$$帧的第$$(i - A_x Y - A_y)$$个 token **并行生成**。由此，最多可并行生成$$\left\lfloor \frac{XY}{A_x Y + A_y} \right\rfloor$$个 token，相比每次仅生成一个 token 的标准自回归注意力，显著提升了并行度并加速了推理。

### 4.6.2 <span style="color: rgb(36,91,219); background-color: inherit">Imagen Video</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">整体生成链路</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">七级级联如何把难题拆开</span>**

Imagen Video 不直接在高清、长序列的像素空间里完成一次生成，而是把任务拆成 **`7`** 个连续扩散模型。冻结的 T5-XXL 先提供文本表示；Base Video Model 生成 **`16×40×24`**、**`3 FPS`** 的低清视频，之后交替经过 **`3`** 个 Temporal Super-Resolution 与 **`3`** 个 Spatial Super-Resolution，最终得到 **`128`** 帧、**`1280×768`**、**`24 FPS`**、约 **`5.3 秒`** 的视频。

> **<span style="color: rgb(36,91,219); background-color: inherit">级联结构的核心取舍</span>**
>
> <span style="color: rgb(100,37,208); background-color: inherit">Base Model 负责“画什么、怎么动”，TSR 负责“中间帧怎么补”，SSR 负责“细节怎么长出来”。</span>每一级只解决一个相对受控的问题，因此训练更稳定，也能把不同分辨率的算力分开配置。代价是链路长，而且早期阶段的构图或运动错误会被后续放大。

![级联链路交替提高帧率与空间分辨率，最终生成 128 帧、1280×768、24 FPS 视频。](../../images/视觉多模态讲义（下）-imagen_pipeline.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">文本条件贯穿全部分辨率</span>**

文本只在低清阶段出现会带来一个问题：超分模块可能把错误纹理放大，却无法根据 Prompt 修正对象属性。这里每一级都通过 Cross-Attention 接收同一组文本 Embedding，使“红色汽车”“水彩风格”“从左向右奔跑”等语义在补帧和增清时仍然有效。T5-XXL 保持冻结，训练重点集中在视频扩散网络本身。

* **<span style="color: rgb(36,91,219); background-color: inherit">Video U-Net：把空间计算和时间计算分开</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">先处理单帧，再沿时间轴混合</span>**

Video U-Net 沿用图像 U-Net 的多尺度 Encoder-Decoder 结构，但在空间模块之后增加时间模块。空间卷积与 Spatial Attention 先在每一帧内部建模物体、纹理和布局；时间模块再把同一空间位置在不同帧上的特征连起来。<span style="color: rgb(100,37,208); background-color: inherit">这是一种 factorized space-time 设计：不把整个视频一次性展平成超长 Token 序列，而是让空间与时间各自承担更合适的计算。</span>

Base Model 面对的是低分辨率但完整时长的视频，使用 Temporal Attention 捕捉跨帧的长距离关系。进入 TSR 和 SSR 后，输入已经有较强的结构先验，主要改用 Temporal Convolution 维持局部连续性并节省显存；最高分辨率 SSR 甚至移除 Spatial Attention，改成全卷积网络，避免注意力在大画布上的二次方开销。

![Video U-Net 先在每帧内做空间计算，再沿时间轴混合特征；Base 使用 Temporal Attention，超分阶段主要使用 Temporal Convolution。](../../images/视觉多模态讲义（下）-imagen_unet.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">超分模型怎样接收上一级结果</span>**

每个超分模型先把上一级视频插值到目标尺寸或目标帧率，再与当前时间步的 noisy input 沿 Channel 维拼接。这样扩散网络不是从零生成，而是在“已有低清结果 + 当前噪声状态”的条件下恢复目标视频。训练时会额外给条件视频加噪，噪声强度也作为条件输入模型。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：为什么要给条件视频加噪
>
> 训练阶段的低清条件来自真实视频下采样，通常比推理阶段上一级生成结果更干净。若完全不处理，超分模型会在训练时依赖“过于完美”的条件，部署时一遇到生成噪点就失稳。Noise Conditioning Augmentation 主动制造这种误差，使各级模型能够独立训练，同时缩小训练与推理的输入分布差距。

* **<span style="color: rgb(36,91,219); background-color: inherit">扩散目标与采样</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">v-prediction 统一不同噪声区间</span>**

连续时间扩散过程使用 cosine noise schedule。与直接预测噪声 ε 或干净样本 x 不同，网络预测由两者组合得到的 velocity：

$$v_t=\alpha_t\epsilon-\sigma_t x$$

在低噪声与高噪声区间，v 的数值尺度更均衡，便于同一网络覆盖完整扩散轨迹。七个阶段都采用这一参数化，因此不同分辨率模型可以复用相近的训练和采样逻辑。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Classifier-Free Guidance 与动态阈值</span>**

训练时随机丢弃文本条件，得到同一个网络的 conditional 与 unconditional 分支；推理时用两者差值把结果推向 Prompt：

$$\tilde{x}_{\theta}(z_t,c)=(1+w)\hat{x}_{\theta}(z_t,c)-w\hat{x}_{\theta}(z_t)$$

Guidance 越大，文本一致性通常越强，但像素也更容易饱和。Dynamic Thresholding 会按当前样本的分位数裁剪并重新缩放预测结果。Base Model 和前两个超分阶段还使用 Oscillating Guidance，在高 Guidance 与低 Guidance 之间交替：高值强化语义，低值缓解过饱和。更高分辨率阶段不继续使用这项技巧，因为容易引入可见闪烁和纹理伪影。

3. **<span style="color: rgb(36,91,219); background-color: inherit">Progressive Distillation 缩短七级采样</span>**

原始级联若每一级都运行大量 DDIM 步，累积延迟会非常高。蒸馏先把 conditional 与 unconditional 的 Guidance 结果吸收到一个学生模型中，再反复让学生用一半步数逼近教师的两步更新。经过多轮迭代，每个子模型都能压到 **`8 步`** 左右，视觉差异仍较小。

![左侧为原始级联采样结果，右侧为每级 8 步的 distillation 结果；主体和构图基本保持。](../../images/视觉多模态讲义（下）-imagen_distillation.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">训练组织与边界</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像与视频联合训练</span>**

静态图像被视为只有一帧的视频。遇到图像样本时，时间卷积分支和跨帧 Attention 会被 Mask，空间 U-Net 仍然正常更新。这样可以利用规模更大的图像数据补足物体、风格与构图知识，同时不让不存在的时间维参与梯度。

2. **<span style="color: rgb(36,91,219); background-color: inherit">各级独立训练</span>**

条件噪声增强使七个阶段不必端到端联合训练，工程上可以按分辨率拆分数据、显存和训练任务。但这也意味着整条链路缺少统一的全局目标：后一级擅长把局部细节做清楚，却无法彻底改正 Base Model 的错误动作、错误计数或物体消失。

* **<span style="color: rgb(36,91,219); background-color: inherit">连续时间扩散与两类采样器</span>**

每个级联模型都把干净视频 *x* 映射到连续时间 *t* 的带噪状态。cosine schedule 决定信号系数与噪声系数，且二者满足平方和为 1：

$$q(z_t\mid x)=\mathcal{N}(z_t;\alpha_t x,\sigma_t^2 I),\qquad z_t=\alpha_t x+\sigma_t\epsilon$$

$$\alpha_t^2+\sigma_t^2=1,\qquad \lambda_t=\log\frac{\alpha_t^2}{\sigma_t^2}$$

<span style="color: rgb(100,37,208); background-color: inherit">连续时间写法让训练可以随机抽取任意噪声强度，而不是把网络绑定到某个固定的离散步数。</span>七个模型虽然分辨率和输入条件不同，但共享同一套时间条件和扩散参数化。

标准噪声目标要求网络从带噪状态 z\_t 中恢复采样噪声：

$$\mathcal{L}_{\epsilon}=\mathbb{E}_{t,x,\epsilon}\left[\left\lVert\epsilon-\hat{\epsilon}_{\theta}(z_t,t,c)\right\rVert_2^2\right]$$

Imagen Video 实际使用 v-parameterization。由网络输出的 v\_t 可以同时解出干净样本和噪声，因此在接近纯数据与接近纯噪声的两端都保持较稳定的目标尺度：

$$v_t=\alpha_t\epsilon-\sigma_t x,\qquad \hat{x}=\alpha_t z_t-\sigma_t\hat{v}_{\theta},\qquad \hat{\epsilon}=\sigma_t z_t+\alpha_t\hat{v}_{\theta}$$

训练目标确定后，采样端有两条路径。Ancestral sampler 按学习到的反向条件分布逐步采样，每一步都可以重新注入随机噪声，适合保留随机性；DDIM 把同一个训练模型改写为近似确定性的轨迹，同一初始噪声会给出更稳定的结果，也更适合做 progressive distillation。最终的蒸馏模型仍可接入随机采样器，在少步数下恢复一定的样本多样性，而不是把整个生成过程锁死为单一路径。

* **<span style="color: rgb(36,91,219); background-color: inherit">级联条件、联合训练与蒸馏细节</span>**

生成侧由 **`1`** 个 Base、**`3`** 个 SSR 和 **`3`** 个 TSR 组成，扩散网络总参数量约 **`11.6B`**；冻结的 T5-XXL 只负责文本编码，不计入这组可训练扩散参数。每一级一次性生成完整的视频块，并不是逐帧自回归，因此不会把前一帧的采样误差机械地滚到下一帧。

级联开始放大时，SSR 先对低分辨率条件做双线性空间上采样，再与目标分辨率的 noisy video 沿通道维拼接；TSR 则在时间轴上重复已有帧，或在待补位置放入空帧，再与目标帧率的 noisy video 拼接。两者都把条件帧显式送进 U-Net，因此超分任务不是“重新生成”，而是受上一级结果约束的条件去噪。

分辨率走到最后一级后，SSR 改用全卷积结构，训练时只取高清目标视频中的随机低分辨率空间裁剪。卷积权重与画布绝对尺寸无关，推理时可以把同一网络滑到完整的 **`1280×768`** 画面上。这一步把显存消耗从“整张高清帧”降到“局部高清块”，也是级联能够落到最终分辨率的重要工程条件。

各级能够独立训练，关键在条件噪声增强。训练时随机采样条件噪声的 SNR，并把噪声级别一并输入超分网络；推理时不再随机，而是给对应级联使用固定 SNR，典型取值为 **`3`** 或 **`5`**。固定非零噪声相当于主动削弱上一级的局部瑕疵，避免下一级把伪影当成必须忠实保留的高置信度结构。

静态图像也沿用同一接口。一个训练样本需要固定长度的视频块时，可以把多张彼此独立的图像打包到同一序列。图像样本经过空间模块，但绕过 temporal convolution，并用 attention mask 阻止不同图像之间发生时间注意力。这样既保持批处理形状一致，又不会把两张无关图片误学成连续动作。

训练链路确定以后，推理端再控制 guidance 与步数。采样开始阶段先使用高 guidance，之后在约 **`15`** 与约 **`1`** 之间振荡：高值拉紧文本语义，低值释放过度饱和。该策略只用在 Base 和前两个超分阶段，空间分辨率超过 **`80×48`** 后停止。蒸馏则先把 CFG 的双分支合成单一 guided model，再反复把两次 DDIM 更新压成一次；完成步数折半后切回随机采样器，每个阶段最终约用 **`8 步`**。

> **<span style="color: rgb(36,91,219); background-color: inherit">主要局限</span>**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">部署复杂</span>**：需要维护 **`7`** 套扩散权重与采样配置，显存调度和故障定位都比单一主干困难。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">误差逐级传递</span>**：低清阶段一旦确定了错误的主体或运动，超分阶段通常只能清晰地重现错误。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">缺少端到端效率</span>**：蒸馏降低了每一级步数，但没有消除多模型串行执行本身的时延。

### 4.6.3 <span style="color: rgb(36,91,219); background-color: inherit">Stable Video Diffusion</span>

在 SVD 之前，有许多视频生成模型都是在图像生成模型 SD 的基础上，添加和视频时序相关的模块，并在小规模高质量视频数据集上微调新模型。而 SVD 作者认为，该领域在训练方法及精制数据集的策略上并未达成统一。<span style="color: rgb(100,37,208); background-color: inherit">这篇文章的主要贡献，也正是提出了一套训练方法与精制数据集的方</span>法。具体而言，<span style="color: rgb(100,37,208); background-color: inherit">SVD 的训练由三个阶段组成：文生图预训练、视频预训练、高质量视频微调</span>。同时，SVD 提出了一种系统性的数据精制流程，包含数据的标注与过滤这两部分的策略。论文会分享诸多的实验成果，包括验证精心构建的数据集对生成高质量视频的必要性、探究视频预训练与微调这两步的重要性、展示基础模型如何为图生视频等下游任务提供强大的运动表示、演示模型如何提供多视角三维先验并可以作为微调多视角扩散模型的基础模型在一轮神经网络推理中同时生成多视角的图片。

总结一下，SVD 并没有强调在模型设计或者采样算法上的创新，而主要宣传了该工作在数据集精制及训练策略上的创新。对于大部分普通研究人员来说，由于没有训练大视频模型的需求，该文章的很多内容都价值不大。这里来大致过一遍这篇文章的主要内容。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Video-LDM 与 SVD</span>**

在阅读正文之前，我们先来回顾一下此前视频生成模型的开发历程，并重点探究 SVD 的模型架构——Video LDM 的具体组成。<span style="color: rgb(100,37,208); background-color: inherit">绝大多数工作在训练一个基于扩散模型的视频生成模型时，都是在预训练的 SD 上加入时序模块，如 </span> <span style="color: rgb(220,155,4); background-color: inherit">3D 卷积</span> <span style="color: rgb(100,37,208); background-color: inherit">，并通过微调把一个图像生成模型转换成视频生成模型</span>。由于 SD 是一种 LDM (Latent Diffusion Model)，所以这些视频模型都可以归类为 Video-LDM。所谓 LDM，就是一种<span style="color: rgb(100,37,208); background-color: inherit">先生成压缩图像，再用解码模型把压缩图像还原成真实图像的模型。而对于视频，Video-LDM 则会先生成边长压缩过的视频，再把压缩视频还原</span>。

![](../../images/视觉多模态讲义（下）-3c5ab2a01c3163d298c44f2c29f52f58.jpeg)

虽然 Video-LDM 严格上来说是一个视频扩散模型的种类，但大家一般会用 Video LDM （没有横杠） 来指代 [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2304.08818)*&#x20;*&#x8FD9;篇工作。这篇论文已在 CVPR 2023 上发布，两个主要作者正是前一年在 CVPR 上发表 SD 论文的主要作者，也是现在这篇 SVD 论文的主要作者。论文中也讲到，SVD 完全复用了 Video LDM 的结构。为了了解 SVD 的模型结构，我们再来回顾一下 Video LDM 的结构。

在 SD 的基础上，Video LDM 做对模型结构了两项改动：**<span style="color: rgb(100,37,208); background-color: inherit">在扩散模型的去噪模型 U-Net 中加入时序层、在对图像压缩和解压的 VAE 的解码器中加入时序层</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">添加时序层</span>**

Video LDM 在 U-Net 中加入时序层的方法与多数同期方法相同，是在每个原来处理图像的空间层后面加上处理视频的时序层。Video LDM 加入的时序层包括 3D 卷积层与时序注意力层。这些新模块本身不难理解，但我们需要着重关注这些新模块是怎么与原模型兼容的。

要兼容各个模块，其实就是要兼容数据的形状。本来，图像生成模型的 U-Net 的输入形状&#x4E3A;**`B C H W`**，分别表示图像数、通道数、高、宽。而视频数据的形状&#x662F;**`B T C H W`** ，即视频数、视频长度、通道数、高、宽。要让视频数据复用之前的图像模型的结构，只要把数据前两维合并，变&#x6210;**`(B T) C H W`**&#x5373;可。这种做法就是<span style="color: rgb(100,37,208); background-color: inherit">把</span>$$B$$<span style="color: rgb(100,37,208); background-color: inherit">组长度为</span>$$T$$<span style="color: rgb(100,37,208); background-color: inherit">的视频看成了</span>$$B \cdot T$$<span style="color: rgb(100,37,208); background-color: inherit">张图片</span>。

对于之前已有的空间层，只要把数据形状变&#x6210;**`(B T) C H W`**&#x5C31;没问题了。而 SVD 又新加入了两种时序层：<span style="color: rgb(100,37,208); background-color: inherit">3D 卷积和时序注意力</span>。我们来看一下数据是怎么经过这些新的时序层的。2D 卷积会&#x5BF9;**`B C H W`**&#x7684;数据的后两个高、宽维度做卷积。类似地，3D 卷积会对数据最后三个时间、高、宽维度做卷积。所以，过 3D 卷积前，要把形状&#x4ECE;**`(B T) C H W`**&#x53D8;&#x6210;**`B C T H W`**，做完卷积再还原。

接下来我们来看新的时序注意力。这个地方稍微有点难理解，我们从最简单的注意力开始一点一点学习。最早的 NLP 中的注意力层的输入形状&#x4E3A;**`B L C`**，表示数据数、token 长度、token 通道数。 $$L$$这一维最为重要，它表示了$$L$$个 token 之间互相交换信息。如果把其拓展成图像空间注意力，则 token 表示图像的每一个像素。在这种注意力层中，$$L$$&#x662F;**`(H W)`**，**`B C H W`**&#x7684;数据会被转换&#x6210;**`B (H W) C`**&#x8F93;入进注意力层。这表示同一组图像中，每个像素两两之间交换信息。而让视频数据过空间注意力层时，只需要把$$B$$换&#x6210;**`(B T)`**&#x5373;可，即把数据形状&#x4ECE;**`(B T) C H W`**&#x53D8;&#x4E3A;**`(B T) (H W) C`**。这表示同一组、同一帧的图像的每个像素之间，两两交换信息。

在 SVD 新加入的时序注意力层中，token 依旧指代是某一组、某一帧上的一个像素。然而，这次我们不是让同一张图像的像素互相交换信息，而是<span style="color: rgb(100,37,208); background-color: inherit">让不同时刻的像素互相交换信息</span>。因此，这次 token 长度$$L$$是$$T$$，它表示要像素在时间维度上交换信息。这样，在视频数据过时序层里的自注意力层时，要把数据形状&#x4ECE;**`(B T) C H W`**&#x53D8;&#x6210;**`(B H W) T C`** 。这表示每一组、图像每一处的像素独立处理，它们仅与同一位置不同时间的像素进行信息交换。

![](../../images/视觉多模态讲义（下）-b46ede7bf457e3ad6ab40ded0ef8aa6b.jpeg)

* **<span style="color: rgb(36,91,219); background-color: inherit">微调 VAE 解码器</span>**

Video LDM 的另一项改动是修改了图像压缩模型 VAE 的解码器。具体来说就是<span style="color: rgb(100,37,208); background-color: inherit">先在 VAE 的解码器中加入类似的时序层，并在 VAE 配套的 GAN 的判别器里也加入了时序层，随后开始微调。在微调时，编码器不变，仅训练解码器和判别器</span>。

![](../../images/视觉多模态讲义（下）-78e7ce5db5ff8dae49a3c13165439e05.jpeg)

以上就是 Video LDM 的模型结构。SVD 对其没有做任何更改，所以也没有在论文里对模型结构做详细介绍。稍有不同的是，Video LDM 仅微调了新加入的模块，而 SVD 在加入新模块后对模型的所有参数都进行了重新训练。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练细节</span>**

SVD 分四节介绍了模型训练过程。<span style="color: rgb(100,37,208); background-color: inherit">第一节介绍了数据精制的过程，后三节分别介绍了训练的三个阶段：文生图预训练、视频预训练、高质量视频微调</span>。

获取了一个大规模视频数据集后，SVD 的数据精制主要由预处理和标注这两步组成。由于视频生成模型主要关注生成同一个场景的视频，而不考虑转场的问题，每段训练视频也应该尽量只包含一个场景。为此，<span style="color: rgb(100,37,208); background-color: inherit">预处理主要是在用一些自动化视频剪切工具把收集到的视频进一步切成连续的片段</span>。经切片后，<span style="color: rgb(100,37,208); background-color: inherit">视频片段数变为原来的4倍。标注主要是给视频加上文字描述，以训练一个文生视频的模型</span>。SVD 在添加文字描述时用到了多个标注模型，并使用 LLM 来润色描述。经预处理和标注后，得到的数据集被称作 **<span style="color: rgb(216,57,49); background-color: inherit">LVD</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">arge </span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ideo </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">ataset）</span>。

SVD 数据精制的细节中，比较值得注意的是有关视频帧数的处理。由于开发团队发现视频数据的播放速度快慢不一，于是他们<span style="color: rgb(100,37,208); background-color: inherit">使用光流预测模型来大致估计每段视频的播放速度（以帧率 FPS 表示），并将视频的帧率也作为标注</span>。这样，在训练时，视频的帧率也可以作为一种约束信息。这样的好处是，在我们在生成视频时，可以用该约束来指定视频的播放速度。

之后我们来看 SVD 模型训练的三个阶段。对于第一个文生图预训练阶段，论文没有对模型结构做过多修改，因为他们在这一步使用了之前训练好的 SD 2.1。不过，SVD 在这一步做了一个非常重要的改进：**<span style="color: rgb(100,37,208); background-color: inherit">SVD 的噪声调度器从原版的 DDPM 改成了 EDM，采样方法也改成了 EDM 的</span>**。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：EDM 的论文全称为 [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/pdf/2206.00364)*&#x20;*。这篇论文用一种概括性较强的数学模型统一表示了此前各种各样的扩散模型结构，并提出了改进版模型的训练及采样策略。简单来说，<span style="color: rgb(100,37,208); background-color: inherit">EDM 把扩散模型不同时刻的噪声强度表示成</span>$$\sigma_t$$<span style="color: rgb(100,37,208); background-color: inherit">，它表示在</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">时刻时，对来自数据集的图像加了标准差为</span>$$\sigma_t$$<span style="color: rgb(100,37,208); background-color: inherit">的高斯噪声</span>$$\mathcal{N}(\mathbf{0}, \sigma_t^2\mathbf{I})$$<span style="color: rgb(100,37,208); background-color: inherit">。一开始，对于没加噪声的图像，</span>$$\sigma_0=0$$<span style="color: rgb(100,37,208); background-color: inherit">。对于最后一个时刻</span>$$T$$<span style="color: rgb(100,37,208); background-color: inherit">的图像，</span>$$\sigma_T$$<span style="color: rgb(100,37,208); background-color: inherit">要足够大，使得原图像的内容被完全破坏</span>。这里时刻$$0$$与时刻$$T$$的定义与 DDPM 论文相同，与 EDM 论文相反。

有了这样一种统一的表示后，EDM 对扩散模型的训练和采样都做了不少改进。这里我们仅关注其中最重要的一条改进：将离散噪声改进成连续噪声。<span style="color: rgb(216,57,49); background-color: inherit">原来 DDPM 的去噪模型会输入时刻</span>$$t$$<span style="color: rgb(216,57,49); background-color: inherit">这个参数</span>。EDM 论文指出，$$t$$<span style="color: rgb(100,37,208); background-color: inherit">实际上表示了噪声强度</span>$$\sigma_t$$<span style="color: rgb(100,37,208); background-color: inherit">，应该把</span>$$\sigma_t$$<span style="color: rgb(100,37,208); background-color: inherit">输入进模型。与其用离散的</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">训练一个只认识离散噪声强度的去噪模型，不如训练一个认识连续噪声强度</span>$$\sigma$$<span style="color: rgb(100,37,208); background-color: inherit">的模型</span>。这样，在采样$$n$$步时，我们不再是选择离散去噪时&#x523B;**`[timestep[n], timestep[n - 1], ..., 0]`**，而是可以选择连续噪声强&#x5EA6;**`[sigma[n], sigma[n - 1], ..., 0]`** 。这样采样更灵活，效果也更好。在第一个训练阶段中，SVD 照搬了 EDM 的这种训练方法，改进了原来的 DDPM。SVD 的默认采样策略也使用了 EDM 的 。我们会在之后的代码实践文章中详细学习这种新采样方法。

对于第二个视频预训练阶段，或许是因为视频模型和图像模型的训练过程毫无区别，论文的介绍重点依然放在了这一阶段的数据处理上，而没有强调训练方法上的创新。简单来看，<span style="color: rgb(100,37,208); background-color: inherit">这一阶段的目标是得到一个过滤后的高质量数据集</span>**`LVD-F`**。为了找到这样一种合适的过滤方案，开发团队先用排列组合生成了大量的过滤方案：对每类指标（文本视频匹配度、美学分数、帧率等）都设&#x7F6E;**`12.5%`**, **`25%`**&#x6216;**`50%`**&#x7684;过滤条件，然后不同指标的条件之间排列组合。之后，开发团队抽取原数据集的一个子集 LVD-10M，用各个方案得到过滤后的视频子集 LVD-10-F。最后，用这样得到的子数据集分别训练模型，比较模型输出的好坏，以决定在完整数据集上使用的最优过滤方案。

在第三个阶段，参考以往多阶段训练图像模型的经验，SVD 也在另一个小而精的视频数据集上进行微调。此数据集的获取方法并没有在论文中给出，大概率是人工手动收集并标注。

* **<span style="color: rgb(36,91,219); background-color: inherit">总结</span>**

Stable Video Diffusion 是在文生图模型 Stable Diffusion 2.1 的基础上添加了和 Video LDM 相同的视频模块微调而成的一套视频生成模型。SVD 的论文主要介绍了其精制数据集的细节，并展示了几个微调基础模型能实现的应用。<span style="color: rgb(46,161,33); background-color: inherit">通过微调基础低分辨率文生视频模型，SVD 可以用于高分辨率文生视频、高分辨率图生视频、视频插帧、多视角生成</span>。

### 4.6.4 <span style="color: rgb(36,91,219); background-color: inherit">Sora</span>

Sora 的关键技术是使用&#x4E86;**`DiT`**，包含三个部分：

> * 一个时空压缩器将原始视频映射到潜在空间
>
> * 一&#x4E2A;**`ViT`**&#x5904;理分词后的潜在表示，并输出去噪后的潜在表示
>
> * 一个类&#x4F3C;**`CLIP`**&#x7684;条件机制接收经 LLM 增强的用户指令以及可能的视觉 prompt ，以引导扩散模型生成具有特定风格或主题的视频

经过多次去噪步骤后，生成视频的潜在表示，然后通过相应的解码器映射回像素空间。

![](../../images/视觉多模态讲义（下）-image-125.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">数据预处理  </span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">可变时长、分辨率与宽高比  </span>**

Sora 的一个显著特点是能够在原始尺寸下训练、理解并生成视频和图像，如下图。传统方法<span style="color: rgb(216,57,49); background-color: inherit">一般会对视频进行缩放、裁剪或调整宽高比，以适应统一的标准</span>，一般是固定低分辨率下的方形短片段。这些样本<span style="color: rgb(216,57,49); background-color: inherit">以较大的时间步长生成，并依赖于单独训练的帧插值和分辨率渲染模型，导致视频内部不一致</span>。 &#x20;

![](../../images/视觉多模态讲义（下）-image-124.png)

Sora 是首个利用视觉数据多样性的模型，<span style="color: rgb(46,161,33); background-color: inherit">能够以多样化的视频和图像格式进行采样，涵盖从宽屏</span>$$1920\times1080\text{p}$$<span style="color: rgb(46,161,33); background-color: inherit">视频到竖屏</span>$$1080\times1920\text{p}$$<span style="color: rgb(46,161,33); background-color: inherit">视频以及其间所有格式，且不损害其原始尺寸</span>。在原始尺寸数据上训练显著提升了生成视频的构图与取景质量。通过保持原始宽高比，Sora 能够实现更自然、连贯的视觉叙事。如右图，<span style="color: rgb(46,161,33); background-color: inherit">Sora 生成的视频取景更佳，确保主体完整呈现在画面中，而非因方形裁剪而出现部分截断的情况</span>。 &#x20;

![裁剪为方形训练                   按原始尺寸训练](../../images/视觉多模态讲义（下）-image-120.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">视觉编码 </span>**

为了有效处理包括图像和视频在内的多样化视觉输入，即时长、分辨率和宽高比各异，关键方法是<span style="color: rgb(100,37,208); background-color: inherit">将所有形式的视觉数据转换为统一表示，从而便于生成模型大规模训练</span>。Sora 通过对视频进行分&#x5757;**`patchify`**&#x6765;做：首先将视频压缩到低维潜在空间，然后将该表示分解为时空&#x5757;**`spacetime patches`**。&#x20;

Sora 的压缩网络主要是降低输入数据的维度，输出在时间和空间上均被压缩的潜在表示，如下图。编码器基&#x4E8E;**`VAE`**&#x6216;**`VQ-VAE`**。但<span style="color: rgb(216,57,49); background-color: inherit">若不采用缩放或裁剪，VAE 很难将任意尺寸的视觉数据映射到统一且固定大小的潜在空间</span>。这里有两种不同的实现方案可以应对该问题：

![](../../images/视觉多模态讲义（下）-image-122.png)

**<span style="color: rgb(222,120,2); background-color: inherit">空间分块压缩</span>**

<span style="color: rgb(100,37,208); background-color: inherit">将视频帧转换为固定大小的 patch ，类似于</span>**`ViT`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`MAE`**<span style="color: rgb(100,37,208); background-color: inherit">中的方法，再将其编码到潜在空间</span>，如右图。这<span style="color: rgb(100,37,208); background-color: inherit">适用于处理不同分辨率和宽高比的视频</span>，因为它通过对单个 patch 进行处理来编码完整帧。随后，这些空间 token <span style="color: rgb(100,37,208); background-color: inherit">按时间顺序组织，形成时空潜在表示</span>。这里有多个因素需要考虑：&#x20;

![](../../images/视觉多模态讲义（下）-image-123.png)

> * **<span style="color: rgb(36,91,219); background-color: inherit">时间维度可变性</span>**：由于<span style="color: rgb(216,57,49); background-color: inherit">训练视频时长各异，潜在表示的时间维度无法固定</span>。解决方法一般是<span style="color: rgb(100,37,208); background-color: inherit">采样固定帧数，对于极短视频可能需要填充或时间插值，或定义一个统一的超长输入长度以供后续处理</span>&#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">训练压缩网络</span>**：对于高分辨率视频，可借助现有预训练压缩网络，&#x5982;**`Stable Diffusion`**<span style="color: rgb(220,155,4); background-color: inherit">中的</span>**`VAE`**；但 Sora <span style="color: rgb(100,37,208); background-color: inherit">采用潜在扩散模型的训练方式从头训练了压缩网络，包含编码器和解码器</span>。这些编码器能高效压缩大尺寸 patch，便于大规模数据管理&#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">时间信息聚合</span>**：之前的操作主要关注空间分块压缩，<span style="color: rgb(100,37,208); background-color: inherit">模型内部需额外机制来聚合时间信息，这对捕捉动态变化很重要</span>

**<span style="color: rgb(222,120,2); background-color: inherit">时空分块压缩</span>**

<span style="color: rgb(46,161,33); background-color: inherit">同时封装视频数据的空间与时间维度，提供更全面的表示，不仅分析静态帧，还考虑帧间运动与变化，从而捕捉视频的动态特性</span>。这里使用 3D 卷积来整合。下图展示时空分块压缩与纯空间分块的对比。与空间分块压缩类似，<span style="color: rgb(216,57,49); background-color: inherit">若采用预设的卷积核参数，如固定核大小、步长和输出通道数，由于输入视频特性不同，潜在空间维度仍会变化，主要受视频时长和分辨率多样性驱动</span>。这其实前面提到的空间分块策略同样适用且有效。Sora 采用时空分块，因其<span style="color: rgb(46,161,33); background-color: inherit">易于实现，且能通过高信息密度 token 有效缩短上下文长度，降低后续时间信息建模的复杂度</span>。

![不同视频分块方法的对比。左图是空间分块方法仅采样 n\_t 帧，并按照 ViT 的方式对每一帧二维图像独立进行 Embedding。右图是时空分块方法，从整个时空输入体中提取非重叠或重叠的 tubelets，并对其进行线性 Embedding。](../../images/视觉多模态讲义（下）-image-119.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">时空潜在块  </span>**

压缩网络仍存在一个关键问题是：<span style="color: rgb(100,37,208); background-color: inherit">在将 patch 输入扩散 Transformer 的输入层之前，如何处理潜在空间维度的可变性，即不同视频类型产生的潜在特征 patch 或 patch 的数量差异</span>。<span style="color: rgb(100,37,208); background-color: inherit">Sora 采用</span> **<span style="color: rgb(216,57,49); background-color: inherit">PNP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">atch </span>**<span style="color: rgb(216,57,49); background-color: inherit">n</span>**<span style="color: rgb(216,57,49); background-color: inherit">’ </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">ack）</span> <span style="color: rgb(100,37,208); background-color: inherit">方法，将来自不同图像的多个 patch 打包到单一序列中</span>，如下图。这其实和自然语言处理中&#x7684;**`example packing`**&#x7C7B;似，通过丢弃部分 token 来高效训练可变长度输入。这里分块与 token 编码需在压缩网络中完成；然后 Sora 对潜在表示进一步分块以生成 Transformer token。这里需解决两个问题：<span style="color: rgb(100,37,208); background-color: inherit">如何紧凑地打包这些 token，以及如何控制哪些 token 应被丢弃</span>。 &#x20;

![](../../images/视觉多模态讲义（下）-image-117.png)

> * **<span style="color: rgb(36,91,219); background-color: inherit">问题 1</span>**：采用简单的贪心策略，<span style="color: rgb(100,37,208); background-color: inherit">将样本依次加入首个仍有足够剩余空间的序列；当无法再容纳新样本时，用填充 token 补全序列，以满足批处理所需的固定序列长度</span>。这种简单打包算法<span style="color: rgb(216,57,49); background-color: inherit">可能导致大量填充，具体取决于输入长度的分布</span>。当然也可<span style="color: rgb(100,37,208); background-color: inherit">通过调节采样分辨率与帧数、优化序列长度并限制填充，以实现高效打包</span>。 &#x20;
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">问题 2</span>**：丢弃相似 token，或使用 PNP 的方法应用丢弃率调度器。但三维一致&#x6027;**`3D Consistency`**&#x662F; Sora 的重要特性之一，丢弃 token 可能在训练中忽略细粒度细节。因此 Sora 应该<span style="color: rgb(100,37,208); background-color: inherit">采用了超长上下文窗口，将所有视频 token 全部打包</span>，长视频的时空潜在块可打包进一个序列，而多个短视频的patch 则拼接至另一序列，但是这在计算上代价高昂，例如<span style="color: rgb(220,155,4); background-color: inherit">多头注意力算子的计算复杂度随序列长度呈平方增长</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像扩散 Transformer</span>**

传统扩散模型主要采用包含下采样和上采样模块的卷积 U-Net 作为去噪网络主干。但 U-Net 架构并非扩散模型表现好的关键。<span style="color: rgb(100,37,208); background-color: inherit">通过引入更灵活的 Transformer 架构，基于 Transformer 的扩散模型能够利用更多训练数据和更大规模的模型参数</span>，DiT 和 U-ViT 是第一次将 Vision Transformer 应用于潜在扩散模型的工作。 &#x20;

![](../../images/视觉多模态讲义（下）-image-114.png)

**<span style="color: rgb(222,120,2); background-color: inherit">DiT</span>**

<span style="color: rgb(100,37,208); background-color: inherit">采用多头自注意力层与逐点前馈网络，并穿插</span>**`LayerNorm`**<span style="color: rgb(100,37,208); background-color: inherit">和缩放层</span>。此外，DiT 通过<span style="color: rgb(100,37,208); background-color: inherit">自适应层归一化</span>**`AdaLN`**<span style="color: rgb(100,37,208); background-color: inherit">引入条件信息</span>，并<span style="color: rgb(100,37,208); background-color: inherit">附加一个 MLP 层用于零初始化</span>**`zero-initializing`**，使得每个残差块初始为恒等函数，从而极大提升了训练稳定性。D<span style="color: rgb(46,161,33); background-color: inherit">iT 的可扩展性与灵活性都很好，现已成为扩散模型的新主干架构</span>。 &#x20;

**<span style="color: rgb(222,120,2); background-color: inherit">U-ViT</span>**

<span style="color: rgb(100,37,208); background-color: inherit">所有输入，包括时间步、条件信息和带噪图像块，均被视为 token，并在浅层与深层 Transformer 层之间引入长的跳跃连接</span>。作者表明基于 CNN 的 U-Net 中的下采样与上采样操作并非总是必要的，<span style="color: rgb(46,161,33); background-color: inherit">U-ViT 在图像生成和文本到图像生成任务中取得了创纪录的 FID 分数</span>。 &#x20;

**<span style="color: rgb(216,57,49); background-color: inherit">MDT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">asked </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">iffusion </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformer）</span> <span style="color: rgb(100,37,208); background-color: inherit">将掩码潜在建模引入扩散过程，以显式增强图像合成中对象语义部分之间的上下文关系学习</span>。MDT 在训练过程中<span style="color: rgb(100,37,208); background-color: inherit">引入一个侧插值分支，用于额外的掩码 token 预测任务</span>，从而提升训练效率，并学习强大的上下文感知位置 Embedding 以用于推理。相比 DiT，MDT 实现了更优性能和更快的学习速度。  &#x5728;**`MDTv2`**&#x4E2D;，作者进一步改进 MDT，采用更高效的宏观网络结构：<span style="color: rgb(100,37,208); background-color: inherit">编码器中引入 U 形长短连接，解码器中则从编码器输入引入密集跳跃连接</span>。此外还结合多种训练策略，<span style="color: rgb(100,37,208); background-color: inherit">包括</span>**`Adan`**<span style="color: rgb(100,37,208); background-color: inherit">优化器、</span>**`Min-SNR`**<span style="color: rgb(100,37,208); background-color: inherit">加权、动态掩码率，以及改进的 power-cosine 加权用于无分类器引导</span>**`classifier-free guidance`**。MDTv2 在合成性能和学习速度上均取得显著提升。 &#x20;

![](../../images/视觉多模态讲义（下）-image-115.png)

**<span style="color: rgb(216,57,49); background-color: inherit">DiffiT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Diff</span>**<span style="color: rgb(216,57,49); background-color: inherit">usion V</span>**<span style="color: rgb(216,57,49); background-color: inherit">i</span>**<span style="color: rgb(216,57,49); background-color: inherit">sion </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformers）</span> <span style="color: rgb(100,37,208); background-color: inherit">不再使用</span>**`AdaLN`**<span style="color: rgb(100,37,208); background-color: inherit">进行时间条件建模，而是引入时间依赖自注意力</span>**`TMSA`**<span style="color: rgb(100,37,208); background-color: inherit">模块，以建模采样时间步上的动态去噪行为</span>。此外，DiffiT 采用两种混合层次架构，<span style="color: rgb(100,37,208); background-color: inherit">分别在像素空间和潜在空间中实现高效去噪，并在多种生成任务中达到新的 SOTA 水平</span>。这些研究在图像潜在扩散中采用 Vision Transformer 取得了显著成果，为其他模态的扩散建模铺平了道路。

* **<span style="color: rgb(36,91,219); background-color: inherit">视频扩散 Transformer</span>**<span style="color: rgb(36,91,219); background-color: inherit">  </span>

扩散 Transformer 在文本到视频生成任务中的也十分有用。由于视频具有时间维度，将 DiT 应用于视频领域面临三个问题：

> 1. 如何在空间和时间上将视频压缩到潜在空间以实现高效去噪
>
> 2. 如何将压缩后的潜在表示转换为块并输入 Transformer
>
> 3. 如何处理长程时空依赖并确保内容一致性

**Imagen Video** 采用级联扩散模型结构，<span style="color: rgb(100,37,208); background-color: inherit">包含</span>**`7`**<span style="color: rgb(100,37,208); background-color: inherit">个子模型，分别执行文本条件视频生成、空间超分辨率和时间超分辨率，将文本 prompt 转化为高清视频</span>：

> 1. <span style="color: rgb(100,37,208); background-color: inherit">冻结的</span>**`T5`**<span style="color: rgb(100,37,208); background-color: inherit">文本编码器从输入文本 prompt 生成上下文 Embedding</span>。这些 Embedding 可以帮助视频与文本对齐，并注入包括基础模型在内的所有级联模型
>
> 2. <span style="color: rgb(100,37,208); background-color: inherit">Embedding 被送入基础模型生成低分辨率视频，再由级联扩散模型逐步提升分辨率</span>。基础视频模型和超分辨率模型<span style="color: rgb(100,37,208); background-color: inherit">均采用</span>**`3D U-Net`**<span style="color: rgb(100,37,208); background-color: inherit">架构，以时空分离方式实现</span>：将时间注意力与卷积层与空间对应模块交织，高效捕捉帧间依赖。架构<span style="color: rgb(100,37,208); background-color: inherit">采用</span>**`v-prediction`**<span style="color: rgb(100,37,208); background-color: inherit">参数化以提升数值稳定性，并使用条件增强支持模型间的并行训练</span>。训练过程联合使用图像和视频数据，将每张图像视为单帧以利用更大规模数据集，并采用无分类器引导提升 prompt 保真度。
>
> 3. <span style="color: rgb(100,37,208); background-color: inherit">应用渐进蒸馏降低采样计算开销，同时保持感知质量</span>。这些方法的结合<span style="color: rgb(46,161,33); background-color: inherit">使 Imagen Video 不仅能生成高保真视频，还展现出卓越的可控性</span>，例如<span style="color: rgb(220,155,4); background-color: inherit">生成多样化视频、文字动画及多种艺术风格内容</span>。

![左图为级联扩散模型，包含一个基础扩散模型和六个在空间和时间维度上运行的上采样模型，构成级联采样流程。文本 Embedding 被注入到所有扩散模型中；右图为视频 U-Net 时空可分离模块。空间操作在各帧上独立进行，并共享参数；而时间操作则混合各帧的激活值，为节省内存，仅在基础模型中使用时间注意力机制。](../../images/视觉多模态讲义（下）-image-118.png)

**Video LDM** 将 2D 潜在扩散模型LDM扩展为视频潜在扩散模型。<span style="color: rgb(100,37,208); background-color: inherit">通过在现有空间层之间插入</span>**`post-hoc temporal layers`**<span style="color: rgb(100,37,208); background-color: inherit">，分别应用于 U-Net 主干和 VAE 解码器，使模型学会对齐各帧</span>。这些时间层<span style="color: rgb(100,37,208); background-color: inherit">在编码后的视频数据上训练，而空间层保持冻结，从而可利用大规模图像数据集进行预训练</span>。LDM 的解码器<span style="color: rgb(100,37,208); background-color: inherit">经微调以在像素空间中实现时间一致性，并对扩散模型上采样器进行时间对齐以提升空间分辨率</span>。为生成超长视频，模型被训练为根据若干上下文帧预测未来帧，从而在采样时支持无分类器引导。为实现高时间分辨率，<span style="color: rgb(100,37,208); background-color: inherit">视频合成分为</span>**<span style="color: rgb(100,37,208); background-color: inherit">关键帧生成</span>**<span style="color: rgb(100,37,208); background-color: inherit">与</span>**<span style="color: rgb(100,37,208); background-color: inherit">关键帧间插值</span>**<span style="color: rgb(100,37,208); background-color: inherit">两个阶段</span>。沿用级联 LDM 思路，扩散模型将 Video LDM 输出进一步放大四倍，在保持时间一致性的同时确保高空间分辨率，这以计算高效的方式生成全局一致的长视频。此外<span style="color: rgb(46,161,33); background-color: inherit">仅在训练时间对齐层即可将预训练图像</span>**`LDM`**<span style="color: rgb(46,161,33); background-color: inherit">转化为文本到视频模型，生成分辨率高达</span>$$1280\times2048$$<span style="color: rgb(46,161,33); background-color: inherit">像素的视频</span>。

![额外的时间层。通过插入学习将帧对齐为时间上一致序列的时间层，一个预训练的潜在扩散模型被转化为视频生成器。在优化过程中，图像主干网络参数保持固定，仅训练时间层的参数。](../../images/视觉多模态讲义（下）-image-121.png)

![视频 LDM 堆栈。首先生成稀疏的关键帧，然后使用相同的 LDM 进行两次时间插值，以实现高帧率。将潜在视频解码到像素空间，并可选地应用一个视频上采样扩散模型。](../../images/视觉多模态讲义（下）-image-113.png)

**<span style="color: rgb(100,37,208); background-color: inherit">Sora</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 能生成高分辨率视频，采用级联扩散模型架构，包含一个基础模型和多个时空精炼模型</span>。由于在高分辨率情况下使用注意力机制计算成本高昂且性能增益有限，基础扩散模型和低分辨率扩散模型中不太可能大量使用注意力模块。<span style="color: rgb(100,37,208); background-color: inherit">为保证时空场景一致性，并且时间一致性对视频生成比空间一致性更重要</span>，Sora 采用一种高效训练策略：<span style="color: rgb(100,37,208); background-color: inherit">使用更长但分辨率较低的视频以强化时间一致性</span>。<span style="color: rgb(100,37,208); background-color: inherit">由于</span>**`v-parameterization`**<span style="color: rgb(100,37,208); background-color: inherit">扩散模型在预测原始潜在</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">或噪声</span>$$\epsilon$$<span style="color: rgb(100,37,208); background-color: inherit">的各类方法中表现更优，Sora 采用 v-parameterization 扩散模型</span>。

为提升训练效率，大多数现有工作利用 Stable Diffusion 的预训练 VAE 编码器初始化。然而，该编码器缺乏时间压缩能力。<span style="color: rgb(216,57,49); background-color: inherit">尽管一些工作提出仅微调解码器以处理时间信息，但其在压缩潜在空间中处理视频时间数据的性能仍不理想</span>。因此 <span style="color: rgb(100,37,208); background-color: inherit">Sora 并未使用现有预训练图像 VAE 编码器进行调优，而是采用了一个时空 VAE 编码器，在图像和视频数据上联合训练，以同时压缩空间与时间信息</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">指令遵循 </span>**&#x20;

用户主要通过自然语言指令与生成式 AI 模型交互。模型指令微调旨在提升 AI 模型准确遵循 prompt 的能力。这种增强的 prompt 遵循能力使模型生成的输出更贴近人类对自然语言查询的回答。为增强文本到视频模型的指令遵循能力，<span style="color: rgb(100,37,208); background-color: inherit">Sora 采用了与</span>**`DALL·E 3`**<span style="color: rgb(100,37,208); background-color: inherit">类似的方法：训练一个描述性字幕生成器</span>**`captioner`**<span style="color: rgb(100,37,208); background-color: inherit">，并利用其生成的数据进行微调</span>。通过指令微调，<span style="color: rgb(46,161,33); background-color: inherit">Sora 能够细致关注指令中的细节，并生成精准符合用户需求的视频</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">文本到图像  </span>**

**DALL·E 3** 通过<span style="color: rgb(100,37,208); background-color: inherit">字幕改进方法解决指令遵循问题</span>，其核心假设是：**<span style="color: rgb(100,37,208); background-color: inherit">模型训练所用文本-图像对的质量决定了最终文本到图像模型的性能</span>**。低质量数据尤其是噪声数据和省略大量视觉信息的简短字幕会导致诸多问题，如<span style="color: rgb(220,155,4); background-color: inherit">忽略关键词、词序错误以及误解用户意图</span>。字幕改进方法通过为现有图像重新生成详细描述性字幕来解决这些问题：<span style="color: rgb(100,37,208); background-color: inherit">首先训练一个视觉-语言模型作为图像字幕生成器，生成精确且描述性强的图像字幕；然后利用这些字幕微调文本到图像模型</span>：DALL·E 3 借鉴对比字幕生成&#x5668;**`CoCa`**&#x65B9;法，结&#x5408;**`CLIP`**&#x67B6;构与语言模型目标联合训练一个图像字幕生成器。<span style="color: rgb(100,37,208); background-color: inherit">字幕生成器包含图像编码器、用于提取语言信息的单模态文本编码器和多模态文本解码器</span>。训练<span style="color: rgb(100,37,208); background-color: inherit">首先在单模态图像与文本 Embedding 之间施加对比损失，随后对多模态解码器输出施加字幕生成损失</span>。该字幕生成器进一步<span style="color: rgb(100,37,208); background-color: inherit">在覆盖主体对象、环境、背景、文字、风格和色彩的高细节图像描述上微调，从而能为图像生成详细描述性字幕</span>。文本到图像模型的训练数据混合了字幕生成器重新标注的数据与人工撰写的真实数据，以确保模型能准确捕捉用户输入。  但这样的改进方法引入一个潜在问题：<span style="color: rgb(216,57,49); background-color: inherit">训练数据中的描述性图像描述与实际用户 prompt 之间可能存在不匹配</span>。DALL·E 3 通过上采样解决此问题：<span style="color: rgb(100,37,208); background-color: inherit">利用 LLM 将简短用户 prompt 重写为详细且冗长的指令，确保推理时模型接收的文本输入与训练时一致</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">文本到视频  </span>**

为增强指令遵循能力，Sora 采用了类似的字幕改进方法：<span style="color: rgb(100,37,208); background-color: inherit">首先训练一个能为视频生成详细描述的视频字幕生成器，然后将其应用于所有训练视频，生成高质量的</span>`视频，描述性字幕`<span style="color: rgb(100,37,208); background-color: inherit">对，用于微调 Sora 以提升其指令遵循能力</span>。这里采&#x7528;**`CoCa`**&#x67B6;构进行视频字幕生成，&#x5373;**`VideoCoCa`**：将视频多帧分别输入图像编码器。<span style="color: rgb(100,37,208); background-color: inherit">VideoCoCa 基于 CoCa，复用预训练图像编码器权重，并独立应用于采样视频帧</span>。所得帧的 <span style="color: rgb(100,37,208); background-color: inherit">token Embedding 被展平并拼接为长序列视频表示，再经生成池化器与对比池化器处理，联合使用对比损失与字幕损失进行训练</span>。为确保用户 prompt 与训练数据中描述性字幕格式一致，Sora 使用额外的 prompt 扩展步骤：<span style="color: rgb(100,37,208); background-color: inherit">使用</span>**`GPT-4V`**<span style="color: rgb(100,37,208); background-color: inherit">将用户输入扩展为详细描述性 prompt</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">prompt 工程 </span>**&#x20;

1. **<span style="color: rgb(36,91,219); background-color: inherit">文本 prompt   </span>**

文本 prompt 工程对引导文本到视频模型生成既视觉震撼又精准满足用户需求的视频至关重要。这涉及精心构造详细描述，以有效弥补人类创造力与 AI 执行能力之间的差距。Sora 的 prompt 涵盖广泛场景。<span style="color: rgb(100,37,208); background-color: inherit">prompt 工程可利用模型的自然语言理解能力，解码复杂指令并渲染为连贯、生动且高质量的视频叙事</span>。如下图，**`a stylish woman walking down a neon-lit Tokyo street...`**&#x8FD9;类精心构造的文本 prompt ，能确保 Sora 生成与预期愿景高度一致的视频。 <span style="color: rgb(100,37,208); background-color: inherit">prompt 工程的质量取决于词汇的精心选择、所提供细节的具体程度，以及对这些细节如何影响模型输出的理解</span>。下图中的 prompt 详细指定了动作、场景、角色外观，甚至场景所需的氛围与情绪。

![](../../images/视觉多模态讲义（下）-image-116.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">图像 prompt </span>** &#x20;

<span style="color: rgb(100,37,208); background-color: inherit">图像 prompt 为待生成视频的内容及其他元素，如</span> <span style="color: rgb(220,155,4); background-color: inherit">角色、场景和情绪等</span> <span style="color: rgb(100,37,208); background-color: inherit">，提供视觉锚点</span>。此外，文本 prompt 可指导模型通过添加运动层次、交互和叙事进展，使静态图像活起来。<span style="color: rgb(100,37,208); background-color: inherit">图像 prompt 的使用使 Sora 能结合视觉与文本信息，将静态图像转化为动态、叙事驱动的视频</span>。下图是使用 DALL·E 生成图像作为 prompt 的 AI 视频示例：**`a Shiba Inu wearing a beret and turtleneck`**、**`a unique monster family`**、**`a cloud forming the word SORA`**&#x4EE5;&#x53CA;**`surfers navigating a tidal wave inside a historic hall`**。这通过 DALL·E 生成图像来指导 Sora 达到效果。

![](../../images/视觉多模态讲义（下）-image-127.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">视频 prompt   </span>**

视频 prompt 也可用于视频生成。近期工作展示了<span style="color: rgb(100,37,208); background-color: inherit">优质的视频 prompt 需兼具具体性与灵活性：既为模型提供清晰目标，如</span> <span style="color: rgb(220,155,4); background-color: inherit">特定对象和视觉主题的呈现</span> <span style="color: rgb(100,37,208); background-color: inherit">，又允许最终输出具有想象性变化</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">在视频扩展任务中， prompt 可指定时间向前或向后的扩展方向及扩展的上下文或主题</span>。下图中视频 prompt 让 Sora 将视频向时间反向扩展，以探索原始起点之前的事件。下图(b)所示的视频到视频编辑任务中，<span style="color: rgb(220,155,4); background-color: inherit">模型需清晰理解期望的变换，如改变视频风格、场景或氛围，或调整光照、情绪等细微方面</span>。图(c)中 prompt <span style="color: rgb(220,155,4); background-color: inherit">令 Sora 连接多个视频，并确保不同场景间对象过渡平滑</span>。

![](../../images/视觉多模态讲义（下）-image-126.png)

### 4.6.5 <span style="color: rgb(36,91,219); background-color: inherit">CogVideoX</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">整体架构：3D Latent 与文本在同一主干中交互</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">从像素视频到 Transformer Token</span>**

输入视频先经过 3D Causal VAE Encoder，压缩成时空 Latent；Latent 再执行 patchify 与线性映射，形成视频 Token。文本侧使用 T5 Encoder 得到文本 Token。两类 Token 拼接后进入 Expert Transformer，在 3D Full Attention 中直接互相读取，最后经过 unpatchify 与 VAE Decoder 还原成视频。

> **<span style="color: rgb(36,91,219); background-color: inherit">Expert 的含义</span>**
>
> <span style="color: rgb(100,37,208); background-color: inherit">“Expert”不是额外的路由器，也不是 MoE；它指文本与视频共用 Attention 交互，但在 AdaLN 调制等位置保留各自参数。</span>文本 Token 与视频 Token 的统计分布不同，完全共享同一组 Scale、Shift 容易互相牵制。Text Expert AdaLN 与 Vision Expert AdaLN 先分别归一化和调制，再把结果送入统一 Attention。

![文本 Token 与视频 Token 进入同一 3D Full Attention，Text/Vision Expert AdaLN 分别调制两种模态。](../../images/视觉多模态讲义（下）-cogvideox_architecture.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">3D Causal VAE</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">8×8×4 压缩与 16 Channel Latent</span>**

VAE 在空间维把高度和宽度各压缩 **`8×`**，在时间维压缩 **`4×`**，Latent Channel 数为 **`16`**。Encoder 与 Decoder 使用对称的多阶段结构，并混合 3D 与 2D Downsample/Upsample：需要跨帧建模的地方用 3D 运算，纯空间缩放则尽量用成熟的 2D 模块控制成本。

2. **<span style="color: rgb(36,91,219); background-color: inherit">时间因果卷积与 Context Parallel</span>**

Causal Convolution 只读取当前帧与历史帧，时间 Padding 放在序列开头，不让未来信息泄漏到过去。长视频训练时，时间维被切给多个设备；每个 rank 只需把卷积核所需的最后 **`k-1`** 个历史切片传给下一个 rank，便能保持因果卷积连续，而不必复制完整视频。

![3D Causal VAE 同时压缩时空维度；右侧展示时间因果卷积在 Context Parallel 下仅传递必要的历史片段。](../../images/视觉多模态讲义（下）-cogvideox_vae.jpg)

3. **<span style="color: rgb(36,91,219); background-color: inherit">VAE 的训练阶段与损失</span>**

第一阶段使用 **`256×256`**、**`17 帧`** 的短片段训练，并在 **`8 FPS`** 与 **`16 FPS`** 之间随机采样；随后扩展到 **`161 帧`** 进行长视频微调。重建目标由加权 L1、LPIPS 与 KL 组成，训练稳定后再加入 3D Discriminator 的 GAN Loss，强化纹理和运动边缘。

* **<span style="color: rgb(36,91,219); background-color: inherit">Expert Transformer 的时空建模</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">3D RoPE 分配时间、高度和宽度坐标</span>**

视频 Token 的位置由时间 t、高度 y、宽度 x 三个轴共同决定。注意力 Channel 按 **`2/8`**、**`3/8`**、**`3/8`** 分给时间、垂直和水平 RoPE。时间轴占比略小，是因为空间纹理需要更高的位置频率；三轴旋转编码又能让不同分辨率和帧数共享同一主干。

2. **<span style="color: rgb(36,91,219); background-color: inherit">为什么使用 3D Full Attention</span>**

空间 Attention 与时间 Attention 分开计算虽然便宜，但一次只能沿一个轴交换信息，处理镜头平移、主体快速穿越画面或大幅形变时需要多层间接传播。CogVideoX 让所有时空 Token 在同一 Attention 中交互，直接建立跨时间、跨位置的对应关系。<span style="color: rgb(100,37,208); background-color: inherit">这提高了大运动建模能力，也让序列长度成为主要算力瓶颈。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Diffusion Transformer 的训练策略</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">v-prediction 与 Zero-SNR</span>**

主干使用 v-prediction，并把噪声调度的末端信噪比压到零。终点若仍残留图像信息，模型在训练中几乎看不到真正的纯噪声，却要在推理第一步从纯噪声开始，容易造成曝光和亮度偏差；Zero-SNR 让训练终点与推理起点一致。

2. **<span style="color: rgb(36,91,219); background-color: inherit">图像与视频联合训练</span>**

图像作为单帧视频进入同一训练管线，补足构图、物体和细粒度纹理。为避免模型在海量图像数据上忽略运动，训练会逐步提高视频比例、长度和分辨率，并在不同阶段控制样本质量门槛。

3. **<span style="color: rgb(36,91,219); background-color: inherit">Multi-Resolution Frame Pack</span>**

传统 batch 往往把视频裁成统一尺寸和帧数，既破坏构图，也产生大量 Padding。Frame Pack 按 Token 预算，把不同时长、分辨率、宽高比的图像与视频拼成一个训练 batch；Attention Mask 隔开样本，3D RoPE 保留各自坐标。这样一个 batch 能同时利用短横屏、长竖屏和静态图像。

![Frame Pack 将不同时长、分辨率和宽高比的图像与视频打包进 batch，减少截断和 Padding。](../../images/视觉多模态讲义（下）-cogvideox_framepack.jpg)

4. **<span style="color: rgb(36,91,219); background-color: inherit">分辨率课程与时间步均衡</span>**

训练从 **`256`** 级分辨率开始，随后进入 **`512`** 与 **`768`** 阶段，最后用高质量数据微调。位置编码在高分辨率阶段采用 RoPE extrapolation，保持局部位置频率，不用插值把纹理频率整体压低。Explicit Uniform Sampling 则把 diffusion timestep 区间均匀分给不同 rank，避免某个 batch 恰好集中在极高噪声或极低噪声区间，降低跨设备 Loss 波动。

* **<span style="color: rgb(36,91,219); background-color: inherit">数据与 Caption 管线</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">从原始视频到单镜头片段</span>**

清洗后保留约 **`3500 万`** 个单镜头片段，平均时长约 **`6 秒`**，并混合约 **`20 亿`** 张过滤图像。过滤器同时检查清晰度、美学、光流运动、字幕和水印等属性，避免静止画面、强压缩或剪辑跳变污染运动学习。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Dense Caption 不只描述首帧</span>**

先生成短视频摘要，再对关键帧做 CogVLM 细粒度描述，最后整合主体属性、动作过程、镜头语言与场景变化。规模化阶段把这套结果蒸馏给专用 Caption 模型。<span style="color: rgb(46,161,33); background-color: inherit">Caption 越能区分“谁在何时做了什么”，Transformer 的跨帧 Attention 才越容易学到与语言一致的运动轨迹。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">3D Causal VAE 与 Expert Transformer 的完整实现</span>**

Encoder 与 Decoder 都由 **`4`** 个对称 stage 组成。空间轴累计下采样 **`8×`**；前两个下采样 stage 同时压缩时间轴，后一个只压空间，时间轴累计压缩 **`4×`**。所有时间卷积保持因果性，使第 *t* 个 latent 只依赖当前与过去帧，既能编码视频，也能把单张图像当作长度为 1 的视频处理。

$$z=E(x)\in\mathbb{R}^{T/4\times H/8\times W/8\times C},\qquad \hat{x}=D(z)$$

完成时空压缩后，VAE latent 进入 Transformer 前只在 *H*、*W* 上做 patchify，不额外合并相邻时间 latent。<span style="color: rgb(100,37,208); background-color: inherit">这让图像与视频共用同一种 token 组织：图像只是时间长度为 1，视频则保留真实的时间序列。</span>如果再做 temporal patchify，单图与短视频会落入不同的 token 拓扑，统一训练反而更难。

Token 的组织方式统一后，再给每个视频 token 注入时间、纵向和横向三组旋转坐标。3D RoPE 比固定的 sinusoidal position embedding 收敛更快；在 3D RoPE 之外继续叠加可学习的绝对位置向量没有带来稳定收益。真正需要保留的是三轴相对位移，而不是把训练分辨率下的每个绝对格点背下来。

位置编码确定后，文本 token 和视频 token 进入共享 attention，并共用 FFN 参数，但分别使用由 timestep 调制的 Expert AdaLN。这样在进入共享算子前，两个模态已经拥有不同的尺度、偏置和门控。进一步为两个模态拆分独立 MLP 会明显增加参数，却没有加快收敛，因此最终只保留归一化层上的 expert 路由。

最终，两类 token 被拼成一个序列，直接做 full attention，并通过 FlashAttention 控制显存。分离式做法需要文本先影响某些视频 token，再靠下一层把信息传给更远位置；full attention 允许任意文本 token 在同一层直接作用于任意时空位置，且更容易沿序列并行。代价是序列长度增大后仍有二次方计算，因此 VAE 压缩与 Frame Pack 必须一起工作。

* **<span style="color: rgb(36,91,219); background-color: inherit">完整训练课程、数据清洗与 I2V 条件</span>**

第一类浪费来自图像与视频 token 数相差巨大，若按样本数固定 batch，显存负载会剧烈波动；第二类浪费来自视频时长不一，强行裁成固定帧数会丢数据或填充大量空位。Frame Pack 先把样本按 token 数分桶，再把多个短样本装入同一个接近定长的 token 包，通过 attention mask 隔离不同样本。优化器看到的是近似恒定的 token 预算，而不是恒定的视频条数。

Frame Pack 稳定了单步 token 数，多机采样还要避免各 rank 看到偏置数据。训练时先把候选样本划成与 data-parallel rank 数相同的区间，第 *r* 个 rank 从自己的区间均匀抽样；每轮重新打乱区间映射，避免某个 rank 长期只见到一种分辨率。扩散训练仍使用标准的随机 timestep 回归：

$$\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{x_0,\epsilon,t,c}\left[\left\lVert\epsilon-\epsilon_{\theta}(x_t,t,c)\right\rVert_2^2\right]$$

基础训练先用规模更大的混合数据学习物体、动作与镜头分布，再把最高质量子集用于最后阶段；高质量 fine-tuning 子集约占原训练集的 **`20%`**。这一步会过滤水印、字幕、低清和异常运动，视觉质量提升明显，但过度过滤也会轻微损伤语义覆盖，因此它被放在训练末段，而不是从头只训“干净小集”。

最后阶段使用的高质量子集，先由自动标签筛选。标签不仅判断清晰度，还覆盖 Editing、Lack of Motion Connectivity、Low Quality、Lecture Type、Text Dominated、Noisy Screenshots 等失败类型。先人工标注约 **`20k`** 视频，再训练 Video-LLaMA 分类器批量过滤。这样能把“画面清楚但并非自然连续视频”的样本从训练分布中剔除。

过滤解决样本质量后，还要补足文本监督。原始短 caption 先来自 Panda-70M；随后每隔约 **`2 秒`** 抽帧，用 CogVLM 生成逐帧稠密描述，再让 GPT-4 汇总为包含主体、动作、背景和镜头的信息完整 caption。用约 **`50k`** 组样本微调 Llama2 后，批量替代昂贵的在线汇总；后续版本进一步使用基于 CogVLM2-Video 与 Llama3 的 CogVLM2-Caption。

同一套表示也可以接入 I2V。首帧先经过 3D Causal VAE，得到与视频 latent 通道和空间尺度一致的条件；条件 latent 与 noisy video latent 沿通道维拼接。训练时还会给首帧条件加入较强噪声，主动缩小“真实首帧编码”与“生成视频 latent”之间的分布差。I2V 的长 Prompt 可由视觉语言模型结合首帧扩写，T2V 则由 caption upsampler 扩展短文本。

> **<span style="color: rgb(36,91,219); background-color: inherit">主要局限</span>**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Full Attention 成本高</span>**：帧数和分辨率增长会同时拉长 Token 序列，显存与计算量上升很快。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">VAE 决定细节上限</span>**：小字、快速纹理和小目标若在 3D 压缩时丢失，Transformer 很难准确恢复。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Caption 误差会变成监督误差</span>**：自动描述若写错动作顺序、主体或镜头，模型会把错误文本与视频绑定。

### 4.6.6 <span style="color: rgb(36,91,219); background-color: inherit">HunyuanVideo</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">整体链路</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">像素、文本与 Flow Matching 主干</span>**

像素视频先由 Causal 3D VAE 压缩为 Latent Token；Prompt 同时进入 MLLM 与 CLIP-Large，分别提供细粒度序列表示和全局语义向量。视频 Token、文本 Token 与 timestep 条件进入 Diffusion Backbone，网络预测从噪声流向真实视频的速度场，最后由 VAE Decoder 把 Latent 还原到像素空间。

![像素视频经 Causal 3D VAE 压缩后与文本条件共同进入 Diffusion Backbone，再由 VAE Decoder 还原。](../../images/视觉多模态讲义（下）-hunyuan_overview.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">13B 模型不是一次拍脑袋定出来的</span>**
>
> 先在图像 DiT 上训练多个规模，拟合参数量、训练计算量与 Loss 的 scaling law；再把图像权重迁移到视频模型，观察视频阶段的缩放趋势。最终选择约 **`13B`** 参数，是训练收益、推理速度和部署成本之间的折中，而不是单纯追求更大。

* **<span style="color: rgb(36,91,219); background-color: inherit">视频数据与结构化 Caption</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">单镜头切分、去重与概念覆盖</span>**

长视频先用 PySceneDetect 切成单镜头片段，再用 Laplacian 清晰度从片头附近选择较清楚的起始帧。内部 VideoCLIP 负责近重复检测和语义 Embedding；Embedding 进一步聚成约 **`1 万`** 个概念中心，用来检查数据是否过度集中在少数人物、场景或风格。

2. **<span style="color: rgb(36,91,219); background-color: inherit">分层质量数据集</span>**

数据按 **`256p`**、**`360p`**、**`540p`**、**`720p`** 与最终 SFT 集逐级收紧。过滤指标包括美学、清晰度、光流运动强度、场景边界、OCR 字幕、水印、Logo 与黑边。最后约 **`100 万`** 条视频再经过人工审查，用于高质量微调。

![分层过滤从原始视频池逐步构建 256p、360p、540p、720p 与高质量微调数据。](../../images/视觉多模态讲义（下）-hunyuan_filtering.png)

3. **<span style="color: rgb(36,91,219); background-color: inherit">JSON Caption 把镜头语言变成监督信号</span>**

Caption 不只是自然语言段落，而是包含短描述、密集描述、转场与运镜、背景、风格、景别、光线、氛围和元数据标签的结构化 JSON。训练时对字段做 dropout 和随机重排，使模型既能接受简短 Prompt，也能接受完整摄影描述。镜头运动分类器覆盖推近、拉远、上下左右平移、上下左右摇摄、左右环绕、静止与手持等 **`14 类`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">Causal 3D VAE</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">4×8×8 时空压缩</span>**

形状为 **`(T+1)×3×H×W`** 的视频，经 Encoder 变成 **`(T/4+1)×16×H/8×W/8`** 的 Latent。首帧单独保留因果起点，后续帧按时间块压缩。所有跨时间卷积都只读取当前与历史帧，因此同一 VAE 可以自然支持图像样本、不同视频长度和流式分块。

![Causal 3D VAE 在时间维压缩 4×、空间维压缩 8×8，并保留首帧的因果结构。](../../images/视觉多模态讲义（下）-hunyuan_vae.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">VAE 损失与课程训练</span>**

VAE 从头训练，视频与图像采样比约为 **`4:1`**。总目标由 L1 Reconstruction、LPIPS、Adversarial Loss 与 KL 组成，对应权重约为 **`1`**、**`0.1`**、**`0.05`**、**`10⁻⁶`**。训练从低分辨率短片段逐步增加到高分辨率长片段，并随机使用 **`1～8`** 的帧采样间隔，让相邻 Latent 覆盖不同速度的运动。

3. **<span style="color: rgb(36,91,219); background-color: inherit">Tiling 解决超大视频解码</span>**

推理时把时间和空间划成有重叠的 Tile，分别编码或解码后在重叠区混合。若模型只在完整画面上训练，Tile 边界会产生接缝；因此 VAE 微调阶段随机打开或关闭 Tiling，让网络提前见到分块上下文不足的情况。

* **<span style="color: rgb(36,91,219); background-color: inherit">Dual-Stream to Single-Stream Hybrid DiT</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">先分开理解，再合并生成</span>**

前 **`20`** 个 Dual-Stream Block 保留两条参数路径：文本与视频分别做 Norm、Scale & Shift、QK-Norm 和 MLP，但 Attention 中允许两种模态交换信息。后 **`40`** 个 Single-Stream Block 把文本 Token 与视频 Token 合并为一个序列，使用统一参数继续深度融合。

![文本与视频先沿各自路径提取特征，再依次经过 Dual-stream 与 Single-stream Block。](../../images/视觉多模态讲义（下）-hunyuan_backbone_overview.png)

![Dual-stream 分开调制文本与视频，Single-stream 合并 Token 完成深度融合，两者都使用 3D RoPE。](../../images/视觉多模态讲义（下）-hunyuan_backbone_blocks.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">Full Attention 与 3D RoPE</span>**

图像被视为单帧视频，不单独维护图像主干。视频 Token 在时间、高度和宽度三个轴使用 3D RoPE，Full Attention 直接连接任意帧、任意空间位置。主干隐藏维度为 **`3072`**，FFN 为 **`12288`**，Attention 使用 **`24`** 个 Head、每个 Head **`128`** 维。

* **<span style="color: rgb(36,91,219); background-color: inherit">文本条件：MLLM、Token Refiner 与 CLIP</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">用 MLLM 代替单一 T5</span>**

Decoder-Only MLLM 经过图文对齐，更擅长解析长 Prompt 中的主体属性、动作关系和摄影细节。System Instruction 会要求模型关注视觉可见内容，减少把抽象修辞直接当成画面元素。与只提供语言统计的文本 Encoder 相比，这种表示更贴近生成任务需要的视觉语义。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Token Refiner 补回右侧上下文</span>**

Decoder-Only MLLM 使用 causal attention，一个 Token 看不到后面的词；但视频条件更适合双向理解。Token Refiner 在 MLLM 输出之上再做双向 Attention，使“一个穿红衣服的人骑着白马”中的早期主体 Token 也能吸收后面的颜色和动作信息。CLIP-Large 的 pooled embedding 经过 MLP 后加入 timestep 条件，为各层提供全局风格与语义约束。

![MLLM 通过 system instruction 强化细节描述，再由双向 Token Refiner 补回 causal attention 看不到后文的问题。](../../images/视觉多模态讲义（下）-hunyuan_text_encoder.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Flow Matching 与渐进式训练</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">直接预测速度场</span>**

令 x₀ 为噪声、x₁ 为真实视频 Latent，线性插值得到中间状态：

$$x_t=(1-t)x_0+t x_1,\quad u_t=\frac{d x_t}{dt}$$

网络输入 xₜ、t 与文本条件，预测速度 vₜ，并最小化：

$$\mathcal{L}_{\text{generation}}=\mathbb{E}_{t,x_0,x_1}\left\lVert v_t-u_t\right\rVert_2^2$$

t 使用 logit-normal 分布采样，把更多训练预算放在信息变化较大的中间区域。推理时从纯噪声出发，用 Euler ODE Solver 沿预测速度积分到数据端。

2. **<span style="color: rgb(36,91,219); background-color: inherit">从图像到长视频的课程</span>**

先在 **`256p`** 图像上建立视觉先验，再混合 **`256p/512p`** 图像；视频阶段按“低分辨率短视频 → 低分辨率长视频 → 高分辨率长视频”推进。图像数据始终保留在 batch 中，防止模型在学习运动后遗忘静态构图与世界知识。

3. **<span style="color: rgb(36,91,219); background-color: inherit">时长与宽高比分桶</span>**

样本按时长、分辨率和宽高比进入不同 Bucket，每个 Bucket 根据 Token 数设置独立 micro-batch size。短低清视频一次装得更多，长高清视频一次装得更少，在不统一裁切画面的前提下尽量打满显存。

* **<span style="color: rgb(36,91,219); background-color: inherit">Prompt Rewrite 与推理加速</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">把用户短提示改成可生成描述</span>**

Hunyuan-Large 先统一多语言表达，补充主体、动作、环境和镜头，再执行 self-revision，检查是否添加了与原意冲突的细节。规模化部署时把这套能力蒸馏到 LoRA 版本，降低在线改写成本。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Timestep Shifting 与 Guidance Distillation</span>**

少步采样时，简单均匀时间步会在高噪声阶段走得过快。Timestep Shifting 把更多步数移向早期去噪：约 **`50 步`** 时可用 shift **`7`**，低于 **`20 步`** 时可提高到 **`17`**。Guidance Distillation 在训练中随机采样 **`1～8`** 的 Guidance Scale，让单次前向逼近 CFG 的双分支结果，推理吞吐可提升约 **`1.9×`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">Scaling Law 与完整预训练课程</span>**

Scaling 实验构造 **`7`** 个 DiT-T2X(I) 规模，参数量从约 **`92M`** 到 **`6.6B`**。各模型固定使用 T5-XXL、同一 3D VAE、Cross-Attention、DDPM 与 v-prediction，并在同一批 **`256px`** 数据上训练。只有主干规模变化，loss-compute 曲线的差异才能归因于参数量，而不是数据或目标函数。

对每个计算预算，取七条训练曲线中 loss 最低的点组成 envelope，再拟合 compute、参数量与最优 loss 的幂律关系。视频 scaling 以对应的图像 checkpoint 初始化，减少从零训练全部视频尺度的成本。最终 **`13B`** 不是单纯追求最大参数量，而是联合考虑拟合趋势、推理成本和训练吞吐后的选择。

模型规模确定后，预训练先从多宽高比的 **`256px`** 图像上学习对象、构图和文本对齐；第二阶段混合 **`256px`** 与 **`512px`**，而不是直接切到纯 512 数据，避免模型在提高细节时遗忘低分辨率能力。不同 bucket 使用动态 micro-batch，使单步 token 与显存负载大致稳定。

![七个图像 DiT 规模的 loss-compute 曲线及最优计算包络点，用于拟合模型规模。](../../images/视觉多模态讲义（下）-hunyuan_scaling.png)

图像能力稳定后，再进入视频阶段。样本按分辨率、宽高比和帧数建立 bucket，并从低分辨率、短片段逐步推进到更高分辨率和更长片段。每个阶段仍混入图像样本：视频负责运动与镜头，图像维持主体细节和构图。如果完全切断图像数据，大模型会在长视频训练中出现明显的静态视觉能力遗忘。

完整预训练结束后，再从自动过滤结果中建立四组高质量子集，再进行人工复核，用较小学习率做空间质量微调。这个阶段不再扩展知识覆盖，目标是提高纹理、主体完整性和高审美样本上的稳定度，因此样本规模可以小，但标签、画质和运动必须更严格。

* **<span style="color: rgb(36,91,219); background-color: inherit">少步推理与大规模训练系统</span>**

线性时间表先定义 *t*=1−*q*/*Q*，再通过 shifting factor *s* 映射为模型真正接收的时间条件：

$$t^{\prime}=\frac{s\,t}{1+(s-1)t}$$

当 *s*>1 时，采样点向高噪声区域集中。**`50 步`** 时经验取 *s*=**`7`**；少于 **`20 步`** 时提高到 *s*=**`17`**。少步采样优先修正大结构，再在后段快速落到干净视频，**`10 步`** 下优于 linear-quadratic scheduler。

![少步数时提高 shifting factor，把采样点集中到高噪声早期；上这个图为 10 步。](../../images/视觉多模态讲义（下）-hunyuan_timestep-1.png)

![少步数时提高 shifting factor，把采样点集中到高噪声早期；这个图为 50 步。](../../images/视觉多模态讲义（下）-hunyuan_timestep.png)

时间步压缩解决了迭代次数，CFG 的双分支成本则由 distillation 处理。学生模型与教师使用相同架构和参数量，并从教师权重初始化；训练时把 guidance scale 作为额外条件，覆盖约 **`1–8`** 的范围，使单次前向直接逼近 CFG 组合后的速度。推理不再分别计算 conditional 与 unconditional 分支，吞吐约提升 **`1.9×`**，同时保留按 guidance scale 调整文本约束强度的能力。

推理侧压缩完成后，还要解决 13B 模型怎样训稳。瓶颈不只是参数量，更是超长时空序列产生的 activation。AngelPTM 负责模型并行与训练算子编排，XingMai 高速互联支撑跨卡通信；二者共同把单模型训练拆成可扩展的集群任务。

训练系统把长序列计算拆成五个并行维度。Tensor Parallel 切分矩阵乘，Sequence Parallel 切分非 attention 序列算子，Context Parallel 通过 Ring Attention 切分长时空 attention，Data Parallel 复制模型处理不同样本，ZeroCache 再分摊优化器状态与梯度。五个维度不是同时盲目开大，而是按分辨率、帧数和集群拓扑选择组合，避免通信量超过算力收益。

并行划分之外，FusedAttention 减少中间张量落显存，activation recomputation 用额外计算换显存，layer-based activation offload 把暂时不用的层激活移出 GPU。训练系统还集成自动故障检测、任务恢复与坏卡隔离，使大规模长周期训练有效运行时间达到约 **`99.5%`**；这部分工程能力与模型结构同样决定 13B 视频模型能否真正训完。

> **<span style="color: rgb(36,91,219); background-color: inherit">主要局限</span>**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">计算门槛高</span>**：13B 主干、长序列 Full Attention 与高分辨率 VAE 解码共同抬高显存和延迟。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">数据链路很重</span>**：质量过滤、结构化 Caption、Prompt Rewrite 与人工 SFT 缺一项都可能明显影响最终表现。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">物理一致性仍有限</span>**：复杂接触、遮挡后身份保持、手部细节和快速镜头下的几何关系仍可能跳变。

### 4.6.7 <span style="color: rgb(36,91,219); background-color: inherit">LTX-Video</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Holistic Latent Diffusion</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Transformer 与 VAE 共同完成去噪</span>**

传统 Latent Diffusion 把职责切得很清楚：Transformer 在 Latent 空间完成所有去噪，VAE Decoder 只负责把最终干净 Latent 还原成像素。LTX-Video 改成两段协作：Transformer 先完成多步 Latent-to-Latent 去噪，最后一步由带 timestep 条件的 VAE Decoder 在解码时直接完成。

<span style="color: rgb(100,37,208); background-color: inherit">这样设计是为了补偿高压缩 Latent 无法显式保存的高频细节。</span>Decoder 不必机械重建所有纹理，而是根据接近干净的 Latent 生成合理的发丝、树叶、皮肤和运动边缘，也不需要额外串联一套像素空间超分模型。

![Transformer 完成多步 Latent-to-Latent 去噪，VAE Decoder 将最后一步去噪与 Latent-to-Pixel 解码合并。](../../images/视觉多模态讲义（下）-ltx_holistic_denoising.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">1:192 压缩把效率问题前移</span>**

Video-VAE 直接执行 **`32×32×8`** 的空间与时间 Downsample，并输出 **`128`** Channel Latent，总压缩比约为 **`1:192`**。Patchify 被移到 VAE Encoder 入口，Transformer 不再把 **`2×2×1`** Latent Patch 重新打包；最终每个 Transformer Token 对应约 **`8192`** 个输入像素位置。

> **<span style="color: rgb(36,91,219); background-color: inherit">为什么高压缩能换来 Full Attention</span>**
>
> Attention 的主要成本随 Token 数近似二次增长。把视频在进入 Transformer 前压得更短，**`1.9B`** 级主干也能让全部时空 Token 直接互相注意，省掉分块 Attention 或单独的时间模块。效率不是只靠少走几步，而是从 Token 数、主干规模和 Decoder 职责一起优化。

* **<span style="color: rgb(36,91,219); background-color: inherit">高压缩 Video-VAE</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Causal Encoder 与 Denoising Decoder</span>**

Encoder 使用 CausalConv3D，首帧单独编码，后续帧按 **`8`** 帧时间块压缩。多级 ResBlock 与 Downsample 逐步降低分辨率，避免 Transformer 入口再做 patchify。Decoder 使用带 timestep Embedding 的 Conditional ResBlock，在各层注入可学习强度的噪声，并逐级 Upsample 到像素视频。

![Causal Encoder 直接产生 32×32×8 高压缩 Latent；Denoising Decoder 接收 timestep 并在多层注入噪声。](../../images/视觉多模态讲义（下）-ltx_vae_architecture.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">Shared Diffusion Objective</span>**

Decoder 接收带少量噪声的 Latent zₜ 与时间步 t，直接预测干净像素视频：

$$x_0=D(z_t,t),\quad t\in[0,0.2]$$

t 的范围只覆盖扩散末端，表示 Transformer 已经完成主要结构去噪，Decoder 专注于最后一段细节恢复。训练时若只给 t=0 的干净 Latent，Decoder 会退化成普通重建器；随机小噪声让它真正学会从不完整 Latent 中生成高频信息。

3. **<span style="color: rgb(36,91,219); background-color: inherit">Reconstruction GAN</span>**

普通 GAN Discriminator 只看一张图，必须同时判断内容是否合理和纹理是否真实。Reconstruction GAN 把原图与重建图作为一对输入，让判别器直接比较“哪个是原始样本”。在高压缩重建中，这种相对判断更容易聚焦模糊、运动纹理错位和局部伪影，也比无条件判别更稳定。

4. **<span style="color: rgb(36,91,219); background-color: inherit">噪声注入、统一方差与 3D DWT</span>**

Decoder 不只在 Latent 入口加噪，还在多层按 Channel 注入独立噪声，为高频细节留下随机自由度。宽 Latent 若逐 Channel 预测 log-variance，KL Loss 容易让部分 Channel 退化到接近标准高斯而不承载重建信息；这里让所有 Channel 共享一个 log-variance，避免“牺牲通道”。此外对输入和重建视频计算 **`8`** 个 3D Discrete Wavelet Transform 分量，并使用 L1 距离约束时空高频。

最终 VAE Loss 组合 MSE Reconstruction、Video-DWT L1、LPIPS 与 Reconstruction-GAN。Causal 3D Convolution 的重建上限略低于非因果结构，但能统一图像/视频训练，并自然支持首帧条件生成，因此工程收益更大。

* **<span style="color: rgb(36,91,219); background-color: inherit">3D Diffusion Transformer</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">PixArt 主干上的稳定性改造</span>**

主干沿用 PixArt-α 的文本条件 Transformer 结构，改用 RMSNorm 与 QK-Norm，并在 Self-Attention 中加入 3D RoPE。AdaLN 根据 timestep 生成 Scale 与 Shift，Cross-Attention 读取 T5-XXL 文本 Embedding，FFN 继续处理每个视频 Token。文本采用 Cross-Attention，而不是把文本与视频合并为 MM-DiT 序列，减少主序列长度。

* **<span style="color: rgb(36,91,219); background-color: inherit">Fractional Coordinate RoPE</span>**

位置坐标不直接使用整数 Token Index，而是用像素位置与真实秒数除以预设最大分辨率和时长，得到归一化小数坐标。时间坐标显式使用原始 FPS，因此同样 **`24 帧`** 在 **`8 FPS`** 和 **`24 FPS`** 下代表不同持续时间。RoPE 频率采用指数间隔，使局部与长距离位置都能获得有效分辨率。

![3D Transformer Block 使用 RMSNorm、QK-Norm、RoPE 与 Cross-Attention。](../../images/视觉多模态讲义（下）-ltx_transformer.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Image-to-Video 的逐 Token 时间步</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">条件帧和待生成帧不必处在同一噪声水平</span>**

训练时偶尔把首帧对应 Token 的 timestep 设为较小随机值，其余 Token 仍使用当前扩散时间步。模型由此学会：低噪声 Token 是可信条件，高噪声 Token 是需要生成的区域。推理时，输入图像经 Causal VAE 编成时间长度为 **`1`** 的 Latent，与随机噪声 Latent 拼接；首帧 Token 使用 t\_c≈0，其他 Token 从 t=1 开始。

![首帧条件 Token 使用较小 timestep，待生成 Token 从纯噪声开始，同一模型即可完成 I2V。](../../images/视觉多模态讲义（下）-ltx_i2v.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">同一主干为何能兼容 T2V 与 I2V</span>**
>
> 区别不在于增加第二个 Encoder 或 Control 分支，而在于给每个 Token 独立的 timestep 与位置。没有条件帧时，全部视频 Token 从高噪声开始；有条件帧时，仅首帧 Token 接近干净状态。Transformer 看到的是同一种序列接口。

* **<span style="color: rgb(36,91,219); background-color: inherit">Rectified Flow 与训练采样</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">线性路径与 velocity 目标</span>**

干净 Latent z₀ 与高斯噪声 ε 之间使用线性插值，网络预测整条路径的速度：

$$z_t=(1-t)z_0+t\epsilon,\quad v=\epsilon-z_0$$

t 从 log-normal 分布采样，并根据当前样本 Token 数向高噪声区域平移：分辨率越高、时长越长，早期去噪越难，需要更多训练概率。分布两端按约 **`0.5%`** 和 **`99.9%`** 分位裁剪，避免极端 timestep 主导梯度。

2. **<span style="color: rgb(36,91,219); background-color: inherit">多分辨率、时长与 Token Drop</span>**

不同分辨率和时长的样本被配置为接近的 Token 预算，图像只是“时长为一帧”的一种组合。训练时随机丢弃 **`0～20%`** 的视频 Token，降低计算量，并迫使 Attention 利用更远的上下文补全缺失位置。

3. **<span style="color: rgb(36,91,219); background-color: inherit">数据过滤与重写</span>**

美学过滤器使用带标签的样本对训练 Siamese Model，学习哪一条视频在构图与观感上更好；同时过滤低运动、黑边、重复和低质量片段，并重新生成细粒度 Caption。低延迟结构降低了模型计算，但最终的文本一致性和运动质量仍高度依赖这条数据管线。

* **<span style="color: rgb(36,91,219); background-color: inherit">高压缩 Video VAE 的完整论证</span>**

对 **`128`** 个视频样本的 latent pixel 做 PCA：训练早期，少数主成分就解释了大部分方差，通道之间还有明显的非对角相关；训练结束时，方差被更均匀地分配到各通道，相关矩阵接近对角。<span style="color: rgb(100,37,208); background-color: inherit">真正降低冗余的是 VAE 的 pixels-to-latents 压缩；在 Transformer 前简单 patchify 只减少 token 数，不会让 latent 自身更有信息密度。</span>

PCA 说明压缩确实在利用通道。具体实现上，空间与时间的高倍下采样直接由 VAE Encoder 学习，Transformer 接收的已经是高压缩 token，而不是先产生冗余 latent、再靠外部 patchify 打包。这使扩散主干的序列长度从源头下降，也让 Decoder 对每个 token 承担更多可学习的像素重建工作。

![PCA 与通道相关矩阵显示：训练推进后，各 latent 通道的方差贡献更均匀，非对角相关显著下降。](../../images/视觉多模态讲义（下）-ltx_latent_redundancy.png)

高压缩降低了 token 数，也把一部分去噪压力留给 Decoder。有限步 Transformer 不可能把 noisy latent 完全推回训练 VAE 时的干净 latent 流形；若 Decoder 只见过干净的 z\_0，推理时残留的扩散噪声就会变成纹理和颜色伪影。训练 Decoder 时额外输入 *t*，并让它直接从轻度加噪的 z\_t 恢复干净像素：

$$\hat{x}=D(z_t,t),\qquad t\sim\mathcal{U}(0,0.2)$$

这相当于把最后一小段去噪职责交给 Decoder，使 VAE 与 diffusion 共用一个重建目标，而不是把两者当成互不相干的模块。

Decoder 能处理轻度噪声后，还要保证重建纹理不发糊。传统 discriminator 分别接收真实或重建样本，需要判断某个局部模糊究竟来自景深还是生成失败；rGAN 把同一样本的原图与重建图成对送入，并随机交换顺序，让判别器只需判断哪一个是真实版本。比较条件被严格对齐后，Patch-GAN 也能把梯度集中到重建差异上，训练比普通 GAN 更稳定。

![传统 GAN 分别判断真伪；rGAN 成对比较同一样本的原图与重建图。](../../images/视觉多模态讲义（下）-ltx_rgan_compare.png)

rGAN 之外，VAE 还用了几项稳定化设计。Encoder 多层注入按通道学习的噪声，避免少数通道塌缩；所有 latent 位置共享统一的 log-variance，防止局部方差成为信息旁路；**`8 级`** DWT loss 同时约束低频轮廓和高频纹理。因果 3D 卷积便于 I2V 和流式使用，非因果版本重建略好但会看到未来帧；完整 3D convolution 的质量也略高于 2D+1D 分解卷积，但计算更重。

* **<span style="color: rgb(36,91,219); background-color: inherit">Transformer、RoPE 与条件注入补全</span>**

视频 Transformer 约 **`1.9B`** 参数，hidden size 为 **`2048`**，堆叠 **`28`** 个 block。每个 block 先做 video self-attention，再通过 cross-attention 读取文本。实验中这种分工清晰的结构优于把文本与视频混入同一序列的 MM-DiT：文本提供条件，视频 token 之间单独完成时空传播。

主干结构确定后，位置编码负责适配可变分辨率和时长。绝对坐标会把模型绑定到训练网格；普通 fractional coordinates 把位置归一化，但不同分辨率和时长的物理尺度仍不一致；最终采用 normalized fractional coordinates：空间坐标以像素相对预设最大分辨率归一化，时间坐标以秒相对最大时长归一化，并显式纳入原始 FPS。这样同一个运动速度在不同帧率下拥有一致的时间位置变化。

坐标归一化只是第一步。频率排布上，常见实现从高频到低频使用 inverse-exponential spacing；受控实验显示，LTX-Video 的 exponential spacing 更有效，相当于截掉一部分过低频率，让有限维度优先覆盖能够区分局部位置变化的频段。最终组合是“归一化分数坐标 + 指数频率”，二者缺一不可。

![三种位置坐标与两种频率排布：最终采用归一化分数坐标和指数递增频率。](../../images/视觉多模态讲义（下）-ltx_rope_compare.png)

位置问题解决后，attention 数值稳定性仍需单独处理。高压缩 token 包含的信息量更大，query/key 的范数容易随训练放大，使 softmax 退化成几乎 one-hot 的注意力。对 Q、K 分别做 RMSNorm 后，点积尺度受控，attention entropy 不会过早坍缩；RMSNorm 比 LayerNorm 更合适，因为它只规范幅度，不额外减去均值，也少一组统计与计算。

* **<span style="color: rgb(36,91,219); background-color: inherit">Rectified Flow 与数据管线补全</span>**

Rectified Flow 直接在数据 latent 与高斯噪声之间定义直线路径，目标速度在理想配对下为常量：

$$z_t=(1-t)z_0+t\epsilon,\qquad v=\epsilon-z_0$$

网络学习条件速度场 v\_θ，推理从 *t*=1 的噪声向 *t*=0 积分，Euler 离散更新为：

$$z_{t-\Delta t}=z_t-\Delta t\,v_{\theta}(z_t,t,c)$$

速度目标确定后，训练效率主要取决于 timestep 如何采样。这里不再均匀抽 *t*，而是从 log-normal 分布分配更多样本给速度预测更难的区间；分辨率、帧数和 token 数越大，调度越向高噪声端平移，以维持有效 SNR。概率密度在 **`0.5`** 与 **`99.9`** 分位处截断，避免分布尾部几乎永远采不到而形成训练盲区。

![不同 token 规模对应不同的 timestep 概率平移，蓝线保留了分布两端的有效采样概率。](../../images/视觉多模态讲义（下）-ltx_timestep_compare.png)

timestep 分布会随 token 数变化，因此同一模型可以按 token budget 混合不同宽高比、分辨率和帧数；图像被视为单帧视频，不另建图像分支。位置坐标使用像素和秒的物理量，timestep 分布又按 token 数修正，因此分辨率变化不会同时破坏位置尺度与噪声尺度。

模型侧的 training recipe 确定后，数据侧先处理审美偏差。审美模型由数万组人工二选一图像对训练。为了避免“漂亮图与普通图内容类别完全不同”造成捷径，每一对候选先通过多标签网络提取标签，只在 top-3 标签有重合的样本之间配对。模型必须学习构图、清晰度和视觉质量，而不能仅凭题材猜审美分数。

审美分数之外，完整处理顺序还包括裁除黑边、估计运动强度、生成缩略图、提取中帧 CLIP embedding、预测审美、聚类去重、过滤低运动与异常片段，再按 bucket resize。最终对全量保留样本重新生成细粒度 caption，并在高审美子集上继续 fine-tuning。数据管线先保证“能动、清楚、不重复”，再通过 recaption 提高文本对齐。

![原始镜头依次经过黑边裁剪、运动估计、缩略图与 CLIP 特征、审美过滤、聚类去重和统一缩放。](../../images/视觉多模态讲义（下）-ltx_data.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">主要局限</span>**
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">高压缩是信息瓶颈</span>**：细小文字、密集纹理、快速运动和小物体最容易在 VAE 阶段丢失。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Decoder 需要“生成”细节</span>**：补出的高频可能视觉合理，却不一定与输入或真实世界精确一致。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">速度不能脱离硬件解释</span>**：采样步数、精度、显卡、分辨率和版本变化都会显著影响端到端延迟。

**<span style="color: rgb(222,120,2); background-color: inherit">模型对比：</span>**&#x56DB;项工作的技术重点分别落在级联放大、长视频语义对齐、大规模开放训练和高压缩低延迟推理。

| **<span style="color: rgb(36,91,219); background-color: inherit">模型</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">生成主干</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">关键设计</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">适合强调的知识点</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| Imagen Video                                                                 | 级联 Video Diffusion                                                             | 时间超分与空间超分交替                                                                    | 早期高清视频生成如何拆解复杂度                                                                    |
| CogVideoX                                                                    | 3D Causal VAE + DiT                                                            | Expert Transformer、3D RoPE                                                     | 开放模型中的长序列与文本对齐                                                                     |
| HunyuanVideo                                                                 | Hybrid DiT + Flow Matching                                                     | MLLM 文本 Encoder、双流到单流                                                          | 大规模视频模型的结构与数据工程                                                                    |
| LTX-Video                                                                    | Holistic Latent Diffusion                                                      | 1:192 压缩、Decoder 参与去噪                                                          | 用高压缩换取低延迟的工程路线                                                                     |

### 4.6.8 <span style="color: rgb(36,91,219); background-color: inherit">Wan</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Wan-VAE</span>**

VAE 可以从高维视觉数据中学习紧凑的潜在表示，尤其是视频数据，有助于扩散模型的可扩展和高效训练。但是为视频生成任务设计有效的 VAE 有很多困难：

> 1. 视频本质上同时具有空间和时间维度，这<span style="color: rgb(216,57,49); background-color: inherit">要求 VAE 能够捕捉复杂的时空依赖关系</span>
>
> 2. 视频包含多个高分辨率像素帧，这种固有的<span style="color: rgb(216,57,49); background-color: inherit">高维特性显著增加了内存消耗和计算成本</span>，使得将 VAE 扩展到长视频序列变得困难
>
> 3. <span style="color: rgb(216,57,49); background-color: inherit">确保时间因果性</span>对于生成真实连贯的视频内容至关重要，但这一约束<span style="color: rgb(216,57,49); background-color: inherit">引入了额外的架构复杂性</span>

Wan 提出了一种专为视频生成设计的新型三维因果 VAE 架构，解决了这些问题：其<span style="color: rgb(100,37,208); background-color: inherit">结合多种策略以提升时空压缩效率、降低内存占用，并确保时间因果性</span>。这些改进使 Wan-VAE 更高效、更具可扩展性，并更适于与 DiT 等基于 diffusion 的生成模型集成。

1. **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

![](../../images/视觉多模态讲义（下）-image-134.png)

为实现高维像素空间与低维潜在空间之间的双向映射， Wan 设计了一个三维因果 VAE，给定输入视频$$V \in \mathbb{R}^{(1+T) \times H \times W \times 3}$$，Wan-VAE 将其时空维度压缩至$$[1 + T/4, H/8, W/8]$$，同时将通道数$$C$$扩展至$$16$$。在模型架构方面，<span style="color: rgb(100,37,208); background-color: inherit">将所有 </span>**<span style="color: rgb(100,37,208); background-color: inherit">GroupNorm</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 层替换为 </span>**<span style="color: rgb(100,37,208); background-color: inherit">RMSNorm</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 层，以保留时间因果性</span>。这可以使&#x7528;**<span style="color: rgb(216,57,49); background-color: inherit">特征缓存机制</span>**，显著提升了推理效率。在空间上采样层中，<span style="color: rgb(100,37,208); background-color: inherit">将输入特征通道数减半，使推理阶段的内存消耗降低</span>**`33%`**。通过精细调整基础通道数，Wan-VAE 是仅&#x6709;**`127M`**&#x53C2;数的紧凑模型，<span style="color: rgb(46,161,33); background-color: inherit">降低了编码时间和内存占用，从而有利于后续 </span>**<span style="color: rgb(46,161,33); background-color: inherit">DiT</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 的训练</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：首帧仅进行空间压缩以更好地处理图像数据，这一设计参考了 MagViT-v2

* **<span style="color: rgb(36,91,219); background-color: inherit">特征缓存机制</span>**

为高效支持任意长度视频的编码与解码，Wan-VAE 在因果卷积模块中实现了特征缓存机制：

视频序列帧数采用$$1 + T$$的输入格式，首先将视频划分为$$1 + T/4$$个块，与潜在特征数量一致。在处理输入视频序列时，<span style="color: rgb(100,37,208); background-color: inherit">模型采用分块策略，每次编码解码仅处理对应单个潜在表示的视频块</span>。基于时间压缩比，<span style="color: rgb(100,37,208); background-color: inherit">每个处理块最多包含</span>**`4`**<span style="color: rgb(100,37,208); background-color: inherit">帧，有效防止内存溢出</span>。为确保上下文块之间的时间连续性，模型会保留前一块的帧级特征缓存，并将这些缓存特征融入后续块的因果卷积计算中。上图展示了特征缓存机制的两种典型场景：

![默认设置：因果卷积不改变帧数，需保留来自历史帧的两个缓存特征，其中卷积核大小为 3。对于初始块，用两个虚拟帧进行零填充以初始化缓存缓冲区；后续块复用前一块的最后两帧作为缓存特征，并丢弃过时的历史数据](../../images/视觉多模态讲义（下）-image-131.png)

![2 倍时间下采样，步长为 2：需采用不同的缓存管理策略：仅对非初始块实施单帧缓存填充，以确保维度一致性。这保证输出序列长度严格遵循下采样比例，同时维持块边界上的因果关系。](../../images/视觉多模态讲义（下）-image-132.png)

特征缓存机制<span style="color: rgb(46,161,33); background-color: inherit">不仅优化了内存使用，还保持了块边界间的特征一致性，从而支持对无限长度视频的稳定推理</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

采用三阶段策略训练 Wan-VAE：

> 1. 构建结构相同的二维图像 VAE 并在图像数据上进行训练
>
> 2. <span style="color: rgb(100,37,208); background-color: inherit">将训练好的二维 VAE 膨胀为三维因果 Wan-VAE</span>，以提供初始的空间压缩先验，提升训练速度。此阶段在低分辨率$$128 \times 128$$&#x548C;**`5`**&#x5E27;的短帧数视频上进行训练以加速收敛。<span style="color: rgb(100,37,208); background-color: inherit">训练损失包括 </span>**<span style="color: rgb(100,37,208); background-color: inherit">L1 重建损失</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">KL 散度损失</span>**<span style="color: rgb(100,37,208); background-color: inherit">和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">LPIPS 感知损失</span>**，加权系数分别为$$3$$、$$3 \times 10^{-6}$$和$$3$$
>
> 3. 在不同分辨率和帧数的高质量视频上<span style="color: rgb(100,37,208); background-color: inherit">对模型进行微调，引入三维判别器的 GAN 损失</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型性能</span>**

<table><colgroup><col width="100"><col width="124"><col width="75"></colgroup>
<thead>
<tr>
<th>模型</th>
<th>压缩率</th>
<th>潜在维度</th>
</tr>
</thead>
<tbody>
<tr>
<td>Open Sora</td>
<td>4 \times 8 \times 8</td>
<td>4</td>
</tr>
<tr>
<td>SVD</td>
<td>1 \times 8 \times 8</td>
<td>4</td>
</tr>
<tr>
<td>Step Video</td>
<td>8 \times 16 \times 16</td>
<td>64</td>
</tr>
<tr>
<td>Mochi</td>
<td>6 \times 8 \times 8</td>
<td>12</td>
</tr>
<tr>
<td>Wan-VAE</td>
<td>4 \times 8 \times 8</td>
<td>16</td>
</tr>
</tbody>
</table>

![](../../images/视觉多模态讲义（下）-image-130.png)

一共测试了 200 个视频，每个视频包含 25 帧，分辨率为$$720 \times 720$$。圆圈大小与模型参数量正相关。从图中可以看出，<span style="color: rgb(46,161,33); background-color: inherit">Wan-VAE 在两项指标上均表现良好，兼具优越的视频质量和高处理效率</span>。在相同硬件环境下， <span style="color: rgb(46,161,33); background-color: inherit">VAE 重建速度比现有 SOTA 方法 HunYuan Video 快</span>**`2.5`**<span style="color: rgb(46,161,33); background-color: inherit">倍</span>。由于模型的小规模设计和特征缓存机制，这一速度优势在更高分辨率下将进一步放大。总的来说，Wan-VAE 为视频重建任务和视频生成训练奠定了基础，验证了该模型设计的有效性。

* **<span style="color: rgb(36,91,219); background-color: inherit">Wan</span>**

![](../../images/视觉多模态讲义（下）-image-128.png)

Wan 的架构基于 DiT 结构，包含三个核心组件：**<span style="color: rgb(100,37,208); background-color: inherit">Wan-VAE</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">DiT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 和</span>**<span style="color: rgb(100,37,208); background-color: inherit">文本编码器</span>**。对于给定视频$$V \in \mathbb{R}^{(1+T) \times H \times W \times 3}$$，Wan-VAE 的编码器将其从像素空间映射到潜在空间$$x \in \mathbb{R}^{(1 + T/4) \times H/8 \times W/8}$$，随后送入 DiT 结构进行处理。

1. **<span style="color: rgb(36,91,219); background-color: inherit">视频 DiT</span>**

DiT 主要由三部分组成：**<span style="color: rgb(100,37,208); background-color: inherit">patchify 模块</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">Transformer 块</span>**<span style="color: rgb(100,37,208); background-color: inherit">和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">unpatchify 模块</span>**。在每个 Transformer 块中，重点建模时空上下文关系，并嵌入文本条件和时间步信息。

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Patchify 模块</span>**：中使用核大小为$$(1, 2, 2)$$的三维卷积，并进行展平操作，将$$x$$转换为形状为$$(B, L, D)$$的特征序列，其中$$B$$为批量大小，$$L = (1 + \frac{T}{4}) \times \frac{H}{16}
>     \times \frac{W}{16}$$为序列长度，$$D$$为潜在维度。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">Transformer 块</span>**：采用<span style="color: rgb(100,37,208); background-color: inherit">交叉注意力融入文本条件</span>，确保模型在长上下文建模下仍具备良好的指令遵循能力；使用<span style="color: rgb(100,37,208); background-color: inherit">带 SiLU 激活函数的 MLP 来处理时间编码，并分别预测六个参数</span>，这个 MLP 在所有 Transformer 块间共享，每个块学习一组独立的偏置。在相同参数规模下，可减少&#x7EA6;**`25%`**&#x7684;参数量，提升性能。

![](../../images/视觉多模态讲义（下）-image-129.png)

文本编码器：Wan 采&#x7528;**`umT5`**&#x5BF9;输入文本进行编码。**umT5** 优点：

> 1. <span style="color: rgb(46,161,33); background-color: inherit">强大的多语言编码能力</span>，能有效理解中英文及输入的视觉文本
>
> 2. 相同条件下，在<span style="color: rgb(46,161,33); background-color: inherit">组合能力上优于其他采用单向注意力机制的大语言模型</span>
>
> 3. <span style="color: rgb(46,161,33); background-color: inherit">收敛更快</span>，在相同参数规模下训练更高效

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

采用 Flow Matching 框架，在图像与视频域中建模统一的去噪扩散过程。**<span style="color: rgb(100,37,208); background-color: inherit">首先在低分辨率图像上进行预训练，随后进行多阶段的图像-视频联合优化</span>**。联合训练过程中，数据的空间分辨率和时间长度随训练阶段逐步提升。

**<span style="color: rgb(222,120,2); background-color: inherit">训练目标</span>**：流匹配为扩散模型中的连续时间生成过程提供了理论支撑，通过常微分方程 ODE 实现稳定训练，避免了迭代速度预测，同时保持与最大似然目标等价。训练时，<span style="color: rgb(100,37,208); background-color: inherit">给定图像或视频潜在变量</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">、随机噪声</span>$$x_0 \sim \mathcal{N}(0, I)$$<span style="color: rgb(100,37,208); background-color: inherit">，以及从 logit-normal 分布中采样的时间步</span>$$t \in [0, 1]$$<span style="color: rgb(100,37,208); background-color: inherit">，中间潜在变量</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">作为模型输入。根据 Rectified Flows，</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit"> 定义为</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">的线性插值</span>：

$$x_t = t x_1 + (1 - t) x_0$$

真实速度$$v_t$$为：

$$v_t = \frac{d x_t}{d t} = x_1 - x_0$$

<span style="color: rgb(100,37,208); background-color: inherit">训练模型来预测速度，因此损失函数可表示为模型输出与</span>$$v_t$$<span style="color: rgb(100,37,208); background-color: inherit">之间的均方误差</span>：

$$\mathcal{L} = \mathbb{E}_{x_0, x_1, c_{\text{txt}}, t} \left\| u(x_t, c_{\text{txt}}, t; \theta) - v_t \right\|^2$$

其中$$c_{\text{txt}}$$是长度为$$512$$&#x7684;**`umT5`**&#x6587;本 Embedding 序列，$$\theta$$为模型参数，$$u(x_t, c_{\text{txt}}, t; \theta)$$表示模型预测的速度。

**<span style="color: rgb(222,120,2); background-color: inherit">图像预训练</span>**：直接在高分辨率图像与长视频序列上进行联合训练比较困难：

> 1. <span style="color: rgb(216,57,49); background-color: inherit">序列长度显著增加</span>，如$$1280 \times 720$$<span style="color: rgb(220,155,4); background-color: inherit">视频包含</span>$$81$$<span style="color: rgb(220,155,4); background-color: inherit">帧，降低了训练吞吐量，在固定 GPU 小时预算下导致数据吞吐不足，阻碍模型收敛</span>
>
> 2. <span style="color: rgb(216,57,49); background-color: inherit">GPU 内存消耗过高</span>，迫使使用次优批量大小，引发梯度方差激增，导致训练不稳定

为缓解这些问题，**<span style="color: rgb(100,37,208); background-color: inherit">Wan</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 首先在</span>$$256$$<span style="color: rgb(100,37,208); background-color: inherit">像素低分辨率文本到图像任务上进行预训练，以强制模型在引入高分辨率视频模态前，建立跨模态语义-文本对齐和几何结构保真能力</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">图像-视频联合训练</span>**：在大规模 256 像素文本到图像预训练后，通过分辨率逐步提升的课程学习策略，实施图像与视频数据的分阶段联合训练。训练包含三个阶段，按空间分辨率区分：

> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 1</span>**：联合训练$$256$$像素图像与$$5$$秒视频片段，视频片段为$$192$$像素，$$16$$FPS
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 2</span>**：将图像与视频分辨率均提升至$$480$$像素，保持$$5$$秒视频时长不变
>
> **<span style="color: rgb(36,91,219); background-color: inherit">Stage 3</span>**：将两者分辨率进一步提升至$$720$$像素

> **<span style="color: rgb(222,120,2); background-color: inherit">训练配置</span>**：采用 **BF16** 混合精度进行训练，优化器为 AdamW，权重衰减系数为$$10^{-3}$$。初始学习率设为 $$10^{-4}$$，并根据 FID 和 CLIP Score 指标的平台期动态衰减。

---

[Previous](10-Diffusion-Model--其他工作.md) | [Contents](../../README.md) | [Next](12-UMM-统一理解生成模型--Diffusion-based.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=11)
