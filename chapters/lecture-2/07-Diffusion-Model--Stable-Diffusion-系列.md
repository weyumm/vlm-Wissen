[Previous](06-Diffusion-Model--Diffusion-基础.md) | [Contents](../../README.md) | [Next](08-Diffusion-Model--Diffusion-的应用.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=7)

## 4.2 <span style="color: rgb(36,91,219); background-color: inherit">Stable Diffusion 系列</span>

### 4.2.1 <span style="color: rgb(36,91,219); background-color: inherit">Stable Diffusion 1</span>

尽管扩散模型可以通过对相应的损失项进行欠采样来忽略感知上无关的细节，但它们仍然需要在像素空间中进行昂贵的函数评估，这导致了巨大的计算时间和能源消耗。

因此通过引入压缩学习阶段与生成学习阶段的显式分离来规避这一缺点，如右图。为实现这一点，作者利用了一种自编码模型，该模型学习了一个与图像空间感知等价但计算复杂度显著降低的空间。

这种方法具有以下几个优势：

> 1. 通过离开高维图像空间，获得了<span style="color: rgb(46,161,33); background-color: inherit">计算效率更高</span>的扩散模型，因为采样是在低维空间中进行的
>
> 2. 利用了扩散模型从&#x5176;**`UNet`**&#x67B6;构继承的归纳偏置，这使得它们特别<span style="color: rgb(46,161,33); background-color: inherit">适合处理具有空间结构的数据，从而避免了先前方法所需的激进且降低质量的压缩级别</span>
>
> 3. 获得了通用的压缩模型，其<span style="color: rgb(46,161,33); background-color: inherit">潜在空间可用于训练多个生成模型，也可用于其他下游应用</span>，例如单图&#x50CF;**`CLIP`**&#x5F15;导的合成

![](../../images/视觉多模态讲义（下）-image-58.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">感知图像压缩</span>**

感知压缩模型是基于先前的工作，并<span style="color: rgb(100,37,208); background-color: inherit">由一个结合了感知损失和基于  patch 的对抗目标训练的自编码器组成</span>。这确保了重建结果被限制在图像流中，通过强制局部真实感避免了仅依赖像素空间损失（如$$L_2$$或$$L_1$$目标）带来的模糊。

更具体地说，<span style="color: rgb(100,37,208); background-color: inherit">给定RGB空间中的图像</span>$$x \in \mathbb{R}^{H \times W \times 3}$$<span style="color: rgb(100,37,208); background-color: inherit">，编码器</span>$$\mathcal{E}$$<span style="color: rgb(100,37,208); background-color: inherit">将</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">编码为潜在表示</span>$$z = \mathcal{E}(x)$$<span style="color: rgb(100,37,208); background-color: inherit">，解码器</span>$$\mathcal{D}$$<span style="color: rgb(100,37,208); background-color: inherit">从潜在表示重建图像，得到</span>$$\tilde{x} = \mathcal{D}(z) = \mathcal{D}(\mathcal{E}(x))$$<span style="color: rgb(100,37,208); background-color: inherit">，其中</span>$$z \in \mathbb{R}^{h \times w \times c}$$。编码器还通过因子$$f = H/h = W/w$$对图像进行下采样，作者研究了不同的下采样因子$$f = 2^m$$，其中$$m \in \mathbb{N}$$。

![](../../images/视觉多模态讲义（下）-image-61.png)

为了避免潜在空间的方差过高，作者尝试了两种不同的正则化方法：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">KL 正则化</span>**：它对学习到的潜在变量施加轻微的 KL 惩罚，使其接近标准正态分布，类似于变分自编码器 VAE
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">VQ 正则化</span>**：在解码器中使用矢量量化层。该模型可以解释为一种 VQGAN，但量化层被解码器吸收

由于后续的扩散模型设计为与学习到的二维潜在空间$$z = \mathcal{E}(x)$$一起工作，因此可以使用相对温和的压缩率实现非常好的重建效果。这与之前的工作形成对比，依赖于对学习空间$$z$$的任意一维排序来对其分布进行自回归建模，从而忽略了$$z$$的大部分内在结构。因此更好地保留了$$x$$的细节 。

* **<span style="color: rgb(36,91,219); background-color: inherit">潜在扩散模型</span>**

扩散模型是概率模型，旨在通过逐步去噪一个正态分布变量来学习数据分布$$p(x)$$，这相当于学习固定长度为$$T$$的马尔可夫链的逆过程。对于图像合成，最成功的模型依赖于$$p(x)$$的重加权变分下界，这反映了去噪得分匹配。

这些模型可以理解为<span style="color: rgb(100,37,208); background-color: inherit">一系列权重相等的去噪自编码器</span>$$\epsilon_\theta(x_t, t); t = 1, \dots, T$$<span style="color: rgb(100,37,208); background-color: inherit">，它们预测输入</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">的去噪情况，其中</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">是输入</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">加噪声得到的</span>。相应的目标函数可以简化为：

$$L_{\text{DM}} = \mathbb{E}_{x, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|_2^2 \right]$$

其中$$t$$从$$\{1, \dots, T\}$$中均匀采样。

1. **<span style="color: rgb(36,91,219); background-color: inherit">潜在表示的生成模型</span>**

通过训练好的感知压缩模型$$\mathcal{E}$$和$$\mathcal{D}$$，可以访问一个高效的低维潜在空间，在该空间中高频、不可感知的细节被抽象化。与高维像素空间相比，该空间更适合基于似然的生成模型，因为它们现在可以专注于数据中重要的语义部分，并且在维度更低、计算效率更高的空间中进行训练。

与之前高度压缩离散潜在空间中依赖自回归、基于注意力的 Transformer 模型的工作不同，<span style="color: rgb(100,37,208); background-color: inherit">扩散模型可以利用模型提供的图像特定归纳偏置</span>。这包括能够主要使用 2D 卷积层构建底&#x5C42;**`UNet`**，并进一步聚焦于感知上最重要的部分，使用重加权边界，现在表示为：

$$L_{\text{LDM}} := \mathbb{E}_{\mathcal{E}(x), \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t)\|_2^2 \right]$$

模型的神经主干$$\epsilon_\theta(\cdot, t)$$是以时间为条件&#x7684;**`UNet`**。由于前向过程是固定的，$$z_t$$可以在训练期间从$$\mathcal{E}$$获得，而来自$$p(z)$$的样本可以通过$$\mathcal{D}$$一次解码到图像空间。

* **<span style="color: rgb(36,91,219); background-color: inherit">条件机制</span>**

与其他类型的生成模型类似，<span style="color: rgb(100,37,208); background-color: inherit">扩散模型原则上能够建模形式为</span>$$p(z|y)$$<span style="color: rgb(100,37,208); background-color: inherit">的条件分布。这可以通过条件去噪自编码器</span>$$\epsilon_\theta(z_t, t, y)$$<span style="color: rgb(100,37,208); background-color: inherit">实现</span>，并通过输入$$y$$控制合成过程，如`文本`<span style="color: rgb(220,155,4); background-color: inherit">、</span>`语义图`<span style="color: rgb(220,155,4); background-color: inherit">或其他图像到图像的翻译任务</span>。

通过<span style="color: rgb(100,37,208); background-color: inherit">在扩散模型底层的</span>**`UNet`**<span style="color: rgb(100,37,208); background-color: inherit">主干中引入交叉注意力机制，将其转变为更加灵活的条件图像生成器</span>。该机制对于学习各种输入模态的注意力模型非常有效。为了从不同模态预处理$$y$$，例如<span style="color: rgb(220,155,4); background-color: inherit">语言 prompt</span>，作者<span style="color: rgb(100,37,208); background-color: inherit">引入了一个特定领域的编码器</span>$$\tau_\theta$$<span style="color: rgb(100,37,208); background-color: inherit">，它将</span>$$y$$<span style="color: rgb(100,37,208); background-color: inherit">投影到一个中间表示</span>$$\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}$$<span style="color: rgb(100,37,208); background-color: inherit">，然后通过一个交叉注意力层将其映射到</span>**`UNet`**<span style="color: rgb(100,37,208); background-color: inherit">的中间层</span>。该交叉注意力层实现了以下公式：

$$\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^T}{\sqrt{d}} \right) \cdot V$$

其中$$Q = W_Q^{(i)} \cdot \phi_i(z_t), \quad K = W_K^{(i)} \cdot \tau_\theta(y), \quad V = W_V^{(i)} \cdot \tau_\theta(y).$$

在这里，$$\phi_i(z_t) \in \mathbb{R}^{N \times d_\epsilon^i}$$为实现$$\epsilon_\theta$$&#x7684;**`UNet`**&#x7684;一个中间表示，而$$W_V^{(i)} \in \mathbb{R}^{d \times d_\epsilon^i}$$、$$W_Q^{(i)} \in \mathbb{R}^{d \times d_\tau}$$和$$W_K^{(i)} \in \mathbb{R}^{d \times d_\tau}$$是可学习的投影矩阵，如上图所示。

基于图像-条件对，通过以下公式学习条件潜在扩散模&#x578B;**`LDM`**：

$$L_{\text{LDM}} := \mathbb{E}_{\mathcal{E}(x), y, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y))\|_2^2 \right]$$

其中$$\tau_\theta$$和$$\epsilon_\theta$$进行联合优化。这种条件机制非常灵活，因为$$\tau_\theta$$可以用特定领域的专家参数化，例如<span style="color: rgb(220,155,4); background-color: inherit">当</span>$$y$$<span style="color: rgb(220,155,4); background-color: inherit">为文本提示时可以使用不带 mask 的 Transformer</span>。

### 4.2.2 <span style="color: rgb(36,91,219); background-color: inherit">Stable Diffusion 2</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">整体结构</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">技术定位</span>**

Stable Diffusion 2 不是另一套独立的生成范式，而是 Latent Diffusion Model 的一组具体实现。它保留 VAE、时序条件 U-Net 和 cross-attention 三段式结构，把文本条件、训练数据、预测目标与可用分辨率更新为第二代配置。

<span style="color: rgb(100,37,208); background-color: inherit">模型不在 RGB pixel space 中直接执行完整扩散过程，而是先把图像压缩到较小的 latent space，再让 U-Net 学习 latent 的去噪方向，最后只调用一次 Decoder 恢复图像。</span>

SD 2.0 的核心变化包括 OpenCLIP ViT-H/14 Text Encoder、原生 **`512×512`** 与 **`768×768`** 两种文生图 checkpoint，以及 Depth2Img、Inpainting 和 ×4 Upscaler。SD 2.1 沿用 2.0 的模型结构，在同一代数据和 checkpoint 上继续训练。

2. **<span style="color: rgb(36,91,219); background-color: inherit">端到端计算链</span>**

文本 prompt 先经过 Tokenizer，再由冻结的 OpenCLIP ViT-H/14 Text Encoder 产生 token Embedding。SD 2 使用 Text Encoder 倒数第二层的输出，单个 token 的上下文维度为 **`1024`**。这些 Embedding 不直接变成图像，而是作为 cross-attention 的 Key 和 Value 注入 U-Net。

训练时，RGB 图像由 VAE Encoder 下采样 **`8`** 倍。尺寸为 **`512×512`** 的图像对应 **`64×64×4`** latent，尺寸为 **`768×768`** 的图像对应 **`96×96×4`** latent。U-Net 接收带噪 latent、timestep 与文本 Embedding，预测当前去噪步骤需要的量。

推理从高斯噪声开始，scheduler 反复调用 U-Net，把 latent 从高噪声状态逐步更新到低噪声状态。最后，VAE Decoder 把结果映射回 RGB 图像。Text Encoder 和 VAE 通常保持冻结，生成阶段的主要迭代计算集中在 U-Net。

3. **<span style="color: rgb(36,91,219); background-color: inherit">SD 2.0 模型家族</span>**

**`stable-diffusion-2-base`** 是 **`512×512`** 文生图模型，使用标准 noise prediction，也就是预测加入 latent 的噪声。它是 768 模型、Depth2Img 和 Inpainting 的起点。

**`stable-diffusion-2`** 面向 **`768×768`** 生成，从 512-base 继续训练，并切换到 v-prediction。它与 SD 1.5 的 U-Net 参数规模接近，但使用不同的 Text Encoder、cross-attention 上下文宽度和预测目标，checkpoint 不能按同一配置直接混用。

**`stable-diffusion-2-depth`** 增加单目相对深度条件；**`stable-diffusion-2-inpainting`** 增加 mask 与 masked-image latent；**`stable-diffusion-x4-upscaler`** 根据低分辨率图像和文本生成四倍尺寸结果。

4. **<span style="color: rgb(36,91,219); background-color: inherit">SD 2.0 与 SD 2.1 的关系</span>**

SD 2.1 没有更换 VAE、U-Net 或 OpenCLIP 路线，512-base 与 768-v 的参数量和结构都与 2.0 相同。变化发生在继续训练阶段：使用更多数据，并把训练集的 NSFW 过滤阈值放宽。

2.1-base 从 2.0 的 512-base 继续训练 **`220k`** steps，数据过滤阈值改为 **`punsafe=0.98`**。2.1 的 768-v 先在 **`punsafe=0.1`** 数据上继续 **`55k`** steps，再在 **`punsafe=0.98`** 数据上训练 **`155k`** steps。

<span style="color: rgb(100,37,208); background-color: inherit">因此，“Stable Diffusion 2”通常表示 2.0 与 2.1 共享的架构代际；2.1 更接近一组在 2.0 checkpoint 上追加训练得到的更新权重。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Latent Diffusion 核心方法</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">VAE 与 perceptual compression</span>**

第一阶段是 KL-regularized VAE。Encoder 把图像 $$x$$ 映射为二维 latent，Decoder 再把 latent 重建为图像：

$$z_0=s\,\mathcal{E}(x),\qquad \hat{x}=\mathcal{D}(z_0/s)$$

其中 $$s$$ 是 latent 的缩放系数，SD 2 的配置取 **`0.18215`**。缩放后的 latent 具有更稳定的数值范围，后续 diffusion Backbone 在这一空间中训练。

VAE 的目标不是逐像素无损压缩。它结合 perceptual loss、patch-based adversarial loss 与很弱的 KL regularization，优先保留能影响人类感知的结构、颜色和纹理，同时允许微小的高频细节损失。

<span style="color: rgb(46,161,33); background-color: inherit">将空间尺寸缩小 8 倍后，U-Net 处理的空间位置数量约降为 pixel space 的六十四分之一，训练和逐步采样都明显更省计算。</span>

<span style="color: rgb(216,57,49); background-color: inherit">VAE 是有损模块。细小文字、极细线条、规则纹理和人脸局部可能在编码或解码阶段发生变化，即使 U-Net 已经生成了合理的 latent。</span>

2. **<span style="color: rgb(36,91,219); background-color: inherit">前向加噪过程</span>**

diffusion 的前向过程固定不学习。它按照预设 noise schedule，在每个 timestep 向干净 latent $$z_0$$ 加入高斯噪声：

$$q(z_t\mid z_0)=\mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\,z_0,\left(1-\bar{\alpha}_t\right)I\right)$$

利用重参数化，可以直接从任意 timestep 采样带噪 latent，而不需要真的执行前面所有加噪步骤：

$$z_t=\sqrt{\bar{\alpha}_t}\,z_0+\sqrt{1-\bar{\alpha}_t}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0,I)$$

$$\bar{\alpha}_t$$ 控制信号保留比例。较小的 $$t$$ 保留更多图像结构，较大的 $$t$$ 更接近纯噪声。训练时随机采样 timestep，使同一个 U-Net 学会处理整个噪声区间。

3. **<span style="color: rgb(36,91,219); background-color: inherit">时序条件 U-Net</span>**

U-Net 是 diffusion 的主 Backbone。输入先在 Encoder 路径逐级下采样，低分辨率层建立大范围语义关系；Decoder 路径逐级上采样，并通过 skip connection 取回早期层保存的局部空间细节。

timestep 会编码成向量并注入多个 residual block，使同一套参数在高噪声阶段学习整体布局，在低噪声阶段修复边缘、纹理和局部形状。SD 2 的 U-Net 约有 **`865M`** 参数，基础通道数为 **`320`**，通道倍率为 **`1、2、4、4`**，每个尺度包含 **`2`** 个 residual blocks。

<span style="color: rgb(100,37,208); background-color: inherit">U-Net 并不是一次输出完整图像，而是在每个 timestep 给出一个局部反向更新方向；scheduler 决定如何把这一预测转换为下一时刻的 latent。</span>

4. **<span style="color: rgb(36,91,219); background-color: inherit">cross-attention 文本条件</span>**

OpenCLIP 产生的文本 token Embedding 通过 cross-attention 进入 U-Net。某一层的图像特征提供 Query，文本特征提供 Key 和 Value：

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\!\left(\frac{QK^{\mathsf{T}}}{\sqrt{d}}\right)V$$

$$Q=W_Q\varphi_i(z_t),\qquad K=W_K\tau(y),\qquad V=W_V\tau(y)$$

$$\varphi_i(z_t)$$ 是 U-Net 第 $$i$$ 个空间层的特征，$$\tau(y)$$ 是文本 Encoder 输出。每个空间位置都可以根据当前图像状态选择不同的文本 token，因此主体、属性、风格和位置描述能够在不同区域产生不同影响。

SD 2 在多个分辨率层中放置 Spatial Transformer。self-attention 负责图像位置之间的关系，cross-attention 负责图像位置与文本 token 之间的关系，MLP 则完成逐位置的非线性变换。

下图完整展示了 latent diffusion 的两阶段结构、U-Net skip connection、concat conditioning 和 cross-attention conditioning。

![Latent Diffusion 以 VAE 映射 pixel space 与 latent space，并通过 U-Net 中的 cross-attention 接收条件。](../../images/视觉多模态讲义（下）-ldm-figure-3.png)

5. **<span style="color: rgb(36,91,219); background-color: inherit">noise prediction 目标</span>**

512-base 使用标准 noise prediction。给定图像、文本、随机 timestep 和噪声，U-Net 预测加到 latent 中的 $$\epsilon$$：

$$\mathcal{L}_{\epsilon}=\mathbb{E}_{x,y,t,\epsilon}\!\left[\left\|\epsilon-\epsilon_{\theta}(z_t,t,\tau(y))\right\|_2^2\right]$$

这个损失可以只在 latent space 中计算。VAE Encoder 提供干净 latent，前向过程直接构造 $$z_t$$，U-Net 的输出与真实噪声做均方误差。Decoder 不需要参与每一步 U-Net 训练。

6. **<span style="color: rgb(36,91,219); background-color: inherit">v-prediction 目标</span>**

768-v 不直接预测噪声，而是预测由干净 latent 与噪声共同构成的 velocity。令 $$\alpha_t^2+\sigma_t^2=1$$ 且 $$z_t=\alpha_tz_0+\sigma_t\epsilon$$，目标定义为：

$$v_t=\alpha_t\epsilon-\sigma_tz_0$$

得到 $$v_t$$ 后，可以同时恢复干净 latent 与噪声估计：

$$\hat{z}_0=\alpha_tz_t-\sigma_tv_{\theta}(z_t,t,c),\qquad \hat{\epsilon}=\sigma_tz_t+\alpha_tv_{\theta}(z_t,t,c)$$

<span style="color: rgb(100,37,208); background-color: inherit">v-prediction 改变的是 U-Net 输出的参数化方式，不是把 diffusion 换成另一种生成模型。scheduler 必须知道 checkpoint 使用 epsilon prediction 还是 v-prediction，否则反向更新的含义会错位。</span>

7. **<span style="color: rgb(36,91,219); background-color: inherit">Classifier-Free Guidance</span>**

Classifier-Free Guidance 同时计算有条件预测与无条件预测，再沿两者之差放大文本条件：

$$\hat{\epsilon}_{\mathrm{cfg}}=\epsilon_{\theta}(z_t,t,c_{\varnothing})+w\left[\epsilon_{\theta}(z_t,t,c)-\epsilon_{\theta}(z_t,t,c_{\varnothing})\right]$$

$$c$$ 是 prompt Embedding，$$c_{\varnothing}$$ 通常来自空文本；使用 negative prompt 时，无条件分支改为负向文本的 Embedding。$$w$$ 是 guidance scale。

<span style="color: rgb(46,161,33); background-color: inherit">增大 guidance scale 通常会提升 prompt 对齐程度，让主体和属性更明确。</span> <span style="color: rgb(216,57,49); background-color: inherit">guidance 过强会牺牲多样性，并可能带来过饱和、边缘发硬、重复纹理和构图失真。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">训练与配置</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Autoencoder 与 latent 配置</span>**

SD 2 使用下采样因子为 **`8`** 的 AutoencoderKL，RGB 输入通道为 **`3`**，latent 通道为 **`4`**。Autoencoder 只负责图像与 latent 之间的双向映射，不接收文本条件。

配置中的 latent scale factor 为 **`0.18215`**。因此，外部工具在加载 checkpoint 时不仅要匹配 U-Net 和 Text Encoder，还要保持相同的 VAE 缩放约定；漏掉缩放会使 U-Net 接收到与训练分布不同的数值范围。

2. **<span style="color: rgb(36,91,219); background-color: inherit">OpenCLIP ViT-H/14 Text Encoder</span>**

SD 2 将 SD 1.x 的 OpenAI CLIP Text Encoder 换成 OpenCLIP ViT-H/14。Text Encoder 固定不训练，U-Net 读取倒数第二层 token Embedding，cross-attention 的 context dimension 为 **`1024`**。

<span style="color: rgb(100,37,208); background-color: inherit">更换 Text Encoder 会同时改变 Tokenizer、词语 Embedding 空间和 prompt 对生成结果的作用方式，所以 SD 1.x 上形成的 prompt 习惯不一定能原样迁移到 SD 2。</span>

文本条件保留 token 序列，而不是先压成单个句向量。这样，U-Net 中不同空间位置可以关注不同词语。倒数第二层输出比最终对比学习投影更适合作为细粒度 token 条件。

3. **<span style="color: rgb(36,91,219); background-color: inherit">512-base 的两阶段训练</span>**

512-base 从头训练。第一阶段在 LAION-5B 子集上以 **`256×256`** 分辨率训练 **`550k`** steps。训练数据要求 aesthetic score 不低于 **`4.5`**，并使用 LAION NSFW classifier，以 **`punsafe=0.1`** 过滤显式成人内容。

第二阶段继续训练 **`850k`** steps，裁剪分辨率提高到 **`512×512`**，只使用原始宽高均不低于 **`512`** 的图像。这个阶段让模型在保持已学语义的同时适应更大的 latent 网格和更细的局部结构。

512-base 使用 epsilon prediction。训练完成后，它既可直接生成 512 图像，也可作为其他 512 分支的初始化权重。

4. **<span style="color: rgb(36,91,219); background-color: inherit">768-v 的继续训练</span>**

768-v 从 512-base 恢复权重，先在相同数据上用 v-objective 训练 **`150k`** steps，使 U-Net 从 epsilon prediction 迁移到 v-prediction。之后在 **`768×768`** 数据子集上继续训练 **`140k`** steps。

原生 768 训练意味着 U-Net 在 **`96×96`** latent 网格上学习布局和细节。它可以生成其他宽高比，但总像素量、长宽边界和构图稳定性仍受训练分辨率影响。

<span style="color: rgb(216,57,49); background-color: inherit">512-base 与 768-v 的 prediction type 不同。仅修改输出宽高而不切换配置，不能把 512 epsilon checkpoint 等价地变成 768 v-prediction checkpoint。</span>

5. **<span style="color: rgb(36,91,219); background-color: inherit">U-Net 结构配置</span>**

SD 2 的 U-Net 输入和输出都是 **`4`** 通道 latent。基础通道数为 **`320`**，四个尺度的 channel multiplier 为 **`1、2、4、4`**，每个尺度包含 **`2`** 个 residual blocks。cross-attention 的 context dimension 是 **`1024`**，与 OpenCLIP token Embedding 对齐。

Spatial Transformer 使用 self-attention、cross-attention 与 MLP。相较 SD 1.x，SD 2 打开 linear projection 形式的 Transformer，并调整 attention head 的配置，以容纳 1024 维文本条件。

<span style="color: rgb(46,161,33); background-color: inherit">卷积路径保留图像的局部归纳偏置，attention 路径建立远距离空间关系并接收文本条件，两类模块在同一个 U-Net 中协作。</span>

6. **<span style="color: rgb(36,91,219); background-color: inherit">优化器、批量与计算资源</span>**

模型训练使用 AdamW。学习率先在 **`10,000`** steps 内 warmup 到 **`0.0001`**，之后保持恒定。Gradient accumulation 为 **`1`**，全局 batch size 为 **`2048`**。

训练资源记录为 **`32×8`** 张 A100 GPU。大批量训练让每一步覆盖更多图文对，但数据过滤、caption 质量和视觉概念分布仍会直接限制模型学到的语义边界。

7. **<span style="color: rgb(36,91,219); background-color: inherit">2.1 的追加训练与数据过滤</span>**

2.1-base 在 2.0 512-base 上继续 **`220k`** steps。2.1 的 768-v 在 2.0 768-v 上先继续 **`55k`** steps，再以放宽后的过滤阈值训练 **`155k`** steps。

这里的 **`punsafe=0.98`** 表示只剔除被 NSFW classifier 判为极高风险的样本，相比 2.0 使用的 **`0.1`** 阈值，保留的数据范围更广。

<span style="color: rgb(100,37,208); background-color: inherit">数据过滤不是只影响安全边界，它还会改变人物、艺术风格、服装、姿态和词语共现关系的覆盖范围，因此 2.1 即使架构不变，prompt 响应也可能明显不同。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">推理与使用</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">Text-to-Image 推理</span>**

Text-to-Image 从 $$z_T\sim\mathcal{N}(0,I)$$ 开始。每个 timestep 都执行 Text Encoder 条件下的 U-Net 预测，scheduler 根据 epsilon 或 velocity 估计更新 latent。经过预设步数后，VAE Decoder 只运行一次。

采样步数决定 U-Net 调用次数，guidance scale 决定文本条件的放大程度，随机 seed 决定初始噪声。相同 prompt、checkpoint、scheduler、步数、guidance scale 与 seed 才能严格复现同一条采样轨迹。

官方评测使用 **`50`** 个 DDIM steps，并比较 **`1.5、2、3、4、5、6、7、8`** 的 guidance scale。不同 scheduler 会产生不同的离散反向路径，不能只看 step 数量判断质量。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Image-to-Image</span>**

Image-to-Image 不从纯噪声开始。输入图像先由 VAE Encoder 变成 $$z_0$$，再根据 strength 对应的 timestep 加噪得到 $$z_t$$，随后执行文本条件去噪。

较低 strength 保留更多原图构图、颜色和边缘，较高 strength 让 latent 更接近噪声，模型拥有更大的重绘空间。它使用同一个文生图 checkpoint，不需要额外的 Image-to-Image 训练分支。

<span style="color: rgb(100,37,208); background-color: inherit">strength 实际控制的是反向过程从哪个噪声阶段开始，而不是简单调节原图与结果图的像素混合比例。</span>

3. **<span style="color: rgb(36,91,219); background-color: inherit">Depth2Img</span>**

Depth2Img 从 512-base 继续微调 **`200k`** steps。MiDaS 的 **`dpt_hybrid`** 先估计输入图像的相对深度，深度图作为额外通道送入 U-Net。

新增输入通道的权重使用零初始化，因此训练开始时仍保持 base checkpoint 的行为，再逐步学习如何利用深度条件。文本决定主体与风格，深度条件约束前后关系、轮廓和大体几何。

<span style="color: rgb(46,161,33); background-color: inherit">与普通 Image-to-Image 相比，Depth2Img 可以大幅改变纹理和对象类别，同时更稳定地保留原图的三维布局。</span> <span style="color: rgb(216,57,49); background-color: inherit">单目深度只有相对尺度，透明物体、镜面、遮挡和非常规透视会把 MiDaS 的误差一并传给生成模型。</span>

4. **<span style="color: rgb(36,91,219); background-color: inherit">Inpainting</span>**

Inpainting 从 512-base 继续训练 **`200k`** steps。训练 mask 采用 LaMa 的合成策略，U-Net 除了接收 **`4`** 通道带噪 latent，还接收 **`1`** 通道 mask 和 **`4`** 通道 masked-image latent，因此输入通道总数为 **`9`**。

新增的 **`5`** 个通道使用零初始化。mask 指出允许修改的区域，masked-image latent 提供未遮挡区域的上下文，文本则描述填充内容。去噪过程中，模型需要同时满足边界连续性、全局语义和 prompt。

<span style="color: rgb(100,37,208); background-color: inherit">Inpainting 不是先生成一张完整图片再贴回原图，而是在每个去噪步骤中持续读取 mask 与上下文条件。</span>

5. **<span style="color: rgb(36,91,219); background-color: inherit">×4 Upscaler</span>**

×4 Upscaler 是 text-guided latent upscaling diffusion model。它在 LAION 中约 **`10M`** 张原始尺寸大于 **`2048×2048`** 的图像上训练 **`1.25M`** steps，训练 crop 为 **`512×512`**。

模型同时接收低分辨率图像、文本条件和 **`noise_level`**。noise level 按预设 schedule 扰动低分辨率条件，使模型能够适应不同强度的压缩、模糊和噪声，而不是机械执行插值。

这一分支也使用 v-objective。U-Net 预测 velocity，scheduler 再把 velocity 转换为当前 step 所需的干净 latent 与噪声估计。

<span style="color: rgb(46,161,33); background-color: inherit">它可以从低分辨率条件生成清晰纹理，并将宽高各放大 4 倍。</span> <span style="color: rgb(216,57,49); background-color: inherit">新增细节来自生成分布，不等于恢复了输入中真实存在但被删除的信息；在文字、人脸和规则图案上可能生成看似清楚却不准确的内容。</span>

### 4.2.3 <span style="color: rgb(36,91,219); background-color: inherit">SDXL</span>

SDXL 主要是对 Stable Diffusion 架构的改进。这些改进是模块化的，可以单独使用或组合使用以扩展任何模型。这些策略虽然说是潜在扩散模型 LDM 的扩展，但其中大多数也适用于像素空间的对应模型。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型架构</span>**

之前的工作证明了扩散模&#x578B;**`DM`**&#x662F;强大的图像生成模型，卷&#x79EF;**`UNet`**&#x67B6;构一直是基于扩散的图像合成的主导架构。然而，随着基础 **DM&#x20;**&#x7684;发展，底层架构不断演进：从添加自注意力和改进的上采样层，到用于文本到图像合成的交叉注意力，再到纯基于 Transformer 的架构。

SDXL 遵循这一趋势，<span style="color: rgb(100,37,208); background-color: inherit">将大部分 Transformer 计算转移到 UNet 中的低级特征</span>。特别是，与原始的 Stable Diffusion 架构相比，<span style="color: rgb(100,37,208); background-color: inherit">在 UNet 内使用异构分布的 Transformer 块：出于效率原因，在最高特征级别省略了 Transformer 块，在较低级别分别使用</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`10`**<span style="color: rgb(100,37,208); background-color: inherit">个块，并完全移除了 UNet 中最低级别，即</span>**`8×`**<span style="color: rgb(100,37,208); background-color: inherit">下采样</span>。Stable Diffusion 1.x、2.x 和 SDXL 架构的比较如右表。

<span style="color: rgb(100,37,208); background-color: inherit">SDXL 选择了一个更强大的预训练文本编码器</span>来进行文本条件化：结合使&#x7528;**`OpenCLIP ViT-bigG`**&#x548C;**`CLIP ViT-L`**，将倒数第二层文本编码器输出沿通道轴连接。

![](../../images/视觉多模态讲义（下）-image-59.png)

除了使用交叉注意力层对模型进行文本输入条件化外，还根&#x636E;**`OpenCLIP`**&#x6A21;型池化文本 Embedding 来调整模型。这些更改使得 UNet 中的模型参数量达&#x5230;**`2.6B`**，文本编码器的总参数量&#x4E3A;**`817M`**。

* **<span style="color: rgb(36,91,219); background-color: inherit">条件化</span>**

**<span style="color: rgb(36,91,219); background-color: inherit">图像大小的条件化</span>**

LDM 范式的缺点之一是由于其两阶段架构，训练模型需要最小图像尺寸。解决此问题的两种主要方法是：<span style="color: rgb(100,37,208); background-color: inherit">丢弃所有低于某个最小分辨率的训练图像</span>，例如，**`Stable Diffusion 1.4/1.5`**<span style="color: rgb(220,155,4); background-color: inherit">丢弃了所有尺寸低于</span>**`512`**<span style="color: rgb(220,155,4); background-color: inherit">像素的图像</span>，<span style="color: rgb(100,37,208); background-color: inherit">或者放大过小的图像</span>。然而，根据所需图像分辨率的不同，<span style="color: rgb(216,57,49); background-color: inherit">前一种方法可能导致大量训练数据被丢弃，从而导致性能损失并损害泛化能力</span>。如右图，对于这种特定的数据选择，丢弃所有低于分辨率$$256^2$$像素的样本会导&#x81F4;**`39%`**&#x6570;据被丢弃。另一方面，第二种方法通常会<span style="color: rgb(100,37,208); background-color: inherit">引入放大的图像</span>，这可能泄漏到最终模型输出中，导致样本模糊等问题。

![](../../images/视觉多模态讲义（下）-image-54.png)

相反，SDXL 在 UNet 模型中对原始图像分辨率进行条件化，这在训练期间很容易获得。具体来说，<span style="color: rgb(100,37,208); background-color: inherit">将原始图像的高度和宽度作为额外条件提供给模型</span>，表示为$$c_\text{size} = (h_{\text{original}}, w_{\text{original}})$$。每个组件都通过傅里叶特征编码独立嵌入，这些编码被连接成一个向量，并通过将其添加到 Timestep Embedding 中馈送到模型。

SDXL 通过在空间大小为$$512^2$$的 ImageNet 上训练和评估三个 LDM 来定量评估这种简单但有效的条件化技术的效果：

> 1. 对于第一个模&#x578B;**`CIN-512-only`**，<span style="color: rgb(100,37,208); background-color: inherit">丢弃所有至少一边小于</span>**`512`**<span style="color: rgb(100,37,208); background-color: inherit">像素的训练样本</span>，结果得到仅包&#x542B;**`7`**&#x4E07;张图像的训练数据集。
>
> 2. 对&#x4E8E;**`CIN-nocond`**，<span style="color: rgb(100,37,208); background-color: inherit">使用所有训练样本，但不进行大小条件化</span>。训练完成后，使&#x7528;**`50`**&#x6B65;**`DDIM`**&#x548C;比例&#x4E3A;**`5`**&#x7684;无分类器引导，为每个模型生&#x6210;**`5000`**&#x4E2A;样本，并计&#x7B97;**`IS`**&#x548C;**`FID`**。
>
> 3. 对&#x4E8E;**`CIN-size-cond`**，始终以$$c_\text{size} = (512, 512)$$进行条件化生成样本。

右表总结了结果，并验证&#x4E86;**`CIN-size-cond`**&#x5728;两项指标上均优于基线模型。**`CIN-512-only`**&#x7684;性能下降可能是因为<span style="color: rgb(100,37,208); background-color: inherit">对小型训练数据集的过拟合</span>，&#x800C;**`CIN-nocond`**&#x6837;本分布中<span style="color: rgb(100,37,208); background-color: inherit">模糊样本模式的影响导致 FID 分数降低</span>。

![](../../images/视觉多模态讲义（下）-image-55.png)

**<span style="color: rgb(36,91,219); background-color: inherit">裁剪参数的条件化</span>**

右图前两行展示了先前 SD 模型的典型失败模式：合成对象被裁剪，例如<span style="color: rgb(220,155,4); background-color: inherit"> SD 1-5 和 SD 2-1 的猫头被切断</span>。这是因为<span style="color: rgb(216,57,49); background-color: inherit">模型训练期间使用了随机裁剪</span>：在深度学习框架中一个批次需要相同大小的张量，因此典型的处理流程是：<span style="color: rgb(216,57,49); background-color: inherit">调整图像大小，使最短边匹配目标尺寸，然后沿较长轴随机裁剪图像</span>。虽然随机裁剪是一种自然的数据增强形式，但它可能泄漏到生成样本中，导致上述恶意效果。

![](../../images/视觉多模态讲义（下）-image-56.png)

为了解决这个问题，SDXL 提出了另一种简单而有效的条件化方法：在数据加载期间，<span style="color: rgb(100,37,208); background-color: inherit">均匀采样裁剪坐标</span>$$c_\text{top}$$<span style="color: rgb(100,37,208); background-color: inherit">和 </span>$$c_\text{left}$$<span style="color: rgb(100,37,208); background-color: inherit">，分别指定沿高度和宽度轴从左上角裁剪的像素数量，并通过傅里叶特征嵌入将其作为条件参数馈送到模型中</span>，类似于上述大小条件化。连接后的嵌入$$c_\text{crop}$$用作附加条件参数。这种技术不仅限于 LDM，还可用于任何 DM。<span style="color: rgb(100,37,208); background-color: inherit">这里裁剪和大小条件化可以轻松组合，在添加到 UNet 的 timestep 之前，沿通道维度连接特征 Embedding</span> 。右图伪代码展示了如何在训练期间如何采样 $$c_\text{crop}$$ 和 $$c_\text{size}$$。

![](../../images/视觉多模态讲义（下）-image-80.png)

由于大规模数据集通常内容是在中心的，因此<span style="color: rgb(100,37,208); background-color: inherit">在推理期间设置</span>$$(c_\text{top}, c_\text{left}) = (0, 0)$$<span style="color: rgb(100,37,208); background-color: inherit">，从训练模型中获得对象居中的样本。通过调整</span>$$(c_\text{top}, c_\text{left})$$<span style="color: rgb(100,37,208); background-color: inherit">，可以成功模拟推理期间的裁剪量</span>。这是一种条件增强形式，已在各种自回归模型和扩散模型中以多种形式使用。

SDXL 受益于裁剪引起的数据增强，同时确保它不会泄漏到生成过程中，并且可以<span style="color: rgb(46,161,33); background-color: inherit">利用它来更好地控制图像合成过程。此外，它易于实现，可以在训练期间在线应用，无需额外的数据预处理</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">多长宽比的训练</span>**

现实世界的数据集包含尺寸和宽高比变化极大的图像，如上所述。尽管<span style="color: rgb(216,57,49); background-color: inherit">文本到图像模型的常见输出分辨率是 </span>$$512 \times 512$$<span style="color: rgb(216,57,49); background-color: inherit">或</span>$$1024 \times 1024$$<span style="color: rgb(216,57,49); background-color: inherit">的方形图像，但这是一个很不自然的选择，因为例如</span>$$16:9$$<span style="color: rgb(216,57,49); background-color: inherit">的照片或长方形屏幕广泛分布和使用</span>。

基于这一想法，作者对模型进行微调以同时处理多种宽高比：遵循常见的做法，<span style="color: rgb(100,37,208); background-color: inherit">将数据按不同宽高比分成多个桶，尽量使像素总数接近</span>$$1024^2$$<span style="color: rgb(100,37,208); background-color: inherit">像素，并根据需要调整高度和宽度，且调整步长为</span>**`64`**<span style="color: rgb(100,37,208); background-color: inherit">的倍数</span>，所有长宽比如右图。在优化过程中，一个训练批次由来自同一桶的图像组成，并在每个训练步骤中交替切换桶大小。此外，模型接收目标大小作为条件输入，表示为整数元组$$c_\text{ar} = (h_{\text{tgt}}, w_{\text{tgt}})$$，这些值通过傅里叶空间嵌入，类似于上述大小和裁剪条件化。<span style="color: rgb(100,37,208); background-color: inherit">SDXL 将多纵横比训练作为预训练阶段后的微调步骤，预训练阶段是在固定长宽比和分辨率下完成的</span>，并通过沿通道连接的方式与条件化技术结合使用。裁剪条件化和多长宽比训练是互补的操作。

![](../../images/视觉多模态讲义（下）-image-79.png)

![](../../images/视觉多模态讲义（下）-image-77.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">改进的自编码器</span>**

Stable Diffusion 是一种 LDM，运行在预训练、可学习的和固定的自编码器潜在空间中。虽然大部分语义合成是由 LDM 完成的，但通过改进自编码器可以提升生成图像中的局部高频细节。

![](../../images/视觉多模态讲义（下）-image-74.png)

为此，SDXL 使用更大的批量大小，&#x5C06;**`9`**&#x6539;&#x4E3A;**`256`**，训练与原始 Stable Diffusion 相同的自编码器架构，并额外使用指数移动平均跟踪权重。由此产生的自编码器在所有评估的重建指标上均优于原始模型，如右表。

* **<span style="color: rgb(36,91,219); background-color: inherit">整体结构</span>**

通过多阶段过程训练最终模型 SDXL。SDXL 使用上述提到的自编码器和离散时间扩散步骤，&#x5373;**`1000`**&#x6B65;。然后使用多阶段训练：

> 1. 在内部数据集上预训练一个基础模型，然后在$$256 \times 256$$分辨率下进&#x884C;**`60`**&#x4E07;次优化步骤，批量大小&#x4E3A;**`2048`**，并使用大小和裁剪条件化。
>
> 2. 在$$512 \times 512$$分辨率的图像上继续训&#x7EC3;**`20`**&#x4E07;次优化步骤，
>
> 3. 利用多长宽比训练结合$$0.05$$的偏移噪声水平，在约$$1024 \times 1024$$像素面积的不同长宽比下训练模型。

**<span style="color: rgb(36,91,219); background-color: inherit">精修阶段</span>**

为了提高样本质量，作者在同一潜在空间中训练了一个单独的 LDM，该模型专注于高质量、高分辨率数据，并对基础模型的样本应用 SDEdit 引入的噪声-去噪过程。这里遵循之前的工作，并将此精修模型专门用于前 200 个噪声尺度。在推理时，从基础 SDXL 渲染潜在变量，并在潜在空间中直接使用精修模型对其进行扩散和去噪，使用相同的文本输入。这一步骤是可选的，但它显著提升了详细背景和人脸的样本质量。

为了评估模型有无精修阶段的性能，作者进行了用户研究，让用户从以下四个模型中选择他们最喜欢的生成结果：<span style="color: rgb(100,37,208); background-color: inherit">SDXL、带精修的 SDXL、Stable Diffusion 1.5 和 Stable Diffusion 2.1</span>。

![](../../images/视觉多模态讲义（下）-image-78.png)

结果显示，带有精修阶段的 SDXL 是最受欢迎的选择，显著优于 Stable Diffusion 1.5 和 2.1：SDXL 带精修：**`48.44%`**，SDXL 基础：**`36.93%`**，Stable Diffusion 1.5：**`7.91%`**，Stable Diffusion 2.1：**`6.71%`**）。

### 4.2.4 <span style="color: rgb(36,91,219); background-color: inherit">Stable Diffusion 3</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">流模型训练</span>**

生成模型通过一个常微分方程 ODE 定义从噪声分布$$p_1$$中采样得到的样本$$x_1$$到数据分布$$p_0$$中采样得到的样本$$x_0$$之间的映射：

$$dy_t = v_\Theta(y_t, t) dt$$

其中速度场$$v$$由神经网络的参数$$\Theta$$参数化。<span style="color: rgb(216,57,49); background-color: inherit">先前的工作是直接通过可微分 ODE 求解器求解该方程，但计算代价高，尤其当</span>$$v_\Theta(y_t, t)$$<span style="color: rgb(216,57,49); background-color: inherit">由大型网络架构参数化时</span>。一种更高效的替代方法是<span style="color: rgb(100,37,208); background-color: inherit">直接回归一个向量场</span>$$u_t$$<span style="color: rgb(100,37,208); background-color: inherit">，生成</span>$$p_0$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$p_1$$<span style="color: rgb(100,37,208); background-color: inherit">之间的概率路径</span>。

为构造这样的$$u_t$$，要定义一个前向过程，对应于$$p_0$$与$$p_1 = \mathcal{N}(0, I)$$之间的概率路径$$p_t$$：

$$z_t = a_t x_0 + b_t \epsilon, \quad  \quad \epsilon \sim \mathcal{N}(0, I)$$

当$$a_0 = 1, b_0 = 0, a_1 = 0, b_1 = 1$$时，边缘分布

$$p_t(z_t) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} p_t(z_t|\epsilon)$$

与数据分布和噪声分布一致。

这里为了表达$$z_t$$、$$x_0$$与$$\epsilon$$之间的关系，引入$$\psi_t$$和$$u_t$$：

$$\psi_t(\cdot|\epsilon) : x_0 \mapsto a_t x_0 + b_t \epsilon$$

$$u_t(z|\epsilon) := \psi'_t(\psi^{-1}_t(z|\epsilon)|\epsilon)$$

由于$$z_t$$可写作 ODE $$z'_t = u_t(z_t|\epsilon)$$的解，且初始值$$z_0 = x_0$$，因此$$u_t(\cdot|\epsilon)$$<span style="color: rgb(100,37,208); background-color: inherit">生成条件概率路径 </span>$$p_t(\cdot|\epsilon)$$<span style="color: rgb(100,37,208); background-color: inherit">。这里可以利用条件向量场</span>$$u_t(\cdot|\epsilon)$$<span style="color: rgb(100,37,208); background-color: inherit">构造一个边缘向量场</span>$$u_t$$，生成边缘概率路径$$p_t$$：

$$u_t(z) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} u_t(z|\epsilon) \frac{p_t(z|\epsilon)}{p_t(z)}$$

由于上式中的边缘化，直接通过流匹配目标

$$L_{\text{FM}} = \mathbb{E}_{t,p_t(z)} \|v_\Theta(z, t) - u_t(z)\|_2^2$$

回归$$u_t$$是不可行的，但<span style="color: rgb(216,57,49); background-color: inherit">条件流匹配 Conditional Flow Matching</span> 目标

$$L_{\text{CFM}} = \mathbb{E}_{t,p_t(z|\epsilon),p(\epsilon)} \|v_\Theta(z, t) - u_t(z|\epsilon)\|_2^2$$

使用条件向量场$$u_t(z|\epsilon)$$提供了一个等价且可计算的目标。

为将损失显式化，将$$\psi'_t(x_0|\epsilon) = a'_t x_0 + b'_t \epsilon$$和$$\psi^{-1}_t(z|\epsilon) = \frac{z - b_t \epsilon}{a_t}$$代入$$u_t(z|\epsilon) := \psi'_t(\psi^{-1}_t(z|\epsilon)|\epsilon)$$，得到：

$$z'_t = u_t(z_t|\epsilon) = \frac{a'_t}{a_t} z_t - \epsilon b_t \left( \frac{a'_t}{a_t} - \frac{b'_t}{b_t} \right)$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：信噪比为$$\lambda_t := \log \frac{a_t^2}{b_t^2}$$

由于 $$\lambda'_t = 2\left( \frac{a'_t}{a_t} - \frac{b'_t}{b_t} \right)$$，可以将上式重写为：

$$u_t(z_t|\epsilon) = \frac{a'_t}{a_t} z_t - \frac{b_t}{2} \lambda'_t \epsilon$$

然后用上式将$$L_{\text{CFM}} = \mathbb{E}_{t,p_t(z|\epsilon),p(\epsilon)} \|v_\Theta(z, t) - u_t(z|\epsilon)\|_2^2$$重新参数化为噪声预测目标：

$$L_{\text{CFM}} = \mathbb{E}_{t,p_t(z|\epsilon),p(\epsilon)} \left\| v_\Theta(z, t) - \frac{a'_t}{a_t} z + \frac{b_t}{2} \lambda'_t \epsilon \right\|_2^2 = \mathbb{E}_{t,p_t(z|\epsilon),p(\epsilon)} \left( -\frac{b_t}{2} \lambda'_t \right)^2 \| \epsilon_\Theta(z, t) - \epsilon \|_2^2$$

这里定义$$\epsilon_\Theta := -\frac{2}{\lambda'_t b_t} \left( v_\Theta - \frac{a'_t}{a_t} z \right)$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：<span style="color: rgb(100,37,208); background-color: inherit">这个目标的最优解在引入时间相关权重后不会改变。因此，可以推导出各种加权损失函数，它们均指向期望解</span>，但可能影响优化轨迹。为了能分析不同方法，包括经典扩散模型，可以将目标写成：
>
> $$L_w(x_0) = -\frac{1}{2} \mathbb{E}_{t \sim U(t), \epsilon \sim \mathcal{N}(0,I)} \left[ w_t \lambda'_t \| \epsilon_\Theta(z_t, t) - \epsilon \|^2 \right]$$
>
> 其中 $$w_t = -\frac{1}{2} \lambda'_t b_t^2$$&#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">Flow Trajectories</span>**

上述形式有很多不同变体：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Rectified Flow（RF）</span>**：定义前向过程为<span style="color: rgb(100,37,208); background-color: inherit">数据分布与标准正态分布之间的直线路径</span>，即：
>
> $$z_t = (1 - t) x_0 + t \epsilon$$
>
> 并使用条件流匹配目标$$L_{\text{CFM}}$$，此时对应的权重为$$w_t^{\text{RF}} = \frac{t}{1 - t}$$。网络输出直接参数化速度场$$v_\Theta$$。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">EDM</span>**：使用如下形式的前向过程：
>
> $$z_t = x_0 + b_t \epsilon$$
>
> 其中$$b_t = \exp F_N^{-1}(t|P_m, P_s^2)$$，$$F_N^{-1}$$为均值$$P_m$$、方差$$P_s^2$$的正态分布的分位函数。这会导致：
>
> $$\lambda_t \sim \mathcal{N}(-2P_m, (2P_s)^2), \quad\quad t \sim U(0,1)$$
>
> 网络通&#x8FC7;**`F-prediction`**&#x53C2;数化，损失可写为$$L_{w_t}^{\text{EDM}}$$，其中：
>
> $$w_t^{\text{EDM}} = \mathcal{N}(\lambda_t | -2P_m, (2P_s)^2) \left( e^{-\lambda_t} + 0.5^2 \right)$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Cosine</span>**：前向过程如下：
>
> $$z_t = \cos\left( \frac{\pi}{2} t \right) x_0 + \sin\left( \frac{\pi}{2} t \right) \epsilon$$
>
> 结&#x5408;**`ϵ-parameterization`**&#x4E0E;损失函数，其对应权重为$$w_t = \text{sech}(\lambda_t / 2)$$。若结&#x5408;**`v-prediction`**&#x635F;失，则权重为$$w_t = e^{-\lambda_t / 2}$$。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">LDM-Linear</span>**：<span style="color: rgb(100,37,208); background-color: inherit">LDM 使用对 DDPM 调度的修改</span>。两者均为方差保持调度，即$$b_t = \sqrt{1 - a_t^2}$$，并基于扩散系数$$\beta_t$$定义离散时间步$$t = 0, \dots, T-1$$下的$$a_t = \left( \prod_{s=0}^t (1 - \beta_s) \right)^{1/2}$$。给定边界值$$\beta_0$$和$$\beta_{T-1}$$，DDPM 使用线性插值 $$\beta_t = \beta_0 + \frac{t}{T-1} (\beta_{T-1} - \beta_0)$$，而 <span style="color: rgb(100,37,208); background-color: inherit">LDM 使用平方根插值</span>$$\beta_t = \left( \sqrt{\beta_0} + \frac{t}{T-1} (\sqrt{\beta_{T-1}} - \sqrt{\beta_0}) \right)^2$$。

RF 损失在区间$$[0,1]$$上对所有时间步均匀训练速度场$$v_\Theta$$。然而速度预测目标$$\epsilon - x_0$$在中间时间步更难预测：<span style="color: rgb(100,37,208); background-color: inherit">当</span>$$t=0$$<span style="color: rgb(100,37,208); background-color: inherit">时最优预测为</span>$$p_1$$<span style="color: rgb(100,37,208); background-color: inherit">的均值，当</span>$$t=1$$<span style="color: rgb(100,37,208); background-color: inherit">时为</span>$$p_0$$<span style="color: rgb(100,37,208); background-color: inherit">的均值。将时间步分布从均匀分布</span>$$U(t)$$<span style="color: rgb(100,37,208); background-color: inherit">改为密度为</span>$$\pi(t)$$<span style="color: rgb(100,37,208); background-color: inherit">的分布，等价于加权损失</span>$$L_{w_t^\pi}$$，其中：

$$w_t^\pi = \frac{t}{1 - t} \pi(t)$$

因此通过对中间时间步更高频采样来赋予其更大权重。以下是用于训练模型的时间步密度$$π(t)$$ ：

> **<span style="color: rgb(36,91,219); background-color: inherit">Logit-Normal Sampling</span>**：在中间步骤赋予更高权重的分布。密度为：
>
> $$\pi_{\text{ln}}(t; m, s) = \frac{1}{s \sqrt{2\pi}} \frac{1}{t(1 - t)} \exp\left( -\frac{(\text{logit}(t) - m)^2}{2s^2} \right)$$
>
> 其中$$\text{logit}(t) = \log \frac{t}{1 - t}$$，$$m$$为位置参数，$$s$$为尺度参数。位置参数使训练时间步偏向数据$$p_0$$（此时$$m<0$$）或噪声$$p_1$$（此时$$m>0$$）。如下图，尺度参数控制分布宽度。一般从正态分布$$u \sim \mathcal{N}(m, s)$$中采样，并通过标&#x51C6;**`Logistic`**&#x51FD;数映射。

![](../../images/视觉多模态讲义（下）-image-66.png)

> **<span style="color: rgb(36,91,219); background-color: inherit">Mode Sampling with Heavy Tails</span>**：logit-normal density 在端点$$0$$和$$1$$处始终为零。这里作者研究这是否对性能有负面影响，因此使用在$$[0,1]$$上严格为正的 density。对尺度参数$$s$$，定义：
>
> $$f_{\text{mode}}(u; s) = 1 - u - s \cdot \left( \cos^2\left( \frac{\pi}{2} u \right) - 1 + u \right)$$
>
> 当$$-1 \leq s \leq \frac{2}{\pi - 2}$$时，该函数单调，可从中采样 implied density：
>
> $$\pi_{\text{mode}}(t; s) = \left| \frac{d}{dt} f_{\text{mode}}^{-1}(t) \right|$$
>
> 上图中尺度参数控制采样时偏向中点$$s>0$$或端点$$s<0$$。当$$s=0$$时，该公式退化为均匀权重 $$\pi_{\text{mode}}(t; 0) = U(t)$$，这&#x5728;**`Rectified Flow`**&#x5E7F;泛使用。

> **<span style="color: rgb(36,91,219); background-color: inherit">CosMap</span>**：在 RF 设定下尝&#x8BD5;**`Cosine`**&#x4F59;弦调度。这里作者寻找映射$$f: u \mapsto t = f(u)$$，$$u \in [0,1]$$，使得对数信噪比匹配余弦调度：$$2 \log \frac{\cos(\frac{\pi}{2} u)}{\sin(\frac{\pi}{2} u)} = 2 \log \frac{1 - f(u)}{f(u)}$$。求解$$f$$，对$$u \sim U(u)$$得：
>
> $$t = f(u) = 1 - \frac{1}{\tan(\frac{\pi}{2} u) + 1}$$
>
> 由此得到密度：
>
> $$\pi_{\text{CosMap}}(t) = \left| \frac{d}{dt} f^{-1}(t) \right| = \frac{2}{\pi - 2\pi t + 2\pi t^2}$$

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

为了实现文本条件图像生成，模型需同时处理文本与图像两种模态。作者使用预训练模型提取表征，整体结构如下图：

![](../../images/视觉多模态讲义（下）-image-67.png)

MMDiT 最初是 Stable Diffusion 3 的核心架构，后来成为很多出色扩散模型的核心架构。其目标是<span style="color: rgb(100,37,208); background-color: inherit">用 Transformer 来进行扩散去噪，同时支持图像 + 文本两个模态。它将</span>**`text token`**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**`image latent token`**<span style="color: rgb(100,37,208); background-color: inherit">映射到一个统一 embedding 空间，通过 Transformer 的 self-attention 来处理模态间以及模态内交互</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">双流设计</span>**

> * MMDiT 对文本和图像采用分支结构：<span style="color: rgb(100,37,208); background-color: inherit">图像 token 和文本 token 各自有自己的</span>$$Q/K/V$$<span style="color: rgb(100,37,208); background-color: inherit">投影，以及 MLP 子层</span>。这样可以让每个模态有自己的特征变换路径
>
> * 在 attention 层中，<span style="color: rgb(100,37,208); background-color: inherit">将图像 token 序列和文本 token 序列两种模态拼接，然后执行 self-attention</span>，从而实现跨模态的信息交互
>
> * attention 输出后，再<span style="color: rgb(100,37,208); background-color: inherit">分别送回图像分支和文本分支的 MLP</span>，以保持每个模态特有的表示能力

这种设计的好处是：**<span style="color: rgb(100,37,208); background-color: inherit">既能通过各自 MLP 和各自的 QKV 保留模态专属变换能力，又能通过拼接 attention 实现跨模态融合</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">Attention 分块</span>**

在 MMDiT 的 self-attention 中，由于拼接了文本和图像 token，因此 attention 矩阵可以被视为四种子类型：

> * **<span style="color: rgb(36,91,219); background-color: inherit">I→I</span>**：image-to-image
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">T→T</span>**：text-to-text
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">I→T</span>**：image-to-text
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">T→I</span>**：text-to-image

这种分块结构非常重要，因为它明确地建模了跨模态（**<span style="color: rgb(100,37,208); background-color: inherit">文本 ↔ 图像</span>**）和模态内（**<span style="color: rgb(100,37,208); background-color: inherit">文本 ↔ 文本、图像 ↔ 图像</span>**）的交互。**<span style="color: rgb(46,161,33); background-color: inherit">I→T</span>**<span style="color: rgb(46,161,33); background-color: inherit"> / </span>**<span style="color: rgb(46,161,33); background-color: inherit">T→I</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 使得图像 token 可以关注文本 token，同样文本 token 也可以关注图像 token，从而实现更强的语义对齐与控制</span>。同时，<span style="color: rgb(46,161,33); background-color: inherit">保留 </span>**<span style="color: rgb(46,161,33); background-color: inherit">I→I</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 自注意力可以维持图像内部结构的一致性</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">条件控制</span>**

> - 使用时间&#x6B65;**`timestep`**&#x4F5C;为条件注入模型，也就是扩散模型中时间嵌&#x5165;**`time embedding`**&#x4F1A;被引入到 Transformer 层中
>
> - 文本条&#x4EF6;**`prompt embedding`**&#x901A;过 attention 影响图像 token
>
> - 为了融合不同模态的信息，在 Transformer 层里用归一化，&#x5982;**`LayerNorm`**、**`AdaLN`**&#x6765;调节各 token 的表示

> 注：在一些实现里，如 PyPI &#x7684;**`mmdit-pytorch`**，可以看&#x5230;**`MMDiTBlock(dim_cond, dim_text, dim_image)`** 这样的模块，其&#x4E2D;**`dim_cond`**&#x53EF;用于时间条件注入。&#x20;

* **<span style="color: rgb(36,91,219); background-color: inherit">位置编码</span>**

> - 对于图像 token，由于它们是从 VAE latent 空间里提取出的 patch，MMDiT 通常对这些 patch 使用 **`2×2`&#x20;**&#x70;atch &#x7684;**<span style="color: rgb(100,37,208); background-color: inherit">绝对位置编码</span>**&#x6765;保留图像空间结构
>
> - 对文本 token 则使用 Transformer 常见的位置编码：类似 T5 &#x7684;**<span style="color: rgb(100,37,208); background-color: inherit">相对位置编码</span>**&#x6765;保持文本顺序语义
>
> - 不同模态的位置编码需要尺度对齐，以避免数值上的冲突

整体网络结构遵循 LDM，在预训练自编码器的潜在空间中训练文本到图像模型。与图像编码类似，遵循先前方法，使用预训练、冻结的文本模型编码文本条件$$c$$。细节如下：

> **<span style="color: rgb(222,120,2); background-color: inherit">图像表征</span>**：遵循 LDM 的方法，使用预训练的自编码器将 RGB 图像$$X \in \mathbb{R}^{H \times W \times 3}$$映射到一个更小的潜在空间$$x = E(X) \in \mathbb{R}^{h \times w \times d}$$。采用空间下采样因子$$8$$，即$$h = \frac{H}{8}$$、$$w = \frac{W}{8}$$。这里<span style="color: rgb(100,37,208); background-color: inherit">在潜在空间中应用</span>$$z_t = a_t x_0 + b_t \epsilon$$<span style="color: rgb(100,37,208); background-color: inherit">的前向过程，当通过</span>$$dy_t = v_\Theta(y_t, t) dt$$<span style="color: rgb(100,37,208); background-color: inherit">采样得到潜在表示</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">后，再通过解码器</span>$$D$$<span style="color: rgb(100,37,208); background-color: inherit">将其解码回像素空间</span>$$X = D(x)$$。然后对潜在变量进行全局归一化，均值和标准差基于训练数据的一个子集计算得出。右图是在不同$$d$$值下，训练随模型容量变化的趋势
>
> ![](../../images/视觉多模态讲义（下）-image-75.png)
>
> **<span style="color: rgb(222,120,2); background-color: inherit">文本表征</span>**：使用预训练且冻结的文本模型对文本条件$$c$$进行编码，即结合使用 CLIP 与 T5。
>
> 1. 使&#x7528;**`CLIP L/14`**&#x548C;**`OpenCLIP bigG/14`**&#x7684;文本编码器对$$c$$进行编码，然后将两个模型的池化输出拼接，其维度分别&#x4E3A;**`768`**&#x548C;**`1280`**，得到向量条件$$c_{\text{vec}} \in \mathbb{R}^{2048}$$
>
> 2. 将两个模型倒数第二层的隐藏表示沿通道维度拼接，得到 CLIP 上下文条件$$c^{\text{CLIP}}_{\text{ctxt}} \in \mathbb{R}^{77 \times 2048}$$
>
> 3. 使&#x7528;**`T5-v1.1-XXL`**&#x7684;编码器将$$c$$编码为最终表示$$c^{\text{T5}}_{\text{ctxt}} \in \mathbb{R}^{77 \times 4096}$$
>
> 4. 将 $$c_{\text{CLIP}}^{\text{ctxt}}$$ 沿通道维度零填充至 4096 维，以匹配 T5 表示，并沿序列维度与$$c^{\text{T5}}_{\text{ctxt}}$$拼接，得到最终的上下文表示$$c_{\text{ctxt}} \in \mathbb{R}^{154 \times 4096}$$

SD3 的架构是基&#x4E8E;**`DiT`**&#x7684;，其仅考虑类别条件图像生成，使用调制机制将扩散时间步和类别标签作为网络条件。类似地，作者使用时间步$$t$$和$$c_{\text{vec}}$$的嵌入作为调制机制的输入。然而，<span style="color: rgb(216,57,49); background-color: inherit">由于池化文本表征仅保留文本输入的粗粒度信息，网络还需序列表征</span>$$c_{\text{ctxt}}$$<span style="color: rgb(216,57,49); background-color: inherit">的信息</span>。因此<span style="color: rgb(100,37,208); background-color: inherit">构建了一个包含文本与图像输入嵌入的序列，为潜在像素表征</span>$$x \in \mathbb{R}^{h \times w \times c}$$<span style="color: rgb(100,37,208); background-color: inherit">添加位置编码，将其展平为</span>$$2 \times 2$$<span style="color: rgb(100,37,208); background-color: inherit">的图块序列，长度为</span>$$\frac{1}{2} h \cdot \frac{1}{2} w$$。将该图块编码与文本编码$$c_{\text{ctxt}}$$嵌入至共同维度后，将两个序列拼接。然后和 DiT 相同的，应用一系列注意力和 MLP 模块。

由于文本与图像嵌入概念差异较大，<span style="color: rgb(100,37,208); background-color: inherit">作者为两种模态使用两组独立权重，这等价于为每种模态使用独立的Transformer，在注意力操作中合并两个模态的序列，使得两种表征既可在各自空间中运作，又能相互参考</span>。

在 scaling 实验中，以模型深度$$d$$，即注意力块数量，参数化模型规模：<span style="color: rgb(100,37,208); background-color: inherit">隐藏层大小设为</span>$$64 \cdot d$$<span style="color: rgb(100,37,208); background-color: inherit">，MLP 块中扩展为 </span>$$4 \cdot 64 \cdot d$$<span style="color: rgb(100,37,208); background-color: inherit"> 通道，注意力头数设为</span>$$d$$。

### 4.2.5 <span style="color: rgb(36,91,219); background-color: inherit">FLUX.1</span>

![](../../images/视觉多模态讲义（下）-image-68.png)

在扩散模型的发展过程中，**FLUX.1** 是非常有代表性的作品，其来自 **<span style="color: rgb(216,57,49); background-color: inherit">BFL</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">B</span>**<span style="color: rgb(216,57,49); background-color: inherit">lack </span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">orest </span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">abs）</span>，由 Stable Diffusion 3 的核心团队打造，可以看作是 SD3 的下一代。与一般的文本到图像扩散模型不同的是，**<span style="color: rgb(46,161,33); background-color: inherit">FLUX.1</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 在架构、训练方式等方面做了大量工程优化，目标是同时兼顾</span>**<span style="color: rgb(46,161,33); background-color: inherit">高质量</span>**<span style="color: rgb(46,161,33); background-color: inherit">与</span>**<span style="color: rgb(46,161,33); background-color: inherit">高效率</span>**。

目前 FLUX.1 有三个规模的模型：**`FLUX.1-pro`**、**`FLUX.1-dev`**、**`FLUX.1-schnell`**，效果逐渐变差，但速度逐渐变快。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

**FLUX.1** 的核心是一&#x4E2A;**<span style="color: rgb(100,37,208); background-color: inherit">流匹配驱动的扩散网络</span>**，整体上由三部分组成：文本编码器 **<span style="color: rgb(36,91,219); background-color: inherit">Text Encoder</span>**、图像 **<span style="color: rgb(36,91,219); background-color: inherit">VAE</span>**、网络主体 **<span style="color: rgb(36,91,219); background-color: inherit">DiT</span>**。

1. **<span style="color: rgb(36,91,219); background-color: inherit">Text Encoder</span>**

**Stable Diffusion 3&#x20;**&#x7684; **Text Encoder&#x20;**&#x4E00;共使用&#x4E86;**`CLIP ViT-L`**、**`OpenCLIP ViT-bigG`**、**`T5-XXL Encoder`**&#x4E09;&#x4E2A;**&#x20;Text Encoder** 模型。

> * 两个 **CLIP Encoder** 提取&#x7684;**`Pooling Text Embedding`**&#x7279;征拼接在一起后&#x4E0E;**`Time Embedding`**&#x76F8;加。
>
> * 两个 **CLIP Encoder** &#x7684;**`Text Embedding`**&#x7279;征进行拼接，在 **Token&#x20;**&#x7EF4;度与 **T5-XXL&#x20;**&#x7684;**`Text Embedding`**&#x62FC;接后送入 **`MM-DiT`** 中

**FLUX.1** 对 **SD3** 的 **Text Encoder** 部分进行了精简优化，<span style="color: rgb(100,37,208); background-color: inherit">只使用了</span>**`CLIP ViT-L`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`T5-XXL Encoder`**<span style="color: rgb(100,37,208); background-color: inherit">两个 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Text Encoder</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 模型，没有用</span>**`OpenCLIP ViT-bigG`**<span style="color: rgb(100,37,208); background-color: inherit">模型：</span>**<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1 </span>**<span style="color: rgb(100,37,208); background-color: inherit">将 </span>**<span style="color: rgb(100,37,208); background-color: inherit">CLIP ViT-L</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的</span>**`Pooling Text Embeddings`**<span style="color: rgb(100,37,208); background-color: inherit">特征与</span>**`Time Embedding`**<span style="color: rgb(100,37,208); background-color: inherit">相加，同时 </span>**<span style="color: rgb(100,37,208); background-color: inherit">T5-XXL</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 提取的</span>**`Text Embedding`**<span style="color: rgb(100,37,208); background-color: inherit">特征直接送入 </span>**<span style="color: rgb(100,37,208); background-color: inherit">MM-DiT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 中</span>。

> **SD3&#x20;**&#x4E2D;**`CLIP Encoder`**&#x7684;特征有较大的作用，<span style="color: rgb(220,155,4); background-color: inherit">可以去掉</span>**<span style="color: rgb(220,155,4); background-color: inherit"> T5-XXL</span>**<span style="color: rgb(220,155,4); background-color: inherit"> 只用 </span>**<span style="color: rgb(220,155,4); background-color: inherit">CLIP Encoder </span>**<span style="color: rgb(220,155,4); background-color: inherit">提取文本特征信息来生成图像</span>。但是 FLUX.1 的改进使得 **<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 比 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SD3</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 更依赖 </span>**<span style="color: rgb(100,37,208); background-color: inherit">T5-XXL</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 提取的文本特征信息</span>。
>
> **`FLUX.1-dev`**&#x548C;**`FLUX.1-schnell`**&#x4E24;个版本的 **Text Encoder** 部分的结构完全一致。

这里使用 **<span style="color: rgb(36,91,219); background-color: inherit">@Rocky Ding</span>** 大佬的图来说明：

![FLUX.1-dev/schnell CLIP ViT-L Text Encoder 网络结构图](../../images/视觉多模态讲义（下）-image-69.png)

![FLUX.1-dev/schnell T5-XXL Text Encoder 完整结构图](../../images/视觉多模态讲义（下）-image-70.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">VAE</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1 VAE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 架构继承了 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SD3 VAE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的</span>**`8`**<span style="color: rgb(100,37,208); background-color: inherit">倍下采样，但在 </span>**<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1 VAE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 输出的 Latent 特征输入扩散模型前，进行了</span>**`Pack Latent`**<span style="color: rgb(100,37,208); background-color: inherit">操作，将 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Latent</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 特征通道数由</span>**`16`**<span style="color: rgb(100,37,208); background-color: inherit">提高到</span>**`64`**，即 **FLUX.1&#x20;**&#x6269;散模型输入通道数是 **64**，是 **SD3** &#x7684;**`4`**&#x500D;。也就是说 **FLUX.1&#x20;**&#x8981;学习拟合的内容是 **SD3** &#x7684;**`4`**&#x500D;，即 **<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 增加了模型参数量级来提升模型容量</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**`Pack Latent`**&#x64CD;作如下：
>
> * **<span style="color: rgb(100,37,208); background-color: inherit">SD3</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 使用下采样卷积</span>来实现 Latent 特征 Patch 化，会<span style="color: rgb(216,57,49); background-color: inherit">通过卷积减少空间分辨率损失一定的特征信息</span>。
>
> * **<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 将像素块直接在通道维度上堆叠</span>实现 Latent 特征 Patch 化。<span style="color: rgb(46,161,33); background-color: inherit">保留了每个像素块的原始分辨率，只将它们从空间维度移动到了通道维度</span>。

**FLUX.1** 在过扩散模型之前将$$2\times2$$特征 Patch 化，这与 **SD3** 的风格是一样的，并在此基础上进行了继承与优化。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**`FLUX.1-dev`**&#x548C;**`FLUX.1-schnell`**&#x4E24;个版本&#x7684;**&#x20;VAE** 结构完全一致。
>
> **<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1 VAE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 没有直接沿用 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SD3 VAE</span>**<span style="color: rgb(100,37,208); background-color: inherit">，而是基于相同结构进行了重新训练，权重不一样</span>。
>
> **<span style="color: rgb(100,37,208); background-color: inherit">SD3</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1 </span>**<span style="color: rgb(100,37,208); background-color: inherit">的 </span>**<span style="color: rgb(100,37,208); background-color: inherit">VAE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 会对编码后的 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Latent </span>**<span style="color: rgb(100,37,208); background-color: inherit">特征做平移和缩放</span>：（之前的 **SD** 系列中 **VAE** 仅做缩放）
>
> 平移和缩放操作能<span style="color: rgb(100,37,208); background-color: inherit">将 Latent 特征分布的均值和方差归一化到</span>**`0`**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`1`**<span style="color: rgb(100,37,208); background-color: inherit">，和扩散过程加的高斯噪声在同一范围内，更加严谨和合理</span>。

**`FLUX.1-dev`**/**`FLUX.1-schnell`**&#x7CFB;列模型的 **VAE** 完整结构图如下：

![FLUX.1-dev/schnell VAE 完整结构图](../../images/视觉多模态讲义（下）-image-71.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">DiT</span>**

**FLUX.1** &#x7684;**`DiT`**&#x5728; **SD3** 的基础上进一步优化，<span style="color: rgb(100,37,208); background-color: inherit">除了有和 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SD3</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 一样的双流 DiT </span>**`MM-DiT`**<span style="color: rgb(100,37,208); background-color: inherit">，还有单流 DiT </span>**`Single-DiT`**<span style="color: rgb(100,37,208); background-color: inherit">。在单流 DiT 中，文本信息和图像信息拼接融合在一起，再送入 Attention 中处理</span>，也就是经典的 DiT 架构。同时在额外条件部分会输入完整&#x7684;**`Text Embedding`**&#x548C;池化过&#x7684;**`Pooled Text Embedding`**。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：先使用 **MM-DiT block&#x20;**&#x5B9E;现两个模态信息融合，然后再接 **Single-DiT Block&#x20;**&#x52A0;深模型深度，<span style="color: rgb(46,161,33); background-color: inherit">增强模型的整体学习能力的同时，还可以节省一些参数</span>。

**<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的</span>**`MM-Single-DiT`**<span style="color: rgb(100,37,208); background-color: inherit">包含</span>**`19`**<span style="color: rgb(100,37,208); background-color: inherit">层 </span>**<span style="color: rgb(100,37,208); background-color: inherit">MM-DiT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> Block 和</span>**`38`**<span style="color: rgb(100,37,208); background-color: inherit">层</span>**<span style="color: rgb(100,37,208); background-color: inherit"> Single-DiT</span>**<span style="color: rgb(100,37,208); background-color: inherit"> Block</span>，详细结构：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">MM-DiT Block</span>**：包含两&#x4E2A;**`AdaLayerNormZero`**&#x5C42;、一&#x4E2A;**`MM-DiT Attention`**&#x6A21;块、两&#x4E2A;**`LayerNorm`**&#x5C42;、两&#x4E2A;**`FeedForward`**&#x5C42;
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">Single-DiT Block</span>**：包含一&#x4E2A;**`AdaLayerNormZero`**&#x5C42;、一&#x4E2A;**`Single-DiT Attention`**&#x6A21;块、两&#x4E2A;**`Linear`**&#x5C42;、一&#x4E2A;**`GELU`**&#x6FC0;活函数
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">MM-DiT Attention</span>**：与 **SD3** 相同，<span style="color: rgb(100,37,208); background-color: inherit">将文本和图像看作同等重要的信息送入 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Attention</span>**
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">Single-DiT Attention</span>**：<span style="color: rgb(100,37,208); background-color: inherit">将文本信息和图像信息融合后，送入经典的 DiT 中的</span>**<span style="color: rgb(100,37,208); background-color: inherit"> Attention</span>**
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">FeedForward</span>**：包&#x62EC;**`GELU`**&#x6FC0;活函数、**`Dropout`**&#x5C42;、**`Linear`**&#x5C42;

**`FLUX.1-dev`**/**`FLUX.1-schnell`**&#x7684;**`MM-Single-DiT`**&#x7684;完整结构图如下：

![FLUX.1-dev/schnell MM-Single-DiT 完整结构图](../../images/视觉多模态讲义（下）-image-72.png)

**FLUX.1** 将得到&#x7684;**`Patch Embedding`**&#x4E0E;**`Positional Embedding`**&#x76F8;加，然后输入到 Transformer 的主架构中。 同时通&#x8FC7;**`adaLN-Zero`**&#x5C42;将文本全局语义信息特&#x5F81;**`CLIP pooled embedding`**&#x548C;**`Timestep Embedding`**&#x52A0;在一起的融合特征作为额外条件注入到 Transformer Block 中。

**FLUX.1&#x20;**&#x7684; Transformer <span style="color: rgb(100,37,208); background-color: inherit">引入了</span>**<span style="color: rgb(100,37,208); background-color: inherit">并行注意力机制</span>**<span style="color: rgb(100,37,208); background-color: inherit">，主要是在 Single-DiT 中使用</span>，进一步优化模型整体的性能。

并行注意力机制<span style="color: rgb(100,37,208); background-color: inherit">把注意力和线性层之间的串联结构转变成并联结构</span>。

> 常规注意力机制需要在计算注意力的前后各经过一次线性层的特征提取，

转换成并联结构后，<span style="color: rgb(100,37,208); background-color: inherit">注意力在计算完成后与 MLP 进行了 add 操作，将特征融合</span>，整体的<span style="color: rgb(46,161,33); background-color: inherit">计算并行度更高，运行效率也随之提升</span>。



![](../../images/视觉多模态讲义（下）-image-76.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">位置编码</span>**

SD3 中采用绝对位置编码方式，但 **<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 采用了</span>**<span style="color: rgb(100,37,208); background-color: inherit">旋转位置编码 RoPE</span>**<span style="color: rgb(100,37,208); background-color: inherit">，RoPE 使得每个位置的 Token 保留了相邻位置的相对关系</span>。相比传统的绝对位置编码，<span style="color: rgb(46,161,33); background-color: inherit">RoPE 更注重局部关系的建模，这种增强的局部敏感性有助于模型捕获图像局部区域之间的细节关联，从而提升模型的生成质量和泛化性能</span>。

具体的操作是<span style="color: rgb(100,37,208); background-color: inherit">将文本的位置编号设为</span>**`(0, 0, 0)`**<span style="color: rgb(100,37,208); background-color: inherit">，图像的位置编号设为</span>**`(0, i, j)`**<span style="color: rgb(100,37,208); background-color: inherit">，之后用标准的旋转位置编码对三个维度的编号编码，再把三组编码拼接</span>。

> **<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：假设<span style="color: rgb(220,155,4); background-color: inherit">位于</span>**`(i, j)`**<span style="color: rgb(220,155,4); background-color: inherit">的图像像素的位置编号是</span>**`(0, i, j)`**<span style="color: rgb(220,155,4); background-color: inherit">，经过特征编码，位置编号会转换成</span>**`[16, 56, 56]`**<span style="color: rgb(220,155,4); background-color: inherit">维度的矩阵，表示第一个维度用长度</span>**`16`**<span style="color: rgb(220,155,4); background-color: inherit">的位置编码，后两维用长度 </span>**`56`**<span style="color: rgb(220,155,4); background-color: inherit"> 的位置编码</span>。再经 RoPE 计算得到旋转式位置编码后拼接到一起，形&#x6210;**`128`**&#x7EF4;的位置编码。前 **16** 个通道是第一维位置编号的位置编码，后面两组 **56** 个通道分别是第二维、第三维位置编号的位置编码。这表示在进行注意力运算时，<span style="color: rgb(100,37,208); background-color: inherit">特征的前 </span>**<span style="color: rgb(100,37,208); background-color: inherit">16</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 个通道不知道位置信息，中间 </span>**<span style="color: rgb(100,37,208); background-color: inherit">56</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 个通道知道垂直的位置信息，最后 </span>**<span style="color: rgb(100,37,208); background-color: inherit">56</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 个通道知道水平的位置信息</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练方法</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">time_shift</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 系列模型和 SD3 一样使用了基于</span>**`Rectified Flow`**<span style="color: rgb(100,37,208); background-color: inherit">采样方法来生成图像</span>，FLUX.1 也设置了一&#x4E2A;**`time_shift`**&#x503C;来平移 Time Step。

在 Rectified Flow 采样中，图像沿着某条高维路线从纯高斯噪声运动到训练集分布中，标准差用于控制不同时刻图像的不确定性。

> <span style="color: rgb(100,37,208); background-color: inherit">时刻为</span>**`0`**<span style="color: rgb(100,37,208); background-color: inherit">时，图像为纯噪声，标准差为</span>**`1`**
>
> <span style="color: rgb(100,37,208); background-color: inherit">时刻为</span>**`1`**<span style="color: rgb(100,37,208); background-color: inherit">时，图像趋近训练集图像分布，此时标准差要尽可能趋于</span>**`0`**

原本对于中间时刻，标准差默认按照时刻线性变化。而 **<span style="color: rgb(100,37,208); background-color: inherit">FLUX.1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 中设置了</span>**`time_shift`**<span style="color: rgb(100,37,208); background-color: inherit">是</span>**`0.5 ~ 1.16`**<span style="color: rgb(100,37,208); background-color: inherit">之间的数，控制的是中间时刻的噪声均值</span>。如右图，&#x5F53;**`time_shift`**&#x503C;越大时，运动线路逐渐上凸：

![](../../images/视觉多模态讲义（下）-image-73.png)

当输入的图像分辨率越大，对应的 tokens 越多，time\_shift 越大，这时要加的噪声就越大。对于分辨率越高的图像，需要加更多噪声来摧毁原图像的分布特征。这与 SD3 中的策略一致。

* **<span style="color: rgb(36,91,219); background-color: inherit">多分辨率</span>**

**FLUX.1** 系列能够对多种图像分辨率和图像长宽比进行灵活生成，能够适&#x5E94;**`0.1-2.0MP`**&#x7684;图像生成任务。图像像素数量越多，图像的分辨率越高，细节表现越丰富。**<span style="color: rgb(46,161,33); background-color: inherit">FLUX.1</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 能够很好的适配各种分辨率的图像生成，这主要得益于 </span>**<span style="color: rgb(46,161,33); background-color: inherit">FLUX.1</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 采用了</span>**<span style="color: rgb(46,161,33); background-color: inherit">多尺度训练 </span>**<span style="color: rgb(46,161,33); background-color: inherit">+ </span>**<span style="color: rgb(46,161,33); background-color: inherit">RoPE 位置编码 </span>**<span style="color: rgb(46,161,33); background-color: inherit">+ </span>**<span style="color: rgb(46,161,33); background-color: inherit">动态 time shift</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 的组合策略</span>。

### 4.2.6 <span style="color: rgb(36,91,219); background-color: inherit">FLUX.2</span>

FLUX.2 系列目前发布了4款模型：**<span style="color: rgb(100,37,208); background-color: inherit">FLUX.2 [pro]、FLUX.2 [flex]、FLUX.2 [dev]、FLUX.2 [klein]</span>**

> * **<span style="color: rgb(36,91,219); background-color: inherit">FLUX.2 [pro]</span>**：闭源商用版本，图像质量对标顶级闭源模型，生成速度更快、成本更低
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">FLUX.2 [flex]</span>**：允许调整步数、引导规模等参数，平衡质量与速度，擅长文本与细节渲染
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">FLUX.2 [dev]</span>**：32B 参数开源版本，集成文生图、多图编辑功能，支持本地部署
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">FLUX.2 [klein]</span>**：开源轻量版，通过蒸馏保留核心能力

![](../../images/视觉多模态讲义（下）-image-86.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">text encoder</span>**

**FLUX.1** 的 text encoder 采用的&#x662F;**`T5-XXL`**&#x548C;**`CLIP`**，**FLUX.2&#x20;**&#x7528;的是一&#x4E2A;**`VLM`**&#x6A21;型：**`24B`**&#x7684;**`Mistral-Small-3.2-24B-Instruct-2506`**，不过这里<span style="color: rgb(100,37,208); background-color: inherit">只用它来编码文本特征</span>，对于图像输入，不会像 Qwen-Image-Edit 那样还对输入图像进行特征编码。

* **<span style="color: rgb(36,91,219); background-color: inherit">VAE</span>**

FLUX.2 的 VAE 在可学习性、质量和压缩率之间进行探索：

> * **<span style="color: rgb(36,91,219); background-color: inherit">可学习性</span>**：<span style="color: rgb(100,37,208); background-color: inherit">DiT 在 VAE 的 latent 特征空间中学习生成新样本的难易程度，如果 latent 特征具有语义化表征可以简化建模任务，因为生成模型只需要捕捉高层语义关系，而不必建模低层感知细节</span>，比如最近的工&#x4F5C;**`RAE`**。但这种方式可能会<span style="color: rgb(216,57,49); background-color: inherit">降低图像质量，同时降低压缩效率</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">质量</span>**：<span style="color: rgb(100,37,208); background-color: inherit">VAE Decoder 能多忠实地从压缩的 latent 中重建原始图像</span>，过度压缩会带来感知失真并丢失细节。通过加入感知损失与对抗训练可以提高重建质量，但<span style="color: rgb(216,57,49); background-color: inherit">压缩比越高，保真度必然下降</span>。如果 VAE 在训练中只追求高保真重建，而<span style="color: rgb(216,57,49); background-color: inherit">不对潜空间进行语义约束，则可能产生包含高频噪声或结构不规则的潜空间，使生成模型难以学习</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">压缩率</span>**：<span style="color: rgb(100,37,208); background-color: inherit">latent 的特征维度，更高的压缩率有助于提高建模效率</span>，但可能同时损害重建质量以及生成模型对真实数据分布的刻画能力。

这三个目标本质上存在冲突：<span style="color: rgb(216,57,49); background-color: inherit">提高压缩率往往会损害重建质量，并可能降低可学习性；为了实现完美的重建，需要降低压缩程度；而为了通过语义结构最大化可学习性，则可能需要牺牲低层次的感知保真度</span>。最佳的折中方式在&#x4E8E;**<span style="color: rgb(100,37,208); background-color: inherit">舍弃不可感知的信息，同时保留生成模型能够高效学习的、语义上有意义的结构</span>**。这个也是 **FLUX.2** 的 VAE 的设计目标，其在保持图像重建质量的基础上提升了可学习性：FLUX.2 的 VAE 进一步<span style="color: rgb(100,37,208); background-color: inherit">提升了 latent 的特征维度，但空间压缩率还是</span>**`8x`**

![](../../images/视觉多模态讲义（下）-image-84.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：**`SD-VAE`**&#x7684; latent 维度&#x662F;**`4`**，**`FLUX.1 VAE`**&#x7684; latent 维度&#x662F;**`16`**，**`FLUX.2 VAE`**&#x7684; latent 维度增加&#x81F3;**`32`**，latent 维度的增加不影响 DiT 的 token 数量，所以不会增加计算量，同时在训练 VAE 时引入了语义正则化来提升可学习性。

* **<span style="color: rgb(36,91,219); background-color: inherit">DiT</span>**

FLUX.2 的 DiT 也进行了 scaling，参数量从原来&#x7684;**`12B`**&#x589E;&#x52A0;**`32B`**。FLUX.2 延续了与 FLUX.1 相同&#x7684;**`MM-DiT`** + 并行 DiT 架构。MM-DiT 块首先在独立的通道中处理图像 latent 表征和文本条件特征（**<span style="color: rgb(100,37,208); background-color: inherit">双流</span>**），仅在注意力操作时将两者结合，并行块则在拼接后的图像和文本通道上操作（**<span style="color: rgb(100,37,208); background-color: inherit">单流</span>**）。

**FLUX.1** 与 **FLUX.2&#x20;**&#x4E2D; DiT 的主要区别如下：

> 1. timestep 和控制条件分别在所有双流和单流 Transformer 块之间共享，不像 FLUX.1 为每个块设置独立的调制参数，这可以降低参数量
>
> 2. 模型中所有层都不使用 bias，包括注意力和前馈网络
>
> 3. 在 FLUX.1 中，单流 Transformer 块将注意力输出与前馈网络输出进行了融合。FLUX.2 的单流块还将注意力的 QKV 与前馈网络输入融合，形成一个完全并行的 Transformer 块

![](../../images/视觉多模态讲义（下）-image-85.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：<span style="color: rgb(100,37,208); background-color: inherit">FLUX.2 中单流 Transformer 块占比更大</span>：包&#x62EC;**`8`**&#x4E2A;双流块&#x548C;**`48`**&#x4E2A;单流块，FLUX.1 分别&#x4E3A;**`19`**/**`38`**。即单流块在 DiT 参数中占比更高：**`Flux.1[dev]-12B`**<span style="color: rgb(220,155,4); background-color: inherit">大约有</span>**`54%`**<span style="color: rgb(220,155,4); background-color: inherit">的总参数位于双流块中，而</span>**`FLUX.2[dev]-32B`**<span style="color: rgb(220,155,4); background-color: inherit">大约有</span>**`24%`**<span style="color: rgb(220,155,4); background-color: inherit">的参数在双流块中，约</span>**`73%`**<span style="color: rgb(220,155,4); background-color: inherit">的参数在单流块中</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">位置编码</span>**

**FLUX.1** 的位置编码采&#x7528;**`3D RoPE`**，两个维度分别编码图像的 **w** 和 **h**，第三维度&#x4E3A;**&#x20;t**，固定&#x4E3A;**`0`**。FLUX.2 的位置编码采&#x7528;**`4D RoPE`**，<span style="color: rgb(100,37,208); background-color: inherit">第一个维度是 </span>**<span style="color: rgb(100,37,208); background-color: inherit">t</span>**<span style="color: rgb(100,37,208); background-color: inherit">，用来区分目标图和输入的条件图，其中目标图设定为 0，而输入条件图会进行依次偏移，分别是10、20、...，这里偏移 scale 是</span>**`10`**，第二和第三维度分别编码图像的 **w** 和 **h**，最后一个维度是 **l**，用来编码文本 token 的序列号，对于 image latents 固定&#x4E3A;**`0`**。**<span style="color: rgb(100,37,208); background-color: inherit">新增的一个维度主要是给文本 token 增加了位置编码</span>**<span style="color: rgb(100,37,208); background-color: inherit">，之前的 FLUX.1 文本 token 的位置编码都固定为</span>**`0`**<span style="color: rgb(100,37,208); background-color: inherit">，是不区分各个 token 的</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">新特点</span>**

1. 支持<span style="color: rgb(100,37,208); background-color: inherit">输入结构化的 JSON prompt</span>，可以精确控制图像的各个方面，对于生产流程和自动化比较有用。支持输入场景描述、主体描述、艺术风格、颜色调色板、光照、背景、构图以及相机参数等来精确控制生成的图像。

2. 支持<span style="color: rgb(100,37,208); background-color: inherit">使用十六进制颜色代码进行精确配色</span>，对于品牌一致性和设计工作比较重要。可以直接使用关键词如 **`color`**<span style="color: rgb(220,155,4); background-color: inherit">或</span>**`hex`**&#x6307;示十六进制颜色代码。

3. <span style="color: rgb(100,37,208); background-color: inherit">多语言理解能力</span>，可以使用母语 prompt，获得更符合文化特色的生成结果

### 4.2.7 <span style="color: rgb(36,91,219); background-color: inherit">FLUX.3</span>

<span style="color: rgb(100,37,208); background-color: inherit">FLUX.3 的核心变化不是在 FLUX.2 上继续堆叠图像生成能力，而是把图像、视频、音频和动作预测放进同一个多模态 Flow Backbone，让内容生成与 Physical AI 共用一套世界表征。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">模型定位</span>**

FLUX.3 的技术跨度和产品成熟度并不同步：统一多模态能力已经进入 Early Access，正式接口、权重和商业条件仍在展开。先区分“模型能够做什么”和“用户现在能够拿到什么”，后续判断才不会把演示能力当成交付能力。

1. **<span style="color: rgb(36,91,219); background-color: inherit">背景</span>**

FLUX.1 和 FLUX.2 的主产品形态是图像生成与图像编辑。<span style="color: rgb(100,37,208); background-color: inherit">FLUX.3 把产品边界扩展为图像、视频、音频和 Action Prediction，并用统一 Transformer 表达这些模态之间的联系</span>。文本仍然承担指令与语义条件的作用，图像提供空间结构，视频补上时间与动力学，音频补充事件与声学因果，动作序列则把内部世界表征连接到真实执行。

<span style="color: rgb(46,161,33); background-color: inherit">统一 Backbone 的工程收益是：同一对象、风格、运动状态和因果关系不需要在多个独立模型之间反复对齐，跨模态条件可以在共享特征空间里传递。</span> <span style="color: rgb(216,57,49); background-color: inherit">这不等于 FLUX.3 已经成为一个可直接下载、可本地部署、接口稳定的通用世界模型。当前能确认的是 Early Access 计划和公开展示的能力边界，不能确认的部分必须继续留空。</span>这种能力扩张是否属于真正的代际变化，要看模型的训练对象和输出空间是否随之改变。把 FLUX.1、FLUX.2 与 FLUX.3 放在一起，差异就不再只是多了视频功能，而是 Backbone 开始承担跨模态建模。

* **<span style="color: rgb(36,91,219); background-color: inherit">从 FLUX.1、FLUX.2 到 FLUX.3</span>**

| **<span style="color: rgb(36,91,219); background-color: inherit">代际</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">核心输出</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">主要条件</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">架构定位</span>**                           |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| FLUX.1                                                                       | 图像                                                                             | 文本，部分版本支持图像条件与局部控制                                                             | 高质量文生图与可控图像生成                                                                                            |
| FLUX.2                                                                       | 图像                                                                             | 文本与多张参考图，强调 in-context 生成和编辑                                                   | 图像生成、编辑、角色与风格一致性                                                                                         |
| FLUX.3                                                                       | 图像、视频、音频、动作                                                                    | 文本、图像、视频、音频与机器人状态                                                              | <span style="color: rgb(100,37,208); background-color: inherit">统一多模态 Flow Backbone，同时服务内容生成与动作预测</span> |

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：代际方向明确以后，仍要把架构路线与产品可用性拆开。统一 Backbone 可以支撑多个输出头，但 Video、Image、Action 和开放权重会按不同节奏落地。

* **<span style="color: rgb(36,91,219); background-color: inherit">产品线</span>**

| **<span style="color: rgb(36,91,219); background-color: inherit">名称</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">能力</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">计划访问方式</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">当前状态</span>**         |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| FLUX 3 Video                                                                 | 视频与原生音频生成、编辑、续写                                                              | API 与 Private Weights                                                            | <span style="color: rgb(46,161,33); background-color: inherit">Early Access 已启动</span> |
| FLUX 3 Image                                                                 | 图像生成与编辑                                                                      | API 与 Private Weights                                                            | 计划在随后数周开放 Early Access                                                                 |
| FLUX-mimic / Action                                                          | 视频与机器人动作联合预测                                                                 | 研究与商业合作伙伴                                                                        | 与 mimic robotics 联合验证                                                                  |
| FLUX 3 Dev                                                                   | 面向内容生成与 Action Prediction 的多模态 Backbone                                      | Open Weights                                                                     | <span style="color: rgb(216,57,49); background-color: inherit">尚未公开下载</span>           |

* **<span style="color: rgb(36,91,219); background-color: inherit">统一多模态架构</span>**

上面解释了哪些能力会先开放，统一架构则解释这些能力为什么能够共享对象、运动和因果信息。核心问题不是把四类数据简单拼接，而是<span style="color: rgb(100,37,208); background-color: inherit">哪些表征应该共享、哪些输入输出必须保留模态差异</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">从不同模态进入同一 Transformer</span>**

<span style="color: rgb(100,37,208); background-color: inherit">每种输入先通过自己的 Encoder 进入对应 Token 空间，再送入共享的 Multimodal Transformer</span>。输出端仍然保留模态专用 Decoder：图像 Decoder 还原空间像素，视频 Decoder 还原时空序列，音频 Decoder 还原波形或声学表示，Action Decoder 输出机器人状态转移或控制序列。

![文本、图像、视频、音频与动作分别编码，在共享 Multimodal Transformer 中交换信息，再由模态专用 Decoder 输出。](../../images/视觉多模态讲义（下）-test-1.jpg)

> **<span style="color: rgb(36,91,219); background-color: inherit">统一 Backbone 中的四类信息流</span>**
>
> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Image Token</span>**：表达对象、材质、文字、构图和空间关系。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">Video Token</span>**：表达跨帧运动、遮挡、接触、惯性和事件顺序。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">Audio Token</span>**：表达语音、环境声，以及撞击、摩擦等事件与声学结果之间的联系。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">Action Token</span>**：表达机器人状态和低维控制信号，把预测出的未来状态映射为可执行动作。

共享 Transformer 解决跨模态交互，算力分配却不会因此平均。视频 Token 同时沿空间和时间展开，训练预算很快被时空建模主导。

* **<span style="color: rgb(36,91,219); background-color: inherit">为什么视频是训练主成本</span>**

视频同时包含空间分辨率和时间长度，模型必须处理运动连续性、物体恒常性、接触关系与因果结果。FLUX.3 的训练成本中，视频预测占比超过 **`95%`**。这说明<span style="color: rgb(100,37,208); background-color: inherit">“统一多模态”并不意味着每种模态的计算权重相同，真正昂贵的是时空建模</span>。在带音频的 **`720p`** 视频表示里，音频 Token 少于总 Token 的 **`0.5%`**。音频维度低得多，但它能提供视觉里没有的因果约束：嘴型要对应语音，撞击时刻要对应声音，环境声要与空间和材质一致。

训练逻辑可以概括为：<span style="color: rgb(100,37,208); background-color: inherit">视频负责逼迫模型学习物理变化，音频负责补充事件因果，图像负责提供高密度空间细节，动作负责把内部预测投射到控制空间。</span>不同模态对训练的贡献并不相同，这也意味着“统一”不能理解成所有组件完全一样。真正共享的是中间世界表征，重建像素、波形和控制量仍需不同的 I/O 路径。

* **<span style="color: rgb(36,91,219); background-color: inherit">统一并不等于完全共享</span>**

共享 Transformer 负责跨模态关系，Encoder 与 Decoder 仍然保留模态差异。这是更现实的工程折中：图像、视频、音频和动作的采样率、局部结构、损失尺度都不同，强行使用完全相同的输入输出头会把表示对齐问题转化为重建瓶颈。<span style="color: rgb(46,161,33); background-color: inherit">共享中间表征、保留模态专用 I/O，使 FLUX.3 可以在统一语义空间中迁移能力，同时避免让一个 Decoder 同时承担像素、波形和控制序列的重建任务。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">Self-Flow 技术机制</span>**

架构给出了信息在哪里交互，Self-Flow 决定这些表征怎样学出来。它试图在一次 pretraining 中同时提高生成质量和中间特征的可迁移性，因此需要从标准 Flow Matching 的局限出发。

1. **<span style="color: rgb(36,91,219); background-color: inherit">标准 Flow Matching 的学习目标</span>**

Flow Matching 从数据样本 $$x_0$$ 与高斯噪声 $$x_1$$ 之间构造连续路径。最直接的 rectified flow 使用线性插值：

$$x_t=(1-t)x_0+t x_1,\quad t\in[0,1]$$

网络 $$f_\theta$$ 预测路径上的速度场，生成损失写成：

$$\mathcal{L}_{\text{gen}}=\mathbb{E}_{x_0,x_1,t}\left\lVert f_\theta(x_t,t)-(x_1-x_0)\right\rVert_2^2$$

<span style="color: rgb(216,57,49); background-color: inherit">单一噪声强度作用于全部 Token 时，网络可以依赖局部纹理与邻域相关性完成去噪，却没有足够压力学习全局语义、长程结构和跨模态因果。</span>问题落在噪声的组织方式上：当所有 Token 处在同一噪声强度，局部线索往往足以完成去噪，模型未必需要建立长程关系。Dual-Timestep Scheduling 在同一样本内引入异质噪声，让训练输入变得更难，也更丰富。

* **<span style="color: rgb(36,91,219); background-color: inherit">Dual-Timestep Scheduling</span>**

Self-Flow 对同一个样本采样两个 timestep，并用随机 mask 把不同 Token 分配到两个噪声强度。较干净的 Token 提供上下文，噪声更重的 Token 迫使网络利用全局关系恢复信息。

> **<span style="color: rgb(36,91,219); background-color: inherit">Dual-Timestep 的训练过程</span>**
>
> 1. **<span style="color: rgb(36,91,219); background-color: inherit">采样 timestep</span>**：独立采样 $$t,s\sim p(t)$$。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">采样 mask</span>**：mask 比例不超过 **`0.5`**，避免大多数 Token 同时落入第二种噪声状态。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">逐 Token 加噪</span>**：mask 内使用 $$s$$，其余 Token 使用 $$t$$。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">保持边缘分布</span>**：任意单个 Token 的 timestep 仍来自同一个 $$p(t)$$，减少与统一 timestep 推理过程之间的分布偏移。

$$\tau_i=\begin{cases}s,&i\in M\\t,&i\notin M\end{cases}$$

$$x_\tau=\operatorname{diag}(\mathbf{1}-\tau)x_0+\operatorname{diag}(\tau)x_1$$

下面的机制图把 mixed-noise student 路径、cleaner EMA teacher 路径、生成损失和表征损失放在同一视图里。

![Self-Flow 使用 mixed-noise Student 与 cleaner EMA Teacher，在同一训练步里同时优化生成损失和表征损失。](../../images/视觉多模态讲义（下）-08-self-flow-method.png)

混合噪声改变了 Student 看到的输入，却还没有约束中间特征应该表达什么。EMA Teacher 提供更干净的表征目标，把“恢复数据”和“学习可迁移特征”合进同一次优化。

* **<span style="color: rgb(36,91,219); background-color: inherit">EMA Teacher 与自表征对齐</span>**

Student 接收混合噪声输入 $$x_\tau$$，EMA Teacher 接收两个 timestep 中更干净的那一个 $$x_{\tau_{\min}}$$。Teacher 不需要额外的 DINO、CLIP 或 V-JEPA，它来自 Student 参数的指数滑动平均。表征损失让 Student 的中间特征逼近 Teacher 在干净视图上的深层特征：

$$\mathcal{L}_{\text{rep}}=-\mathbb{E}\,\cos\left(h_\theta^{(l)}(x_\tau,\tau),f_{\theta'}^{(k)}(x_{\tau_{\min}},\tau_{\min})\right)$$

最终目标把生成与表征学习放在同一次 pretraining 中：

$$\mathcal{L}=\mathcal{L}_{\text{gen}}+\gamma\mathcal{L}_{\text{rep}}$$

<span style="color: rgb(46,161,33); background-color: inherit">Self-Flow 的价值不只是提高采样质量，它还让中间特征更适合下游理解与控制任务，并避免把生成模型绑定到固定的外部 Encoder。</span>生成指标和下游特征同时改善，并不意味着机制解释已经唯一确定。区分跨 Token 信息传递与噪声增强效应，才能避免把实验相关性写成因果结论。

* **<span style="color: rgb(36,91,219); background-color: inherit">机制解释需要保留的边界</span>**

Dual-Timestep Scheduling 的收益不能只归因于“干净 Token 帮助噪声 Token”。后续受控实验在阻断不同 timestep Token 之间的 attention 后，生成指标没有下降，部分设置反而提高。这说明异质噪声本身也像一种沿噪声轴展开的数据增强。

<span style="color: rgb(216,57,49); background-color: inherit">更稳妥的结论是：Self-Flow 同时改变了训练样本的噪声组合和表征对齐目标。现有证据足以支持它提高生成与表征质量，但还不足以把全部增益归结为单一的跨 Token 信息传递机制。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">生成与编辑能力矩阵</span>**

训练机制最终要落到可见的输入输出上。FLUX.3 的能力可以沿时间维度理解：图像处理单帧空间结构，视频扩展到连续状态，音频再把事件与声学结果绑定在同一条时间线上。

1. **<span style="color: rgb(36,91,219); background-color: inherit">视频与原生音频</span>**

单次生成最长可达 **`20 秒`**，视频输出自带原生音频。公开的初步评测使用 **`10 秒`**、**`720p`**、带音频的视频。两组数字对应不同语境，不能把评测设置误写成最大能力。

> **<span style="color: rgb(36,91,219); background-color: inherit">FLUX 3 Video 的输入输出组合</span>**
>
> 1. **<span style="color: rgb(36,91,219); background-color: inherit">Text-to-Video</span>**：从文本直接生成视频与音频。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">Image-to-Video</span>**：把起始帧动画化，或把参考图中的人物、对象和风格迁移到视频。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">Video-to-Video</span>**：保留参考片段中的主体身份或核心元素，改变场景和叙事语境。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">Video-Audio Continuation</span>**：在输入视频和音频后继续生成。
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">Keyframe-to-Video</span>**：用多个关键时刻约束中间过渡。
>
> 6. **<span style="color: rgb(36,91,219); background-color: inherit">Multilingual Dialogue</span>**：同步生成多语言对话、口型与场景声。
>
> 7. **<span style="color: rgb(36,91,219); background-color: inherit">Typography and Motion Design</span>**：生成带文字和动态排版的视频设计。
>
> 8. **<span style="color: rgb(36,91,219); background-color: inherit">Agentic Chaining</span>**：由外部工作流串联多个短片段，构成长镜头或多镜头序列。

<span style="color: rgb(216,57,49); background-color: inherit">Agentic Chaining 是片段编排能力，不代表基础模型一次即可原生生成数分钟、全程一致的长视频。长序列仍然依赖参考图、状态管理、镜头规划和失败重试。</span>视频能力检验跨帧连续性，图像能力则更集中地暴露文字、构图、材质和参考条件的一致性。两者共用 Backbone，但失败模式与评测方法并不相同。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像生成与编辑</span>**

FLUX 3 Image 延续 FLUX.2 的生成与编辑路线，但底层已换成统一多模态 Backbone。中期训练结果覆盖摄影、产品设计、绘画、平面插画和动态影像帧，重点提升复杂 prompt、跨语言文字渲染、风格跨度和参考条件一致性。

| **<span style="color: rgb(36,91,219); background-color: inherit">能力</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">输入条件</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">工程关注点</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| 图像合成                                                                         | 文本，可扩展参考图                                                                      | 复杂 prompt 遵循、文字正确率、长尾对象与细节稳定性                                                   |
| 图像编辑                                                                         | 原图、参考图与编辑指令                                                                    | 未编辑区域保持、身份一致、局部几何与材质连续性                                                         |
| 风格迁移                                                                         | 文本风格描述或视觉参考                                                                    | 内容结构与风格特征解耦，避免只做颜色滤镜                                                            |
| 多语言文字                                                                        | 包含文本的 prompt 或参考图                                                              | 字形、拼写、排版、透视与遮挡关系                                                                |

图像和视频解释了视觉输出，原生音频决定这套系统能否把事件时间、声源位置和可见动作统一起来。若音频只是生成完成后的配音，共享时空表征的价值就体现不出来。

* **<span style="color: rgb(36,91,219); background-color: inherit">音频不是独立外挂</span>**

视频、音频联合生成的关键不在于“视频生成后再配音”，而在于两种模态同时受同一个事件状态约束。口型同步、物体撞击声、空间混响和镜头内声源位置都可以从共享时空表征中取得条件。<span style="color: rgb(46,161,33); background-color: inherit">联合生成减少了离线 TTS、音效检索、时间轴对齐和后期重混的人工步骤，尤其适合需要大量短视频变体的生产链路。</span>

<span style="color: rgb(216,57,49); background-color: inherit">公开展示仍不足以证明复杂多说话人对话、长时间声学一致性、歌词级文本控制或专业混音质量。音频能力需要单独评测，不能由画面观感代替。</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">评测</span>**

能力展示只能说明模型会生成什么，评测才回答它在什么条件下更好。训练机制实验、Early Access 偏好比较和真实生产指标必须分开看，因为它们对应的结论强度完全不同。

1. **<span style="color: rgb(36,91,219); background-color: inherit">Self-Flow 的研究结果</span>**

Self-Flow 的小规模实验使用图像、视频和音频专用 autoencoder，在统一 Flow Transformer 上比较 vanilla Flow Matching、外部表征对齐和内部自表征对齐。它验证的是训练机制，不是 FLUX.3 产品模型的最终绝对分数。

![Self-Flow 同时降低视频 FVD、图像 FID 与音频 FAD，并在机器人控制微调中更快达到更高成功率。](../../images/视觉多模态讲义（下）-test.jpg)

| **<span style="color: rgb(36,91,219); background-color: inherit">任务</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Flow Matching</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">Self-Flow</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">解读</span>** |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 视频 FVD                                                                       | 72.9                                                                                    | <span style="color: rgb(46,161,33); background-color: inherit">66.3</span>          | 下降 **`9.1%`**，时序生成误差更低                                                       |
| 图像 FID                                                                       | 4.04                                                                                    | <span style="color: rgb(46,161,33); background-color: inherit">3.69</span>          | 下降 **`8.5%`**，图像分布更接近真实数据                                                    |
| 音频 FAD                                                                       | 153.0                                                                                   | <span style="color: rgb(46,161,33); background-color: inherit">149.8</span>         | 下降 **`2.1%`**，提升存在但幅度较小                                                      |
| 机器人成功率                                                                       | 35%                                                                                     | <span style="color: rgb(46,161,33); background-color: inherit">47%</span>           | 在相同微调步数下学习更快，复杂任务差距更明显                                                       |

Self-Flow 的实验支持训练方法有效，却不能直接替代 FLUX.3 产品评测。转向 Early Access 结果时，比较对象从受控研究模型变成候选产品，指标也从 FID、FVD 变成人类偏好。

* **<span style="color: rgb(36,91,219); background-color: inherit">FLUX.3 Early Access 的偏好评测</span>**

FLUX.3 Video 的初步人类偏好评测比较了多家同期视频模型。得分表示二选一时选择 FLUX.3 的比例，**`50%`** 代表双方持平，不是综合质量分。

![Early Access 候选模型的人类偏好比例；越靠右表示在该组二选一比较中越常选择 FLUX.3。](../../images/视觉多模态讲义（下）-test-4.jpg)

> **<span style="color: rgb(36,91,219); background-color: inherit">这张偏好图不能证明什么</span>**
>
> 1. **<span style="color: rgb(36,91,219); background-color: inherit">不是独立榜单</span>**：评测由模型开发方组织，尚未看到完整 prompt 集、样本量、随机化和评审细则。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">不是同一差距</span>**：**`52%`** 接近持平，**`93%`** 才是很强的单边偏好，不能都写成“领先”。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">不是生产指标</span>**：人类偏好没有覆盖延迟、价格、失败率、可重复性、并发和审核成本。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">不是最终版本</span>**：测试对象是 Early Access 候选模型，后续接口和模型可能继续变化。

人类偏好适合回答“哪段结果更讨喜”，不能覆盖稳定性、成本和失败恢复。真正的接入判断需要一套贴合业务分布的评测集，把主观质量和工程约束放进同一个记录体系。

* **<span style="color: rgb(36,91,219); background-color: inherit">真正需要建立的评测集</span>**

| **<span style="color: rgb(36,91,219); background-color: inherit">维度</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">测试对象</span>** | **<span style="color: rgb(36,91,219); background-color: inherit">失败定义</span>** |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| Prompt 遵循                                                                    | 对象数量、空间关系、动作顺序、禁用元素                                                            | 遗漏约束、对象交换、事件顺序错误                                                               |
| 人物一致性                                                                        | 跨镜头脸部、服装、配饰、体型                                                                 | 身份漂移、局部重绘、正侧脸不一致                                                               |
| 物理合理性                                                                        | 重力、接触、遮挡、刚柔体运动                                                                 | 穿模、瞬移、质量感错误、因果顺序错                                                              |
| 音画同步                                                                         | 口型、撞击、脚步、环境声                                                                   | 时间偏移、错误声源、场景切换后残留                                                              |
| 文字与排版                                                                        | 多语言拼写、动态文字、透视                                                                  | 错字、字形崩坏、帧间闪烁                                                                   |
| 工程成本                                                                         | 端到端延迟、重试次数、价格、审核                                                               | 达到可用结果的总成本超过现有 pipeline                                                        |

* **<span style="color: rgb(36,91,219); background-color: inherit">FLUX-mimic 与 Physical AI</span>**

前面的评测仍围绕内容生成，FLUX-mimic 把同一套世界表征推进到动作预测。判断这条路线是否成立，<span style="color: rgb(100,37,208); background-color: inherit">要看 Future Feature 能否被轻量 Action Decoder 读取，也要看加入控制信号后是否会持续损害原有视频能力</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">从视频 Backbone 解码机器人动作</span>**

FLUX-mimic 不需要重新训练一套与视频无关的机器人 foundation model。文本 Encoder 生成 Task Token，视频 Encoder 把历史观测转成 History Token，FLUX Backbone 预测 Future Feature；轻量 Action Decoder 再结合机器人状态，输出后续动作序列。

![FLUX-mimic 从 FLUX Backbone 的中间 Future Feature 解码机器人动作，同时保留 Video Decoder 预测未来画面。](../../images/视觉多模态讲义（下）-test-2.jpg)

<span style="color: rgb(100,37,208); background-color: inherit">这条路径的关键不是把像素直接映射成控制量，而是从视频预测过程中抽取包含任务、历史状态和未来变化的中间特征，再用 Action Decoder 读取可执行部分。</span>有了从 Future Feature 到动作序列的解码路径，紧接着的问题是多一种模态会不会扰乱已经学好的生成分布。训练初期的质量回落和后续恢复，正好检验 Backbone 是否真的容纳了动作信息。

* **<span style="color: rgb(36,91,219); background-color: inherit">加入 Action Prediction 后的适应过程</span>**

把动作模态加入训练 curriculum 后，Text-to-Video 与 Image-to-Video 的人类评分一度下降最多 **`10%`**。经过 **`3500 steps`**，视频质量恢复到加入动作前的水平，同时模型已经具备 Action Prediction。

<span style="color: rgb(46,161,33); background-color: inherit">这个结果支持“动作是同一物理过程的另一种观测”这一架构假设：模型可以在不永久牺牲视频生成质量的前提下吸收低维控制信号。</span>

视频质量能够恢复，只说明多模态训练在当前设置下没有造成永久冲突；它还不能回答开放环境中的可靠性。数据规模、机器人本体差异和安全控制决定了这条路线离通用部署还有多远。

![加入动作模态后视频质量先下降，继续训练后 Text-to-Video 与 Image-to-Video 都恢复到原有水平。](../../images/视觉多模态讲义（下）-test-3.jpg)

* **<span style="color: rgb(36,91,219); background-color: inherit">训练规模与现实边界</span>**

FLUX.3 使用数千万小时通用视频学习广泛的世界动力学，并使用数十万小时面向人类与机器人操作的视频强化 manipulation 表征。FLUX-mimic 进一步在工业机器人任务上做适配，并进入 Audi 生产环境中的测试与部署链路。

> **<span style="color: rgb(36,91,219); background-color: inherit">Physical AI 结论不能外推过度</span>**
>
> 1. **<span style="color: rgb(36,91,219); background-color: inherit">场景边界</span>**：工业工位通常具有固定相机、固定工作区和受控对象分布，不能直接代表开放家庭环境。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">硬件边界</span>**：不同机器人本体的关节、夹爪、控制频率和安全约束不同，Action Decoder 需要适配。
>
> 3. **<span style="color: rgb(36,91,219); background-color: inherit">安全边界</span>**：视频预测合理不等于控制动作安全，仍需碰撞检测、速度限制、急停与独立安全控制器。
>
> 4. **<span style="color: rgb(36,91,219); background-color: inherit">评测边界</span>**：公开曲线说明特征可迁移，不等于已经解决长时任务、异常恢复与零样本部署。

---

[Previous](06-Diffusion-Model--Diffusion-基础.md) | [Contents](../../README.md) | [Next](08-Diffusion-Model--Diffusion-的应用.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=7)
