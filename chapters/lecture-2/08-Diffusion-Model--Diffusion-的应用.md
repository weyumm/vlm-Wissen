[Previous](07-Diffusion-Model--Stable-Diffusion-系列.md) | [Contents](../../README.md) | [Next](09-Diffusion-Model--OpenAI-工作.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=8)

## 4.3 <span style="color: rgb(36,91,219); background-color: inherit">Diffusion 的应用</span>

### 4.3.1 <span style="color: rgb(36,91,219); background-color: inherit">SDEdit</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">用预训练 SDE 直接处理用户引导图</span>**

SDEdit 接收一张已经在 RGB 像素空间表达意图的引导图 $$x^{(g)}$$，例如粗线稿、在真实图像上涂改的笔触，或贴入目标图像的局部图块。它不训练新的条件网络，也不需要已编辑/未编辑图像对，而是<span style="color: rgb(100,37,208); background-color: inherit">先向引导图加入适量高斯噪声，抹平笔触、拼接边缘等不自然细节，再从中间时刻启动预训练生成 SDE 的逆过程</span>。

同一过程可以覆盖笔触生成、局部编辑和图像合成。完整输入—输出形式如下。

![SDEdit 在笔触生成、局部编辑和图像合成中的统一输入输出形式](../../images/视觉多模态讲义（下）-sdedit-fig1-overview.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">前向 SDE 与噪声扰动</span>**

设 $$x(0)\sim p_0=p_{\mathrm{data}}$$ 是真实图像，连续时间 $$t\in[0,1]$$。前向 SDE 把数据分布逐步变成高斯噪声。给定 $$x(0)$$，任意时刻的扰动样本可以写为：

$$x(t)=\alpha(t)x(0)+\sigma(t)z,\qquad z\sim\mathcal{N}(0,I)$$

$$\alpha(t)$$ 控制保留的数据信号强度，$$\sigma(t)$$ 控制噪声强度。常见形式包括：

> * **<span style="color: rgb(36,91,219); background-color: inherit">VE-SDE</span>**：$$\alpha(t)=1$$，$$\sigma(1)$$ 足够大，使终点分布接近 $$\mathcal{N}(0,\sigma^2(1)I)$$。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">VP-SDE</span>**：$$\alpha^2(t)+\sigma^2(t)=1$$，并且 $$t\rightarrow1$$ 时 $$\alpha(t)\rightarrow0$$，终点分布为 $$\mathcal{N}(0,I)$$。

两种 SDE 的具体形式不同，但都把图像分布连续推向高斯分布。SDEdit 的主过程可以在二者上实现；核心推导以 VE-SDE 为例。

* **<span style="color: rgb(36,91,219); background-color: inherit">逆 SDE 与 score 模型</span>**

图像生成等价于从噪声状态反向积分到数据状态。VE-SDE 的逆过程为：

$$dx(t)=\left[-\frac{d\sigma^2(t)}{dt}\nabla_x\log p_t(x)\right]dt+\sqrt{\frac{d\sigma^2(t)}{dt}}\,d\bar w$$

$$\bar w$$ 是时间从 $$1$$ 流向 $$0$$ 时的 Wiener 过程，$$\nabla_x\log p_t(x)$$ 是噪声扰动分布的 score。用参数模型 $$s_\theta(x(t),t)$$ 逼近 score 时，单个时刻的去噪 score matching 目标为：

$$\mathcal{L}_t=\mathbb{E}_{x(0)\sim p_{\mathrm{data}},\,z\sim\mathcal{N}(0,I)}\left[\left\|\sigma_t s_\theta(x(t),t)-z\right\|_2^2\right]$$

整体训练目标是不同 $$t$$ 上损失的加权和。训练完成后，用 Euler–Maruyama 离散化逆 SDE。由 $$t+\Delta t$$ 更新到 $$t$$：

$$x(t)=x(t+\Delta t)+\left(\sigma^2(t)-\sigma^2(t+\Delta t)\right)s_\theta(x(t),t)+\sqrt{\sigma^2(t)-\sigma^2(t+\Delta t)}\,z$$

普通无条件生成从 $$t=1$$ 的高斯噪声开始。SDEdit 的关键变化是：<span style="color: rgb(100,37,208); background-color: inherit">逆过程不必从 </span>$$1$$<span style="color: rgb(100,37,208); background-color: inherit"> 开始，可以从任意中间时刻 </span>$$t_0\in(0,1)$$<span style="color: rgb(100,37,208); background-color: inherit"> 开始</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">SDEdit 采样过程</span>**

对用户引导图 $$x^{(g)}$$ 和选定的起始时刻 $$t_0$$，先采样：

$$x^{(g)}(t_0)\sim\mathcal{N}\!\left(x^{(g)},\sigma^2(t_0)I\right)$$

然后以 $$x^{(g)}(t_0)$$ 作为逆 SDE 初值，从 $$t_0$$ 迭代到 $$0$$ 得到 $$x(0)$$。记作：

$$x(0)=\operatorname{SDEdit}(x^{(g)};t_0,\theta)$$

加噪把引导图从非自然图像空间推向模型熟悉的噪声数据分布；逆 SDE 再沿 score 场逐步投影回自然图像流形。该路径完整示意如下。

![SDEdit 从引导图加噪并沿逆 SDE 返回自然图像流形的过程](../../images/视觉多模态讲义（下）-sdedit-fig2-process.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">起始时刻控制真实感与忠实度</span>**

$$t_0$$ 是除求解器离散步数外最关键的超参数：

> * $$t_0$$ 较小：加入噪声较少，逆过程较短，结果更接近引导图，但笔触、粘贴边缘和不自然结构可能被保留。
>
> * $$t_0$$ 较大：引导信息被破坏得更多，逆过程有更大自由度回到自然图像流形，结果更真实，但可能偏离用户给定的布局、颜色或局部内容。

$$t_0=0$$ 等价于直接保留引导图，$$t_0=1$$ 则接近忽略引导的随机无条件采样。应在二者之间选择兼顾 realism 与 faithfulness 的中间区间。下图完整展示了 $$t_0$$ 增大时的连续变化。

![SDEdit 起始时刻 t0 对真实感与引导忠实度的连续控制](../../images/视觉多模态讲义（下）-sdedit-fig3-tradeoff.png)

<span style="color: rgb(216,57,49); background-color: inherit">SDEdit 不知道“真实图像如何被用户操作变成引导图”的测量函数</span>，因此它不是常规已知退化算子的逆问题求解。方法成立的条件是：用户意图已经以可直接扰动的 RGB 引导图表达，且预训练 SDE 的数据分布覆盖目标图像域。

### 4.3.2 <span style="color: rgb(36,91,219); background-color: inherit">Textual Inversion</span>

Textual Inversion 的目标是实现由语言引导的、针对用户指定新概念的生成。因此作者希望<span style="color: rgb(100,37,208); background-color: inherit">将这些概念编码进一个预训练文本到图像模型的中间表示中。理想情况下，这一过程应能充分利用此类模型所蕴含的丰富语义与视觉先验，并利用它来指导对这些概念进行直观的视觉变换</span>。

很直观的一个想法是考虑在文本到图像模型常用的文本编码器的词嵌入阶段寻找合适的候选表征。在此阶段，离散的输入文本首先被转换为连续的向量表示，这种形式便于直接优化。

之前已经有工作表明该嵌入空间具有足够的表达能力以捕捉基本的图像语义。但是这些方法依赖的是对比学习或语言补全目标，二者均不要求对图像有深入的视觉理解。这些方法<span style="color: rgb(216,57,49); background-color: inherit">无法准确捕捉概念的外观特征，若尝试将其用于图像合成，会导致明显的视觉失真</span>。作者的目标是找到能够指导生成的伪词，而生成本质上是一个视觉任务。因此，作者提出通过视觉重建目标来寻找这样的伪词。

有了这个想法，作者将其应用于潜在扩散模型 **<span style="color: rgb(216,57,49); background-color: inherit">LDM</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">atent </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">iffusion </span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">odel）</span>的核心细节。

LDM 包含两个核心组件。

> * **<span style="color: rgb(36,91,219); background-color: inherit">自编码器</span>**：在大量图像数据上进行预训练。编码器$$E$$学习将图像$$x \in \mathcal{D}_x$$映射为一个空间潜在码$$z = E(x)$$，并通过 KL 散度损失或向量量化进行正则化。解码器$$D$$学习将这些潜在码还原为图像，使得$$D(E(x)) \approx x$$。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">扩散模型</span>**：用于在学习到的潜在空间中生成代码。该扩散模型可以以类别标签、分割掩码等作为条件。令$$c_\theta(y)$$表示将条件输入$$y$$映射为条件向量的模型，则 LDM 的损失函数定义为：
>
> $$\mathcal{L}_{\text{LDM}} := \mathbb{E}_{z \sim E(x), y, \epsilon \sim \mathcal{N}(0,I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\|^2_2 \right]$$
>
> 其中$$t$$是时间步，$$z_t$$是在时间$$t$$被加入噪声的潜在表示，$$\epsilon$$是未缩放的噪声样本，$$\epsilon_\theta$$是去噪网络。直观上，<span style="color: rgb(100,37,208); background-color: inherit">该目标是正确去除添加到图像潜在表示上的噪声。训练过程中，</span>$$c_\theta$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\epsilon_\theta$$<span style="color: rgb(100,37,208); background-color: inherit">联合优化以最小化 LDM 损失。在推理阶段，从随机噪声张量开始，通过迭代去噪生成新的潜在码</span>$$z_0$$<span style="color: rgb(100,37,208); background-color: inherit">，最终通过预训练解码器将其转换为图像：</span>$$x' = D(z_0)$$。

作者采用 Rombach 等人公开发布&#x7684;**`1.4B`**&#x6A21;型，该模型&#x5728;**`LAION-400M`**&#x6570;据集上进行了预训练。在此模型中，$$c_\theta$$由 BERT 文本编码器实现，$$y$$为文本 prompt。

* **<span style="color: rgb(36,91,219); background-color: inherit">Text embedding</span>**

文本编码器模型首先进行文本处理。输入字符串中的每个词或子词被转换为一个token，即某个预定义词典中的索引。每个 token 对应一个唯一的嵌入向量，可通过索引查找获得。这些嵌入向量通常作为文本编码器 $$c_\theta$$ 训练的一部分被学习。

作者选择该嵌入空间作为反演的目标空间。设定一个占位符字符串$$S^*$$，用于表示作者希望学习的新概念。然后在嵌入过程中进行干预，将该 token 对应的向量替换为一个新的可学习嵌入$$v^*$$，实质上是将该概念注入词汇表中。这样一来就可以像使用普通词汇一样，构造包含该概念的新句子。

![](../../images/视觉多模态讲义（下）-image-81.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Textual Inversion</span>**

为了找到这些新的嵌入，作者使用一小组图像，通常&#x4E3A;**`3`**&#x81F3;**`5`**&#x5F20;，这些图像在不同背景或姿态下展示了目标概念。作者通过直接优化来求解$$v^*$$，即在从该小样本集中采样的图像上最小化 LDM 损失：

$$v^* = \arg\min_v \mathbb{E}_{z \sim E(x), y, \epsilon \sim \mathcal{N}(0,I), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c_\theta(y))\|^2_2 \right]$$

该优化过程复用了原始 LDM 模型的训练方案，但保持$$c_\theta$$和$$\epsilon_\theta$$固定不变。这是一个重建任务，因此需要促使所学习的嵌入捕捉到该概念独有的精细视觉特征。

* **<span style="color: rgb(36,91,219); background-color: inherit">实现细节</span>**

作者沿用 LDM 的原始超参数设置。词嵌入使用目标对象的单字粗略描述词的嵌入进行初始化。实验&#x5728;**`2`**&#x5757;V100 GPU上进行，批大小&#x4E3A;**`4`**。基础学习率设&#x4E3A;**`0.005`**，并根据GPU数量和批量大小进行缩放，得到实际学习率&#x4E3A;**`0.04`**。所有结果均通&#x8FC7;**`5000`**&#x6B65;优化生成。作者发现这些参数在大多数情况下表现良好。但对于某些概念，减少优化步数或提高学习率可能获得更优结果。

### 4.3.3 <span style="color: rgb(36,91,219); background-color: inherit">DreamBooth</span>

给定仅少数随意拍摄的某一特定主体的图像，&#x5373;**`3`**&#x5230;**`5`**&#x5F20;，且无任何文本描述，DreamBooth 的目标是<span style="color: rgb(100,37,208); background-color: inherit">生成该主体的新图像，要求细节保真度高，并能根据文本 prompt 进行可控变化</span>。可能的变化包括改变主体所处的场景、颜色或形状等属性，调整主体的姿态、视角，以及其他语义层面的修改。并且不对输入图像的拍摄条件做任何限制，主体图像可具有不同的上下文环境。

* **<span style="color: rgb(36,91,219); background-color: inherit">扩散模型</span>**

扩散模型通过逐步去噪从高斯分布中采样的变量来学习数据分布。这里关注预训练的文本到图像扩散模型 $$\hat{x}_\theta$$，该模型接收初始噪声图$$\epsilon \sim \mathcal{N}(0, I)$$和一个由文本编码器$$\Gamma$$根据文本 prompt $$P$$生成的条件向量$$c = \Gamma(P)$$，并生成图像$$x_{\text{gen}} = \hat{x}_\theta(\epsilon, c)$$。模型通过平方误差损失函数进行训练，用于对不同程度加噪的图像或潜在编码 $$z_t := \alpha_t x + \sigma_t \epsilon$$ 进行去噪：

$$\mathbb{E}_{x,c,\epsilon,t} \left[ w_t \left\| \hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x \right\|^2_2 \right]$$

其中$$x$$为真实图像，$$c$$为条件向量，一般来自文本 prompt，$$\alpha_t$$、$$\sigma_t$$、$$w_t$$为控制噪声调度和样本质量的系数，是扩散过程时间$$t \sim U([0, 1])$$的函数。

![](../../images/视觉多模态讲义（下）-image-82.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">扩散模型的个性化</span>**

**DreamBooth&#x20;**&#x7684;首要任务是将特定主体实例植入模型的输出域中，从而能够通过查询模型生成该主体在不同情境下的新图像。一个自然的想法是使用该主体的少量样本数据集对模型进行微调。然而，在少样本场景下微调生成模型如 GAN 需格外谨慎，<span style="color: rgb(216,57,49); background-color: inherit">容易导致过拟合、模式坍缩，或未能充分捕捉目标分布</span>。之前一些工作提出若干方法避免这些问题，但这些方法主要目标是生成符合目标分布的图像，而不要求保持主体一致性。

> 作者发现一个有趣的现象：<span style="color: rgb(100,37,208); background-color: inherit">若采用扩散损失并精心设计微调方案，大型文本到图像扩散模型能够在不遗忘原有知识、也不过拟合于少量训练图像的前提下，有效融合新信息</span>。

1. **<span style="color: rgb(36,91,219); background-color: inherit">prompt 设计</span>**

**DreamBooth&#x20;**&#x7684;目标是将一个新&#x7684;**`(unique identifier, subject)`**&#x5BF9;植入扩散模型的词典中。为了避免详细描述每组图像的繁琐过程，采用简化策略：将所有主体图像统一标注&#x4E3A;**`a [identifier] [class noun]`**，其&#x4E2D;**`[identifier]`**&#x662F;与该主体关联的唯一标识符，**`[class noun]`**&#x662F;主体的粗粒度类别描述，&#x5982;**`cat`**、**`dog`**、**`watch`**&#x7B49;。<span style="color: rgb(100,37,208); background-color: inherit">类别描述可由用户指定，或通过分类器自动获取。在 prompt 中引入类别描述，是为了将该类别的先验知识与唯一标识符绑定</span>。<span style="color: rgb(216,57,49); background-color: inherit">若使用错误或缺失类别描述，会导致训练时间增加、语言漂移加剧，且性能下降</span>。本质上是利用模型对该类别的已有先验，并将其与主体唯一标识符的嵌入耦合，从而在不同上下文中生成主体的新姿态和形态。

* **<span style="color: rgb(36,91,219); background-color: inherit">稀有 token 标识符</span>**

作者发现常见的英文词汇，&#x5982;**`unique`**、**`special`**&#x4F5C;为标识符效果不佳，因为模型需学习将其原始语义解耦，并重新关联至主体。因此，<span style="color: rgb(100,37,208); background-color: inherit">理想的标识符应在语言模型和扩散模型中均具有较弱的先验</span>。一种做法是随机拼接英文字母生成罕见标识符，&#x5982;**`xxy5syt00`**，但实际上分词器可能将每个字母单独切分，而这些字母本身在模型中已有较强先验，导致类似常见词的问题。

**DreamBooth&#x20;**&#x7684;方法是：<span style="color: rgb(100,37,208); background-color: inherit">在词汇表中查找稀有 token ，再将其逆映射回文本空间，以最小化标识符具有强先验的概率。在词汇表中查找稀有 token 序列</span>$$f(\hat{V})$$<span style="color: rgb(100,37,208); background-color: inherit">，其中</span>$$f$$<span style="color: rgb(100,37,208); background-color: inherit">为分词器，将字符序列映射为 token，</span>$$\hat{V}$$<span style="color: rgb(100,37,208); background-color: inherit">为这些 token 解码后的文本。该序列长度可变，较短的序列</span>$$k = \{1, ..., 3\}$$<span style="color: rgb(100,37,208); background-color: inherit">效果良好。然后通过分词器的逆操作作用于</span>$$f(\hat{V})$$<span style="color: rgb(100,37,208); background-color: inherit">，得到一组字符序列作为唯一标识符</span>$$\hat{V}$$。对&#x4E8E;**`Imagen`**&#x6A21;型，作者发现使用均匀随机采样、对应不超&#x8FC7;**`3`**&#x4E2A;Unicode字符、且位&#x4E8E;**`T5-XXL`**&#x5206;词器范围 $$\{5000, ..., 10000\}$$ 内的 token ，效果较好。

* **<span style="color: rgb(36,91,219); background-color: inherit">损失函数</span>**

为最大化主体保真度，最佳策略是微调模型的所有层，包括依赖文本嵌入的层。但这会引发语言漂移问题：<span style="color: rgb(216,57,49); background-color: inherit">一个在大规模语料上预训练的模型，在特定任务微调后，会逐渐丧失语言的语法和语义知识</span>。这是首次发现扩散模型中存在类似现象——模型在微调后会逐渐遗忘如何生成与目标主体同类的其他实例。

另一个问题是输出多样性下降。文本到图像扩散模型天然具有高度的输出多样性。在少量图像上微调时，仍希望模型能生成主体在新视角、新姿态下的图像。然而，存在输出姿态和视角趋于训练图像的风险。尤其在训练轮数过多时，这种情况尤为明显。

![](../../images/视觉多模态讲义（下）-image-83.png)

为缓解上述两个问题，提出一种自生成&#x7684;**<span style="color: rgb(216,57,49); background-color: inherit">类别特定先验保持损失</span>**<span style="color: rgb(216,57,49); background-color: inherit"> class-specific prior preservation loss</span>，以促进多样性并抑制语言漂移。核心思想是：<span style="color: rgb(100,37,208); background-color: inherit">利用模型自身生成的样本来监督微调过程，使其在开始少样本微调后仍能保留类别先验知识</span>。使用冻结的预训练扩散模型，通过采样器生成先验数据$$x_{\text{pr}} = \hat{x}(z_{t_1}, c_{\text{pr}})$$，其中初始噪声 $$z_{t_1} \sim \mathcal{N}(0, I)$$，条件向量$$c_{\text{pr}} := \Gamma(f(\text{``a [class noun]''}))$$。损失函数变为：

$$\mathbb{E}_{x,c,\epsilon,\epsilon',t} \left[ w_t \left\| \hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x \right\|^2_2 + \lambda w_{t'} \left\| \hat{x}_\theta(\alpha_{t'} x_{\text{pr}} + \sigma_{t'} \epsilon', c_{\text{pr}}) - x_{\text{pr}} \right\|^2_2 \right]$$

其中第二项为先验保持项，用于以模型自生成图像监督训练；$$\lambda$$控制该项的相对权重。<span style="color: rgb(46,161,33); background-color: inherit">该损失在提升输出多样性和缓解语言漂移方面非常有效。此外可以在不引发过拟合的前提下进行更多轮次的训练</span>。

> * **<span style="color: rgb(36,91,219); background-color: inherit">Imagen</span>**：使用$$\lambda = 1$$、学习率$$10^{-5}$$，在单块 TPUv4 上，Imagen 的训练耗时约5分钟
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Stable Diffusion</span>**：使用$$5 \times 10^{-6}$$，训练约1000轮，输入3-5张图像，即可获得良好效果。在 NVIDIA A100 上，Stable Diffusion 的训练约需5分钟

在此过程中，约生成1000&#x4E2A;**`a [class noun]`**&#x6837;本。

### 4.3.4 <span style="color: rgb(36,91,219); background-color: inherit">InstructPix2Pix</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">两阶段方法：合成编辑数据，再训练前向编辑模型</span>**

InstructPix2Pix 把指令式图像编辑写成监督学习问题。第一阶段自动构造“编辑前图像、编辑指令、编辑后图像”三元组；第二阶段用这些三元组训练条件扩散模型。推理时只需一张真实输入图和一条自然语言编辑指令，不需要用户提供掩码、目标图像完整描述，也不需要对单个输入执行优化或微调。

完整数据生成与模型训练管线如下。

![InstructPix2Pix 的合成编辑数据生成与条件扩散训练管线](../../images/视觉多模态讲义（下）-instructpix2pix-fig2-pipeline.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">用 GPT-3 生成指令与成对描述</span>**

文本阶段的输入是一条真实图像描述 $$c_{\mathrm{before}}$$，输出是编辑指令 $$c_T$$ 和编辑后的描述 $$c_{\mathrm{after}}$$。监督数据先由人工构造 **`700`** 个三元组：

$$\left(c_{\mathrm{before}},\;c_T,\;c_{\mathrm{after}}\right)$$

输入描述来自 LAION-Aesthetics V2 6.5+，人工编写对应编辑指令和编辑后描述。GPT-3 Davinci 用默认参数微调 **`1`** 个 epoch，然后对去重后的 LAION 图像描述批量生成文本三元组，最终得到 **`454,445`** 条指令与描述对。

只在文本域生成编辑计划有两个好处：编辑类型可以覆盖对象、属性、背景、风格和布局变化；同时，编辑前后描述与指令由同一次生成保持语义对应。LAION 的噪声则在后续图像配对过滤和条件引导中进一步抑制。

* **<span style="color: rgb(36,91,219); background-color: inherit">用 Prompt-to-Prompt 生成配对图像</span>**

若分别用 Stable Diffusion 对 $$c_{\mathrm{before}}$$ 和 $$c_{\mathrm{after}}$$ 独立采样，即使两条描述只改了一个词，构图、人物身份和背景也可能完全变化。这种图像对无法监督“编辑”，只会让模型学习重新生成。

因此，生成编辑后图像时使用 Prompt-to-Prompt，在部分去噪步骤共享编辑前后两次生成的 cross-attention 权重。共享比例记为 $$p$$：较大的 $$p$$ 保持更强的空间一致性，较小的 $$p$$ 允许更大结构变化。

下图对比独立生成与共享 cross-attention 后的配对结果。

![Prompt-to-Prompt 共享注意力前后生成图像对的一致性差异](../../images/视觉多模态讲义（下）-instructpix2pix-fig3-prompt-to-prompt.png)

不同编辑所需的共享比例不同，无法只根据文本可靠预测最佳 $$p$$。对每个描述对，方法生成 **`100`** 组候选图像，且每组的共享比例从以下分布随机采样：

$$p\sim\mathcal{U}(0.1,0.9)$$

随后用 CLIP 方向相似度过滤。设图像编码为 $$E_I(\cdot)$$，文本编码为 $$E_T(\cdot)$$，图像变化方向与文本变化方向分别为：

$$\Delta I=E_I(x_{\mathrm{after}})-E_I(x_{\mathrm{before}}),\qquad \Delta T=E_T(c_{\mathrm{after}})-E_T(c_{\mathrm{before}})$$

按 $$\Delta I$$ 与 $$\Delta T$$ 的余弦相似度选择候选，使图像发生的语义变化与文本要求的变化一致。该过滤既控制图像对质量，也缓解 Prompt-to-Prompt 或 Stable Diffusion 偶发失败。

* **<span style="color: rgb(36,91,219); background-color: inherit">基于 Stable Diffusion 的编辑网络</span>**

编辑模型从预训练 Stable Diffusion 初始化，在 VAE 潜空间工作。输入图像 $$c_I$$ 经 VAE 编码器 $$\mathcal{E}$$ 得到条件潜变量 $$\mathcal{E}(c_I)$$；目标编辑图像 $$x$$ 的潜变量 $$z=\mathcal{E}(x)$$ 在时刻 $$t$$ 加噪得到 $$z_t$$。U-Net 同时接收噪声潜变量、输入图像潜变量和文本指令：

$$\epsilon_\theta\!\left(z_t,t,\mathcal{E}(c_I),c_T\right)$$

训练目标是预测加入目标潜变量的噪声：

$$\mathcal{L}=\mathbb{E}_{\mathcal{E}(x),\mathcal{E}(c_I),c_T,\epsilon,t}\left[\left\|\epsilon-\epsilon_\theta(z_t,t,\mathcal{E}(c_I),c_T)\right\|_2^2\right]$$

为了加入图像条件，第一层卷积增加输入通道，把 $$z_t$$ 与 $$\mathcal{E}(c_I)$$ 沿通道维拼接。原有通道权重继承预训练 Stable Diffusion；新增通道对应的权重初始化为零。<span style="color: rgb(100,37,208); background-color: inherit">这样初始化时模型仍保持原有去噪函数，图像条件的影响从零开始学习</span>。文本条件接口保持不变，但输入从完整图像描述改为编辑指令 $$c_T$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">双条件 classifier-free guidance</span>**

普通 classifier-free guidance 在条件预测与无条件预测之间外推：

$$\tilde\epsilon_\theta(z_t,c)=\epsilon_\theta(z_t,\varnothing)+s\left(\epsilon_\theta(z_t,c)-\epsilon_\theta(z_t,\varnothing)\right)$$

InstructPix2Pix 有输入图像 $$c_I$$ 和文本指令 $$c_T$$ 两个条件。训练时，**`5%`** 样本只丢弃图像条件，**`5%`** 只丢弃文本条件，另有 **`5%`** 同时丢弃两个条件。模型因此能估计三个必要状态：完全无条件、仅图像条件、图像加文本条件。

推理时使用两个独立引导尺度：

$$\begin{aligned}\tilde\epsilon_\theta(z_t,c_I,c_T)=&\;\epsilon_\theta(z_t,\varnothing,\varnothing)\\&+s_I\left(\epsilon_\theta(z_t,c_I,\varnothing)-\epsilon_\theta(z_t,\varnothing,\varnothing)\right)\\&+s_T\left(\epsilon_\theta(z_t,c_I,c_T)-\epsilon_\theta(z_t,c_I,\varnothing)\right)\end{aligned}$$

$$s_I$$ 放大输入图像条件相对于无条件预测的方向，值越大，输出越保持原图空间结构；$$s_T$$ 放大文本指令在给定输入图基础上的增量方向，值越大，编辑越强。两者分别控制“保留多少原图”和“执行多强编辑”。

右图完整展示两个引导尺度的独立作用。



![InstructPix2Pix 中图像引导尺度与文本引导尺度的独立作用](../../images/视觉多模态讲义（下）-instructpix2pix-fig4-guidance.png)

### 4.3.5 <span style="color: rgb(36,91,219); background-color: inherit">ControlNet</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">动机</span>**

在 Stable Diffusion 这样的生成模型风靡后，创作者面临了一个难题：虽然写下的文字可以帮助 AI 生成图像，但<span style="color: rgb(216,57,49); background-color: inherit">想要精准控制画面结构、姿态或布局，单靠一句 prompt 往往得经过反复调整，效率低且缺乏直觉反馈</span>。

为了解决这个问题，**ControlNet** 被提出。它的核心思路是<span style="color: rgb(100,37,208); background-color: inherit">在</span>**<span style="color: rgb(100,37,208); background-color: inherit">冻结预训练生成模型</span>**<span style="color: rgb(100,37,208); background-color: inherit">的基础上，增添</span>**<span style="color: rgb(100,37,208); background-color: inherit">可训练的控制分支</span>**<span style="color: rgb(100,37,208); background-color: inherit">，通过</span>**<span style="color: rgb(216,57,49); background-color: inherit">零卷积</span>**<span style="color: rgb(216,57,49); background-color: inherit"> zero convolution</span> <span style="color: rgb(100,37,208); background-color: inherit"> 将控制信息稳定注入生成流程中</span>。这样设计既保留了原始模型的表达能力，也避免了因微调导致的破坏性干扰，并且使得多个类型的条件输入都能被自然整合进图像生成过程，如<span style="color: rgb(220,155,4); background-color: inherit">边缘图、深度图、人体骨架、分割图</span>等，从而实现对图像结构、姿势和场景布局的细致掌控

* **<span style="color: rgb(36,91,219); background-color: inherit">网络结构</span>**

ControlNet <span style="color: rgb(100,37,208); background-color: inherit">将额外的条件注入神经网络的各个模块中</span>。网络模块指 ResNet 模块、conv-bn-relu 模块、多头注意力模块、Transformer 模块等。假设$$  F(\cdot; \Theta)  $$是一个已训练好的神经模块，其参数为$$\Theta$$，它将输入特征图$$  x  $$映射为输出特征图$$y$$，即：

$$y = F(x; \Theta)$$

![](../../images/视觉多模态讲义（下）-image-88.png)

$$x$$和$$  y  $$通常是二维特征图，即$$x \in \mathbb{R}^{h \times w \times c}$$，其中$$  h, w, c  $$分别表示特征图的高度、宽度和通道数。

为了向这样一个预训练的神经模块添加 ControlNet，一般<span style="color: rgb(100,37,208); background-color: inherit">将原始模块的参数</span>$$  \Theta  $$<span style="color: rgb(100,37,208); background-color: inherit">冻住，同时复制一个可训练的网络模块，设参数为</span>$$\Theta_c$$。这个可训练的网络模块接收一个外部的条件向量$$  c  $$作为输入。当这种结构应用于 Stable Diffusion时，冻住的参数保留了用数十亿图像训练出的、可用于生产的模型；而<span style="color: rgb(100,37,208); background-color: inherit">复制出的可训练部分则复用这一大规模预训练模型，建立起一个深层、鲁棒且强大的主干网络，以处理多样化的输入条件</span>。

可训练的网络模块通过零卷积层 zero convolution layers 与锁定的模型相连，一般记作$$Z(\cdot; \cdot)$$，**<span style="color: rgb(100,37,208); background-color: inherit">其是一个</span>$$  1 \times 1  $$<span style="color: rgb(100,37,208); background-color: inherit">卷积层，权重和偏置均初始化为零</span>**。为了构建 ControlNet，作者使用两个零卷积，参数分别记为$$  \Theta_{z1}  $$和$$\Theta_{z2}$$。完整的 ControlNet 计算如下：

$$y_c = F(x; \Theta) + Z\left(F\left(x + Z(c; \Theta_{z1}); \Theta_c\right); \Theta_{z2}\right)$$

其中 $$  y_c  $$ 是 ControlNet 模块的输出。

在训练的第一步中，由于零卷积层的权重和偏置均初始化为零，则两个$$  Z(\cdot; \cdot)  $$项均为零，因此：

$$y_c = y.$$

这样，**<span style="color: rgb(100,37,208); background-color: inherit">在训练初期，有害的噪声不会影响可训练模块中神经网络层的隐藏状态</span>**。由于$$Z(c; \Theta_{z1}) = 0$$，且可训练模块仍然接收原始输入图像$$x$$，因此这个可训练模块是功能完整的，且保留了大规模预训练模型的能力，可作为后续学习的主干网络。**<span style="color: rgb(100,37,208); background-color: inherit">零卷积通过在训练初期消除梯度中的随机噪声来保护这个主干结构</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">ControlNet 的用法</span>**

Stable Diffusion 本质上是一个带有编码器、中间模块和残差连接解码器的 U-Net。<span style="color: rgb(100,37,208); background-color: inherit">编码器和解码器各包含</span>**`12`**<span style="color: rgb(100,37,208); background-color: inherit">个模块，算上中间模块一共包含</span>**`25`**<span style="color: rgb(100,37,208); background-color: inherit">个模块。这</span>**`25`**<span style="color: rgb(100,37,208); background-color: inherit">个模块中，有</span>**`8`**<span style="color: rgb(100,37,208); background-color: inherit">个是下采样或上采样卷积层，其余</span>**`17`**<span style="color: rgb(100,37,208); background-color: inherit">个是主模块，每个主模块包含</span>**`4`**<span style="color: rgb(100,37,208); background-color: inherit">个</span>**`ResNet`**<span style="color: rgb(100,37,208); background-color: inherit">层和</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">个</span>**`ViT`**<span style="color: rgb(100,37,208); background-color: inherit">。每个</span>**`ViT`**<span style="color: rgb(100,37,208); background-color: inherit">包含多个交叉注意力和自注意力机制</span>。如右图，**`SD Encoder Block A`**<span style="color: rgb(220,155,4); background-color: inherit">包含 4 个 ResNet 层和 2 个 ViT，而</span>**`×3`**<span style="color: rgb(220,155,4); background-color: inherit">表示该模块重复三次。文本 prompt 通过 CLIP 文本编码器编码，扩散时间步则通过位置编码的时间编码器进行编码</span>。

ControlNet 应用在 U-Net 的每一级编码器，即<span style="color: rgb(100,37,208); background-color: inherit">为 Stable Diffusion 的 12 个编码模块和 1 个中间模块分别复制出一个可训练的网络模块</span>。这 12 个编码模块<span style="color: rgb(100,37,208); background-color: inherit">分布在 4 种分辨率上：</span>**`64×64`**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**`32×32`**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**`16×16`**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**`8×8`**，每种分辨率重复 3 次。这些模块的<span style="color: rgb(100,37,208); background-color: inherit">输出被加到 U-Net 的 12 个跳跃连接和 1 个中间模块上</span>。由于ControlNet 是一种即插即用的模块，因此很容易适用于其他模型。



![](../../images/视觉多模态讲义（下）-image-92.png)

由于原始模型的参数是被冻结的，微调过程中不需要在原始编码器中进行梯度计算，加快了训练速度并节省了 GPU 内存。<span style="color: rgb(46,161,33); background-color: inherit">在单张 A100 40GB 显卡上使用 ControlNet 优化 Stable Diffusion 时，训练仅需多出约</span>**`23%`**<span style="color: rgb(46,161,33); background-color: inherit">的 GPU 内存和</span>**`34%`**<span style="color: rgb(46,161,33); background-color: inherit">的时间</span>。

为了将 ControlNet 添加到 Stable Diffusion 中，首先要<span style="color: rgb(100,37,208); background-color: inherit">将每个输入条件图像，如</span> <span style="color: rgb(220,155,4); background-color: inherit">边缘、姿态、深度</span> <span style="color: rgb(100,37,208); background-color: inherit">等，从</span>$$  512 \times 512  $$<span style="color: rgb(100,37,208); background-color: inherit">的输入尺寸转换为与 Stable Diffusion 匹配的</span>$$  64 \times 64  $$<span style="color: rgb(100,37,208); background-color: inherit">特征空间向量</span>。这里作者使用一个小型网络 $$E(\cdot)$$，包含 4 个卷积层，卷积核为$$4 \times 4$$，步幅为$$2 \times 2$$，激活函数为 ReLU，通道数分别&#x4E3A;**`16`**、**`32`**、**`64`**、**`128`**，权重使用高斯初始化，并与整个模型一起联合训练，将图像空间中的条件$$  c_i  $$编码为特征空间中的条件向量$$c_f$$：

$$c_f = E(c_i)$$

这个条件向量 $$  c_f  $$ 被送入 ControlNet。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：Stable Diffusion 使用潜在空间作为训练域，即<span style="color: rgb(100,37,208); background-color: inherit">使用 VAE 将</span>$$  512 \times 512  $$<span style="color: rgb(100,37,208); background-color: inherit">的像素空间图像转换为更小的</span>$$  64 \times 64  $$<span style="color: rgb(100,37,208); background-color: inherit">潜在图像</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练</span>**

**ControlNet** 学习目标：给定输入图像$$z_0$$，逐步向图像添加噪声，生成带噪图像$$z_t$$，其中$$  t  $$表示噪声添加的次数。<span style="color: rgb(100,37,208); background-color: inherit">给定一组条件，包括时间步</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">、文本prompt </span>$$  c_t  $$<span style="color: rgb(100,37,208); background-color: inherit">以及任务特定的条件</span>$$c_f$$<span style="color: rgb(100,37,208); background-color: inherit">，要学习一个网络</span>$$  \epsilon_\theta  $$<span style="color: rgb(100,37,208); background-color: inherit">来预测添加到带噪图像</span>$$  z_t  $$<span style="color: rgb(100,37,208); background-color: inherit">上的噪声</span>，损失函数为：

$$L = \mathbb{E}_{z_0, t, c_t, c_f, \epsilon \sim \mathcal{N}(0,1)} \left[ \left\| \epsilon - \epsilon_\theta(z_t, t, c_t, c_f) \right\|_2^2 \right]$$

其中 $$  L  $$ 是整个扩散模型的总体学习目标，也是 ControlNet 的学习目标。

在训练过程中，作者团队随机&#x5C06;**`50%`**&#x7684;文本 prompt  $$  c_t  $$ 替换为空字符串。这种方法增强了 ControlNet 直接从输入条件图像，如边缘、姿态、深度等，识别语义信息的能力，从而替代文本 prompt 的作用。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：有意思的观察
>
> ControlNet 并非逐步学习控制条件，而是突然较好地遵循输入条件图像，如下图，作者将这一现象称&#x4E3A;**<span style="color: rgb(216,57,49); background-color: inherit">突然收敛现象</span>**<span style="color: rgb(216,57,49); background-color: inherit"> sudden convergence phenomenon</span>。

![](../../images/视觉多模态讲义（下）-image-91.png)

![](../../images/视觉多模态讲义（下）-image-90.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">推理</span>**

作者通过多种方式进一步控制 ControlNet 的额外条件影响去噪扩散过程：

1. **<span style="color: rgb(36,91,219); background-color: inherit">Classifier-free guidance 的分辨率加权</span>**

Stable Diffusion 通常使用 **<span style="color: rgb(216,57,49); background-color: inherit">CFG</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">C</span>**<span style="color: rgb(216,57,49); background-color: inherit">lassifier-</span>**<span style="color: rgb(216,57,49); background-color: inherit">F</span>**<span style="color: rgb(216,57,49); background-color: inherit">ree </span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">uidance）</span>来生成高质量图像。公式为：

$$\epsilon_{\text{prd}} = \epsilon_{\text{uc}} + \beta_{\text{cfg}} (\epsilon_c - \epsilon_{\text{uc}})$$

其中$$  \epsilon_{\text{prd}}  $$是模型的最终输出，$$\epsilon_{\text{uc}}$$是无条件输出，$$\epsilon_c$$是条件输出，$$\beta_{\text{cfg}}$$是用户指定的权重。

<span style="color: rgb(100,37,208); background-color: inherit">当通过 ControlNet 添加条件图像时，可以被同时添加到</span>$$  \epsilon_{\text{uc}}  $$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$  \epsilon_c  $$<span style="color: rgb(100,37,208); background-color: inherit">中，也可以只添加到</span>$$  \epsilon_c  $$<span style="color: rgb(100,37,208); background-color: inherit">中</span>。在一些挑战性场景中,例如<span style="color: rgb(220,155,4); background-color: inherit">无文本 prompt</span>，

> * 若同时添加到两者中，会完全消除 CFG 引导，如右图(b)
>
> * 若仅添加到$$\epsilon_c$$，则引导会过强，如右图(c)

![](../../images/视觉多模态讲义（下）-image-87.png)

作者的解决方案是：<span style="color: rgb(100,37,208); background-color: inherit">首先将条件图像仅添加到</span>$$  \epsilon_c  $$<span style="color: rgb(100,37,208); background-color: inherit">中，然后根据每个模块的分辨率，对 Stable Diffusion 与 ControlNet 之间的每条连接乘以一个权重</span>$$w_i$$：

$$w_i = \frac{64}{h_i}$$

其中$$  h_i  $$是第$$  i  $$个模块的空间尺寸，例如$$h_1 = 8, h_2 = 16, \dots, h_{13} = 64$$。通过降低 CFG 引导的强度，可以实现上图(d)所示的效果，作者将这种方法称为 **<span style="color: rgb(216,57,49); background-color: inherit">CFG 分辨率加权</span>**。

* **<span style="color: rgb(36,91,219); background-color: inherit">组合多个 ControlNet</span>**

为了将多个条件图像应用于同一个 Stable Diffusion 实例，例如 <span style="color: rgb(220,155,4); background-color: inherit">Canny 边缘和姿态</span>，可以直接将对应 ControlNet 的输出相加到 Stable Diffusion 模型中，并且无需额外的加权或线性插值。

![](../../images/视觉多模态讲义（下）-image-89.png)

### 4.3.6 <span style="color: rgb(36,91,219); background-color: inherit">IP-Adapter</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">冻结文生图模型，只增加图像提示分支</span>**

IP-Adapter 让预训练文生图扩散模型在保留文本提示能力的同时，额外接收一张图像提示。基础模型以 Stable Diffusion 为例：文本经冻结的 CLIP 文本编码器得到文本特征，带 cross-attention 的 U-Net 在潜空间预测噪声。训练 IP-Adapter 时，原始文本编码器、U-Net 和 VAE 全部冻结，只更新图像投影网络和新增的图像 cross-attention 参数。

整体架构如下。蓝色模块是冻结的基础模型，红色模块是新增并训练的 IP-Adapter。

![IP-Adapter 的图像提示编码器与解耦 cross-attention 结构](../../images/视觉多模态讲义（下）-ip-adapter-fig2-architecture.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">基础扩散目标</span>**

给定真实潜变量 $$x_0$$、条件 $$c$$、时间步 $$t$$ 与高斯噪声 $$\epsilon$$，前向加噪写为：

$$x_t=\alpha_t x_0+\sigma_t\epsilon$$

噪声预测模型 $$\epsilon_\theta$$ 的简化训练目标为：

$$\mathcal{L}_{\mathrm{simple}}=\mathbb{E}_{x_0,\epsilon\sim\mathcal{N}(0,I),c,t}\left[\left\|\epsilon-\epsilon_\theta(x_t,c,t)\right\|_2^2\right]$$

条件生成通常使用 classifier-free guidance，把条件预测和无条件预测线性组合：

$$\hat\epsilon_\theta(x_t,c,t)=w\epsilon_\theta(x_t,c,t)+(1-w)\epsilon_\theta(x_t,t)$$

$$w$$ 控制采样结果与条件的一致程度。IP-Adapter 保留这一扩散训练和采样框架，只改变图像条件进入 U-Net 的方式。

* **<span style="color: rgb(36,91,219); background-color: inherit">图像提示编码器</span>**

图像提示先经过冻结的 CLIP 图像编码器。方法使用 CLIP 的全局图像嵌入，因为它与图像描述语义对齐，并同时包含内容与风格信息。一个小型可训练投影网络把全局嵌入分解为长度为 $$N$$ 的图像特征序列：

$$c_i=\operatorname{LN}(W_i e_{\mathrm{CLIP}}+b_i)$$

原始实现取 **`N=4`**。图像特征维度与基础扩散模型的文本特征维度相同；投影网络由线性层和 Layer Normalization 构成。

* **<span style="color: rgb(36,91,219); background-color: inherit">解耦 cross-attention</span>**

原始 Stable Diffusion 的 cross-attention 使用 U-Net 特征 $$Z$$ 产生 query，用文本特征 $$c_t$$ 产生 key 与 value：

$$Z'=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}\right)V$$

$$Q=ZW_q,\qquad K=c_tW_k,\qquad V=c_tW_v$$

如果直接把图像特征与文本特征拼接后送入同一个冻结 cross-attention，二者会共享同一组 key/value 投影，图像特征难以充分嵌入预训练 U-Net。IP-Adapter 为每个原始文本 cross-attention 并联一个新的图像 cross-attention：

$$Z''=\operatorname{Softmax}\left(\frac{Q(K')^\top}{\sqrt d}\right)V'$$

$$K'=c_iW'_k,\qquad V'=c_iW'_v$$

文本分支与图像分支共享同一个 query $$Q$$，但拥有独立的 key/value 投影。最终输出是两条注意力分支之和：

$$Z_{\mathrm{new}}=\operatorname{Softmax}\left(\frac{QK^\top}{\sqrt d}\right)V+\operatorname{Softmax}\left(\frac{Q(K')^\top}{\sqrt d}\right)V'$$

每个 U-Net cross-attention 层只新增 $$W'_k$$ 和 $$W'_v$$ 两组参数。它们从原始文本分支的 $$W_k$$、$$W_v$$ 初始化，以加快收敛；训练时原始 $$W_q,W_k,W_v$$ 全部冻结。

> **<span style="color: rgb(36,91,219); background-color: inherit">解耦的含义</span>**
>
> <span style="color: rgb(100,37,208); background-color: inherit">文本与图像不在输入 token 维度提前拼接，而是分别完成 cross-attention，再在输出特征上相加。</span>这样既不破坏原有文本注意力，又给图像提示独立的投影空间。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练目标与条件丢弃</span>**

训练数据是图像—文本对。文本条件为 $$c_t$$，图像提示条件为 $$c_i$$，只优化 IP-Adapter 参数：

$$\mathcal{L}_{\mathrm{IP}}=\mathbb{E}_{x_0,\epsilon,c_t,c_i,t}\left[\left\|\epsilon-\epsilon_\theta(x_t,c_t,c_i,t)\right\|_2^2\right]$$

为支持 classifier-free guidance，训练时随机丢弃图像条件；丢弃时直接把 CLIP 图像嵌入置零。条件与无条件预测组合为：

$$\hat\epsilon_\theta(x_t,c_t,c_i,t)=w\epsilon_\theta(x_t,c_t,c_i,t)+(1-w)\epsilon_\theta(x_t,t)$$

* **<span style="color: rgb(36,91,219); background-color: inherit">推理时独立调节图像提示强度</span>**

由于文本和图像 cross-attention 已经解耦，推理时可以只缩放图像分支：

$$Z_{\mathrm{new}}=\operatorname{Attention}(Q,K,V)+\lambda\operatorname{Attention}(Q,K',V')$$

$$\lambda$$ 是图像提示权重。$$\lambda=0$$ 时，新增分支完全关闭，模型退化为原始文生图扩散模型；提高 $$\lambda$$ 会增强参考图像内容或风格对生成结果的约束。<span style="color: rgb(46,161,33); background-color: inherit">文本条件与图像条件可以在同一 U-Net 中独立控制，而不需要重新训练基础模型</span>。

---

[Previous](07-Diffusion-Model--Stable-Diffusion-系列.md) | [Contents](../../README.md) | [Next](09-Diffusion-Model--OpenAI-工作.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=8)
