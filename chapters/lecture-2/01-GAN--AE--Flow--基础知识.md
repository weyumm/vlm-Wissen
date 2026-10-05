[Contents](../../README.md) | [Next](02-GAN--AE--Flow--GAN-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=1)

# 3. <span style="color: rgb(36,91,219); background-color: inherit">GAN &amp; AE &amp; Flow</span>

## 3.1 <span style="color: rgb(36,91,219); background-color: inherit">基础知识</span>

### 3.1.1 <span style="color: rgb(36,91,219); background-color: inherit">数学知识</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">雅可比矩阵与行列式</span>**

给定一个将$$n$$维输入向量$$\mathbf{x}$$映射到$$m$$维输出向量的函数$$\mathbf{f}: \mathbb{R}^n \mapsto \mathbb{R}^m$$，该函数<span style="color: rgb(100,37,208); background-color: inherit">所有一阶偏导数构成的矩阵称为</span>**<span style="color: rgb(100,37,208); background-color: inherit">雅可比矩阵</span>**$$\mathbf{J}$$，其中第$$i$$行第$$j$$列的元素为：

$$\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$$

即：

$$\mathbf{J} =
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

<span style="color: rgb(100,37,208); background-color: inherit">行列式是一个实数，它是对一个方阵中所有元素计算得到的函数。行列式都是方阵。行列式的绝对值可以理解为</span>**<span style="color: rgb(100,37,208); background-color: inherit">矩阵在乘法过程中对空间的拉伸或压缩程度的度量</span>**。

一个 $$n \times n$$ 矩阵 $$M$$ 的行列式定义为：

$$\det M = \det
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{n1} & a_{n2} & \cdots & a_{nn}
\end{bmatrix} = \sum_{j_1 j_2 \ldots j_n} (-1)^{\tau(j_1 j_2 \ldots j_n)} a_{1j_1} a_{2j_2} \ldots a_{nj_n}$$

其中，求和下标$$j_1 j_2 \ldots j_n$$是集合$$\{1, 2, \ldots, n\}$$的所有排列，共有$$n!$$项；$$\tau(\cdot)$$表示排列。

一个方阵$$M$$的行列式可用于判断其是否可逆：

> * 若$$\det(M) = 0$$，则$$M$$不可逆，即为<span style="color: rgb(100,37,208); background-color: inherit">奇异矩阵</span>，其行或列线性相关，或某行/列为全零
>
> * 若$$\det(M) \neq 0$$，则$$M$$可逆

矩阵乘积的行列式等于各矩阵行列式的乘积：

$$\det(AB) = \det(A)\det(B)$$

* **<span style="color: rgb(36,91,219); background-color: inherit">变量替换定理</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">单变量版本</span>**

给定一个随机变量$$z$$及其已知的概率密度函数$$z \sim \pi(z)$$，然后希望使用一个一一映射函数$$x = f(z)$$构造一个新的随机变量。函数$$f$$是可逆的，因此有$$z = f^{-1}(x)$$。现在的问题是：<span style="color: rgb(100,37,208); background-color: inherit">如何推导新变量的未知概率密度函数 </span>$$p(x)$$<span style="color: rgb(100,37,208); background-color: inherit">？</span>

根据概率分布的定义：

$$\int p(x)\,dx = \int \pi(z)\,dz = 1$$

通过变量替换可得：

$$p(x) = \pi(z) \left| \frac{dz}{dx} \right| = \pi(f^{-1}(x)) \left| \frac{df^{-1}}{dx} \right| = \pi(f^{-1}(x)) \left| (f^{-1})'(x) \right|$$

根据定义，积分$$\int \pi(z)\,dz$$是无限多个无穷小宽度$$\Delta z$$的矩形面积之和。在位置$$z$$处，矩形的高度是密度函数$$\pi(z)$$的值。当进行变量替换$$z = f^{-1}(x)$$时，有：

$$\frac{\Delta z}{\Delta x} = (f^{-1}(x))' \quad \text{且} \quad \Delta z = (f^{-1}(x))' \Delta x$$

这里$$\left|(f^{-1}(x))'\right|$$表示在变量$$z$$和$$x$$的两个不同坐标系下，矩形面积之间的比例关系。



* **<span style="color: rgb(36,91,219); background-color: inherit">多变量版本</span>**

设$$z \sim \pi(z)$$，$$x = f(z)$$，$$z = f^{-1}(x)$$，则：

$$p(x) = \pi(z) \left| \det \frac{dz}{dx} \right| = \pi(f^{-1}(x)) \left| \det \frac{df^{-1}}{dx} \right|$$

其中$$\det \frac{\partial f}{\partial x}$$是函数$$f$$的雅可比行列式。

### 3.1.2 <span style="color: rgb(36,91,219); background-color: inherit">GAN 基础</span>

生成对抗网络<span style="color: rgb(216,57,49); background-color: inherit"> </span>**<span style="color: rgb(216,57,49); background-color: inherit">GAN</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">enerative </span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">dversarial </span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">etwork）</span>是深度生成模型中的一项里程碑式工作，由 Ian Goodfellow 等人在 2014 年提出。其核心思想源于博弈论中&#x7684;**<span style="color: rgb(100,37,208); background-color: inherit">零和博弈</span>**<span style="color: rgb(100,37,208); background-color: inherit">，通过两个神经网络——生成器 Generator 与判别器 Discriminator 之间的动态对抗，实现对复杂数据分布的建模与采样</span>。这种对抗式的训练机制赋予了 GAN 强大的生成能力，广泛应用于图像生成、风格迁移、数据增强等领域。

GAN 的本质是一场由生成器$$G$$和判别器$$D$$共同参与&#x7684;**<span style="color: rgb(216,57,49); background-color: inherit">极小极大博弈</span>**<span style="color: rgb(216,57,49); background-color: inherit"> minimax game</span>。二者在训练过程中不断博弈、相互促进，最终趋于一个理想的平衡状态。

> * **<span style="color: rgb(36,91,219); background-color: inherit">判别器</span>**$$D$$：扮演侦探角色，<span style="color: rgb(100,37,208); background-color: inherit">接收一个数据样本</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">，输出一个标量值</span>$$D(x) \in [0, 1]$$<span style="color: rgb(100,37,208); background-color: inherit">，表示该样本为真实数据的概率</span>。其目标是尽可能准确地区分真实样本与生成样本。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">生成器</span>**$$G$$：扮演伪造者角色，<span style="color: rgb(100,37,208); background-color: inherit">接收一个从先验分布</span>$$z \sim p_z(z)$$<span style="color: rgb(100,37,208); background-color: inherit">中采样的随机噪声向量</span>$$z$$<span style="color: rgb(100,37,208); background-color: inherit">，将其映射为一个合成样本</span>$$G(z)$$。其目标是生成逼真的数据，使判别器难以分辨真伪。

这两个网络在对抗中共同进化：<span style="color: rgb(100,37,208); background-color: inherit">生成器越强，生成的样本越真实；判别器越强，识别能力越精准</span>。整个过程其实是一场猫鼠游戏，推动双方性能不断提升。

* **<span style="color: rgb(36,91,219); background-color: inherit">目标函数</span>**

GAN 的训练目标由一个统一的价值函数 $$V(D, G)$$ 来刻画，同时反映了判别器的判别能力和生成器的欺骗能力：

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

这一目标函数包含两个关键项：

> * 第一项$$\mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)]$$<span style="color: rgb(100,37,208); background-color: inherit">鼓励判别器对真实样本赋予高置信度</span>，即$$D(x) \to 1$$，从而提升其识别真实数据的能力
>
> * 第二项$$\mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$<span style="color: rgb(100,37,208); background-color: inherit">鼓励判别器将生成样本识别为假</span>，即$$D(G(z)) \to 0$$

从博弈角度理解：

> * 判别器$$D$$<span style="color: rgb(100,37,208); background-color: inherit">试图最大化</span>$$V(D, G)$$，以增强其辨别能力
>
> * 生成器$$G$$<span style="color: rgb(100,37,208); background-color: inherit">试图最小化</span>$$V(D, G)$$，从而欺骗判别器

整个学习过程可形式化为一个极小极大优化问题：

$$\min_G \max_D V(D, G)$$

这正是 GAN 对抗机制的数学核心。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练过程</span>**

GAN 的训练采&#x7528;**<span style="color: rgb(216,57,49); background-color: inherit">交替优化</span>**<span style="color: rgb(216,57,49); background-color: inherit"> alternating optimization </span>的方式，分别更新判别器和生成器的参数，通常使用基于梯度的优化算法，&#x5982;**`Adam`**。

> * **<span style="color: rgb(36,91,219); background-color: inherit">固定生成器，优化判别器</span>**
>
> 在每一轮迭代中，首先固定生成器$$G$$，通过梯度上升法最大化价值函数，以增强判别器的判别能力：
>
> $$\nabla_{\theta_d} \frac{1}{m} \sum_{i=1}^m \left[ \log D(x^{(i)}) + \log(1 - D(G(z^{(i)}))) \right]$$
>
> 其中，$$x^{(i)} \sim p_{\text{data}}$$为真实样本，$$z^{(i)} \sim p_z$$为噪声输入。此时，判别器执行一个标准的二分类任务：真实样本标签为 1，生成样本标签为 0。

> * **<span style="color: rgb(36,91,219); background-color: inherit">固定判别器，优化生成器</span>**
>
> 随后固定判别器$$D$$，通过梯度下降法更新生成器参数$$\theta_g$$，使其生成的样本更接近真实分布：
>
> $$\nabla_{\theta_g} \frac{1}{m} \sum_{i=1}^m \log(1 - D(G(z^{(i)})))$$
>
> 然而，在训练初期，<span style="color: rgb(216,57,49); background-color: inherit">若生成样本质量较差，导致</span>$$D(G(z)) \approx 0$$<span style="color: rgb(216,57,49); background-color: inherit">，则</span>$$\log(1 - D(G(z)))$$<span style="color: rgb(216,57,49); background-color: inherit">的梯度极小，生成器难以有效学习</span>，即梯度消失问题。
>
> 为缓解这一问题，<span style="color: rgb(100,37,208); background-color: inherit">实践中常采用替代目标函数</span>：
>
> $$\max_G \mathbb{E}_{z \sim p_z(z)}[\log D(G(z))]$$
>
> 即鼓励生成器直接最大化判别器对其输出样本的真实性评分。这一策略在训练早期能提供更强的梯度信号，显著提升收敛速度和稳定性。

* **<span style="color: rgb(36,91,219); background-color: inherit">理论分析</span>**

Goodfellow 等人在原始论文中从理论上证明了 GAN 的收敛性质：在理想条件下，即模型容量足够、优化充分，GAN 的训练过程将收敛至唯一&#x7684;**<span style="color: rgb(216,57,49); background-color: inherit">纳什均衡</span>**<span style="color: rgb(216,57,49); background-color: inherit"> Nash equilibrium</span>，此时系统达到全局最优。

**<span style="color: rgb(36,91,219); background-color: inherit">最优判别器的形式</span>**

对于任意固定的生成器$$G$$，存在一个最优判别器$$D^*(x)$$，解析表达式为：

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

该公式表明，<span style="color: rgb(100,37,208); background-color: inherit">最优判别器根据真实数据与生成数据在点</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">处的概率密度之比进行决策</span>。当$$p_{\text{data}}(x) = p_g(x)$$时，$$D^*(x) = 0.5$$，说明判别器无法做出判断。

**<span style="color: rgb(36,91,219); background-color: inherit">最优生成器的目标</span>**

当生成器成功学习到真实数据分布时，即满足：

$$p_g = p_{\text{data}}$$

此时，生成样本与真实样本在统计上完全一致，判别器对任何输入都输出$$D^*(x) = 0.5$$，意味着它已完全被欺骗，系统达到理想状态。

**<span style="color: rgb(36,91,219); background-color: inherit">GAN 与 JS 散度的联系</span>**

一个关键的理论是：<span style="color: rgb(100,37,208); background-color: inherit">当判别器处于最优状态</span>$$D^*(x)$$<span style="color: rgb(100,37,208); background-color: inherit">时，GAN 的训练目标可转化为对某种概率距离的最小化</span>。

将 $$D^*(x)$$ 代入原价值函数，并对判别器部分取最大值，得到：

$$\max_D V(D, G) = \int p_{\text{data}}(x) \log D^*(x) \, dx + \int p_g(x) \log(1 - D^*(x)) \, dx$$

化简后可得：

$$\min_G V(D^*, G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \parallel p_g)$$

其中，Jensen-Shannon 散度 **<span style="color: rgb(216,57,49); background-color: inherit">JSD</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">J</span>**<span style="color: rgb(216,57,49); background-color: inherit">ensen-</span>**<span style="color: rgb(216,57,49); background-color: inherit">S</span>**<span style="color: rgb(216,57,49); background-color: inherit">hannon </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">ivergence）</span>定义为：

$$\text{JSD}(P \parallel Q) = \frac{1}{2} \text{KL}\left(P \parallel M\right) + \frac{1}{2} \text{KL}\left(Q \parallel M\right), \quad M = \frac{1}{2}(P + Q)$$

这里 $$\text{KL}(\cdot \parallel \cdot)$$ 表示 KL 散度。



这一结果说明生成器的优化过程等价于最小化真实数据分&#x5E03;**$$p_{\text{data}}$$**&#x4E0E;生成分&#x5E03;**$$p_g$$**&#x4E4B;间的 Jensen-Shannon 散度。当两个分布完全一致时，$$\text{JSD} = 0$$，价值函数达到最小值，系统收敛至纳什均衡。

---

[Contents](../../README.md) | [Next](02-GAN--AE--Flow--GAN-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=1)
