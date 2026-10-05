[Previous](05-GAN--AE--Flow--总结.md) | [Contents](../../README.md) | [Next](07-Diffusion-Model--Stable-Diffusion-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=6)

# 4. <span style="color: rgb(36,91,219); background-color: inherit">Diffusion Model</span>

## 4.1 <span style="color: rgb(36,91,219); background-color: inherit">Diffusion 基础</span>

### 4.1.1 **<span style="color: rgb(36,91,219); background-color: inherit">DDPM</span>**

![](../../images/视觉多模态讲义（下）-image-46.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">理论基础</span>**

扩散模型是一种潜在变量模型，形式为$$p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}$$，其中，$$\mathbf{x}_1, \dots, \mathbf{x}_T$$是与数据$$\mathbf{x}_0 \sim q(\mathbf{x}_0)$$具有相同维度的潜在变量。联合分布$$p_\theta(\mathbf{x}_{0:T})$$被称为**逆过程**，被定义为一个具有学习到的高斯转移的马尔可夫链，从初始分布 $$p(\mathbf{x}_T) = \mathcal{N}(\mathbf{x}_T; \mathbf{0}, \mathbf{I})$$ 开始：

$$p_\theta(\mathbf{x}_{0:T}) := p(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_t | \mathbf{x}_{t-1})$$

$$p_\theta(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_\theta(\mathbf{x}_{t-1}, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_{t-1}, t))$$

<span style="color: rgb(100,37,208); background-color: inherit">扩散模型与其他类型的潜在变量模型的区别在于，近似后验分布</span>$$q(\mathbf{x}_{1:T} | \mathbf{x}_0)$$<span style="color: rgb(100,37,208); background-color: inherit">，称为 </span>**<span style="color: rgb(100,37,208); background-color: inherit">前向过程</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 或 </span>**<span style="color: rgb(100,37,208); background-color: inherit">扩散过程</span>**<span style="color: rgb(100,37,208); background-color: inherit">，被固定为一个马尔可夫链</span>，该链根据方差调度 $$\beta_1, \dots, \beta_T$$ 逐渐向数据添加高斯噪声：

$$q(\mathbf{x}_{1:T} | \mathbf{x}_0) := \prod_{t=1}^T q(\mathbf{x}_t | \mathbf{x}_{t-1}), \quad q(\mathbf{x}_t | \mathbf{x}_{t-1}) := \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

训练通过优化通常的变分下界（负对数似然）进行：

$$\mathbb{E}\left[-\log p_\theta(\mathbf{x}_0)\right] \leq \mathbb{E}_q\left[-\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)}\right] = \mathbb{E}_q\left[-\log p(\mathbf{x}_T) - \sum_{t \geq 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})}\right] =: L$$

<span style="color: rgb(100,37,208); background-color: inherit">前向过程中的参数</span>$$\beta_t$$<span style="color: rgb(100,37,208); background-color: inherit">可以通过重新参数化学习，也可以作为超参数保持不变</span>。逆过程的表达能力部分由$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$中高斯条件的选择保证，因为当$$\beta_t$$较小时，两个过程具有相同的函数形式。前向过程的一个显著性质是，它允许在任意时间步$$t$$以闭式采样$$\mathbf{x}_t$$：使用记号$$\alpha_t := 1 - \beta_t$$和$$\bar{\alpha}_t := \prod_{s=1}^t \alpha_s$$，因此有：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

因此，通过随机梯度下降优化$$L$$的随机项可以实现高效的训练。进一步的改进可以通过重写$$L$$来减少方差：

$$\mathbb{E}_q \left[ \frac{D_{\text{KL}}(q(\mathbf{x}_{T} | \mathbf{x}_0) \| p(\mathbf{x}_T))}{L_T} + \sum_{t > 1} \frac{D_{\text{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t))}{L_{t-1}} - \log p_\theta(\mathbf{x}_0 | \mathbf{x}_1) \right]$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：公式推导
>
> $$\begin{aligned}
> L &= \mathbb{E}_q \left[ -\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} | \mathbf{x}_0)} \right] \\
> &= \mathbb{E}_q \left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})} \right] \\
> &= \mathbb{E}_q \left[ -\log p(\mathbf{x}_T) - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_t | \mathbf{x}_{t-1})} - \log \frac{p_\theta(\mathbf{x}_0 | \mathbf{x}_1)}{q(\mathbf{x}_1 | \mathbf{x}_0)} \right] \\
> &= \mathbb{E}_q \left[ -\log \frac{p(\mathbf{x}_T)}{q(\mathbf{x}_T | \mathbf{x}_0)} - \sum_{t > 1} \log \frac{p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)}{q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0)} - \log p_\theta(\mathbf{x}_0 | \mathbf{x}_1) \right] \\
> &= \mathbb{E}_q \left[ D_{\text{KL}}(q(\mathbf{x}_T | \mathbf{x}_0) \| p(\mathbf{x}_T)) + \sum_{t > 1} D_{\text{KL}}(q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0 | \mathbf{x}_1) \right]
> \end{aligned}$$

这里<span style="color: rgb(100,37,208); background-color: inherit">使用 KL 散度直接比较</span>$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$<span style="color: rgb(100,37,208); background-color: inherit">和前向过程的后验分布，后者在条件化于</span>$$\mathbf{x}_0$$<span style="color: rgb(100,37,208); background-color: inherit">时是可处理的</span>：

$$q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_{t-1}; \boldsymbol{\tilde{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}\right)$$

其中$$\boldsymbol{\tilde{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) := \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0 + \frac{\sqrt{1 - \bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_t} \mathbf{x}_t$$，$$\tilde{\beta}_t := \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

因此，上式中的所有 KL 散度都是高斯分布之间的比较，因此它们可以通过闭式表达式的方式计算。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型结构</span>**

扩散模型看似是一类受限的隐变量模型，但它们在实现过程中允许很大的自由度。因此需要选择前向过程的方差 $$\beta_t$$以及反向过程的模型架构和高斯分布参数化形式。

**<span style="color: rgb(36,91,219); background-color: inherit">前向过程</span>**

作者没有重新参数化学习前向过程的方差$$\beta_t$$，而是将其固定为常数。因此在实现中，近似后验分布$$q$$没有可学习的参数，所以$$L_T$$在训练过程中是常数，可以忽略。

**<span style="color: rgb(36,91,219); background-color: inherit">反向过程</span>**

在反向过程中，需要在$$1 < t \leq T$$时选择$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$，其中$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$。首先将$$\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)$$设置为与时间相关的常数矩阵$$\sigma_t^2 \mathbf{I}$$。实验表明，两种选择$$\sigma_t^2 = \beta_t$$ 和 $$\sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$ 的结果相似。第一种选择对于 $$\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$ 是最优的，而第二种选择对于确定性地设置为一个点的 $$\mathbf{x}_0$$ 是最优的。这两种选择分别对应于具有坐标独立单位方差的数据在逆过程熵上的上界和下界。

其次，为了表示均值$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$$，作者提出了一种特定的参数化方法，该方法受到以下对$$L_t$$的分析的启发。对于 $$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2 \mathbf{I})$$，可以写出：

$$L_{t-1} = \mathbb{E}_q \left[ \frac{1}{2 \sigma_t^2} \| \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) - \boldsymbol{\mu}_\theta(\mathbf{x}_t, t) \|^2 \right] + C$$

其中$$C$$是一个不依赖于$$\theta$$的常数。因此可以看到$$\boldsymbol{\mu}_\theta$$的最直接参数化是一个预测$$\tilde{\boldsymbol{\mu}}_t$$的模型，即前向过程后验均值。然而可以通过重新参数化上述$$q(\mathbf{x}_t | \mathbf{x}_0)$$，令$$\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}) = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}$$，其中 $$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$，并应用前向过程进一步扩展上式：

$$\begin{aligned}
L_{t-1} - C &= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{1}{2 \sigma_t^2} \left\| \tilde{\boldsymbol{\mu}}_t \left( \mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}), \frac{1}{\sqrt{\bar{\alpha}_t}} (\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}) - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}) \right) - \boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}), t) \right\|^2 \right] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{1}{2 \sigma_t^2} \left\| \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}) - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right) - \boldsymbol{\mu}_\theta(\mathbf{x}_t(\mathbf{x}_0, \boldsymbol{\epsilon}), t) \right\|^2 \right]
\end{aligned}$$

这表明，$$\boldsymbol{\mu}_\theta$$必须根据$$\mathbf{x}_t$$预测$$\frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon} \right)$$。由于$$\mathbf{x}_t$$可以作为模型的输入，因此可以选择以下参数化：

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \tilde{\boldsymbol{\mu}}_t \left( \mathbf{x}_t, \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t) \right) \right) = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right)$$

其中是一个函数逼近器，旨在根据$$\mathbf{x}_t$$预测$$\boldsymbol{\epsilon}$$。为了从$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$$中采样$$\mathbf{x}_{t-1}$$，需要计算$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\bar{\alpha}_t}} \left( \mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sigma_t \mathbf{z}$$，其中$$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$。此外，通过参数化上式，$$L_{t-1} - C$$可以简化为：

$$\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \bar{\alpha}_t (1 - \bar{\alpha}_t)} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t) \right\|^2 \right]$$

这类似于在多个噪声尺度上进行去噪评分匹配，其中这些尺度由$$t$$索引。

![](../../images/视觉多模态讲义（下）-image-47.png)

总结来说，可以训练逆过程均值逼近器$$\boldsymbol{\mu}_\theta$$来预测$$\tilde{\boldsymbol{\mu}}_t$$，或者通过修改其参数化，可以训练它来预测 $$\boldsymbol{\epsilon}$$。当然也可以预测$$\mathbf{x}_0$$，但效果会比较差。$$\boldsymbol{\epsilon}$$-预测简化了扩散模型的变分下界，使其类似于去噪评分匹配目标。

**<span style="color: rgb(36,91,219); background-color: inherit">反向过程解码</span>**

假设图像数据由整数$$\{0, 1, \dots, 255\}$$组成，并线性缩放到区间$$[-1, 1]$$。这确保了神经网络的逆过程从标准正态先验$$p(\mathbf{x}_T)$$开始，对一致缩放后的输入进行操作。为了获得离散对数似然，将逆过程的最后一项设置为一个独立的离散解码器，该解码器由高斯分布$$\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \sigma^2 \mathbf{I})$$衍生而来：

$$p_\theta(\mathbf{x}_0 | \mathbf{x}_1) = \prod_{i=1}^D \int_{\delta_-(x_i^0)}^{\delta_+(x_i^0)} \mathcal{N}(x; \mu_\theta^i(\mathbf{x}_1, 1), \sigma^2) \, dx$$

其中

$$\delta_+(x) =
\begin{cases}
\infty & \text{if } x = 1 \\
x + \frac{1}{255} & \text{if } x < 1
\end{cases}, \quad
\delta_-(x) =
\begin{cases}
-\infty & \text{if } x = -1 \\
x - \frac{1}{255} & \text{if } x > -1
\end{cases}$$

这里，$$D$$是数据的维度，上标$$i$$表示提取一个坐标分量。类似于变分自编码&#x5668;**`VAE`**&#x7684;解码器和自回归模型中使用的离散化连续分布，确保变分下界是一个离散数据的无损码长，无需向数据添加噪声，也无需将缩放操作的雅可比行列式纳入对数似然中。在采样结束时，可以无噪声地输出$$\boldsymbol{\mu}_\theta(\mathbf{x}_1, 1)$$。

**<span style="color: rgb(36,91,219); background-color: inherit">简化的训练目标</span>**

通过上述定义的反向过程和解码器，变分下界显然可以关于$$\theta$$求导，并且可以直接用于训练。然而训练以下变种的变分下界对采样质量是有益的：

$$L_{\text{simple}}(\theta) := \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta \left( \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t \right) \right\|^2 \right]$$

其中$$t$$在$$1$$和$$T$$之间均匀分布。当$$t = 1$$时，该情况对应于$$p_\theta(\mathbf{x}_0 | \mathbf{x}_1)$$中离散解码器定义中的积分被高斯概率密度函数乘以区间宽度近似，这里忽略了$$\sigma_1^2$$和边缘效应。上图 Algorithm 1 显示了使用此简化目标的完整训练过程。

由于上述简化目标舍弃了$$\mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \left[ \frac{\beta_t^2}{2 \sigma_t^2 \bar{\alpha}_t (1 - \bar{\alpha}_t)} \left\| \boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \boldsymbol{\epsilon}, t) \right\|^2 \right]$$中的权重，它是一个加权的变分下界，强调了重建的不同方面。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

首先实现时间步嵌入的功能

然后实现 U-Net 中&#x7684;**`ResidualBlock`**&#x548C;**`AttentionBlock`**，以及上采样和下采样的过程

然后将这几个组件组装为 U-Net：

实现调度逻辑：

最后实现高斯扩散过程：

训练

### 4.1.2 **<span style="color: rgb(36,91,219); background-color: inherit">DDIM</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">背景知识</span>**

给定来自数据分布$$q(x_0)$$的样本，目标是<span style="color: rgb(100,37,208); background-color: inherit">学习一个模型分布</span>$$p_\theta(x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">，使其逼近</span>$$q(x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">并且易于采样</span>。DDPM 是隐变量模型，形式为：

$$p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}, \quad\quad p_\theta(x_{0:T}) := p_\theta(x_T) \prod_{t=1}^T p_\theta^{(t)}(x_{t-1}|x_t)$$

其中$$x_1, \dots, x_T$$是与$$x_0$$处于同一样本空间的隐变量，记为$$\mathcal{X}$$。<span style="color: rgb(100,37,208); background-color: inherit">参数</span>$$\theta$$<span style="color: rgb(100,37,208); background-color: inherit">通过最大化一个变分下界来拟合数据分布 </span>$$q(x_0)$$：

$$\max_\theta \mathbb{E}_{q(x_0)}[\log p_\theta(x_0)] \leq \max_\theta \mathbb{E}_{q(x_0,x_{1:T})}[\log p_\theta(x_{0:T}) - \log q(x_{1:T}|x_0)]$$

其中$$q(x_{1:T}|x_0)$$是隐变量上的某种推理分布。与 VAE 等典型的隐变量模型不同的是，<span style="color: rgb(100,37,208); background-color: inherit">DDPM 使用固定的推理过程 </span>$$q(x_{1:T}|x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">，而不是可训练的过程，且隐变量维度相对较高</span>。

如下是具有高斯转移的马尔可夫链，其参数由递减序列$$\alpha_{1:T} \in (0, 1]^T$$控制：

$$q(x_{1:T}|x_0) := \prod_{t=1}^T q(x_t|x_{t-1}), \quad\quad q(x_t|x_{t-1}) := \mathcal{N}\left( \sqrt{\frac{\alpha_t}{\alpha_{t-1}}} x_{t-1}, \left(1 - \frac{\alpha_t}{\alpha_{t-1}}\right) I \right)$$

其中协方差矩阵对角线元素保证为正。由于<span style="color: rgb(100,37,208); background-color: inherit">采样过程具有自回归性质</span>，即从$$x_0$$到$$x_T$$，被称为前向过程。这里<span style="color: rgb(100,37,208); background-color: inherit">将隐变量模型</span>$$p_\theta(x_{0:T})$$<span style="color: rgb(100,37,208); background-color: inherit">称为生成过程，即从</span>$$x_T$$<span style="color: rgb(100,37,208); background-color: inherit">向</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">采样的马尔可夫链</span>。因为它近似了难以计算的逆过程$$q(x_{t-1}|x_t)$$。

直观上，<span style="color: rgb(100,37,208); background-color: inherit">前向过程逐步向观测</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">添加噪声，而生成过程则逐步对含噪观测进行去噪</span>，如右图所示：

![](../../images/视觉多模态讲义（下）-image-48.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：前向过程的一个特殊性质是
>
> $$q(x_t|x_0) := \int q(x_{1:t}|x_0) \, dx_{1:(t-1)} = \mathcal{N}(x_t; \sqrt{\alpha_t} x_0, (1 - \alpha_t) I)$$

因此<span style="color: rgb(100,37,208); background-color: inherit">可以将</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">表示为</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">与噪声变量</span>$$\epsilon$$<span style="color: rgb(100,37,208); background-color: inherit">的线性组合</span>：

$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon, \quad\quad \epsilon \sim \mathcal{N}(0, I)$$

当$$\alpha_T$$足够接近 0 时，$$q(x_T|x_0)$$对所有$$x_0$$均收敛为标准高斯分布，因此设定$$p_\theta(x_T) := \mathcal{N}(0, I)$$。<span style="color: rgb(100,37,208); background-color: inherit">若所有条件分布均建模为高斯分布，则均值函数可以训练并保证方差固定</span>，因此上述优化$$\theta$$的目标可简化为：

$$L_\gamma(\epsilon_\theta) := \sum_{t=1}^T \gamma_t \mathbb{E}_{x_0 \sim q(x_0), \epsilon_t \sim \mathcal{N}(0,I)} \left[ \left\| \epsilon_\theta^{(t)}(\sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon_t) - \epsilon_t \right\|_2^2 \right]$$

其中$$\epsilon_\theta := \{ \epsilon_\theta^{(t)} \}_{t=1}^T$$是一组$$T$$个函数，每个$$\epsilon_\theta^{(t)}: \mathcal{X} \to \mathcal{X}$$是具有可训练参数$$\theta^{(t)}$$的函数，$$\gamma := [\gamma_1, \dots, \gamma_T]$$是依赖于$$\alpha_{1:T}$$的正系数向量。一些工作为最大化训练模型的生成性能，优化$$\gamma = 1$$的目标，这与基于分数匹配的噪声条件分数网络所使用的目标一致。<span style="color: rgb(100,37,208); background-color: inherit">从训练好的模型中采样</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">时，首先从先验</span>$$p_\theta(x_T)$$<span style="color: rgb(100,37,208); background-color: inherit">中采样</span>$$x_T$$<span style="color: rgb(100,37,208); background-color: inherit">，然后依次从生成过程中采样</span>$$x_{t-1}$$。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：前向过程的长度$$T$$是 DDPM 中的一个重要超参数。从变分角度看，较大的$$T$$使得逆过程更接近高斯分布，从而使用高斯条件分布建模的生成过程成为良好近似，这促使选择较大的$$T$$值，如$$T=1000$$。但是<span style="color: rgb(216,57,49); background-color: inherit">由于所有</span>$$T$$<span style="color: rgb(216,57,49); background-color: inherit">步必须顺序执行才能获得样本</span>$$x_0$$<span style="color: rgb(216,57,49); background-color: inherit">，DDPM 的采样速度远慢于其他深度生成模型，这使其在计算资源有限或延迟敏感的任务中不具实用性</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">非马尔可夫前向过程</span>**

由于生成模型近似推理过程的逆过程，需要重新看推理过程，减少生成模型所需的迭代次数。从之前描述可以看出，<span style="color: rgb(100,37,208); background-color: inherit">DDPM 目标</span>$$L_\gamma$$<span style="color: rgb(100,37,208); background-color: inherit">的形式仅依赖于边缘分布</span>$$q(x_t|x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">，而不直接依赖联合分布</span>$$q(x_{1:T}|x_0)$$。

因为存在许多具有相同边缘分布的推理分布，这里作者探索非马尔可夫的替代推理过程，从而导出新的生成过程，如右图所示。

![](../../images/视觉多模态讲义（下）-image-62.png)

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：这些非马尔可夫推理过程其实是与 DDPM 具有相同的代理目标函数。并且非马尔可夫视角也适用于高斯情形之外，如离散数据：
>
> 对于一个类别型变量$$x_0$$，其具有$$K$$个可能取值的 one-hot 向量，首先，<span style="color: rgb(100,37,208); background-color: inherit">令</span>$$q(x_t|x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">为如下类别分布</span>：
>
> $$q(x_t|x_0) = \text{Cat}(\alpha_t x_0 + (1 - \alpha_t) \mathbf{1}_K)$$
>
> 其中$$\mathbf{1}_K \in \mathbb{R}^K$$是所有元素均为$$1/K$$的向量，$$\alpha_t$$从$$t=0$$时的$$\alpha_0 = 1$$递减至$$t=T$$时的$$\alpha_T = 0$$。然后<span style="color: rgb(100,37,208); background-color: inherit">定义</span>$$q(x_{t-1}|x_t, x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">为混合分布</span>：
>
> $$q(x_{t-1}|x_t, x_0) = 
> \begin{cases}
> \text{Cat}(x_t) & \text{概率为 } \sigma_t \\
> \text{Cat}(x_0) & \text{概率为 } (\alpha_{t-1} - \sigma_t \alpha_t) \\
> \text{Cat}(\mathbf{1}_K) & \text{概率为 } (1 - \alpha_{t-1}) - (1 - \alpha_t)\sigma_t
> \end{cases}$$
>
> 或等价表示为：
>
> $$q(x_{t-1}|x_t, x_0) = \text{Cat}\left( \sigma_t x_t + (\alpha_{t-1} - \sigma_t \alpha_t) x_0 + \left((1 - \alpha_{t-1}) - (1 - \alpha_t)\sigma_t\right) \mathbf{1}_K \right)$$
>
> 这和之前$$q(x_t|x_0)$$的定义相同。
>
> 类似地，<span style="color: rgb(100,37,208); background-color: inherit">定义逆向过程 </span>$$p_\theta(x_{t-1}|x_t)$$<span style="color: rgb(100,37,208); background-color: inherit"> 为</span>：
>
> $$p_\theta(x_{t-1}|x_t) = \text{Cat}\left( \sigma_t x_t + (\alpha_{t-1} - \sigma_t \alpha_t) f_\theta^{(t)}(x_t) + \left((1 - \alpha_{t-1}) - (1 - \alpha_t)\sigma_t\right) \mathbf{1}_K \right)$$
>
> 其中$$f_\theta^{(t)}(x_t)$$将$$x_t$$映射为一个$$K$$维向量。<span style="color: rgb(100,37,208); background-color: inherit">当</span>$$(1 - \alpha_{t-1}) - (1 - \alpha_t)\sigma_t \to 0$$<span style="color: rgb(100,37,208); background-color: inherit">时，采样过程的随机性将降低，即它将以高概率选择</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">或预测的</span>$$x_0$$。
>
> $$D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t))$$是两个类别分布之间的 KL 散度。因此所得的变分目标函数也易于优化。由于 KL 散度是凸函数，因此有上界：
>
> $$D_{\text{KL}}(q(x_{t-1}|x_t, x_0) \,\|\, p_\theta(x_{t-1}|x_t)) \leq (\alpha_{t-1} - \sigma_t \alpha_t) D_{\text{KL}}(\text{Cat}(x_0) \,\|\, \text{Cat}(f_\theta^{(t)}(x_t)))$$
>
> <span style="color: rgb(100,37,208); background-color: inherit">当右侧趋近于零时该上界是紧的</span>
>
> 右侧本质上是一个多类分类损失，因此可以得出类似结论：**$$\sigma_t$$<span style="color: rgb(100,37,208); background-color: inherit">的变化不会影响目标函数</span>**。

**<span style="color: rgb(36,91,219); background-color: inherit">非马尔可夫前向过程</span>**

假设一组由实向量$$\sigma \in \mathbb{R}_{\geq 0}^T$$索引的推理分布：

$$q_\sigma(x_{1:T}|x_0) := q_\sigma(x_T|x_0) \prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)$$

其中$$q_\sigma(x_T|x_0) = \mathcal{N}(\sqrt{\alpha_T} x_0, (1 - \alpha_T) I)$$，且对所有$$t > 1$$，

$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left( \sqrt{\alpha_{t-1}} x_0 + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1 - \alpha_t}}, \sigma_t^2 I \right)$$

均值函数的选择保证了对所有$$t$$，有$$q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t} x_0, (1 - \alpha_t) I)$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：证明
>
> 假设对任意$$t \leq T$$，$$q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t} x_0, (1 - \alpha_t) I)$$成立，若：
>
> $$q_\sigma(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}} x_0, (1 - \alpha_{t-1}) I)$$
>
> 则可对$$t$$从$$T$$到$$1$$进行归纳证明，因为基础情形$$t = T$$已成立。
>
> 首先，有：
>
> $$q_\sigma(x_{t-1}|x_0) := \int_{x_t} q_\sigma(x_t|x_0) q_\sigma(x_{t-1}|x_t, x_0) \, dx_t$$
>
> 且
>
> $$q_\sigma(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t} x_0, (1 - \alpha_t) I) $$
>
> $$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left( \sqrt{\alpha_{t-1}} x_0 + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1 - \alpha_t}}, \sigma_t^2 I \right)$$
>
> 由前人的工作可知$$q_\sigma(x_{t-1}|x_0)$$为高斯分布，这里记作$$\mathcal{N}(\mu_{t-1}, \Sigma_{t-1})$$，其中
>
> $$\begin{aligned}
> \mu_{t-1} &= \sqrt{\alpha_{t-1}} x_0 + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \frac{\sqrt{\alpha_t} x_0 - \sqrt{\alpha_t} x_0}{\sqrt{1 - \alpha_t}} \\
> &= \sqrt{\alpha_{t-1}} x_0
> \end{aligned}$$
>
> 且
>
> $$\Sigma_{t-1} = \sigma_t^2 I + \frac{1 - \alpha_{t-1} - \sigma_t^2}{1 - \alpha_t} (1 - \alpha_t) I = (1 - \alpha_{t-1}) I $$
>
> 因此，
>
> $$q_\sigma(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\alpha_{t-1}} x_0, (1 - \alpha_{t-1}) I)$$

从而定义了一个符合所需边缘分布的联合推理分布。前向过程可通过贝叶斯规则推导：

$$q_\sigma(x_t|x_{t-1}, x_0) = \frac{q_\sigma(x_{t-1}|x_t, x_0) q_\sigma(x_t|x_0)}{q_\sigma(x_{t-1}|x_0)}$$

这个分布也是高斯的，但前向过程不再是马尔可夫的，因为每个$$x_t$$可能同时依赖于$$x_{t-1}$$和$$x_0$$。$$\sigma$$<span style="color: rgb(100,37,208); background-color: inherit">的大小控制前向过程的随机性程度，当</span>$$\sigma \to 0$$<span style="color: rgb(100,37,208); background-color: inherit">时，达到极端情形：只要由</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">和某个</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">对应的</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">，则</span>$$x_{t-1}$$<span style="color: rgb(100,37,208); background-color: inherit">即被确定</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">统一的变分推理目标</span>**

定义一个可训练的生成过程$$p_\theta(x_{0:T})$$，其中每个$$p_\theta^{(t)}(x_{t-1}|x_t)$$利用了对$$q_\sigma(x_{t-1}|x_t, x_0)$$的认知。直观上，<span style="color: rgb(100,37,208); background-color: inherit">给定含噪观测</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">，首先预测对应的</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">，然后通过已定义的逆条件分布</span>$$q_\sigma(x_{t-1}|x_t, x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">采样</span>$$x_{t-1}$$。

对于某些$$x_0 \sim q(x_0)$$和$$\epsilon_t \sim \mathcal{N}(0, I)$$，可通过$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon$$得到$$x_t$$。模型$$\epsilon_\theta^{(t)}(x_t)$$在无$$x_0$$信息的情况下仅根据$$x_t$$预测$$\epsilon_t$$。重写$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon$$即可预测去噪后的观测值，也就是给定$$x_t$$对$$x_0$$的预测：

$$f_\theta^{(t)}(x_t) := \left( x_t - \sqrt{1 - \alpha_t} \cdot \epsilon_\theta^{(t)}(x_t) \right) / \sqrt{\alpha_t}$$

然后定义生成过程：固定先验$$p_\theta(x_T) = \mathcal{N}(0, I)$$，且

$$p_\theta^{(t)}(x_{t-1}|x_t) = 
\begin{cases}
\mathcal{N}(f_\theta^{(1)}(x_1), \sigma_1^2 I) & \text{if } t = 1 \\
q_\sigma(x_{t-1}|x_t, f_\theta^{(t)}(x_t)) & \text{otherwise}
\end{cases}$$

其中 $$q_\sigma(x_{t-1}|x_t, f_\theta^{(t)}(x_t))$$ 的定义与$$q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left( \sqrt{\alpha_{t-1}} x_0 + \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \frac{x_t - \sqrt{\alpha_t} x_0}{\sqrt{1 - \alpha_t}}, \sigma_t^2 I \right)$$相同，仅将$$x_0$$替换为$$f_\theta^{(t)}(x_t)$$。<span style="color: rgb(100,37,208); background-color: inherit">为确保生成过程在整个空间有支撑，在</span>$$t=1$$<span style="color: rgb(100,37,208); background-color: inherit">时额外添加高斯噪声</span>，其协方差为$$\sigma_1^2 I$$。

然后通过以下变分推理目标优化$$\theta$$，这个目标是$$\epsilon_\theta$$的泛函：

$$J_\sigma(\epsilon_\theta) := \mathbb{E}_{x_{0:T} \sim q_\sigma(x_{0:T})} [\log q_\sigma(x_{1:T}|x_0) - \log p_\theta(x_{0:T})]$$



$$= \mathbb{E}_{x_{0:T} \sim q_\sigma(x_{0:T})} \left[ \log q_\sigma(x_T|x_0) + \sum_{t=2}^T \log q_\sigma(x_{t-1}|x_t, x_0) - \sum_{t=1}^T \log p_\theta^{(t)}(x_{t-1}|x_t) - \log p_\theta(x_T) \right]$$

这里<span style="color: rgb(100,37,208); background-color: inherit">根据</span>$$q_\sigma(x_{1:T}|x_0) := q_\sigma(x_T|x_0) \prod_{t=2}^T q_\sigma(x_{t-1}|x_t, x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">分解</span>$$q_\sigma(x_{1:T}|x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">，根据</span>$$p_\theta(x_0) = \int p_\theta(x_{0:T}) \, dx_{1:T}$$<span style="color: rgb(100,37,208); background-color: inherit">分解</span>$$p_\theta(x_{0:T})$$。

从$$J_\sigma$$的定义看，<span style="color: rgb(216,57,49); background-color: inherit">对每个</span>$$\sigma$$<span style="color: rgb(216,57,49); background-color: inherit">的选择都需要训练不同的模型，因为其对应不同的变分目标</span>。但其实$$J_\sigma$$等价于某个权重$$\gamma$$下的$$L_\gamma$$，因为：

> **<span style="color: rgb(222,120,2); background-color: inherit">定理</span>**：对任意$$\sigma > 0$$，存在$$\gamma \in \mathbb{R}_{>0}^T$$和$$C \in \mathbb{R}$$，使得$$J_\sigma = L_\gamma + C$$

目标$$L_\gamma$$的特殊之处在于：<span style="color: rgb(100,37,208); background-color: inherit">若模型</span>$$\epsilon_\theta^{(t)}$$<span style="color: rgb(100,37,208); background-color: inherit">的参数</span>$$\theta$$<span style="color: rgb(100,37,208); background-color: inherit">在不同</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">间不共享，则</span>$$\epsilon_\theta$$<span style="color: rgb(100,37,208); background-color: inherit">的最优解不依赖于权重</span>$$\gamma$$<span style="color: rgb(100,37,208); background-color: inherit">，因为全局最优可通过独立最大化求和中的每一项实现</span>。这一性质具有双重含义：

> 证明了<span style="color: rgb(100,37,208); background-color: inherit">在 DDPM 中使用</span>$$L_1$$<span style="color: rgb(100,37,208); background-color: inherit">作为变分下界的代理目标是合理的</span>
>
> 上述定理表明$$J_\sigma$$等价于某个$$L_\gamma$$，故$$J_\sigma$$<span style="color: rgb(100,37,208); background-color: inherit">的最优解也与</span>$$L_1$$<span style="color: rgb(100,37,208); background-color: inherit">相同</span>

因此若模型$$\epsilon_\theta$$中不同$$t$$的参数不共享，则使用$$L_1$$目标也可作为变分目标$$J_\sigma$$的代理目标。

* **<span style="color: rgb(36,91,219); background-color: inherit">从广义生成过程中采样</span>**

以$$L_1$$为目标函数，如上所述，本质上可以将预训练好的 DDPM 模型直接作为这些新目标的解，只需通过调整$$\sigma$$来寻找更符合需求的生成过程，而无需重新训练模型。

**<span style="color: rgb(36,91,219); background-color: inherit">去噪扩散隐式模型 </span> <span style="color: rgb(216,57,49); background-color: inherit">DDIM（Denoising Diffusion Implicit Model）</span>**

由$$p_\theta^{(t)}(x_{t-1}|x_t) = 
\begin{cases}
\mathcal{N}(f_\theta^{(1)}(x_1), \sigma_1^2 I) & \text{if } t = 1 \\
q_\sigma(x_{t-1}|x_t, f_\theta^{(t)}(x_t)) & \text{otherwise}
\end{cases}$$中的$$p_\theta(x_{1:T})$$，可以从样本$$x_t$$生成样本$$x_{t-1}$$，更新公式如下：

$$x_{t-1} = \underbrace{ \sqrt{\alpha_{t-1}} \left( \frac{x_t - \sqrt{1 - \alpha_t} \epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}} \right) }_{\text{“预测的 } x_0\text{”}} + \underbrace{ \sqrt{1 - \alpha_{t-1} - \sigma_t^2} \cdot \epsilon_\theta^{(t)}(x_t) }_{\text{“指向 } x_t \text{ 的方向”}} + \underbrace{ \sigma_t \epsilon_t }_{\text{随机噪声}} $$

其中$$\epsilon_t \sim \mathcal{N}(0, I)$$是与$$x_t$$独立的标准高斯噪声，这里定义$$\alpha_0 := 1$$。<span style="color: rgb(100,37,208); background-color: inherit">不同的</span>$$\sigma$$<span style="color: rgb(100,37,208); background-color: inherit">取值对应不同的生成过程，但均可使用同一个模型</span>$$\epsilon_\theta$$<span style="color: rgb(100,37,208); background-color: inherit">，因此无需重新训练模型。当对所有</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">取</span>$$\sigma_t = \sqrt{(1 - \alpha_{t-1})/(1 - \alpha_t)} \cdot \sqrt{1 - \alpha_t / \alpha_{t-1}}$$<span style="color: rgb(100,37,208); background-color: inherit">时，前向过程恢复为马尔可夫过程，生成过程则退化为标准 DDPM</span>。

<span style="color: rgb(100,37,208); background-color: inherit">当对所有</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">取</span>$$\sigma_t = 0$$<span style="color: rgb(100,37,208); background-color: inherit">时，前向过程在给定</span>$$x_{t-1}$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">的条件下变为确定性过程</span>，这里$$t=1$$除外。在生成过程中，随机噪声$$\epsilon_t$$前的系数变为零。此时模型成为一个隐式概率模型 Implicit Probabilistic Model，即样本通过一个固定过程从隐变量生成，即从$$x_T$$到$$x_0$$。作者将其命名为<span style="color: rgb(216,57,49); background-color: inherit">去噪扩散隐式模型 </span>**<span style="color: rgb(216,57,49); background-color: inherit">DDIM</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">enoising </span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">iffusion </span>**<span style="color: rgb(216,57,49); background-color: inherit">I</span>**<span style="color: rgb(216,57,49); background-color: inherit">mplicit </span>**<span style="color: rgb(216,57,49); background-color: inherit">M</span>**<span style="color: rgb(216,57,49); background-color: inherit">odel）</span>，因为它是使用 DDPM 目标训练的隐式概率模型（前向过程不再是扩散过程）。

**<span style="color: rgb(36,91,219); background-color: inherit">加速生成</span>**

之前<span style="color: rgb(216,57,49); background-color: inherit">将生成过程视为对逆过程的近似，由于前向过程有</span>$$T$$<span style="color: rgb(216,57,49); background-color: inherit">步，生成过程也被迫执行</span>$$T$$<span style="color: rgb(216,57,49); background-color: inherit">步采样</span>。但由于去噪目标$$L_1$$不依赖于具体的前向过程，<span style="color: rgb(46,161,33); background-color: inherit">只要</span>$$q_\sigma(x_t|x_0)$$<span style="color: rgb(46,161,33); background-color: inherit">保持不变，因此可以考虑长度小于</span>$$T$$<span style="color: rgb(46,161,33); background-color: inherit">的前向过程，从而在不重新训练模型的前提下加速生成过程</span>。

这里不再定义前向过程在全部隐变量$$x_{1:T}$$上，而是定义在一个子集$$\{x_{\tau_1}, \dots, x_{\tau_S}\}$$上，其中$$\tau$$是$$[1, \dots, T]$$的一个长度为$$S$$的递增子序列。

这里要<span style="color: rgb(100,37,208); background-color: inherit">特别定义</span>$$x_{\tau_1}, \dots, x_{\tau_S}$$<span style="color: rgb(100,37,208); background-color: inherit">上的顺序前向过程，使得</span>$$q(x_{\tau_i}|x_0) = \mathcal{N}(\sqrt{\alpha_{\tau_i}} x_0, (1 - \alpha_{\tau_i}) I)$$<span style="color: rgb(100,37,208); background-color: inherit">保持与原始边缘分布一致</span>，如右图。

![](../../images/视觉多模态讲义（下）-image-64.png)

生成过程现在根据$$\text{reversed}(\tau)$$对隐变量进行采样，称此为采样轨迹。当采样轨迹长度远小于$$T$$时，<span style="color: rgb(46,161,33); background-color: inherit">由于采样过程是迭代的，计算效率将获得显著提升</span>。由上面得到的结论，可以直接使用通过$$L_1$$目标训练的模型，训练过程无需任何改动。<span style="color: rgb(100,37,208); background-color: inherit">只需对上述 DDIM 公式中的更新步骤稍作调整，即可获得新的、更快的生成过程，这适用于 DDPM、DDIM 以及</span>$$p_\theta^{(t)}(x_{t-1}|x_t) = 
\begin{cases}
\mathcal{N}(f_\theta^{(1)}(x_1), \sigma_1^2 I) & \text{if } t = 1 \\
q_\sigma(x_{t-1}|x_t, f_\theta^{(t)}(x_t)) & \text{otherwise}
\end{cases}$$<span style="color: rgb(100,37,208); background-color: inherit">中所有考虑的生成过程</span>。这意味着可以用任意多步的前向过程训练模型，但在生成过程中仅采样其中部分步骤。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：在加速情形下，可以将推理过程分解为：
>
> $$q_{\sigma,\tau}(x_{1:T}|x_0) = q_{\sigma,\tau}(x_{\tau_S}|x_0) \prod_{i=1}^S q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, x_0) \prod_{t \in \bar{\tau}} q_{\sigma,\tau}(x_t|x_0) $$
>
> 其中$$\tau$$是$$[1, \dots, T]$$的一个长度为$$S$$的子序列，且满足$$\tau_S = T$$，<span style="color: rgb(100,37,208); background-color: inherit">令</span>$$\bar{\tau} := \{1, \dots, T\} \setminus \tau$$<span style="color: rgb(100,37,208); background-color: inherit">为其补集。</span>$$\{x_{\tau_i}\}_{i=1}^S$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">构成一个链式图模型，而</span>$$\{x_t\}_{t \in \bar{\tau}}$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">构成一个星型图模型</span>。这里定义：
>
> $$q_{\sigma,\tau}(x_t|x_0) = \mathcal{N}(\sqrt{\alpha_t} x_0, (1 - \alpha_t) I) \quad \forall t \in \bar{\tau} \cup \{T\} $$
>
> $$q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, x_0) = \mathcal{N}\left( \sqrt{\alpha_{\tau_{i-1}}} x_0 + \sqrt{1 - \alpha_{\tau_{i-1}} - \sigma_{\tau_i}^2} \cdot \frac{x_{\tau_i} - \sqrt{\alpha_{\tau_i}} x_0}{\sqrt{1 - \alpha_{\tau_i}}}, \sigma_{\tau_i}^2 I \right) \quad \forall i \in [S]$$
>
> 系数的选择满足：
>
> $$q_{\sigma,\tau}(x_{\tau_i}|x_0) = \mathcal{N}(\sqrt{\alpha_{\tau_i}} x_0, (1 - \alpha_{\tau_i}) I) \quad \forall i \in [S]$$
>
> 即<span style="color: rgb(100,37,208); background-color: inherit">保持边缘分布一致</span>。对应的生成过程定义为：
>
> $$p_\theta(x_{0:T}) := p_\theta(x_T) \prod_{i=1}^S \underbrace{p_\theta^{(\tau_i)}(x_{\tau_{i-1}}|x_{\tau_i})}_{\text{用于生成样本}} \times \prod_{t \in \bar{\tau}} \underbrace{p_\theta^{(t)}(x_0|x_t)}_{\text{用于变分目标}}$$
>
> 其中仅部分模型实际用于生成样本。各条件分布为：
>
> $$p_\theta^{(\tau_i)}(x_{\tau_{i-1}}|x_{\tau_i}) = q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, f_\theta^{(\tau_i)}(x_{\tau_i})) \quad \text{if } i \in [S], i > 1$$
>
> $$p_\theta^{(t)}(x_0|x_t) = \mathcal{N}(f_\theta^{(t)}(x_t), \sigma_t^2 I) \quad \text{otherwise}$$
>
> 这里利用了推理过程中的$$q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, x_0)$$。最终的变分目标函数为：
>
> $$J(\epsilon_\theta) = \mathbb{E}_{x_{0:T} \sim q_{\sigma,\tau}(x_{0:T})} [\log q_{\sigma,\tau}(x_{1:T}|x_0) - \log p_\theta(x_{0:T})]$$
>
> $$= \mathbb{E}_{x_{0:T} \sim q_{\sigma,\tau}(x_{0:T})} \left[ \sum_{t \in \bar{\tau}} D_{\text{KL}}(q_{\sigma,\tau}(x_t|x_0) \,\|\, p_\theta^{(t)}(x_0|x_t)) + \sum_{i=1}^L D_{\text{KL}}(q_{\sigma,\tau}(x_{\tau_{i-1}}|x_{\tau_i}, x_0) \,\|\, p_\theta^{(\tau_i)}(x_{\tau_{i-1}}|x_{\tau_i})) \right]$$
>
> 这里为简洁起见，定义$$x_{\tau_{L+1}} = \emptyset$$。其中每个 KL 散度均在两个方差与$$\theta$$无关的高斯分布之间计算。可证明该变分目标$$J$$可转化为形如$$L_\gamma$$的目标函数。

**<span style="color: rgb(36,91,219); background-color: inherit">Neural ODE</span>**

此外，可以根据 DDIM 公式重写 DDIM 的迭代步骤，其与求解常微分方程 ODE 的欧拉积分形式的相似性变得更加明显：

$$\frac{x_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}} = \frac{x_t}{\sqrt{\alpha_t}} + \left( \sqrt{\frac{1 - \alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}} - \sqrt{\frac{1 - \alpha_t}{\alpha_t}} \right) \cdot \epsilon_\theta^{(t)}(x_t)$$

为推导对应的 ODE，可以<span style="color: rgb(100,37,208); background-color: inherit">用</span>$$\sigma$$<span style="color: rgb(100,37,208); background-color: inherit">重参数化</span>$$\sqrt{1 - \alpha}/\sqrt{\alpha}$$<span style="color: rgb(100,37,208); background-color: inherit">，用</span>$$\bar{x}$$<span style="color: rgb(100,37,208); background-color: inherit">重参数化</span>$$x/\sqrt{\alpha}$$。在连续情形下，$$\sigma$$和$$\bar{x}$$均为$$t$$的函数，其中$$\sigma: \mathbb{R}_{\geq 0} \to \mathbb{R}_{\geq 0}$$是连续递增函数，且满足$$\sigma(0) = 0$$。上式可视为 ODE 的欧拉方法近似：

$$d\bar{x}(t) = \epsilon_\theta^{(t)}\left( \bar{x}(t) \sqrt{\sigma^2(t) + 1} \right) d\sigma(t)$$

初始条件为$$x(T) \sim \mathcal{N}(0, \sigma(T))$$，其中$$\sigma(T)$$非常大，对应$$\alpha \approx 0$$的情形。这表明<span style="color: rgb(100,37,208); background-color: inherit">若离散化步数足够多，可逆向执行生成过程，从</span>$$t=0$$<span style="color: rgb(100,37,208); background-color: inherit">到</span>$$T$$<span style="color: rgb(100,37,208); background-color: inherit">，将</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">编码为</span>$$x_T$$，并模拟上式中 ODE 的逆过程。这<span style="color: rgb(46,161,33); background-color: inherit">与 DDPM 不同，DDIM 可用于获取观测数据的编码，形式为</span>$$x_T$$<span style="color: rgb(46,161,33); background-color: inherit">，这对需要模型隐表示的下游任务非常有用</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

这里&#x4EE5;**`cifar10`**&#x6570;据集为例，首先构&#x9020;**`dataset`**&#x7C7B;

这里和 **DDPM&#x20;**&#x4E00;样的，实现一个包&#x542B;**`ResNet Block`**&#x548C;**`Attention Block`**&#x7684;**`U-Net`**

1. **<span style="color: rgb(36,91,219); background-color: inherit">首先实现时间步的编码以及激活函数和归一化</span>**：

2. **<span style="color: rgb(36,91,219); background-color: inherit">实现上采样和下采样的过程</span>**：

3. **<span style="color: rgb(36,91,219); background-color: inherit">实现 Resnet Block</span>**：

4. **<span style="color: rgb(36,91,219); background-color: inherit">实现 Attention Block</span>**：

5. **<span style="color: rgb(36,91,219); background-color: inherit">完整的 UNet 模型</span>**：

用于扩散模型DDIM中预测噪声$$\epsilon$$。包&#x62EC;**`time embedding`**、下采&#x6837;**`down`**、中间 **`bottleneck`**、上采&#x6837;**`up`**&#x4E09;部分 + **`skip connections`** + **`attention`**。

最后是训练部分，要实现优化&#x5668;**`optimizer`**&#x548C;**`schedule`**

然后构造训练类：

### 4.1.3 <span style="color: rgb(36,91,219); background-color: inherit">Flow Matching</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">连续归一化流 Continuous Normalizing Flow</span>**

设$$\mathbb{R}^d$$表示数据空间，其中数据点为$$x = (x_1, \dots, x_d) \in \mathbb{R}^d$$这里介绍两个重要概念：

> * **<span style="color: rgb(36,91,219); background-color: inherit">概率密度路径 probability density path</span>** $$p : [0, 1] \times \mathbb{R}^d \to \mathbb{R}_{>0}$$，它是一个时间依赖的概率密度函数，即$$\int p_t(x)\,dx = 1$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">时间依赖的向量场 time-dependent vector field</span>** $$v : [0, 1] \times \mathbb{R}^d \to \mathbb{R}^d$$

向量场$$v_t$$可用于构造一个时间依赖的微分同胚映射，称为流 Flow，记作$$\phi : [0, 1] \times \mathbb{R}^d \to \mathbb{R}^d$$，由如下常微分方程 ODE 定义：

$$\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))$$

$$\phi_0(x) = x$$

之前一些工作使用神经网络$$v_t(x; \theta)$$对向量场$$v_t$$建模，其中$$\theta \in \mathbb{R}^p$$<span style="color: rgb(100,37,208); background-color: inherit">为可学习参数，从而得到流</span>$$\phi_t$$<span style="color: rgb(100,37,208); background-color: inherit">的深度参数化模型，称为</span>**<span style="color: rgb(216,57,49); background-color: inherit">连续归一化流 CNF</span>**。CNF 用于将简单的先验密度$$p_0$$（例如纯噪声）通过前推方程重塑为更复杂的密度 $$p_1$$：

$$p_t = [\phi_t]_* p_0$$

其中前推算子$$*$$定义为：

$$[\phi_t]_* p_0(x) = p_0(\phi_t^{-1}(x)) \det\left( \frac{\partial \phi_t^{-1}}{\partial x}(x) \right) $$

若向量场$$v_t$$的流$$\phi_t$$满足方程$$p_t = [\phi_t]_* p_0$$，则称该向量场生成概率密度路径$$p_t$$。检验一个向量场是否生成概率路径的一种实用方法是使用连续性方程。

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：<span style="color: rgb(100,37,208); background-color: inherit">检验一个向量场</span>$$v_t$$<span style="color: rgb(100,37,208); background-color: inherit">是否生成概率路径</span>$$p_t$$<span style="color: rgb(100,37,208); background-color: inherit">的一种方法是使用连续性方程</span>。它是一个偏微分方程 PDE，提供了向量场$$v_t$$生成$$p_t$$的充要条件：
>
> $$\frac{d}{dt} p_t(x) + \text{div}(p_t(x) v_t(x)) = 0$$
>
> 其中散度算子$$\text{div}$$是关于空间变量$$x = (x_1, \dots, x_d)$$定义的，即$$\text{div} = \sum_{i=1}^d \frac{\partial}{\partial x_i}$$

* **<span style="color: rgb(36,91,219); background-color: inherit">流匹配 Flow Matching</span>**

设$$x_1$$为服从某个未知数据分布$$q(x_1)$$的随机变量。假设仅能访问来自$$q(x_1)$$的数据样本，而无法访问其密度函数本身。设$$p_t$$为概率路径，满足$$p_0 = p$$为简单分布，例如<span style="color: rgb(220,155,4); background-color: inherit">标准正态分布</span>$$p(x) = \mathcal{N}(x|0, I)$$，且$$p_1$$在分布上近似等于$$q$$。<span style="color: rgb(100,37,208); background-color: inherit">流匹配目标旨在匹配这一目标概率路径，从而实现从</span>$$p_0$$<span style="color: rgb(100,37,208); background-color: inherit">到</span>$$p_1$$<span style="color: rgb(100,37,208); background-color: inherit">的流动</span>。

给定目标概率密度路径$$p_t(x)$$及其对应的生成该路径的向量场$$u_t(x)$$，定义流匹配 FM 目标为：

$$L_{\text{FM}}(\theta) = \mathbb{E}_{t, p_t(x)} \left[ \| v_t(x) - u_t(x) \|^2 \right]$$

其中$$\theta$$表示 CNF 向量场$$v_t$$的可学习参数，$$t \sim \mathcal{U}[0, 1]$$为均匀分布，且$$x \sim p_t(x)$$。也就是说 <span style="color: rgb(100,37,208); background-color: inherit">FM 损失使用神经网络</span>$$v_t$$<span style="color: rgb(100,37,208); background-color: inherit">回归向量场</span>$$u_t$$<span style="color: rgb(100,37,208); background-color: inherit">。当损失为零时，所学 CNF 模型将生成</span>$$p_t(x)$$。

流匹配看起来很完美，但在实践中直接使用是不行的，因为事先不知道合适的$$p_t$$和$$u_t$$是什么。存在许多满足$$p_1(x) \approx q(x)$$的概率路径选择，更重要的是无法获得生成所需$$p_t$$的闭式$$u_t$$。

1. **<span style="color: rgb(36,91,219); background-color: inherit">从条件概率路径与向量场构造</span>$$p_t$$<span style="color: rgb(36,91,219); background-color: inherit">与</span>$$u_t$$**

构造目标概率路径的一种简单方法是通过更简单概率路径的混合：<span style="color: rgb(100,37,208); background-color: inherit">对于特定数据样本</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">，记</span>$$p_t(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">为条件概率路径，满足在</span>$$t=0$$<span style="color: rgb(100,37,208); background-color: inherit">时</span>$$p_0(x|x_1) = p(x)$$<span style="color: rgb(100,37,208); background-color: inherit">，并在</span>$$t=1$$<span style="color: rgb(100,37,208); background-color: inherit">时设计</span>$$p_1(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">为集中在</span>$$x=x_1$$<span style="color: rgb(100,37,208); background-color: inherit">附近的分布</span>，例如 $$p_1(x|x_1) = \mathcal{N}(x|x_1, \sigma^2 I)$$<span style="color: rgb(220,155,4); background-color: inherit">，即均值为</span>$$x_1$$<span style="color: rgb(220,155,4); background-color: inherit">、标准差</span>$$\sigma > 0$$<span style="color: rgb(220,155,4); background-color: inherit">足够小的正态分布</span>。对条件概率路径关于$$q(x_1)$$边缘化，得到边缘概率路径：

$$p_t(x) = \int p_t(x|x_1) q(x_1)\, dx_1$$

特别地，<span style="color: rgb(100,37,208); background-color: inherit">在</span>$$t=1$$<span style="color: rgb(100,37,208); background-color: inherit">时，边缘概率</span>$$p_1$$<span style="color: rgb(100,37,208); background-color: inherit">是一个混合分布，近似于数据分布</span>$$q$$：

$$p_1(x) = \int p_1(x|x_1) q(x_1)\, dx_1 \approx q(x)$$

当然也可以<span style="color: rgb(100,37,208); background-color: inherit">边缘化条件向量场</span>，定义边缘向量场，假设对所有$$t$$和$$x$$有$$p_t(x) > 0$$：

$$u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1) q(x_1)}{p_t(x)}\, dx_1$$

其中$$u_t(\cdot|x_1) : \mathbb{R}^d \to \mathbb{R}^d$$是生成条件概率路径$$p_t(\cdot|x_1)$$的条件向量场。<span style="color: rgb(100,37,208); background-color: inherit">以这种方式聚合条件向量场实际上能得到建模边缘概率路径的正确向量场</span>。

> **<span style="color: rgb(36,91,219); background-color: inherit">结论 1</span>**：边缘向量场$$u_t(x)$$可以生成边缘概率路径$$p_t(x)$$

这<span style="color: rgb(100,37,208); background-color: inherit">在条件向量场（生成条件概率路径）与边缘向量场（生成边缘概率路径）之间建立了联系</span>，能够将未知且难以处理的边缘向量场分解为更简单的条件向量场，后者仅依赖于单个数据样本，因而更容易定义。

> **<span style="color: rgb(36,91,219); background-color: inherit">定理 1</span>**：给定生成条件概率路径$$p_t(x|x_1)$$的向量场$$u_t(x|x_1)$$，对任意分布$$q(x_1)$$，方&#x7A0B;**$$u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1) q(x_1)}{p_t(x)}\, dx_1$$**&#x4E2D;的边缘向量场$$u_t$$生成方&#x7A0B;**$$p_t(x) = \int p_t(x|x_1) q(x_1)\, dx_1$$**&#x4E2D;的边缘概率路径$$p_t$$，即$$u_t$$与$$p_t$$满足连续性方程$$\frac{d}{dt} p_t(x) + \text{div}(p_t(x) v_t(x)) = 0$$。

* **<span style="color: rgb(36,91,219); background-color: inherit">条件流匹配 Conditional Flow Matching</span>**

由于<span style="color: rgb(216,57,49); background-color: inherit">边缘概率路径和向量场定义中存在难以处理的积分，即</span>**$$u_t(x) = \int u_t(x|x_1) \frac{p_t(x|x_1) q(x_1)}{p_t(x)}\, dx_1$$**<span style="color: rgb(216,57,49); background-color: inherit">和</span>**$$p_t(x) = \int p_t(x|x_1) q(x_1)\, dx_1$$**<span style="color: rgb(216,57,49); background-color: inherit">，直接计算</span>$$u_t$$<span style="color: rgb(216,57,49); background-color: inherit">仍不可行，进而无法直接计算原始流匹配目标的无偏估计</span>。因此作者提出一个更简单的替代目标，它与原始目标具有相同的最优解。

具体来说，条件流匹配 CFM 目标：

$$L_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p_t(x|x_1)} \left[ \| v_t(x) - u_t(x|x_1) \|^2 \right]$$

其中$$t \sim \mathcal{U}[0, 1]$$，$$x_1 \sim q(x_1)$$，且现在$$x \sim p_t(x|x_1)$$。与 FM 目标不同，<span style="color: rgb(100,37,208); background-color: inherit">只要能高效地从</span>$$p_t(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">采样并计算</span>$$u_t(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">，CFM 目标就能轻松采样无偏估计</span>，而这二者均可轻松实现，因为它们是基于单样本定义的。

> **<span style="color: rgb(36,91,219); background-color: inherit">结论 2</span>**：流匹配目标$$L_{\text{FM}}(\theta)$$与条件流匹配目标$$L_{\text{CFM}}(\theta)$$关于$$\theta$$具有相同的梯度

这也就是说<span style="color: rgb(100,37,208); background-color: inherit">优化 CFM 目标等价于优化 FM 目标。因此可以在无需访问边缘概率路径或边缘向量场的情况下，训练 CNF 以生成边缘概率路径</span>$$p_t$$，在$$t=1$$时近似未知数据分布$$q$$。这里只需设计合适的条件概率路径与向量场：

> **<span style="color: rgb(36,91,219); background-color: inherit">定理 2</span>**：假设对所有$$x \in \mathbb{R}^d$$和$$t \in [0, 1]$$有$$p_t(x) > 0$$，则$$L_{\text{CFM}}$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$L_{\text{FM}}$$<span style="color: rgb(100,37,208); background-color: inherit">在相差一个与</span>$$\theta$$<span style="color: rgb(100,37,208); background-color: inherit">无关的常数下相等</span>。因此$$\nabla_\theta L_{\text{FM}}(\theta) = \nabla_\theta L_{\text{CFM}}(\theta)$$

* **<span style="color: rgb(36,91,219); background-color: inherit">条件概率路径与向量场</span>**

条件流匹配目标<span style="color: rgb(100,37,208); background-color: inherit">适用于任意选择的条件概率路径和条件向量场</span>。假设有如下形式的条件概率路径：

$$p_t(x|x_1) = \mathcal{N}(x \mid \mu_t(x_1), \sigma_t(x_1)^2 I)$$

其中$$\mu : [0, 1] \times \mathbb{R}^d \to \mathbb{R}^d$$是高斯分布的时间依赖均值，$$\sigma : [0, 1] \times \mathbb{R} \to \mathbb{R}_{>0}$$是时间依赖的标量标准差。<span style="color: rgb(100,37,208); background-color: inherit">设</span>$$\mu_0(x_1) = 0$$<span style="color: rgb(100,37,208); background-color: inherit">、</span>$$\sigma_0(x_1) = 1$$<span style="color: rgb(100,37,208); background-color: inherit">，使得所有条件概率路径在</span>$$t=0$$<span style="color: rgb(100,37,208); background-color: inherit">时均收敛到相同的标准化高斯噪声分布</span>$$p(x) = \mathcal{N}(x|0, I)$$。在$$t=1$$时，<span style="color: rgb(100,37,208); background-color: inherit">设</span>$$\mu_1(x_1) = x_1$$<span style="color: rgb(100,37,208); background-color: inherit">、</span>$$\sigma_1(x_1) = \sigma_{\min}$$<span style="color: rgb(100,37,208); background-color: inherit">，使得</span>$$p_1(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">为以</span>$$x_1$$<span style="color: rgb(100,37,208); background-color: inherit">为中心的集中高斯分布</span>。

对于任意特定概率路径，存在无穷多个生成它的向量场，例如<span style="color: rgb(220,155,4); background-color: inherit">通过在连续性方程中添加无散度分量</span>，但绝大多数向量场包含不改变底层分布的成分，例如<span style="color: rgb(220,155,4); background-color: inherit">当分布具有旋转不变性时的旋转分量，导致不必要的额外计算</span>。这里选择使用对应于高斯分布规范变换的最简单向量场。具体而言，以$$x_1$$为条件的流：

$$\psi_t(x) = \sigma_t(x_1) x + \mu_t(x_1)$$

当$$x$$服从标准高斯分布时，$$\psi_t(x)$$是将$$x$$映射到均值为$$\mu_t(x_1)$$、标准差为$$\sigma_t(x_1)$$的正态分布随机变量的仿射变换。也就是说，<span style="color: rgb(100,37,208); background-color: inherit">根据方程</span>$$[\phi_t]_* p_0(x) = p_0(\phi_t^{-1}(x)) \det\left( \frac{\partial \phi_t^{-1}}{\partial x}(x) \right) $$<span style="color: rgb(100,37,208); background-color: inherit">，</span>$$\psi_t$$<span style="color: rgb(100,37,208); background-color: inherit">将噪声分布</span>$$p_0(x|x_1) = p(x)$$<span style="color: rgb(100,37,208); background-color: inherit">前推至 </span>$$p_t(x|x_1)$$，即：

$$[\psi_t]_* p(x) = p_t(x|x_1)$$

进而提供了生成条件概率路径的向量场：

$$\frac{d}{dt} \psi_t(x) = u_t(\psi_t(x) | x_1)$$

<span style="color: rgb(100,37,208); background-color: inherit">将</span>$$p_t(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">用</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">重新参数化，并将上式代入 CFM 损失</span>，得到：

$$L_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \left[ \left\| v_t(\psi_t(x_0)) - \frac{d}{dt} \psi_t(x_0) \right\|^2 \right] $$

由于$$\psi_t$$是一个简单可逆的仿射映射，<span style="color: rgb(100,37,208); background-color: inherit">可以利用方程</span>$$\frac{d}{dt} \psi_t(x) = u_t(\psi_t(x) | x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">闭式求解</span>$$u_t$$。记$$f'$$表示关于时间的导数，即$$f' = \frac{d}{dt} f$$，其中$$f$$为时间依赖函数。

> **<span style="color: rgb(36,91,219); background-color: inherit">定理 3</span>**：设$$p_t(x|x_1)$$为方程$$p_t(x|x_1) = \mathcal{N}(x \mid \mu_t(x_1), \sigma_t(x_1)^2 I)$$所示的高斯概率路径，$$\psi_t$$为对应的流映射$$\psi_t(x) = \sigma_t(x_1) x + \mu_t(x_1)$$。则定义$$\psi_t$$的唯一向量场形式为：
>
> $$u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \mu_t'(x_1)$$
>
> 因此，$$u_t(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">可以生成高斯路径</span>$$p_t(x|x_1)$$

上述公式对任意函数$$\mu_t(x_1)$$和$$\sigma_t(x_1)$$均完全通用，只要满足边界条件的可微函数即可。但是还存在一些特殊情形，恢复了先前扩散过程中使用的概率路径。由于直接处理概率路径可以完全脱离扩散过程的推理框架，因此可以直接基&#x4E8E;**`Wasserstein-2`**&#x6700;优传输解构造一个概率路径。

> **<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：**<span style="color: rgb(100,37,208); background-color: inherit">扩散条件向量场 Diffusion conditional VFs</span>**
>
> 扩散模型从数据点开始，逐步添加噪声直至近似纯噪声。这可表述为随机过程，<span style="color: rgb(100,37,208); background-color: inherit">为了在任意时间</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">获得闭式表达，需满足严格条件，从而导出具有特定均值</span>$$\mu_t(x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">和标准差</span>$$\sigma_t(x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">的高斯条件概率路径</span>$$p_t(x|x_1)$$。例如反向（噪声→数据）的 **<span style="color: rgb(216,57,49); background-color: inherit">VE</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ariance </span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">xploding）</span>路径形式为：
>
> $$p_t(x) = \mathcal{N}(x|x_1, \sigma_{1-t}^2 I)$$
>
> 其中$$\sigma_t$$为递增函数，$$\sigma_0 = 0$$，$$\sigma_1 \gg 1$$。由上式可得$$\mu_t(x_1) = x_1$$，$$\sigma_t(x_1) = \sigma_{1-t}$$。代入定理 3 的方程$$u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \mu_t'(x_1)$$，得到：
>
> $$u_t(x|x_1) = -\frac{\sigma_{1-t}'}{\sigma_{1-t}} (x - x_1)$$
>
> 反向（噪声→数据）&#x7684;**&#x20;<span style="color: rgb(216,57,49); background-color: inherit">VP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">ariance </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">reserving）</span>扩散路径形式为：
>
> $$p_t(x|x_1) = \mathcal{N}\left( x \mid \alpha_{1-t} x_1, \sqrt{1 - \alpha_{1-t}^2} I \right), \quad \alpha_t = e^{-\frac{1}{2} T(t)}, \quad T(t) = \int_0^t \beta(s)\, ds $$
>
> 其中$$\beta$$为噪声尺度函数。由上式可得$$\mu_t(x_1) = \alpha_{1-t} x_1$$，$$\sigma_t(x_1) = \sqrt{1 - \alpha_{1-t}^2}$$。代入定理 3 的方程$$u_t(x|x_1) = \frac{\sigma_t'(x_1)}{\sigma_t(x_1)} (x - \mu_t(x_1)) + \mu_t'(x_1)$$，得到：
>
> $$u_t(x|x_1) = \frac{\alpha_{1-t}'}{1 - \alpha_{1-t}^2} (\alpha_{1-t} x - x_1) = -\frac{T'(1 - t)}{2} \left[ \frac{e^{-T(1-t)} x - e^{-\frac{1}{2} T(1-t)} x_1}{1 - e^{-T(1-t)}} \right] $$

这里构造的<span style="color: rgb(100,37,208); background-color: inherit">条件向量场</span>$$u_t(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">实际上与此前在确定性概率流中使用的向量场在这些条件扩散过程下完全一致</span>。但是将扩散条件向量场与流匹配目标结合，提供了一种更具吸引力的训练替代方案。<span style="color: rgb(46,161,33); background-color: inherit">相较于现有的分数匹配方法，其更稳定、鲁棒</span>。

由于这些概率路径此前是从扩散过程推导而来，它们实际上并未在有限时间内达到真正的噪声分布。$$p_0(x)$$<span style="color: rgb(100,37,208); background-color: inherit">仅由合适的高斯分布近似用于采样和似然评估</span>。相比之下，构造对概率路径具有完全控制权，可直接设定$$\mu_t$$和 $$\sigma_t$$。

> **<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：**<span style="color: rgb(100,37,208); background-color: inherit">最优传输条件向量场 Optimal Transport conditional VFs</span>**
>
> &#x20;一种更自然的条件概率路径选择是<span style="color: rgb(100,37,208); background-color: inherit">让均值和标准差随时间线性变化</span>，即：
>
> $$\mu_t(x) = t x_1, \quad \sigma_t(x) = 1 - (1 - \sigma_{\min}) t$$
>
> 根据定理 3，该路径由如下向量场生成：
>
> $$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min}) t}$$
>
> <span style="color: rgb(100,37,208); background-color: inherit">与扩散条件向量场</span>$$u_t(x|x_1) = \frac{\alpha_{1-t}'}{1 - \alpha_{1-t}^2} (\alpha_{1-t} x - x_1) = -\frac{T'(1 - t)}{2} \left[ \frac{e^{-T(1-t)} x - e^{-\frac{1}{2} T(1-t)} x_1}{1 - e^{-T(1-t)}} \right] $$<span style="color: rgb(100,37,208); background-color: inherit">不同，该向量场对所有</span>$$t \in [0, 1]$$<span style="color: rgb(100,37,208); background-color: inherit">均有定义</span>。对应的条件流为：
>
> $$\psi_t(x) = (1 - (1 - \sigma_{\min}) t) x + t x_1$$
>
> 此时，CFM 损失$$L_{\text{CFM}}(\theta)$$形式为：
>
> $$L_{\text{CFM}}(\theta) = \mathbb{E}_{t, q(x_1), p(x_0)} \left[ \left\| v_t(\psi_t(x_0)) - \left( x_1 - (1 - \sigma_{\min}) x_0 \right) \right\|^2 \right]$$
>
> 允许均值和标准差线性变化不仅可以产生简单直观的路径，而且在如下意义上是最优的：<span style="color: rgb(100,37,208); background-color: inherit">条件流</span>$$\psi_t(x)$$<span style="color: rgb(100,37,208); background-color: inherit">实际上是两个高斯分布</span>$$p_0(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">与</span>$$p_1(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">之间的</span>**<span style="color: rgb(216,57,49); background-color: inherit">最优传输</span>**<span style="color: rgb(216,57,49); background-color: inherit"> </span>**<span style="color: rgb(216,57,49); background-color: inherit">OT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">O</span>**<span style="color: rgb(216,57,49); background-color: inherit">ptimal </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransport）</span> <span style="color: rgb(100,37,208); background-color: inherit">位移映射</span>。OT interpolant 定义为：
>
> $$p_t = [(1 - t) \text{id} + t \psi]_* p_0$$
>
> 其中$$\psi : \mathbb{R}^d \to \mathbb{R}^d$$是将$$p_0$$推送至$$p_1$$的 OT 映射，$$\text{id}$$表示恒等映射，即$$\text{id}(x) = x$$，$$(1 - t)\text{id} + t\psi$$称为 OT 位移映射。

直观上，<span style="color: rgb(100,37,208); background-color: inherit">粒子在 OT 位移映射下始终沿直线轨迹以恒定速度移动</span>。右图是 Diffusion 与 OT 条件向量场的采样路径。Diffusion 路径的采样轨迹可能超出最终样本，导致不必要的回溯，而 OT 路径则保证保持直线。

![](../../images/视觉多模态讲义（下）-image-65.png)

下图展示了扩散条件分数函数，即$$\nabla \log p_t(x|x_1)$$，其中$$p_t(x|x_1) = \mathcal{N}\left( x \mid \alpha_{1-t} x_1, \sqrt{1 - \alpha_{1-t}^2} I \right)$$，与 OT 条件向量场$$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min}) t}$$。两例中的起始$$p_0$$与终止$$p_1$$高斯分布完全相同。<span style="color: rgb(46,161,33); background-color: inherit">OT 向量场在时间上方向恒定，可能使回归任务更简单</span>。这一性质也<span style="color: rgb(100,37,208); background-color: inherit">可直接由方程</span>$$u_t(x|x_1) = \frac{x_1 - (1 - \sigma_{\min}) x}{1 - (1 - \sigma_{\min}) t}$$<span style="color: rgb(100,37,208); background-color: inherit">验证，因为该向量场可写为</span>$$u_t(x|x_1) = g(t) h(x|x_1)$$<span style="color: rgb(100,37,208); background-color: inherit">的形式</span>。

![](../../images/视觉多模态讲义（下）-image-52.png)

右图展示了扩散向量场的可视化。<span style="color: rgb(216,57,49); background-color: inherit">尽管条件流是最优的，但这并不能说明边缘向量场是最优传输 OT 解</span>。但还是希望边缘向量场能尽可能保持相对简单。

![](../../images/视觉多模态讲义（下）-image-63.png)

### 4.1.4 **<span style="color: rgb(36,91,219); background-color: inherit">DiT</span>**

* **<span style="color: rgb(36,91,219); background-color: inherit">扩散模型基础</span>**

1. **<span style="color: rgb(36,91,219); background-color: inherit">扩散模型</span>**

高斯扩散模型假设一个前向加噪过程，该过程<span style="color: rgb(100,37,208); background-color: inherit">逐渐对真实数据</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">添加噪声</span>：$$q(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1 - \bar{\alpha}_t)I)$$，其中常数$$\bar{\alpha}_t$$是超参数。通过应用重参数化技巧，可以采样$$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon_t$$，其中$$\epsilon_t \sim \mathcal{N}(0, I)$$。

扩散模型学习反向过程，这个过程<span style="color: rgb(100,37,208); background-color: inherit">反转前向过程的损坏：</span>$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t), \Sigma_\theta(x_t))$$<span style="color: rgb(100,37,208); background-color: inherit">，其中神经网络用于预测</span>$$p_\theta$$<span style="color: rgb(100,37,208); background-color: inherit">的统计量</span>。反向过程模型使用变分下界来训练$$x_0$$的对数似然，这可以写为$$L(\theta) = -p(x_0|x_1) + \sum_t D_{KL}(q^*(x_{t-1}|x_t, x_0)||p_\theta(x_{t-1}|x_t))$$。由于$$q^*$$和$$p_\theta$$都是高斯分布，$$D_{KL}$$可以通过两个分布的均值和协方差来计算。通过将$$\mu_\theta$$重新参数化为噪声预测网络$$\epsilon_\theta$$，模型可以使用<span style="color: rgb(100,37,208); background-color: inherit">预测噪声</span>$$\epsilon_\theta(x_t)$$<span style="color: rgb(100,37,208); background-color: inherit">和真实采样的高斯噪声</span>$$\epsilon_t$$<span style="color: rgb(100,37,208); background-color: inherit">之间的简单均方误差进行训练</span>：$$L_{\text{simple}}(\theta) = ||\epsilon_\theta(x_t) - \epsilon_t||_2^2$$。然而，为了训练具有学习到的反向过程协方差$$\Sigma_\theta$$的扩散模型，需要优化完整的$$D_{KL}$$项。即用$$L_{\text{simple}}$$训练$$\epsilon_\theta$$，并用完整的$$L$$训练$$\Sigma_\theta$$。当$$p_\theta$$被训练完成，可以通过初始化$$x_{t_{\text{max}}} \sim \mathcal{N}(0, I)$$并通过重参数化技巧采样 $$x_{t-1} \sim p_\theta(x_{t-1}|x_t)$$ 来生成新图像。

* **<span style="color: rgb(36,91,219); background-color: inherit">无分类器引导 Classifier-free guidance</span>**

条件扩散模型接受额外信息作为输入，例如类别标签$$c$$。在这种情况下，反向过程变为$$p_\theta(x_{t-1}|x_t, c)$$，其中$$\epsilon_\theta$$和$$\Sigma_\theta$$均以$$c$$为条件。在此设置中，<span style="color: rgb(100,37,208); background-color: inherit">无分类器引导鼓励采样过程找到使</span>$$\log p(c|x)$$<span style="color: rgb(100,37,208); background-color: inherit">较高的</span>$$x$$。根据贝叶斯规则，$$\log p(c|x) \propto \log p(x|c) - \log p(x)$$，因此$$\nabla_x \log p(c|x) \propto \nabla_x \log p(x|c) - \nabla_x \log p(x)$$。通过将扩散模型的输出解释为得分函数，**DDPM** 采样过程可以通过以下方式引导采样具有较高$$p(x|c)$$的$$x$$：

$$\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot \nabla_x \log p(x|c) \propto \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

其中$$s > 1$$表示引导的尺度（当$$s = 1$$时恢复标准采样）。<span style="color: rgb(100,37,208); background-color: inherit">通过在训练期间随机丢弃</span>$$c$$<span style="color: rgb(100,37,208); background-color: inherit">并将其替换为学习到的空嵌入</span>$$\emptyset$$<span style="color: rgb(100,37,208); background-color: inherit">，可以评估</span>$$c = \emptyset$$<span style="color: rgb(100,37,208); background-color: inherit">的扩散模型</span>。无分类器引导广泛被认为比通用采样技术显著提高了样本质量。

* **<span style="color: rgb(36,91,219); background-color: inherit">潜在扩散模型 Latent Diffusion Models</span>**

直接在高分辨率像素空间中训练扩散模型可能会计算成本过高。潜在扩散模型通过两阶段方法解决了这一问题：

> 学习一个**自编码器**，该编码器使用学习到的编码器$$E$$将图像压缩为较小的空间表示
>
> 训练图像编码$$z = E(x)$$的扩散模型，而不是原始图像$$x$$的扩散模型。然后可以通过从扩散模型中采样$$z$$并随后使用学习到的解码器将其解码为图像$$x = D(z)$$来生成新图像

潜在扩散模型在性能良好的同时，使用的浮点运算量仅为像素空间扩散模型的一小部分。由于计算效率十分重要，这使它们成为架构探索的一个有吸引力的起点。作者就是&#x5C06;**`DiT`**&#x5E94;用于潜在空间，使其图像生成流程成为一种混合方法：使用现成的卷&#x79EF;**`VAE`**&#x548C;基&#x4E8E;**`Transformer`**&#x7684; **DDPM**。

* **<span style="color: rgb(36,91,219); background-color: inherit">Diffusion Transformer 设计</span>**

**<span style="color: rgb(216,57,49); background-color: inherit">DiT</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Di</span>**<span style="color: rgb(216,57,49); background-color: inherit">ffusion </span>**<span style="color: rgb(216,57,49); background-color: inherit">T</span>**<span style="color: rgb(216,57,49); background-color: inherit">ransformer）</span>是用于扩散模型的新架构。他遵循标准 Transformer 架构，以保留其扩展特性。由于重点是训练图像的 DDPM，特别是图像的空间表示，DiT 基于 ViT 架构，作用在图像块序列上。DiT 保留了许多 ViT 的优点。

1. **<span style="color: rgb(36,91,219); background-color: inherit">图像分块</span>**

**DiT&#x20;**&#x7684;输入是一个空间表示$$z$$，例如，对于$$256 \times 256 \times 3$$的图像，$$z$$的形状为$$32 \times 32 \times 4$$。DiT 的第一层是分块，它通过线性层编码输入中的每个块，将空间输入转换为长度为$$T$$的 token 序列，每个 token 的维度为$$d$$。在分块之后，对所有输入 token 应用标准 ViT 正余弦位置编码。<span style="color: rgb(100,37,208); background-color: inherit">分块操作创建的 token 数量</span>$$T$$<span style="color: rgb(100,37,208); background-color: inherit">由块大小超参数</span>$$p$$<span style="color: rgb(100,37,208); background-color: inherit">决定</span>。如右图，将$$p$$减半会使$$T$$增加四倍，因此至少会使 Transformer 的总运算量增加四倍。这里虽然 对浮点运算量有显著影响，但改变$$p$$对下游参数计数没有明显影响。

实践中作者设定$$p = 2, 4, 8$$。

![](../../images/视觉多模态讲义（下）-image-60.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">DiT 块设计</span>**

在分块之后，输入 token 由一系列 Transformer 块处理。除了噪声图像输入外，扩散模型还会处理额外的条件信息，例如噪声时间步$$t$$、类别标签$$c$$、自然语言等。作者探索了四种不同的 Transformer 块变体，它们以不同的方式处理条件输入。这些设计对标准 ViT 块设计进行了小修改，如下图。

![](../../images/视觉多模态讲义（下）-image-53.png)

> * **<span style="color: rgb(36,91,219); background-color: inherit">In-context conditioning</span>**：<span style="color: rgb(100,37,208); background-color: inherit">简单地将</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$c$$<span style="color: rgb(100,37,208); background-color: inherit">的向量嵌入作为两个附加 token 追加到输入序列中，并将它们与图像 token 同等对待</span>。这类似于 ViT 中&#x7684;**`[CLS]`** token，可以直接使用标准 ViT 块。在最后一个块之后，从序列中移除条件 token。这种方法<span style="color: rgb(46,161,33); background-color: inherit">对模型的浮点运算量几乎没有影响</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Cross-attention block</span>**：<span style="color: rgb(100,37,208); background-color: inherit">将</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$c$$<span style="color: rgb(100,37,208); background-color: inherit">的嵌入连接成一个长度为</span>**`2`**<span style="color: rgb(100,37,208); background-color: inherit">的序列，与图像 token 序列分开</span>。Transformer 块修改为在多头自注意力块之后包含一个额外的多头交叉注意力层，这与 Transformer 原始解码器设计类似，<span style="color: rgb(216,57,49); background-color: inherit">交叉注意力对模型的浮点运算量影响最大，大约增加了</span>**`15%`**<span style="color: rgb(216,57,49); background-color: inherit">的开销</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Adaptive layer norm (adaLN) block</span>**：鉴于自适应归一化层在 GAN 和使用 UNet 主干的扩散模型中的广泛应用，作者探索了<span style="color: rgb(100,37,208); background-color: inherit">用自适应层归一化 adaLN 替换 Transformer 块中的标准层归一化</span>。这里不是直接学习逐维的缩放和偏移参数$$\gamma$$和$$\beta$$，而是<span style="color: rgb(100,37,208); background-color: inherit">从</span>$$t$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$c$$<span style="color: rgb(100,37,208); background-color: inherit">的嵌入向量之和中学习它们</span>。<span style="color: rgb(46,161,33); background-color: inherit">adaLN 增加的浮点运算量最少，因此计算效率最高</span>。且它是唯一一种对所有 token 应用相同函数的条件化机制。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">adaLN-Zero block</span>**：ResNet 展示了将每个残差块初始化为恒等函数是有益的，将每个块中的最终批量归一化缩放因子$$\gamma$$初始化为零可以加速训练。以 U-Net 为主干的扩散模型也采用了类似的初始化策略，在任何残差连接之前将每个块中的最终卷积层初始化为零。遵循这个设计理念，作者对 adaLN DiT 块进行修改，实现了相同的效果。<span style="color: rgb(100,37,208); background-color: inherit">除了学习</span>$$\gamma$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\beta$$<span style="color: rgb(100,37,208); background-color: inherit">外，还学习了应用于 DiT 块内任何残差连接之前的逐维缩放参数</span>$$\alpha$$。这里将 MLP 初始化为输出全零向量$$\alpha$$；这将整个 DiT 块初始化为恒等函数。<span style="color: rgb(46,161,33); background-color: inherit">与普通的 adaLN 块一样，adaLN-Zero 对模型的浮点运算量几乎没有影响</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">模型规模</span>**

应用$$N$$个 DiT 块，每个块隐藏维度大小为$$d$$，并且使用标准 Transformer 配置，联合缩放$$N$$、$$d$$和注意力头的数量。论文使用四种配置：**`DiT-S`**、**`DiT-B`**、**`DiT-L`**&#x548C;**`DiT-XL`**。覆盖了广泛的模型规模和运算量，范围&#x4ECE;**`0.3`**&#x5230;**`118.6`**&#x47;flops。

![](../../images/视觉多模态讲义（下）-image-57.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">Transformer 解码器</span>**

在最后一个 DiT 块之后，将图像 token 序列解码为输出噪声预测和输出对角协方差预测。这两个输出的形状等于原始空间输入，这里<span style="color: rgb(100,37,208); background-color: inherit">使用标准线性解码器来输出</span>；最终应用层归一化，并<span style="color: rgb(100,37,208); background-color: inherit">线性解码每个 token 为</span>$$p \times p \times 2C$$<span style="color: rgb(100,37,208); background-color: inherit">张量，其中</span>$$C$$<span style="color: rgb(100,37,208); background-color: inherit">是输入到 DiT 的空间输入的通道数。最后，将解码后的标记重新排列为其原始空间布局，以获得预测的噪声和协方差</span>。

---

[Previous](05-GAN--AE--Flow--总结.md) | [Contents](../../README.md) | [Next](07-Diffusion-Model--Stable-Diffusion-系列.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=6)
