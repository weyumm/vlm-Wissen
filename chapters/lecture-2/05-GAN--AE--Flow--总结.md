[Previous](04-GAN--AE--Flow--Flow-based-model.md) | [Contents](../../README.md) | [Next](06-Diffusion-Model--Diffusion-基础.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=5)

## 3.5 <span style="color: rgb(36,91,219); background-color: inherit">总结</span>

### 3.5.1 <span style="color: rgb(36,91,219); background-color: inherit">四种方法区别</span>

假设有一批大小为$$N$$的真实图片数据集合，$$S_x=\{x_1, x_2, \cdots, x_N\}$$

> ✏️ * 如果有一个完美的分布$$p(x)$$能生成集合$$S_x$$中的元素，就意味着要求每次采样出来的$$x' \sim p(x)$$都满足$$x' \in S_x$$
>
> ✏️ * 当集合元素$$N$$比较小的时候，$$p(x)$$可以强行背下来即可；但是<span style="color: rgb(216,57,49); background-color: inherit">当</span>$$N$$<span style="color: rgb(216,57,49); background-color: inherit">变得非常大的时候，完美分布</span>$$p(x)$$<span style="color: rgb(216,57,49); background-color: inherit">就几乎不可能获取了</span>

而 AIGC 的任务就是逼近这个完美分布$$p(x)$$，这里的$$x$$可以是图像，可以是视频，可以是音频等。那具体如何逼近这个完美分布$$p(x)$$呢？主要有以下几种方法：

![](../../images/视觉多模态讲义（下）-BLyXbjjSsotcayxaf0kcFZHWn3c.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">极大似然估计</span>**

估计分布$$p(x)$$有个最直接的方法，就是极大似然估计。<span style="color: rgb(100,37,208); background-color: inherit">先假设</span>$$p(x)$$<span style="color: rgb(100,37,208); background-color: inherit">的形式为</span>$$p_\theta(x)$$<span style="color: rgb(100,37,208); background-color: inherit">，然后待定其中的参数</span>$$\theta$$。例如<span style="color: rgb(220,155,4); background-color: inherit">假设</span>$$x$$<span style="color: rgb(220,155,4); background-color: inherit">来自多元高斯分布</span>$$\mathcal{N}(\mu, \Sigma)$$<span style="color: rgb(220,155,4); background-color: inherit">，即</span>$$x \sim \mathcal{N}(\mu, \Sigma)$$<span style="color: rgb(220,155,4); background-color: inherit">，其中高斯分布的维度和图片的像素个数一致</span>。于是这里的待定参数就是$$\theta=(\mu, \Sigma)$$，此时似然估计函数为

$$L_\theta(S_x)=p(S_x;\theta)=\prod_{i=1}^N p(x_i;\theta)$$

做最大似然估计，用梯度下降或者公式推导求解出最佳参数$$\hat{\theta}$$

$$(\hat{\mu}, \hat{\Sigma})=\hat{\theta}=\text{argmax}_\theta L_\theta(S_x)=\text{argmax}_\theta(-\sum_{i=1}^N \ln p(x_i;\theta))$$

这样就求得了$$p(x)$$的分布为$$\mathcal{N}(\hat{\mu}, \hat{\Sigma})$$。这样做简单直观，但是弊端也很明显：

> 🍰 * **$$p(x)$$<span style="color: rgb(36,91,219); background-color: inherit">形式未知</span>**：需要丰富的领域知识才能笃定$$p(x)$$就是某个形式，其实对于复杂问题来说没人知道分布的参数化表达式是啥
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">参数量</span>$$\theta$$<span style="color: rgb(36,91,219); background-color: inherit">的空间太大</span>**：多元高斯分布的维度跟图片像素数一样大，例如 ImageNet 是$$\mathbb{R}^{3 \times 224 \times 224}$$维度，那么意味着&#x662F;**`150528`**&#x5143;高斯分布，那需要海量的数据才能估计得准确

* **<span style="color: rgb(36,91,219); background-color: inherit">隐变量估计</span>**

为了解决上述两个问题，更加广泛使用的方式是隐变量估计法，<span style="color: rgb(100,37,208); background-color: inherit">目前生成模型主流的 4 大类方法本质上都是生成模型</span>：

![](../../images/视觉多模态讲义（下）-Y8aUbPM4GoEsKuxEalucZV5nnQe.png)

首先来说一下什么是隐变量模型，就是引入隐变量$$z$$，期望从简单的分布$$p(z)$$出发，用积分公式来间接求出复杂的$$p(x)$$：

$$p(x)=\int p(x|z) p(z) \text{d}z$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：
>
> * 这里的隐变量满足两个性质：
>
>   1. **$$p(z)$$<span style="color: rgb(36,91,219); background-color: inherit">分布表达式简单</span>**，例如是<span style="color: rgb(220,155,4); background-color: inherit">普通的多元高斯分布</span>
>
>   2. **$$z$$<span style="color: rgb(36,91,219); background-color: inherit">自由度低</span>**，例如<span style="color: rgb(220,155,4); background-color: inherit">对于多元高斯分布，维度为</span>$$d$$<span style="color: rgb(220,155,4); background-color: inherit">，其中</span>$$d \ll 3 \times 224 \times 224$$
>
> * $$p(x|z)$$可以是个确定性的过程$$p_\theta(x|z)$$，即$$x=f(z)$$，$$f$$用神经网络来表示
>
> * 有了万能拟合函数$$f$$，从简单分布$$p(z)$$生成任意分布$$p(x)$$过程为
>
>   * **<span style="color: rgb(36,91,219); background-color: inherit">Step 1</span>**: 假设想得到$$p(x=x')$$的概率，那么就先研究$$f$$找到所有$$z$$使得$$x'=f(z)$$
>
>   📚 * **<span style="color: rgb(36,91,219); background-color: inherit">Step 2</span>**: 假设$$z$$的解集合为$$\{z_1, z_2, \cdots, z_M\}$$，则<span style="color: rgb(100,37,208); background-color: inherit">把这些</span>$$z$$<span style="color: rgb(100,37,208); background-color: inherit">出现的概率全部加起来就是</span>$$x'$$<span style="color: rgb(100,37,208); background-color: inherit">出现的概率，即可以得到</span>$$p(x=x')=\sum_{i=1}^Mp(z=z_i)$$

但是<span style="color: rgb(216,57,49); background-color: inherit">要找到 Step 2 里面</span>$$z$$<span style="color: rgb(216,57,49); background-color: inherit">的解集合是很难的，所以虽然</span>$$p(z)$$<span style="color: rgb(216,57,49); background-color: inherit">很简单，</span>$$f$$<span style="color: rgb(216,57,49); background-color: inherit">是个确定过程，但由于</span>$$f$$<span style="color: rgb(216,57,49); background-color: inherit">不一定可逆，</span>$$p(x)$$<span style="color: rgb(216,57,49); background-color: inherit">仍然很难直接算出表达式</span>。目前能做的是<span style="color: rgb(100,37,208); background-color: inherit">数值模拟的方式采样出</span>$$x' \sim p(x)$$<span style="color: rgb(100,37,208); background-color: inherit">，常用的为蒙特卡洛方法</span>：

> 🚅 1. 首先依$$p(z)$$的分布采样$$K$$次$$z$$，得到数组$$T_z=[z_1, z_2, \cdots, z_K]$$，其中$$z_i \sim p(z)$$
>
> 2. 根据$$x=f(z)$$计算得到数组$$T_x=[x_1, x_2, \cdots, x_K]$$
>
> 🚅 3. 注意如果$$p(z_i)$$概率更高，那么$$z_i$$出现在$$T_z$$的次数会越多，相应的$$x_i$$出现在$$T_x$$的概率就会高
>
> 4. 只需要简单对$$T_x$$取均值即可得到本次采样最终的$$x'=\frac{1}{K}\sum_{i=1}^Kx_i$$

蒙特卡洛方法简单，但是也<span style="color: rgb(100,37,208); background-color: inherit">有 2 个可改进方向</span>：

> * **<span style="color: rgb(36,91,219); background-color: inherit">提高采样效率（VAE 方向）</span>**：这里的$$K$$可能需要非常大才能得到对应满意的结果，也就是生成一个$$x’$$需要很多次$$z$$的采样。因此一种思路是<span style="color: rgb(100,37,208); background-color: inherit">提高</span>$$z$$<span style="color: rgb(100,37,208); background-color: inherit">的采样效率，最好采样一次就能完成任务</span>
>
> 🌅 * **<span style="color: rgb(36,91,219); background-color: inherit">改变损失误差（GAN 方向）</span>**：在实际网络训练中，蒙特卡洛采样过程每次得到的$$x'$$会启发式地和真实图片$$x$$做均方误差。但没有证据表明均方误差就是最优度量，因此另外一种思路是<span style="color: rgb(100,37,208); background-color: inherit">用神经网络</span>$$D(x, x')$$<span style="color: rgb(100,37,208); background-color: inherit">来隐式学这个度量方式</span>

* **<span style="color: rgb(36,91,219); background-color: inherit">VAE</span>**

在隐变量的积分式中：

$$p(x)=\int p(x|z) p(z) \text{d}z$$

如何提高$$z$$的采样效率呢？一种思路是<span style="color: rgb(100,37,208); background-color: inherit">用</span>$$p(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">来估计</span>$$p(z)$$<span style="color: rgb(100,37,208); background-color: inherit">，因为有了</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">的信息，</span>$$z$$<span style="color: rgb(100,37,208); background-color: inherit">怎么着都应该变得确定一点。这里研究的变量是</span>$$z$$<span style="color: rgb(100,37,208); background-color: inherit">，因此</span>**$$p(z)$$<span style="color: rgb(100,37,208); background-color: inherit">是先验分布，</span>$$p(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">是后验分布</span>**。但即便如此，**<span style="color: rgb(216,57,49); background-color: inherit">无中生有出</span>$$p(z|x)$$<span style="color: rgb(216,57,49); background-color: inherit">会陷入鸡生蛋和蛋生鸡的窘境</span>**，因此<span style="color: rgb(100,37,208); background-color: inherit">需要一个具体可优化模型来逼近</span>$$p(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">，于是引入新的概率分布</span>$$q_\theta(z|x)$$：

> * **<span style="color: rgb(36,91,219); background-color: inherit">Step 1</span>**：先想办法构造一个新的分布$$q_\theta(z|x)$$，使得$$z' \sim q_\theta(z|x)$$
>
> 🎹 * **<span style="color: rgb(36,91,219); background-color: inherit">Step 2</span>**：然后不断优化$$q_\theta(z|x)$$靠近$$p(z|x)$$，使得最终近似满足$$z'\sim p(z|x)$$

<span style="color: rgb(100,37,208); background-color: inherit">引入的</span>$$q_\theta(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">是一系列的分布家族，并且需要在里面做优化，选择最合适的</span>$$q_{\theta^*}(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">分布，这个过程就是</span>**<span style="color: rgb(100,37,208); background-color: inherit">变分</span>**。这里用 KL 衡量一下这两个分布的差距

$$\begin{aligned}\text{KL}(q_\theta(z|x) ||p(z|x))&=\int q_\theta(z|x)\ln \frac{q_\theta(z|x)}{p(z|x)} \text{d}z \\&= \mathbb{E}_{z\sim q_\theta(z|x)}[\ln q_\theta(z|x) - \ln p(z|x)]\\&= \mathbb{E}_{z\sim q_\theta(z|x)}[\ln q_\theta(z|x) - \ln \frac{p(x|z)p(z)}{p(x)}]\\&= \mathbb{E}_{z\sim q_\theta(z|x)}[\ln q_\theta(z|x) - \ln p(x|z) - \ln p(z) + \ln p(x)]\\&=\text{KL}(q_\theta(z|x)||p(z))-\mathbb{E}_{z\sim q_\theta(z|x)}\ln p(x|z) + \ln p(x)\end{aligned}$$

可以看到，**<span style="color: rgb(100,37,208); background-color: inherit">通过变分绕过了虚无缥缈的</span>$$p(z|x)$$**<span style="color: rgb(100,37,208); background-color: inherit">，留下的 3 项都是可以分析的对象</span>。目标&#x662F;**<span style="color: rgb(100,37,208); background-color: inherit">希望左侧的 KL 距离越来越小，并且要重点关注</span>$$q_\theta(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">要怎么变才能逼近</span>$$p(z|x)$$**，那就可以逐项分析：

> ⛱️ * 右边第一项$$\text{KL}(q_\theta(z|x)||p(z))$$，希望尽可能小，也就是<span style="color: rgb(100,37,208); background-color: inherit">新引入的采样过程得到的</span>$$z'$$<span style="color: rgb(100,37,208); background-color: inherit">不能离原来的标准多元高斯分布假设</span>$$p(z)$$<span style="color: rgb(100,37,208); background-color: inherit">太远</span>。这里是希望$$z'$$的方差变小，而不是完全变个样。这就是正则项要尽可能小
>
> * 右边第二项$$-\mathbb{E}_{z\sim q_\theta(z|x)}\ln p(x|z)$$，希望尽可能小，也就是$$\mathbb{E}_{z\sim q_\theta(z|x)}\ln p(x|z)$$<span style="color: rgb(100,37,208); background-color: inherit">要尽可能大，含义就是每张图的似然</span>$$\ln p(x|z)$$<span style="color: rgb(100,37,208); background-color: inherit">在所有</span>$$z$$<span style="color: rgb(100,37,208); background-color: inherit">采样中要尽可能解释观测数据</span>$$x$$，这就是重建误差要尽可能小
>
> * 右边第三项$$\ln p(x)$$，是个<span style="color: rgb(100,37,208); background-color: inherit">跟</span>$$q_\theta(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">没啥关系的常数，不随</span>$$q_\theta(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">变化而变化</span>，忽略

上式也常常写成这个形式：

$$\begin{aligned}\ln p(x) &=\text{KL}(q_\theta(z|x) ||p(z|x)) -\text{KL}(q_\theta(z|x)||p(z))+\mathbb{E}_{z\sim q_\theta(z|x)}\ln p(x|z)\\&\ge -\text{KL}(q_\theta(z|x)||p(z))+\mathbb{E}_{z\sim q_\theta(z|x)}\ln p(x|z) \\&= \text{Variational Lower Bound}\end{aligned}$$

要最大化$$p(x)$$，只需要不断提&#x9AD8;**`Variational Lower Bound`**&#x5373;可，即最大&#x5316;**`VLB`**。由此知道了第一项就是正则项，第二项就是重建误差，整个 VAE 的网络也就得出来了：

> $$q_\theta(z|x)$$过程是采样过程。而神经网络$$g$$是个确定性过程，本身没有随机性，因此<span style="color: rgb(100,37,208); background-color: inherit">为了让网络参数可学习，需要引入重参数化技巧</span>：
>
> * $$g$$过程直接预测每个维度的均值$$\mu_i$$和方差$$\sigma_i$$，然后叠加随机数发生器$$\epsilon_i \sim \mathcal{N}(0, 1)$$，得到采样结果$$z'_i = \sigma_i \cdot \epsilon_i+ \mu_i$$
>
> * 每个维度$$i$$都操作完之后得到$$z'=[z'_1, z'_2, \cdots, z'_d]$$
>
> * 所有$$\mu_i$$组成$$\mu$$，所有$$\sigma_i$$组成$$\Sigma$$，则$$z' \sim \mathcal{N}(\mu, \Sigma)$$

![](../../images/视觉多模态讲义（下）-L4evb2ktUo56HyxljAKcGqoDnXc.png)

> $$p_\theta(x|z)$$过程是个确定性过程，当$$z'$$被采样出来之后，直接调用 decoder 网络就能得到重建的$$\hat{X}$$

对 VAE 网络有了认识，再回到 VAE 的 loss，包含两项：

> 🎉 * **<span style="color: rgb(36,91,219); background-color: inherit">最小化正则项</span>**：从$$\text{KL}(q_\theta(z|x)||p(z))$$转为$$\text{KL}(\mathcal{N}(\mu,\Sigma)||\mathcal{N}(0, I))$$，有闭式解，即最小化$$\sum_{j=1}^d\frac{1}{2}(-\ln \sigma_j^2 + \sigma_j^2 + \mu_j^2 -1)$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">最小化重建误差项</span>**：$$\mathbb{E}_{z\sim q_\theta(z|x)}\ln p(x|z)$$可以转换为要求$$x_i$$和$$\hat{x}_i$$的均方误差尽可能小，即最小化$$\sum_{i=1}^N||x_i - \hat{x}^i||^2_2$$

最终 VAE 的 loss 为：

$$\sum_{i=1}^N (\frac{1}{2}\sum_{j=1}^d(\sigma_{i,j}^2 +\ln \frac{1}{\sigma_{i,j}^2} + \mu_{i,j}^2 -1) + ||x_i - \hat{x}^i||^2_2)$$

模型收敛之后<span style="color: rgb(100,37,208); background-color: inherit">把 encoder 网络丢弃，每次按标准多元高斯分布采样</span>$$z$$<span style="color: rgb(100,37,208); background-color: inherit">，然后经过 decoder 网络输出</span>$$\hat{x}$$<span style="color: rgb(100,37,208); background-color: inherit">，完成一次采样</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">GAN</span>**

VAE 的思路是<span style="color: rgb(100,37,208); background-color: inherit">提高</span>$$p(z)$$<span style="color: rgb(100,37,208); background-color: inherit">采样效率，参数化后验分布</span>$$q_\theta(z|x)$$<span style="color: rgb(100,37,208); background-color: inherit">，引入数据依赖直接得到均值</span>$$\mu$$<span style="color: rgb(100,37,208); background-color: inherit">和方差</span>$$\Sigma$$<span style="color: rgb(100,37,208); background-color: inherit">，使得</span>$$q_\theta(z|x)=\mathcal{N}(\mu, \Sigma)$$

GAN 的思路&#x662F;**<span style="color: rgb(100,37,208); background-color: inherit">保留</span>$$p(z)$$<span style="color: rgb(100,37,208); background-color: inherit">是固定的标准多元高斯分布，额外引入新网络</span>$$D(x', x)$$<span style="color: rgb(100,37,208); background-color: inherit">对生成的</span>$$x'$$<span style="color: rgb(100,37,208); background-color: inherit">和真实的</span>$$x$$<span style="color: rgb(100,37,208); background-color: inherit">做分类</span>**，而不是用固定的均方误差损失

GAN的数学化表达如下：

![](../../images/视觉多模态讲义（下）-FFdtbmHkxoKtHJxIG5Cc2Yxbnsf.png)

$$-\mathbb{E}_{x\sim P_r}[\log D(x)]-\mathbb{E}_{x\sim P_g}[\log(1- D(x))]$$

* **<span style="color: rgb(36,91,219); background-color: inherit">Flow Model</span>**

前面是如何用神经网络$$f$$来表示$$p_\theta(x|z)$$来建立$$z$$和$$x$$的关系，方法如下

> * **<span style="color: rgb(36,91,219); background-color: inherit">Step 1</span>**: 假设想得到$$p(x=x')$$的概率，那么就先研究$$f$$找到所有$$z$$使得$$x'=f(z)$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Step 2</span>**: 假设$$z$$的解集合为$$\{z_1, z_2, \cdots, z_M\}$$，则把这些$$z$$出现的概率全部加起来就是$$x'$$出现的概率，即可以得到$$p(x=x')=\sum_{i=1}^Mp(z=z_i)$$

但想得到 Step 2 里面$$z$$的解集合是很难的，所以<span style="color: rgb(216,57,49); background-color: inherit">虽然</span>$$p(z)$$<span style="color: rgb(216,57,49); background-color: inherit">很简单，</span>$$f$$<span style="color: rgb(216,57,49); background-color: inherit">是个确定过程，但由于</span>$$f$$<span style="color: rgb(216,57,49); background-color: inherit">不一定可逆， </span>$$p(x)$$<span style="color: rgb(216,57,49); background-color: inherit">仍然很难直接算出表达式</span>。于是退而求其次，不求$$p(x)$$的表达式了，只求蒙特卡洛的方式能采样出$$x' \sim p(x)$$就好。但如果想硬上求出表达式呢，其实也可以，只是需要满足

> * $$f$$函数是可逆的，即知道了$$x=f(z)$$，那么容易推导出来$$z =f^{-1}(x)$$
>
> * 因此$$f$$是一一映射的，同时$$x$$和$$z$$要求是维度一样的，如果$$x \in \mathbb{R}^{D}$$那么$$z \in \mathbb{R}^{D}$$
>
> ✍️ * 既然$$f$$是一一映射的，也用不着积分了，直接变量替换$$z=f^{-1}(x) \sim \mathcal{N}(0, I)$$

变量替换后概率密度为：

$$p(x)=\frac{1}{(2\pi)^{D/2}}\exp(-\frac{1}{2}||f^{-1}(x)||^2)\bigg |\det[\frac{\partial {f^{-1}}}{\partial x}]\bigg|$$

可见除了要求$$f$$可逆，还要求逆函数$$f^{-1}(x)$$的行列式计算也要简单。一个比较经典的使用 flow model 的工作&#x662F;**`RevNet`**：

> * 把$$x$$分成两部分$$x_1$$和$$x_2$$，把$$y$$也分成两部分$$y_1$$和$$y_2$$
>
> 🏆 * 输入和输出 tensor shape 一样的函数$$\mathcal{F}$$和$$\mathcal{G}$$

那么可以计算前馈和反传，如右图：

![](../../images/视觉多模态讲义（下）-C0oebOKLhosGQBxp7l0cr1Qvn4d.png)

![前馈                                  反传](../../images/视觉多模态讲义（下）-Lz7FbHcifoOt4WxtR1OcTjcYnic.jpg)

* **<span style="color: rgb(36,91,219); background-color: inherit">Diffusion</span>**

VAE 的思路是提高$$p(z)$$采样效率：

> 👍 * **<span style="color: rgb(36,91,219); background-color: inherit">Encoder</span>**：用参数化的后验分布模型$$q_\theta(z|x)$$，直接预测均值$$\mu$$和方差$$\Sigma$$，使得$$q_\theta(z|x)=\mathcal{N}(\mu, \Sigma)$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">Decoder</span>**：采样得到$$z$$之后，经过参数化的$$p_\theta(x|z)$$生成最后的$$\hat{x}$$

如果$$q_\theta(z|x)$$不是一步生成的，而是经过$$T$$步，那么 Encoder 会变为：

$$q_\theta(z|x)=q_\theta(z|x_{T-1})\cdot q_\theta(x_{T-1}|x_{T-2})\cdots q_\theta(x_{t}|x_{t-1}) \cdots q_\theta(x_2|x_1) \cdot q_\theta(x_1|x)$$

Decoder 会变成：

$$p_\theta(x|z)=p_\theta(x|x_1)\cdot p_\theta(x_1|x_2)\cdots p_\theta(x_{t-1}|x_t) \cdots p_\theta(x_{T-2}|x_{T-1})\cdot p_\theta(x_{T-1}|z)$$

这个就是 diffusion model 的思想

![](../../images/视觉多模态讲义（下）-G8gCbvKC6oIGdexSc4lcWOavnrb.png)

> 🌰 * 从右往左的 encoder 是无参数的$$q(x_t|x_{t-1})$$。不像 VAE 是带超参数的，这是<span style="color: rgb(100,37,208); background-color: inherit">人为定义的过程，从原始清晰图</span>$$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">开始，每次转换成新的高斯噪声</span>，逐渐变成标准多元高斯变量$$x_T$$。
>
> * 从左往右的 decoder 是带参数的$$p_\theta(x_{t-1}|x_t)$$。不是像 VAE 一样直接预测$$\hat{x}$$，而是<span style="color: rgb(100,37,208); background-color: inherit">预测高斯噪声，并且会减去这个高斯噪声得到更清晰的图片</span>
>
> * 另外中间隐变量$$x_T$$的维度变成了跟原始图片$$x_0$$一样大

首先定义递增的常量序列$$\beta_t$$，满足$$0<\beta_1 \lt \beta_2 \lt \cdots \lt \beta_T<1$$。定义观测原图为随机变量$$x_0$$，然后定义从随机变量$$x_{t-1}$$到随机变量$$x_t$$的分布关系为：

$$q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{1-\beta_t}x_{t-1}, \beta_t )$$

即$$x_t$$是均值$$\sqrt{1-\beta_t}x_{t-1}$$且方差为$$\beta_t$$的高斯分布，用类似重参数分解可以得到：

$$x_t = \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t} \epsilon_{t-1}, \ \epsilon_i \sim \mathcal{N}(0, 1)$$

<span style="color: rgb(100,37,208); background-color: inherit">其中</span>$$x_{t-1}$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\epsilon_{t-1}$$<span style="color: rgb(100,37,208); background-color: inherit">前面的两个系数平方和等于</span>$$1$$<span style="color: rgb(100,37,208); background-color: inherit">，</span>$$\beta_t$$<span style="color: rgb(100,37,208); background-color: inherit">单调递增且</span>$$\beta_t \in (0, 1)$$<span style="color: rgb(100,37,208); background-color: inherit">，则可以保证</span>$$t=0$$<span style="color: rgb(100,37,208); background-color: inherit">时候方差几乎为</span>$$0$$<span style="color: rgb(100,37,208); background-color: inherit">，</span>$$t=T$$<span style="color: rgb(100,37,208); background-color: inherit">时方差几乎为</span>$$1$$。如果定义$$\alpha_t=1-\beta_t$$，$$\bar{\alpha}_t=\prod_{i=1}^t \alpha_i$$，$$\bar{\epsilon}_k \sim \mathcal{N}(0, 1)$$，代表$$k$$个高斯分布合并之后的新高斯分布，那么递推展开可以得到：

$$\begin{aligned}x_t&=\sqrt{\alpha_t} x_{t-1} + \sqrt{1-\alpha_t} \epsilon_{t-1} \\ &=\sqrt{\alpha_t \alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_t \alpha_{t-1}} \bar{\epsilon}_2 \\&= \cdots \\&=\sqrt{\bar{\alpha_t}}x_0 +\sqrt{1-\bar{\alpha_t}}\bar{\epsilon}_t\end{aligned}$$

> **<span style="color: rgb(222,120,2); background-color: inherit">注</span>**：
>
> * 这里用了方差的性质，即<span style="color: rgb(100,37,208); background-color: inherit">两个高斯分布的和还是高斯分布，并且新方差等于这两个高斯分布的方差</span>
>
> * $$\bar{\epsilon}_k$$是$$k$$个高斯分布合并之后的新高斯分布
>
> * $$x_0$$<span style="color: rgb(100,37,208); background-color: inherit">和</span>$$\bar{\epsilon}_t$$<span style="color: rgb(100,37,208); background-color: inherit">前面两个系数的平方和仍然是</span>$$1$$

观察$$x_t$$可以发现

> * 随着$$t \rightarrow T，\sqrt{\bar{\alpha_t}} \rightarrow 0$$且$$\sqrt{1-\bar{\alpha_t}} \rightarrow 1$$，因此$$x_t \rightarrow \mathcal{N}(0, 1)$$，<span style="color: rgb(100,37,208); background-color: inherit">逐渐变成标准高斯分布</span>，极端情况下$$x_T =\mathcal{N}(0, 1)$$
>
> * <span style="color: rgb(100,37,208); background-color: inherit">不仅</span>$$q(x_t|x_{t-1})=\mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t )$$<span style="color: rgb(100,37,208); background-color: inherit">可以直接计算，并且</span>$$q(x_t|x_0)=\mathcal{N}(\sqrt{\bar{\alpha_t}}x_0 , 1-\bar{\alpha_t})$$<span style="color: rgb(100,37,208); background-color: inherit">也可以直接计算</span>
>
> * 整个 Encoder 过程的是完全透明的，可以高效的计算中间任意分布$$q(x_t|x_0)$$的方式

有了 Encoder 过程后，可以寻找优化目标，这里有两种方法：

> * **<span style="color: rgb(36,91,219); background-color: inherit">方法 1</span>**：最小化$$D_{\text{KL}}(q(x_{1:T}|x_0)||p_\theta(x_{1:T}|x_0))$$，和前面的 VAE 一样
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">方法 2</span>**：最小化 CE 损失$$-\mathbb{E}_{q(x_0)}\log p_\theta(x_0)$$

这其实都是要最小化$$L_{t-1}$$：

$$L_{t-1}=\text{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))$$

其中$$q(x_{t-1}|x_t, x_0)$$可以用贝叶斯公式求得，$$p_\theta(x_{t-1}|x_t)$$是$$q(x_{t-1}|x_t)$$的参数化建模，假设也符合高斯分布，只需关注均值和方差即可

1. **<span style="color: rgb(36,91,219); background-color: inherit">分析</span>$$\mathbf{q(x_{t-1}|x_t, x_0)}$$**

可以使用使用贝叶斯公式

$$\begin{aligned}q(x_{t-1}|x_t, x_0)&=q(x_t|x_{t-1},x_0) \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)} \\&=q(x_t|x_{t-1}) \frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}\end{aligned}$$

<span style="color: rgb(100,37,208); background-color: inherit">每个</span>$$q(\cdot)$$<span style="color: rgb(100,37,208); background-color: inherit">代表的是高斯函数的密度函数，因此可以带入得到</span>$$q(x_{t-1}|x_t, x_0)$$<span style="color: rgb(100,37,208); background-color: inherit">的密度函数，并且也是高斯分布</span>。此时是可以用待定系数的方式推导求出$$q(x_{t-1}|x_t, x_0)$$的均值$$\mu_t(x_t, x_0)$$和方差$$\sigma_t^2(x_t, x_0)$$的，如下所示

$$\begin{aligned}\mu_t(x_t,x_0)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)}{1-\bar{\alpha}_t}x_0\\\sigma_t^2(x_t, x_0)&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t\end{aligned}$$

把式中的$$x_0$$代换掉，可以得

$$\begin{aligned}\mu_t(x_t,x_0)&=\frac{1}{\sqrt{\alpha_t}} (x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \bar{\epsilon}_t)\\\sigma_t^2(x_t, x_0)&=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \cdot \beta_t\end{aligned}$$

分析这个方差公式$$\sigma_t^2(x_t, x_0)$$，其实有好几种选择

> * **<span style="color: rgb(36,91,219); background-color: inherit">选择 1</span>**：$$\bar{\alpha}_{t-1}$$和$$\bar{\alpha}_t$$是通过预先设置$$\beta_t$$直接计算而来，运算结果是常量
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">选择 2</span>**：DDPM 里面进一步发现可以简化成$$\sigma_t^2(x_t, x_0)=\beta_t$$
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">选择 3</span>**：Improved DDPM 里面发现也可以设置成可学

这里使用第 2 种选择$$\sigma_t^2(x_t, x_0)=\beta_t$$，因此

$$q(x_{t-1}|x_t, x_0)=\mathcal{N}(x_{t-1};\mu_t(x_t, x_0), \beta_t)$$

* **<span style="color: rgb(36,91,219); background-color: inherit">分析</span>$$\mathbf{p_\theta(x_{t-1}|x_t)}$$**

由于已经假设$$p_\theta(x_{t-1}|x_t)$$是高斯分布且满足

$$p_\theta(x_{t-1}|x_t)=\mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t)$$

那么当计算两个高斯分布的 KL 散度，利用现成结论公式可以得到

$$\begin{aligned}L_{t-1}&=\text{KL}(\mathcal{N}(x_{t-1};\mu_t(x_t, x_0), \beta_t)||\mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t))\\&\propto||\mu_t(x_t, x_0)-\mu_\theta(x_t, t)||^2\end{aligned}$$

DDPM 的贡献点之一，就是发现与其让网络输出$$\mu_\theta(x_t, t)$$预测$$\mu_t(x_t, x_0)$$，不如让网络输出$$\epsilon_\theta(x_t,t)$$预测$$\bar{\epsilon}_t$$。$$\mu_\theta(x_t, t)$$和$$\epsilon_\theta(x_t,t)$$关系如下

$$\mu_\theta(x_t, t)= \frac{1}{\sqrt{\alpha_t}} (x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t,t))$$

于是终于得到了最终的 Loss 表达式

$$\begin{aligned}L_{t-1} &\propto ||\bar{\epsilon}_t-\epsilon_\theta(x_t,t)||^2 \\&= ||\bar{\epsilon}_t-\epsilon_\theta(\sqrt{\bar{\alpha_t}}x_0 +\sqrt{1-\bar{\alpha_t}}\bar{\epsilon}_t,t)||^2\end{aligned}$$

这里$$\epsilon_\theta(\sqrt{\bar{\alpha_t}}x_0 +\sqrt{1-\bar{\alpha_t}}\bar{\epsilon}_t,t)$$是个U-Net网络，并且时间步t需要做 embedding 灌入网络

整体的训练和推理流程如下：

![训练                                                            推理](../../images/视觉多模态讲义（下）-image-45.png)

这里再完整梳理一下核心逻辑：

> * 首先建模 Encoder 过程为清晰图$$x_0$$到标准高斯分布$$x_T$$的扩散过程，相应的 Decoder 过程为标准高斯分布$$x_T$$到清晰图$$x_0$$的采样过程。<span style="color: rgb(100,37,208); background-color: inherit">最后需要的是 Decoder 过程</span>。
>
> * <span style="color: rgb(100,37,208); background-color: inherit">定义单调递增的序列</span>$$\beta_t$$<span style="color: rgb(100,37,208); background-color: inherit">，并定义</span>$$q(x_t|x_{t-1})$$<span style="color: rgb(100,37,208); background-color: inherit">的递推表达式</span>
>
> * 利用<span style="color: rgb(100,37,208); background-color: inherit">递推式可以求出 Encoder 的每一步</span>$$x_t$$<span style="color: rgb(100,37,208); background-color: inherit">的分布</span>$$q(x_t|x_0)$$
>
> * 利用<span style="color: rgb(100,37,208); background-color: inherit">贝叶斯公式可以求出分布</span>$$q(x_{t-1}|x_t, x_0)$$，并且满足$$\mathcal{N}(x_{t-1};\mu_t(x_t, x_0),\sigma_t^2(x_t, x_0))$$
>
> * 变分分析得出最终的优化 loss 目标为$$L_{t-1}=\text{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))$$
>
> * <span style="color: rgb(100,37,208); background-color: inherit">设置</span>$$p_\theta(x_{t-1}|x_t)$$<span style="color: rgb(100,37,208); background-color: inherit">为高斯分布，</span>并且满足$$\mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \beta_t)$$
>
> * <span style="color: rgb(100,37,208); background-color: inherit">带入</span>$$L_{t-1}$$<span style="color: rgb(100,37,208); background-color: inherit">求解，化简表达式</span>，最终发现网络预测噪声$$\epsilon_\theta(x_t,t)$$来逼近$$\bar{\epsilon}_t$$即可

---

[Previous](04-GAN--AE--Flow--Flow-based-model.md) | [Contents](../../README.md) | [Next](06-Diffusion-Model--Diffusion-基础.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-2.html#c=5)
