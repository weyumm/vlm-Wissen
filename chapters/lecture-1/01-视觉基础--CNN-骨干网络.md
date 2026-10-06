[Contents](../../README.md) | [Next](02-视觉基础--Transformer-骨干网络.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=1)

* **<span style="color: rgb(36,91,219); background-color: inherit">笔记标题样式及颜色含义</span>**

![](../../images/视觉多模态讲义（上）-image-12.png)

# 1. <span style="color: rgb(36,91,219); background-color: inherit">视觉基础</span>

## 1.1 <span style="color: rgb(36,91,219); background-color: inherit">CNN 骨干网络</span>

### 1.1.1 <span style="color: rgb(36,91,219); background-color: inherit">卷积与池化</span>

<span style="color: rgb(36,91,219); background-color: inherit">卷积神经网络 </span>CNN 是一种特殊类型的深度神经网络，其结构如右图，<span style="color: rgb(100,37,208); background-color: inherit">由卷积层，池化层，全连接层等各种类型的结构构成</span>。在图像处理等领域，与普通的深度神经网络相比，CNN拥有更好的处理效果。<span style="color: rgb(100,37,208); background-color: inherit">CNN 中有两层结构是其特有的：</span>**<span style="color: rgb(100,37,208); background-color: inherit">卷积</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">池化</span>**。

![](../../images/视觉多模态讲义（上）-image-11.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">卷积层</span>**

卷积层是 CNN 网络中独立的一层，用来进行一种称之为卷积操作，该操作的示意图右图：

**<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：原始图像用一&#x4E2A;**`8x8`**&#x7684;矩阵表示，对于这个矩阵，使用一&#x4E2A;**`3X3`**&#x7684;矩阵对其进行映射运算，图中的红色光柱标记了第一个九宫格的中心&#x70B9;**`6`**，映射的过程其实是取&#x4EE5;**`6`**&#x4E3A;中心&#x7684;**`3×3`**&#x56FE;像区域，将区域内每个元素&#x4E0E;**`3×3`**&#x5377;积核对应位置的元素相乘，再&#x5C06;**`9`**&#x4E2A;乘积求和，最后总得到的值&#x4E3A;**`-3`**。

只需要定义移动的步长，就可以遍历图像中的所有点，比如定义步长为`1`，遍历过程如下图所示

![](../../images/视觉多模态讲义（上）-image-13.png)

![](../../images/视觉多模态讲义（上）-image-10.png)

上述过程是基于无填充的情况，在有填充的情况下，对于图像边缘的点，会&#x7528;**`0`**&#x586B;充，如右图

遍历所有点可以得到一张新的图，这样的图称之为特征图，示例如下

![](../../images/视觉多模态讲义（上）-image.png)

![](../../images/视觉多模态讲义（上）-image-1.png)

所以卷积层是用来做特征提取的。

* **<span style="color: rgb(36,91,219); background-color: inherit">池化层</span>**

池化是一种下采样技术，本质是基于滑动窗口的思想，可以去除特征图中的冗余信息，降低特征图的维度。事先定义好窗口和步长，然后对原始图像进行分割，对于分割的子窗口，采取某种策略取一个样本点出来，常用的有两种策略

> 1. &#x53D6;**<span style="color: rgb(36,91,219); background-color: inherit">最大值</span>**，对应max pooling
>
> ✍️ 2. &#x53D6;**<span style="color: rgb(36,91,219); background-color: inherit">平均值</span>**，对应average pooling

**<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：定义窗口大小&#x4E3A;**`2X2`**, 步长&#x4E3A;**`2`**，两种策略的结果如右图

图像被分割成&#x4E86;**`4`**&#x4E2A;子窗口，然后每个子窗口内取最大值或者平均值，就构成了一幅新图像。新图像的维度比原始图像小，达到了降维的目的。

![](../../images/视觉多模态讲义（上）-image-2.png)

### 1.1.2 <span style="color: rgb(36,91,219); background-color: inherit">LeNet</span>

卷积神经网络 **<span style="color: rgb(216,57,49); background-color: inherit">CNN</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">C</span>**<span style="color: rgb(216,57,49); background-color: inherit">onvolutional </span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">eural </span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">etwork）</span>是深度学习领域中最具影响力的模型之一，而 **LeNet** 则是 CNN 的开山之作。这一节将详细介绍 LeNet。

LeNet 由 **Yann LeCun** 于 1998 年在 **AT\&T** 贝尔实验室提出，最初被应用于手写字符识别任务。<span style="color: rgb(100,37,208); background-color: inherit">这篇工作标志着卷积神经网络首次通过反向传播算法成功训练，并在实际应用中展现了卓越的性能</span>。LeCun 的研究不仅奠定了 CNN 的基础，还为后续的深度学习发展提供了重要的理论支持。

尽管当时的计算资源有限，但 LeNet 的设计理念却极具前瞻性。它通过引入卷积层和池化层的层次结构，实现了高效的特征提取与分类任务。这一创新使得 LeNet 在手写数字识别任务中表现优异，尤其是在 MNIST 数据集上的应用广受关注。

* **<span style="color: rgb(36,91,219); background-color: inherit">核心思想</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">LeNet-5</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的设计体现了卷积神经网络的三大核心思想：</span>**<span style="color: rgb(100,37,208); background-color: inherit">局部感受野</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 、</span>**<span style="color: rgb(100,37,208); background-color: inherit">权值共享</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">池化操作</span>**：

1. **<span style="color: rgb(36,91,219); background-color: inherit">局部感受野</span>**：局部感受野的思想来源于生物视觉系统的研究，尤其是猫的视觉神经元实验。在 LeNet-5 中，每个神经元只对输入图像的局部区域敏感，而不是对整个图像进行处理。这种设计显著减少了参数数量，并使网络能够专注于提取局部特征，例如边缘、角点等。

   > 🏖️ 1. **<span style="color: rgb(36,91,219); background-color: inherit">实现方式</span>**：通过卷积核（滤波器）对输入图像进行滑动窗口操作，提取局部特征。
   >
   > 2. **<span style="color: rgb(36,91,219); background-color: inherit">优点</span>**：降低了计算复杂度，增强了模型的可解释性。

2. **<span style="color: rgb(36,91,219); background-color: inherit">权值共享</span>**：权值共享是卷积神经网络的核心特性之一，它允许同一个卷积核在整个输入图像上重复使用。这种机制不仅减少了模型的参数数量，还提高了特征提取的一致性。

   > 🥛 1. **<span style="color: rgb(36,91,219); background-color: inherit">实现方式</span>**：卷积层中的每个卷积核在输入图像的不同位置上共享相同的权重和偏置。
   >
   > 2. **<span style="color: rgb(36,91,219); background-color: inherit">优点</span>**：大幅减少了模型的存储需求，同时增强了模型对平移不变性的鲁棒性。

3. **<span style="color: rgb(36,91,219); background-color: inherit">池化操作</span>**：池化操作通过对特征图进行下采样，降低其空间维度，从而减少计算量并增强模型的鲁棒性。LeNet-5 中采用的是平均池化 Average Pooling，但现在 CNN 更常用最大池化 Max Pooling 以保留更显著的特征。

   > 📌 1. **<span style="color: rgb(36,91,219); background-color: inherit">实现方式</span>**：将特征图划分为若干个 2×2 的区域，取每个区域的平均值或最大值作为输出。
   >
   > 2. **<span style="color: rgb(36,91,219); background-color: inherit">优点</span>**：减少了特征图的空间尺寸，缓解了过拟合问题。

* **<span style="color: rgb(36,91,219); background-color: inherit">网络结构</span>**

![](../../images/视觉多模态讲义（上）-image-3.png)

**LeNet-5** 的设计体现了卷积、池化和全连接层的有机结合。以下是对 **LeNet-5** 的网络结构详解：

> 🌟 1. **<span style="color: rgb(36,91,219); background-color: inherit">输入层</span>**：**LeNet-5** 的输入层接受大小&#x4E3A;**`32×32`**&#x7684;灰度图像。尽管 MNIST 数据集中的手写数字图像原始尺寸&#x4E3A;**`28×28`**，但在输入到网络之前，这些图像会被填充&#x5230;**`32×32`**，以便更好地适应卷积操作的需求。

> 2. **<span style="color: rgb(36,91,219); background-color: inherit">第一卷积块</span>**：**LeNet-5** 的第一个卷积块由一个卷积层和一个池化层组成。
>
>    🌰 1. **<span style="color: rgb(36,91,219); background-color: inherit">卷积层</span>**：第一卷积层使&#x7528;**`6`**&#x4E2A;**`5×5`**&#x7684;卷积核对输入图像进行卷积操作，生&#x6210;**`6`**&#x4E2A;特征图。每个卷积核通过滑动窗口的方式提取局部特征，例如边缘、角点等。由于输入图像是单通道的灰度图像，因此卷积核的深度&#x4E3A;**`1`**。
>
>       1. 输出特征图的尺寸&#x4E3A;**`28×28`**，这是通过公式$$(W−F+2P)/S+1$$计算得出的，其中$$W=32$$是输入尺寸，$$F=5$$是卷积核尺寸，$$P=0$$是填充，$$S=1$$是步幅。
>
>       2. 参数数量为$$5×5×1×6+6=156$$，包括卷积核权重和偏置项。
>
>    2. **<span style="color: rgb(36,91,219); background-color: inherit">池化层</span>**：池化层采用平均池化 Average Pooling，将每&#x4E2A;**`2×2`**&#x7684;区域映射为一个值，从而将特征图的空间尺寸减半。输出特征图的尺寸&#x4E3A;**`14×14`**。
>
>       1. 池化操作不仅减少了计算量，还增强了模型对输入图像平移和缩放的鲁棒性。

> 3. **<span style="color: rgb(36,91,219); background-color: inherit">第二卷积块</span>**：第二个卷积块进一步提取更高级的特征。
>
>    🍞 1. **<span style="color: rgb(36,91,219); background-color: inherit">卷积层</span>**：第二卷积层使&#x7528;**`16`**&#x4E2A;**`5×5`**&#x7684;卷积核对前一层&#x7684;**`6`**&#x4E2A;特征图进行卷积操作，生&#x6210;**`16`**&#x4E2A;新的特征图。为了减少参数数量，这里采用了部分连接的方式：并非所有输入特征图都与所有输出特征图相连。
>
>       1. 输出特征图的尺寸&#x4E3A;**`10×10`**，计算方式同上。
>
>       2. 参数数量为$$5×5×6×16+16=2,416$$。
>
>    2. **<span style="color: rgb(36,91,219); background-color: inherit">池化层</span>**：再次使用平均池化，将特征图的空间尺寸减半。输出特征图的尺寸&#x4E3A;**`5×5`**。

> 4. **<span style="color: rgb(36,91,219); background-color: inherit">全连接层</span>**：在经过两个卷积块后，LeNet-5 将高维特征映射到低维空间，并通过全连接层完成分类任务：
>
>    1. **<span style="color: rgb(36,91,219); background-color: inherit">展平操作</span>**：在进入全连接层之前，特征图被展平为一维向量。展平后的向量长度为$$16×5×5=400$$。
>
>    🎹 2. **<span style="color: rgb(36,91,219); background-color: inherit">第一全连接层</span>**：第一全连接层&#x6709;**`120`**&#x4E2A;神经元，用于整合卷积层提取的特征。参数数量为 $$400×120+120=48,120$$。
>
>    3. **<span style="color: rgb(36,91,219); background-color: inherit">第二全连接层</span>**：第二全连接层&#x6709;**`84`**&#x4E2A;神经元，进一步压缩特征表示。参数数量为$$120×84+84=10,164$$。

> 🎉 5. **<span style="color: rgb(36,91,219); background-color: inherit">输出层</span>**：输出层&#x6709;**`10`**&#x4E2A;神经元，对应于 **MNIST** 数据集中的 0 到 9 &#x5171;**`10`**&#x4E2A;类别。使用 **Softmax** 激活函数输出每个类别的概率分布。参数数量为$$84×10+10=850$$。

**<span style="color: rgb(36,91,219); background-color: inherit">激活函数</span>**：**LeNet-5** 使用非线性激活函数增强模型的表达能力。早期版本中通常采用 **Sigmoid** 或 **Tanh** 函数，但现在实现中更倾向于使用 **ReLU** 以加速训练并缓解梯度消失问题。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

以下是一个简单的基于 PyTorch 的 LeNet实现：

```python
LeNet-5
Pythonimport torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # 第一层卷积
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)  # 第一层池化
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 第二层卷积
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)  # 第二层池化
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 全连接层1
        self.fc2 = nn.Linear(120, 84)  # 全连接层2
        self.fc3 = nn.Linear(84, 10)  # 输出层
```


**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

LeNet 作为卷积神经网络的奠基之作，不仅在技术上开创了先河，还在实际应用中取得了显著成效。它的设计理念至今仍然影响着深度学习的发展方向。通过学习 LeNet，可以更好地理解卷积神经网络的工作原理，并为进一步探索复杂的深度学习模型打下坚实的基础。

### 1.1.3 <span style="color: rgb(36,91,219); background-color: inherit">AlexNet</span>

在深度学习的发展历程中，2012 年是一个重要的年份。这一年，**<span style="color: rgb(100,37,208); background-color: inherit">AlexNet</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的出现彻底改变了计算机视觉领域，并为深度学习的复兴奠定了坚实的基础</span>。AlexNet 是由 Alex Krizhevsky、Ilya Sutskever 以及他们的导师 Geoffrey Hinton 共同设计并提出的。<span style="color: rgb(46,161,33); background-color: inherit">这一模型在 ImageNet 大规模视觉识别挑战赛 ILSVRC 中取得了突破性的成绩</span>，将错误率从此前&#x7684;**`26.2%`**&#x5927;幅降低&#x81F3;**`15.3%`**，震惊了整个学术界。

* **<span style="color: rgb(36,91,219); background-color: inherit">网络架构</span>**

**AlexNet** 的架构可以看作是 **LeNet** 的扩展版本，但其规模和复杂性远远超过了当时的其他模型。**AlexNet** 包&#x542B;**`5`**&#x4E2A;卷积层、**`3`**&#x4E2A;全连接层，以及一些关键的设计创新，使其成为深度学习历史上的经典之作。

**<span style="color: rgb(36,91,219); background-color: inherit">整体结构</span>**

![](../../images/视觉多模态讲义（上）-image-4.png)

**AlexNet** 的整体结构包括： &#x20;

> * **<span style="color: rgb(36,91,219); background-color: inherit">输入层</span>**：接&#x6536;**`224×224`**&#x5927;小的 RGB 图像作为输入。为了增强数据多样性，输入图像会经过随机裁剪和水平翻转等数据增强操作。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">卷积层</span>**：&#x5171;**`5`**&#x4E2A;卷积层，其中前两个卷积层后接局部响应归一&#x5316;**`LRN`**&#x548C;最大池&#x5316;**`Max-Pooling`**，后续卷积层则直接堆叠。这些卷积层通过不同的滤波器逐步提取图像特征。
>
> 🚅 * **<span style="color: rgb(36,91,219); background-color: inherit">全连接层</span>**：包&#x542B;**`3`**&#x4E2A;全连接层，其中前两个全连接层使&#x7528;**`ReLU`**&#x6FC0;活函数，并结&#x5408;**`Dropout`**&#x8FDB;行正则化。最后一个全连接层输&#x51FA;**`1000`**&#x4E2A;类别的概率分布，对应于ImageNet数据集的分类任务。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">输出层</span>**：通&#x8FC7;**`Softmax`**&#x51FD;数计算每个类别的概率值。

**<span style="color: rgb(36,91,219); background-color: inherit">卷积层的设计</span>**

**AlexNet** 的卷积层设计体现了当时的技术创新，以下是各层的具体参数和功能：

> 🌟 * **<span style="color: rgb(36,91,219); background-color: inherit">C1 层</span>**：第一个卷积层使&#x7528;**`96`**&#x4E2A;大小&#x4E3A;**`11×11`**、步长&#x4E3A;**`4`**&#x7684;卷积核，生&#x6210;**`96`**&#x4E2A;特征图。由于卷积核较大，**<span style="color: rgb(100,37,208); background-color: inherit">C1</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 层主要捕捉图像中的低级特征，例如边缘和纹理</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">C2 层</span>**：第二个卷积层使&#x7528;**`256个5×5`**&#x7684;卷积核，步长&#x4E3A;**`1`**，并<span style="color: rgb(100,37,208); background-color: inherit">采用零填充 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Zero Padding</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 以保持特征图的尺寸不变。</span>**<span style="color: rgb(100,37,208); background-color: inherit">C2</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 层进一步组合低级特征，形成更复杂的模式</span>。
>
> * **<span style="color: rgb(36,91,219); background-color: inherit">C3-C5 层</span>**：后续的三个卷积层分别使&#x7528;**`3×3`**&#x7684;小卷积核，逐步提取更高层次的特征，例如物体的部分结构或整体形状。这种<span style="color: rgb(46,161,33); background-color: inherit">逐层抽象的方式使得 </span>**<span style="color: rgb(46,161,33); background-color: inherit">AlexNet</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 能够有效地处理复杂的视觉任务</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">激活函数</span>**

在 **AlexNet** 之前，<span style="color: rgb(216,57,49); background-color: inherit">大多数神经网络使用的是</span>**`Sigmoid`**<span style="color: rgb(216,57,49); background-color: inherit">或</span>**`Tanh`**<span style="color: rgb(216,57,49); background-color: inherit">激活函数。然而，这些函数在深层网络中容易导致梯度消失问题</span>。**<span style="color: rgb(46,161,33); background-color: inherit">AlexNet</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 首次引入了 </span>**<span style="color: rgb(216,57,49); background-color: inherit">ReLU</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Re</span>**<span style="color: rgb(216,57,49); background-color: inherit">ctified </span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">inear </span>**<span style="color: rgb(216,57,49); background-color: inherit">U</span>**<span style="color: rgb(216,57,49); background-color: inherit">nit）</span> <span style="color: rgb(46,161,33); background-color: inherit">作为激活函数，显著加速了训练过程，并缓解了梯度消失的问题</span>。 &#x20;

**ReLU** 的公式为$$f(x) = \max(0, x)$$，<span style="color: rgb(46,161,33); background-color: inherit">它通过将负值直接置零的方式避免了梯度饱和现象，从而提高了网络的收敛速度。此外，</span>**<span style="color: rgb(46,161,33); background-color: inherit">ReLU</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 的计算效率较高，因为它只涉及简单的比较和赋值操作</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">局部响应归一化 LRN</span>**

AlexNet 在前两个卷积层之后引入了局部响应归一化 **<span style="color: rgb(216,57,49); background-color: inherit">LRN</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">L</span>**<span style="color: rgb(216,57,49); background-color: inherit">ocal </span>**<span style="color: rgb(216,57,49); background-color: inherit">R</span>**<span style="color: rgb(216,57,49); background-color: inherit">esponse </span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">ormalization）</span>。这一技术的灵感来源于生物学中的侧抑制现象，旨在<span style="color: rgb(46,161,33); background-color: inherit">增强对高激活值的响应，同时抑制邻近区域的低激活值</span>。 &#x20;

具体来说，**LRN** 通过对同一位置不同通道的激活值进行归一化，使得某些通道的响应更加突出。尽管 **LRN** 在后来的研究中逐渐被批归一化 Batch Normalization 等技术取代，但在当时，它确实为提升模型性能提供了一定的帮助。

**<span style="color: rgb(36,91,219); background-color: inherit">池化层</span>**

**AlexNet** 在前两个卷积层后使用了最大池化 Max-Pooling 操作，窗口大小&#x4E3A;**`3×3`**，步长&#x4E3A;**`2`**。<span style="color: rgb(46,161,33); background-color: inherit">最大池化通过对局部区域取最大值的方式，减少了特征图的尺寸，同时保留了最重要的信息</span>。 &#x20;

池化层的主要作用是<span style="color: rgb(46,161,33); background-color: inherit">降低特征图的空间维度，从而减少计算量，并增强模型的鲁棒性</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">池化操作可以忽略图像中小的平移变化，使模型更具泛化能力</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">全连接层与 Dropout</span>**

**AlexNet** 的全连接层包含两个隐藏层和一个输出层，其中前两个全连接层各&#x6709;**`4096`**&#x4E2A;神经元。为了防止过拟合，**AlexNet** 在全连接层中引入了 **Dropout** 技术。 &#x20;

**<span style="color: rgb(100,37,208); background-color: inherit">Dropout</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的核心思想是在训练过程中随机丢弃一部分神经元，使得每次训练时模型只能依赖部分神经元完成任务</span>。这种方法有效地<span style="color: rgb(46,161,33); background-color: inherit">避免了模型对特定神经元的过度依赖，从而增强了模型的泛化能力</span>。在 AlexNet 中，Dropout 的概率通常设置&#x4E3A;**`0.5`**。

**<span style="color: rgb(36,91,219); background-color: inherit">多 GPU 训练</span>**

由于当时的硬件限制，单个 GPU 无法支持如此庞大的模型训练。**<span style="color: rgb(100,37,208); background-color: inherit">AlexNet</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 巧妙地将网络分为两路，分别在两个 GPU 上并行计算，并在特定层进行通信和合并</span>。这种设计不仅<span style="color: rgb(46,161,33); background-color: inherit">解决了硬件瓶颈，还启发了后续分布式训练的研究</span>。 &#x20;

具体而言，<span style="color: rgb(100,37,208); background-color: inherit">网络的前几层分别在两个 GPU 上独立运行，而在第三卷积层后才进行跨 GPU 的信息交换</span>。这种分而治之的策略极大地提高了训练效率，同时也展示了如何利用有限资源实现复杂模型的能力。

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()

        # 定义卷积层部分
        self.features = nn.Sequential(
            # 第一层卷积：输入通道3（RGB图像），输出通道96，卷积核大小11x11，步长4，无填充
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # 使用ReLU激活函数
```


**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**AlexNet** 的成功不仅仅体现在其卓越的性能上，更在于它推动了深度学习的普及和应用。在 ImageNet 竞赛中，**AlexNet** 的表现远超传统方法，证明了深度卷积神经网络在处理复杂视觉任务中的巨大潜力。此后，基于深度学习的计算机视觉技术迅速崛起，**AlexNet** 也被广泛应用于图像分类、目标检测、语义分割等多个领域。

**AlexNet** 标志着深度学习时代的到来。它的成功不仅归功于强大的硬件支持和大规模数据集 ImageNet，还得益于一系列创新的设计理念。<span style="color: rgb(100,37,208); background-color: inherit">从 ReLU 激活函数到 Dropout 正则化，从多 GPU 训练到深层网络结构，</span>**<span style="color: rgb(100,37,208); background-color: inherit">AlexNet</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 为后续的深度学习研究提供了宝贵的经验和启示</span>。

### 1.1.4 <span style="color: rgb(36,91,219); background-color: inherit">VGG</span>

在深度学习的发展历程中，**VGG**（Very Deep Convolutional Networks for Large-Scale Image Recognition）是另一座重要的里程碑。这一模型由牛津大学的视觉几何组 **<span style="color: rgb(216,57,49); background-color: inherit">VGG</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">V</span>**<span style="color: rgb(216,57,49); background-color: inherit">isual </span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">eometry </span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">roup）</span>于 2014 年提出，它以其简洁的设计和卓越的性能迅速成为当时最受欢迎的卷积神经网络之一。**VGG** 不仅推动了图像分类和目标检测领域的发展，还为后续的研究者提供了宝贵的架构设计经验。

在 **VGG** 之前，卷积神经网络的设计往往倾向于使用较大的卷积核，&#x5982;**`7×7`**&#x6216;**`11×11`**，以捕捉更大的感受野。然而，随着计算资源的限制以及对更深层次网络的需求增加，研究者们开始思考如何在保持高性能的同时简化网络结构。**<span style="color: rgb(100,37,208); background-color: inherit">VGG</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的核心思想是通过堆叠多个小尺寸的</span>**`3×3`**<span style="color: rgb(100,37,208); background-color: inherit">卷积核来构建更深的网络</span>。这种设计利用了小卷积核的优势，<span style="color: rgb(46,161,33); background-color: inherit">即能够减少参数数量、增强非线性表达能力，并逐步扩大感受野</span>。

此外，**VGG** 的研究团队希望通过实验验证一个假设：**<span style="color: rgb(100,37,208); background-color: inherit">加深网络深度是否能持续提升模型性能？</span>**&#x8FD9;一问题的答案在 **VGG** 的工作中得到了明确的肯定回应，同时也揭示了深层网络在实际应用中的潜力。

* **<span style="color: rgb(36,91,219); background-color: inherit">网络结构</span>**

VGG的主要贡献在于其系统化的网络设计方法。作者提出了多个不同深度的网络配置，其中最著名的两个版本分别&#x662F;**`VGG-16`**&#x548C;**`VGG-19`**，分别包&#x542B;**`16`**&#x5C42;&#x548C;**`19`**&#x5C42;可训练的权重层。这些网络的共同特点包括：

1. **<span style="color: rgb(36,91,219); background-color: inherit">小卷积核的堆叠</span>**：所有的卷积层均采&#x7528;**`3×3`**&#x7684;小型滤波器，步幅&#x4E3A;**`1`**，填充&#x4E3A;**`1`**，从而<span style="color: rgb(100,37,208); background-color: inherit">确保输入和输出的空间维度保持一致。通过连续堆叠多个这样的卷积层，模型能够在不显著增加参数量的情况下扩展感受野</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">两个</span>**`3×3`**<span style="color: rgb(220,155,4); background-color: inherit">的卷积层相当于一个</span>**`5×5`**<span style="color: rgb(220,155,4); background-color: inherit">的感受野，参数量分别是</span>$$2\times3\times3=18$$<span style="color: rgb(220,155,4); background-color: inherit">和</span>$$5\times5=25$$<span style="color: rgb(220,155,4); background-color: inherit">；而三个</span>**`3×3`**<span style="color: rgb(220,155,4); background-color: inherit">的卷积层则等效于一个</span>**`7×7`**<span style="color: rgb(220,155,4); background-color: inherit">的感受野，参数量分别是</span>$$3\times3\times3=27$$<span style="color: rgb(220,155,4); background-color: inherit">和</span>$$7\times7=49$$。这种设计<span style="color: rgb(100,37,208); background-color: inherit">不仅减少了参数数量，还通过多次激活函数的非线性变换增强了特征的表达能力</span>。

![](../../images/视觉多模态讲义（上）-image-8.png)

2. **<span style="color: rgb(36,91,219); background-color: inherit">规则的网络结构</span>**：每个卷积块通常包&#x542B;**`2`**&#x5230;**`4`**&#x4E2A;卷积层，随后接一个最大池化层，用于下采样特征图的空间分辨率。具体来说，**VGG** 的网络结构可以分为五个主要的卷积块，每个块的输出通道数逐渐增加：&#x4ECE;**`64`**&#x5230;**`512`**。最大池化层的核大小&#x4E3A;**`2×2`**，步幅&#x4E3A;**`2`**，这使得特征图的空间尺寸在每个块后减半。<span style="color: rgb(46,161,33); background-color: inherit">这种模块化的设计使得网络易于实现和扩展</span>。

3. **<span style="color: rgb(36,91,219); background-color: inherit">通道数的增长</span>**：随着网络的加深，每个卷积块的输出通道数逐渐增加，&#x4ECE;**`64`**&#x5230;**`512`**，<span style="color: rgb(46,161,33); background-color: inherit">以便提取更高层次的抽象特征</span>。第一个卷积块的输出通道数&#x4E3A;**`64`**，第二个&#x4E3A;**`128`**，第三个和第四个&#x4E3A;**`256`**，第五个&#x4E3A;**`512`**。这种逐层增加通道数的设计策略，<span style="color: rgb(46,161,33); background-color: inherit">使得网络能够逐步捕捉从低级到高级的特征表示</span>。

4) **<span style="color: rgb(36,91,219); background-color: inherit">全连接层的使用</span>**：在最后的卷积层之后，**VGG** 引入了三个全连接层，其中前两个包&#x542B;**`4096`**&#x4E2A;神经元，最后一个用于分类任务的输出。尽管全连接层增加了模型的参数量，但它们在当时被认为是<span style="color: rgb(100,37,208); background-color: inherit">提取全局信息的关键组件</span>。为了防止过拟合，全连接层之间通常会加&#x5165;**`Dropout`**&#x6B63;则化，丢弃率&#x4E3A;**`0.5`**。

5) **<span style="color: rgb(36,91,219); background-color: inherit">ReLU激活函数</span>**：所有隐藏层均采&#x7528;**`ReLU`**&#x4F5C;为激活函数，这不仅加速了训练过程，还有效缓解了梯度消失问题。相比于传统的 Sigmoid 或 Tanh 函数，**<span style="color: rgb(46,161,33); background-color: inherit">ReLU</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 在深层网络中表现出了更强的鲁棒性和效率</span>。

6. **<span style="color: rgb(36,91,219); background-color: inherit">Softmax输出层</span>**：最后一层是一&#x4E2A;**`Softmax`**&#x5C42;，用于多类别分类任务。通过 **Softmax** 函数，模型<span style="color: rgb(46,161,33); background-color: inherit">将输出转化为概率分布，便于评估每个类别的置信度</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">参数计算与复杂性分析</span>**

**VGG-16** 的参数量高达&#x7EA6;**`1.38`**&#x4EBF;，而 **VGG-19** 的参数量更是达到了&#x7EA6;**`1.44`**&#x4EBF;。这些<span style="color: rgb(100,37,208); background-color: inherit">参数主要集中在全连接层中，尤其是最后三个全连接层占据了大部分的计算开销</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">在</span>**`VGG-16`**<span style="color: rgb(220,155,4); background-color: inherit">中，仅最后三个全连接层就包含了超过</span>**`1`**<span style="color: rgb(220,155,4); background-color: inherit">亿个参数</span>。相比之下，卷积层的参数量相对较少，但由于其需要处理高分辨率的特征图，计算成本仍然很高。

为了进一步说明这一点，我们可以通过公式计算每一层的参数量。

1. **对于卷积层**，其参数量由以下公式决定：

$$\text{参数量} = (\text{卷积核宽度} \times \text{卷积核高度} \times \text{输入通道数} + 1) \times \text{输出通道数}$$

例如，<span style="color: rgb(220,155,4); background-color: inherit">第一层卷积层的参数量为：</span>$$(3 \times 3 \times 3 + 1) \times 64 = 1792$$

* **对于全连接层**，其参数量则由输入神经元数和输出神经元数决定：

$$\text{参数量} = \text{输入神经元数} \times \text{输出神经元数} + \text{偏置项}$$

例如，<span style="color: rgb(220,155,4); background-color: inherit">第一个全连接层的参数量为：</span>$$(7 \times 7 \times 512 + 1) \times 4096 = 102764544$$

由此可见，<span style="color: rgb(100,37,208); background-color: inherit">全连接层是 </span>**<span style="color: rgb(100,37,208); background-color: inherit">VGG</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 模型参数量的主要来源</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">计算复杂度分析</span>**

除了参数量，VGG  的计算复杂度也值得关注。计算复杂度通常用浮点运算次数 FLOPs 来衡量，主要包括卷积操作和全连接操作的计算量。

1. **卷积层的计算复杂度**卷积层的计算复杂度由以下公式决定：

   $$\text{FLOPs} = \text{输出高度} \times \text{输出宽度} \times \text{输出通道数} \times (\text{卷积核宽度} \times \text{卷积核高度} \times \text{输入通道数} + 1)$$

   例如，<span style="color: rgb(220,155,4); background-color: inherit">第一层卷积层的FLOPs为：</span>$$(224 \times 224) \times 64 \times (3 \times 3 \times 3 + 1) = 118M$$

2. **全连接层的计算复杂度**全连接层的计算复杂度由以下公式决定：

   $$\text{FLOPs} = \text{输入神经元数} \times \text{输出神经元数} + \text{偏置项}$$

   例如，<span style="color: rgb(220,155,4); background-color: inherit">第一个全连接层的FLOPs为：</span>$$(7 \times 7 \times 512) \times 4096 = 102764544$$

* **<span style="color: rgb(36,91,219); background-color: inherit">VGG的局限性与改进方向</span>**

尽管VGG在理论和实践上都取得了巨大的成功，但它也存在一些明显的局限性：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">计算成本高昂</span>**：**VGG** 的深层结构和大量参数使其<span style="color: rgb(216,57,49); background-color: inherit">在训练和推理过程中需要消耗大量的计算资源</span>。例如，**<span style="color: rgb(220,155,4); background-color: inherit">VGG-16</span>**<span style="color: rgb(220,155,4); background-color: inherit"> 的参数量超过 1 亿</span>，这在当时的硬件条件下是一个不小的挑战。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">梯度消失问题</span>**：随着网络深度的增加，<span style="color: rgb(216,57,49); background-color: inherit">传统的激活函数，如 ReLU，可能会导致梯度消失问题，从而阻碍模型的进一步优化</span>。这一问题在后来的 ResNet 中通过引入残差连接得到了有效解决。
>
> 🍰 3) **<span style="color: rgb(36,91,219); background-color: inherit">全连接层的冗余性</span>**：<span style="color: rgb(216,57,49); background-color: inherit">全连接层占据了大部分的参数量，但其作用却相对有限</span>。后续的研究表明，通过全局平均池化 Global Average Pooling 可以替代全连接层，从而显著降低模型的复杂度。
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">内存瓶颈</span>**：在训练过程中，**VGG** 的深层结构可能导致显存不足的问题。尤其是在早期的硬件环境下，这种限制更加明显\[\[7]]。

为了克服这些局限性，后续的研究提出了多种改进方案，例如<span style="color: rgb(220,155,4); background-color: inherit">使用更高效的卷积操作：</span>**<span style="color: rgb(220,155,4); background-color: inherit">深度可分离卷积</span>**<span style="color: rgb(220,155,4); background-color: inherit">、引入</span>**<span style="color: rgb(220,155,4); background-color: inherit">跳跃连接</span>**<span style="color: rgb(220,155,4); background-color: inherit">以及设计</span>**<span style="color: rgb(220,155,4); background-color: inherit">轻量级网络架构</span>**<span style="color: rgb(220,155,4); background-color: inherit">等</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

这里以构建 VGG-16为例

```python
import torch
import torch.nn as nn

# 定义VGG块
def make_vgg_block(in_channels, out_channels, num_conv_layers, use_batchnorm=False):
    """
    构建VGG块，包含多个卷积层和一个最大池化层。
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :param num_conv_layers: 卷积层数量
    :param use_batchnorm: 是否使用BatchNorm
    :return: 一个VGG块（nn.Sequential）
    """
```


**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**VGG** &#x5728;**`ILSVRC-2014`**&#x7ADE;赛中取得了优异的成绩，其 Top-5 错误率仅&#x4E3A;**`7.3%`**，仅次于同年提出&#x7684;**`GoogLeNet`**。虽然 **VGG** 的计算复杂度较高，但它凭借简单的架构和强大的性能赢得了广泛的认可。更重要的是，**<span style="color: rgb(100,37,208); background-color: inherit">VGG</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的成功证明了</span>**<span style="color: rgb(100,37,208); background-color: inherit">增加网络深度</span>**<span style="color: rgb(100,37,208); background-color: inherit">可以显著提升模型的表现力，这一发现为后续 ResNet 等更深层次网络的发展奠定了基础</span>。

除了在图像分类领域的成就，**VGG** 还在其他任务中展现了其价值。例如，<span style="color: rgb(220,155,4); background-color: inherit">在风格迁移技术中，预训练的 VGG 模型被广泛用于提取图像的内容和风格特征</span>。此外，由于其模块化的设计，**<span style="color: rgb(100,37,208); background-color: inherit">VGG</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的中间层特征可以直接迁移到新的任务上，极大地促进了迁移学习的发展</span>。

**VGG** 以其简洁而深刻的设计理念，为深度卷积神经网络的研究开辟了新的道路。<span style="color: rgb(100,37,208); background-color: inherit">它不仅展示了深度网络在图像识别任务中的巨大潜力，还启发了无数后续工作</span>。正如科学研究中的许多经典案例一样，**VGG** 的意义不仅在于其本身，更在于它为整个领域带来的深远影响。

### 1.1.5 <span style="color: rgb(36,91,219); background-color: inherit">GoogLeNet</span>

**GoogLeNet** 是由 Google 团队于 2014 年提出的另一个具有里程碑意义的网络，并在当年的 ImageNet 大规模视觉识别挑战赛 ILSVRC 中一举夺魁。**<span style="color: rgb(46,161,33); background-color: inherit">GoogLeNet</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 不仅以其卓越的性能闻名，更因其创新的设计理念和高效的计算架构成为卷积神经网络领域的经典之作</span>。

随着深度学习技术的快速发展，Google 的作者团队逐渐意识到，增加网络深度可以显著提升模型的表达能力。然而，简单的堆叠卷积层会导致参数量急剧膨胀，从而引发计算资源消耗过大和过拟合问题。为了解决这些问题，GoogLeNet 的设计者们提出了一个核心思想：<span style="color: rgb(100,37,208); background-color: inherit">通过稀疏连接和多尺度特征提取来优化网络结构，同时保持较高的计算效率</span>。

传统的卷积神经网络通常采用逐层堆叠的方式，每一层都对输入特征图进行密集的卷积操作。这种设计虽然简单直观，但随着网络层数的增加，计算复杂度和参数量会迅速增长。例如，<span style="color: rgb(220,155,4); background-color: inherit">VGG 网络通过堆叠多个</span>**`3×3`**<span style="color: rgb(220,155,4); background-color: inherit">卷积层实现了高性能，但其参数量高达</span>**`1.4`**<span style="color: rgb(220,155,4); background-color: inherit">亿，这对硬件资源提出了极高的要求</span>。**GoogLeNet** 的目标是设计一种既能保持高性能，又能大幅降低参数量的网络结构。

GoogLeNet的名字也颇具深意。它不仅是对 LeNet 的致敬，还象征着深度学习领域的传承与发展。这一命名方式体现了研究者对历史的尊重以及对未来探索的期待。

* **<span style="color: rgb(36,91,219); background-color: inherit">Inception 模块</span>**

![](../../images/视觉多模态讲义（上）-image-5.png)

**GoogLeNet** 的最大亮点在于其引入&#x4E86;**`Inception`**&#x6A21;块，如上图所示，这是一种全新的卷积网络设计范式。Inception 模块的核心思想是通过并行计算多个不同尺度的卷积操作，从而捕获图像中的多层次特征。具体来说，每个Inception模块包含以下几种操作：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">1×1 卷积</span>**：<span style="color: rgb(100,37,208); background-color: inherit">用于降维和减少计算量</span>。1×1卷积的主要作用是对输入特征图的通道数进行压缩，从而降低后续卷积操作的计算复杂度。例如，假设输入特征图的尺寸为$$H \times W \times C$$，其中$$C$$表示通道数，那&#x4E48;**`1×1`**&#x5377;积可以将通道数从$$C$$降到$$C'$$，从而显著减少后续卷积操作的计算量。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">3×3 卷积</span>**：<span style="color: rgb(100,37,208); background-color: inherit">捕捉局部特征</span>。**`3×3`**&#x5377;积核是卷积神经网络中最常用的卷积核尺寸之一，能够有效提取局部空间信息。相比于更大的卷积核，**3×3** 卷积核的感受野较小，但计算成本较低，适合提取细粒度特征。
>
> 🌟 3) **<span style="color: rgb(36,91,219); background-color: inherit">5×5 卷积</span>**：<span style="color: rgb(100,37,208); background-color: inherit">捕获更大范围的空间信息</span>。相比于 **3×3&#x20;**&#x5377;积，**`5×5`**&#x5377;积核的感受野更大，适合提取全局特征。然而，**5×5** 卷积的计算成本较高，因此 GoogLeNet 采用了瓶颈层 Bottleneck Layer 的设计，先通过 **1×1** 卷积降低输入通道数，再进行 **5×5** 卷积操作。
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">3×3 最大池化</span>**：<span style="color: rgb(100,37,208); background-color: inherit">提取全局特征</span>。最大池化操作通过对特征图进行下采样，保留最显著的特征，同时降低特征图的分辨率。这种操作不仅减少了计算量，还能增强模型对输入变化的鲁棒性。

这些操作的结果会被拼接在一起，形成一个多通道的输出特征图。这种设计既保留了网络的稀疏性，又利用了密集矩阵的高计算性能。

为了进一步优化 Inception 模块的性能，GoogLeNet还采用了瓶颈层 Bottleneck Layer 的设计。<span style="color: rgb(46,161,33); background-color: inherit">通过在大尺寸卷积操作前加入 </span>**<span style="color: rgb(46,161,33); background-color: inherit">1×1</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 卷积层，模型能够有效降低输入特征图的维度，从而大幅减少计算开销</span>。例如，<span style="color: rgb(220,155,4); background-color: inherit">在处理</span>**`5×5`**<span style="color: rgb(220,155,4); background-color: inherit">卷积时，先通过</span>**`1×1`**<span style="color: rgb(220,155,4); background-color: inherit">卷积将输入通道数从</span>**`256`**<span style="color: rgb(220,155,4); background-color: inherit">降到</span>**`64`**<span style="color: rgb(220,155,4); background-color: inherit">，然后再进行</span>**`5×5`**<span style="color: rgb(220,155,4); background-color: inherit">卷积操作</span>。这种设计使得计算量从原本的$$256 \times 5 \times 5 = 6400$$降低到$$64 \times 5 \times 5 + 256 \times 1 \times 1 = 1600 + 256 = 1856$$，显著提升了计算效率。

此外，**<span style="color: rgb(100,37,208); background-color: inherit">Inception</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 模块的设计还体现了</span>**<span style="color: rgb(100,37,208); background-color: inherit">分而治之</span>**<span style="color: rgb(100,37,208); background-color: inherit">的思想</span>。通过并行计算不同尺度的卷积操作，<span style="color: rgb(46,161,33); background-color: inherit">模型能够在不增加参数数量的情况下提高网络的深度和表征能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">辅助分类器：提升训练稳定性</span>**

![](../../images/视觉多模态讲义（上）-image-6.png)

除了 **Inception** 模块外，**GoogLeNet** 还引入&#x4E86;**<span style="color: rgb(216,57,49); background-color: inherit">辅助分类器</span>**<span style="color: rgb(216,57,49); background-color: inherit"> Auxiliary Classifier</span> 的设计，如上图分支所示。<span style="color: rgb(216,57,49); background-color: inherit">在深层网络中，梯度消失问题常常导致训练困难</span>。为了解决这一问题，**<span style="color: rgb(46,161,33); background-color: inherit">GoogLeNet </span>**<span style="color: rgb(46,161,33); background-color: inherit">在网络的中间层添加了两个辅助分类器。这些分类器能够提供额外的监督信号，帮助网络更好地收敛</span>。

辅助分类器的具体实现如下：

> 1. 在网络的中间层，通常是倒数第二个和第三个 Inception 模块之后，分别添加一个辅助分类器
>
> 🍰 2. 每个辅助分类器包含一&#x4E2A;**`5×5`**&#x7684;平均池化层、一&#x4E2A;**`1×1`**&#x5377;积层用于降维、两个全连接层以及一&#x4E2A;**`Softmax`**&#x5206;类器
>
> 3) 在训练阶段，<span style="color: rgb(100,37,208); background-color: inherit">辅助分类器的损失函数与主分类器的损失函数按一定权重相加，共同指导网络的优化过程</span>

辅助分类器仅在训练阶段使用，在测试阶段会被移除。这种设计<span style="color: rgb(46,161,33); background-color: inherit">不仅提高了模型的训练效率，还增强了模型的泛化能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">网络结构</span>**

**GoogLeNet** 的整体架构由多个 **Inception** 模块串联而成，如下表所示，总共&#x6709;**`22`**&#x5C42;，<span style="color: rgb(100,37,208); background-color: inherit">包括</span>**<span style="color: rgb(100,37,208); background-color: inherit">卷积层</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">池化层</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">全连接层</span>**。<span style="color: rgb(46,161,33); background-color: inherit">相比之前的 VGG 网络，</span>**<span style="color: rgb(46,161,33); background-color: inherit">GoogLeNet</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 的参数量显著减少，仅为</span>**`500`**<span style="color: rgb(46,161,33); background-color: inherit">万左右，而 VGG 的参数量则高达</span>**`1.4`**<span style="color: rgb(46,161,33); background-color: inherit">亿</span>。这种参数量的大幅降低使得 **GoogLeNet** 在实际应用中更加高效。

![](../../images/视觉多模态讲义（上）-image-7.png)

以下是 GoogLeNet 的详细结构设计：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">输入层</span>**：接&#x53D7;**`224×224`**&#x5927;小的RGB图像作为输入
>
> 📚 2. **<span style="color: rgb(36,91,219); background-color: inherit">卷积层 1</span>**：包含一&#x4E2A;**`7×7`**&#x7684;卷积层，步长&#x4E3A;**`2`**，输出通道数&#x4E3A;**`64`**，后接一&#x4E2A;**`3×3`**&#x7684;最大池化层，步长&#x4E3A;**`2`**
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">卷积层 2</span>**：包含一&#x4E2A;**`1×1`**&#x5377;积层和一&#x4E2A;**`3×3`**&#x7684;卷积层，输出通道数&#x4E3A;**`192`**，用于进一步提取低级特征
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">Inception 模块</span>**：包&#x542B;**`9`**&#x4E2A; **Inception** 模块，分为三个主要部分：
>
>    * <span style="color: rgb(36,91,219); background-color: inherit">第一部分</span>：**`2`**&#x4E2A; **Inception** 模块，逐步增加网络深度
>
>    * <span style="color: rgb(36,91,219); background-color: inherit">第二部分</span>：**`5`**&#x4E2A; **Inception** 模块，结合辅助分类器
>
>    * <span style="color: rgb(36,91,219); background-color: inherit">第三部分</span>：**`2`**&#x4E2A; **Inception** 模块，进一步提升特征表达能力
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">全局平均池化层</span>**：取代传统的全连接层，通过全局平均池化操作将特征图转换为固定长度的向量
>
> 6. **<span style="color: rgb(36,91,219); background-color: inherit">输出层</span>**：包含一个 **Softmax** 分类器，用于生成最终的分类结果

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

以下是基于 PyTorch 实现 **GoogLeNet&#x20;**&#x7684;完整代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义Inception模块
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        """
```


**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

在 2014 年的 ImageNet 竞赛中，**<span style="color: rgb(46,161,33); background-color: inherit">GoogLeNet</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 以</span>**`6.67%`**<span style="color: rgb(46,161,33); background-color: inherit">的 Top-5 错误率夺得冠军，远超第二名的表现。这一成绩充分证明了 </span>**<span style="color: rgb(46,161,33); background-color: inherit">GoogLeNet </span>**<span style="color: rgb(46,161,33); background-color: inherit">在图像分类任务中的优越性</span>。此外，**GoogLeNet&#x20;**&#x7684;成功也启发了后续许多研究工作，例如 ResNet 等。

**GoogLeNet&#x20;**&#x7684;提出标志着卷积神经网络设计进入了一个新的阶段。<span style="color: rgb(46,161,33); background-color: inherit">通过引入 </span>**<span style="color: rgb(46,161,33); background-color: inherit">Inception </span>**<span style="color: rgb(46,161,33); background-color: inherit">模块、辅助分类器以及全局平均池化等创新技术，</span>**<span style="color: rgb(46,161,33); background-color: inherit">GoogLeNet </span>**<span style="color: rgb(46,161,33); background-color: inherit">不仅实现了更高的计算效率，还显著提升了模型的性能</span>。它的设计理念深刻影响了后续的研究，并为深度学习领域的发展奠定了坚实的基础。

### 1.1.6 <span style="color: rgb(36,91,219); background-color: inherit">ResNet</span>

在深度学习的发展历程中，神经网络的深度一直是研究的核心问题之一。理论上，更深的网络能够捕捉更复杂的特征，从而提升模型性能。然而，实际训练中，<span style="color: rgb(216,57,49); background-color: inherit">随着网络深度的增加，梯度消失和梯度爆炸等问题逐渐显现，导致模型难以收敛</span>。此外，当网络达到一定深度时，甚至会<span style="color: rgb(216,57,49); background-color: inherit">出现退化现象——即更深的网络反而表现得比浅层网络更差</span>。如右图所示

![](../../images/视觉多模态讲义（上）-image-9.png)

为了解决这些问题，凯明在 2015 年提出了深度残差网络 **<span style="color: rgb(216,57,49); background-color: inherit">ResNet</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Res</span>**<span style="color: rgb(216,57,49); background-color: inherit">idual </span>**<span style="color: rgb(216,57,49); background-color: inherit">Net</span>**<span style="color: rgb(216,57,49); background-color: inherit">work）</span>，并在 ImageNet 竞赛中取得了突破性成果。

* **<span style="color: rgb(36,91,219); background-color: inherit">核心思想：残差学习</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">ResNet</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 的核心思想是引入</span>**<span style="color: rgb(100,37,208); background-color: inherit">残差学习框架</span>**<span style="color: rgb(100,37,208); background-color: inherit">，通过构建</span>**<span style="color: rgb(100,37,208); background-color: inherit">残差块</span>**<span style="color: rgb(100,37,208); background-color: inherit">来缓解深层网络的训练困难</span>。传统的神经网络试图直接拟合目标函数$$H(x)$$，而 **<span style="color: rgb(100,37,208); background-color: inherit">ResNet</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 则通过引入一个恒等映射，将目标变为拟合残差</span>$$F(x) = H(x) - x$$<span style="color: rgb(100,37,208); background-color: inherit">。也就是说网络不再需要直接学习输出，而是学习输入与输出之间的差异，即残差</span>。这种设计使得即使在网络较深的情况下，信息也能更容易地通过网络传播。

为了实现这一目标，**ResNet&#x20;**&#x5728;每个残差块中加入&#x4E86;**<span style="color: rgb(216,57,49); background-color: inherit">跳跃连接</span>**<span style="color: rgb(216,57,49); background-color: inherit"> skip connection</span>，也称为捷径连接。跳跃连接将输入直接加到输出上，形成如下形式的表达式：

$$y = F(x, \{W_i\}) + x$$

其中，$$x$$是输入，$$F(x, \{W_i\})$$是残差函数，$$y$$是输出。这种结构不仅简化了优化过程，还有效缓解了梯度消失问题。

* **<span style="color: rgb(36,91,219); background-color: inherit">网络结构</span>**

**ResNet** 的基本单元是残差块，其结构如右图所示。<span style="color: rgb(100,37,208); background-color: inherit">每个残差块通常由两到三个卷积层组成，并通过跳跃连接将输入直接传递到输出</span>。如果输入和输出的维度不一致，则可以通过线性投影（通常&#x662F;**`1x1`**&#x5377;积）来调整维度。

> 🌟 **ResNet&#x20;**&#x7684;设计非常灵活，可以根据任务需求堆叠不同数量的残差块。例如，经典&#x7684;**`ResNet-18`**、**`ResNet-34`**、**`ResNet-50`**、**`ResNet-101`**&#x548C;**`ResNet-152`**&#x5206;别包&#x542B;**`18`**&#x5C42;、**`34`**&#x5C42;、**`50`**&#x5C42;、**`101`**&#x5C42;&#x548C;**`152`**&#x5C42;网络。尽管这些网络的深度差异显著，但它们都基于相同的残差学习框架。

![](../../images/视觉多模态讲义（上）-image-23.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">输入</span>**：ResNet 的输入部分通常由一&#x4E2A;**`7x7`**&#x7684;卷积层组成，步长&#x4E3A;**`2`**，填充&#x4E3A;**`3`**，用于对输入图像进行初步特征提取。随后是一个最大池化层 Max Pooling，进一步压缩特征图的空间尺寸。这一部分的设计可以快速降低输入数据的空间分辨率，同时保留重要的低级特征。

2. **<span style="color: rgb(36,91,219); background-color: inherit">卷积</span>**：中间卷积部分是 ResNet 的核心，由多个阶段组成，每个阶段包含若干个残差块。根据网络深度的不同，这些阶段的数量和每个阶段中的残差块数量也会有所不同。例如，**<span style="color: rgb(220,155,4); background-color: inherit">ResNet-18</span>**<span style="color: rgb(220,155,4); background-color: inherit"> 和 </span>**<span style="color: rgb(220,155,4); background-color: inherit">ResNet-34</span>**<span style="color: rgb(220,155,4); background-color: inherit"> 使用的是</span>**<span style="color: rgb(220,155,4); background-color: inherit">基本残差块</span>**<span style="color: rgb(220,155,4); background-color: inherit">，而 ResNet-50 及更深的网络则使用</span>**<span style="color: rgb(220,155,4); background-color: inherit">瓶颈残差块</span>**。

   > ⛱️ **<span style="color: rgb(36,91,219); background-color: inherit">基本残差块</span>**：基本残差块由两&#x4E2A;**`3x3`**&#x7684;卷积层组成，每个卷积层后接批量归一化 **BN** 和 **ReLU** 激活函数。跳跃连接直接将输入添加到输出上。如果输入和输出的通道数不一致，则通&#x8FC7;**`1x1`**&#x5377;积调整维度。

   ![](../../images/视觉多模态讲义（上）-image-28.png)

   > 🍞 **<span style="color: rgb(36,91,219); background-color: inherit">瓶颈残差块</span>**：瓶颈残差块是 ResNet-50 及更深网络的核心组件。它由三个卷积层组成：第一&#x4E2A;**`1x1`**&#x5377;积层用于降维，减少计算量；第二&#x4E2A;**`3x3`**&#x5377;积层用于提取特征；第三&#x4E2A;**`1x1`**&#x5377;积层用于升维，恢复通道数。跳跃连接同样直接将输入添加到输出上，必要时通&#x8FC7;**`1x1`**&#x5377;积调整维度

   ![](../../images/视觉多模态讲义（上）-image-22.png)

   不同深度的 ResNet 在中间卷积部分的设计上有所区别。以下是几种经典 ResNet 的结构特点：

   > * **<span style="color: rgb(36,91,219); background-color: inherit">ResNet-18 和 ResNet-34</span>**：使用基本残差块，分别包&#x542B;**`18`**&#x5C42;&#x548C;**`34`**&#x5C42;网络。这两个网络适用于较小的数据集或计算资源有限的场景。
   >
   > * **<span style="color: rgb(36,91,219); background-color: inherit">ResNet-50、ResNet-101 和 ResNet-152</span>**：使用瓶颈残差块，分别包&#x542B;**`50`**&#x5C42;、**`101`**&#x5C42;&#x548C;**`152`**&#x5C42;网络。这些网络在大规模数据集上表现出色，但计算成本较高

3. **<span style="color: rgb(36,91,219); background-color: inherit">跳跃连接</span>**：跳跃连接是ResNet的关键创新点之一。对于输入和输出维度一致的情况，跳跃连接可以直接将输入添加到输出上。而对于维度不一致的情况，如通道数变化或空间分辨率变化，可以通过以下两种方式实现跳跃连接：

   > **<span style="color: rgb(36,91,219); background-color: inherit">零填充</span>**：在通道维度上用零填充输入，使其与输出的维度一致。
   >
   > 🎨 **<span style="color: rgb(36,91,219); background-color: inherit">1x1卷积</span>**：通&#x8FC7;**`1x1`**&#x5377;积调整输入的通道数和空间分辨率，使其与输出匹配

   第二种方法更为常用，因为它能够在调整维度的同时引入额外的非线性变换，从而增强模型的表达能力。

4. **<span style="color: rgb(36,91,219); background-color: inherit">输出</span>**：ResNet 的输出部分通常由全局平均池化 Global Average Pooling 和全连接层组成。全局平均池化将每个特征图的空间尺寸压缩&#x4E3A;**`1x1`**，从而大幅减少参数数量。最后，全连接层将特征映射到类别空间，完成分类任务。

ResNet 的5种规格如右图所示。

**<span style="color: rgb(222,120,2); background-color: inherit">例</span>**：ResNet-50具体结构如下

**`Stage 1`**：1&#x4E2A;**`7x7`**&#x5377;积层和最大池化层

**`Stage 2`**：3个瓶颈残差块

**`Stage 3`**：4个瓶颈残差块

**`Stage 4`**：6个瓶颈残差块

**`Stage 5`**：3个瓶颈残差块

![](../../images/视觉多模态讲义（上）-image-27.png)

每个阶段的输出特征图的<span style="color: rgb(100,37,208); background-color: inherit">空间分辨率逐级减半，而通道数逐级加倍</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">训练技巧</span>**

除了创新的网络结构，**ResNet** 的成功还得益于一些关键的训练技巧。首先，<span style="color: rgb(100,37,208); background-color: inherit">批量归一化 </span>**<span style="color: rgb(100,37,208); background-color: inherit">BN</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 被广泛应用于每一层卷积之后</span>，以加速训练并提高模型的泛化能力。其次，**<span style="color: rgb(100,37,208); background-color: inherit">ReLU</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 激活函数作为非线性变换的主要手段</span>，进一步提升了网络的表达能力。最后，<span style="color: rgb(100,37,208); background-color: inherit">全局平均池化 Global Average Pooling 替代了全连接层</span>，从而减少了模型参数的数量，降低了过拟合的风险。

此外，**ResNet** 的训练过程中还采用了数据增强技术，如<span style="color: rgb(100,37,208); background-color: inherit">随机裁剪、水平翻转</span>等，以提高模型的鲁棒性。这些技术的结合使得 **ResNet** 能够在大规模数据集，如 ImageNet 上取得优异的表现。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

以下是一个使用 PyTorch 实现的 **ResNet-50** 代码：

这展示了一个基本的残差块。通过堆叠多个这样的模块，可以搭建出完整的 ResNet 模型。

**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

自 **ResNet** 提出以来，在计算机视觉领域产生了深远的影响。它不仅在图像分类任务中表现出色，还在目标检测、语义分割等任务中展现了强大的潜力。ResNet 的成功证明了深度神经网络的潜力，并启发了许多后续工作，&#x5982;**`DenseNet`**、**`EfficientNet`**&#x7B49;。

在实际应用中，**ResNet&#x20;**&#x7684;预训练模型被广泛使用。通过迁移学习，研究人员可以利用在大规模数据集，如ImageNet 上预训练的 **ResNet** 模型，快速适应新的任务或领域。例如，<span style="color: rgb(220,155,4); background-color: inherit">在医学图像分析、自动驾驶等领域，ResNet 的变体已经被成功应用于解决实际问题</span>。

**ResNet** 通过引入残差学习框架和跳跃连接，成功解决了深层神经网络的训练难题，成为深度学习领域的里程碑之一。它的设计理念不仅推动了计算机视觉技术的进步，也为其他领域的研究提供了宝贵的借鉴。

### 1.1.7 <span style="color: rgb(36,91,219); background-color: inherit">DenseNet</span>

**<span style="color: rgb(216,57,49); background-color: inherit">DenseNet</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">Dense</span>**<span style="color: rgb(216,57,49); background-color: inherit">ly Connected Convolutional </span>**<span style="color: rgb(216,57,49); background-color: inherit">Net</span>**<span style="color: rgb(216,57,49); background-color: inherit">works）</span>是一种具有开创性的深度学习架构，其通过引入密集连接机制，显著提升了特征传播效率和参数利用率。

* **<span style="color: rgb(36,91,219); background-color: inherit">核心思想</span>**

**DenseNet&#x20;**&#x7684;核心创新在于密集连接 dense connections 的设计。<span style="color: rgb(216,57,49); background-color: inherit">在传统的卷积神经网络中，信息在层与层之间逐级传递，每一层只能利用前一层的输出</span>。然而，在 **DenseNet&#x20;**&#x4E2D;，<span style="color: rgb(46,161,33); background-color: inherit">每一层不仅接收其前一层的输出，还直接连接到所有之前层的输出。这种设计使得每一层的输入是由前面所有层的特征图拼接而成的，从而实现了特征的高效复用</span>。

具体来说，假设第$$l$$层的输入为$$[x_0, x_1, ..., x_{l-1}]$$，其中$$x_i$$表示第$$i$$层的输出特征图，那么第$$l$$层的输出可以表示为：

![](../../images/视觉多模态讲义（上）-image-26.png)

$$x_l = H_l([x_0, x_1, ..., x_{l-1}])$$

其中$$H_l$$是一个非线性变换，通常由<span style="color: rgb(100,37,208); background-color: inherit">批量归一化 Batch Normalization、ReLU 激活函数和</span>**`3×3`**<span style="color: rgb(100,37,208); background-color: inherit">卷积组成</span>。这种密集连接机制<span style="color: rgb(46,161,33); background-color: inherit">不仅增强了特征传播，还缓解了梯度消失问题，因为梯度可以直接通过短连接传播到浅层</span>。

此外，由于每一层都可以直接访问前面所有层的特征图，**<span style="color: rgb(46,161,33); background-color: inherit">DenseNet</span>**<span style="color: rgb(46,161,33); background-color: inherit"> 在训练过程中能够更好地保留低层特征，这对于需要多层次上下文信息的任务尤为重要</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">网络结构</span>**

![](../../images/视觉多模态讲义（上）-image-14.png)

**<span style="color: rgb(100,37,208); background-color: inherit">DenseNet </span>**<span style="color: rgb(100,37,208); background-color: inherit">的网络结构由多个</span>**<span style="color: rgb(100,37,208); background-color: inherit">密集块</span>**<span style="color: rgb(100,37,208); background-color: inherit"> Dense Block 和</span>**<span style="color: rgb(100,37,208); background-color: inherit">过渡层</span>**<span style="color: rgb(100,37,208); background-color: inherit"> Transition Layer 交替组成</span>。每个密集块内部实现了层与层之间的密集连接，而过渡层则用于控制模型的复杂度。

1. **<span style="color: rgb(36,91,219); background-color: inherit">密集块 Dense Block</span>**：<span style="color: rgb(100,37,208); background-color: inherit">在密集块中，每一层的输入是由前面所有层的输出特征图拼接而成的</span>。这种拼接操作<span style="color: rgb(46,161,33); background-color: inherit">不仅保留了原始信息，还允许网络动态地选择性利用这些特征，从而提高了模型的表达能力</span>。为了控制模型的规模，**DenseNet&#x20;**&#x5F15;入了一个超参数——增长率$$k$$，它表示每一层输出的特征图数量。较小的增长率可以显著减少模型的参数量，同时保持较高的性能。例如，**`DenseNet-121`**<span style="color: rgb(220,155,4); background-color: inherit">的增长率为</span>**`32`**<span style="color: rgb(220,155,4); background-color: inherit">，那么每一层只增加</span>$$32$$<span style="color: rgb(220,155,4); background-color: inherit">个新的特征图，而不会显著增加计算成本；</span>**`DenseNet-201`**<span style="color: rgb(220,155,4); background-color: inherit">的增长率为</span>**`48`**<span style="color: rgb(220,155,4); background-color: inherit">，每一层只增加</span>$$48$$<span style="color: rgb(220,155,4); background-color: inherit">个新的特征图</span>。<span style="color: rgb(100,37,208); background-color: inherit">密集块中的每一层通常采用</span>**<span style="color: rgb(100,37,208); background-color: inherit">批量归一化</span>**<span style="color: rgb(100,37,208); background-color: inherit"> Batch Normalization、ReLU </span>**<span style="color: rgb(100,37,208); background-color: inherit">激活函数</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`3×3`<span style="color: rgb(100,37,208); background-color: inherit">卷积</span>**<span style="color: rgb(100,37,208); background-color: inherit">的组合结构</span>。这种简单的层设计<span style="color: rgb(46,161,33); background-color: inherit">进一步简化了网络结构，同时确保了每一层都能专注于提取新的特征</span>。

2. **<span style="color: rgb(36,91,219); background-color: inherit">过渡层 Transition Layer</span>**：<span style="color: rgb(100,37,208); background-color: inherit">过渡层位于两个密集块之间，用于降维操作，控制过渡层中特征图数量的缩减比例</span>。过渡层通常由一&#x4E2A;**`1×1`**&#x5377;积和一&#x4E2A;**`2×2`**&#x5E73;均池化组成。**<span style="color: rgb(100,37,208); background-color: inherit">1×1 </span>**<span style="color: rgb(100,37,208); background-color: inherit">卷积用于减少特征图的数量，而平均池化则用于降低特征图的空间尺寸</span>。通过设置压缩因子$$θ$$，可以进一步控制特征图数量的缩减比例。通过设置$$θ<1$$，可以在不显著影响性能的情况下进一步压缩模型规模。例如，<span style="color: rgb(220,155,4); background-color: inherit">当</span>$$θ=0.5$$<span style="color: rgb(220,155,4); background-color: inherit">时，过渡层会将特征图数量减半</span>。<span style="color: rgb(46,161,33); background-color: inherit">过渡层的作用不仅仅是降维，它还在一定程度上起到了正则化的效果，有助于防止过拟合</span>。

**DenseNet&#x20;**&#x7F51;络也具有多种参数配置，如下表所示：

![](../../images/视觉多模态讲义（上）-image-15.png)

* **<span style="color: rgb(36,91,219); background-color: inherit">技术优势</span>**

**DenseNet&#x20;**&#x7684;设计带来了多方面的技术优势：

> ⛱️ 1. **<span style="color: rgb(36,91,219); background-color: inherit">参数效率高</span>**：在 ImageNet 分类任务上，**<span style="color: rgb(46,161,33); background-color: inherit">DenseNet </span>**<span style="color: rgb(46,161,33); background-color: inherit">在达到与 ResNet 相当准确率的同时，所需的参数量不到 ResNet 的一半</span>。这得益于密集连接机制对特征的充分利用，避免了冗余计算。
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">隐式正则化效果</span>**：通过拼接不同层的特征图，**<span style="color: rgb(100,37,208); background-color: inherit">DenseNet </span>**<span style="color: rgb(100,37,208); background-color: inherit">在训练过程中自然地引入了一种隐式的正则化效果</span>。这种正则化有助于提高模型的稳定性，尤其是在小样本数据集上表现尤为突出。
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">梯度流动更顺畅</span>**：密集连接的设计使得梯度可以直接传播到浅层，<span style="color: rgb(46,161,33); background-color: inherit">有效缓解了深度网络训练中的梯度消失问题</span>。此外，每一层都可以获得来自损失函数的更多监督信息，类似于 **<span style="color: rgb(216,57,49); background-color: inherit">DSN</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">D</span>**<span style="color: rgb(216,57,49); background-color: inherit">eeply </span>**<span style="color: rgb(216,57,49); background-color: inherit">S</span>**<span style="color: rgb(216,57,49); background-color: inherit">upervised </span>**<span style="color: rgb(216,57,49); background-color: inherit">N</span>**<span style="color: rgb(216,57,49); background-color: inherit">ets）</span>的设计。
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">特征多样性增强</span>**：每一层的输入是由前面所有层的特征图拼接而成的，因此<span style="color: rgb(46,161,33); background-color: inherit">每一层都能接触到更丰富的特征组合</span>。这种多样性增强了模型的表达能力，使其能够更好地捕捉复杂的模式。
>
> 5. **<span style="color: rgb(36,91,219); background-color: inherit">抗过拟合能力强</span>**：由于密集连接机制能够更好地利用特征，并且<span style="color: rgb(46,161,33); background-color: inherit">通过过渡层实现了特征图数量的压缩，</span>**<span style="color: rgb(46,161,33); background-color: inherit">DenseNet </span>**<span style="color: rgb(46,161,33); background-color: inherit">在小样本场景下表现出更强的泛化能力</span>。

* **<span style="color: rgb(36,91,219); background-color: inherit">性能表现</span>**

**DenseNet&#x20;**&#x5728;多个基准数据集上展现了出色的性能。例如，<span style="color: rgb(220,155,4); background-color: inherit">在 ImageNet 分类任务中，</span>**`DenseNet-121`**<span style="color: rgb(220,155,4); background-color: inherit">以</span>**`121`**<span style="color: rgb(220,155,4); background-color: inherit">层的深度达到了与</span>**`ResNet-152`**<span style="color: rgb(220,155,4); background-color: inherit">相当的准确率，但参数量仅为后者的一半</span>。此外，在 CIFAR-10 和 CIFAR-100 数据集上，**DenseNet&#x20;**&#x540C;样表现出色，证明了其<span style="color: rgb(46,161,33); background-color: inherit">在小样本场景下的强大泛化能力</span>。除此之外，**<span style="color: rgb(46,161,33); background-color: inherit">DenseNet </span>**<span style="color: rgb(46,161,33); background-color: inherit">在相同的数据集上不仅参数量更少，而且训练时间更短</span>。

**DenseNet&#x20;**&#x7684;独特结构使其在多种应用场景中表现出色。例如，<span style="color: rgb(220,155,4); background-color: inherit">在迁移学习中，</span>**<span style="color: rgb(220,155,4); background-color: inherit">DenseNet </span>**<span style="color: rgb(220,155,4); background-color: inherit">的预训练模型可以通过微调快速适应新的任务</span>。此外，**<span style="color: rgb(100,37,208); background-color: inherit">DenseNet </span>**<span style="color: rgb(100,37,208); background-color: inherit">在需要多层次上下文信息的任务中也展现了强大的能力，如图像分割、目标检测，因为密集连接能够更好地保留和传递上下文信息</span>。

在实际应用中，**DenseNet&#x20;**&#x8FD8;被广泛用于无人机高分辨率影像处理领域。例如，<span style="color: rgb(220,155,4); background-color: inherit">基于</span>**`DenseNet121_BL`**<span style="color: rgb(220,155,4); background-color: inherit">和</span>**`DenseNet169_BL`**<span style="color: rgb(220,155,4); background-color: inherit">网络模型的树种分类研究在验证集上取得了</span>**`89.17%`**<span style="color: rgb(220,155,4); background-color: inherit">的分类正确率</span>。

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

以下是一个基于 PyTorch 的 **DenseNet&#x20;**&#x5B9E;现：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
```


**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**<span style="color: rgb(100,37,208); background-color: inherit">DenseNet </span>**<span style="color: rgb(100,37,208); background-color: inherit">通过引入密集连接机制，开创了一种全新的网络设计范式。它以高效的特征复用、优异的参数效率和强大的泛化能力，成为深度学习领域的重要网络结构</span>。无论是理论研究还是实际应用，**DenseNet&#x20;**&#x90FD;为我们提供了宝贵的启示，展示了如何通过巧妙的设计实现性能与效率的双赢。

### 1.1.8 <span style="color: rgb(36,91,219); background-color: inherit">SENet</span>

深度学习领域中，卷积神经网络作为计算机视觉任务的核心工具，其设计初衷是通过局部感受野提取空间和通道维度的特征。然而，<span style="color: rgb(216,57,49); background-color: inherit">在 </span>**<span style="color: rgb(216,57,49); background-color: inherit">SENet </span>**<span style="color: rgb(216,57,49); background-color: inherit">提出之前，研究者们更多关注于如何优化空间维度的特征提取，而对通道维度的建模能力却鲜有系统性的探索</span>。**<span style="color: rgb(216,57,49); background-color: inherit">SENet</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">S</span>**<span style="color: rgb(216,57,49); background-color: inherit">queeze-and-</span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">xcitation </span>**<span style="color: rgb(216,57,49); background-color: inherit">Net</span>**<span style="color: rgb(216,57,49); background-color: inherit">works）</span>的出现填补了这一空白，它通过引入一种全新的模块——**<span style="color: rgb(216,57,49); background-color: inherit">SE</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">S</span>**<span style="color: rgb(216,57,49); background-color: inherit">queeze-and-</span>**<span style="color: rgb(216,57,49); background-color: inherit">E</span>**<span style="color: rgb(216,57,49); background-color: inherit">xcitation）</span>块，为通道间的关系建模提供了强有力的工具。

* **<span style="color: rgb(36,91,219); background-color: inherit">核心思想</span>**

**SENet&#x20;**&#x7684;核心思想是<span style="color: rgb(100,37,208); background-color: inherit">通过显式地建模通道间的依赖关系，实现对特征图的自适应重新校准</span>。在<span style="color: rgb(216,57,49); background-color: inherit">传统的卷积操作中，每个通道的特征图被视为独立的单元，忽略了它们之间的潜在关联</span>。**SENet&#x20;**&#x5219;通过引入 **SE&#x20;**&#x5757;，动态地调整每个通道的重要性，从而增强网络对关键特征的关注能力。这种机制可以看作是一种通道注意力机制，它让网络能够根据输入数据自适应地分配资源，提升模型的表示能力。

具体而言，**<span style="color: rgb(100,37,208); background-color: inherit">SE </span>**<span style="color: rgb(100,37,208); background-color: inherit">块的工作流程分为三个步骤：</span>**`Squeeze`<span style="color: rgb(100,37,208); background-color: inherit">压缩</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**`Excitation`<span style="color: rgb(100,37,208); background-color: inherit">激励</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**`Scale`<span style="color: rgb(100,37,208); background-color: inherit">缩放</span>**。 &#x20;

![](../../images/视觉多模态讲义（上）-image-21.png)

1. **<span style="color: rgb(36,91,219); background-color: inherit">Squeeze 压缩</span>**：在这一步骤中，**<span style="color: rgb(100,37,208); background-color: inherit">SE </span>**<span style="color: rgb(100,37,208); background-color: inherit">块通过对每个通道的特征图进行全局平均池化 </span>**<span style="color: rgb(216,57,49); background-color: inherit">GAP</span>**<span style="color: rgb(216,57,49); background-color: inherit">（</span>**<span style="color: rgb(216,57,49); background-color: inherit">G</span>**<span style="color: rgb(216,57,49); background-color: inherit">lobal </span>**<span style="color: rgb(216,57,49); background-color: inherit">A</span>**<span style="color: rgb(216,57,49); background-color: inherit">verage </span>**<span style="color: rgb(216,57,49); background-color: inherit">P</span>**<span style="color: rgb(216,57,49); background-color: inherit">ooling）</span> <span style="color: rgb(100,37,208); background-color: inherit">，将空间维度的信息压缩为一个标量，得到每个通道的全局描述</span>。全局平均池化的公式如下：

$$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i,j)$$

其中，$$z_c$$表示第$$c$$个通道的压缩值，$$x_c(i,j)$$是第$$c$$个通道在位置$$(i,j)$$的特征值，$$H$$和$$W$$分别是特征图的高度和宽度。这种操作不仅有效地捕捉了全局上下文信息，还避免了引入过多的参数。

2. **<span style="color: rgb(36,91,219); background-color: inherit">Excitation 激励</span>**：在这一步骤中，**<span style="color: rgb(100,37,208); background-color: inherit">SE </span>**<span style="color: rgb(100,37,208); background-color: inherit">块利用两层全连接网络（或等价的卷积操作）生成通道权重，捕捉通道间的非线性依赖关系</span>。第一层通过降维减少通道数，第二层通过升维恢复原始通道数。具体公式如下：

$$s = \sigma(W_2 \delta(W_1 z))$$

其中，$$z$$是经过 **Squeeze&#x20;**&#x6B65;骤得到的通道描述向量，$$W_1$$和$$W_2$$分别是降维和升维的权重矩阵，$$\delta$$是激活函数，如 ReLU，$$\sigma$$是 Sigmoid 函数。通过这种瓶颈结构，**SE&#x20;**&#x5757;能够在保持轻量化的同时，捕捉到通道间的复杂依赖关系 。

3. **<span style="color: rgb(36,91,219); background-color: inherit">Scale 缩放</span>**：最后，**<span style="color: rgb(100,37,208); background-color: inherit">SE </span>**<span style="color: rgb(100,37,208); background-color: inherit">块将生成的通道权重</span>$$s$$<span style="color: rgb(100,37,208); background-color: inherit">与原始特征图相乘，实现对通道特征的重新校准</span>。公式如下：

$$\hat{x}_c = s_c \cdot x_c$$

其中，$$\hat{x}_c$$是重新校准后的特征图，$$s_c$$是第$$c$$个通道的权重，$$x_c$$是原始特征图。通过这种方式，**SE&#x20;**&#x5757;能够增强重要通道的特征表达，同时抑制不重要的通道。

* **<span style="color: rgb(36,91,219); background-color: inherit">网络结构</span>**

**SENet** 的一个重要特点是其模块化的特性，使得 **SE&#x20;**&#x5757;可以无缝嵌入到现有的卷积神经网络架构中。以下&#x662F;**`SE-Inception Module`**&#x548C;**`SE-ResNet Module`**&#x7684;具体构造。

**<span style="color: rgb(36,91,219); background-color: inherit">SE-Inception Module</span>**

**Inception&#x20;**&#x6A21;块核心思想是通过并行地使用不同尺度的卷积核，&#x5982;**`1×1`**、**`3×3`**、**`5×5`**，以及最大池化操作来捕捉多尺度的特征信息。**<span style="color: rgb(100,37,208); background-color: inherit">SE-Inception Module</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 在标准 Inception 模块的基础上引入了 </span>**<span style="color: rgb(100,37,208); background-color: inherit">SE</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 块</span>，具体构造如下：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">原始 Inception 模块</span>**&#x7531;四个分支组成：
>
>    * 一&#x4E2A;**`1×1`**&#x5377;积分支
>
>    * 一&#x4E2A;**`3×3`**&#x5377;积分支，前&#x63A5;**`1×1`**&#x5377;积降维
>
>    * 一&#x4E2A;**`5×5`**&#x5377;积分支，前&#x63A5;**`1×1`**&#x5377;积降维
>
>    * 一&#x4E2A;**`3×3`**&#x6700;大池化分支，后&#x63A5;**`1×1`**&#x5377;积升维
>
> ❤️ 2. **<span style="color: rgb(36,91,219); background-color: inherit">嵌入 SE 块</span>**：在 **Inception&#x20;**&#x6A21;块的输出特征图上添加 **SE&#x20;**&#x5757;，通过对每个通道进行全局平均池化（**`Squeeze`**）、两层全连接网络生成通道权重（**`Excitation`**），最后通过缩放操作（**`Scale`**）重新校准特征图

![](../../images/视觉多模态讲义（上）-image-25.png)

这种设计使得 **SE-Inception Module** 能够<span style="color: rgb(46,161,33); background-color: inherit">在保持原有结构优势的同时，增强通道间的依赖关系建模能力</span>。

**<span style="color: rgb(36,91,219); background-color: inherit">SE-ResNet Module</span>**

**ResNet&#x20;**&#x662F;深度学习领域的重要突破，其核心是通过跳跃连接解决了深层网络中的梯度消失问题。**SE-ResNet Module&#x20;**&#x5728; **ResNet&#x20;**&#x7684;基础上引入了SE块，具体构造如下：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">原始 ResNet 模块</span>**&#x7531;三个卷积层组成：
>
>    * 第一层&#x4E3A;**`1×1`**&#x5377;积，用于降维
>
>    * 第二层&#x4E3A;**`3×3`**&#x5377;积，用于提取空间特征
>
>    * 第三层&#x4E3A;**`1×1`**&#x5377;积，用于升维
>
>    最终输出通过跳跃连接与输入相加，形成残差结构。
>
> 📌 2. **<span style="color: rgb(36,91,219); background-color: inherit">嵌入 SE 块</span>**：在 ResNet 模块的输出特征图上添加 **SE&#x20;**&#x5757;，<span style="color: rgb(100,37,208); background-color: inherit">通过 </span>**<span style="color: rgb(100,37,208); background-color: inherit">Squeeze-Excitation-Scale</span>**<span style="color: rgb(100,37,208); background-color: inherit"> 三步操作对通道特征进行重新校准</span>。具体而言，**SE** 块被放置在残差连接之后，即：
>
>    $$y=F(x,{W_i})+x$$
>
>    其中$$F(x,{W_i})$$表示经过 **SE&#x20;**&#x5757;重新校准后的特征图

![](../../images/视觉多模态讲义（上）-image-24.png)

**<span style="color: rgb(36,91,219); background-color: inherit">结构设计</span>**

为了更直观地理解 **SENet** 对经典网络的改进效果，这里&#x4EE5;**`ResNet-50`**、**`SE-ResNet-50`**&#x548C;**`SE-ResNeXt-50`**&#x4E3A;例，&#x4ECE;**<span style="color: rgb(100,37,208); background-color: inherit">参数量</span>**<span style="color: rgb(100,37,208); background-color: inherit">、</span>**<span style="color: rgb(100,37,208); background-color: inherit">计算复杂度</span>**<span style="color: rgb(100,37,208); background-color: inherit">和</span>**<span style="color: rgb(100,37,208); background-color: inherit">性能</span>**<span style="color: rgb(100,37,208); background-color: inherit">三个方面进行对比</span>。

![](../../images/视觉多模态讲义（上）-image-16.png)

**<span style="color: rgb(222,120,2); background-color: inherit">ResNet-50</span>**

**ResNet-50** 是一个经典的深度残差网络，包&#x542B;**`50`**&#x5C42;卷积层，分&#x4E3A;**`4`**&#x4E2A;阶段，每个阶段通过重复堆叠基本残差模块构建。

* 参数量：&#x7EA6;**`25.6M`**

* 计算量（FLOPs）：&#x7EA6;**`4.1G`**

* 性能：在 ImageNet 数据集上的 Top-1 准确率&#x4E3A;**`76.1%`**&#x20;



**<span style="color: rgb(222,120,2); background-color: inherit">SE-ResNet-50</span>**

**SE-ResNet-50** 在 **ResNet-50** 的基础上，在每个残差模块中嵌入了 **SE** 块。

* 参数量：增加至&#x7EA6;**`28.1M`**，增加了&#x7EA6;**`10%`**

* 计算量：仅增加了不&#x5230;**`1%`**，约&#x4E3A;**`4.14G`**

* 性能：在 ImageNet 数据集上的 Top-1 准确率达&#x5230;**`76.9%`**，相比 ResNet-50 提高&#x4E86;**`0.8%`**

**<span style="color: rgb(222,120,2); background-color: inherit">SE-ResNeXt-50</span>**

**ResNeXt** 是一种<span style="color: rgb(100,37,208); background-color: inherit">基于分组卷积的改进版 </span>**<span style="color: rgb(100,37,208); background-color: inherit">ResNet</span>**<span style="color: rgb(100,37,208); background-color: inherit">，其核心思想是通过基数 cardinality 来扩展网络宽度</span>，从而提高模型的表示能力。

* 参数量：&#x7EA6;**`25.0M`**

* 计算量：&#x7EA6;**`4.2G`**

* 性能：在 ImageNet 数据集上的 Top-1 准确率达&#x5230;**`77.5%`**，相比ResNeXt-50提高&#x4E86;**`0.7%`**



* **<span style="color: rgb(36,91,219); background-color: inherit">技术要点</span>**

尽管 **SENet&#x20;**&#x5F15;入了额外的计算步骤，但其设计非常注重效率。**SE&#x20;**&#x5757;的计算开销相对较小，尤其在大规模数据集和复杂任务中，性能提升远超额外计算成本。以下是 **SENet&#x20;**&#x5728;技术实现上的几个关键点：

> 1. **<span style="color: rgb(36,91,219); background-color: inherit">全局平均池化的巧妙应用</span>**：全局平均池化操作不仅有效地压缩了空间信息，还避免了引入过多的参数。这使得<span style="color: rgb(100,37,208); background-color: inherit"> </span>**<span style="color: rgb(100,37,208); background-color: inherit">SE </span>**<span style="color: rgb(100,37,208); background-color: inherit">块能够在保持轻量化的同时，捕捉到全局上下文信息</span>
>
> ❤️ 2. **<span style="color: rgb(36,91,219); background-color: inherit">瓶颈结构的设计</span>**：在 **Excitation&#x20;**&#x9636;段，**SE&#x20;**&#x5757;采用了瓶颈结构 Bottleneck Structure，即先通过降维减少通道数，再通过升维恢复原始通道数。假设原始通道数为$$C$$，降维比例为$$r$$，则中间层的通道数为$$C/r$$。这种设计<span style="color: rgb(46,161,33); background-color: inherit">大幅降低了计算复杂度，同时保留了足够的表达能力</span>
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">可嵌入性与普适性</span>**：**SE** 块是一种高度模块化的组件，可以无缝嵌入到现有的卷积神经网络架构中。无论&#x662F;**`ResNet`**、**`Inception`**&#x8FD8;&#x662F;**`DenseNet`**，只需在每个卷积块后添加一个 **SE&#x20;**&#x5757;，即可显著提升模型性能。这种普适性使得 **SENet&#x20;**&#x6210;为一种极具吸引力的通用工具
>
> 4) **<span style="color: rgb(36,91,219); background-color: inherit">参数与计算量的权衡</span>**：基础模型增加 **SE&#x20;**&#x6A21;块后会使得整体模型的参数增加&#x7EA6;**`10%`**，但计算量增加不多。例如，<span style="color: rgb(220,155,4); background-color: inherit">在</span>**`ResNet-50`**<span style="color: rgb(220,155,4); background-color: inherit">上添加 </span>**<span style="color: rgb(220,155,4); background-color: inherit">SE </span>**<span style="color: rgb(220,155,4); background-color: inherit">模块后，参数量从</span>**`25.6M`**<span style="color: rgb(220,155,4); background-color: inherit">增加到</span>**`28.1M`**<span style="color: rgb(220,155,4); background-color: inherit">，而 FLOPs 仅增加了不到</span>**`1%`**。这种轻量化设计使得 **SENet&#x20;**&#x5728;实际应用中具有很高的性价比

**<span style="color: rgb(222,120,2); background-color: inherit">代码实现</span>**

这里以 SE-ResNet-50 为例

首先，定义一个通用的 **SE** 块，它将被嵌入到 **ResNet-50** 的残差模块中：

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        SE Block的实现
        :param channel: 输入特征图的通道数
        :param reduction: 降维比例，默认为16
        """
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.excitation = nn.Sequential(
```


然后基于 ResNet-50 的结构，嵌入 SE 块以构建 **SE-ResNet-50**：

```python
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
```


**<span style="color: rgb(222,120,2); background-color: inherit">总结</span>**

**SENet&#x20;**&#x7684;主要创新点可以总结为以下几点： &#x20;

> 🚅 1. **<span style="color: rgb(36,91,219); background-color: inherit">通道注意力机制的引入</span>**：**SENet&#x20;**&#x9996;次系统性地提出了通道注意力机制，通过显式建模通道间的依赖关系，增强了网络对重要特征的聚焦能力
>
> 2. **<span style="color: rgb(36,91,219); background-color: inherit">自适应特征重新校准</span>**：通过动态调整每个通道的重要性，**SENet&#x20;**&#x5B9E;现了对特征图的自适应重新校准，从而提升了模型的表示能力
>
> 3) **<span style="color: rgb(36,91,219); background-color: inherit">轻量化与高效性</span>**：尽管引入了额外的计算步骤，**SE&#x20;**&#x5757;的设计非常注重效率，能够在不显著增加计算开销的情况下带来显著的性能提升

自 **SENet&#x20;**&#x63D0;出以来，其思想已被广泛应用于各类计算机视觉任务中，包括<span style="color: rgb(220,155,4); background-color: inherit">图像分类、目标检测、语义分割</span>等。例如，<span style="color: rgb(220,155,4); background-color: inherit">在 ImageNet 图像分类任务中，基于 </span>**<span style="color: rgb(220,155,4); background-color: inherit">SENet </span>**<span style="color: rgb(220,155,4); background-color: inherit">的模型 </span>**<span style="color: rgb(220,155,4); background-color: inherit">SE-ResNet </span>**<span style="color: rgb(220,155,4); background-color: inherit">以显著的优势赢得了 2017 年 ILSVRC 竞赛的冠军</span>。此外，**SENet&#x20;**&#x7684;思想也被后续许多工作所借鉴，例如 **<span style="color: rgb(220,155,4); background-color: inherit">EfficientNet</span>**<span style="color: rgb(220,155,4); background-color: inherit">、</span>**<span style="color: rgb(220,155,4); background-color: inherit">MobileNetV3 </span>**<span style="color: rgb(220,155,4); background-color: inherit">等轻量化网络，均采用了类似的</span>**<span style="color: rgb(100,37,208); background-color: inherit">通道注意力机制</span>**。

**<span style="color: rgb(100,37,208); background-color: inherit">SENet </span>**<span style="color: rgb(100,37,208); background-color: inherit">的提出标志着</span>**<span style="color: rgb(100,37,208); background-color: inherit">通道注意力机制</span>**<span style="color: rgb(100,37,208); background-color: inherit">在深度学习领域的兴起</span>。它通过一种简单而优雅的方式，显著提升了卷积神经网络的性能，同时保持了较低的计算开销。随着深度学习技术的不断发展，通道注意力机制有望与其他注意力机制，如<span style="color: rgb(220,155,4); background-color: inherit">空间注意力、时间注意力</span>相结合，进一步推动神经网络架构的演进。就像 SENet 作者展示的，<span style="color: rgb(100,37,208); background-color: inherit">关注细节并深入挖掘潜在的优化空间，往往能够带来意想不到的突破</span>。

---

[Contents](../../README.md) | [Next](02-视觉基础--Transformer-骨干网络.md) | [Visual website](https://weyumm.github.io/vlm-Wissen/lecture-1.html#c=1)
