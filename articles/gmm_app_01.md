#! https://zhuanlan.zhihu.com/p/525111181
# [论文笔记] GMM与深度网络结合的异常检测任务：DAGMM

论文总结系列：[Zong B, Song Q, Min M R, et al. Deep autoencoding gaussian mixture model for unsupervised anomaly detection[C]//ICLR. 2018.](https://openreview.net/pdf?id=BJJLHbb0-)

这是一篇把深度网络嵌套在GMM的要素的异常检测算法。

该工作认为高维数据的异常检测会造成后验概率密度$\mathbb{P}(v=j | X_i, \Mu, \Sigma)$的估计困难。为了解决此问题，很多研究采用了两阶段学习的范式，先降维、再在低维空间里学习GMM的隐变量。作者认为两阶段的训练目标不一致，降维任务的目标可能与后验概率估计任务的目标不一致，造成模型性能不佳（结果是一个sub-optimal performance）。作者提出了一种joint learning的Deep Autoencoding Gaussian Mixture Model (DAGMM)模型：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051932011.png)

- **Compression Network:** 学习高维输入的降维表示，同时充分考虑到**重建损失(reconstruction error)**，目的是为了防止降维造成的信息损失。

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051936405.png)

$z_c$是降维编码后的表示，$h$是编码器。$g$是解码器，得到了重建表达$x'$。这样可以用$f$计算原始输入和解码器重建后的损失$z_r$。然后将他和编码后的降维表示$z_c$拼起来组成了压缩表达$z$输入给后面的评估网络使用。

前半部的网络只是为了降维，而与GMM结合的部分主要在第二部分的评估网络。如何结合呢，我们再次回忆GMM最重要的一个公式，也即是优化迭代的目标函数：

$$
Q\left(\theta, \theta^{t}\right)=\sum_{i=1}^{N} \sum_{w_{i}} \log p\left(x_{i}, w_{i} \mid \theta\right) p\left(w_{i} \mid x_{i}, \theta^{t}\right)
$$

E-step是用现有参数估计出后验$p (w_{i} \mid x_{i}, \theta^{t})$ ，然后对似然中的参数$\theta$的更新也都是基于这个后验进行的。这个后验在GMM中可以理解为样本$i$从属于类别$k$的后验概率，用$\gamma_{ik}$表示。我们知道，在生成模型的后验概率计算中，分母是个积分，维度一高就容易算不动，于是才有了很多研究围绕如何估计后验展开。GMM也一样，$\gamma_{ik}$的计算在高维大样本的情境下效率低下。于是一个很自然的想法就产生了：用神经网络来估计这个后验。

以上这个逻辑是我认为的用编解码器与GMM结合相关研究的基础逻辑。基于此, 作者提出的第二部分预估网络：

-  **Estimation Network:** 

首先，用MLN对输入的压缩表达进行继续学习，得到$p$, 然后用softmax得到多分类结果。这个就是后验结果。

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051950528.png)

然后就是GMM的M-step, 利用上一轮参数计算得到的后验更新GMM的参数：先验$\phi_k$（每个类别占比概率）、高斯分布的均值和方差：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051959432.png)

到这里有人会疑惑还计算这些做什么，不是已经有了学习目标$\gamma_{ik}$了？大家别忘了这是一个无监督模型，没有与$\gamma_{ik}$对应的label, 我们的优化目标函数需要与GMM模型保持一致：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051959220.png)

这样对GMM模型重要参数的估计体现在神经网络的loss函数上，估计误差就会体现在神经网络的参数上。这也是为什么我们用神经网络分类器得到的后验概率$\gamma_{ik}$可以直接用在GMM流程中的原因。

神经网络不但用分类器把GMM中难计算的后验概率解决了。并且最优化目标的参数梯度计算

$$\theta^{t+1}=\underset{\theta}{\operatorname{argmax}} \int_{w} \log [p(x, w \mid \theta)] p\left(w \mid x, \theta^{t}\right) d w=\mathbb{E}_{w \mid x, \theta^{t}}[\log p(x, w \mid \theta)]$$

也可以用神经网络的BP算法来优化。神经网络的拟合目标函数（objective  function）可以更复杂，于是除了GMM的优化目标外，可以把降维任务的目标、正则化项也加上：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206052006029.png)

$L$就是重建error，也融合进整个网络的优化目标中。

总结下DAGMM的主要思想（神经网络和GMM如何结合的？）：

- E-step: 用softmax分类器学习GMM后验$\gamma_{ik}$，并用它计算GMM的其他隐含参数
- M-step: 将计算得到的隐含参数继续计算GMM优化目标似然，作为神经网络的优化目标。这样在BP的同时达到等同于GMM参数的优化过程。
- 依靠神经网络的拟合能力，还可以把降维损失一并放在优化目标函数中优化。
- 使用GMM的优化目标、降维重建损失作为无监督任务的优化目标