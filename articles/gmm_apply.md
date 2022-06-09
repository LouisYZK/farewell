#! https://zhuanlan.zhihu.com/p/526531960
# [Python-SML] 高斯混合模型GMM的应用

GMM第一篇介绍了建模思想、EM解法和手写实现。现在我们知道了GMM是一个假定数据服从线性高斯混合模型的统计模型参数推断。参数推断完成后，计算出的后验概率$p(v_i = j | x_i, \theta_j)$ ,即$p(sample\ i\ belong\ to\ class\ j | x_i,\theta)$ 非常有用。可以作为聚类结果使用。因此如果我们的特征集高维、需要做无监督聚类，选择GMM是很合适的。

下面搜一些中英文paper看看国内外学术圈都是如何应用（牵强附会）的（逃...）(PS: 随缘挑选，如果有更代表性的工作大家可以补充~)

## 异常检测

### 直接使用GMM无监督分类

高维数据聚类，用GMM检测出异常类[1]。这篇文章是说，在火电厂巡检系统中导出的数据有很多异常值，这些数据干扰了后续环节系统的调控。因此需要设计异常检测算法把这些数据剔除。把数据聚成两类，GMM收敛后得出样本属于异常类的概率，设定阈值判断。

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051012925.png)

### GMM与深度网络结合：DAGMM
除了直接用GMM聚类，把深度网络嵌套在GMM的要素的算法也很多。

工作[2]认为高维数据的异常检测会造成后验概率密度$\mathbb{P}(v=j | X_i, \Mu, \Sigma)$的估计困难。为了解决此问题，很多研究采用了两阶段学习的范式，先降维、再在低维空间里学习GMM的隐变量。[2]认为两阶段的训练目标不一致，降维任务的目标可能与后验概率估计任务的目标不一致，造成模型性能不佳（结果是一个sub-optimal performance）。[2]提出了一种joint learning的Deep Autoencoding Gaussian Mixture Model (DAGMM)模型：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051932011.png)

- **Compression Network:** 学习高维输入的降维表示，同时充分考虑到**重建损失(reconstruction error)**，目的是为了防止降维造成的信息损失。

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206051936405.png)

$z_c$是降维编码后的表示，$h$是编码器。$g$是解码器，得到了重建表达$x'$。这样可以用$f$计算原始输入和解码器重建后的损失$z_r$。然后将他和编码后的降维表示$z_c$拼起来组成了压缩表达$z$输入给后面的评估网络使用。

前半部的网络只是为了降维，而与GMM结合的部分主要在第二部分的评估网络。如何结合呢，我们再次回忆GMM最重要的一个公式，也即是优化迭代的目标函数：

$$
Q\left(\theta, \theta^{t}\right)=\sum_{i=1}^{N} \sum_{w_{i}} \log p\left(x_{i}, w_{i} \mid \theta\right) p\left(w_{i} \mid x_{i}, \theta^{t}\right)
$$

E-step是用现有参数估计出后验$p (w_{i} \mid x_{i}, \theta^{t})$ ，然后对似然中的参数$\theta$的更新也都是基于这个后验进行的。这个后验在GMM中可以理解为样本$i$从属于类别$k$的后验概率，用$\gamma_{ik}$表示。我们知道，在生成模型的后验概率计算中，分母是个积分，维度一高就容易算不动，于是才有了很多研究围绕如何估计后验展开。GMM也一样，$\gamma_{ik}$的计算在高维大样本的情境下效率低下。于是一个很自然的想法就产生了：用神经网络来估计这个后验。

以上这个逻辑是我认为的用编解码器与GMM结合相关研究的基础逻辑。基于此，[2]提出的第二部分预估网络：

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

## 推荐系统
上文DAGMM是直接对GMM的要素做了「深度网络」化。推荐系统应用部分我选取了一个更加纯粹的应用模式。所谓纯粹GMM就是直接用生成模型进行建模。我选取了IJCAI2016的一个工作MPMA[3]，是用贝叶斯概率视角(MP, Matrix Probalistic)对推荐系统中的协同过滤CF方法的基石--协同矩阵（item-user matric）的估计（MA, Matrix Approximation）问题进行重新建模的。

还是先回顾一下传统的矩阵估计方法。从原始的SVD分解，到用深度网络去学习user和item表示（NeuralCF），再到双塔模型... 而MPMA的工作将MA的估计考虑的更多，将每个矩阵元素$R_{ij}$（用户i->商品j）融合进了local和global的表示：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062059460.png)

上图的1,2,3分别是建模的三个隐变量，表示构成$R_{ij}$的三个隐含类别。分别来自：基于用户的local表示、基于物品的local表示和全局表示。local和global分别学习自不同的协同矩阵。将矩阵元素用概率写出来：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091731470.png)

这是GMM建模的第一步，相当于写出了似然的形式（PS: 疑惑为啥要建模成正态分布，二项不应该更合理吗？）。$R_{ij}$ 是我们观测的数据。但不同于传统GMM，这里的均值参数复杂了一些。对user-local，item-local，global分别建模，参数量有所增加。对于均值参数依赖的商品embedding、user embedding以及各个local组的embedding, 也对其假设先验：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062122579.png)

我们是想用MAP（最大后验估计）来估计出这些。作者对后验

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062124270.png)

进行了很长的推导，最后得出了优化目标：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062125875.png)

作者分析，这个形式其实就是SVD优化目标的子集。只是加入了两个local组合的优化。目标函数(11)相当于原始SVD分解的优化目标，尽量还原R的结构；（12）（13）相当于两个local R的SVD分解优化目标。（14）是正则项。这个正则项怎么来的呢？回忆一下用MAP来推导线性回归模型，加入我们用概率对观测数据的似然建模，对方差参数假定不同的先验分布，就回得到Lasso和Redge形式的正则项，这里也是一样。

这个目标函数非凸，作者想用SGD来优化，推导出了各自的偏导：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062127554.png)

那么问题来了，以上 都是在讨论各类U\V的优化，似乎我们忘记了GMM的核心--隐参数还有一部分未被提及：各类别的概率先验、后验的方差$\sigma_{1,2,3}$。作者把UV的优化步骤算法写完了，紧接着才想起来这部分，依然用EM来优化：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062131889.png)

只不过这里M步就不再迭代优化UV了（用上文 的SGD来优化）。那么完整的M步就是在（21）步完成后，再进行（15）-（16）的UV优化。训练完成后，我们就可以用学到的所有参数来泛化估计用户i与物品j之间的推荐系数：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062138439.png)

总结下MAMP的GMM应用：

- 用传统的概率视角对观测数据协同矩阵建模似然。均值参数来自对每个物品和用户的表达估计。
- 将lcoal 和global的三个集合作为隐含类别。
- 用SGD + MAP估计均值参数，用EM估计GMM模型剩下的参数。

综上，这篇论文把协同过滤的矩阵估计问题建模为蕴含三个隐含组合的估计，并使用GMM模型描述，EM算法解决。特别的是部分参数来自协同矩阵的学习。

## 自然语言处理
GMM在NLP的应用我选取了ACL2017的一项工作：多模态高斯词嵌入，或者叫混合高斯词嵌入（Mixtured Gaussian Word Embedding）。从名字就可以看出他是继承自2014年的[高斯词嵌入](https://openreview.net/forum?id=WNtoc0ULITO)。

首先总结下高斯词嵌入的动机，基本上2013年`Word2vec`之后提出的词嵌入模型都是对static embedding的改进。高斯词嵌入不再把词向量学习成固定的向量，而是一个概率分布。每个词有自己的方差。这个方差可以表达很多东西：

- 不确定性（uncertainty），例如词频低的词不确定就高，代表我们学习的不够拟合。词频高的词学习的较好，方差小
- 包含性，把词向量学习成概率分布的好处是可以分布可以表达词与词之间的包含性（entailment），例如music就包含了pop，jazz等。而`Word2Vec`学习到的static embedding是对称的，并不能表达蕴含词义。

2014年的高斯分布embedding是单模态的，这篇论文把工作拓展到多模态，也就是一个词由混合高斯组成：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206081711181.png)

这么做的动机，一个是水...哦不是，最重要的是多模态混合高斯相比单模态有诸多好处，除了高斯embedding表达的蕴含关系，在多义词的表达上有明显优势：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206072142441.png)

上图看明白就基本明白多模态高斯混合embedding的动机了。

- 上图：高斯混合embedding, 拿「rock」举例，他的多个语义被建模成了两部分高斯混合，可以看到在「stone, basalt」、「music」两个语义区分别学习到了两个高斯分布。
- 下图：单模态高斯embedding, 「rock」的方差被不必要地放大，以蕴含多个语义。同时还有一个缺点是，蕴含的相似语义可能会被分开，像music下的「pop」和「jazz」。而多模态music会建模成蕴含「pop」和「jazz」的分布，同义词会更加紧凑。

这篇论文剩下的采样、训练模式与skip-gram无差别。目前函数还是中心词$w$与上下文$c$的相似性，加上负采样加速收敛。目标函数：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206081956273.png)

其中，$E_{\theta}$是一种能衡量两个分布之间的差距的能量函数，因为这里的$c,w$都是概率分布的embedding了，不能再简单使用点积了。相比Word2vec和高斯Embedding，多模态混合高斯的训练参数: $\theta = \{ \vec{\mu}_{w,i}, p_{w,i}, \Sigma_{w,i} \}$, 明显多了不少。

对目标函数中的能量函数，作者没有用自然想到的KL散度，而是用了Expected Likelihood Kernel的函数：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206081959810.png)

然后训练学习得到的就是每个单词$w$的 $\{ \vec{\mu}_{w,i}, p_{w,i}, \Sigma_{w,i}  \}$。可以用在后续的近义词、多含义分析中。

总结：

- 多模态混合高斯词向量是2014年Gaussian Embedding工作的多模态补充与拓展
- 用多模态高斯分布表示一个词向量具有多含义学习、蕴含关系学习、不确定性学习等优势
- 目标函数采用skip-gram和负采样。为了衡量分布向量间的差异，定义了能量函数
- 直接把所有参数放进目标函数梯度下降优化，没有使用EM算法

## 计算机视觉：Fisher Vector
研究CV的老哥们肯定知道2013年左右的上古图像特征FV（Fisher Vectors）[6]。这是GMM直接用在图像特征提取上。首先一个图像可以又$t$个descriptor描述：$X= \{ x_t, t=1,...,T  \}$. 这个$x_t$可能来自其他local特征如SIFT，那么这个序列（可能是长度不一的）就完整地描述了图像全部特征。

现在为了解决每个图像的描述$X$可能长度不一的问题，想到可以用概率来对$X$建模，pdf为$\mu_{\lambda}(X)$。那么$X$就可以用log-likelyhood的梯度来描述，这个梯度是定长的，只与参数维度相关：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091141359.png)

再对$X$的各个特征做iid假设：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091144526.png)

而FV选择的对$X$概率建模的$\mu_{\lambda}$就是GMM，是基于更早的一项工作Fisher Kernel[7]。$\mu_{\lambda}(x) = \sum_{i}^K w_iu_i{x}$,其中参数就是GMM中的三个参数：$\lambda = \{ w_i, \mu_i, \Sigma_i, i = 1...K \}$。

各个参数梯度的计算与GMM中EM的推导一样：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091618125.png)

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091618087.png)

FV认为GMM建模的log似然函数的梯度能表示图像特征的原因是，似然的梯度是图像数据不断去拟合GMM隐变量分布的过程，这个过程能反应图像的诸多局部动态特征[7]。

FV是早期CV领域图像特征提取的方法，用的很普遍。现在google schoolar一下FV近些年的应用已经偏向时间序列方面了，例如语音信号、多模态序列、EEG信号、文本等的特征处理了。

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091643454.png)

因为这些长序列数据的处理有一个核心问题：序列长度的padding与对齐，如何把不同长度的序列更好地放在一起训练。相关工作往往期望一种特征提取能统一长度同时又不丢失信息，FV的过程就非常合适。举个例子，下图是一个多模态的Audio + Video多模态序列特征处理的一个流程[8]:

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091649790.png)

将拼接好的多模态序列特征用GMM+FV处理成定长的向量。

## 总结
GMM作为一种使用多个高斯分布拟合任意概率分布的通用方法有诸多应用。应用的方式主要有：

- 直接用为一种聚类算法。`sklearn.mixture.GaussianMixtureModel`
- 仍然是无监督问题，与NN结合。后验部分深度网络代替，同时把参数优化目标放进损失函数
- 仅利用GMM的方式概率建模，不用EM算法求解。例如本文的高斯混合词向量和Fisher Vector.

> 估计很少有人把本文看完吧。这像是一篇综述，说来搞笑，硕士最后一个月的时间在这里老老实实写论文综述。这也是我领悟的一个关于学习的方式。以前第一次学习GMM，看过公式推导，看完了就完了。没有思考过这东西有啥用。完成硕士论文写作后，发现万事都有其动机逻辑链条。作为一种算法，无论它基于的理论还是应用都有其动机。于是在毕业季回顾系列我期望把这种写论文的思维方式再应用一下。毕竟这是读研究生收获的为数不多的财富。


[1] 吴铮,张悦,董泽.基于改进高斯混合模型的热工过程异常值检测[J/OL].系统仿真学报:1-12[2022-05-11].DOI:10.16182/j.issn1004731x.joss.22-0047.
[2] [Zong B, Song Q, Min M R, et al. Deep autoencoding gaussian mixture model for unsupervised anomaly detection[C]//ICLR. 2018.](https://openreview.net/pdf?id=BJJLHbb0-)
[3] [Chen C, Li D, Lv Q, et al. MPMA: Mixture Probabilistic Matrix Approximation for Collaborative Filtering[C]//IJCAI. 2016: 1382-1388.](http://recmind.cn/papers/mpma_ijcai16.pdf)

[4] [Athiwaratkun B, Wilson A G. Multimodal Word Distributions[C]//ACL (1). 2017.](https://arxiv.org/pdf/1704.08424.pdf)

[5] [Vilnis L, McCallum A. Word Representations via Gaussian Embedding[C]//ICLR. 2015.](https://openreview.net/forum?id=WNtoc0ULITO)

[6] Sánchez J, Perronnin F, Mensink T, et al. Image classification with the fisher vector: Theory and practice[J]. International journal of computer vision, 2013, 105(3): 222-245.

[7] Perronnin F, Dance C. Fisher kernels on visual vocabularies for image categorization[C]//2007 IEEE conference on computer vision and pattern recognition. IEEE, 2007: 1-8.

[8] Zhang Z, Lin W, Liu M, et al. Multimodal deep learning framework for mental disorder recognition[C]//2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020). IEEE, 2020: 344-350.


