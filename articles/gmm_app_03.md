#! https://zhuanlan.zhihu.com/p/526118956
# [论文笔记] 多模态混合高斯词向量 Multimodal Word Distributions
[Athiwaratkun B, Wilson A G. Multimodal Word Distributions[C]//ACL (1). 2017.](https://arxiv.org/pdf/1704.08424.pdf)

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

