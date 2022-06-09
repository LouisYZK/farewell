#! https://zhuanlan.zhihu.com/p/525597265
# [论文笔记] 高斯混合模型GMM与协同过滤：混合概率模型的协同矩阵估计MPMA

[Chen C, Li D, Lv Q, et al. MPMA: Mixture Probabilistic Matrix Approximation for Collaborative Filtering[C]//IJCAI. 2016: 1382-1388.](http://recmind.cn/papers/mpma_ijcai16.pdf)

关于高斯混合模型GMM的应用，推荐系统应用部分我选取了一个更加纯粹的应用模式。所谓纯粹GMM就是直接用生成模型进行建模。我选取了IJCAI2016的一个工作MPMA，是用贝叶斯概率视角(MP, Matrix Probalistic)对推荐系统中的协同过滤CF方法的基石--协同矩阵（item-user matric）的估计（MA, Matrix Approximation）问题进行重新建模的。

还是先回顾一下传统的矩阵估计方法。从原始的SVD分解，到用深度网络去学习user和item表示（NeuralCF），再到双塔模型... 而MPMA的工作将MA的估计考虑的更多，将每个矩阵元素$R_{ij}$（用户i->商品j）融合进了local和global的表示：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206062059460.png)

上图的1,2,3分别是建模的三个隐变量，表示构成$R_{ij}$的三个隐含类别。分别来自：基于用户的local表示、基于物品的local表示和全局表示。local和global分别学习自不同的协同矩阵。将矩阵元素用概率写出来：

![image-20220606211752865](/Users/yangzhikai/Library/Application Support/typora-user-images/image-20220606211752865.png)

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