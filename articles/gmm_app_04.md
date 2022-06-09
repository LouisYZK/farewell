#! https://zhuanlan.zhihu.com/p/526572416
# [论文闭集] GMM的计算机视觉特征应用：Fisher Vectors

研究CV的老哥们肯定知道2013年左右的上古图像特征FV（Fisher Vectors）[1]。这是GMM直接用在图像特征提取上。首先一个图像可以又$t$个descriptor描述：$X= \{ x_t, t=1,...,T  \}$. 这个$x_t$可能来自其他local特征如SIFT，那么这个序列（可能是长度不一的）就完整地描述了图像全部特征。

现在为了解决每个图像的描述$X$可能长度不一的问题，想到可以用概率来对$X$建模，pdf为$\mu_{\lambda}(X)$。那么$X$就可以用log-likelyhood的梯度来描述，这个梯度是定长的，只与参数维度相关：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091141359.png)

再对$X$的各个特征做iid假设：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091144526.png)

而FV选择的对$X$概率建模的$\mu_{\lambda}$就是GMM，是基于更早的一项工作Fisher Kernel[7]。$\mu_{\lambda}(x) = \sum_{i}^K w_iu_i{x}$,其中参数就是GMM中的三个参数：$\lambda = \{ w_i, \mu_i, \Sigma_i, i = 1...K \}$。

各个参数梯度的计算与GMM中EM的推导一样：

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091618125.png)

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091618087.png)

FV认为GMM建模的log似然函数的梯度能表示图像特征的原因是，似然的梯度是图像数据不断去拟合GMM隐变量分布的过程，这个过程能反应图像的诸多局部动态特征[2]。

FV是早期CV领域图像特征提取的方法，用的很普遍。现在google schoolar一下FV近些年的应用已经偏向时间序列方面了，例如语音信号、多模态序列、EEG信号、文本等的特征处理了。

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091643454.png)

因为这些长序列数据的处理有一个核心问题：序列长度的padding与对齐，如何把不同长度的序列更好地放在一起训练。相关工作往往期望一种特征提取能统一长度同时又不丢失信息，FV的过程就非常合适。举个例子，下图是一个多模态的Audio + Video多模态序列特征处理的一个流程[3]:

![](https://raw.githubusercontent.com/LouisYZK/picrepo/main/202206091649790.png)

将拼接好的多模态序列特征用GMM+FV处理成定长的向量。

[1] Sánchez J, Perronnin F, Mensink T, et al. Image classification with the fisher vector: Theory and practice[J]. International journal of computer vision, 2013, 105(3): 222-245.

[2] Perronnin F, Dance C. Fisher kernels on visual vocabularies for image categorization[C]//2007 IEEE conference on computer vision and pattern recognition. IEEE, 2007: 1-8.

[3] Zhang Z, Lin W, Liu M, et al. Multimodal deep learning framework for mental disorder recognition[C]//2020 15th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2020). IEEE, 2020: 344-350.
