---
layout:     post
date:       2020-03-09
tag:        note
author:     BY Zhi-kai Yang
---
>Mar 09, 2020,  Zhengzhou
>
>统计推断04： 数据简化原理与点估计

## 充分性原理

参数$\theta$ 的一个充分统计量在某种意义上提炼了样本中有关$\theta$的 全部信息，即除充分统计量的值意外，样本中其余信息不能再提供关于$\theta$的任何信息。这就是数据简化充分性原理的基本思想。

### 充分统计量

**定义** 如果样本$\mathbb{X}$ 在已知统计量$T(\mathbb{X})$ 取值时的条件分布与$\theta$ 无关，则称统计量$T(\mathbb{X})$ 是$\theta$ 的充分统计量；

按照上述定义验证统计量$T(X)$是否是$\theta$ 的充分统计量需要证明对任意的$\mathbf{x}$ 和$t$， 条件概率$P_\theta(\mathbf{X}=\mathbf{x}| T(\mathbf{X}) = T(\mathbf{x}))$ 与$\theta$无关. 所以
$$
P_\theta(\mathbf{X}=\mathbf{x}| T(\mathbf{X}) = T(\mathbf{x})) = \frac{P_\theta(\mathbf{X}=\mathbf{x} ,T(\mathbf{X})=T(x))}{P_\theta( T(\mathbf{X})} \\ =\frac{P_\theta{(X=x)}}{P_\theta(T(X))} ,\ because \ \{X=x\} \in \{T(X) = T(x)\} \\
=\frac{p(\boldsymbol{x} | \theta)}{q(T(\boldsymbol{x}) | \theta)}
$$
于是按照定义，上式的结果应该是与$\theta$ 无关的常函数。这就导出的下述定理：

**定理** 设$p(x| \theta)$ 为样本$X$ 的联合概率密度函数，$q(t|\theta)$ 为$T(X)$ 的概率密度函数，如果对样本空间中的任意$x$, 比值$p(x|\theta)/ q(T(x)|\theta)$ 都是$\theta$的常函数，则$T(X)$是$\theta$的充分统计量。

运用上述定理是很难求得充分统计量的，因为上诉定理更适合来验证。下面的定理能简化我们计算充分统计量：

**因子分解定理** 设$f(x|\theta)$ 为样本$X$ 的联合概率密度函数，统计量$T(X)$是$\theta$的充分统计量当且仅当存在函数$g(t|\theta)$ 和$h(x)$, 使得对女人一样本点$x$以及参数$\theta$ 都有
$$
f(x|\theta) = g(T(x)|\theta)h(x)
$$
充分统计量也可以是一个向量，比如$T(X) = (T_1(X),...,T_r(X))$, 当参数本身是向量即$\theta = (\theta_1,...,\theta_s)$ 时常常出现这种情况，通常情况下$r=s$ 但也有例外；对于向量形式充分统计量，因子分解定理仍然适用。

利用因子分解定理很容易求得指数型分布的充分统计量，下面的结论很重要：

**定理**： 设随机样本服从指数族分布，即
$$
f(x | \boldsymbol{\theta})=h(x) c(\boldsymbol{\theta}) \exp \left(\sum_{i=1}^{k} w_{i}(\boldsymbol{\theta}) t_{i}(x)\right)
$$
其中$\theta =(\theta_1,...,\theta_d)$, $d \le k$, 则
$$
T(\boldsymbol{X})=\left(\sum_{j=1}^{n} t_{1}\left(X_{j}\right), \cdots, \sum_{j=1}^{n} t_{k}\left(X_{j}\right)\right)
$$
是$\mathbf{\theta}$的充分统计量

### 极小充分统计量

样本$X$ 本身就是一个充分统计量，且任何一一映射下的$T^\star(X)$都是充分统计量。当然我们希望用最少的数据来毫不损失地代表$\theta$.

**定义**  如果对其余任一充分统计量$T'(X)$,  $T(X)$都是$T'(X)$的函数，则称$T(X)$是极小充分统计量； 

**定理** 设$f(x|\theta)$ 是样本$X$的概率密度函数，如果存在函数$T(x)$ 使得对任意两个样本点$x,y$, 比值$f(x,\theta)/ f(y, \theta)$ 是$\theta$的常函数当且仅当$T(x) = T(y)$， 则$T(X)$ 是$\theta$的极小充分统计量。

### 辅助统计量

**定义** 如果统计量$S(X)$ 的分布与$\theta$ 无关，则称$S(X)$ 为辅助统计量；

**定义** 设$f(t|\theta)$ 是统计量$T(X)$ 的概率密度函数，如果满足：对任意$\theta$ 都有$E_\theta g(T) =0$  那么对任意$\theta$ 都有$P_\theta(g(T) =0)=1$ .则称该概率分布族是完备的，或称$T(X)$是完备统计量；

**Basu定理** 设$T(X)$ 是完备的极小充分统计量，则$T(X)$ 与任意辅助统计量都独立

## 似然原理

### 似然函数

**定义** 设$f(x|\theta)$ 为样本$\mathbf{X} =(X_1,...,X_n)$ 的联合概率密度函数，如果观测到$\mathbf{X} = \mathbf{x}$ 则称$\theta$ 的函数
$$
L(\theta|\mathbf{x}) = f(\mathbf{x}|\theta)
$$
为似然函数。

考察似然函数$L(\theta|\mathbf{x})$时，我们制定观测到的样本点为$x$, 让$\theta$ 在参数空间上任意取值；

似然原理揭示了似然函数实现数据简化的原理：

**似然原理** 设样本点$x$ 和$y$ 满足$L(\theta|x)$ 和$L(\theta|y)$成比例，即存在常数$C(x,y)$使得对任意$\theta$ 都有
$$
L(\theta|x) = C(x,y)L(\theta|y)
$$
则由$x$和$y$出发的所做的对$\theta$的推断完全相同

## 点估计

样本的任何一个函数$W(X_1,...,X_n)$ 称为一个点估计量 point-estimator, 即任意一个统计量就是一个点估计量。

需要注意的是估计值estimate 是样本观测的函数$W(x_1,...,x_n)$ 估计量是随机变量$\mathbf{X}$的函数。估计量都需要经过评价来确定其价值；

## 求估计量的方法

### 矩法

令前$k$阶的样本矩与总体矩相等就可以联立成发表赶车报告组，求解就可以得到矩估计量：
$$
\begin{aligned} m_{1} &=\frac{1}{n} \sum_{i=1}^{n} X_{i}^{1}, \quad \mu_{1}^{\prime}=\mathrm{EX}^{1} \\ m_{2} &=\frac{1}{n} \sum_{i=1}^{n} X_{i}^{2}, \quad \mu_{2}^{\prime}=\mathrm{EX}^{2} \\ & \vdots \\ m_{k} &=\frac{1}{n} \sum_{i=1}^{n} X_{i}^{k}, \quad \mu_{k}=\mathrm{EX}^{k} \end{aligned}
$$

$$
\begin{aligned} m_{1}=& \mu_{1}^{\prime}\left(\theta_{1}, \cdots, \theta_{k}\right) \\ m_{2}=& \mu_{2}^{\prime}\left(\theta_{1}, \cdots, \theta_{k}\right) \\ \vdots & \\ m_{k}=& \mu_{k}^{\prime}\left(\theta_{1}, \cdots, \theta_{k}\right) \end{aligned}
$$

### 极大似然估计量

**定义** 对每一个固定的样本点$\mathbf{x}$ , 令$\hat{\theta}(\mathbf{x})$是参数$\theta$的一个取值，他使得$L(\theta|x)$作为$\theta$的函数在该处取得最大值，那么，基于样本$\mathbf{X}$ 的极大似然估计量MLE,就是$\hat{\mathbf{\theta}}(\mathbf{X})$

**极大似然估计的不变性** 若$\hat{\theta}$ 是$\theta$的MLE, 则对于$\theta$的任何函数$\tau(\theta)$, $\tau(\hat{\theta})$是$\tau(\theta)$的MLE

使用这个定理，我们现在看到正态均值平方$\theta^2$的MLE是$\bar{X}^2$ .

### Bayes 估计量

频率派的统计学认为$\theta$ 是一个未知但固定的量，在Bayes方法中，$\theta$被考虑为一个其变化可被一个概率分布描述的量，该分布为先验分布。这是一个主主观的分布，建立在试验者的信念上，而且见到抽样数据前已经用公式确定好了。 然后从以$\theta$为指标的总体中抽取一组样本，先验分布通过样本信息得到校正。这个被校正的分布称谓后验分布。

Bayes推断的公式为：
$$
\pi(\theta|x) = \frac{f(x|\theta)\pi(\theta)}{m(\theta)} \\ 
= \frac{f(x|\theta)\pi(\theta)}{\int f(x|\theta)\pi(\theta)d\theta}
$$
注意这个后验分布是一个条件分布，其条件建立在观测样本上，现在用这个后验分布来作出关于$\theta$的推断，而$\theta$被考虑为一个随机的量。例如后验分布的均值就可以被用作$\theta$的点估计。

>二项分布的贝叶斯估计：
>
>假设$p$的先验分布是beta分布，求得后验分布为：
>$$
>f(p | y)=\frac{f(y, p)}{f(y)}=\frac{\Gamma(n+\alpha+\beta)}{\Gamma(y+\alpha) \Gamma(n-y+\beta)} p^{y+\alpha-1}(1-p)^{n-y+\beta-1}
>$$
>这是一个$beta(y+\alpha, n-y+\beta)$分布。 注意这里$p$是变动的，$y$是固定的。$p$的一个贝叶斯估计量就是后验分布的期望：
>$$
>\hat{p} = \frac{y+\alpha}{\alpha +\beta + n}
>$$
>先验分布的均值是$\alpha/ \alpha + \beta$ ，如果没有先验时，我们根据样本对$p$的估计是$y/n$ . 而$p$的贝叶斯分布则结合了这两部的信息，可以吧贝叶斯估计写作：
>$$
>\hat{p}_{B}=\left(\frac{n}{\alpha+\beta+n}\right)\left(\frac{y}{n}\right)+\left(\frac{\alpha+\beta}{\alpha+\beta+n}\right)\left(\frac{\alpha}{\alpha+\beta}\right)
>$$
>这样看贝叶斯估计就可以表示成先验均值和样本均值的一个线性组合。

为什么在二项分布的估计是先验分布选择beta分布？下面引出一个共轭分布的定义：

**定义** 设$\mathcal{F}$ 是概率密度函数或概率质量函数$f(x|\theta)$的类（以$\theta$为指标） 称一个先验分布类$\prod$ 为$\mathcal{F}$ 的一个共轭族，如果对所有的$f \in \mathcal{F}$ 所有的$\prod$中的鲜艳发呢不和所有的$x \in X$, 其后验分布仍在$\prod$中。

贝塔分布就是二项分布族的共轭分布。这样我们那一个贝塔分布当做先验分布开始，我们将以一个贝塔分布的后验分布结束，先验分布的校正表现为对其参数的校正，这在数学上十分方便。

### EM算法

EM算法其实是求收敛MLE的一种算法。他把一个难以处理的似然函数最大化问题用一个易于最大化的序列取代，而其极限是原始问题的解。

EM算法允许我们在对$L(\theta|y)$求极大时只通过处理$L(\theta|y,x)$  以及给定的$y$和$\theta$的条件下$X$ 的条件pmf或pdf，
$$
k(x|\theta, y) = \frac{f(y,x|\theta)}{g(y|\theta)}
$$
两边取对数：
$$
logL(\theta|y) = log L(\theta|y,x) - logk(x|\theta, y)
$$
因为$x$ 是缺失数据或者不可测的。我们可以把上式的右边替换成关于条件分布$k(x|\theta',y)$的期望：
$$
\log L(\theta | \boldsymbol{y})=\mathrm{E}\left[\log L(\theta | \boldsymbol{y}, \boldsymbol{X}) | \theta^{\prime}, \boldsymbol{y}\right]-\mathrm{E}\left[\log k(\boldsymbol{X} | \theta, \boldsymbol{y}) | \theta^{\prime}, \boldsymbol{y}\right]
$$
现在EM算法就是从一个初始值$\theta_0$开始， 不断迭代：
$$
\theta^{(r+1)}= \max_\theta E[\log L(\theta|y,X)|\theta',y]
$$
E步骤就是指计算对数似然函数的期望，M步骤就是求其最大值。

## 估计量的评价方法

评价统计量的基本准则

### 均方误差

**定义** 参数$\theta$的估计量$W$的均方误差MSE是由$E_\theta(W-\theta)^2$定义的关于$\theta$的函数

他有一个实际意义上的解释：
$$
E_\theta(W-\theta)^2 = Var_\theta W + (E_\theta W-\theta)^2 = Var_\theta W +(Bias_\theta W)^2
$$
**定义**： 偏差的定义即为$Bias_\theta = E_\theta W -\theta$, 当一个估计量当他的偏差恒等于0，则称为无偏的，他满足$E_\theta W =\theta$ 对所有$\theta$成立

这样MSE就包含两部分，一个是估计量的变异性（精度），其次是他的偏差。如果一个估计量是无偏的，则他的MSE就是他的方差。

### 最佳无偏估计量

**定义** 估计量$W^\star$ 称为$\tau(\theta)$的最佳无偏估计量如果他满足无偏，且他的方差比其余任何无偏估计量的方差都要小。也称其为一致最小方差无偏估计量。

找到最佳无偏估计其实并不容易，人们也许很想拥有一个更具理性的研究手段，假定为了估计具有分布$f(x|\theta)$的一个参数$\tau(\theta)$，我们能为$\tau(\theta)$的任何无偏估计的方差具体指出一个下界，记做$B(\theta)$ ,如果我们能找到一个无偏估计量$W^\star$ 满足$VarW^\star = B(\theta)$ 我们就找到了一个最佳无偏估计的方法，这就是采用C-R 下界的方法：

**定理 Cramer-Rao 不等式**

令$W(X) = W(X_1,...,X_n)$是任意一个估计量，满足：
$$
\frac{\mathrm{d}}{\mathrm{d} \theta} \mathrm{E}_{\theta} W(\boldsymbol{X})=\int_{X} \frac{\partial}{\partial \theta}[W(\boldsymbol{x}) f(\boldsymbol{x} | \theta)] \mathrm{d} \boldsymbol{x}\\
Var_\theta W(X) <  \infty
$$
则有
$$
\operatorname{Var}_{\theta}(W(x)) \geqslant \frac{\left(\frac{\mathrm{d}}{\mathrm{d} \theta} \mathrm{E}_{\theta} W(\boldsymbol{X})\right)^{2}}{\mathrm{E}_{\theta}\left(\left(\frac{\partial}{\partial \theta} \log f(\boldsymbol{X} | \theta)\right)^{2}\right)}
$$
如果我们添上独立样本的假定，那么下界的计算就简单了，分母上的联合分布的期望变为一元计算：

**C-R不等式，iid情况**
$$
\operatorname{Var}_{\theta}(W(\boldsymbol{X})) \geqslant \frac{\left(\frac{\mathrm{d}}{\mathrm{d} \theta} \mathrm{E}_{\theta} W(\boldsymbol{X})\right)^{2}}{n \mathrm{E}_{\theta}\left(\left(\frac{\partial}{\partial \theta} \log f(X | \theta)\right)^{2}\right)}
$$
$\mathrm{E}_{\theta}\left(\left(\frac{\partial}{\partial \theta} \log f(\boldsymbol{X} | \theta)\right)^{2}\right)$ 叫做样本的信息数，也叫Fisher信息量，这个术语反应当信息量增大时，我们就掌握$\theta$更多的信息，从而就有一个较小的最佳无偏估计方差的界。

Fisher信息数其实就是MLE方程一阶导数的二阶矩，他有更多统计学的意义，可以参考[回答](https://www.zhihu.com/question/26561604/answer/33275982)

### 充分性（Sufficiency）和无偏性

本节的主定理吧充分统计量和无偏估计联系起来，下述是著名定理：

**Rao-Blackwell定理** 设$W$是$\tau(\theta)$的一个无偏估计量，而$T$是关于$\theta$的一个充分统计量。 定义$\phi(T) = E(W|T)$ 则$E_\theta \phi(T) \le Var_\theta W$ 对所有$\theta$成立；即是说$\phi(T)$ 是一个一致较优的无偏估计量。

因此，对任何一个无偏估计量，求其在给定一个充分统计量时的条件期望将导致一个一致改善，所以我们再求最佳无偏估计量时只需要考虑那些是充分统计量的函数的统计量就行了。

**定理** 如果$W$是$\tau(\theta)$的一个最佳无偏估计量，则$W$是唯一的。

本节的理论告诉我们如果能找到任意的一个无偏估计量，我们就能找到那个最佳无偏估计量。若$T$是参数$\theta$是一个完全充分统计量，$h(X_1,...,X_n)$是一个无偏估计量，则$\phi(\theta) = E(h(X_1,...,X_n)|T)$是最佳无偏估计量。
