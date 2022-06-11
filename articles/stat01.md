#! https://zhuanlan.zhihu.com/p/504933802
---
layout:     post
date:       2020-02-24
tag:        note
author:     BY Zhi-kai Yang
---

> 统计推断-1 
>
> Feb.14,2020,  Zhengzhou, Henan
>
> 内容：常见分布族

这是《统计推断》的第三章节，涵盖的分布比我本科阶段学习的概统课程中的分布要多一些。

## 离散分布

### 离散均匀分布

如果$$P(X=x|N) = \frac{1}{N}, x=1,2,3,..,N$$ 则称随机变量$$X$$ 服从参数为$$(1,N)$$ 的离散均匀分布；
$$
EX= \frac{N+1}{2} \\ VarX = EX^2 - E^2X = \frac{(N+1)(N-1)}{12}
$$

### 超几何分布

超几何分布常见于有限总体样本时建模，他满足
$$
P(X=x|N,M,K) = \frac{\tbinom{M}{x}\tbinom{N-M}{K-x}}{\tbinom{N}{K}}
$$
次模型可以借助传统的“摸球问题”理解，共有样本$$N$$个球，其中红球$$M$$个，剩余是黑球，在一次试验中摸取$$K$$个球，其中正好有$$x$$个红球的概率如上式表达。
$$
EX = \frac{KM}{N} \\ VarX = \frac{KM}{N}{\frac{(N-M)(N-K)}{N(N-1)}}
$$

### 二项分布

设有$$n$$相同且独立的Bernoulli试验，每个试验成功的概率都是$$p$$, 定义随机变量$$X_i, X_2, ...,X_n$$为 $X_i = 1$ 的概率为$p$, $X_i= 0$的概率为$(1-p)$; 则随机变量
$$
Y = \sum_{i=1}^n X_i
$$
服从参数为$(n，p)$的二项分布；
$$
P(Y=y | n, p)=\left(\begin{array}{l}{n} \\ {y}\end{array}\right) p^{y}(1-p)^{n-y}, \quad y=0,1,2, \cdots, n
$$

$$
EX = np, \ VarX = np(1-p)
$$

### Poisson分布

泊松分布适用于对等待事务的概率的建模，例如我们认为某事件发生的概率随等待的时间越长而越大。变量$X$ 服从参数为$\lambda$ 的泊松分布：
$$
P(X=x | \lambda)=\frac{\mathrm{e}^{-\lambda} \lambda^{x}}{x !}, x=0,1, \cdots
$$

$$
EX = VarX = \lambda
$$

> 当二项分布中的$n$很大、$p$ 很小时， 我们可以使$\lambda = np$的泊松分布来近似二项分布。

### 负二项分布

设有一列独立的成功概率为$p$的Bernoulli试验，以随机变量$X$记该序列中第$r$个成功试验出现的位置，则
$$
P(X=x | r, p)=\left(\begin{array}{l}{x-1} \\ {r-1}\end{array}\right) p^{r}(1-p)^{x-r}, \quad x=r, r+1, \cdots
$$
称$X$ 服从参数为$r,p$ 的负二项分布，即为$NB(r,p)$

另一种更常用的形式为：
$$
P(Y=y)=\left(\begin{array}{c}{r+y-1} \\ {y}\end{array}\right) p^{r}(1-p)^{y}, \quad y=0,1, \cdots
$$

$$
EY = r\frac{1-p}{p} \\
VarY =  r\frac{1-p}{p^2}
$$

### 几何分布

几何分布式负二项分布的特例，也是等待时间分布的一种最为简单的形式：
$$
P(X=x | p)=p(1-p)^{x-1}, \quad x=1,2, \cdots
$$
这里的$X$ 可以理解为**出现第一个成功试验时总的试验个数**， 即我们等待的是"一个成功”。
$$
EX = \frac{1}{p} \\ VarX = \frac{1-p}{p^2}
$$
几何分布有一个无记忆性的性质，对任意整数$s > t$, 有
$$
P(X>s |X>t) = P(X>s-t)
$$
也就是说犯错的概率仅仅和次数有关，与出现的位置无关。

由于无记忆性，几何分布不能用来建模那些报废（死亡）概率随时间增大的事务的寿命。

## 连续分布

###  $\Gamma$分布

首先定义Gamma函数如下：
$$
\Gamma(\alpha)=\int_{0}^{+\infty} t^{\alpha-1} \mathrm{e}^{-t} \mathrm{d} t
$$
有如下的性质： 
$$
\Gamma(\alpha+1)= \alpha \Gamma(\alpha) \\ \Gamma(n) = (n-1)! \\ \Gamma(1) = 1,\ \Gamma(\frac{1}{2}) = \pi^{1/2}
$$
参数为$(\alpha, \beta)$的 Gamma分布的的概率密度函数如下：
$$
f(x | \alpha, \beta)=\frac{1}{\Gamma(\alpha) \beta^{\alpha}} x^{a-1} \mathrm{e}^{-x / \beta}, \quad 0<x<+\infty, \quad \alpha>0, \quad \beta>0
$$

$$
EX = \alpha \beta \\ VarX = \alpha \beta^2
$$

$\alpha$ 影响宽度， $\beta$ 影响陡峭程度。

Gamma分布有很多重要的特例，令$\alpha = \frac{p}{2}$ , 其中$p$为整数且 $\beta = 2$ 则概率密度函数此时为：
$$
f(x | p)=\frac{1}{\Gamma(p / 2) 2^{p / 2}} x^{p / 2-1} \mathrm{e}^{-x / 2}, \quad 0<x<\infty
$$
这就是**自由度为$p$ 的卡方分布** 。卡方分布在统计推断中发挥着重要作用，特别是对正态分布总体的抽样。

若令$\alpha =1$ 此时
$$
f(x | \beta)=\frac{1}{\beta} \mathrm{e}^{-x / \beta}, \quad 0<x<\infty
$$
此即尺度参数为$\beta$ 的指数概率密度函数。与离散情况下的几何分布类似，指数分布也可以用于建模“寿命”. 事实上，指数分布也具有几何分布的'无记忆性'。

### 正态分布

密度函数
$$
f\left(x | \mu, \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi} \sigma} \mathrm{e}^{-(x-\mu)^{2} /\left(2 \sigma^{2}\right)}, \quad-\infty<x<+\infty
$$
上式的推导来自于高斯对误差分布的建模得到的。

正态分布的应用极为广规范，其中重要的一项应用就是我们可以用它来逼近其他分布（这在某种程度上由中心极限定理保证）。

### Beta 分布

贝塔分布是$(0,1)$ 区间上含有两个参数的一类连续分布，参数为$(\alpha, \beta)$ 的贝塔概率密度函数为：
$$
f(x | \alpha, \beta)=\frac{1}{\mathrm{B}(\alpha, \beta)} x^{\alpha-1}(1-x)^{\beta-1}, \quad 0<x<1, \quad \alpha>0, \quad \beta>0 \\
\mathrm{B}(\alpha, \beta)=\frac{\Gamma(\alpha) \Gamma(\beta)}{\Gamma(\alpha+\beta)} \\
EX = \frac{\alpha}{\alpha + \beta}
$$
贝塔分布是为数极少的几类被命名、且在有限区间上取得概率为1的分布之一，因而常用于建模0到1内的各种比例。

### 对数正态分布

如果随机变量$X$ 的自然对数服从正态分布，$logX \sim N(\mu, \delta^2)$ , 称$X$ 服从对数正态分布，X的分布函数：
$$
f\left(x | \mu, \sigma^{2}\right)=\frac{1}{\sqrt{2 \pi} \sigma} \frac{1}{x} \mathrm{e}^{-(\log x-\mu)^{2} /\left(2 \sigma^{2}\right)}, 0<x<+\infty,-\infty<\mu<+\infty, \sigma>0
$$
对数正态分布的形状与$\Gamma$分布族类似，他广泛应用于某些右偏变量的建模，例如工资就是一个右偏变量。

## 指数族分布

一个分布可以成为指数族时，他的概率密度函数可以表示为：
$$
f(x | \boldsymbol{\theta})=h(x) c(\boldsymbol{\theta}) \exp \left(\sum_{i=1}^{k} w_{i}(\boldsymbol{\theta}) t_{i}(x)\right)
$$

## 位置与尺度族

位置族、尺度族、位置-尺度族。

**定理**  设$f(x)$ 是概率密度函数，$\mu$ 和 $\sigma >0$ 为任意指定参数，则函数
$$
g(x|\mu, \sigma) = \frac{1}{\sigma}f(\frac{x-\mu}{\sigma})
$$
也是概率密度函数。

设$f(x)$ 是概率密度函数，则称概率密度函数族$f(x - \mu)$ 是标准概率密度函数为$f(x)$ 的位置族 (location family with standard pdf $f(x)$)

设$f(x)$ 是概率密度函数，则称概率密度函数族$\frac{x}{\sigma} f(\frac{x}{\sigma})，\ \sigma >0$ 是$f(x)$的尺度族(scale family).

设$f(x)$是概率密度函数，则称概率密度函数族 $\frac{x -\mu}{\sigma} f(\frac{x-\mu}{\sigma})，\ \sigma >0$ 是$f(x)$的位置-尺度族；

位置-尺度族随机变量的概率可以通过标准化变量$Z$ 按下式计算
$$
P(X \leqslant x)=P\left(\frac{X-\mu}{\sigma} \leqslant \frac{x-\mu}{\sigma}\right)=P\left(Z \leqslant \frac{x-\mu}{\sigma}\right)
$$

## 不等式与恒等式

这里介绍一些统计中经常使用的不等式、恒等式。

### 概率不等式

**Chebychev不等式** ：设X为随机变量，$g(x)$为非负函数，则对任意的$r >0$ 有
$$
P(g(X) \gt r) \le \frac{Eg(X)}{r}
$$

### 恒等式

很多分布都能发现递推关系，如之前的Possion分布中有
$$
P(X = x+1) = \frac{\lambda}{x+1}P(X=x)
$$
绝大部分离散分布都有类似上式的递推关系，甚至对于某些连续分布，其变形后的关系也成立。

**定理**  设$X_{\alpha ,\beta}$ 服从参数为$(\alpha, \beta)$的伽马分布，其概率密度函数为$f(x|\alpha,\beta)$ 其中$\alpha >1$,则对任意常数$a, b$,有
$$
P\left(a<X_{a, \beta}<b\right)=\beta(f(a | \alpha, \beta)-f(b | \alpha, \beta))+P\left(a<X_{a-1, \beta}<b\right)
$$
**定理 Stein引理** 设$X \sim N(\theta, \sigma^2)$ , $g$ 是满足$E|g'(X)| < + \infty$ 的可导函数，则
$$
\mathrm{E}[g(X)(X-\theta)]=\sigma^{2} \mathrm{E} g^{\prime}(X)
$$
Stein引理是直接使用分部积分法得到的有用等式，对于其他许多分布，利用分布积分法也能得到类似的概率恒等式。

**定理**  设$\chi_p^2$ 是自由度为$p$的随机变量，则对任意函数$h(x)$ ,有

$$\operatorname{Eh}\left(\chi_{\rho}^{2}\right)=p \mathrm{E}\left(\frac{h\left(\chi_{p+2}^{2}\right)}{\chi_{p+2}^{2}}\right) $$
