# [Python-SML] 高斯混合模型GMM与实现

> 毕业回顾系列之「python-SML统计机器学习系列」，review总结学校所学，以期后续重拾与应用。不同于学生时代的细致探索，这次review的特点：
> - 简明数学原理
> - 工程实现导向
> - 案例导向，从论文举例

## GMM建模

GMM的产生还是来源于统计学对问题的建模思路。

统计学中我们通常会用已知分布去拟合观测到的数据，用极大似然估计MLE或极大后验估计MAP对参数进行推断。最常用来拟合的分布就是高斯分布。我们可以简单地将数据建模为单一的高斯分布，若想让模型复杂一些，数据分布服从多个高斯分布的混合；或者有了数据分布依赖一些隐变量（Latent Variable）的先验，我们可以用GMM建模。下面是GMM的pdf（概率密度函数）:
$$
p(x)=\sum_{k=1}^{K} w_{k} \mathcal{N}\left(\mu_{k}, \Sigma_{k}\right)
$$
其中$w_k$是模型中每个高斯分布的概率，换言之就是分布的分布。这里回忆一下[常用分布族](https://zhuanlan.zhihu.com/p/504933802)的知识，如果认为模型中有两个类别的高斯（二项分布）, $w$ 的先验可以认为从$\beta$分布产出。如果认为模型中大于两个高斯（多项式分布），$w$ 的先验可以认为从Dirichlet分布产出。

好了，GMM建模完毕，可以看出，GMM是一个典型的生成模型，因为我们没有把每个类的概率看成一个固定不变的参数，而是来自先验分布。由此，我们可以画出GMM的概率图模型：

<img src="https://raw.githubusercontent.com/LouisYZK/picrepo/main/202205082044619.png" style="zoom:50%;" />

其中$\alpha$ 表示多个高斯类别是否均匀。$V_n$ 是由每个高斯类别概率依据Dirichlet分布生成的样本个数。数据$x_i$ (图中为$y_i$ )则由不同的高斯分布参数产生。以两个类别画出符合GMM分布的数据分布：

<img src="https://raw.githubusercontent.com/LouisYZK/picrepo/main/202205082048875.png" style="zoom: 67%;" />

## EM算法

隐变量分布的参数$w$ 未知，无法直接用MLE算出GMM的pdf，这种含有隐变量的生成模型非常适合用EM算法解决。

EM算法的提出就是为了解决含有隐变量的生成模型，他的核心思想在于对数据的对数概率$ \log p(x|\theta)$的分解，用假设的隐变量分布$q(w)$进行：
$$
\log p(x \mid \theta)=\log p(w, x \mid \theta)-\log p(w \mid x, \theta)=\log \frac{p(w, x \mid \theta)}{q(w)}-\log \frac{p(w \mid x, \theta)}{q(w)}
$$
两边同时对$p(w|x, \theta)$ 取期望：
$$
\begin{array}{l}\text { Left : } \int_{w} q(w) \log p(x \mid \theta) d w=\log p(x \mid \theta) \\ \text { Right: } \int_{z} q(w) \log \frac{p(w, x \mid \theta)}{q(w)} d w-\int_{z} q(w) \log \frac{p(w \mid x, \theta)}{q(w)} d z=E L B O+K L(p(w \mid x, \theta), q(w))\end{array}
$$
可以看出最大化对数概率$ \log p(x|\theta)$的方法就是让$E L B O$最大，让我们假定的隐变量分布$q(w)$ 和后验分布$p(w|x, \theta)$ 的KL散度最小。最优化的思想是固定一个参数，去优化另一个参数。也就衍生出了EM中的E步和M步：

### E-step

固定上一步骤的$\theta^t$ ，优化KL散度，使得分布$q(w)$最优（KL散度下降）。优于可以数学证明这一步的解其实是$q(w) = p(w|x, \theta^t)$ ，所以狭义的EM算法，这一步也被成为Evaluation。即用$q(w) = p(w|x, \theta^t)$去估计$p(x|\theta)$。

### M-step

这一步就是固定了隐变量分布$q(w)$, 来优化参数$\theta$:
$$
\hat{\theta}=\underset{\theta}{\operatorname{argmax}} \int_{w} q^{t+1}(w) \log \frac{p(x, w \mid \theta)}{q^{t+1}(w)} d w
$$
把两个步骤整合一下，其实EM算法就一个关键公式（我们唯一需要记住的公式）：
$$
\theta^{t+1}=\underset{\theta}{\operatorname{argmax}} \int_{w} \log [p(x, w \mid \theta)] p\left(w \mid x, \theta^{t}\right) d w=\mathbb{E}_{w \mid x, \theta^{t}}[\log p(x, w \mid \theta)]
$$
可以从数学严格证明$\log p\left(x \mid \theta^{t}\right) \leq \log p\left(x \mid \theta^{t+1}\right)$ 成立.

## EM求解GMM

我们需要做的很简单，把GMM模型的参数假设套入EM关键迭代公式，算出来每个参数的梯度即可。

我们对EM迭代式进一步计算、简化：
$$
Q\left(\theta, \theta^{t}\right)=\sum_{i=1}^{N} \sum_{w_{i}} \log p\left(x_{i}, w_{i} \mid \theta\right) p\left(w_{i} \mid x_{i}, \theta^{t}\right)
$$
对于联合分布$ p(x, w|\theta)$ ，我们一般处理为能计算出的先验乘以似然：
$$
p(x, w \mid \theta)=p(w \mid \theta) p(x \mid w, \theta)=p_{w} \mathcal{N}\left(x \mid \mu_{w}, \Sigma_{w}\right)
$$
对于后验$p(w|x, \theta)$，他的计算是整个生成模型最麻烦之处，需要动用积分:
$$
p\left(w \mid x, \theta^{t}\right)=\frac{p\left(x, w \mid \theta^{t}\right)}{p\left(x \mid \theta^{t}\right)}=\frac{p_{w}^{t} \mathcal{N}\left(x \mid \mu_{w}^{t}, \Sigma_{w}^{t}\right)}{\sum_{k} p_{k}^{t} \mathcal{N}\left(x \mid \mu_{k}^{t}, \Sigma_{k}^{t}\right)}
$$
在GMM模型中，$p_k = w_k$ 表示每个高斯类别$v_i$的概率，我们把GMM中上述后验概率记做$r_{ji}$ 表示样本$x_i$属于类别$w_j$ :
$$
r_{j, i}= p\left(w \mid x, \theta^{t}\right) =\frac{\omega_{j} \mathcal{N} \left(y_{i} \mid \mu_{j}, \Sigma_{j}\right)}{\sum_{j} \omega_{j}  \mathcal{N} \left(y_{i} \mid \mu_{j}, \Sigma_{j}\right)}
$$
则GMM的EM优化目标函数可以写为：
$$
Q(\theta, \theta^t) = \sum_{i} \sum_{j} r_{j, i}(\theta^t) \left(\log P\left(v_{i}=j\right)+\log P\left(x_{i} \mid v_{i}=j\right)\right) \\
= \sum_{i} \sum_{j} r_{j, i}(\theta^t)( \omega_j + \mathcal{N}(x_i|\mu_j, \Sigma_j))
$$
这个式子不复杂，可以直接计算出各个参数的梯度：
$$
\begin{array}{c}
\omega_j  \leftarrow \frac{1}{N}\sum_i r_{j,i} \\
\mu_{j} \leftarrow \frac{1}{\sum_{i} r_{j, i}} \sum_{i} r_{j, i} x_{i} \\ 
\sigma_{j} \leftarrow \frac{1}{\sum_{i} r_{j, i}} \sum_{i} r_{j, i}\left(x_{i}-\mu_{j}\right)\left(x_{i}-\mu_{j}\right)^T
\end{array}
$$

## 用Python来写算法

如果目前仍然对EM目标函数的公式头疼，那么我们用伪代码来描述下$Q(\theta, \theta^t)$是咋算的：

```python
Q = .0
for ind in range(N_samples):
    for w in range(N_latent_gmm):
        p_w = class_prob[w]  ## 每个高斯类别概率
        mu_w, sigma_w = mus[w], sigmas[w]  ## 对应类别高斯参数
        p_x_w = Gassian(X[ind], mu_w, sigma_w) ## 计算似然 p(x| w)
        ## 计算后验 p(w|x) 也就是公式中分母积分部分
        px = .0
        for ww in range(N_latent_gmm):
            p_w = class_prob[w]
            mu_w, sigma_w = mus[w], sigmas[w]
        p_x_w = Gassian(X[ind], mu_w, sigma_w)
        px += p_w * p_x_w
    p_w_x = (p_w * p_x_w) / px
    Q += p_w_x * (p_w + p_x_w)  ## EM目标函数 Q += 后验*（隐变量先验 + 对应似然）
```

不过我们并不需要写代码计算目标函数，我们只需要算参数的梯度，我们可以设计一个GMM类，来进行参数梯度最优求解。下面我们使用python实现一个手动梯度下降，适配任意类个数、数据维度的GMM类。这一步就是考察我们的代码基本功了，如何将线性代数中的各种运算转化为以numpy矩阵算法包为代表的的代码。

同手写其他机器学习算法一样，一般是初始化参数、计算梯度、更新参数几个步骤。

### 初始化参数

我们采取一定的策略来初始化GMM模型的三个参数：各高斯类别概率、各高斯分布期望、各高斯分布方差。

```python
from sklearn.cluster import KMeans

class GMM:
    def __init__(self, n_class=None) -> None:
        """
        Args:
            data: [n_smaples, n_dim] raw data 
            n_class: if you know the latent gassian class num
        """
        self.n_epochs = 100
        self.n_class = n_class if n_class is not None else self.n_explore_class

    def init_params(self, data):
        self.data = data
        self.n_dim = data.shape[1]
        self.n_sample = data.shape[0]
				
        ## 1.采用了Kmeans初始化
        km = KMeans(self.n_class)
        km.fit(self.data)
        self.mus = []
        for ind in range(self.n_class):
            self.mus.append(np.mean(self.data[km.labels_ == ind], axis=0))
        self.vars = []
        for ind in range(self.n_class):
            self.vars.append(np.cov(self.data[km.labels_ == ind], rowvar=False))
        self.class_prob = np.random.rand(self.n_class)
        self.class_prob = self.class_prob / np.sum(self.class_prob)
        print(f'Init params: mus: {self.mus}\n vars: {self.vars}\n class_prob: {self.class_prob}')
```

上面我们仿照sklearn的GMM实现用Kmeans初始聚类来初始化参数。

### 计算梯度

从建模部分我们知道，GMM各参数的梯度首先要经过E步骤，即用上一步的参数计算出后验概率。再基于这个后验经M步算出梯度：

```python
from scipy import stats

class GMM:
    def e_step(self):
        """
        Calculate posterior prob given last time params.
        p_(z|x, \theta) = p(z=i | x_i) \ sum of tital probs

        Return:
            posteriors: [n_sample, n_class] reprent the probs of each item
                        belongs to each gaussian class.
        """
        models = [ stats.multivariate_normal(self.mus[ind], self.vars[ind]) 
                        for ind in range(self.n_class)]
        total_probs = []
        for ind in range(self.n_sample):
            probs = []
            x_i = self.data[ind, :]
            ## Integral part in posteriors（后验概率分母中的积分部分）:
            for g_cls in range(self.n_class):
                probs.append(self.class_prob[g_cls] * models[g_cls].pdf(x_i))
            probs = np.array(probs)
            probs /= probs.sum()
            total_probs.append(probs)
        return np.array(total_probs)

    def m_step(self, posterior):
        """Maximization step in EM algorithm, use last time posterior p(z|x)
        to calculate params gratitude.

        Args:
            posterior: [n_sample, n_class] p(z=i | x_i, \theta_t)

        Return:
            Each class param's gratitude in current time step
            grad_class_prob: scatter of class j
            grad_mus:        [,dim] jth class mus
            grad_sigma:      [, dim, dim] jth class sigma
        """
        for cls in range(self.n_class):
            ## class_prob gratitudes
            grad_class_prob = posterior[:, cls].sum() / self.n_sample

            ## mu_j <- (\sum_i p(z_j|x_i) * x_i) / sum_i p(z_j |x_i)
            grad_mus = np.zeros(self.n_dim)
            for ind in range(self.n_sample):
                grad_mus += posterior[ind, cls] * self.data[ind, :]
            grad_mus /= posterior[:, cls].sum()

            ## sigma_j <-  (\sum_i p(z_j|x_i) * (x_i - \mu_j)^2) / sum_i p(z_j |x_i)
            grad_sigma = np.zeros((self.n_dim, self.n_dim))
            for ind in range(self.n_sample):
                grad_sigma += posterior[ind, cls] * \
                        np.dot((self.data[ind, :] - self.mus[cls]), 
                                self.data[ind, :] - self.mus[cls].T)
            grad_sigma /= posterior[:, cls].sum()
            yield grad_class_prob, grad_mus, grad_sigma

```

这部分代码难点在于如何把数学公式转为代码，最直接明了的办法是将公式中的矩阵运算统一转换为 `for` 循环。$\sum_i^N \sum_j^{class}$ 就是两层循环。当然也可以直接用矩阵乘法，注意维度即可。而当连加的维度增长至三维以上，矩阵乘法的代码难度上升，此时用`for` 循环配合SIMD优化是工业界常用的解决方案。

### 参数更新

```python
class GMM:
     def fit(self, data):
        """process of gratitude dereasing of params in GMM
        """
        self.init_params(data)
        for e in range(self.n_epochs):
            ## e-step: 计算后验
            posterior = self.e_step()
            ## m-step: 计算梯度，并更新参数
            for cls, (grad_class, grad_mu, grad_sigma) in \
                zip(range(self.n_class), self.m_step(posterior)):
                self.class_prob[cls] += 1e-3 *grad_class
                self.mus[cls] += 1e-3 * grad_mu
                self.vars[cls] += 1e-3 * grad_sigma
            self.class_prob /= self.class_prob.sum()
            print (e)

    def pred(self, data):
        self.data = data
        self.n_sample = data.shape[0]
        assert self.n_dim == data.shape[1], "Wrong dim size !"
        res = self.e_step()
        return res.argmax(axis=1)
```

### 与sklearn比较下~

下面我们生成一个多维度算例，用sklearn的GMM包和我们手写的比对下：

```python

if __name__ == '__main__':
    n_class = 4
    n_dim = 4
    n_objects = 200

    ## generate gassian dist params:
    class_distance = 20
    class_diff = 2
    mus = [np.random.random(n_dim) * class_distance * i for i in range(1, n_class+1)]
    vars = [np.eye(n_dim) * class_diff * i for i in range(1, n_class+1)]

    ## use dirichlet dist to generate each gaussian class probability
    a = np.ones(n_class)
    n = 1
    p = len(a)
    rd = np.random.gamma(np.repeat(a, n), n, p)
    rd = np.divide(rd, np.repeat(np.sum(rd), p))
    theta = rd
    print (f'{n_class} classes prob: {theta}')

    ## use multinomial to generate each class's sample numbers
    r = np.random.multinomial(n_objects, theta)
    print(f'The number of objects in each classes from 1 to {n_class}: {r}')

    ## generate data:
    data = [np.random.multivariate_normal(mus[i], vars[i], r[i]) for i in range(0, n_class)]
    test_data = [np.random.multivariate_normal(mus[i], vars[i], 20) for i in range(0, n_class)]
    data=np.concatenate(data, axis=0).reshape(-1, n_dim)
    test_data = np.concatenate(test_data, axis=0).reshape(-1, n_dim)

    gmm = GMM(n_class=4)
    gmm.fit(data)
    pred = gmm.pred(test_data)
    print (pred, pred.shape)

    sk_gmm = GaussianMixture(n_components=4)
    sk_gmm.fit(data)
    pred = sk_gmm.predict(test_data)
    print (pred, pred.shape)
```

结果：

```bash
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] (80,)
[2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1] (80,)
```

不能说很相似，只能说一模一样。

完整代码[在此](https://github.com/LouisYZK/farewell/blob/main/articles/code/GMM.py)。

## 总结

- 高斯混合模型建模过程为多个高斯分布线性相加；不同的高斯分布组成视为模型中的隐变量。
- 高斯混合模型是生成模型：狄利克雷分布生成各类别的个数和概率；多项式分布生成各类别的样本数；多维高斯分布生成观测数据。
- 高斯分布的似然函数中含有对隐变量后验概率的连加，无法求出梯度的解析解。
- EM算法适合用于求解含有隐变量的生成模型，适合GMM的参数求解。对EM算法的核心公式求导$Q\left(\theta, \theta^{t}\right)=\sum_{i=1}^{N} \sum_{w_{i}} \log p\left(x_{i}, w_{i} \mid \theta\right) p\left(w_{i} \mid x_{i}, \theta^{t}\right)$ ，可以很直观地计算出各参数的梯度。
- 我们用python手写了一个适配各种类别数目、多维度高斯分布的GMM算法，与sklearn实现的结果一致。

## Reference

[1] https://github.com/GaborLengyel/Finite-Gaussian-Mixture-models/blob/master/Finite%20Gaussian%20Mixture%20models.ipynb

[2] 《统计机器学习》-EM算法及高斯混合模型，李航著

[3] [slearn-GMM](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_covariances.html#sphx-glr-auto-examples-mixture-plot-gmm-covariances-py)




