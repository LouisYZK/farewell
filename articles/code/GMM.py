import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


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

    @property
    def n_explore_class(self):
        return 4

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

    def fit(self, data):
        """process of gratitude dereasing of params in GMM
        """
        self.init_params(data)
        for e in range(self.n_epochs):
            posterior = self.e_step()
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



if __name__ == '__main__':
    n_class = 6
    n_dim = 10
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
    # print (f'ACC: {}')

    sk_gmm = GaussianMixture(n_components=4)
    sk_gmm.fit(data)
    pred = sk_gmm.predict(test_data)
    print (pred, pred.shape)