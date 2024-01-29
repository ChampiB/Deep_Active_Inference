import math
import torch
from zoo.helpers.KMeans import KMeans


class GaussianMixture:
    """
    Implement useful function for inference in Gaussian mixture models.
    """

    @staticmethod
    def expected_log_det_Λ(v_hat, W_hat, dataset_size=-1):
        log_det = []
        n_states = len(v_hat)
        for k in range(n_states):
            digamma_sum = sum([torch.digamma((v_hat[k] - i) / 2) for i in range(n_states)])
            log_det.append(n_states * math.log(2) + torch.logdet(W_hat[k]) + digamma_sum)
        log_det = torch.tensor(log_det)
        return log_det if dataset_size <= 0 else GaussianMixture.repeat_across(log_det, dataset_size, dim=0)

    @staticmethod
    def expected_quadratic_form(x, m_hat, β_hat, v_hat, W_hat):
        dataset_size = len(x)
        n_states = len(v_hat)
        quadratic_form = torch.zeros([dataset_size, n_states])
        for n in range(dataset_size):
            for k in range(n_states):
                diff = x[n] - m_hat[k]
                quadratic_form[n][k] = n_states / β_hat[k]
                quadratic_form[n][k] += v_hat[k] * torch.matmul(torch.matmul(diff.t(), W_hat[k]), diff)
        return quadratic_form

    @staticmethod
    def repeat_across(x, n, dim):
        """
        Repeat a 1D tensor across a newly created dimension
        :param x: the tensor
        :param n: the size of the newly created dimension
        :param dim: the index at which the dimension should be created
        :return: the expanded data
        """
        return torch.unsqueeze(x, dim=dim).repeat(n, 1)

    @staticmethod
    def expected_log_D(d, dataset_size=-1):
        sum_d = d.sum()
        log_D = torch.digamma(d) - torch.digamma(sum_d)
        return log_D if dataset_size <= 0 else GaussianMixture.repeat_across(log_D, dataset_size, dim=0)

    @staticmethod
    def expected_log_B(b):
        n_states = b.shape[1]
        digamma_sum_b = torch.digamma(b.sum(dim=1)).unsqueeze(dim=1).repeat(1, n_states, 1)
        return torch.digamma(b) - digamma_sum_b

    @staticmethod
    def k_means_init(x, v, v_hat):

        # Retrieve the number of states.
        n_states = len(v)

        # Perform K-means to initialize the parameter of the posterior over latent variables at time step 1.
        μ, r = KMeans.run(x[-1], n_states)

        # Estimate the covariance of the clusters and use it to initialize the Wishart prior and posterior.
        precision = KMeans.precision(x[-1], r)
        W_hat = [precision[k] / v_hat[k] for k in range(n_states)]
        W = [precision[k] / v[k] for k in range(n_states)]
        if len(x) == 1:
            return μ, μ, r, W_hat, W

        # Perform K-means to initialize means of prior and posterior distributions at time step 0.
        r0 = KMeans.update_responsibilities(x[0], μ)
        return μ, μ, r, W_hat, W, r0

    @staticmethod
    def gm_x_bar(r, x, N):
        dataset_size = r.shape[0]
        n_states = r.shape[1]
        return [sum(r[n][k] * x[n] for n in range(dataset_size)) / N[k] for k in range(n_states)]

    @staticmethod
    def tgm_x_bar(r, x0, x1, N):
        dataset_size = r[0].shape[0]
        n_states = r[0].shape[1]
        return [
            sum([r[0][n][k] * x0[n] + r[1][n][k] * x1[n] for n in range(dataset_size)]) / N[k]
            for k in range(n_states)
        ]
