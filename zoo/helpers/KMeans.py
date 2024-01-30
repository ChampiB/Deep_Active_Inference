from random import randint
import torch


class KMeans:
    """
    Implement the K-means algorithm taking random each action.
    """

    @staticmethod
    def run(x, n_clusters):
        μ = [x[randint(0, len(x) - 1)] for _ in range(n_clusters)]
        r = torch.zeros([len(x), n_clusters])
        for i in range(20):  # TODO implement a better stopping condition
            r = KMeans.update_responsibilities(x, μ)
            μ = KMeans.update_means(x, r)
        return μ, r

    @staticmethod
    def update_responsibilities(x, μ):
        r = torch.zeros([len(x), len(μ)])
        for n in range(len(x)):
            distances = [torch.pow(x[n] - μ[k], 2).sum().sqrt() for k in range(len(μ))]
            best_k = distances.index(min(distances))
            r[n][best_k] = 1
        return r

    @staticmethod
    def update_means(x, r):
        μ = []
        sum_r = r.sum(dim=0)
        for k in range(r.shape[1]):
            μ_k = sum([r[n][k] * x[n] for n in range(len(x))]) / (sum_r[k] + 0.00001)
            μ.append(μ_k)
        return μ

    @staticmethod
    def precision(x, r):
        precision = []
        for k in range(r.shape[1]):
            x_k = [torch.unsqueeze(x[n], dim=0) for n in range(len(x)) if r[n][k] == 1]
            if len(x_k) >= 2:
                x_k = torch.concat(x_k, dim=0).t()
                precision.append(torch.inverse(torch.cov(x_k)))
            else:
                precision.append(torch.eye(x[0].shape[0]))
        return precision
