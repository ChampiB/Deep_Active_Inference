from torch.distributions.multivariate_normal import MultivariateNormal
from torch import zeros, eye
from zoo.helpers.Device import Device
import torch


def entropy_gaussian(log_var, sum_dims=None):
    """
    Compute the entropy of a Gaussian distribution
    :param log_var: the logarithm of the variance parameter
    :param sum_dims: the dimensions along which to sum over before to return, by default only dimension one
    :return: the entropy of a Gaussian distribution
    """
    ln2pie = 1.23247435026
    sum_dims = [1] if sum_dims is None else sum_dims
    return log_var.size()[1] * 0.5 * ln2pie + 0.5 * log_var.sum(sum_dims)


def kl_div_categorical(pi_hat, pi):
    """
    Compute the KL-divergence between two categorical distribution.
    :param pi_hat: the parameters of the first categorical distribution.
    :param pi: the parameters of the second categorical distribution.
    :return: the KL-divergence.
    """
    shift = 0.00001
    kl = pi_hat * ((pi_hat + shift).log() - (pi + shift).log())
    return kl.sum()


def kl_div_gaussian(mean_hat, log_var_hat, mean=None, log_var=None, sum_dims=None, min_var=10e-4):
    """
    Compute the KL-divergence between two Gaussian distributions
    :param mean_hat: the mean of the first Gaussian distribution
    :param log_var_hat: the logarithm of variance of the first Gaussian distribution
    :param mean: the mean of the second Gaussian distribution
    :param log_var: the logarithm of variance of the second Gaussian distribution
    :param sum_dims: the dimensions along which to sum over before to return, by default all of them
    :param min_var: the minimal variance allowed to avoid division by zero
    :return: the KL-divergence between the two Gaussian distributions
    """

    # Initialise the mean and log variance vectors to zero, if they are not provided as parameters.
    if mean is None:
        mean = torch.zeros_like(mean_hat)
    if log_var is None:
        log_var = torch.zeros_like(log_var_hat)

    # Compute the KL-divergence
    var = log_var.exp()
    var = torch.clamp(var, min=min_var)
    kl_div = log_var - log_var_hat + torch.exp(log_var_hat - log_var) + (mean - mean_hat) ** 2 / var

    if sum_dims is None:
        return 0.5 * kl_div.sum(dim=1).mean()
    else:
        return 0.5 * kl_div.sum(dim=sum_dims)


def log_bernoulli_with_logits(obs, alpha):
    """
    Compute the log probability of the observation (obs), given the logits (alpha), assuming
    a bernoulli distribution, c.f.
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    :param obs: the observation
    :param alpha: the logits
    :return: the log probability of the observation
    """
    one = torch.ones_like(alpha)
    zero = torch.zeros_like(alpha)
    out = - torch.maximum(alpha, zero) + alpha * obs - torch.log(one + torch.exp(-torch.abs(alpha)))
    return out.sum(dim=(1, 2, 3)).mean()


def reparameterize(mean, log_var):
    """
    Perform the reparameterization trick
    :param mean: the mean of the Gaussian
    :param log_var: the log of the variance of the Gaussian
    :return: a sample from the Gaussian on which back-propagation can be performed
    """
    nb_states = mean.shape[1]
    epsilon = MultivariateNormal(zeros(nb_states), eye(nb_states)).sample([mean.shape[0]]).to(Device.get())
    return epsilon * torch.exp(0.5 * log_var) + mean


def compute_info_gain(g_value, mean, log_var, mean_hat, log_var_hat):
    """
    Compute the information gain.
    :param g_value: the definition of the efe to use, i.e., reward or efe.
    :param mean_hat: the mean from the encoder.
    :param log_var_hat: the log variance from the encoder.
    :param mean: the mean from the transition.
    :param log_var: the log variance from the transition.
    :return: the information gain.
    """
    info_gain = torch.zeros([1]).to(Device.get())
    if g_value == "old_efe":
        info_gain = -kl_div_gaussian(mean, log_var, mean_hat, log_var_hat)
    if g_value == "efe":
        info_gain = kl_div_gaussian(mean_hat, log_var_hat, mean, log_var)
    return info_gain
