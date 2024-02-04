import math
import torch
from torch import outer, matmul, inverse, mvlgamma, logdet, trace, digamma, unsqueeze, concat, zeros, softmax, tensor
from torch.special import gammaln
from zoo.helpers.MatPlotLib import MatPlotLib


class GaussianMixture:

    def __init__(self, x, W, m, v, β, d, r_hat=None, ε=1e-7):

        # Store a small number used for numerical stability.
        self.ε = ε

        # Store the data.
        self.data = concat([obs.unsqueeze(dim=0) for obs in x], dim=0)

        # Store the prior parameters.
        self.W = W
        self.m = m
        self.v = v
        self.β = β
        self.d = d

        # Initialize the posterior parameters.
        self.W_hat = [W_k.clone() for W_k in W]
        self.m_hat = [m_k.clone() for m_k in m]
        self.v_hat = v.clone()
        self.β_hat = β.clone()
        self.d_hat = d.clone()
        self.r_hat = softmax(torch.ones([x.shape[0], d.shape[0]]), dim=1) if r_hat is None else r_hat + ε

        # Pre-compute useful terms.
        self.digamma_d_hat, self.sum_d_hat, self.digamma_sum_d_hat, self.expected_log_D = self.pre_compute_d_terms()
        self.Ns, self.x_bar, self.S = self.pre_compute_z_terms()
        self.expected_log_det_Λ = self.pre_compute_μ_and_Λ_terms()
        self.ln2pi = math.log(2 * math.pi)
        self.ln2 = math.log(2)

        # The variational free energy.
        self.F = self.compute_vfe()

    def learn(self, debug=False, verbose=True, threshold=1):

        # Display debug information, if needed.
        if debug is True:
            print(f"Initial VFE: {self.vfe}")
            self.show("Gaussian Mixture: before optimization")

        # Perform variational inference, while the variational free energy has not converged.
        vfe = math.inf
        i = 0
        while abs(vfe - self.vfe) > threshold:

            # Update the current variational free energy.
            vfe = self.vfe

            # Perform variational inference.
            self.update_z()
            self.update_d()
            self.update_μ_and_Λ()
            self.update_vfe()

            # Display debug information, if needed.
            if verbose is True:
                i += 1
                print(f"Iteration {i}, VFE reduced by: {float(vfe - self.vfe)}, new VFE: {self.vfe}")

        # Display debug information, if needed.
        if debug is True:
            print(f"Final VFE: {self.vfe}")
            self.show("Gaussian Mixture: after optimization")

    def data_of_component(self, k):
        return self.x[self.r_hat.argmax(dim=1) == k]

    @property
    def active_components(self):
        return set(self.r_hat.argmax(dim=1).tolist())

    def params(self, k):
        return (
            self.W[k], self.m[k], self.v[k], self.β[k], self.d[k],
            self.W_hat[k], self.m_hat[k], self.v_hat[k], self.β_hat[k], self.d_hat[k]
        )

    def update_d(self):
        self.d_hat = self.d + self.Ns
        self.digamma_d_hat, self.sum_d_hat, self.digamma_sum_d_hat, self.expected_log_D = self.pre_compute_d_terms()

    def pre_compute_d_terms(self):
        digamma_d_hat = digamma(self.d_hat)
        sum_d_hat = self.d_hat.sum()
        digamma_sum_d_hat = digamma(sum_d_hat)
        expected_log_D = digamma_d_hat - digamma_sum_d_hat
        return digamma_d_hat, sum_d_hat, digamma_sum_d_hat, expected_log_D

    def update_z(self):

        # Compute the non-normalized state probabilities.
        expected_log_D = unsqueeze(self.expected_log_D, dim=0).repeat(self.N, 1)
        quadratic_form = self.expected_quadratic_form()
        log_det = unsqueeze(self.expected_log_det_Λ, dim=0).repeat(self.N, 1)
        log_ρ = zeros([self.N, self.K])
        log_ρ += expected_log_D - 0.5 * (self.K * self.ln2pi - log_det + quadratic_form)

        # Normalize the state probabilities.
        self.r_hat = softmax(log_ρ, dim=1) + self.ε

        # Pre-compute useful terms.
        self.Ns, self.x_bar, self.S = self.pre_compute_z_terms()

    def expected_quadratic_form(self):
        dataset_size = self.N
        n_states = self.K
        quadratic_form = zeros([dataset_size, n_states])
        for n in range(dataset_size):
            for k in range(n_states):
                diff = self.x[n] - self.m_hat[k]
                quadratic_form[n][k] = n_states / self.β_hat[k]
                quadratic_form[n][k] += self.v_hat[k] * matmul(matmul(diff.t(), self.W_hat[k]), diff)
        return quadratic_form

    def pre_compute_z_terms(self):
        Ns = self.compute_Ns()
        x_bar = self.compute_x_bar(Ns)
        S = [self.compute_S(Ns, x_bar, k) for k in range(self.K)]
        return Ns, x_bar, S

    def compute_Ns(self):
        return self.r_hat.sum(dim=0)

    def compute_x_bar(self, Ns):
        dataset_size = self.r_hat.shape[0]
        n_states = self.r_hat.shape[1]
        x_bar = [sum(self.r_hat[n][k] * self.x[n] for n in range(dataset_size)) / Ns[k] for k in range(n_states)]
        return concat([x_bar_k.unsqueeze(dim=0) for x_bar_k in x_bar], dim=0)

    def compute_S(self, Ns, x_bar, k):
        n_observations = x_bar.shape[1]
        S = zeros([n_observations, n_observations])
        for n in range(self.N):
            diff = self.x[n] - x_bar[k]
            S += self.r_hat[n][k] * outer(diff, diff)
        return S / Ns[k]

    def update_μ_and_Λ(self):

        # Update the parameters of the posterior over μ and Λ.
        self.v_hat = self.v + self.Ns
        self.β_hat = self.β + self.Ns
        self.m_hat = [(self.β[k] * self.m[k] + self.Ns[k] * self.x_bar[k]) / self.β_hat[k] for k in range(self.K)]
        self.W_hat = [self.compute_W_hat(self.Ns, k, self.x_bar) for k in range(self.K)]

        # Pre-compute useful terms.
        self.expected_log_det_Λ = self.pre_compute_μ_and_Λ_terms()

    def compute_W_hat(self, N, k, x_bar):
        W_hat = inverse(self.W[k]) + self.Ns[k] * self.S[k]
        x = x_bar[k] - self.m[k]
        W_hat += (self.β[k] * N[k] / self.β_hat[k]) * outer(x, x)
        return inverse(W_hat)

    def pre_compute_μ_and_Λ_terms(self):
        log_det = []
        n_states = self.K
        for k in range(n_states):
            digamma_sum = sum([digamma((self.v_hat[k] - i) / 2) for i in range(n_states)])
            log_det.append(n_states * math.log(2) + logdet(self.W_hat[k]) + digamma_sum)
        return tensor(log_det)

    @staticmethod
    def beta_ln(x):
        beta_ln = 0
        for i in range(x.shape[0]):
            beta_ln += gammaln(x[i])
        beta_ln -= gammaln(x.sum())
        return beta_ln

    def compute_vfe(self):

        # Pre-compute useful terms.
        log_β_hat = self.β_hat.log()
        log_β = self.β.log()

        # Add part of E[P(D)] and E[Q(D)].
        F = - self.beta_ln(self.d_hat) + self.beta_ln(self.d)

        for k in range(self.K):

            # Add E[P(Z|D)], as well as part of E[P(D)] and E[Q(D)].
            F += (self.d_hat[k] - self.d[k] - self.Ns[k]) * self.expected_log_D[k]

            # Add E[Q(μ|Λ)].
            F += 0.5 * (self.K * (log_β_hat[k] - self.ln2pi) + self.expected_log_det_Λ[k] - self.K)

            # Add E[P(μ|Λ)].
            diff = self.m_hat[k] - self.m[k]
            quadratic_form = (
                self.K * self.β[k] / self.β_hat[k] +
                self.β[k] * self.v_hat[k] * matmul(matmul(diff.t(), self.W_hat[k]), diff)
            )
            F -= 0.5 * (self.K * (log_β[k] - self.ln2pi) + self.expected_log_det_Λ[k] - quadratic_form)

            # Add E[Q(Λ)].
            F += -0.5 * (
                self.v_hat[k] * self.K * self.ln2 + self.v_hat[k] * logdet(self.W_hat[k]) -
                (self.v_hat[k] - self.K - 1) * self.expected_log_det_Λ[k] + self.v_hat[k] * self.K
            ) - mvlgamma(self.v_hat[k] / 2, self.K)

            # Add E[P(Λ)].
            F += 0.5 * (
                self.v[k] * self.K * self.ln2 + self.v[k] * logdet(self.W[k]) -
                (self.v[k] - self.K - 1) * self.expected_log_det_Λ[k] +
                self.v_hat[k] * trace(matmul(inverse(self.W[k]), self.W_hat[k]))
            ) - mvlgamma(self.v[k] / 2, self.K)

            # Add E[P(X|Z,μ,Λ)].
            diff = self.x_bar[k] - self.m[k]
            F -= 0.5 * self.Ns[k] * (
                self.expected_log_det_Λ[k] - self.K / self.β_hat[k] -
                self.v_hat[k] * trace(matmul(self.S[k], self.W[k])) -
                self.v_hat[k] * matmul(matmul(diff.t(), self.W[k]), diff) - self.K * self.ln2pi
            )

        # Add E[Q(Z)].
        log_r_hat = self.r_hat.log()
        F += (self.r_hat * log_r_hat).sum()
        return F

    def update_vfe(self):
        self.F = self.compute_vfe()

    def use_posterior_as_empirical_prior(self):
        self.W = self.W_hat
        self.m = self.m_hat
        self.v = self.v_hat
        self.β = self.β_hat
        self.d = self.d_hat

    def show(self, title="Gaussian Mixture"):
        MatPlotLib.draw_gm_graph(title=title, data=self.x, params=(self.m_hat, self.v_hat, self.W_hat), r=self.r_hat)

    @property
    def vfe(self):
        return self.F

    @property
    def x(self):
        return self.data

    @x.setter
    def x(self, new_data):
        self.data = torch.concat([obs.unsqueeze(dim=0) for obs in new_data], dim=0)
        self.update_z()

    @property
    def K(self):
        return len(self.W)

    @property
    def N(self):
        return len(self.data)
