import numpy as np
from numba import jit
from math import lgamma
from scipy.stats import beta

import matplotlib.pyplot as plt


@jit
def h(a, b, c, d):
    num = lgamma(a + c) + lgamma(b + d) + lgamma(a + b) + lgamma(c + d)
    den = lgamma(a) + lgamma(b) + lgamma(c) + lgamma(d) + lgamma(a + b + c + d)
    return np.exp(num - den)


@jit
def g0(a, b, c):
    return np.exp(lgamma(a + b) + lgamma(a + c) - (lgamma(a + b + c) + lgamma(a)))


@jit
def hiter(a, b, c, d):
    while d > 1:
        d -= 1
        yield h(a, b, c, d) / d


def g(a, b, c, d):
    return g0(a, b, c) + sum(hiter(a, b, c, d))


def calc_prob_between(beta1, beta2):
    return g(beta1.args[0], beta1.args[1], beta2.args[0], beta2.args[1])


class Bayesian:
    """Class calculating the"""

    def __init__(
        self,
        A_impressions: int,
        A_conversions: int,
        B_impressions: int,
        B_conversions: int,
    ) -> None:
        self.a_imp = A_impressions
        self.a_conv = A_conversions
        self.b_imp = B_impressions
        self.b_conv = B_conversions

        self.A_beta, self.B_beta = self._calc_beta_dist()

    def _calc_beta_dist(self) -> None:
        alpha_A, beta_A = self.a_conv + 1, self.a_imp + 1
        alpha_B, beta_B = self.b_conv + 1, self.b_imp + 1
        A_beta = beta(alpha_A, beta_A)
        B_beta = beta(alpha_B, beta_B)
        return A_beta, B_beta

    def _get_prop(self):
        return calc_prob_between(self.B_beta, self.A_beta)

    def get_uplift(self):
        self.uplift = (self.B_beta.mean() - self.A_beta.mean()) / self.A_beta.mean()
        self.prob = self._get_prop()

        result = f"""
        Uplift using B compared to A = {round(self.uplift*100, 1)}%
        \n
        With a probability of {round(self.prob*100, 1)}%
        
        """

        return result

    def plot(self, names=["Group A", "Group_B"], x_start=0, x_stop=0.05):
        x = np.linspace(x_start, x_stop, 100)
        betas = [self.A_beta, self.B_beta]
        for f, name in zip(betas, names):
            y = f.pdf(x)
            plt.plot(x, y, label=f"{name}")
        plt.legend()
        plt.show()


class Frequentist:
    pass
