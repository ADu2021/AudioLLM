from scipy.special import binom, betaln
import numpy as np

def beta_binomial_pmf(n, k, alpha, beta):
    """
    Compute the probability mass function of the beta-binomial distribution.
    
    :param n: number of trials
    :param k: number of successes
    :param alpha: alpha parameter of the Beta distribution
    :param beta: beta parameter of the Beta distribution
    :return: probability of observing k successes out of n trials
    """
    if alpha == 0:
        return 1 if (k == 0) else 0
    log_pmf = (np.log(binom(n, k)) +
               betaln(k + alpha, n - k + beta) -
               betaln(alpha, beta))
    return np.exp(log_pmf)
def get_prior(text_length, unit_length, w=1.0):
    prior = [[beta_binomial_pmf(text_length, k, w*t, w*(unit_length-t+1)) for k in range(text_length)] for t in range(unit_length)]
    # shape [unit_length, text_length]
    return prior
# Example usage
n = 10  # number of trials
alpha = 2  # alpha parameter of the Beta distribution
beta = 2   # beta parameter of the Beta distribution
k_values = np.arange(0, n+1)  # possible values of k from 0 to n
pmf_values = [beta_binomial_pmf(n, k, alpha, beta) for k in k_values]

print("k values:", k_values)
print("PMF values:", pmf_values)

print()
import ipdb; ipdb.set_trace();
get_prior(20, 300, 2)