import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable, norm
from scipy.special import gamma

seed = 1
alpha = 1.8
c = 1000
rng = np.random.default_rng(seed=seed)
dt = 1



def sample(n_samples):
  epochs = rng.exponential(scale=1 / dt, size=(int(c * dt), n_samples))
  epochs = epochs.cumsum(axis=0)
  jsize = np.power(epochs, -1. / alpha)
  sum0 = np.sum(jsize, axis=0)
  
  b_M = alpha / (alpha-1.) * (c ** ((alpha - 1.) / alpha)) if alpha >= 1.0 else 0
  samples = sum0 - b_M

  return samples


def stable_scaling_factor(alpha):
    tmp1 = gamma(2 - alpha) 
    tmp2 = np.cos(np.pi * alpha / 2)
    C_alpha = (1 - alpha) / (tmp1 * tmp2)
    factor = C_alpha ** (-1. / alpha)
    return factor



beta = 1.0

fig, ax = plt.subplots(nrows=1, ncols=1)
y = sample(10000)
scale = stable_scaling_factor(alpha=alpha)
x = np.linspace(levy_stable.ppf(0.01, alpha, beta, scale=scale),
                levy_stable.ppf(0.99, alpha, beta, scale=scale), 100)
_ = ax.plot(x, levy_stable.pdf(x, alpha, beta, scale=scale), '-', lw=5, alpha=0.6, label='levy_stable pdf') 

ax.hist(y, bins='auto', density=True)
ax.set_xlim([x[0], x[-1]])
plt.show()