import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os

from scipy.stats import levy_stable, norm, kstest
from scipy.special import gamma




def format_plt(font_size=15):
    """
    Formats matplotlib to appear use Latex + Seaborn theme.
    """
    global plt
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],  # or another LaTeX font
            "axes.grid": False,
            "text.color": "black",
            "axes.titlesize": font_size,
            'axes.labelsize': font_size,
            'xtick.labelsize': font_size,
            'ytick.labelsize': font_size,
        }
    )
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.style.use("seaborn-v0_8-bright")


def symmetric_stable_scaling_factor(alpha, sigma_W2):
    # Assume symmetric, mu_W = 0
    sigma = np.sqrt(sigma_W2)
    raw_moments = (sigma ** alpha) * (2 ** (alpha / 2)) * gamma((alpha + 1) / 2) / np.sqrt(np.pi)
    tmp1 = gamma(2 - alpha) 
    tmp2 = np.cos(np.pi * alpha / 2)
    C_alpha = (1 - alpha) / (tmp1 * tmp2)
    factor = C_alpha ** (-1. / alpha)
    return factor * raw_moments


def plot_levy_normal_residuals(fig, ax, sigma_W2, truth, alpha, beta, title=None, xlabel=None, ylabel=None):
    # n_lines = 5
    # cmap = mpl.colormaps['tab20b']
    # colors = cmap(np.linspace(0, 1, n_lines))
    colors=['black', 'black', 'black']

    scale = symmetric_stable_scaling_factor(alpha=alpha, sigma_W2=sigma_W2)
    # scale=1
    y = np.diff([data.state_vector[0] for data in truth])
    # y /= scale
    # y = (y - y.mean()) / y.std()
    # fig, ax = plt.subplots(nrows=1, ncols=1)

    # ls_rvs = levy_stable(alpha=alpha, beta=beta, scale=scale)
    ls_rvs = levy_stable(alpha=alpha, beta=beta, scale=scale)
    x = np.linspace(ls_rvs.ppf(0.01), ls_rvs.ppf(0.99), 100)
    ax.plot(x, ls_rvs.pdf(x), '-', label=r'$\alpha$-stable PDF', color=colors[0])        
    ax.hist(y, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color=colors[2], label="Noise samples")
    ax.plot(x, norm.pdf(x), '--', label='Gaussian PDF', color=colors[1])    

    ax.set_xlim([x[0], x[-1]])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    results = kstest(y, ls_rvs.cdf) 
    print("KS test p-value:", results.pvalue)
    return fig

def save_fig(fig, filename, dir_="./assets/", form=".pdf"):
    path = os.path.join(dir_, filename + form)
    fig.savefig(path, bbox_inches='tight')

def plot_as_process_1d(fig, ax, truth, alpha, title=None, xlabel=None, ylabel=None, figsize=None):
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    y = np.array([data.state_vector[0] for data in truth])
    norm_rvs = norm.rvs(size=y.size)
    x = np.cumsum(norm_rvs)
    x -= x[0]
   
    # ax.plot(y, color='k', label=r"$\alpha$-stable process, $\alpha$" + f"={alpha}")
    ax.plot(y, color='k', label=r"$\alpha$-stable process")
    ax.plot(x, linestyle='dotted', color='k', label=r"Gaussian process")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gnvm_process_1d(fig, ax, truth, beta, nu, title=None, xlabel=None, ylabel=None, figsize=None):
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    y = np.array([data.state_vector[0] for data in truth])
    norm_rvs = norm.rvs(size=y.size)
    x = np.cumsum(norm_rvs)
    x -= x[0]
   
    # ax.plot(y, color='k', label=r"Gamma NVM process, $\beta$=" + f"{beta:.2f}, " + r"$\nu$="+ f"{nu:.2f}")
    ax.plot(y, color='k', label=r"VG Process")

    ax.plot(x, linestyle='dotted', color='k', label=r"Gaussian process")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_gnvm_residuals(fig, ax, truth, title=None, xlabel=None, ylabel=None):
    # n_lines = 5
    # cmap = mpl.colormaps['tab20b']
    # colors = cmap(np.linspace(0, 1, n_lines))
    colors=['black', 'black', 'black']

    y = np.diff([data.state_vector[0] for data in truth])
    # y = (y - np.mean(y)) / y.std()
    # fig, ax = plt.subplots(nrows=1, ncols=1)

    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)
    ax.plot(x, norm.pdf(x), '--', label='Gaussian PDF', color=colors[1])    
    ax.hist(y, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color=colors[2], label="Noise samples")
    ax.set_xlim([x[0], x[-1]])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    results = kstest(y, norm.cdf) 
    print("KS test p-value:", results.pvalue)
    return fig


def plot_tsnvm_process_1d(fig, ax, truth, beta, alpha, sigma_W2, title=None, xlabel=None, ylabel=None, figsize=None):
    # sigma_W = np.sqrt(sigma_W2)
    # scale = (2 ** alpha) * alpha * (1 / gamma(1 - alpha)) * sigma_W
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    y = np.array([data.state_vector[0] for data in truth])
    norm_rvs = norm.rvs(size=y.size)
    x = np.cumsum(norm_rvs)
    x -= x[0]
   
    # ax.plot(y, color='k', label=r"Tempered Stable NVM process, $\alpha$=" + f"{alpha:.2f}, " + r"$\beta$="+ f"{beta:.2f}")
    ax.plot(y, color='k', label=r"NTS process")
    ax.plot(x, linestyle='dotted', color='k', label=r"Gaussian process")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_tsnvm_residuals(fig, ax, truth, alpha, sigma_W2, title=None, xlabel=None, ylabel=None):
    colors=['black', 'black', 'black']
    y = np.diff([data.state_vector[0] for data in truth])
    y = (y - y.mean()) / y.std()
    # fig, ax = plt.subplots(nrows=1, ncols=1)

    x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)
    ax.hist(y, density=True, bins='auto', histtype='stepfilled', alpha=0.2, color=colors[2], label="Noise samples")
    ax.plot(x, norm.pdf(x), '--', label='Gaussian PDF', color=colors[1])    
    ax.set_xlim([x[0], x[-1]])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()

    results = kstest(y, norm.cdf) 
    print("KS test p-value:", results.pvalue)
    return fig





