import numpy as np
from scipy.special import gammainc, gammaincc, gamma
from abc import ABC, abstractmethod


class CustomException(Exception):
    pass


"""
Instead of letting Gaussian driver generate mean and covariance,
directly generate noise. Model should be deterministic, while
noise driver should be random (seeded). Better separation then
having both the model and driver be random.

As the system states and matrices will be obtained directly from
the model, forcing the Gaussian driver to produce a mean vector and
covariance matrices introduces additional complexity to the overall
state space equation, rather than just passting the noise 
parameters to the random normal generator directly.

In research literature, it is common to use a matrix to represent
multi-dimensional states. In Stone Soup, each model here is treated
as an independent sytem and uses for loop instead of diagonal matrices.

Instead of using diagonal matrices, assign a priority/score to each model.
If the same driver is being shared across multiple models, assign a score
0 as main, while others are assigned 1 as slaves.

Or just use a list keeping references of each model bound the driver.

Caching by time stamp just feels wrong.

A limitation of the master-slave issue: the master must be "simulated" first
before the slaves. Assume the first model called to be the master. Or else,
the slave would use previously generated jumps. Hence, the master must always
come before the slaves. 
"""


class BaseDriver(ABC):
    """
    This class should not be initialised. It is parent class to handle the
    master-slave interaction for sharing noise drivers across multiple classes
    """

    def __init__(self, seed):
        self._models = set()
        self._master = None
        self._rng = np.random.default_rng(seed=seed)

    def elect_master(self, model):
        assert model in self._models
        self._master = model

    def add_model(self, model):
        # First model added is master by default
        if self._master is None:
            self._master = model
        self._models.add(model)

    def _model_is_master(self, model):
        assert self._master is not None
        return id(self._master) == id(model)
    
    @abstractmethod
    def noise(self):
        """
        returns driving noise term
        """
        pass
    

class GaussianDriver(BaseDriver):
    def __init__(self, mu_W, sigma_W, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._noises = []
        self._mu_W = mu_W
        self._sigma_W = sigma_W

    def noise(self, model, dt):
        if self._model_is_master(model):
            # Isotropic noise
            omega_func = model.omega_func(dt)
            omega = omega_func()
            R = (
                np.linalg.cholesky(omega)
                if np.all(np.linalg.eigvals(omega) > 0)
                else np.zeros_like(omega)
            )
            dims = R.shape[0]
            noise = R @ self._rng.normal(loc=self._mu_W, scale=self._sigma_W, size=(dims, 1))
            # print(noise)
            self._noises.append(noise)
        else:
            assert self._noises
            noise = self._noises[-1]  # Take last noise generated
        
        return noise


class NonGaussianDriver(BaseDriver):
    def __init__(self, c, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._Gammavs = []  # Gammav latent
        self._Vvs = []  # Vv latent
        self._c = c

    def _latents(self, model, dt, rate=1.):
        """
        Sample gammav and vv is master, else return from hist
        """
        if self._model_is_master(model):
            cutoff = self._c * dt
            init = self._rng.exponential(scale=1.0, size=max(int(np.ceil(1.1 * cutoff)), 1)).cumsum()
            epochs = [init]
            last_epoch = init[-1]

            while last_epoch < cutoff:
                epoch_seq = self._rng.exponential(scale=rate, size=int(np.ceil(0.1 * cutoff)))
                epoch_seq[0] += last_epoch
                epoch_seq = epoch_seq.cumsum()  # generates a sequence of time stamps
                last_epoch = epoch_seq[-1]  
                epochs.append(epoch_seq)

            epochs = np.concatenate(epochs)
            epochs = epochs[epochs < cutoff]
            jtimes = self._rng.uniform(low=0.0, high=dt, size=epochs.size)
            self._Gammavs.append(epochs)
            self._Vvs.append(jtimes)
        else:
            epochs = self._Gammavs[-1]
            jtimes = self._Vvs[-1]
        
        return epochs, jtimes

    @abstractmethod
    def _hfunc(self, gammav):
        """
        returns jump sizes
        """
        pass


class AlphaStableDriver(NonGaussianDriver):
    def __init__(self, alpha, mu_W, sigma_W, noise_case=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alpha = alpha
        self._mu_W = mu_W
        self._sigma_w = sigma_W
        self._noise_case = noise_case

    def _hfunc(self, gammav):
        return np.power(gammav, -1.0 / self._alpha)

    def _Ac(self):
        if 1 <= self._alpha < 2:
            term1 = self._mu_W
            term2 = self._alpha / (self._alpha - 1)
            term3 = self._c ** (1 - (1 / self._alpha))
            return term1 * term2 * term3
        elif 0 < self._alpha < 1:
            return 0
        else:
            raise CustomException("alpha must be 0 < alpha < 2")
    
    def _mean(self, model, dt, jsizes, jtimes):
        ft_func = model.ft_func(dt)
        ft = ft_func(jtimes)
        m = (dt ** (1.0 / self._alpha)) * np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)
        return self._mu_W * m

    def _covar(self, model, dt, jsizes2, jtimes):
        ft2_func = model.ft2_func(dt)
        ft2 = ft2_func(jtimes)
        s = (dt ** (2.0 / self._alpha)) * np.sum(jsizes2.reshape(-1, 1, 1) * ft2, axis=0)
        return (self._sigma_w ** 2) * s

    def _residual_constant(self):
        return self._alpha / (2.0 - self._alpha) * np.power(self._c, 1.0 - 2.0 / self._alpha)

    def _residual(self, model, dt):
        E_ft2_func = model.E_ft2_func(dt)
        E_ft2 = E_ft2_func()
        # print(E_ft2)
        if self._noise_case == 1:
            Sigma = 0
        elif self._noise_case == 2:
            sigma_W2 = self._sigma_w ** 2
            mu_W2 = self._mu_W ** 2
            Sigma = (sigma_W2 + mu_W2) * E_ft2
        elif self._noise_case == 3:
            Sigma = self._sigma_W2 * E_ft2
        else:
            raise CustomException("invalid noise case")
        return self._residual_constant() * Sigma

    def _centring(self, model, dt):
        E_ft_func = model.E_ft_func(dt)
        E_ft = E_ft_func()
        return -self._Ac() * E_ft

    def noise(self, model, dt):
        assert hasattr(model, "A")
        assert hasattr(model, "h")
        epochs, jtimes = self._latents(model, dt)
        jsizes = self._hfunc(epochs)
        jsizes2 = jsizes**2

        noise_mean = self._mean(model, dt, jsizes, jtimes) + self._centring(model, dt)
        noise_covar = self._covar(model, dt, jsizes2, jtimes) + self._residual(model, dt)
        
        # print(noise_covar)
        R = np.linalg.cholesky(noise_covar) if np.all(np.linalg.eigvals(noise_covar) > 0) else np.zeros_like(noise_covar)

        dims = R.shape[0]
        noise = noise_mean + R @ self._rng.normal(size=(dims, 1))
        return noise



class NormalGammaDriver(NonGaussianDriver):
    """
    This is a mean mixture levy process, not sure if i should make
    a separate parent class.

    Recall the form of the Lévy measure of NVM processes from (6). 
    An NVM process need not be compensated, and hence we may take 
    bi = 0 for all i in the shot noise representation, provided,
    """
    def __init__(self, nu, beta, mu_W, sigma_W, noise_case=2, *args, **kwargs):
        """
        Compared to Barndorff-Nielson
        beta = gamma**2/2
        C = ni
        """
        super().__init__(*args, **kwargs)
        self._nu = nu # scale parameter
        self._beta = beta # shape parameter
        self._mu_W = mu_W
        self._sigma_w = sigma_W
        self._noise_case = noise_case
        self.debug_jumps = []
        print("Truncation:", self._hfunc(self._c)) # for debug

    def _hfunc(self, gammav):
        return 1. / (self._beta * (np.exp(gammav / self._nu) - 1.))

    def _thinning(self, z):
        return (1. + self._beta * z) * np.exp(-self._beta * z)

    def _mean(self, model, dt, jsizes, jtimes):
        """
        Try alpha = 2? Not sure if delta_t term is correct
        Actually i think there is none, need to confirm
        From the point process simulation paper, the jump times 
        are not limited to a unit interval (see Algorithm2)
        """
        ft_func = model.ft_func(dt)
        ft = ft_func(jtimes)
        # m = (dt ** (1.0 / 2)) * np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)
        m = np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)
        return self._mu_W * m

    def _covar(self, model, dt, jsizes2, jtimes):
        """
        Try alpha = 2? Not sure if delta_t term is correct
        Actually i think there is none, need to confirm
        """
        ft2_func = model.ft2_func(dt)
        ft2 = ft2_func(jtimes)
        # s = (dt ** (2.0 / 2)) * np.sum(jsizes2.reshape(-1, 1, 1) * ft2, axis=0)
        s = np.sum(jsizes2.reshape(-1, 1, 1) * ft2, axis=0)
        return (self._sigma_w ** 2) * s
    
    def _residual(self, model, dt):
        def incgammal(s, x):
            return gammainc(s, x) * gamma(s)

        def unit_expected_residual_jumps(): #M(1)
            return (self._nu / self._beta) * incgammal(1., self._beta * truncation)

        def unit_variance_residual_jumps(): #M(2)
            return (self._nu / self._beta ** 2) * incgammal(2., self._beta * truncation)

        truncation = self._hfunc(self._c * dt) # epsilon in paper
        E_ft2_func = model.E_ft2_func(dt)
        E_ft2 = E_ft2_func()

        if self._noise_case == 1:
            cov_const0 = 0.
            cov_const1 = 0.            
        elif self._noise_case == 2:
            cov_const0 = unit_expected_residual_jumps()
            cov_const1 = unit_variance_residual_jumps()
        elif self._noise_case == 3:
            cov_const0 = unit_expected_residual_jumps()
            cov_const1 = 0.
        else:
            raise CustomException("invalid noise case")
        
        var = (self._sigma_w ** 2) * cov_const0
        mu2 = (self._mu_W ** 2) * cov_const1
        return (mu2 + var) * E_ft2
    
    def noise(self, model, dt):
        assert hasattr(model, "A")
        assert hasattr(model, "h")
        
        epochs, jtimes = self._latents(model, dt)

        # Rejection sampling
        z = self._hfunc(epochs)
        thinning_prob = self._thinning(z)
        u = self._rng.uniform(low=0., high=1.0, size=thinning_prob.shape)
        z = z[u < thinning_prob]
        jtimes = jtimes[u < thinning_prob]        
        jsizes = self._mu_W * z + self._sigma_w * np.sqrt(z) * self._rng.normal(size=z.shape)
        jsizes2 = jsizes ** 2
        self.debug_jumps.append(np.sum(jsizes))
        noise_mean = self._mean(model, dt, jsizes, jtimes) 
        noise_covar = self._covar(model, dt, jsizes2, jtimes) + self._residual(model, dt)

        """
        Is there no centering term?
        """

        R = np.linalg.cholesky(noise_covar) if np.all(np.linalg.eigvals(noise_covar) > 0) else np.zeros_like(noise_covar)
        dims = R.shape[0]
        noise = noise_mean + R @ self._rng.normal(size=(dims, 1))
        return noise


class NormalTemperedStableDriver(NonGaussianDriver):
    def __init__(self, nu, beta, kappa, mu_W, sigma_W, noise_case=2, *args, **kwargs):
        assert (0. < kappa < 1.)
        assert (nu > 0.0 and beta >= 0.0)
        super().__init__(*args, **kwargs)
        self._nu = nu # scale parameter
        self._beta = beta # shape parameter
        self._kappa = kappa
        self._mu_W = mu_W
        self._sigma_w = sigma_W
        self._noise_case = noise_case
        self.debug_jumps = []
        print("Truncation:", self._hfunc(self._c)) # for debug

    def _hfunc(self, gammav):
        return np.power((self._kappa / self._nu) * gammav, -1. / self._kappa)

    def _thinning(self, z):
        return  np.exp(-self._beta * z)

    def _mean(self, model, dt, jsizes, jtimes):
        ft_func = model.ft_func(dt)
        ft = ft_func(jtimes)
        # m = (dt ** (1.0 / 2)) * np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)
        m = np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)
        return self._mu_W * m

    def _covar(self, model, dt, jsizes2, jtimes):
        ft2_func = model.ft2_func(dt)
        ft2 = ft2_func(jtimes)
        # s = (dt ** (2.0 / 2)) * np.sum(jsizes2.reshape(-1, 1, 1) * ft2, axis=0)
        s = np.sum(jsizes2.reshape(-1, 1, 1) * ft2, axis=0)
        return (self._sigma_w ** 2) * s
    
    def _residual(self, model, dt):
        def incgammal(s, x):
            return gammainc(s, x) * gamma(s)

        def unit_expected_residual_jumps(): #M(1)
            return (self._nu * self._beta ** (self._kappa - 1.)) * incgammal(1. - self._kappa, self._beta * truncation)

        def unit_variance_residual_jumps(): #M(2)
            return (self._nu * self._beta ** (self._kappa - 2.)) * incgammal(2. - self._kappa, self._beta * truncation)

        truncation = self._hfunc(self._c * dt) # epsilon in paper
        E_ft2_func = model.E_ft2_func(dt)
        E_ft2 = E_ft2_func()

        if self._noise_case == 1:
            cov_const0 = 0.
            cov_const1 = 0.            
        elif self._noise_case == 2:
            cov_const0 = unit_expected_residual_jumps()
            cov_const1 = unit_variance_residual_jumps()
        elif self._noise_case == 3:
            cov_const0 = unit_expected_residual_jumps()
            cov_const1 = 0.
        else:
            raise CustomException("invalid noise case")
        
        var = (self._sigma_w ** 2) * cov_const0
        mu2 = (self._mu_W ** 2) * cov_const1
        return (mu2 + var) * E_ft2
    
    def noise(self, model, dt):
        assert hasattr(model, "A")
        assert hasattr(model, "h")
        
        epochs, jtimes = self._latents(model, dt)

        # Rejection sampling
        z = self._hfunc(epochs)
        thinning_prob = self._thinning(z)
        u = self._rng.uniform(low=0., high=1., size=thinning_prob.shape)
        z = z[u < thinning_prob]
        jtimes = jtimes[u < thinning_prob]
        
        jsizes = (self._mu_W * z) + (self._sigma_w * np.sqrt(z)) * self._rng.normal(size=z.shape)
        jsizes2 = jsizes ** 2
        self.debug_jumps.append(np.sum(jsizes))
        noise_mean = self._mean(model, dt, jsizes, jtimes) 
        noise_covar = self._covar(model, dt, jsizes2, jtimes) + self._residual(model, dt)

        """
        Is there no centering term?
        """

        R = np.linalg.cholesky(noise_covar) if np.all(np.linalg.eigvals(noise_covar) > 0) else np.zeros_like(noise_covar)
        dims = R.shape[0]
        noise = noise_mean + R @ self._rng.normal(size=(dims, 1))
        return noise
