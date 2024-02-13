from mystonesoup.base import Driver, ExtendedModel, CustomException, NonGaussianDriver
from stonesoup.base import Base, Property
from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix
from typing import Iterable, Tuple
from datetime import timedelta
from abc import abstractmethod
from scipy.special import gammainc, gammaincc, gamma
import numpy as np


class GaussianDriver(Driver):
    mu_W: float = Property(doc="Gaussian mean.")
    sigma_W: float = Property(doc="Gaussian standard deviation.")
    _noises: Iterable[np.ndarray] = Property(default=None, doc="Noise history.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._noises = []

    def noise(self, num_samples: int, model: ExtendedModel, time_interval: timedelta) -> np.ndarray:
        if self._model_is_master(model):
            # Isotropic noise
            covar = self.covar(model=model, time_interval=time_interval)
            R = (
                np.linalg.cholesky(covar)
                if np.all(np.linalg.eigvals(covar) > 0)
                else np.zeros_like(covar)
            )
            noise = R @ self._rng.normal(
                loc=self.mu_W, scale=self.sigma_W, size=(R.shape[1], num_samples)
            )
            # print(noise)
            self._noises.append(noise)
        else:
            assert self._noises
            noise = self._noises[-1]  # Take last noise generated
        return noise

    def covar(self, model: ExtendedModel, time_interval: timedelta) -> CovarianceMatrix:
        dt = time_interval.total_seconds()
        omega_func = model.omega_func(dt=dt)
        omega = omega_func()
        return omega

    def mean(self, model: ExtendedModel, **kwargs) -> StateVector:
        ndim = model.ndim_state
        return np.ones(ndim) * self.mu_W


class AlphaStableDriver(NonGaussianDriver):
    alpha: float = Property(doc="Alpha parameter.")
    mu_W: float = Property(doc="Conditional Gaussian mean.")
    sigma_W: float = Property(doc="Conditional Gaussian variance.")
    noise_case: int = Property(doc="Noise case, either 1, 2 or 3.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self._alpha = alpha
        # self._mu_W = mu_W
        # self._sigma_W = sigma_W
        # self.noise_case = noise_case

    def _hfunc(self, gammav: np.ndarray) -> np.ndarray:
        return np.power(gammav, -1.0 / self.alpha)

    def _Ac(self) -> float:
        if 1 <= self.alpha < 2:
            term1 = self.mu_W
            term2 = self.alpha / (self.alpha - 1)
            term3 = self.c ** (1 - (1 / self.alpha))
            return term1 * term2 * term3
        elif 0 < self.alpha < 1:
            return 0
        else:
            raise CustomException("alpha must be 0 < alpha < 2")

    def _mean(
        self, model: ExtendedModel, dt: float, jsizes: np.ndarray, jtimes: np.ndarray
    ) -> np.ndarray:
        ft_func = model.ft_func(dt=dt)
        ft = ft_func(jtimes=jtimes)
        # print(jsizes.shape)
       
        m = (dt ** (1.0 / self.alpha)) * np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)

        return self.mu_W * m

    def _covar(
        self, model: ExtendedModel, dt: float, jsizes2: np.ndarray, jtimes: np.ndarray
    ) -> np.ndarray:
        
        ft2_func = model.ft2_func(dt=dt)
        ft2 = ft2_func(jtimes=jtimes)
        # print(ft2.shape)
        # print(jsizes2)
        s = (dt ** (2.0 / self.alpha)) * np.sum(jsizes2.reshape(-1, 1, 1) * ft2, axis=0)
        omega_func = model.omega_func(dt=dt)
        omega = omega_func()
        
        # raise Excepti on
        return omega + (self.sigma_W**2) * s

    def _residual_constant(self) -> float:
        return self.alpha / (2.0 - self.alpha) * np.power(self.c, 1.0 - 2.0 / self.alpha)

    def _residual(self, model: ExtendedModel, dt: float) -> np.ndarray:
        E_ft2_func = model.E_ft2_func(dt=dt)
        E_ft2 = E_ft2_func()
        if self.noise_case == 1:
            Sigma = 0
        elif self.noise_case == 2:
            sigma_W2 = self.sigma_W**2
            mu_W2 = self.mu_W**2
            Sigma = (sigma_W2 + mu_W2) * E_ft2
        elif self.noise_case == 3:
            sigma_W2 = self.sigma_W**2
            Sigma = sigma_W2 * E_ft2
        else:
            raise CustomException("invalid noise case")
        return self._residual_constant() * Sigma

    def _centring(self, model: ExtendedModel, dt: float) -> np.ndarray:
        E_ft_func = model.E_ft_func(dt=dt)
        E_ft = E_ft_func()
        return -self._Ac() * E_ft

    def noise(self, num_samples: int, model: ExtendedModel, time_interval: timedelta) -> np.ndarray:
        dt = time_interval.total_seconds()
        epochs, jtimes = self.latents(model=model, dt=dt, num_samples=num_samples)
    
        noise_mean = self.mean(model=model, time_interval=time_interval, epochs=epochs, jtimes=jtimes)  
        noise_covar = self.covar(model=model, time_interval=time_interval, epochs=epochs, jtimes=jtimes) 
        R = (
            np.linalg.cholesky(noise_covar)
            if np.all(np.linalg.eigvals(noise_covar) > 0)
            else np.zeros_like(noise_covar)
        )
        noise = noise_mean + R @ self._rng.normal(size=(R.shape[1], num_samples))
        return noise

    def covar(self, model: ExtendedModel, time_interval: timedelta, epochs: np.ndarray, jtimes: np.ndarray, *args, **kwargs) -> CovarianceMatrix:
        jsizes = self._hfunc(gammav=epochs)
        jsizes2 = jsizes**2
        dt = time_interval.total_seconds()
        return self._covar(
            model=model, dt=dt, jsizes2=jsizes2, jtimes=jtimes
        ) + self._residual(model=model, dt=dt)


    def mean(self, model: ExtendedModel, time_interval: timedelta, epochs: np.ndarray, jtimes: np.ndarray, *args, **kwargs) -> StateVector:
        jsizes = self._hfunc(gammav=epochs)
        dt = time_interval.total_seconds()
        return self._mean(model=model, dt=dt, jsizes=jsizes, jtimes=jtimes) + self._centring(
            model=model, dt=dt
        )


class NormalGammaDriver(NonGaussianDriver):
    nu: float = Property(doc="Scale parameter")
    beta: float = Property(doc="Shape parameter")
    mu_W: float = Property(doc="Conditional Gaussian mean.")
    sigma_W: float = Property(doc="Conditional Gaussian variance.")
    noise_case: int = Property(doc="Noise case, either 1, 2 or 3.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self._alpha = alpha
        # self._mu_W = mu_W
        # self._sigma_W = sigma_W
        # self.noise_case = noise_case
        self._jtimes_cache = np.zeros(1)
        self._jsizes_cache = np.zeros(1)

    def _hfunc(self, gammav: np.ndarray) -> np.ndarray:
        return 1. / (self.beta * (np.exp(gammav / self.nu) - 1.))
    
    def latents(self, *args, **kwargs):
        # Whenever latents are generated, clear jtimes and jsizes cache
        self._jtimes_cache = None
        self._jsizes_cache = None
        return super().latents(*args, **kwargs)
        
    def get_jumps(self, epochs: np.ndarray, jtimes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._jtimes_cache is None:
            z = self._hfunc(gammav=epochs)
            thinning_prob = self._thinning(z)
            u = self._rng.uniform(low=0., high=1., size=thinning_prob.shape)
            z = z[u < thinning_prob]
            self._jtimes_cache = jtimes[u < thinning_prob]
            self._jsizes_cache = (self.mu_W * z) + (self.sigma_W * np.sqrt(z)) * self._rng.normal(size=z.shape)

        return self._jsizes_cache, self._jtimes_cache

    def _thinning(self, z: np.ndarray):
        return (1. + self.beta * z) * np.exp(-self.beta * z)
    

    def _mean(
        self, model: ExtendedModel, dt: float, jsizes: np.ndarray, jtimes: np.ndarray
    ) -> np.ndarray:
        ft_func = model.ft_func(dt=dt)
        ft = ft_func(jtimes=jtimes)
        # Not sure if we need to account for delta_t for non-unit time interval
        m = np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)

        return self.mu_W * m

    def _covar(
        self, model: ExtendedModel, dt: float, jsizes2: np.ndarray, jtimes: np.ndarray
    ) -> np.ndarray:
        
        ft2_func = model.ft2_func(dt=dt)
        ft2 = ft2_func(jtimes=jtimes)
        # Not sure if we need to account for delta_t for non-unit time interval
        s = np.sum(jsizes2.reshape(-1, 1, 1) * ft2, axis=0)
        omega_func = model.omega_func(dt=dt)
        omega = omega_func()
        
        return omega + (self.sigma_W**2) * s

    def _residual(self, model: ExtendedModel, dt: float) -> np.ndarray:
        def incgammal(s, x):
            return gammainc(s, x) * gamma(s)

        def unit_expected_residual_jumps(): #M(1)
            return (self.nu / self.beta) * incgammal(1., self.beta * truncation)

        def unit_variance_residual_jumps(): #M(2)
            return (self.nu / self.beta ** 2) * incgammal(2., self.beta * truncation)

        truncation = self._hfunc(self.c * dt) # epsilon in paper
        E_ft2_func = model.E_ft2_func(dt)
        E_ft2 = E_ft2_func()

        if self.noise_case == 1:
            cov_const0 = 0.
            cov_const1 = 0.            
        elif self.noise_case == 2:
            cov_const0 = unit_expected_residual_jumps()
            cov_const1 = unit_variance_residual_jumps()
        elif self.noise_case == 3:
            cov_const0 = unit_expected_residual_jumps()
            cov_const1 = 0.
        else:
            raise CustomException("invalid noise case")
        
        var = (self.sigma_W ** 2) * cov_const0
        mu2 = (self.mu_W ** 2) * cov_const1
        return (mu2 + var) * E_ft2

    def noise(self, num_samples: int, model: ExtendedModel, time_interval: timedelta) -> np.ndarray:
        dt = time_interval.total_seconds()
        epochs, jtimes = self.latents(model=model, dt=dt, num_samples=num_samples)
        
        noise_mean = self.mean(model=model, time_interval=time_interval, epochs=epochs, jtimes=jtimes)  
        noise_covar = self.covar(model=model, time_interval=time_interval, epochs=epochs, jtimes=jtimes) 
        R = (
            np.linalg.cholesky(noise_covar)
            if np.all(np.linalg.eigvals(noise_covar) > 0)
            else np.zeros_like(noise_covar)
        )
        noise = noise_mean + R @ self._rng.normal(size=(R.shape[1], num_samples))
        return noise

    def covar(self, model: ExtendedModel, time_interval: timedelta, epochs: np.ndarray, jtimes: np.ndarray, *args, **kwargs) -> CovarianceMatrix:
        jsizes, jtimes = self.get_jumps(epochs=epochs, jtimes=jtimes)
        jsizes2 = jsizes ** 2
        dt = time_interval.total_seconds()
        return self._covar(
            model=model, dt=dt, jsizes2=jsizes2, jtimes=jtimes
        ) + self._residual(model=model, dt=dt)


    def mean(self, model: ExtendedModel, time_interval: timedelta, epochs: np.ndarray, jtimes: np.ndarray, *args, **kwargs) -> StateVector:
        jsizes, jtimes = self.get_jumps(epochs=epochs, jtimes=jtimes)
        dt = time_interval.total_seconds()
        return self._mean(model=model, dt=dt, jsizes=jsizes, jtimes=jtimes)
