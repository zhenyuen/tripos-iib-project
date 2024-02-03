from mystonesoup.base import Driver, ExtendedModel, CustomException, NonGaussianDriver
from stonesoup.base import Base, Property
from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix
from typing import Iterable, Tuple
from datetime import timedelta
from abc import abstractmethod
import numpy as np


class GaussianDriver(Driver):
    mu_W: np.double = Property(doc="Gaussian mean.")
    sigma_W: np.double = Property(doc="Gaussian standard deviation.")
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
    alpha: np.double = Property(doc="Alpha parameter.")
    mu_W: np.double = Property(doc="Conditional Gaussian mean.")
    sigma_W: np.double = Property(doc="Conditional Gaussian variance.")
    noise_case: int = Property(doc="Noise case, either 1, 2 or 3.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self._alpha = alpha
        # self._mu_W = mu_W
        # self._sigma_W = sigma_W
        # self._noise_case = noise_case

    def _hfunc(self, gammav: np.ndarray) -> np.ndarray:
        return np.power(gammav, -1.0 / self.alpha)

    def _Ac(self) -> np.double:
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
        self, model: ExtendedModel, dt: np.double, jsizes: np.ndarray, jtimes: np.ndarray
    ) -> np.ndarray:
        ft_func = model.ft_func(dt=dt)
        ft = ft_func(jtimes=jtimes)
        # print(jsizes.shape)
       
        m = (dt ** (1.0 / self.alpha)) * np.sum(jsizes.reshape(-1, 1, 1) * ft, axis=0)

        return self.mu_W * m

    def _covar(
        self, model: ExtendedModel, dt: np.double, jsizes2: np.ndarray, jtimes: np.ndarray
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

    def _residual_constant(self) -> np.double:
        return self.alpha / (2.0 - self.alpha) * np.power(self.c, 1.0 - 2.0 / self.alpha)

    def _residual(self, model: ExtendedModel, dt: np.double) -> np.ndarray:
        E_ft2_func = model.E_ft2_func(dt=dt)
        E_ft2 = E_ft2_func()
        if self.noise_case == 1:
            Sigma = 0
        elif self.noise_case == 2:
            sigma_W2 = self.sigma_W**2
            mu_W2 = self.mu_W**2
            Sigma = (sigma_W2 + mu_W2) * E_ft2
        elif self.noise_case == 3:
            Sigma = self.sigma_W2 * E_ft2
        else:
            raise CustomException("invalid noise case")
        return self._residual_constant() * Sigma

    def _centring(self, model: ExtendedModel, dt: np.double) -> np.ndarray:
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
        
