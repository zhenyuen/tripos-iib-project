import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import quad, quad_vec
from abc import ABC, abstractmethod


class CombinedModel:
    def __init__(self, *models):
        self._models = models

    # def _tensorise(self):
    #     self.A = np.diag([model.A for model in self._models])
    #     self.H = np.diag([model.H for model in self._models])
    #     self.G = np.diag([model.G for model in self._models])
    #     self.B = np.concatenate([model.B.T for model in self._models]).T

    def n_models(self):
        return len(self._models)

    def function(self, state, noise, time_interval):
        next_state = np.copy(state)
        for k, model in enumerate(self._models):
            next_state[:, k : k + 1] = model.function(state[:, k : k + 1], noise, time_interval)
        return next_state


"""
Model must be registered/bound to driver externally. The base model
class should not modify the driver for clarity and prone to less bugs.
"""


class BaseModel(ABC):
    def __init__(self, gaussian_driver=None, non_gaussian_driver=None):
        self._gaussian_driver = gaussian_driver
        self._non_gaussian_driver = non_gaussian_driver
        if gaussian_driver:
            gaussian_driver.add_model(self)
        if non_gaussian_driver:
            non_gaussian_driver.add_model(self)
        # The below parameters are defined for formality only. Recommend hard
        # coding analytic expressions rather than using them.
        self.A = None
        self.b = None
        self.h = None
        self.g = None

    def function(self, state, noise, time_interval):
        dt = time_interval.total_seconds()
        if noise:
            g_noise = self._gaussian_noise(state, dt)
            non_g_noise = self._nongaussian_noise(state, dt)
        else:
            g_noise = np.zeros_like(state)
            non_g_noise = np.zeros_like(state)
        # print(self._beta(dt).shape)
        # print(g_noise.shape)
        # print(non_g_noise.shape)
        return self._eAdt(dt) @ state + self._beta(dt) + g_noise + non_g_noise

    @abstractmethod
    def _eAdt(self, dt):
        """
        Model drift
        """
        pass

    @abstractmethod
    def _beta(self, dt):
        """
        p_t() = exp(At) @ b
        E[p_t()] ? Not sure what to denote this integral by
        """
        pass

    @abstractmethod
    def ft_func(self, dt):
        """
        Returns function handle implementing f_t() = exp(At) @ h
        """
        pass

    @abstractmethod
    def E_ft_func(self, dt):
        """
        Returns function handle implementing E[f_t()]
        """
        pass

    @abstractmethod
    def ft2_func(self, dt):
        """
        Returns function handle implementing f_t() = exp(At) @ h
        """
        pass

    @abstractmethod
    def E_ft2_func(self, dt):
        """
        Returns function handle implementing E[f_t() @ f_t().T]
        """
        pass

    @abstractmethod
    def omega_func(self):
        """
        q_t() = exp(At) @ g
        Returns function handle implementing E[q_t() @ q_t().T]
        Only called by Gaussian driver, NOT the non-Gaussian driver
        """
        pass

    def _gaussian_noise(self, state, dt):
        if self._gaussian_driver is None:
            return np.zeros_like(state)
        return self._gaussian_driver.noise(self, dt)

    def _nongaussian_noise(self, state, dt):
        if self._non_gaussian_driver is None:
            return np.zeros_like(state)
        return self._non_gaussian_driver.noise(self, dt)


class Process(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = np.zeros(shape=(1, 1))
        self.b = np.zeros(shape=(1, 1))
        self.h = np.eye(1)
        self.g = np.eye(1)

    def _eAdt(self, dt):
        return np.eye(1)

    def _beta(self, dt):
        return np.zeros((1, 1))

    def ft_func(self, dt):
        def inner(jtimes):
            return np.ones((1, 1))

        return inner

    def E_ft_func(self, dt):
        def inner():
            return dt * np.ones((1, 1))

        return inner

    def ft2_func(self, dt):
        def inner(jtimes):
            return np.ones((1, 1))

        return inner

    def E_ft2_func(self, dt):
        def inner():
            return dt * np.ones((1, 1))

        return inner

    def omega_func(self, dt):
        def inner():
            return dt * np.ones((1, 1))

        return inner


class Langevin(BaseModel):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theta = theta
        self.A = np.array([[0, 1.0], [0.0, theta]])
        self.b = np.array([0.0, 0.0]).reshape((2, 1))
        self.h = np.array([0.0, 1.0]).reshape((2, 1))
        self.g = np.array([0.0, 1.0]).reshape((2, 1))

    def _eAdt(self, dt):
        eA0 = np.array([[0, 1.0 / self._theta], [0.0, 1.0]])
        eA1 = np.array([[1, -1.0 / self._theta], [0.0, 0.0]])
        eAdt = np.exp(self._theta * dt) * eA0 + eA1
        # exp_A_delta_t
        return eAdt

    def _beta(self, dt):
        return np.zeros((2, 1))

    def ft_func(self, dt):
        """
        Summing terms here in the inner function would be more efficient,
        as returning a scalar faster than a vector. That being said, the
        summation is part of the point process simulation and the driver
        is responsible for it, not the model.
        """

        def inner(jtimes):
            v1 = np.array([[1.0 / self._theta], [1.0]])
            v2 = np.array([[-1.0 / self._theta], [0.0]])
            term1 = np.exp(self._theta * (dt - jtimes)).reshape(-1, 1, 1)
            term2 = np.ones_like(jtimes).reshape(-1, 1, 1)
            return term1 * v1 + term2 * v2

        return inner

    def ft2_func(self, dt):
        def inner(jtimes):
            M1 = np.array([[1.0 / (self._theta**2), 1.0 / self._theta], [1.0 / self._theta, 1.0]])
            M2 = np.array(
                [[-2.0 / (self._theta**2), -1.0 / self._theta], [-1.0 / self._theta, 0.0]]
            )
            M3 = np.array([[1.0 / (self._theta**2), 0.0], [0.0, 0.0]])
            term1 = np.exp(2 * self._theta * (dt - jtimes)).reshape(-1, 1, 1)
            term2 = np.exp(self._theta * (dt - jtimes)).reshape(-1, 1, 1)
            term3 = np.ones_like(jtimes).reshape(-1, 1, 1)
            return term1 * M1 + term2 * M2 + term3 * M3

        return inner

    def E_ft_func(self, dt):
        def inner():
            v1 = np.array([[1.0 / self._theta], [1.0]])
            v2 = np.array([[-1.0 / self._theta], [0.0]])
            term1 = (np.exp(self._theta * dt) - 1.0) / self._theta * v1
            term2 = dt * v2
            return term1 + term2

        return inner

    def E_ft2_func(self, dt):
        def inner():
            M1 = np.array([[1.0 / (self._theta**2), 1.0 / self._theta], [1.0 / self._theta, 1.0]])
            M2 = np.array(
                [[-2.0 / (self._theta**2), -1.0 / self._theta], [-1.0 / self._theta, 0.0]]
            )
            M3 = np.array([[1.0 / (self._theta**2), 0.0], [0.0, 0.0]])
            term1 = (np.exp(2.0 * self._theta * dt) - 1.0) / (2.0 * self._theta) * M1
            term2 = (np.exp(self._theta * dt) - 1.0) / self._theta * M2
            term3 = dt * M3
            return term1 + term2 + term3

        return inner

    def omega_func(self, dt):
        # As h and g are the same, omega is the same as E_ft
        return self.E_ft2_func(dt)


class Singer(BaseModel):
    def __init__(self, theta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._theta = theta
        self.A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, theta]])
        self.b = np.array([0.0, 0.0, 0.0]).reshape((3, 1))
        self.h = np.array([0.0, 0.0, 1.0]).reshape((3, 1))
        self.g = np.array([0.0, 0.0, 1.0]).reshape((3, 1)) # ACtually should be zero ,but dont care, just dont pass a gaussian driver

    def _integrate(self, func, a, b):
        res, err = quad_vec(func, a=a, b=b)
        # print("Integration error:", err)
        return res

    def _eAdt(self, dt):
        return expm((self.A * dt))

    def _beta(self, dt):
        func = lambda u: self._eAdt(dt - u) @ self.g
        return self._integrate(func=func, a=0, b=dt)
    
    def ft_func(self, dt):
        """
        Summing terms here in the inner function would be more efficient,
        as returning a scalar faster than a vector. That being said, the
        summation is part of the point process simulation and the driver
        is responsible for it, not the model.
        """

        def inner(jtimes):
            return expm((self._eAdt(dt - jtimes.reshape((-1, 1, 1))))) @ self.h

        return inner

    def ft2_func(self, dt):
        def inner(jtimes):
            eAdt_h = self.ft_func(dt)(jtimes)
            outer_products = np.einsum('ijk,ilk->ijl', eAdt_h, eAdt_h)
            return outer_products
        return inner

    def E_ft_func(self, dt):
        def inner():
            func = lambda u: self._eAdt(dt - u) @ self.h
            return self._integrate(func=func, a=0, b=dt)

        return inner

    def E_ft2_func(self, dt):
        def inner():
            def func(u):
                tmp = self._eAdt(dt - u) @ self.h
                return tmp @ tmp.T
            return self._integrate(func=func, a=0, b=dt)
    
        return inner

    def omega_func(self, dt):
        def inner():
            def func(u):
                tmp = self._eAdt(dt - u) @ self.g
                return tmp @ tmp.T
            return self._integrate(func=func, a=0, b=dt)
        
        return inner


class ERV(BaseModel):
    def __init__(self, eta, rho, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eta = eta
        self._rho = rho
        self._p = p
        self.A = np.array([[0, 1.0], [-self._eta, -self._rho]])
        self.b = self._eta * np.array([0.0, self._p]).reshape((2, 1))
        self.h = np.array([0.0, 1.0]).reshape((2, 1))
        self.g = np.array([0.0, 1.0]).reshape((2, 1))

    def _integrate(self, func, a, b):
        res, err = quad_vec(func, a=a, b=b)
        # print("Integration error:", err)
        return res

    def _eAdt(self, dt):
        return expm((self.A * dt))

    def _beta(self, dt):
        func = lambda u: self._eAdt(dt - u) @ self.g
        return self._integrate(func=func, a=0, b=dt)
    
    def ft_func(self, dt):
        """
        Summing terms here in the inner function would be more efficient,
        as returning a scalar faster than a vector. That being said, the
        summation is part of the point process simulation and the driver
        is responsible for it, not the model.
        """

        def inner(jtimes):
            return expm((self._eAdt(dt - jtimes.reshape((-1, 1, 1))))) @ self.h

        return inner

    def ft2_func(self, dt):
        def inner(jtimes):
            eAdt_h = self.ft_func(dt)(jtimes)
            outer_products = np.einsum('ijk,ilk->ijl', eAdt_h, eAdt_h)
            return outer_products
        return inner

    def E_ft_func(self, dt):
        def inner():
            func = lambda u: self._eAdt(dt - u) @ self.h
            return self._integrate(func=func, a=0, b=dt)

        return inner

    def E_ft2_func(self, dt):
        def inner():
            def func(u):
                tmp = self._eAdt(dt - u) @ self.h
                return tmp @ tmp.T
            return self._integrate(func=func, a=0, b=dt)
    
        return inner

    def omega_func(self, dt):
        def inner():
            def func(u):
                tmp = self._eAdt(dt - u) @ self.g
                return tmp @ tmp.T
            return self._integrate(func=func, a=0, b=dt)
        
        return inner
