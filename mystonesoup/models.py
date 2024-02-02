from mystonesoup.base import LinearNonGaussianTransitionExtModel
from datetime import timedelta
from typing import Callable
from stonesoup.base import Property
import numpy as np

class Process(LinearNonGaussianTransitionExtModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = np.zeros(shape=(1, 1))
        self.b = np.zeros(shape=(1, 1))
        self.h = np.eye(1)
        self.g = np.eye(1)

    def matrix(self, **kwargs) -> np.ndarray:
        return np.eye(1)

    def ext_input(self, **kwargs) -> np.ndarray:
        return np.zeros((1, 1))

    def ft_func(self, **kwargs) -> Callable:
        def inner(**inner_kwargs) -> np.ndarray:
            return np.ones((1, 1))

        return inner

    def E_ft_func(self, dt: np.double, **kwargs) -> Callable:
        def inner(**inner_kwargs):
            return dt * np.ones((1, 1))

        return inner

    def ft2_func(self, **kwargs) -> Callable:
        def inner(**inner_kwargs):
            return np.eye(1)

        return inner

    def E_ft2_func(self, dt: np.double, **kwargs) -> Callable:
        def inner(**inner_kwargs):
            return dt * np.eye(1)

        return inner

    def omega_func(self, dt: np.double, **kwargs) -> Callable:
        def inner(**inner_kwargs):
            return dt * np.eye(1)

        return inner


class Langevin(LinearNonGaussianTransitionExtModel):

    theta: np.double = Property(doc="Theta parameter.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = np.array([[0, 1.0], [0.0, self.theta]])
        self.b = np.array([0.0, 0.0]).reshape((2, 1))
        self.h = np.array([0.0, 1.0]).reshape((2, 1))
        self.g = np.array([0.0, 1.0]).reshape((2, 1))

    def matrix(self, time_interval: timedelta, **kwargs) -> np.ndarray:
        dt = time_interval.total_seconds()
        eA0 = np.array([[0, 1.0 / self.theta], [0.0, 1.0]])
        eA1 = np.array([[1, -1.0 / self.theta], [0.0, 0.0]])
        eAdt = np.exp(self.theta * dt) * eA0 + eA1
        # exp_A_delta_t
        return eAdt

    def ext_input(self, **kwargs) -> np.ndarray:
        return np.zeros((2, 1))

    def ft_func(self, dt: np.double, **kwargs) -> Callable:
        """
        Summing terms here in the inner function would be more efficient,
        as returning a scalar faster than a vector. That being said, the
        summation is part of the point process simulation and the driver
        is responsible for it, not the model.
        """

        def inner(jtimes):
            # need to replace n with a better name
            n = jtimes.shape[1]
            v1 = np.tile(np.array([[1.0 / self.theta], [1.0]]), (1, n))
            v2 = np.tile(np.array([[-1.0 / self.theta], [0.0]]), (1, n))

            term1 = np.exp(self.theta * (dt - jtimes))[:, np.newaxis, :]
            term2 = np.ones_like(jtimes)[:, np.newaxis, :]
            return term1 * v1 + term2 * v2

        return inner

    def ft2_func(self, dt: np.double, **kwargs) -> Callable:
        def inner(jtimes):
            M1 = np.array([[1.0 / (self.theta**2), 1.0 / self.theta], [1.0 / self.theta, 1.0]])
            M2 = np.array(
                [[-2.0 / (self.theta**2), -1.0 / self.theta], [-1.0 / self.theta, 0.0]]
            )
            M3 = np.array([[1.0 / (self.theta**2), 0.0], [0.0, 0.0]])
            # need to replace n with a better name
            n = jtimes.shape[1]
            M1 = np.tile(M1, (n, 1, 1)).T
            M2 = np.tile(M2, (n, 1, 1)).T
            M3 = np.tile(M3, (n, 1, 1)).T
            term1 = np.exp(2 * self.theta * (dt - jtimes))[:, np.newaxis, np.newaxis, :]
            term2 = np.exp(self.theta * (dt - jtimes))[:, np.newaxis, np.newaxis, :]
            term3 = np.ones_like(jtimes)[:, np.newaxis, np.newaxis, :]
            return term1 * M1 + term2 * M2 + term3 * M3

        return inner

    def E_ft_func(self, dt: np.double, **kwargs) -> Callable:
        def inner():
            v1 = np.array([[1.0 / self.theta], [1.0]])
            v2 = np.array([[-1.0 / self.theta], [0.0]])
            term1 = (np.exp(self.theta * dt) - 1.0) / self.theta * v1
            term2 = dt * v2
            return (term1 + term2)

        return inner

    def E_ft2_func(self, dt: np.double, **kwargs) -> Callable:
        def inner():
            M1 = np.array([[1.0 / (self.theta**2), 1.0 / self.theta], [1.0 / self.theta, 1.0]])
            M2 = np.array(
                [[-2.0 / (self.theta**2), -1.0 / self.theta], [-1.0 / self.theta, 0.0]]
            )
            M3 = np.array([[1.0 / (self.theta**2), 0.0], [0.0, 0.0]])
            term1 = (np.exp(2.0 * self.theta * dt) - 1.0) / (2.0 * self.theta) * M1
            term2 = (np.exp(self.theta * dt) - 1.0) / self.theta * M2
            term3 = dt * M3
            return (term1 + term2 + term3)[..., np.newaxis]

        return inner

    def omega_func(self, dt: np.double, **kwargs) -> Callable:
        def inner():
            M1 = np.array([[1.0 / (self.theta**2), 1.0 / self.theta], [1.0 / self.theta, 1.0]])
            M2 = np.array(
                [[-2.0 / (self.theta**2), -1.0 / self.theta], [-1.0 / self.theta, 0.0]]
            )
            M3 = np.array([[1.0 / (self.theta**2), 0.0], [0.0, 0.0]])
            term1 = (np.exp(2.0 * self.theta * dt) - 1.0) / (2.0 * self.theta) * M1
            term2 = (np.exp(self.theta * dt) - 1.0) / self.theta * M2
            term3 = dt * M3
            return (term1 + term2 + term3)[..., np.newaxis]

        return inner