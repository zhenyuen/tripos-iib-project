from abc import abstractmethod
from stonesoup.models.base import Base, Model, LinearModel
from stonesoup.models.transition import TransitionModel
from stonesoup.base import Base, Property
from stonesoup.types.numeric import Probability
from stonesoup.types.state import State
from stonesoup.types.array import StateVector, StateVectors, CovarianceMatrix
from typing import Optional, Iterable, Union, Callable, Tuple
from datetime import datetime, timedelta
from scipy.linalg import block_diag
import numpy as np
from typing import Set, Sequence

import copy


class CustomException(Exception):
    pass


class ExtendedModel(Model):
    """ExtendedModel class

    Base/Abstract class for all extended models
    Extended models can handle external inputs"""

    # Not sure about naming the matrices/vectors below or if they should be explicit properties.
    A: Optional[np.ndarray] = Property(default=None, doc="Transition matrix A.")
    b: Optional[np.ndarray] = Property(default=None, doc="Input transform vector b.")
    h: Optional[np.ndarray] = Property9(default=None, doc="Non-Gaussian noise transform vector, h")
    g: Optional[np.ndarray] = Property(default=None, doc="Gaussian noise transform vector, g")

    @abstractmethod
    def ext_input(self, **kwargs) -> np.ndarray:
        """External input vector"""
        raise NotImplementedError


class Driver(Base):
    """Driver type

    Base/Abstract class for all noise driving process."""

    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")
    _rng: np.random.Generator = Property(default=None, doc="Random number generator.")
    # _models: Set[int] = Property(default=None, doc="A set of models that are registered to this driver instance.")
    _master: ExtendedModel = Property(
        default=None, doc="Model responsible for generating latent variables."
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed=self.seed)
        # self._models = set()

    def register_model(self, model: ExtendedModel) -> None:
        """Register model to driver

        Parameters
        ----------
        model : :type:`ExtendedModel`
            A GeneralModel instance

        """
        # First model added is master by default
        if self._master is None:
            self._master = model
        # self._models.add(id(model))

    def _model_is_master(self, model) -> bool:
        """Check if model is the driver master

        Parameters
        ----------
        model : :type:`GeneralModel`
            A GeneralModel instance

        """
        if self._master is None:
            raise AttributeError("Master is not assigned.")
        return id(self._master) == id(model)

    @abstractmethod
    def noise(self, model: ExtendedModel, time_interval: timedelta, num_samples: int) -> np.ndarray:
        """
        returns driving noise term
        """
        pass


class NonGaussianDriver(Driver):
    _Gammavs: Iterable[np.double] = Property(default=None, doc="Epochs history.")
    _Vvs: Iterable[np.double] = Property(default=None, doc="Jump times history.")
    c: np.double = Property(doc="Truncation parameter, expected no. jumps per unit time.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._Gammavs = []  # Gammav latent
        self._Vvs = []  # Vv latent

    def latents(
        self, model: ExtendedModel, dt: np.double, num_samples: int, rate: np.double = 1.0
    ) -> Tuple[np.ndarray]:
        """
        Sample gammav and vv is master, else return from hist
        """
        if self._model_is_master(model):
            # cutoff = self.c * dt
            # init = self._rng.exponential(
            #     scale=rate, size=max(int(np.ceil(1.1 * cutoff)), num_samples)
            # ).cumsum()
            # epochs = [init]
            # last_epoch = init[-1]

            # while last_epoch < cutoff:
            #     epoch_seq = self._rng.exponential(scale=rate, size=int(np.ceil(0.1 * cutoff)))
            #     epoch_seq[0] += last_epoch
            #     epoch_seq = epoch_seq.cumsum()  # generates a sequence of time stamps
            #     last_epoch = epoch_seq[-1]
            #     epochs.append(epoch_seq)

            # epochs = np.concatenate(epochs)
            # epochs = epochs[epochs < cutoff]
            
            epochs = self._rng.exponential(scale=rate, size=(self.c, num_samples))
            epochs = epochs.cumsum(axis=0)
            # Generate jump times
            jtimes = self._rng.uniform(low=0.0, high=dt, size=epochs.shape)
            self._Gammavs.append(epochs)
            self._Vvs.append(jtimes)
        else:
            epochs = self._Gammavs[-1]
            jtimes = self._Vvs[-1]
        return epochs, jtimes

    @abstractmethod
    def _hfunc(self, gammav: np.ndarray) -> np.ndarray:
        """
        returns jump sizes
        """
        pass


class LinearExtModel(ExtendedModel):
    """LinearExtendedModel class

    Base/Abstract class for all linear extended models"""

    def function(
        self, state: State, noise: Union[bool, np.ndarray] = False, **kwargs
    ) -> Union[StateVector, StateVectors]:
        """Model linear function :math:`f_k(x(k),w(k)) = F_k(x_k) + w_k`

        Parameters
        ----------
        state: State
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        : :class:`StateVector` or :class:`StateVectors`
            The StateVector(s) with the model function evaluated.
        """
        num_samples = state.state_vector.shape[1]
        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=num_samples, **kwargs)
            else:
                noise = 0
        return self.matrix(**kwargs) @ state.state_vector + self.ext_input(**kwargs) + noise


class NonGaussianExtModel(ExtendedModel):
    """NonGaussianExtendedModel class

    Abstract class for non-Gaussian (and Gaussian) extended models"""

    g_driver: Optional[Driver] = Property(default=None, doc="Gaussian driver.")
    ng_driver: Optional[Driver] = Property(default=None, doc="Non-Gaussian driver.")

    def latents(self, num_samples: int, time_interval: timedelta, *args, **kwargs):
        if self.ng_driver is None:
            return np.zeros((1, num_samples)), np.zeros((1, num_samples))
        else:
            dt = time_interval.total_seconds()
            return self.ng_driver.latents(model=self, num_samples=num_samples, dt=dt, *args, **kwargs)
        
    def rvs(self, num_samples: int = 1, **kwargs) -> Union[StateVector, StateVectors]:
        r"""Model noise/sample generation function

        Generates noise samples from the model.

        In mathematical terms, this can be written as: *PLACEHOLDER*

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:attr:`ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """
        g_noise = self._gaussian_noise(num_samples=num_samples, **kwargs)
        non_g_noise = self._nongaussian_noise(num_samples=num_samples, **kwargs)

        noise = np.atleast_2d(g_noise + non_g_noise)
        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)

    @abstractmethod
    def ft_func(self, dt: np.double, **kwargs) -> Callable:
        """
        Returns function handle implementing f_t() = exp(At) @ h
        """
        pass

    @abstractmethod
    def E_ft_func(self, dt: np.double, **kwargs) -> Callable:
        """
        Returns function handle implementing E[f_t()]
        """
        pass

    @abstractmethod
    def ft2_func(self, dt: np.double, **kwargs) -> Callable:
        """
        Returns function handle implementing f_t() = exp(At) @ h
        """
        pass

    @abstractmethod
    def E_ft2_func(self, dt: np.double, **kwargs) -> Callable:
        """
        Returns function handle implementing E[f_t() @ f_t().T]
        """
        pass

    @abstractmethod
    def omega_func(self, dt: np.double, **kwargs) -> Callable:
        """
        q_t() = exp(At) @ g
        Returns function handle implementing E[q_t() @ q_t().T]
        Only called by Gaussian driver, NOT the non-Gaussian driver
        """
        pass

    def _gaussian_noise(self, num_samples: int, time_interval: timedelta, **kwargs) -> np.ndarray:
        if self.g_driver is None:
            return 0
        return self.g_driver.noise(num_samples=num_samples, model=self, time_interval=time_interval)

    def _nongaussian_noise(
        self, num_samples: int, time_interval: timedelta, **kwargs
    ) -> np.ndarray:
        if self.ng_driver is None:
            return 0
        return self.ng_driver.noise(
            num_samples=num_samples, model=self, time_interval=time_interval
        )

    def pdf(self, state1: State, state2: State, **kwargs) -> Union[Probability, np.ndarray]:
        raise NotImplementedError

    def covar(self, *args, **kwargs) -> CovarianceMatrix:
        """Model covariance"""
        covar = 0
        if self.g_driver:
            covar += self.g_driver.covar(model=self, *args, **kwargs)
        if self.ng_driver:
            covar += self.ng_driver.covar(model=self, *args, **kwargs)
        return covar

    def mean(self, *args, **kwargs) -> StateVector:
        """Model mean"""
        mean = 0
        if self.g_driver:
            mean += self.g_driver.mean(model=self, *args, **kwargs)
        if self.ng_driver:
            mean += self.ng_driver.mean(model=self, *args, **kwargs)
        return mean


class LinearNonGaussianTransitionExtModel(TransitionModel, LinearExtModel, NonGaussianExtModel):
    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """
        # Use dummy value of dt=1
        return self.matrix(time_interval=timedelta(seconds=1)).shape[0]


class CombinedNonGaussianTransitionExtModel(TransitionModel, NonGaussianExtModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Gaussian.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """
    model_list: Sequence[NonGaussianExtModel] = Property(doc="List of Transition Models.")

    def function(self, state, noise=False, **kwargs) -> StateVector:
        """Applies each transition model in :py:attr:`~model_list` in turn to the state's
        corresponding state vector components.
        For example, in a 3D state space, with :py:attr:`~model_list` = [modelA(ndim_state=2),
        modelB(ndim_state=1)], this would apply modelA to the state vector's 1st and 2nd elements,
        then modelB to the remaining 3rd element.

        Parameters
        ----------
        state : :class:`stonesoup.state.State`
            The state to be transitioned according to the models in :py:attr:`~model_list`.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_state, 1`). The resultant state vector of the transition.
        """

        temp_state = copy.copy(state)
        ndim_count = 0
        if state.state_vector.shape[1] == 1:
            state_vector = np.zeros(state.state_vector.shape).view(StateVector)
        else:
            state_vector = np.zeros(state.state_vector.shape).view(StateVectors)
        # To handle explicit noise vector(s) passed in we set the noise for the individual models
        # to False and add the noise later. When noise is Boolean, we just pass in that value.
        if noise is None:
            noise = False
        if isinstance(noise, bool):
            noise_loop = noise
        else:
            noise_loop = False
        for model in self.model_list:
            temp_state.state_vector = state.state_vector[
                ndim_count : model.ndim_state + ndim_count, :
            ]
            state_vector[ndim_count : model.ndim_state + ndim_count, :] += model.function(
                temp_state, noise=noise_loop, **kwargs
            )
            ndim_count += model.ndim_state
        if isinstance(noise, bool):
            noise = 0

        return state_vector + noise

    def jacobian(self, state, **kwargs):
        return NotImplementedError

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list)

    def ft_func(self, **kwargs) -> Callable:
        def inner(**inner_kwargs) -> np.ndarray:
            tmp = [model.ft_func(**kwargs)(**inner_kwargs) for model in self.model_list]
            return np.vstack(tmp)

        return inner

    def E_ft_func(self, **kwargs) -> Callable:
        def inner(**inner_kwargs) -> np.ndarray:
            tmp = [model.E_ft_func(**kwargs)(**inner_kwargs) for model in self.model_list]
            return np.vstack(tmp)

        return inner

    def ft2_func(self, **kwargs) -> Callable:
        def inner(**inner_kwargs) -> np.ndarray:
            tmp = [model.ft2_func(**kwargs)(**inner_kwargs) for model in self.model_list]
            return block_diag(*tmp)

        return inner

    def E_ft2_func(self, **kwargs) -> Callable:
        def inner(**inner_kwargs) -> np.ndarray:
            tmp = [model.E_ft2_func(**kwargs)(**inner_kwargs) for model in self.model_list]
            return block_diag(*tmp)

        return inner

    def omega_func(self, **kwargs) -> Callable:
        def inner(**inner_kwargs) -> np.ndarray:
            tmp = [model.omega_func(**kwargs)(**inner_kwargs) for model in self.model_list]
            return block_diag(*tmp)

        return inner

    def covar(self, epochs_l, jtimes_l, *args, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        covar_list = [model.covar(epochs=epochs_l[i], jtimes=jtimes_l[i], *args, **kwargs) for i, model in enumerate(self.model_list)]
        return block_diag(*covar_list)

    def mean(self, epochs_l, jtimes_l, *args, **kwargs) -> StateVector:
        """Returns the transition model noise mean vector.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise mean.
        """
        mean_list = [model.mean(epochs=epochs_l[i], jtimes=jtimes_l[i], *args, **kwargs) for i, model in enumerate(self.model_list)]
        return np.vstack(mean_list)
    
    def latents(self, *args, **kwargs) -> np.ndarray:
        # Need to change this function, breaks compatability
        # A single model would return only one sequence, not a list of sequences
        epochs_list, jtime_list = [], []
        for model in self.model_list:
            e,j = model.latents(*args, **kwargs)
            epochs_list.append(e)
            jtime_list.append(j)
        return np.array(epochs_list), np.array(jtime_list)



class CombinedLinearNonGaussianTransitionExtModel(
    CombinedNonGaussianTransitionExtModel, LinearExtModel
):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Linear and Gaussian.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """

    def matrix(self, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """

        transition_matrices = [model.matrix(**kwargs) for model in self.model_list]
        return block_diag(*transition_matrices)

    def ext_input(self, **kwargs) -> np.ndarray:
        ext_inputs = [model.ext_input(**kwargs) for model in self.model_list]
        return np.vstack(ext_inputs)
