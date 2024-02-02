from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.state import ParticleState, GaussianState, StateVector, StateVectors
from stonesoup.types.prediction import Prediction
from stonesoup.types.update import Update
from stonesoup.base import Base, Property, clearable_cached_property
import numpy as np


class RBParticleState(ParticleState):
    """
    RBParticle type

    A RBParticle type which contains a state, weight, mean vector and covariance matrix
    """

    # need to do dimensionality check on this covariance, need it to be 3-d
    covariance: np.ndarray = Property(doc="Three dimensional array (mxmxn) of covariance matrices")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # if state_vector or covariance modified, it will clear the cache
    @clearable_cached_property("state_vector", "covariance")
    def gaussian_states(self, *args, **kwargs):
        gaussian_states = [
            GaussianState(
                state_vector=self.state_vector[:, n],
                covar=self.covariance[:, :, n],
                timestamp=self.timestamp,
            )
            for n in range(self.state_vector.shape[1])
        ]
        return gaussian_states


class RBParticleStateUpdate(Update, RBParticleState):
    """RBStateUpdate type

    This is a simple RBParticle state update object.
    """


class RBParticleStatePrediction(Prediction, RBParticleState):
    """RBStateUpdate type

    This is a simple RBParticle state update object.
    """


class RBParticleUpdater(ParticleUpdater):
    pass


class RBParticlePredictor(ParticlePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.k_predictor = KalmanPredictor(*args, **kwargs)

    def sample_latent_pair(self, delta_t, c, n_particles):
        epochs = np.random.exponential(scale=1, size=(c * delta_t, n_particles))
        # cumulative sum each column
        epochs = epochs.cumsum(axis=0)
        # Generate jump times
        jtimes = np.random.uniform(low=0., high=delta_t, size=epochs.size)
        return epochs, jtimes

    def predict(self, prior, timestamp=None, **kwargs):
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None
        num_samples = len(prior.weight)
        model = self.transition_model
        epochs_l, jtimes_l = model.latents(num_samples=num_samples, time_interval=time_interval)
        process_mean = model.mean(time_interval=time_interval, epochs_l=epochs_l, jtimes_l=jtimes_l)
        process_covar = model.covar(time_interval=time_interval, epochs_l=epochs_l, jtimes_l=jtimes_l)
        print(process_covar.shape)
        print(process_mean.shape)
        F = model.matrix(time_interval=time_interval, **kwargs)
        new_state_vector = F @ prior.state_vector + process_mean + model.ext_input(**kwargs)
        # new_covariance = F @ prior.covariance @ F.T + process_covar
        
        new_covariance = np.einsum('ij,jkl->ikl', F, np.einsum('ijl,jk->ikl', prior.covariance, F.T)) + process_covar
        
        return Prediction.from_state(prior,
                                     parent=prior,
                                     state_vector=new_state_vector,
                                     covariance=new_covariance,
                                     timestamp=timestamp,
                                     transition_model=self.transition_model)
        
