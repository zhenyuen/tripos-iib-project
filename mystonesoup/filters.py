from stonesoup.predictor.particle import ParticlePredictor
from stonesoup.updater.particle import ParticleUpdater
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.state import ParticleState, CovarianceMatrix, GaussianState, StateVector, StateVectors
from stonesoup.types.prediction import Prediction, MeasurementPrediction, GaussianMeasurementPrediction
from stonesoup.types.update import Update
from stonesoup.base import Base, Property, clearable_cached_property
from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from functools import lru_cache

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
        if self.fixed_covar is not None:
            raise AttributeError("Fixed covariance should be none")

    # # if state_vector or covariance modified, it will clear the cache
    # @clearable_cached_property("state_vector", "covariance")
    # def gaussian_states(self, *args, **kwargs):
    #     gaussian_states = [
    #         GaussianState(
    #             state_vector=self.state_vector[:, n],
    #             covar=self.covariance[:, :, n],
    #             timestamp=self.timestamp,
    #         )
    #         for n in range(self.state_vector.shape[1])
    #     ]
    #     return gaussian_states
    
    @clearable_cached_property('state_vector', 'weight', 'covariance')
    def covar(self):
        """Sample covariance matrix for particles"""
        covariance = self.covariance
        mu = self.state_vector
        weighted_covar = np.sum(self.weight[np.newaxis, np.newaxis, :] * covariance, axis=len(covariance.shape)-1)
        mu_bar = np.sum(self.weight[np.newaxis, :] * mu, axis=1)
        tmp = mu - mu_bar[:, np.newaxis]
        weighted_mean = np.sum(self.weight[np.newaxis, np.newaxis, :] * (np.einsum("ik,kj->ijk", tmp, tmp.T)), axis=len(covariance.shape)-1)
        return weighted_mean + weighted_covar


class RBParticleStateUpdate(Update, RBParticleState):
    """RBStateUpdate type

    This is a simple RBParticle state update object.
    """
    """Particle Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.ParticleState`
            The state posterior
        """
    pass


class RBParticleMeasurementPrediction(MeasurementPrediction, RBParticleState):
    cross_covar: np.ndarray = Property(
    default=None, doc="The state-measurement cross covariance matrix")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cross_covar is not None \
                and self.cross_covar.shape[1] != self.state_vector.shape[0]:
            raise ValueError("cross_covar should have the same number of "
                             "columns as the number of rows in state_vector")

class RBParticleStatePrediction(Prediction, RBParticleState):
    """RBStateUpdate type

    This is a simple RBParticle state update object.
    """
    pass


class RBParticleUpdater(ParticleUpdater):
    def _measurement_matrix(self, predicted_state=None, measurement_model=None,
                            **kwargs):
        r"""This is straightforward Kalman so just get the Matrix from the
        measurement model.

        Parameters
        ----------
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`, :math:`P_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            Passed to :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`numpy.ndarray`
            The measurement matrix, :math:`H_k`

        """
        return self._check_measurement_model(
            measurement_model).matrix(**kwargs)

    def _measurement_cross_covariance(self, predicted_state, measurement_matrix):
        """
        Return the measurement cross covariance matrix, :math:`P_{k~k-1} H_k^T`

        Parameters
        ----------
        predicted_state : :class:`GaussianState`
            The predicted state which contains the covariance matrix :math:`P` as :attr:`.covar`
            attribute
        measurement_matrix : numpy.array
            The measurement matrix, :math:`H`

        Returns
        -------
        :  numpy.ndarray
            The measurement cross-covariance matrix

        """
        num_samples = predicted_state.covariance.shape[-1]
        return np.einsum('ijk,jm->imk', predicted_state.covariance, measurement_matrix.T)

    def _innovation_covariance(self, m_cross_cov, meas_mat, meas_mod):
        """Compute the innovation covariance

        Parameters
        ----------
        m_cross_cov : numpy.ndarray
            The measurement cross covariance matrix
        meas_mat : numpy.ndarray
            Measurement matrix
        meas_mod : :class:~.MeasurementModel`
            Measurement model

        Returns
        -------
        : numpy.ndarray
            The innovation covariance

        """
        meas_covar = meas_mod.covar()        
        return np.einsum('ij,jkl->ikl', meas_mat, m_cross_cov) + meas_covar[..., np.newaxis]

    def _posterior_mean(self, predicted_state, kalman_gain, measurement, measurement_prediction):
        r"""Compute the posterior mean, :math:`\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k
        \mathbf{y}_k`, where the innovation :math:`\mathbf{y}_k = \mathbf{z}_k -
        h(\mathbf{x}_{k|k-1}).

        Parameters
        ----------
        predicted_state : :class:`State`, :class:`Prediction`
            The predicted state
        kalman_gain : numpy.ndarray
            Kalman gain
        measurement : :class:`Detection`
            The measurement
        measurement_prediction : :class:`MeasurementPrediction`
            Predicted measurement

        Returns
        -------
        : :class:`StateVector`
            The posterior mean estimate
        """
        post_mean = predicted_state.state_vector + \
            kalman_gain @ (measurement.state_vector - measurement_prediction.state_vector)
        return post_mean.view(StateVector)
                 
    def _posterior_covariance(self, hypothesis):
        """
        Return the posterior covariance for a given hypothesis

        Parameters
        ----------
        hypothesis: :class:`~.Hypothesis`
            A hypothesised association between state prediction and measurement. It returns the
            measurement prediction which in turn contains the measurement cross covariance,
            :math:`P_{k|k-1} H_k^T and the innovation covariance,
            :math:`S = H_k P_{k|k-1} H_k^T + R`

        Returns
        -------
        : :class:`~.CovarianceMatrix`
            The posterior covariance matrix rendered via the Kalman update process.
        : numpy.ndarray
            The Kalman gain, :math:`K = P_{k|k-1} H_k^T S^{-1}`

        """
        predicted_covar = hypothesis.prediction.covariance
        post_cov = np.zeros_like(predicted_covar)
        num_samples = post_cov.shape[-1]

        for p in range(num_samples):
            mp_covar = hypothesis.measurement_prediction.covariance[..., p]
            mp_cross_covar =  hypothesis.measurement_prediction.cross_covar[..., p]
            
            kalman_gain = mp_cross_covar @ np.linalg.inv(mp_covar)
            post_cov[..., p] = predicted_covar[..., p] - kalman_gain @ mp_covar @ kalman_gain.T
            
        return post_cov.view(CovarianceMatrix), kalman_gain

    def update(self, hypothesis, **kwargs):
        predicted_state = hypothesis.prediction

        if hypothesis.measurement.measurement_model is None:
            measurement_model = self.measurement_model
        else:
            measurement_model = hypothesis.measurement.measurement_model


        if hypothesis.measurement_prediction is None:
            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model, **kwargs)


        posterior_covariance, kalman_gain = self._posterior_covariance(hypothesis)

        # Posterior mean
        posterior_mean = self._posterior_mean(predicted_state, kalman_gain,
                                              hypothesis.measurement,
                                              hypothesis.measurement_prediction)

        posterior = Update.from_state(
            state_vector=posterior_mean,
            covariance=posterior_covariance,
            state=hypothesis.prediction,
            hypothesis=hypothesis,
            timestamp=hypothesis.prediction.timestamp
        )

        new_weight = posterior.log_weight + measurement_model.logpdf(
            hypothesis.measurement, posterior, **kwargs)


        # Normalise the weights
        new_weight -= logsumexp(new_weight) 

        posterior.log_weight = new_weight

        # Resample
        resample_flag = True
        if self.resampler is not None:
            resampled_state = self.resampler.resample(posterior)
            if resampled_state == posterior:
                resample_flag = False
            posterior = resampled_state

        if self.regulariser is not None and resample_flag:
            prior = hypothesis.prediction.parent
            posterior = self.regulariser.regularise(prior, posterior)
        return  posterior


    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        predicted_state : :class:`~.GaussianState`
            The predicted state :math:`\mathbf{x}_{k|k-1}`, :math:`P_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used
        **kwargs : various
            These are passed to :meth:`~.MeasurementModel.function` and
            :meth:`~.MeasurementModel.matrix`

        Returns
        -------
        : :class:`GaussianMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state, **kwargs)        
        hh = self._measurement_matrix(predicted_state=predicted_state,
                                      measurement_model=measurement_model,
                                      **kwargs)

        # The measurement cross covariance and innovation covariance
        meas_cross_cov = self._measurement_cross_covariance(predicted_state, hh)
        innov_cov = self._innovation_covariance(meas_cross_cov, hh, measurement_model)

        return MeasurementPrediction.from_state(
            predicted_state, pred_meas, innov_cov, cross_covar=meas_cross_cov)


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
        
        new_state_vector = np.zeros_like(prior.state_vector)
        new_covariance = np.zeros_like(prior.covariance)
        for p in range(num_samples):
            epochs = epochs_l[..., p]
            jtimes = jtimes_l[..., p]

            process_mean = model.mean(time_interval=time_interval, epochs_l=epochs, jtimes_l=jtimes)
            process_covar = model.covar(time_interval=time_interval, epochs_l=epochs, jtimes_l=jtimes)
            F = model.matrix(time_interval=time_interval, **kwargs)
            # print(prior.covariance[..., p].shape)
            mean = F @ prior.state_vector[..., p:p+1] + process_mean + model.ext_input(**kwargs)
            covar = F @ prior.covariance[..., p] @ F.T + process_covar

            new_covariance[..., p] = covar
            new_state_vector[..., p] = multivariate_normal.rvs(mean.flatten(), covar)


        return Prediction.from_state(prior,
                                     parent=prior,
                                     state_vector=new_state_vector,
                                     covariance=new_covariance,
                                     timestamp=timestamp,
                                     transition_model=self.transition_model)
        
