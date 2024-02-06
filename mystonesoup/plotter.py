from stonesoup.plotter import Plotterly, AnimatedPlotterly
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from itertools import chain
from datetime import datetime, timedelta
try:
    from plotly import colors
except ImportError:
    colors = None
try:
    import plotly.graph_objects as go
except ImportError:
    go = None


class MyPlotterly(Plotterly):
    @staticmethod
    def _generate_ellipse_points(state, mapping, n_points=30):
        """Generate error ellipse points for given state and mapping"""
        return generate_ellipse_points(state=state, mapping=mapping, n_points=n_points)
        

class MyAnimatedPlotterly(AnimatedPlotterly):
    def _plot_particles_and_ellipses(self, tracks, mapping, resize, method="uncertainty"):
        """Generate error ellipse points for given state and mapping"""
        data = [dict() for _ in tracks]
        trace_base = len(self.fig.data)
        for n, track in enumerate(tracks):

            # initialise arrays that store particle/ellipse for later plotting
            data[n].update(x=np.array([0 for _ in range(len(track))], dtype=object),
                           y=np.array([0 for _ in range(len(track))], dtype=object))

            for k, state in enumerate(track):

                # find data points
                if method == "uncertainty":

                    data_x, data_y = MyPlotterly._generate_ellipse_points(state, mapping)
                    data_x = list(data_x)
                    data_y = list(data_y)
                    data_x.append(np.nan)  # necessary to draw multiple ellipses at once
                    data_y.append(np.nan)
                    data[n]["x"][k] = data_x
                    data[n]["y"][k] = data_y

                elif method == "particles":

                    data_xy = state.state_vector[mapping[:2], :]
                    data[n]["x"][k] = data_xy[0]
                    data[n]["y"][k] = data_xy[1]

                else:
                    raise ValueError("Should be 'uncertainty' or 'particles'")

        for frame in self.fig.frames:

            frame_time = datetime.fromisoformat(frame.name)

            data_ = list(frame.data)  # current data in frame
            traces_ = list(frame.traces)  # current traces in frame

            data_.append(go.Scatter(x=[-np.inf], y=[np.inf]))  # add empty data for legend trace
            traces_.append(trace_base - len(tracks) - 1)  # ensure correct trace

            for n, track in enumerate(tracks):
                # now plot the data
                _x = list(chain(*data[n]["x"][tuple(self.all_masks[frame_time][n])]))
                _y = list(chain(*data[n]["y"][tuple(self.all_masks[frame_time][n])]))
                _x.append(np.inf)
                _y.append(np.inf)
                data_.append(go.Scatter(x=_x, y=_y))
                traces_.append(trace_base - len(tracks) + n)

            frame.data = data_
            frame.traces = traces_

        if resize:
            self._resize(data, type="particle_or_uncertainty")
        
        
def get_covar(state):
    covariance = state.covariance
    mu = state.state_vector
    weighted_covar = np.sum(state.weight[np.newaxis, np.newaxis, :] * covariance, axis=len(covariance.shape)-1)
    mu_bar = np.mean(mu, axis=1)
    tmp = mu - mu_bar[:, np.newaxis]
    weighted_mean = np.sum(state.weight[np.newaxis, np.newaxis, :] * (np.einsum("ik,kj->ijk", tmp, tmp.T)), axis=len(covariance.shape)-1)
    return weighted_mean + weighted_covar



def generate_ellipse_points(state, mapping, n_points=30):
    covar = get_covar(state)
    HH = np.eye(state.ndim)[mapping, :]  # Get position mapping matrix
    w, v = np.linalg.eig(HH @ covar @ HH.T)
    max_ind = np.argmax(w)
    min_ind = np.argmin(w)
    orient = np.arctan2(v[1, max_ind], v[0, max_ind])
    a = np.sqrt(w[max_ind])
    b = np.sqrt(w[min_ind])
    m = 1 - (b**2 / a**2)

    def func(x):
        return np.sqrt(1 - (m**2 * np.sin(x)**2))

    def func2(z):
        return quad(func, 0, z)[0]

    c = 4 * a * func2(np.pi / 2)

    points = []
    for n in range(n_points):
        def func3(x):
            return n/n_points*c - a*func2(x)

        points.append((brentq(func3, 0, 2 * np.pi, xtol=1e-4)))

    c, s = np.cos(orient), np.sin(orient)
    rotational_matrix = np.array(((c, -s), (s, c)))
    points.append(points[0])
    points = np.array([[a * np.sin(i), b * np.cos(i)] for i in points])
    points = rotational_matrix @ points.T
    return points + state.mean[mapping[:2], :]
