import numpy as np
import matplotlib.pyplot as plt
import sklearn

# Bayesian Optimization
from bayes_opt import UtilityFunction


# Create a Bayesian Optimization class
class BayesianOptimization(object):
    """
    Bayesian Optimization class
    """
    def __init__(self, f, bounds, x0, kernel = None, random_state=None, acq='ucb', kappa=2.576, xi=0.0, **kwargs):
        """
        :param f:
            Function to be optimized.
        :param bounds:
            Array of size 2 where bounds[0] = lower bound and bounds[1] = upper bound
        :param x0:
            Array of size n where x0[i] = starting point
        :param random_state:
            Random seed
        :param acq:
            Acquisition function to be used, defaults to Upper Confidence Bound.
        :param kappa:
            Controls exploration exploitation trade-off. Larger kappa --> more exploration.
        :param xi:
            Controls exploration exploitation trade-off. Larger xi --> less exploration.
        :param kwargs:
            Additional arguments for the acquisition function.
        """
        self.f = f
        self.bounds = bounds
        self.x0 = x0
        self.random_state = random_state

        # expolaration exploitation trade-off
        self.kappa = kappa

        # Initialize acquisition function
        if acq not in ['ucb', 'ei', 'poi', 'gp']:
            raise ValueError('Invalid acquisition function.')
        else:
            self.acq = acq

        if kernel is not None:
            self.kernel = kernel
        else:
            kernel_1 = sklearn.gaussian_process.kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
            kernel_2 = sklearn.gaussian_process.kernels.WhiteKernel(noise_level=1e-1, noise_level_bounds=(1e-2, 1e+1))
            kernel_3 = sklearn.gaussian_process.kernels.ConstantKernel(constant_value=1.0, constant_value_bounds=(1e-2, 1e+1))
            self.kernel = kernel_1 * kernel_2 * kernel_3

        self.gp = sklearn.gaussian_process.GaussianProcessRegressor(kernel=self.kernel, alpha=0.0, n_restarts_optimizer=10)



        # Constants
        self.START = False

        # X and Y arrays
        self.X = []
        self.Y = []


    def _first_step(self):
        """
        First step of the optimization.
        """
        for i in len(x0):
            y = self.f(x0)
            self.X.append(x0)
            self.Y.append(y)
        self.gp.fit(self.X, self.y)




    def step(self):
        """
        One step of the optimization.
        """
        # Sample next point

        if self.START is False:

        else:
            self.
        self.gp.fit(self.X, self.Y)






