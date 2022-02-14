import numpy as np
import matplotlib.pyplot as plt
import sklearn
import warnings

# Bayesian Optimization
from bayes_opt.util import UtilityFunction, acq_max, ensure_rng


# Create a Bayesian Optimization class
class BayesianOptimization(object):
    """
    Bayesian Optimization class
    """
    def __init__(self, f, bounds, x0, kernel = None, random_state=None, acq='ucb', kappa=2.576, xi=0.001, **kwargs):
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
        self.random_state = ensure_rng(random_state)

        # expolaration exploitation trade-off
        self.kappa = kappa

        # Initialize acquisition function
        if acq not in ['ucb', 'ei', 'poi']:
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


        # Initialize utility function
        self.utility = UtilityFunction(kind=self.acq, kappa=self.kappa, xi=xi)

        # Constants
        self.START = False

        # X and Y arrays
        self.X = []
        self.Y = []


    def _first_step(self):
        """
        First step of the optimization.
        """
        for i in range(len(self.x0)):
            y = self.f(self.x0)
            self.X.append(self.x0)
            self.Y.append(y)
            # Update the GP
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(self.X, self.Y)
        self.utility.update_params()
        return acq_max(self.utility.utility, self.gp, max(self.Y), self.bounds, self.random_state)




    def simulate_steps(self, n = 10):
        """
        Simulate n steps of the optimization.
        """
        # Sample next point

        for i in range(n):

            # First iteration
            if self.START is False:
                self.START = True
                next_x = self._first_step()
                self.X.append(next_x)
                self.Y.append(self.f(next_x))

            # Update utility function
            self.utility.update_params()

            # Obtain next best sample
            x_max = acq_max(self.utility.utility, self.gp, max(self.Y), self.bounds, self.random_state)

            # Add to X and Y arrays
            self.X.append(x_max)
            self.Y.append(self.f(x_max))

            # Update the GP
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(self.X, self.Y)


        return (self.X, self.Y)



if __name__ == '__main__':

    # Define the objective function
    def objective(x):
        return x[0]**2 + x[1]**2

    # Define the bounds
    bounds = np.array([[-5, 5], [-5, 5]])

    # Initialize the optimizer
    optimizer = BayesianOptimization(objective, bounds, x0 = [3, -3])

    # Simulate the optimization process
    X, Y = optimizer.simulate_steps(n = 10)

    # Plot the results
    plt.plot(X, Y)
    print(*list(zip(X,Y)), sep = '\n')
    plt.show()





