import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class SDE:
    def __init__(self, model, X0, params, T0=0, T=1):
        """
        Initialise the SDE.
        Attributes:
            model (str): The type of SDE, either Black-Scholes, Ornstein-Uhlenbeck or Heston,
            X0 (float): Intial value of the process,
            params (dict): Different parameters for the process,
            T0 (float): Time start (Default 0)
            T (float): Maturity, end of time (Default 1).
        """
        self.model = model
        self.X0 = X0
        self.T0 = T0
        self.T = T
        self.params = params


    def drift_diffusion(self, x):
        """
        Computes the drift and diffusion of a process for a certain value of X_t.
        Depends on the type of model. Two models accepted: Black-Scholes (BS for short)
        and Ornstein-Uhlenbeck (OU). Needed for the simulation.
        Attributes:
            x (float): Value of x.
        Return:
            drift (float): Value of the drift
            diffusion (float): Value of the diffusion
        """
        if self.model in ["Black-Scholes", 'BS']:
            drift = self.params['mu']*x
            diffusion = self.params["sigma"]*x
            return (drift, diffusion)
        if self.model in ["Ornstein-Uhlenbeck", 'OU']:
            drift = -self.params['k']*(x-self.params['m'])
            diffusion = self.params['sigma']
            return (drift, diffusion)

    def simulate_EM(self, delta_t=0.01, nb_simu=1, seed=None):
        """
        Simulate processes (nb_simu processes) of the SDE whith an EulerScheme.
        Attributes:
            delta_t (float): delta_t is the time step for the EulerScheme.
            nb_simu (int): number of simulations
            seed (int or None): seed if we want to replicate the experience.
                Default : no seed.
        Return:
            X (pd.DataFrame): Simulated processes. Each column is a process.
                Index are the timesteps, columns are the indices of the simulated
                processes.
        """
        if seed is not None:
            np.random.seed(seed)
        # Know number of decimals needed with respect to the delta of time
        decimals = max(0, int(-np.log10(delta_t)))

        if self.model in ["Black-Scholes", 'BS', "Ornstein-Uhlenbeck", 'OU']:
            X = pd.DataFrame(columns=range(nb_simu), index=np.around(np.linspace(self.T0,
                             self.T, int((self.T - self.T0)/delta_t)), decimals=decimals))
            for i in range(nb_simu):
                X[i].iloc[0] = self.X0
                for k in range(1, len(X)):
                    drift, diffusion = self.drift_diffusion(X[i].iloc[k-1])
                    delta_W = np.random.normal(0, np.sqrt(delta_t))
                    X[i].iloc[k] = X[i].iloc[k-1] + drift * delta_t + diffusion * delta_W
        elif self.model == "Heston":
            X = pd.DataFrame(columns=range(nb_simu), index=np.around(np.linspace(
                            self.T0, self.T, int((self.T - self.T0)/delta_t)),
                            decimals = 2))
            v = pd.DataFrame(columns=range(nb_simu), index=np.around(np.linspace(
                            self.T0, self.T, int((self.T - self.T0)/delta_t)),
                            decimals = 2))
            for i in range(nb_simu):
                X[i].iloc[0] = self.X0
                v[i].iloc[0] = self.params['v0']
                for k in range(1, len(X)):
                    delta_W = np.random.normal(0, np.sqrt(delta_t))
                    delta_Z = np.random.normal(0, np.sqrt(delta_t))
                    X[i].iloc[k] = X[i].iloc[k-1] + self.params['mu'] * X[i].iloc[k-1] * delta_t + \
                                    np.sqrt(v[i].iloc[k-1]) * X[i].iloc[k-1] * delta_W
                    v[i].iloc[k] = v[i].iloc[k-1] - self.params['k'] * (v[i].iloc[k-1] \
                                    - self.params['m']) * delta_t + self.params["sigma"] \
                                    * np.sqrt(v[i].iloc[k-1]) * delta_Z
        return X

    def BlackScholes(self, delta_t=0.01, nb_simu=1):
        time = np.round([self.T0 + delta_t*i for i in range(int(self.T / delta_t) + 1)], 2)
        simulation = pd.DataFrame(index=time)
        for i in range(nb_simu):
            Z = np.random.normal(0, np.sqrt(delta_t), len(time))
            W = np.cumsum(Z)
            simu = np.array([self.X0*np.exp((self.params['mu'] - (self.params['sigma']**2)/2)*h + self.params['sigma']*w) for h, w in zip(time, W)])
            simulation[i] = simu
        return simulation



    def plot_df(self, t=None, X=None, title=None, max_nb=5, nb_simu=1):
        """
        Plots simulated processes.
        Attributes:
            t (list or None): t is a list of timesteps that will be marked in the plot.
                By default not plotted.
            X (pd.DataFrame): Simulated process. By default None, the process is
                simulated within the method.
            title (str): title of the plot. Default = the method name of the SDE.
            max_nb (int): max_nb of simulations we want. (Default = 5)
            nb_simu (int): Number of simulations we want. (Default = 1)
        Return:
            X (pd.DataFrame): Either the passed argument X, or the simulated process.

        """
        if X is None:
            X = self.simulate_EM(nb_simu=nb_simu)
        for i in range(min(max_nb, nb_simu)):
            p = plt.plot(X[i])
            if t is not None:
                plt.plot(t, X[i].loc[t], "o", color=p[0].get_color())
        plt.xlabel("time")
        if title is None:
            title = self.model
        plt.title(title)
        return X

    def save_csv(self, t=None, X=None, title=None, nb_simu=1):
        if X is None:
            X = self.simulate_EM(nb_simu=nb_simu)
        if title is None:
            title = self.model + '.csv'
        X.to_csv(title)
        if t is not None:
            t_df = X.loc[t]
            t_df.to_csv('t_' + title)
        return X
