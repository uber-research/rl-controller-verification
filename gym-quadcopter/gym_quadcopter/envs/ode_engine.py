import numpy as np 

def rKN(x, fx, n, hs):
    k_1 = np.zeros(14)
    k_2 = np.zeros(14)
    k_3 = np.zeros(14)
    k_4 = np.zeros(14)
    xk = np.zeros(14)
    y = fx(0., x)
    for i in range(n):
        k_1[i] = (y[i] * hs)
    for i in range(n):
        xk[i] = x[i] + k_1[i] * 0.5

    yxk = fx(0., xk)
    for i in range(n):
        k_2[i] = (yxk[i] * hs)
    for i in range(n):
        xk[i] = x[i] + k_2[i] * 0.5

    yxk = fx(0., xk)
    for i in range(n):
        k_3[i] = yxk[i] * hs
    for i in range(n):
        xk[i] = x[i] + k_3[i]

    yxk = fx(0., xk)
    for i in range(n):
        k_4[i] = yxk[i] * hs

    x2 = np.zeros(n)
    for i in range(n):
        x2[i] = x[i] + (k_1[i] + 2 * (k_2[i] + k_3[i]) + k_4[i]) / 6
    return x2





class ODE_Engine:
    def __init__(self, dt, dt_commands, T, ODE, n, initial_t=0):
        self._dt = dt
        self._dt_commands = dt_commands
        self._T = T
        self._ODE = ODE
        self._n = n

        # Internal Clock
        self.t = initial_t
    
    @property
    def dt(self):
        return self._dt

    @property
    def dt_commands(self):
        return self._dt_commands

    @property
    def T(self):
        return self._T

    @property
    def ODE(self):
        return self._ODE

    @property
    def n(self):
        """ ODE State Space
        """
        return self._n

    def step(self, state):
        """ Computes the next state for dt timestep 
        """
        return rKN(state, self.ODE, self.n, self.dt)

