import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
import math


class Quadcopter:

    def __init__(self, init_state, dt=0.01, dt_commands=0.01, T=5.):
        self.init_state = init_state
        self.state = np.array(init_state)
        self.dt = dt
        self.dt_commands = dt_commands
        self.t = 0.0
        self.T = T
        self.trajectory = [self.state]
        self.eval_times = [self.t]
        self.command_memory = []
        self.params = {
            "C1": 0.04076521,
            "C2": 380.8359,
            "CT": 1.285e-8,
            "CD": 7.645e-11,
            "d": 0.046 / np.sqrt(2),
            "Ix": 1.657171e-5,
            "Iy": 1.6655602e-5,
            "Iz": 2.9261652e-5,
            "m": 0.028,
            "g": 9.81
        }
        self.f = self.ODE()
        self.query = np.array([self.state[0], 0., 0., 0.0])
        self.query_duration = np.random.uniform(0.9, 2.0)
        self.query_active = True
        self.query_memory = {"queries": [], "t": []}
        self.commands = np.zeros(4)
        self.coeff = np.array([100000., 200., 200., 200.])
        self.bias = np.array([0., -100., -100., -100.])
        self.random_query(attitude=True, altitude=True)

    def simulation(self):
        while self.t < self.T:
            if self.query_duration <= 0:
                self.random_query(attitude=True, altitude=True)
            if self.next_step(self.PID()):
                break
            self.query_duration -= self.dt
        return np.array(self.trajectory, dtype="float"), np.array(self.eval_times)

    def reset(self):
        self.state = np.array(self.init_state)
        self.t = 0.0
        self.trajectory = [self.state]
        self.eval_times = [self.t]
        self.command_memory = []
        self.query_duration = np.random.uniform(0.9, 2.0)
        self.query_active = True

    def next_step(self, commands, normalized=False):
        if normalized:
            self.commands = commands * self.coeff + self.bias
        else:
            self.commands = commands
        self.command_memory.append(np.copy(self.commands))
        self.step_RKn(14)
        self.trajectory.append(self.state)
        self.t += self.dt
        self.eval_times.append(self.t)
        return self.check_stop()

    def random_trajectory(self, N=10):
        return 0

    def PID(self):
        c = np.zeros(4, dtype="float")
        c[0] = 1500. * (25 * (2 * (self.query[0] - self.state[0]) - self.state[3]) + 15 * self.state[10]) + 36000.
        c[1] = 550. * (self.query[1] - self.state[7]) + 100. * self.state[11]
        c[2] = 550. * (self.query[2] - self.state[8]) + 100. * self.state[12]
        c[3] = 750. * (self.query[3] - self.state[9]) + 46.7 * self.state[13]
        c[0] = max(0, min(c[0], 200000.))
        return c

    def ODE(self):
        def f(t, x):
            m = self.M()
            f = self.F()
            ode = np.empty(14)
            ode[0] = -np.sin(x[5]) * x[1] + np.cos(x[5]) * np.sin(x[4]) * x[2] + np.cos(x[5]) * np.cos(x[4]) * x[3]
            ode[1] = x[9] * x[2] - x[8] * x[3] + np.sin(x[5]) * self.params["g"]
            ode[2] = -x[9] * x[1] + x[7] * x[3] - np.cos(x[5]) * np.sin(x[4]) * self.params["g"]
            ode[3] = x[8] * x[1] - x[7] * x[2] - np.cos(x[5]) * np.cos(x[4]) * self.params["g"] + f / self.params["m"]
            ode[4] = x[7] + np.cos(x[4]) * np.tan(x[5]) * x[9] + np.tan(x[5]) * np.sin(x[4]) * x[8]
            ode[5] = np.cos(x[4]) * x[8] - np.sin(x[4]) * x[9]
            ode[6] = x[9] * np.cos(x[4]) / np.cos(x[5]) + x[8] * np.sin(x[4]) / np.cos(x[5])
            ode[7] = (self.params["Iy"] - self.params["Iz"]) / self.params["Ix"] * x[8] * x[9] + 1. / self.params["Ix"] * m[0]
            ode[8] = (self.params["Iz"] - self.params["Ix"]) / self.params["Iy"] * x[7] * x[9] + 1. / self.params["Iy"] * m[1]
            # EG corrected may 2020 x8x9->x7x9
            ode[9] = (self.params["Ix"] - self.params["Iy"]) / self.params["Iz"] * x[7] * x[8] + 1. / self.params["Iz"] * m[2]
            # EG corrected may 2020 x8x9->x7x8
            ode[10] = 2 * (self.query[0] - self.state[0]) - self.state[3]
            ode[11] = self.query[1] - self.state[7]
            ode[12] = self.query[2] - self.state[8]
            ode[13] = self.query[3] - self.state[9]
            return ode
        return f

    def simulationScipy(self):
        print(int(self.dt_commands / self.dt))
        for i in range(int(self.T / self.dt)):
            if i % (int(self.dt_commands / self.dt)):
                self.PID()
            y = solve_ivp(
                fun=self.ODE(),
                t_span=[0, self.dt],
                y0=self.state,
                t_eval=[0, self.dt],
            )
            self.state = y['y'][:, 1]
            self.trajectory.append(y['y'][:, 1])
            self.t += self.dt
            self.eval_times.append(y['t'][1])

        return np.array(self.trajectory, dtype="float"), np.array(self.eval_times)

    def step_euler(self):
        self.state = self.state + self.dt * self.ODE()(0.0, self.state)

    def check_stop(self):
        return self.state[0] < -2 or self.t > self.T

    def random_query(self, attitude=False, altitude=False):
        self.query_duration = np.random.uniform(0.3, 1.)
        if self.query_active:
            self.query[1:] = np.zeros(3)
        else:
            if attitude:
                self.query[1] = np.random.uniform(-0.3, 0.3)
                self.query[2] = np.random.uniform(-0.3, 0.3)
            if altitude:
                self.query[0] = np.random.uniform(-0., 0.)
        self.query_memory['queries'].append(np.copy(self.query))
        self.query_memory['queries'].append(np.copy(self.query))
        self.query_memory['t'].append(self.t)
        self.query_memory['t'].append(self.t + self.query_duration)
        self.query_active = not self.query_active

    @staticmethod
    def rKN(x, fx, n, hs):
        k1 = np.zeros(14)
        k2 = np.zeros(14)
        k3 = np.zeros(14)
        k4 = np.zeros(14)
        xk = np.zeros(14)
        y = fx(0., x)
        for i in range(n):
            k1[i] = (y[i] * hs)
        for i in range(n):
            xk[i] = x[i] + k1[i] * 0.5

        yxk = fx(0., xk)
        for i in range(n):
            k2[i] = (yxk[i] * hs)
        for i in range(n):
            xk[i] = x[i] + k2[i] * 0.5

        yxk = fx(0., xk)
        for i in range(n):
            k3[i] = yxk[i] * hs
        for i in range(n):
            xk[i] = x[i] + k3[i]

        yxk = fx(0., xk)
        for i in range(n):
            k4[i] = yxk[i] * hs

        x2 = np.zeros(n)
        for i in range(n):
            x2[i] = x[i] + (k1[i] + 2 * (k2[i] + k3[i]) + k4[i]) / 6
        return x2

    def step_RKn(self, n):
        self.state = Quadcopter.rKN(self.state, self.ODE(), n, self.dt)
        TWO_PI = 2 * math.pi
        self.state[4] = (self.state[4] + math.pi) % TWO_PI - math.pi
        self.state[5] = (self.state[5] + math.pi) % TWO_PI - math.pi
        self.state[6] = (self.state[6] + math.pi) % TWO_PI - math.pi

    def M(self):
        m = np.zeros(3)
        m[0] = 4 * self.params["CT"] * self.params["C1"] * self.params["d"] * (
            self.params["C2"] + self.params["C1"] * self.commands[0]
        ) * self.commands[1]
        m[0] += -4 * self.params["C1"]**2 * self.params["CT"] * self.params["d"] * self.commands[2] * self.commands[3]

        m[1] = -4 * self.params["C1"]**2 * self.params["CT"] * self.params["d"] * self.commands[1] * self.commands[3]
        m[1] += 4 * self.params["CT"] * self.params["C1"] * self.params["d"] * (
            self.params["C2"] + self.params["C1"] * self.commands[0]
        ) * self.commands[2]

        m[2] = -2 * self.params["C1"]**2 * self.params["CT"] * self.params["d"] * self.commands[0] * self.commands[1]
        m[2] += 8 * self.params["CD"] * self.params["C1"] * (
            self.params["C2"] + self.params["C1"] * self.commands[0]
        ) * self.commands[3]
        return m

    def F(self):
        f = (self.params["CT"] * (self.params["C1"]**2)) * (np.sum(self.commands[1:]**2) + 4 * self.commands[0]**2)
        f += 8 * self.params["CT"] * self.params["C1"] * self.params["C2"] * self.commands[0]
        f += 4 * self.params["CT"] * self.params["C2"]**2
        return f
