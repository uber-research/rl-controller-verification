import json
import math
import numpy as np
import os.path


class Quadcopter:

    def __init__(
        self,
        init_state=np.zeros(10),
        dt=0.01,
        dt_commands=0.03,
        T=5.
    ):
        self.init_state = init_state
        self.state = np.copy(init_state)
        self.dt = dt
        self.dt_commands = dt_commands
        self.t = 0.0
        self.T = T
        self.command_memory = []

        DIR_NAME = os.path.dirname(__file__)
        filename = os.path.join(DIR_NAME, 'config.json')
        with open(filename) as config:
            self.params = json.load(config)["quadcopter"]

        self.f = self.ODE()
        self.commands = np.zeros(4)
        self.coeff = np.array([100000., 200., 200., 200.])
        self.bias = np.array([0., -100., -100., -100.])
        self.command_factor = int(self.dt_commands / self.dt)

    def reset(self):
        self.state = np.array(self.init_state)
        self.state[4:7] = np.random.uniform(-math.pi / 6, math.pi / 6, 3)
        # self.state[4:7] = np.array([math.pi / 6, -0.1, 0.3])
        self.t = 0.0
        self.query_duration = np.random.uniform(0.9, 2.0)
        self.query_active = True

    def next_step(self, commands):
        self.commands = commands
        for _ in range(self.command_factor):
            self.step_RKn(10)
            self.t += self.dt
            if self.check_stop():
                break
        return self.check_stop()

    def ODE(self):
        def f(t, x):
            m = self.M()
            f = self.F()
            ode = np.empty(10)
            ode[0] = -np.sin(x[5]) * x[1] + np.cos(x[5]) * np.sin(x[4]) * x[2]
            ode[0] += np.cos(x[5]) * np.cos(x[4]) * x[3]

            ode[1] = x[9] * x[2] - x[8] * x[3]
            ode[1] += np.sin(x[5]) * self.params["g"]

            ode[2] = -x[9] * x[1] + x[7] * x[3] - np.cos(
                x[5]) * np.sin(x[4]) * self.params["g"]

            ode[3] = x[8] * x[1] - x[7] * x[2] - np.cos(
                x[5]) * np.cos(x[4]) * self.params["g"] + f / self.params["m"]
            ode[4] = x[7] + np.cos(x[4]) * np.tan(
                x[5]) * x[9] + np.tan(x[5]) * np.sin(x[4]) * x[8]
            ode[5] = np.cos(x[4]) * x[8] - np.sin(x[4]) * x[9]
            ode[6] = x[9] * np.cos(x[4]) / np.cos(
                x[5]) + x[8] * np.sin(x[4]) / np.cos(x[5])
            ode[7] = (
                self.params["Iy"] - self.params["Iz"]
            ) / self.params["Ix"] * x[8] * x[9] + 1. / self.params["Ix"] * m[0]
            ode[8] = (
                self.params["Iz"] - self.params["Ix"]
            ) / self.params["Iy"] * x[7] * x[9] + 1. / self.params["Iy"] * m[1]
            # corrected EG may 2020 (x8*x9->x7*x9)
            ode[9] = (
                self.params["Ix"] - self.params["Iy"]
            ) / self.params["Iz"] * x[7] * x[8] + 1. / self.params["Iz"] * m[2]
            # corrected EG may 2020 (x8*x9->x7*x8)
            return ode
        return f

    def step_euler(self):
        self.state = self.state + self.dt * self.ODE()(0.0, self.state)

    def check_stop(self):
        return self.state[0] < -5. or self.t > self.T

    @staticmethod
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

    def step_RKn(self, n):
        self.state = Quadcopter.rKN(self.state, self.ODE(), n, self.dt)
        # self.state[4] = (self.state[4]) % TWO_PI
        # self.state[5] = (self.state[5]) % TWO_PI
        # self.state[6] = (self.state[6]) % TWO_PI

    def M(self):
        m = np.zeros(3)
        m[0] = 4 * self.params["CT"] * self.params["C1"] * self.params["d"] * (
            self.params["C2"] + self.params["C1"] * self.commands[0]
        ) * self.commands[1]
        m[0] += -4 * self.params["C1"]**2 * self.params["CT"] * self.params["d"] * self.commands[2] * self.commands[3]  # noqa E261

        m[1] = -4 * self.params["C1"]**2 * self.params["CT"] * self.params["d"] * self.commands[1] * self.commands[3]  # noqa E261
        m[1] += 4 * self.params["CT"] * self.params["C1"] * self.params["d"] * (  # noqa E261
            self.params["C2"] + self.params["C1"] * self.commands[0]
        ) * self.commands[2]

        m[2] = -2 * self.params["C1"]**2 * self.params["CD"] * self.commands[1] * self.commands[2]  # noqa E261
        m[2] += 8 * self.params["CD"] * self.params["C1"] * (
            self.params["C2"] + self.params["C1"] * self.commands[0]
        ) * self.commands[3]
        return m

    def F(self):
        f = (
            self.params["CT"] * (self.params["C1"]**2)
        ) * (4 * (
            self.commands[1]**2 + self.commands[0]**2
        ) + self.commands[1]**2 + self.commands[2]**2)

        f += 8 * self.params["CT"] * self.params["C1"] * self.params["C2"] * self.commands[0]  # noqa E261
        f += 4 * self.params["CT"] * self.params["C2"]**2
        return f
