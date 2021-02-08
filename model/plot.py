import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import math
import pdb
import logging

def display_trajectory(Q, y,t, w, cmd, t_cmd, title, query_memory=None):
    logging.info("display_trajectory")
    w = np.array(w)
    if query_memory is not None:
        if query_memory.get("queries", None) is not None:
            memo = np.array(query_memory['queries'], dtype="float")

        if query_memory.get("rate_queries", None) is not None:
            memo2 = np.array(query_memory['rate_queries'], dtype="float")
        queries_t = query_memory['t']

    f = plt.figure('1', figsize=(30,30))
    plt.clf()
    plt.suptitle(title)
    plt.subplot(541)
    plt.title(f"{title} - Altitude")
    plt.plot(t, y[:, 0], label='Altitude')
    if query_memory.get("queries", None) is not None:
        plt.plot(query_memory['t'], memo[:, 0], label='query')
    plt.xlabel("time (s)")
    plt.legend()

    plt.subplot(542)
    plt.title(f"{title} - Vertical Speed")
    v = - np.sin(y[:, 5]) * y[:, 1] + np.cos(y[:, 5]) * np.sin(y[:, 4]) * y[:, 2] + np.cos(y[:, 5]) * np.cos(y[:, 4]) * y[:, 3]
    plt.plot(t, v, label='Vertical Speed')
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.legend()

    plt.subplot(543)
    plt.title(f"{title} - Thrust")
    plt.plot(t_cmd, cmd[:, 0])
    plt.ylim(40000, 70000)
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()



    plt.subplot(545)
    plt.title(f"{title} - Roll Angle")
    plt.plot(t, y[:, 4], label='Roll')
    if query_memory.get("queries", None) is not None:
        plt.plot(query_memory['t'], memo[:, 1], label='query')
    plt.yticks([- math.pi, 0, math.pi], [r'$-\pi$', 0, r'$\pi$'])
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.legend()

    plt.subplot(546)
    plt.title(f"{title} - Pitch Angle")
    plt.plot(t, y[:, 5], label='Pitch')
    if query_memory.get("queries", None) is not None:
        plt.plot(query_memory['t'], memo[:, 2], label='query')
    plt.yticks([- math.pi, 0, math.pi], [r'$-\pi$', 0, r'$\pi$'])
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.legend()

    plt.subplot(547)
    plt.title(f"{title} - Yaw Angle")
    plt.plot(t, y[:, 6], label="Yaw")
    if query_memory.get("queries", None) is not None:
        plt.plot(query_memory['t'], memo[:, 3], label='query')
    plt.yticks([- math.pi, 0, math.pi], [r'$-\pi$', 0, r'$\pi$'])
    plt.xlabel("time (s)")
    plt.ylabel("angle (rad)")
    plt.legend()

    plt.subplot(549)
    plt.title(f"{title} - Roll rate")
    plt.plot(t, y[:, 7], label='roll rate')
    if query_memory is not None and query_memory.get("rate_queries", None) is not None:
        plt.plot(queries_t, memo2[:, 0], label='query')
    plt.xlabel("time (s)")
    plt.ylabel("angular rate (rad/s)")
    plt.legend()

    plt.subplot(5,4,10)
    plt.title(f"{title} - Pitch rate")
    plt.plot(t, y[:, 8], label='pitch rate')
    if query_memory is not None and query_memory.get("rate_queries", None) is not None:
        plt.plot(queries_t, memo2[:, 1], label='query')
    plt.xlabel("time (s)")
    plt.ylabel("angular rate (rad/s)")
    plt.legend()

    plt.subplot(5,4,11)
    plt.title(f"{title} - Yaw rate")
    plt.plot(t, y[:, 9], label="yaw rate")
    if query_memory is not None and query_memory.get("rate_queries", None) is not None:
        plt.plot(queries_t, memo2[:, 2], label='query')
    plt.xlabel("time (s)")
    plt.ylabel("angular rate (rad/s)")
    plt.legend()

    plt.subplot(5, 4, 13)
    plt.title("Roll cmd")
    plt.plot(t_cmd, cmd[:, 1])
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()

    plt.subplot(5, 4, 14)
    plt.title("Pitch cmd")
    plt.plot(t_cmd, cmd[:, 2])
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()

    plt.subplot(5, 4, 15)
    plt.title("Yaw cmd")
    plt.plot(t_cmd, cmd[:, 3])
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()

    plt.subplot(5, 4, 17)
    plt.title("w_1")
    plt.plot(t_cmd, w[:, 0])
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()

    plt.subplot(5, 4, 18)
    plt.title("w_2")
    plt.plot(t_cmd, w[:, 1])
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()

    plt.subplot(5, 4, 19)
    plt.title("w_3")
    plt.plot(t_cmd, w[:, 2])
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()

    plt.subplot(5, 4, 20)
    plt.title("w_4")
    plt.plot(t_cmd, w[:, 3])
    plt.xlabel("time (s)")
    plt.ylabel("rotor speed (rad/s)")
    plt.legend()
    return f


def animate_trajectory(Q, y, acceleration_factor=15, zoom_factor=3, notebook=False, title=""):
    # First set up the figure, the ay[:, s, and the plot element we want to animate
    fig = plt.figure("Figure 2")
    plt.title(title)
    y = y[::acceleration_factor]
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-8, 2)

    line1, = ax.plot([], [], [], lw=2, c='b')
    line2, = ax.plot([], [], [], lw=2, c='r')

    d = Q.params['d']
    p1 = np.array([0, -zoom_factor * d, 0])
    p2 = np.array([0, zoom_factor * d, 0])
    p3 = np.array([-zoom_factor * d, 0, 0])
    p4 = np.array([zoom_factor * d, 0, 0])

    # initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        line1.set_3d_properties([])
        line2.set_data([], [])
        line2.set_3d_properties([])
        return line1, line2

    # animation function.  This is called sequentially
    def animate(i):
        r = R.from_euler('zyx', [y[i, 6], y[i, 5], y[i, 4]], degrees=False)
        p1R = r.apply(p1)
        p2R = r.apply(p2)
        p3R = r.apply(p3)
        p4R = r.apply(p4)
        line1.set_data([p1R[0], p2R[0]], [p1R[1], p2R[1]])
        line1.set_3d_properties([y[i, 0], y[i, 0]])
        line2.set_data([p3R[0], p4R[0]], [p3R[1], p4R[1]])
        line2.set_3d_properties([y[i, 0], y[i, 0]])
        return line1, line2

    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(y),
        interval=Q.dt * 1000. * acceleration_factor,
        blit=True
    )
    return ani
