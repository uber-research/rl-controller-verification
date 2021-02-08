import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation as R

def simple_plot(Quadcopter, y, t):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[0 for i in t],
                y=[0 for i in t],
                z=y[:, 0]
            )
        ]
    )
    fig.show()


def animate_trajectory(Q, y, t, zoom_factor=1):

    N = len(y)
    d = Q.params['d']
    _p1 = np.array([0, -zoom_factor * d, 0])
    _p2 = np.array([0, zoom_factor * d, 0])
    _p3 = np.array([-zoom_factor * d, 0, 0])
    _p4 = np.array([zoom_factor * d, 0, 0])
    p = np.zeros((3, 4))
    p[:, 0] = _p1
    p[:, 1] = _p2
    p[:, 2] = _p3
    p[:, 3] = _p4

    p1, p2 = np.zeros((N, 3)), np.zeros((N, 3))
    p3, p4 = np.zeros((N, 3)), np.zeros((N, 3))
    top = np.zeros((N, 3))
    top[:, 2] += 0.2 * zoom_factor
    c = np.zeros((N, 3))
    c[:, 2] = y[:, 0]

    for i in range(N):
        r = R.from_euler('zyx', [y[i, 6], y[i, 5], y[i, 4]], degrees=False)
        z = y[i, 0]
        p1[i] = r.apply(_p1)
        p2[i] = r.apply(_p2)
        p3[i] = r.apply(_p3)
        p4[i] = r.apply(_p4)
        top[i] = r.apply(top[i])
        p1[i, 2] += z
        p2[i, 2] += z
        p3[i, 2] += z
        p4[i, 2] += z
        top[i, 2] += z
    # Create figure
    # fig = go.Figure(
    #     data = [],
    #     frames=[
    #         go.Frame(
    #             data=[go.Scatter3d(
    #                 x=[y[k, 0]],
    #                 y=[y[k, 3]],
    #                 z=[0]
    #             ) for k in range(len(y))]
    #         )
    #     ]
    # )

    # Generate curve data
    print(Q.dt_commands * 1000 / Q.T)
    k = 1
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=[-1, -1, 1, 1],
                y=[-1, 1, -1, 1],
                z=[-5, -5, -5, -5],
                # colorbar_title='z',
                # colorscale=[[0, 'gold'],
                #             [0.5, 'mediumturquoise'],
                #             [1, 'magenta']],
                # # Intensity of each vertex, which will be interpolated and color-coded
                # intensity=[0, 0.33, 0.66, 1],
                # i, j and k give the vertices of triangles
                # here we represent the 4 triangles of the tetrahedron surface
                # i=[0, 2],
                # j=[1, 3],
                # k=[1, 3],
                name='floor',
                contour=dict(show=True, color="red", width=10),
                showscale=True
            ),
            {
                "type":"scatter3d",
                "x": [p1[k, 0], p2[k, 0], p3[k, 0], p4[k, 0], c[k, 0]],
                "y": [p1[k, 1], p2[k, 1], p3[k, 1], p4[k, 1], c[k, 1]],
                "z": [p1[k, 2], p2[k, 2], p3[k, 2], p4[k, 2], c[k, 2]],
                "mode": "markers",
                "marker": dict(color="red", size=4),
                "name": "Rotors",
                "text": "ok"
            },
            {
                "type":"scatter3d",
                "x": [c[k, 0]],
                "y": [c[k, 1]],
                "z": [c[k, 2]],
                "mode": "markers",
                "name": "Center",
                "marker": dict(color="blue", size=8)
            },
            {
                "type": "scatter3d",
                "x":[p2[k, 0], p1[k, 0]],
                "y":[p2[k, 1], p1[k, 1]],
                "z":[p2[k, 2], p1[k, 2]],
                "mode": "lines",
                "line": dict(color="purple", width=5),
                "name": "Line 1",
            },
            {
                "type": "scatter3d",
                "x":[p3[k, 0], p4[k, 0]],
                "y":[p3[k, 1], p4[k, 1]],
                "z":[p3[k, 2], p4[k, 2]],
                "mode": "lines",
                "line": dict(color="yellow", width=5),
                "name": "Line 2",
            },
            {
               "type": "scatter3d",
                "x":[top[k, 0]],
                "y":[top[k, 1]],
                "z":[top[k, 2]],
                "mode": "markers",
                "marker": dict(color="green", size=2),
                "name": "Top",
            }
        ],
        layout={"updatemenus": [dict(
            type="buttons",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[
                        None,
                        {
                            "frame": {"duration": Q.dt_commands * 1000 / Q.T, 'redraw': True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[
                        [None],
                        {
                            "frame": {"duration": 0, 'redraw': True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                )]
            )],
            "scene": dict(
                xaxis=dict(
                    range=[-0.3, 0.3],
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    range=[-0.3, 0.3],
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    range=[-5, 1],
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='manual'
            ),
        },
        frames=[go.Frame(
            data=[
                go.Mesh3d(
                    x=[-1, -1, 1, 1],
                    y=[-1, 1, -1, 1],
                    z=[-5, -5, -5, -5],
                    # colorbar_title='z',
                    # colorscale=[[0, 'gold'],
                    #             [0.5, 'mediumturquoise'],
                    #             [1, 'magenta']],
                    # # Intensity of each vertex, which will be interpolated and color-coded
                    # intensity=[0, 0.33, 0.66, 1],
                    # i, j and k give the vertices of triangles
                    # here we represent the 4 triangles of the tetrahedron surface
                    # i=[0, 2],
                    # j=l1, 3],
                    # k=[1, 3],
                    name='floor',
                    color='grey',
                    showscale=True
                ),
                {
                    "type": "scatter3d",
                    "x": [p1[k, 0], p2[k, 0], p3[k, 0], p4[k, 0], c[k, 0]],
                    "y": [p1[k, 1], p2[k, 1], p3[k, 1], p4[k, 1], c[k, 1]],
                    "z": [p1[k, 2], p2[k, 2], p3[k, 2], p4[k, 2], c[k, 2]],
                    "mode": "markers",
                    "marker": dict(color="red", size=4),
                    "name": "Rotors",
                    "text": "ok"
                },
                {
                    "type":"scatter3d",
                    "x": [c[k, 0]],
                    "y": [c[k, 1]],
                    "z": [c[k, 2]],
                    "mode": "markers",
                    "name": "Center",
                    "marker": dict(color="blue", size=8)
                },
                {
                    "type": "scatter3d",
                    "x": [p2[k, 0], p1[k, 0]],
                    "y": [p2[k, 1], p1[k, 1]],
                    "z": [p2[k, 2], p1[k, 2]],
                    "mode": "lines",
                    "line": dict(color="purple", width=5),
                    "name": "Line 1",
                    "visible": True,
                },
                {
                    "type": "scatter3d",
                    "x":[p3[k, 0], p4[k, 0]],
                    "y":[p3[k, 1], p4[k, 1]],
                    "z":[p3[k, 2], p4[k, 2]],
                    "mode": "lines",
                    "line": dict(color="yellow", width=5),
                    "name": "Line 2",
                    "visible": True,
                },
                {
                    "type": "scatter3d",
                    "x":[top[k, 0]],
                    "y":[top[k, 1]],
                    "z":[top[k, 2]],
                    "mode": "markers",
                    "marker": dict(color="green", size=2),
                    "name": "Top",
                }
            ]
        ) for k in range(N)]
    )
    fig.show()
