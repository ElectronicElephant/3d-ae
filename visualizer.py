from io import BytesIO

import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
from io import StringIO

def plot_as_f(*args, **kwargs):
    plt.close()
    plot(*args, **kwargs)

    fp = BytesIO()
    plt.savefig(fp, format="png")
    fp.seek(0)

    return fp


def plot(array1, array2=None):
    r, g, b = np.indices((33, 33, 33)) / 32.0

    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121, projection='3d')
    ax.voxels(r, g, b, array1, linewidth=0.5)

    if array2 is not None:
        ax = fig.add_subplot(122, projection='3d')
        ax.voxels(r, g, b, array2, linewidth=0.5)


def plot_volume(array: np.ndarray):
    _, x, y, z = array.shape
    X, Y, Z = np.mgrid[:x, :y, :z]

    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=array.flatten(),
        isomin=0,
        isomax=1,
        opacity=0.1,
        opacityscale="max",
        surface_count=5
    ))

    f = StringIO()
    fig.write_html(f, full_html=False)

    f.seek(0)
    return f.read()
