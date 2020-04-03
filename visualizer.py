from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np


# from mpl_toolkits.mplot3d import Axes3D


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
