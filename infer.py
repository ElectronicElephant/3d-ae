from main import DenoisingAutoEncoder
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from torch import FloatTensor
import open3d as o3d

import random


M = np.array([[0.80656762, -0.5868724, -0.07091862],
              [0.3770505, 0.418344, 0.82632997],
              [-0.45528188, -0.6932309, 0.55870326]])


def PointCloud(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


ckpt = torch.load("epoch=119.ckpt", map_location=torch.device("cuda"))
hparams = ckpt.get("hparams")
state_dict = ckpt.get("state_dict")
model = DenoisingAutoEncoder(hparams=hparams, train_set=("s2_train.npz", "filled_dense"), val_set=("s2_test.npz", "filled_dense"))
model.load_state_dict(state_dict)
# voxels = np.load("s2_test.npz")["filled_dense"]
voxels = np.load("s2_train.npz")["filled_dense"]
model.bs = 1
model.freeze()

index = list(range(len(voxels)))
random.shuffle(index)

for idx in index:
    output_voxel = model(FloatTensor([[voxels[idx]]]))
    out_voxel = output_voxel[0].numpy()[0][0]

    out = (out_voxel > 0.5).astype(np.int)
    out_coords = np.floor(np.array(np.nonzero(out))).T
    voxel_coords = np.floor(np.array(np.nonzero(voxels[idx]))).T

    pcd = PointCloud(voxel_coords)
    pcd.estimate_normals()
    pcd.translate([0.6 * 128, 0, 0])

    pcd2 = PointCloud(out_coords)
    pcd2.estimate_normals()
    pcd2.translate([-0.6 * 128, 0, 0])

    o3d.visualization.draw_geometries([pcd, pcd2])
