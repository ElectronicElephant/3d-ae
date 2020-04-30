from main import DenoisingAutoEncoder
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from torch import FloatTensor

ckpt = torch.load("epoch=74_11.ckpt", map_location=torch.device("cuda"))
hparams = ckpt.get("hparams")
state_dict = ckpt.get("state_dict")
model = DenoisingAutoEncoder(hparams=hparams, train_set=("val.npz", "arr_0"), val_set=("val.npz", "arr_0"))
model.load_state_dict(state_dict)
voxels = np.load("test.npz")["arr_0"]
model.bs = 1
model.freeze()

"""
voxel_list = []
for i in [3,33,123, 223]:
    print(i)
    voxel = voxels[i]

    output_voxel = model(FloatTensor([[voxel]]))

    out_voxel = output_voxel[0].numpy()[0][0].astype(bool)

    voxel_list.append(out_voxel)

np_out = np.stack(voxel_list)
np.savez_compressed("output.npz", np_out)
"""
import binvox_rw
output_voxel = model(FloatTensor([[voxels[33]]]))
out_voxel = output_voxel[0].numpy()[0][0]
# out_voxel = voxels[133]
out_voxel[out_voxel>=0.3] = 1
out_voxel[out_voxel<0.3] = 0
out_voxel=out_voxel.astype(int)
vox = binvox_rw.Voxels(out_voxel, [128,128,128], [0,0,0], 1, "xyz")
vox.write(open("output.binvox", mode="wb"))
# ---
"""
verts, faces, normals, values = measure.marching_cubes_lewiner(out_voxel, 0.1, step_size=1)

# Display resulting triangular mesh using Matplotlib. This can also be done
# with mayavi (see skimage.measure.marching_cubes_lewiner docstring).
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# Fancy indexing: `verts[faces]` to generate a collection of triangles
mesh = Poly3DCollection(verts[faces])
# mesh.set_edgecolor('k')
ax.add_collection3d(mesh)

ax.set_xlabel("x-axis")
ax.set_ylabel("y-axis")
ax.set_zlabel("z-axis")

ax.set_xlim3d(0, 128)
ax.set_ylim3d(0, 128)
ax.set_zlim3d(0, 128)

plt.tight_layout()
plt.show()
"""
