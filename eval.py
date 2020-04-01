import argparse
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import torch
import yaml

from ae.loader import get_loader
from ae.models import get_model
from ae.utils import convert_state_dict


def plot(array1, array2=None):
    r, g, b = np.indices((33,33,33)) / 32.0

    fig = plt.figure()

    ax = fig.add_subplot(121, projection='3d')
    ax.voxels(r, g, b, array1, linewidth=0.5)

    if array2 is not None:
        ax = fig.add_subplot(122, projection='3d')
        ax.voxels(r, g, b, array2, linewidth=0.5)

    plt.show()


def generateShapeVector(cfg):
    '''
    PLEASE BE SURE TO USE BATCH_SIZE = 1
    :param cfg:
    :return:
    '''
    from torch.utils import data
    # Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(cfg["testing"]["save_path"]):
        os.makedirs(cfg["testing"]["save_path"])
        os.makedirs(cfg["testing"]["save_path"] + "/vectors")
        if cfg["testing"]["save_recon"]:
            os.makedirs(cfg["testing"]["save_path"] + "/recons")

    # setup dataloader
    data_loader = get_loader(cfg["data"]["type"])
    data_path = cfg["data"]["path"]

    v_loader = data_loader(
        data_path,
        array_name=cfg["data"]["array_name"],
        is_transform=False,
        dim=cfg["data"]["dim"],
        test_mode=True
    )

    val_loader = data.DataLoader(
        v_loader, batch_size=cfg["testing"]["batch_size"],
        num_workers=cfg["testing"]["n_workers"],
        shuffle=False,
    )

    # Setup Model
    model_path = cfg["testing"]["model_path"]
    model = get_model(cfg["model"])
    state = convert_state_dict(torch.load(model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for data_counter, data in enumerate(val_loader):
        data_ = data[0].to(device)
        recon_image, vector = model(data_)
        # print(recon_image.shape)
        # print(vector.shape)

        if cfg["testing"]["vis"]:
            source = data[0].detach().cpu().numpy().reshape(32, 32, 32)
            dest = recon_image.detach().cpu().numpy().reshape(32, 32, 32)

            print(np.unique(dest))

            plot(source, dest > 0.1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?", type=str, default="configs/dae.yaml", help="Configuration file to use")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    generateShapeVector(cfg)
