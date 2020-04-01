import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ae.loader import get_loader
from ae.models import get_model
from ae.utils import convert_state_dict

cat_dict_ = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
             9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
             16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
             24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
             34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
             40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
             46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana',
             53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza',
             60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
             70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
             78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock',
             86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}


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

    # Setup Dataloader
    data_loader = get_loader("basic")
    data_path = cfg["testing"]["path"]

    v_loader = data_loader(
        data_path,
        is_transform=False,
        # test_file=cfg["testing"]["test_file"],
        train_file=cfg["testing"]["test_file"],
        img_size=(cfg["testing"]["img_h"], cfg["testing"]["img_w"]),
        test_mode=True,
    )

    val_loader = data.DataLoader(
        v_loader,
        batch_size=cfg["testing"]["batch_size"],
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

    data_counter = 0
    # Initialize the (image, feature) set
    img_fea_dict = {xx: [] for xx in cat_dict_.values()}

    # Configs
    show_image_freq = 0

    # Method 2 to avoid memory bombs
    last_list = []
    last_cat = 0
    current_cat = -1

    for data in val_loader:
        data_counter += 1

        data_ = data[0].to(device)
        recon_image, vector = model(data_)
        # print(recon_image.shape)
        # print(vector.shape)

        if cfg["testing"]["save_recon"]:
            if show_image_freq and data_counter % show_image_freq == 0:
                fig, axs = plt.subplots(2, 1)
                fig.tight_layout()
                axs[0].imshow(data[0].detach().cpu().numpy()[0][0])
                axs[1].imshow(recon_image.detach().cpu().numpy()[0][0])
                plt.show()

            for i in range(len(data[1])):
                current_cat = data[1][i]
                if last_cat == 0:
                    last_cat = current_cat
                # img_fea_dict[data[1][i]].append((data[0].detach().cpu().numpy()[i][0], vector.detach().cpu().numpy()[i]))

                if data[1][i] != last_cat:
                    np.save(f'data/img_features/{last_cat}', last_list)
                    print(f'{last_cat} saved. {len(last_list)} images.')
                    last_list = []
                    last_cat = current_cat

                if current_cat != 'person':
                    last_list.append((data[0].detach().cpu().numpy()[i][0], vector.detach().cpu().numpy()[i]))

            # print(img_fea_dict)
            # print(img_fea_dict.shape)
            # break

        if data_counter % 1 == 0:
            print(f"{data_counter} of {len(val_loader)} epo processed.")

    # if cfg["testing"]["save_recon"]:
    #     for cat in img_fea_dict.keys():
    #         np.save(f'data/img_features/{cat}', img_fea_dict[cat])
    #         print(f'{cat} saved. {len(img_fea_dict[cat])} images.')
    np.save(f'data/img_features/{last_cat}', last_list)
    print(f'{last_cat} saved. {len(last_list)} images.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?", type=str, default="configs/dae.yaml", help="Configuration file to use")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    generateShapeVector(cfg)
