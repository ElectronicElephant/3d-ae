import argparse
import os
import random
import shutil
import time

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils import data

from ae.augmentations import get_composed_augmentations
from ae.loader import get_loader
from ae.loss import get_loss_function
from ae.metrics import averageMeter
from ae.models import get_model
from ae.optimizers import get_optimizer
from ae.schedulers import get_scheduler
from ae.utils import get_logger


def train(cfg, writer, logger):
    # setup seeds
    seed_num = 1337
    torch.manual_seed(cfg.get("seed", seed_num))
    torch.cuda.manual_seed(cfg.get("seed", seed_num))
    np.random.seed(cfg.get("seed", seed_num))
    random.seed(cfg.get("seed", seed_num))

    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup augmentations
    augmentations = cfg["training"].get("augmentations", None)
    data_aug = get_composed_augmentations(augmentations)

    # setup dataloader
    data_loader = get_loader(cfg["data"]["type"])
    data_path = cfg["data"]["path"]

    t_loader = data_loader(
        data_path,
        array_name=cfg["data"]["array_name"],
        ae_type=cfg["model"]["name"],
        is_transform=True,
        dim=cfg["data"]["dim"],
        augmentations=data_aug,
    )

    train_loader = data.DataLoader(
        t_loader, batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
    )

    # setup model
    model = get_model(cfg["model"]).to(device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k: v for k, v in cfg["training"]["optimizer"].items() if k != "name"}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg["training"]["lr_schedule"])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg["training"]["resume"] is not None:
        if os.path.isfile(cfg["training"]["resume"]):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg["training"]["resume"])
            )
            checkpoint = torch.load(cfg["training"]["resume"])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg["training"]["resume"], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg["training"]["resume"]))

    time_meter = averageMeter()
    i = start_iter
    flag = True

    while i <= cfg["training"]["train_iters"] and flag:
        for data_pack in train_loader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            if cfg["model"]["name"] == "dae":
                images = data_pack[1]
            else:
                images = data_pack[0]
            images = images.to(device)
            target = data_pack[0].to(device)

            optimizer.zero_grad()
            if cfg["model"]["name"] == "vae":
                recon_images, _, mu, logvar = model(images)
                loss = loss_fn(input=recon_images, target=target, mu=mu, logvar=logvar)
            else:
                recon_images, _ = model(images)
                loss = loss_fn(input=recon_images, target=target)
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg["training"]["print_interval"] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(
                    i + 1,
                    cfg["training"]["train_iters"],
                    loss.item(),
                    time_meter.avg / cfg["training"]["batch_size"],
                )

                print(print_str)
                logger.info(print_str)
                writer.add_scalar("loss/train_loss", loss.item(), i + 1)
                time_meter.reset()
            if (i + 1) % cfg["training"]["save_interval"] == 0:
                current_loss = loss.item()
                state = {
                    "epoch": i + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "loss": current_loss,
                }
                save_path = os.path.join(
                    writer.file_writer.get_logdir(),
                    "{}_{}_{}_model.pkl".format(cfg["model"]["name"], cfg["data"]["dataset"], str(i + 1))
                )
                torch.save(state, save_path)
            if (i + 1) == cfg["training"]["train_iters"]:
                flag = False
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument("--config", nargs="?", type=str, default="configs/dae.yaml",
                        help="Configuration file to use", )
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], str(run_id))
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    train(cfg, writer, logger)
