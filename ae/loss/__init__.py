import functools
import logging

from ae.loss.loss import (
    dae_loss,
    vae_loss,
)

logger = logging.getLogger("ae")

key2loss = {
    "dae": dae_loss,
    "vae": vae_loss,
}


def get_loss_function(cfg):
    if cfg["training"]["loss"] is None:
        raise ValueError("No default loss function, you must determine at least one")
    else:
        loss_dict = cfg["training"]["loss"]
        loss_name = loss_dict["name"]
        loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

        if loss_name not in key2loss:
            raise NotImplementedError("Loss {} not implemented".format(loss_name))

        logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name], **loss_params)
