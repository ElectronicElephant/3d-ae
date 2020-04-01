import copy

from ae.models.dae import DAE
from ae.models.vae import VAE


# from ae.models.aae import WAAE


def get_model(model_dict, version=None):
    name = model_dict["name"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("name")

    model = model(**param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "dae": DAE,
            "vae": VAE,
            #            "waae": WAAE,
        }[name]
    except:
        raise ("Model {} not available".format(name))
