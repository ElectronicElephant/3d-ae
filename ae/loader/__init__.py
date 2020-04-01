from ae.loader.basic_loader import BasicLoader
from ae.loader.npz_loader import NPZLoader


def get_loader(name):
    return {
        "basic": BasicLoader,
        "npz": NPZLoader
    }[name]
