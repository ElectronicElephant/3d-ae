from ae.loader.basic_loader import BasicLoader


def get_loader(name):
    return {
        "basic": BasicLoader
    }[name]
