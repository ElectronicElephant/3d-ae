from itertools import product
from typing import Generator, Tuple, Union

import nvgpu


class NoGpuAvailable(Exception):
    pass


def get_free_gpu() -> int:
    try:
        return next(map(lambda x: x[0],
                        filter(lambda x: x[1] < 150,
                               map(lambda x: (x[0], x[1]["mem_used"]),
                                   enumerate(nvgpu.gpu_info())))))
    except StopIteration:
        raise NoGpuAvailable("No free gpu available.")


def searcher(hparams: dict) -> Generator[dict, None, None]:
    grids = {key: grid for key, grid in hparams.items() if isinstance(grid, list)}
    keys = list(grids.keys())
    grids = list(grids.values())
    for params in product(*grids):
        output = hparams.copy()
        params = {keys[idx]: value for idx, value in enumerate(params)}
        output.update(params)
        yield output


def conv_size(input_dim: Union[int, Tuple[int]], kernel: int, stride: int, padding: int) \
        -> Tuple[bool, Union[int, Tuple[int]]]:
    def _conv_size(w: int, k: int, s: int, p: int) -> Tuple[bool, int]:
        f: float = (w - k + 2 * p) / s + 1
        return f.is_integer(), int(f)

    if isinstance(input_dim, (tuple, list)):
        _dim = [_conv_size(dim, kernel, stride, padding) for dim in input_dim]

        output_dim = tuple(dim[1] for dim in _dim)
        is_aligned = all([dim[0] for dim in _dim])

        return is_aligned, output_dim
    elif isinstance(input_dim, int):
        return _conv_size(input_dim, kernel, stride, padding)
    else:
        raise TypeError


def validate_conv(dim, layers):
    for layer in layers:
        aligned, dim = conv_size(dim, *layer[2:])
        if not aligned:
            return None
    return dim
