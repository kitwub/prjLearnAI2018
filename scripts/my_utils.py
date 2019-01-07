from inspect import currentframe
import numpy as np
import random
import cupy as cp


def set_random_seed(seed):
    # set Python random seed
    random.seed(seed)

    # set NumPy random seed
    np.random.seed(seed)

    # set Chainer(CuPy) random seed
    cp.random.seed(seed)

# def get_str_of_val_name_on_code(*args):
#     names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
#     # return [names.get(id(arg),'???') for arg in args]
#     return [names[id(arg)] for arg in args]

# def chkprint(*args):
#     names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
#     print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))