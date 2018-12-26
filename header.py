import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.backend.tensorflow_backend import set_session
import numpy as np
import pandas as pd

set_session(
    tf.Session(
        config=tf.ConfigProto(
            gpu_options=tf.GPUOptions(
                # visible_device_list="0", # (bad effect?)specify GPU number
                allow_growth=True))))

