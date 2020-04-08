#!/usr/bin/env python


#
# Author: yasser hifny
#

from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility
import sys
import codecs
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.layers import Embedding, Dense, Input, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model, load_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import RMSprop, Adam, Nadam
from keras.layers import Conv1D, MaxPooling1D
from keras.engine.topology import Layer
from keras import initializers
from keras.preprocessing import sequence
from keras.utils import np_utils
import keras
from keras import optimizers
from keras.models import load_model


import glob
import os
from keras.utils import multi_gpu_model
# import tensorflow as tf
from keras.callbacks import Callback
import warnings
# import tensorflow as tf
import random

random.seed(9001)
import argparse
from numpy import newaxis
from gated_cnn import  GatedConvBlock
from GCNN import GatedConv1D
from novograd import NovoGrad
from my_layers import ContextExpansion
from transformer import Position_Embedding, MultiHeadAttention, LayerNormalization, Gelu




from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
#from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader, MatrixWriter
import config_kaldi as config
import os



#    model = load_model(model_name)
kalid_model = load_model(sys.argv[1], custom_objects={'GatedConvBlock':GatedConvBlock,
                                                        'NovoGrad': NovoGrad, 
                                                        'ContextExpansion' : ContextExpansion,
                                                        'Position_Embedding':Position_Embedding, 
                                                        'MultiHeadAttention':MultiHeadAttention,
                                                        'LayerNormalization':LayerNormalization,
                                                        'Gelu':Gelu})
model = Model(inputs=[kalid_model.input[0] if type(kalid_model.input) is list else kalid_model.input], outputs=[kalid_model.get_layer('output_tri').output])
model.summary()

model.save (os.path.split(os.path.abspath(sys.argv[1]))[0]+'/model_small.hdf5',include_optimizer=False)