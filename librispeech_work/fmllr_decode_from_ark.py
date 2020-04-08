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





from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
#from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader, MatrixWriter
import config_kaldi as config

print (config.graph_file )
print (config.words_mapping_file)

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 15
decoder_opts.max_active = 7000
asr = MappedLatticeFasterRecognizer.from_files(
    config.final_model, config.graph_file, config.words_mapping_file,
    acoustic_scale=0.0625, decoder_opts=decoder_opts)

print (asr)




#
loglikes_rspecifier = "ark:"+sys.argv[2]

# output
out_file = open(sys.argv[4], "w")



# decode
with SequentialMatrixReader(loglikes_rspecifier) as f:
    for (fkey, loglikes)   in f:
        print ('processing: ', fkey, flush=True)        	
        out = asr.decode(loglikes)
        out_file.write("%s %s\n" %(fkey, out["text"]))

