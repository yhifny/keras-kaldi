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
from kaldi.util.table import SequentialMatrixReader
import config_kaldi as config

print (config.graph_file )
print (config.words_mapping_file)

# Construct recognizer
decoder_opts = LatticeFasterDecoderOptions()
decoder_opts.beam = 13
decoder_opts.max_active = 7000
asr = MappedLatticeFasterRecognizer.from_files(
    config.final_model, config.graph_file, config.words_mapping_file,
    acoustic_scale=1.0, decoder_opts=decoder_opts)

print (asr)

def normalize( feature,  feats_mean, feats_std, eps=1e-14):
    return (feature - feats_mean) / (feats_std + eps)



#    model = load_model(model_name)
kalid_model = load_model(sys.argv[1], custom_objects={'GatedConvBlock':GatedConvBlock, 'NovoGrad': NovoGrad, 'ContextExpansion' : ContextExpansion})
model = Model(inputs=[kalid_model.input[0] if type(kalid_model.input) is list else kalid_model.input], outputs=[kalid_model.get_layer('output_tri').output])
model.summary()

# 
feat_norm_file = "mean_std_fmllr.npz"
feats_mean = np.load(feat_norm_file)['mean']
feats_std = np.load(feat_norm_file)['std']

#
feats_rspecifier = config.fmllr_dev_feats_rspecifier
if sys.argv[2]== 'test': feats_rspecifier = config.fmllr_test_feats_rspecifier

# read priors
priors = np.genfromtxt (sys.argv[3], delimiter=',')

# output
out_file = open(sys.argv[4], "w")

# decode
with SequentialMatrixReader(feats_rspecifier) as f:
    for (fkey, feats)   in f:
        print ('processing: ', fkey, flush=True)        	
        feats=normalize(feats.numpy(), feats_mean, feats_std)[newaxis,...]
        loglikes = np.log (model.predict(feats)[0,:,:] / priors)
        loglikes [loglikes == -np.inf] = -100        
        out = asr.decode(Matrix(loglikes))
        out_file.write("%s %s\n" %(fkey, out["text"]))
    
