#!/usr/bin/env python


#
# Author: yasser hifny
#

from __future__ import print_function

import numpy as np

np.random.seed(1337)  # for reproducibility
import sys
import codecs
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, Dense, Input, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import SpatialDropout1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing import sequence
import tensorflow.keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model

from tensorflow_asr.models.encoders.conformer import ConformerEncoder
import model as model_definition


import glob
import os
# import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import warnings
# import tensorflow as tf
import random

random.seed(9001)
import argparse
from numpy import newaxis
#from gated_cnn import  GatedConvBlock
from GCNN import GatedConv1D
from novograd import NovoGrad
from my_layers import ContextExpansion
from transformer import Position_Embedding, MultiHeadAttention, LayerNormalization, Gelu
from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.models import model_from_json
import yaml

from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
#from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader, MatrixWriter
import config_kaldi as config
#from tensorflow.keras_transformer.transformer import TransformerBlock

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

   

with open(sys.argv[5]) as stream:
    try:
        cfg=yaml.safe_load(stream)
        print(cfg)
    except yaml.YAMLError as exc:
        print(exc)



def normalize( feature,  feats_mean, feats_std, eps=1e-14):
    return (feature - feats_mean) / (feats_std + eps)


get_custom_objects().update({
    #'GatedConvBlock': GatedConvBlock,
    'NovoGrad': NovoGrad,
    'ContextExpansion': ContextExpansion,
    'Position_Embedding': Position_Embedding,
})


feat_norm_file = "mean_std_fmllr.npz"
feats_mean = np.load(feat_norm_file)['mean']
feats_std = np.load(feat_norm_file)['std']
feats_variance = feats_std**2

#json_file = open(os.path.dirname(sys.argv[1])+'/model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json, custom_objects={'ConformerEncoder':ConformerEncoder({'type': 'conv2d', 'filters': 144, 'kernel_size': 3, 'strides': 1})})


model , parallel_model = model_definition.create_model(cfg, feats_mean, feats_variance)


# load weights into new model
model.load_weights(sys.argv[1])
print("Loaded model from disk")

model.summary()

feats_rspecifier = config.fmllr_dev_feats_rspecifier
if sys.argv[2]== 'test': feats_rspecifier = config.fmllr_test_feats_rspecifier

# read priors
priors = np.genfromtxt (sys.argv[3], delimiter=',')

# output
out_file = open(sys.argv[4], "w")

# posterior writer
posterior_writer = MatrixWriter("ark:"+sys.argv[4]+'.ark')


# decode
with SequentialMatrixReader(feats_rspecifier) as f:
    for (fkey, feats)   in f:
        print ('processing: ', fkey, flush=True)        	
        feats= feats.numpy()[newaxis,...]
        #feats_len = [len(x) for x in feats]
        feats_len = np.asarray(len(feats))
        print(feats.shape, feats_len)
        if cfg['model']['name'] == 'contextnet':
          loglikes = np.log (model.predict([feats,np.asarray([feats_len])])[0][0,:,:] / priors)
        else: 
          loglikes = np.log (model.predict(feats)[0][0,:,:] / priors)
        loglikes [loglikes == -np.inf] = -100        
        out = asr.decode(Matrix(loglikes))
        out_file.write("%s %s\n" %(fkey, out["text"]))
        posterior_writer[fkey]=Matrix(loglikes)

posterior_writer.close()
