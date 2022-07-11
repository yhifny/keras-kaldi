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





from kaldi.alignment import MappedAligner

#from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader, IntVectorWriter
import config_kaldi as config



aligner = MappedAligner.from_files(config.final_model, 
            config.tree_file, 
            config.lexicon_file, 
            config.words_mapping_file, 
            config.disambig_file, 
            graph_compiler_opts=None, 
            beam=200.0, transition_scale=1.0, 
            self_loop_scale=1.0, 
            acoustic_scale=1.0)

def normalize( feature,  feats_mean, feats_std, eps=1e-14):
    return (feature - feats_mean) / (feats_std + eps)



#    model = load_model(model_name)
models_dir_path = "/media/lumi/alpha/gmm*/asr_work*/model.hdf5" 
models=[]

for  filename in glob.glob(models_dir_path):
    print (filename)
    kalid_model = load_model(filename, custom_objects={'GatedConvBlock':GatedConvBlock, 'NovoGrad': NovoGrad})
    model = Model(inputs=[kalid_model.input[0] if type(kalid_model.input) is list else              kalid_model.input], outputs=[kalid_model.get_layer('output_tri').output])
    model.summary()
    models.append(model)

# 
feat_norm_file = "mean_std_fmllr.npz"
feats_mean = np.load(feat_norm_file)['mean']
feats_std = np.load(feat_norm_file)['std']

#
feats_rspecifier = config.fmllr_train_feats_rspecifier

# read priors
priors = np.genfromtxt (sys.argv[3], delimiter=',')

# output

alignment_wspecifier="ark:"+ sys.argv[4]
alignment_writer = IntVectorWriter(alignment_wspecifier)
# align
with SequentialMatrixReader(feats_rspecifier) as f, open(config.train_transcription_file) as t:
    for (fkey, feats), line in zip(f, t):
        tkey, text = line.strip().split(None, 1)
        assert(fkey == tkey)
        print ('processing: ', fkey, flush=True)        	
        feats=normalize(feats.numpy(), feats_mean, feats_std)[newaxis,...]
        pred_stack = np.empty((0, len(models)), float)
        pred_list=[]
        for model in models:
            pred=model.predict(feats)[0,:,:]
            pred_list.append(pred)
        mean_pred=np.rollaxis(np.dstack(pred_list),-1).mean(axis=0)
        loglikes = np.log (mean_pred / priors)
        loglikes [loglikes == -np.inf] = -100        
        out = aligner.align(Matrix(loglikes), text)
        print(fkey, out["alignment"], flush=True)
        alignment_writer[fkey]= out["alignment"]
        phone_alignment = aligner.to_phone_alignment(out["alignment"])
        print(fkey, phone_alignment, flush=True)

alignment_writer.close()