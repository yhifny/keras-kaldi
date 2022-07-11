#!/usr/bin/env python

from __future__ import print_function
#
# Author: yasser hifny
#


import numpy as np

np.random.seed(1337)  # for reproducibility
import sys
import codecs
from keras.preprocessing import sequence
from keras.utils import np_utils
import cPickle as pickle
from keras.layers import Embedding, Dense, Input, LSTM, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import SpatialDropout1D, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model, load_model
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



from kaldi.alignment import GmmAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.util.table import SequentialMatrixReader

# Construct aligner
aligner = GmmAligner.from_files("gmm-boost-silence --boost=1.0 1 final.mdl - |",
                                "tree", "/mnt/c/yasser/kaldi/egs/timit/s5/data/lang/L.fst", "/mnt/c/yasser/kaldi/egs/timit/s5/data/lang/words.txt", "/mnt/c/yasser/kaldi/egs/timit/s5/data/lang/phones/disambig.int",
                                self_loop_scale=0.1)
phones = SymbolTable.read_text("phones.txt")
wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
                                     "word_boundary.int")

# Define feature pipeline as a Kaldi rspecifier
feats_rspecifier = (
    "ark:compute-mfcc-feats --config=mfcc.conf scp:wav.scp ark:-"
    " | apply-cmvn-sliding --cmn-window=10000 --center=true ark:- ark:-"
    " | add-deltas ark:- ark:- |"
    )


def normalize( feature,  feats_mean, feats_std, eps=1e-14):
    return (feature - feats_mean) / (feats_std + eps)


    model_dir_path = "/work/bcn_emotion/exp_emotion/model_final/*.hdf5"
    models = glob.glob(model_dir_path)


# Align
with SequentialMatrixReader(feats_rspecifier) as f, open("text") as t:
    for (fkey, feats), line in zip(f, t):
        tkey, text = line.strip().split(None, 1)
        assert(fkey == tkey)
        out = aligner.align(feats, text)
        print(fkey, out["alignment"], flush=True)
        phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
        print(fkey, phone_alignment, flush=True)
        word_alignment = aligner.to_word_alignment(out["best_path"], wb_info)
        print(fkey, word_alignment, flush=True)