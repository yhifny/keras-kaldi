#!/bin/bash

export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH

export CUDA_VISIBLE_DEVICES=1

source  activate  tf_gpu






#inputs
IN_MODEL=/media/lumi/alpha/asr_work_cnn_gru_seq/model.hdf5
DATA_SET=train_clean_100
IN_DIR=$(dirname "${IN_MODEL}")
mkdir -p $IN_DIR




python ./fmllr_align_fusion.py $IN_MODEL $DATA_SET ./priors.csv  $IN_DIR/train_trans_id_dnn.trk >&  $IN_DIR/log_align.txt


 



