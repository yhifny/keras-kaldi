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



SCRIPT_PATH=/home/lumi/Dropbox/ALL/timit_work


#inputs
IN_MODEL=/media/lumi/alpha/asr_work_fusion/model.hdf5
DATA_SET=test
IN_DIR=$(dirname "${IN_MODEL}")
mkdir -p $IN_DIR



rm -f ${IN_DIR}/temp_results.txt
rm -f ${IN_DIR}/log.txt

python $SCRIPT_PATH/fmllr_align_fusion.py $IN_MODEL $DATA_SET $SCRIPT_PATH/priors.csv  $IN_DIR/train_align_dnn.trk >&  $IN_DIR/log.txt


 



