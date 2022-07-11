#!/bin/bash

export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH

export CUDA_VISIBLE_DEVICES=0

source  activate  tf_gpu



SCRIPT_PATH=/home/lumi/Dropbox/ALL/timit_work


#inputs
IN_MODEL=/media/lumi/alpha/asr_work_timit/model.hdf5
DATA_SET=dev
IN_DIR=$(dirname "${IN_MODEL}")



rm -f ${IN_DIR}/temp_results.txt
rm -f ${IN_DIR}/log.txt

python $SCRIPT_PATH/fmllr_decode.py $IN_MODEL $DATA_SET $SCRIPT_PATH/priors.csv  $IN_DIR/out.txt >&  $IN_DIR/log.txt

$SCRIPT_PATH/score_timit/score48.sh /media/lumi/alpha/kaldi/egs/timit/s5/data/$DATA_SET/text  $IN_DIR/out.txt >>  $IN_DIR/log.txt

cat /home/lumi/Dropbox/ALL/timit_work/score_timit/log_scoring/hyp48.txt.sclite.sys   

 



