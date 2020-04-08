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



SCRIPT_PATH=/home/lumi/Dropbox/ALL/librispeech_work/score_librispeech
GENSCLITE=$SCRIPT_PATH/GenScliteNum.py
SCLITE=$SCRIPT_PATH/sclite

#inputs
IN_MODEL=/media/lumi/alpha/asr_work/model.hdf5
ARK=/media/lumi/alpha/asr_work/test.ark
DATA_SET=test_clean
IN_DIR=$(dirname "${IN_MODEL}")




python ./fmllr_decode.py $IN_MODEL $DATA_SET ./priors.csv  $IN_DIR/out.txt >&  $IN_DIR/log.txt

#python ./fmllr_decode_from_ark.py $IN_MODEL $ARK ./priors.csv  $IN_DIR/out.txt >&  $IN_DIR/log.txt


cat /media/lumi/alpha/kaldi/egs/librispeech/s5/data/$DATA_SET/text | cut -d" " -f2- > $IN_DIR/ref.txt
cat $IN_DIR/out.txt | cut -d" " -f2- > $IN_DIR/hyp.txt
python $GENSCLITE $IN_DIR/ref.txt > $IN_DIR/ref.txt.sclite  
python $GENSCLITE $IN_DIR/hyp.txt > $IN_DIR/hyp.txt.sclite
$SCLITE -r $IN_DIR/ref.txt.sclite -h $IN_DIR/hyp.txt.sclite -i rm -o all  
cat $IN_DIR/hyp.txt.sclite.sys   

 



