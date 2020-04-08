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



SCRIPT_PATH=/home/lumi/Dropbox/ALL/librispeech_work
SCORING_PATH=$SCRIPT_PATH/score_librispeech
GENSCLITE=$SCORING_PATH/GenScliteNum.py
SCLITE=$SCORING_PATH/sclite




#inputs
IN_MODEL=$1
DATA_SET=$2
IN_DIR=$(dirname "${IN_MODEL}")
REF_KALDI=/media/lumi/alpha/kaldi/egs/librispeech/s5/data/$DATA_SET/text
HYP_KALDI=$IN_DIR/out.txt




rm -f ${IN_DIR}/temp_results.txt
rm -f ${IN_DIR}/log.txt


# decoding
python $SCRIPT_PATH/fmllr_decode.py $IN_MODEL $DATA_SET $SCRIPT_PATH/priors.csv  $IN_DIR/out.txt >&  $IN_DIR/log.txt

#scoring
cat $REF_KALDI | cut -d" " -f2- > $IN_DIR/ref.txt
cat $HYP_KALDI | cut -d" " -f2- > $IN_DIR/hyp.txt
python $GENSCLITE $IN_DIR/ref.txt > $IN_DIR/ref.txt.sclite  
python $GENSCLITE $IN_DIR/hyp.txt > $IN_DIR/hyp.txt.sclite
$SCLITE -r $IN_DIR/ref.txt.sclite -h $IN_DIR/hyp.txt.sclite -i rm -o all

# report results
grep Sum $IN_DIR/hyp.txt.sclite.sys | awk '{print $11}' > ${IN_DIR}/temp_results.txt  

 



