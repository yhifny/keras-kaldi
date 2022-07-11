#!/bin/bash
# -------------------------------------------------------

SCORING_PATH=/lfs01/workdirs/hlwn026u2/keras-kaldi/timit_work/score_timit
LOG_DIR=$SCORING_PATH/log_scoring
GENSCLITE=$SCORING_PATH/GenScliteNum.py
SCLITE=$SCORING_PATH/sclite

REF48=$1
HYP48=$2

##
## LOG DIR
##
mkdir -p $LOG_DIR
rm -f $LOG_DIR/*

cat $REF48 | cut -d" " -f2- > $LOG_DIR/ref48.txt
cat $HYP48 | cut -d" " -f2- > $LOG_DIR/hyp48.txt
		
$SCORING_PATH/timit48Totimit39.sh $LOG_DIR/ref48.txt 	$LOG_DIR/ref39.txt 	$SCORING_PATH/phone39.lst		
$SCORING_PATH/timit48Totimit39.sh $LOG_DIR/hyp48.txt 	$LOG_DIR/hyp39.txt	$SCORING_PATH/phone39.lst
		
		

python $GENSCLITE $LOG_DIR/ref39.txt > $LOG_DIR/ref39.txt.sclite  
python $GENSCLITE $LOG_DIR/hyp39.txt > $LOG_DIR/hyp39.txt.sclite
$SCLITE -r $LOG_DIR/ref39.txt.sclite -h $LOG_DIR/hyp39.txt.sclite -i rm -o all  



