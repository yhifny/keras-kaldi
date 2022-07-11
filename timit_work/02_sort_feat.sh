export KALDI_ROOT=/lfs01/workdirs/hlwn026u2/keras-kaldi/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


DATA_DIR=/home/hlwn026u2/data/keras-kaldi//kaldi/egs/timit/s5/data/train
MIN_LEN=0

feat-to-len scp:$DATA_DIR/feats.scp ark,t:- | awk '{print $2}' > len.tmp || exit 1;
paste -d " " $DATA_DIR/feats.scp len.tmp | sort -k3 -n - |
    awk -v m=$MIN_LEN '{ if ($3 >= m) {print $1 " " $2} }' > $DATA_DIR/feats_sorted.scp || exit 1;
    
rm  len.tmp   

DATA_DIR=/home/hlwn026u2/data/keras-kaldi//kaldi/egs/timit/s5/data/dev
MIN_LEN=0

feat-to-len scp:$DATA_DIR/feats.scp ark,t:- | awk '{print $2}' > len.tmp || exit 1;
paste -d " " $DATA_DIR/feats.scp len.tmp | sort -k3 -n - |
    awk -v m=$MIN_LEN '{ if ($3 >= m) {print $1 " " $2} }' > $DATA_DIR/feats_sorted.scp || exit 1;
    
rm  len.tmp  