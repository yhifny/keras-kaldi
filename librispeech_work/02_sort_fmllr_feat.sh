export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH

### train set
TRAIN_DATA_DIR=/media/lumi/alpha/kaldi/egs/librispeech/s5/fmllr/train_clean_100
MIN_LEN=0

feat-to-len scp:$TRAIN_DATA_DIR/feats.scp ark,t:- | awk '{print $2}' > len.tmp || exit 1;
paste -d " " $TRAIN_DATA_DIR/feats.scp len.tmp | sort -k3 -n - |
    awk -v m=$MIN_LEN '{ if ($3 >= m) {print $1 " " $2} }' > $TRAIN_DATA_DIR/feats_sorted.scp || exit 1;
    
rm  len.tmp

### dev set
DEV_DATA_DIR=/media/lumi/alpha/kaldi/egs/librispeech/s5/fmllr/dev_clean
MIN_LEN=0

feat-to-len scp:$DEV_DATA_DIR/feats.scp ark,t:- | awk '{print $2}' > len.tmp || exit 1;
paste -d " " $DEV_DATA_DIR/feats.scp len.tmp | sort -k3 -n - |
    awk -v m=$MIN_LEN '{ if ($3 >= m) {print $1 " " $2} }' > $DEV_DATA_DIR/feats_sorted.scp || exit 1;
    
rm  len.tmp   