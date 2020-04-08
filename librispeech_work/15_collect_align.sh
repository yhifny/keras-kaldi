export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


MODEL=/media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b/final.alimdl
#ali-to-pdf $MODEL 'ark:gunzip -c /mnt/c/yasser/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/ali.*.gz |' 'ark:|gzip -c > egs/librispeech/s5/exp/tri4b_ali_clean_100/ali.gz'

# train set
ali-to-pdf $MODEL ark:/media/lumi/alpha/asr_work_cnn_gru_seq/train_trans_id_dnn.trk 'ark,t:-' \
> /media/lumi/alpha/asr_work_cnn_gru_seq/ali_tri.txt 

