export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


MODEL=/media/lumi/alpha/kaldi/egs/timit/s5/exp/tri3/final.mdl
#ali-to-pdf $MODEL 'ark:gunzip -c /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/ali.*.gz |' 'ark:|gzip -c > egs/timit/s5/exp/tri3_ali/ali.gz'

ali-to-pdf $MODEL ark:/media/lumi/alpha/kaldi/egs/timit/s5/exp/tri3_ali/train_align_dnn.trk ark,t:- \
> /media/lumi/alpha/kaldi/egs/timit/s5/exp/tri3_ali/ali_tri_dnn.txt 


ali-to-phones --per-frame=true $MODEL ark:/media/lumi/alpha/kaldi/egs/timit/s5/exp/tri3_ali/train_align_dnn.trk   ark,t:- \
> /media/lumi/alpha/kaldi/egs/timit/s5/exp/tri3_ali/ali_mono_dnn.txt 
