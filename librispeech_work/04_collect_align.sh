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
ali-to-pdf $MODEL 'ark:gunzip -c /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/ali.*.gz |' 'ark,t:-' \
> /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/ali_tri.txt 

ali-to-phones --per-frame=true $MODEL 'ark:gunzip -c /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/ali.*.gz |' 'ark,t:-' \
> /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/ali_mono.txt 

# dev set
ali-to-pdf $MODEL 'ark:gunzip -c /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/ali.*.gz |' 'ark,t:-' \
> /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/ali_tri.txt 


ali-to-phones --per-frame=true $MODEL 'ark:gunzip -c /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/ali.*.gz |' 'ark,t:-' \
> /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/ali_mono.txt 
