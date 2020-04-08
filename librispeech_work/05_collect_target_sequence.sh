export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


MODEL=/media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b/final.alimdl



ali-to-phones  $MODEL 'ark:gunzip -c /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/ali.*.gz |' 'ark,t:-' \
> /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/targets_mono.txt 

ali-to-phones  $MODEL 'ark:gunzip -c /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/ali.*.gz |' 'ark,t:-' \
> /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/targets_mono.txt 

python gen_target_sequence.py /media/lumi/alpha/kaldi/egs/librispeech/s5/data/lang/phones.txt\
            /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/targets_mono.txt \
            /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/targets_mono_one_state.txt
           
python gen_target_sequence.py /media/lumi/alpha/kaldi/egs/librispeech/s5/data/lang/phones.txt\
            /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/targets_mono.txt \
            /media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/targets_mono_one_state.txt
       
           