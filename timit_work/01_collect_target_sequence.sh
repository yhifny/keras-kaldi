export KALDI_ROOT=/lfs01/workdirs/hlwn026u2/keras-kaldi/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


MODEL=/home/hlwn026u2/data/keras-kaldi//kaldi/egs/timit/s5/exp/tri3/final.mdl
#ali-to-pdf $MODEL 'ark:gunzip -c /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/ali.*.gz |' 'ark:|gzip -c > egs/timit/s5/exp/tri3_ali/ali.gz'



ali-to-phones  $MODEL 'ark:gunzip -c /home/hlwn026u2/data/keras-kaldi//kaldi/egs/timit/s5/exp/tri3_ali/ali.*.gz |' 'ark,t:-' \
> /home/hlwn026u2/data/keras-kaldi//kaldi/egs/timit/s5/exp/tri3_ali/targets_mono.txt 


ali-to-phones  $MODEL 'ark:gunzip -c /home/hlwn026u2/data/keras-kaldi//kaldi/egs/timit/s5/exp/tri3_ali_dev/ali.*.gz |' 'ark,t:-' \
> /home/hlwn026u2/data/keras-kaldi//kaldi/egs/timit/s5/exp/tri3_ali_dev/targets_mono.txt 