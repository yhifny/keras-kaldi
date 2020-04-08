export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


copy-feats "ark,s,cs:apply-cmvn  --utt2spk=ark:/media/lumi/alpha/kaldi/egs/timit/s5/data/train/utt2spk scp:/media/lumi/alpha/kaldi/egs/timit/s5/data/train/cmvn.scp scp:/media/lumi/alpha/kaldi/egs/timit/s5/data/train/feats.scp ark:- |splice-feats --left-context=3 --right-context=3 ark:- ark:- |transform-feats /media/lumi/alpha/kaldi/egs/timit/s5/exp/tri3/final.mat ark:- ark:- |transform-feats --utt2spk=ark:/media/lumi/alpha/kaldi/egs/timit/s5/data/train/utt2spk 'ark:cat /media/lumi/alpha/kaldi/egs/timit/s5/exp/tri3_ali/trans.* |'  ark:- ark:- |"  ark:/media/lumi/alpha/kaldi/egs/timit/s5/exp/train_fmllr.ark



#'ark,s,cs:apply-cmvn --utt2spk=ark:data/test/split10/7/utt2spk scp:data/test/split10/7/cmvn.scp scp:data/test/split10/7/feats.scp ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats exp/tri3b/final.mat ark:- ark:- | transform-feats --utt2spk=ark:data/test/split10/7/utt2spk "ark:cat exp/tri3b/decode/trans.* |" ark:- ark:- |' ark,scp:/Users/enzyme156/desktop/KALDI/data-fmllr-tri3b/test/data/feats_fmllr_test.7.ark,/Users/enzyme156/desktop/KALDI/data-fmllr-tri3b/test/data/feats_fmllr_test.7.scp   