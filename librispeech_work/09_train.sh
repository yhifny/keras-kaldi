export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH

export CUDA_VISIBLE_DEVICES=0,1

source  activate  tf_gpu

rm -rf /media/lumi/alpha/asr_work
mkdir -p /media/lumi/alpha/asr_work
python train.py config_train/config_train_cnn_bn_big_gated.yaml | tee /media/lumi/alpha/asr_work/my_log.txt 
