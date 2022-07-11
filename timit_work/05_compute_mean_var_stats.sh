export KALDI_ROOT=/lfs01/workdirs/hlwn026u2/keras-kaldi/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH

#LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KALDI_ROOT/src/base:$KALDI_ROOT/src/hmm:$KALDI_ROOT/src/lat:$KALDI_ROOT/src/util:$KALDI_ROOT/src/feat:$KALDI_ROOT/src/nnet3:$KALDI_ROOT/src/matrix:$KALDI_ROOT/src/cudamatrix:$KALDI_ROOT/src/rnnlm:$KALDI_ROOT/tools/openfst/lib:/lfs01/workdirs/hlwn026u2/keras-kaldi/conda_envs/conda_env/lib:/lib64
#export LD_LIBRARY_PATH


python compute_mean_var_stats.py