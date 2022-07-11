#!/bin/bash


export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


export CUDA_VISIBLE_DEVICES=0,1

source  activate  tf_gpu

DATA_SET=test

python evaluate_multi_gpu.py \
            --dataset $DATA_SET\
            --gpuids  0,1 \
            --priors  priors.csv \
            --feat_norm_file mean_std_fmllr.npz\
            --models /media/lumi/alpha/asr_work_timit/model.hdf5\
            --ref_file /media/lumi/alpha/kaldi/egs/timit/s5/data/$DATA_SET/text\
            --hyp_output  /media/lumi/alpha/asr_work_timit/out.txt
