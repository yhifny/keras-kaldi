#!/bin/sh
#######SBATCH --mem=100G
#SBATCH -J timit_ja  #train_model
#SBATCH -o output_timit_jasper.txt #standard output file
#SBATCH -e errors_timit_jasper.txt #standard error file
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --time=240:00:00

export KALDI_ROOT=/lfs01/workdirs/hlwn026u2/keras-kaldi/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lfs01/workdirs/hlwn026u2/keras-kaldi/conda_envs/conda_env/lib

export CUDA_VISIBLE_DEVICES=0,1

#module load Anaconda3
#source activate /lfs01/workdirs/hlwn026u2/keras-kaldi/conda_envs/conda_env

rm -r /lfs01/workdirs/hlwn026u2/keras-kaldi/timit_work//asr_work_timit_jasper
mkdir -p /lfs01/workdirs/hlwn026u2/keras-kaldi/timit_work//asr_work_timit_jasper

# config_train_transformer
python train_conformer.py config_train/config_train_jasper.yaml| tee /lfs01/workdirs/hlwn026u2/keras-kaldi/timit_work//asr_work_timit_jasper/my_log.txt 
