#!/bin/bash

#!/bin/sh
#SBATCH --mem=120G
#SBATCH -J timit  #train_model
#SBATCH -o output_timit_decode.txt #standard output file
#SBATCH -e errors_timit_decode.txt #standard error file
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
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

export CUDA_VISIBLE_DEVICES=0

module load Anaconda3
source activate /lfs01/workdirs/hlwn026u2/keras-kaldi/conda_envs/conda_env
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lfs01/workdirs/hlwn026u2/keras-kaldi/conda_envs/conda_env/lib



SCRIPT_PATH=./


#inputs
IN_MODEL=./asr_work_timit/model.hdf5
DATA_SET=test
IN_DIR=$(dirname "${IN_MODEL}")



rm -f ${IN_DIR}/temp_results.txt
rm -f ${IN_DIR}/log.txt

python $SCRIPT_PATH/fmllr_decode.py $IN_MODEL $DATA_SET $SCRIPT_PATH/priors.csv  $IN_DIR/out.txt >&  $IN_DIR/log.txt

$SCRIPT_PATH/score_timit/score.sh /lfs01/workdirs/hlwn026u2/keras-kaldi/kaldi/egs/timit/s5/data/$DATA_SET/text  $IN_DIR/out.txt >>  $IN_DIR/log.txt

cat ./score_timit/log_scoring/hyp39.txt.sclite.sys   

 



