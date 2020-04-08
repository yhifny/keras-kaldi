#!/usr/bin/env python

from __future__ import print_function
import os
from subprocess import Popen, PIPE, DEVNULL

task_dir        = '/media/lumi/alpha/kaldi/egs/librispeech/s5'
train_fmllr_dir = '/media/lumi/alpha/kaldi/egs/librispeech/s5/fmllr/train_clean_100'
dev_fmllr_dir   = '/media/lumi/alpha/kaldi/egs/librispeech/s5/fmllr/dev_clean'
test_fmllr_dir  = '/media/lumi/alpha/kaldi/egs/librispeech/s5/fmllr/test_clean'
train_align_dir = '/media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_clean_100/'
dev_align_dir   = '/media/lumi/alpha/kaldi/egs/librispeech/s5/exp/tri4b_ali_dev_clean_100/'
final_model_dir = os.path.join (task_dir, 'exp', 'tri4b')

train_feats_scp         = '%s/feats_sorted.scp'%(train_fmllr_dir)
dev_feats_scp           = '%s/feats_sorted.scp'%(dev_fmllr_dir) 
pdf_train_rspecifier    = "ark,t:%s/ali_tri.txt"% (train_align_dir)
pdf_dev_rspecifier      = "ark,t:%s/ali_tri.txt"% (dev_align_dir)
mono_train_rspecifier   = "ark,t:%s/ali_mono.txt"% (train_align_dir)
mono_dev_rspecifier     = "ark,t:%s/ali_mono.txt"% (dev_align_dir)

mono_targets_train_rspecifier = "ark,t:%s/targets_mono.txt"% (train_align_dir)
mono_targets_dev_rspecifier   = "ark,t:%s/targets_mono.txt"% (dev_align_dir)

final_fmllr_train_rspecifier = "ark:%s/feats_fmllr_train_clean_100.1.ark"% (os.path.join (train_fmllr_dir, 'data'))

fmllr_train_feats_rspecifier = final_fmllr_train_rspecifier
fmllr_dev_feats_rspecifier   = "ark:%s/feats_fmllr_dev_clean.1.ark"% (os.path.join (dev_fmllr_dir, 'data'))
final_fmllr_dev_rspecifier   = fmllr_dev_feats_rspecifier
fmllr_test_feats_rspecifier  = "ark:%s/feats_fmllr_test_clean.1.ark"% (os.path.join (test_fmllr_dir, 'data'))

graph_file =  (os.path.join (task_dir, 'exp', 'tri4b','graph_tgsmall', 'HCLG.fst' ))
words_mapping_file =  (os.path.join (task_dir, 'exp', 'tri4b','graph_tgsmall', 'words.txt' ))
final_model =  (os.path.join (task_dir, 'exp', 'tri4b', 'final.mdl'))
tree_file =  (os.path.join (task_dir, 'exp', 'tri4b', 'tree'))
lexicon_file = (os.path.join (task_dir, 'data', 'lang', 'L.fst'))
train_transcription_file = (os.path.join (task_dir, 'data', 'train_clean_100', 'text'))
disambig_file= (os.path.join (task_dir, 'data', 'lang', 'phones', 'disambig.int')) 
words_mapping_file2 =  (os.path.join (task_dir, 'data', 'lang', 'words.txt' ))




def get_num_classes ():
    p1 = Popen (['am-info', final_model], stdout=PIPE)
    modelInfo = p1.stdout.read().splitlines()
    for line in modelInfo:
        if b'number of pdfs' in line:
            return int(line.split()[-1])

