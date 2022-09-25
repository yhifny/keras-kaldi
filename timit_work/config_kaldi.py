#!/usr/bin/env python

from __future__ import print_function
import os
from subprocess import Popen, PIPE, DEVNULL

num_jobs=10
task_dir        = "/lfs01/workdirs/hlwn026u2/keras-kaldi//kaldi/egs/timit/s5"
train_data_dir  = os.path.join (task_dir, 'data', 'train')
dev_data_dir    = os.path.join (task_dir, 'data', 'dev')
test_data_dir   = os.path.join (task_dir, 'data', 'test')
train_trans_dir = os.path.join (task_dir, 'exp', 'tri3_ali')
dev_trans_dir   = os.path.join (task_dir, 'exp', 'tri3', 'decode_dev')
test_trans_dir  = os.path.join (task_dir, 'exp', 'tri3', 'decode_test')
train_align_dir = os.path.join (task_dir, 'exp', 'tri3_ali')
dev_align_dir = os.path.join (task_dir, 'exp', 'tri3_ali_dev')
final_model_dir = os.path.join (task_dir, 'exp', 'tri3')


                   
fmllr_train_feats_rspecifier = "ark,s,cs:apply-cmvn  --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp scp:%s/feats.scp ark:- |" % (train_data_dir,train_data_dir,train_data_dir) +\
                       "splice-feats --left-context=3 --right-context=3 ark:- ark:- |" +\
                       "transform-feats %s/final.mat ark:- ark:- |" % (final_model_dir) +\
                       "transform-feats --utt2spk=ark:%s/utt2spk 'ark:cat %s/trans.* |'  ark:- ark:- |" % (train_data_dir, train_trans_dir)

fmllr_train_feats_rspecifier = "ark,s,cs:apply-cmvn  --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp scp:%s/feats.scp ark:- |" % (train_data_dir,train_data_dir,train_data_dir) +\
                       "splice-feats --left-context=3 --right-context=3 ark:- ark:- |" +\
                       "transform-feats %s/final.mat ark:- ark:- |" % (final_model_dir) +\
                       "transform-feats --utt2spk=ark:%s/utt2spk 'ark:cat %s/trans.* |'  ark:- ark:- |" % (train_data_dir, train_trans_dir)

                    
fmllr_dev_feats_rspecifier = "ark,s,cs:apply-cmvn  --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp scp:%s/feats.scp ark:- |" % (dev_data_dir,dev_data_dir,dev_data_dir) +\
                       "splice-feats --left-context=3 --right-context=3 ark:- ark:- |" +\
                       "transform-feats %s/final.mat ark:- ark:- |" % (final_model_dir) +\
                       "transform-feats --utt2spk=ark:%s/utt2spk 'ark:cat %s/trans.* |'  ark:- ark:- |" % (dev_data_dir, dev_trans_dir)

fmllr_test_feats_rspecifier = "ark,s,cs:apply-cmvn  --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp scp:%s/feats.scp ark:- |" % (test_data_dir,test_data_dir,test_data_dir) +\
                       "splice-feats --left-context=3 --right-context=3 ark:- ark:- |" +\
                       "transform-feats %s/final.mat ark:- ark:- |" % (final_model_dir) +\
                       "transform-feats --utt2spk=ark:%s/utt2spk 'ark:cat %s/trans.* |'  ark:- ark:- |" % (test_data_dir, test_trans_dir)

train_feats_scp = '%s/feats_sorted.scp'%(train_data_dir) 
dev_feats_scp = '%s/feats_sorted.scp'%(dev_data_dir)

#pdf_train_rspecifier = 'ark,s,cs:ali-to-pdf  %s/final.mdl ark:"gunzip -c %s/ali.*.gz |" ark,t:- |' % (final_model_dir, train_align_dir)
#pdf_train_rspecifier = "ark:gunzip -c %s/ali.gz |"% (train_align_dir)
pdf_train_rspecifier = "ark,t:%s/ali_tri.txt"% (train_align_dir) 
mono_train_rspecifier = "ark,t:%s/ali_mono.txt"% (train_align_dir)
mono_targets_train_rspecifier = "ark,t:%s/targets_mono.txt"% (train_align_dir)
pdf_dev_rspecifier = "ark,t:%s/ali_tri.txt"% (dev_align_dir) 
mono_dev_rspecifier = "ark,t:%s/ali_mono.txt"% (dev_align_dir)
mono_targets_dev_rspecifier = "ark,t:%s/targets_mono.txt"% (dev_align_dir)


final_fmllr_train_rspecifier = "ark:%s/train_fmllr.ark"% (os.path.join (task_dir, 'exp'))
final_fmllr_dev_rspecifier = "ark:%s/dev_fmllr.ark"% (os.path.join (task_dir, 'exp'))

graph_file =  (os.path.join (task_dir, 'exp', 'tri3','graph', 'HCLG.fst' ))
words_mapping_file =  (os.path.join (task_dir, 'exp', 'tri3','graph', 'words.txt' ))
final_model =  (os.path.join (task_dir, 'exp', 'tri3', 'final.mdl'))
tree_file =  (os.path.join (task_dir, 'exp', 'tri3', 'tree'))
lexicon_file = (os.path.join (task_dir, 'data', 'lang', 'L.fst'))
train_transcription_file = (os.path.join (task_dir, 'data', 'train', 'text'))
disambig_file= (os.path.join (task_dir, 'data', 'lang', 'phones', 'disambig.int')) 



def get_num_classes ():
    p1 = Popen (['am-info', final_model], stdout=PIPE)
    modelInfo = p1.stdout.read().splitlines()
    for line in modelInfo:
        if b'number of pdfs' in line:
            return int(line.split()[-1])

