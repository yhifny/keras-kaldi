#!/usr/bin/env python

from __future__ import print_function
import re
import sys
from collections import OrderedDict
from kaldi.alignment import GmmAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.util.table import RandomAccessMatrixReader, SequentialMatrixReader,RandomAccessIntVectorReader, SequentialIntVectorReader


feat_ids_file="/mnt/c/yasser/kaldi/egs/timit/s5/data/train/feats.scp"
feat_ids=OrderedDict((i.split()[0],int(re.search('\.(.*?)\.', i.split()[1]).group(1))) 
                for i in open(feat_ids_file, 'r').readlines())
#print (feat_ids.keys())

# with SequentialMatrixReader(scp_rspecifier) as f:
    # for i in f:
        # print(i)
        # break




feats_rspecifier= ( "ark,s,cs:apply-cmvn  --utt2spk=ark:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/utt2spk scp:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/cmvn.scp scp:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/feats.scp ark:- |" 
                      "splice-feats --left-context=3 --right-context=3 ark:- ark:- |"
                      "transform-feats /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3/final.mat ark:- ark:- |" 
                      "transform-feats --utt2spk=ark:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/utt2spk ark:/mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/trans.1 ark:- ark:- |"
                  )
                  
feats_rspecifier_all= ( "ark,s,cs:apply-cmvn  --utt2spk=ark:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/utt2spk scp:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/cmvn.scp scp:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/feats.scp ark:- |" 
                      "splice-feats --left-context=3 --right-context=3 ark:- ark:- |"
                      "transform-feats /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3/final.mat ark:- ark:- |" 
                      "transform-feats --utt2spk=ark:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/utt2spk ark:/mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/trans.* ark:- ark:- |"
                  )                 

with SeuentialMatrixReader(feats_rspecifier) as f:
    for key, feats in f
        print (len(f['mbjv0_sx77']))
    #for  i in f:
    #    print(i)
        #break

alignment_rspecifier="ark,s,cs:gunzip -c /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/ali.1.gz |"
with RandomAccessIntVectorReader(alignment_rspecifier) as f:
    print (len(f['faem0_si2022']))
    print (f['faem0_si2022'])
  
sys.exit()  
    # f.close()

state_rspecifier="ark,s,cs:ali-to-pdf  /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3/final.alimdl 'ark:gunzip -c /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/ali.1.gz |' ark,t:- |"
w=RandomAccessIntVectorReader(state_rspecifier)
print (len(w['mbjv0_sx77']))
print (w['mbjv0_sx77'])

state_rspecifier="ark,s,cs:ali-to-pdf  /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3/final.mdl 'ark:gunzip -c /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/ali.1.gz |' ark,t:- |"
w=RandomAccessIntVectorReader(state_rspecifier)
print (len(w['mbjv0_sx77']))
print (w['mbjv0_sx77'])


#print (len(w['mbjv0_sx77']))
#print (w['mbjv0_sx77'])


# 
# print (len(w[feat_ids[0]]))
# w.close()

# print("======================================")