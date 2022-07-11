#!/usr/bin/env python

from __future__ import print_function

from kaldi.alignment import GmmAligner
from kaldi.fstext import SymbolTable
from kaldi.lat.align import WordBoundaryInfoNewOpts, WordBoundaryInfo
from kaldi.util.table import SequentialMatrixReader

# Construct aligner
aligner = GmmAligner.from_files("gmm-boost-silence --boost=1.0 1 /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3/final.mdl - |",
                                "/mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/tree", "/mnt/c/yasser/kaldi/egs/timit/s5/data/lang/L.fst", "/mnt/c/yasser/kaldi/egs/timit/s5/data/lang/words.txt", "/mnt/c/yasser/kaldi/egs/timit/s5/data/lang/phones/disambig.int",
                                self_loop_scale=0.1, beam=10)
phones = SymbolTable.read_text("/mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/phones.txt")
#wb_info = WordBoundaryInfo.from_file(WordBoundaryInfoNewOpts(),
#                                     "/mnt/c/yasser/kaldi/egs/timit/s5/data/lang/phones/wdisambig_words.int")

# Define feature pipeline as a Kaldi rspecifier
feats_rspecifier= ( "ark,s,cs:apply-cmvn  --utt2spk=ark:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/utt2spk scp:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/cmvn.scp scp:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/feats.scp ark:- |" 
                      "splice-feats --left-context=3 --right-context=3 ark:- ark:- |"
                      "transform-feats /mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3/final.mat ark:- ark:- |" 
                      "transform-feats --utt2spk=ark:/mnt/c/yasser/kaldi/egs/timit/s5/data/train/split3/1/utt2spk ark:/mnt/c/yasser/kaldi/egs/timit/s5/exp/tri3_ali/trans.1 ark:- ark:- |"
                  )

# Align
c=0
with SequentialMatrixReader(feats_rspecifier) as f, open("/mnt/c/yasser/kaldi/egs/timit/s5/data/train/text") as t:
    for (fkey, feats), line in zip(f, t):
        tkey, text = line.strip().split(None, 1)
        assert(fkey == tkey)
        out = aligner.align(feats, text)
        print(fkey, out["alignment"], flush=True)
        print (len(out["alignment"]))
        phone_alignment = aligner.to_phone_alignment(out["alignment"], phones)
        print(fkey, phone_alignment, flush=True)
        #word_alignment = aligner.to_word_alignment(out["best_path"], wb_info)
        #print(fkey, word_alignment, flush=True)
        c=c+1
        if c==2:
            break;