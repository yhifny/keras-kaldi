#!/bin/bash

##  Decode the DNN model
##  Copyright (C) 2016 D S Pavan Kumar
##  dspavankumar [at] gmail [dot] com
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.

export KALDI_ROOT=/media/lumi/alpha/kaldi
PATH=$PATH:$KALDI_ROOT/tools/openfst
PATH=$PATH:$KALDI_ROOT/src/featbin
PATH=$PATH:$KALDI_ROOT/src/gmmbin
PATH=$PATH:$KALDI_ROOT/src/bin
PATH=$PATH:$KALDI_ROOT//src/nnetbin
export PATH


## Begin configuration section
stage=0
nj=1
cmd=run.pl

max_active=7000 # max-active
beam=15.0 # beam used
latbeam=7.0 # beam used in getting lattices
acwt=0.1 # acoustic weight used in getting lattices

skip_scoring=false # whether to skip WER scoring
scoring_opts=

splice_size=
norm_vars=
add_deltas=

## End configuration section


[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;




data=data/test_clean
graphdir=exp/tri4b/graph_tgsmall
dir=exp/keras # remove any trailing slash.
cmd=run.pl

srcdir=`dirname $dir`; # assume model directory one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
cp exp/tri4b/final.mdl exp
cp exp/tri4b/final.mdl exp/keras

split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data/feats.scp exp/tri4b/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

finalfeats="ark,s,cs: cat /media/lumi/alpha/asr_work/test.ark |"
latgen-faster-mapped --max-active=$max_active --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt exp/tri4b/final.mdl $graphdir/HCLG.fst "$finalfeats" "ark:|gzip -c > $dir/lat.1.gz"

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
fi


exit 0;