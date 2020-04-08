. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)


mkdir -p exp/tri4b/decode_tgsmall_train_clean_100 && cp exp/tri4b/trans.* exp/tri4b/decode_tgsmall_train_clean_100/

gmmdir=exp/tri4b

for chunk in train_clean_100 dev_clean test_clean; do
    dir=fmllr/$chunk
    steps/nnet/make_fmllr_feats.sh --nj 1 --cmd "$train_cmd" \
        --transform-dir $gmmdir/decode_tgsmall_$chunk \
            $dir data/$chunk $gmmdir $dir/log $dir/data || exit 1

    compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
done

#copy-feats 'ark,s,cs:apply-cmvn  --utt2spk=ark:utt2spk  ark:data/cmvn_speaker.ark scp:feats.scp ark:- |' ark,t:-
#copy-feats 'ark,s,cs:apply-cmvn  --utt2spk=ark:utt2spk  ark:data/cmvn_speaker.ark scp:feats.scp ark:- |' ark:train.ark