. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)


steps/align_fmllr.sh --nj 10 data/train_clean_100 data/lang exp/tri4b exp/tri4b_ali_clean_100
steps/align_fmllr.sh --nj 10 data/dev_clean data/lang exp/tri4b exp/tri4b_ali_dev_clean_100