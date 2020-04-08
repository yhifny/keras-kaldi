#
# Author: yasser hifny
#
import sys
from kaldi.util.table import SequentialIntVectorReader, IntVectorWriter


def gen_one_state_targets(target_vec, phone_table):
    
    target_vec_by_name =[phone_table[i] for  i in target_vec]
    print (target_vec_by_name,flush=True)

    return target_vec

phone_table = {}
for i in open(sys.argv[1]).readlines():
    phone_table [int(i.split()[1])] = i.split()[0]

target_rspecifier = "ark:"+ sys.argv[2]

one_state_target_wspecifier="ark:"+ sys.argv[3]
target_writer = IntVectorWriter(one_state_target_wspecifier)


with SequentialIntVectorReader(target_rspecifier) as f:
    for (fkey, target_vec)   in f:
        print ('processing: ', fkey, "target len:", len(target_vec), flush=True)        	
        target_writer[fkey] = gen_one_state_targets(target_vec, phone_table)

target_writer.close()
        