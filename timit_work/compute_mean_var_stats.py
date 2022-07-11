#!/usr/bin/env python

import numpy as np
from kaldi.util.table import SequentialMatrixReader
import config_kaldi as config

def compute_mean_var_large_scale_maximum_likelihood(feats_rspecifier_all):

    N=0
    mean_acc =  None
    cov_acc  =  None
    with SequentialMatrixReader(feats_rspecifier_all) as f:
        for i, (key, feats) in enumerate(f):
            print ("processing file: ", i, key, flush=True)            
            feats_data=feats.numpy()
            if i == 0:
                N_FEAT_COEFFS=feats_data.shape[1]
                mean_acc =  np.zeros((1,N_FEAT_COEFFS), dtype='float32')
                cov_acc=  np.zeros((N_FEAT_COEFFS,N_FEAT_COEFFS), dtype='float32')

            mean_acc=mean_acc+np.sum(feats_data, axis=0)
            cov_acc=cov_acc + (feats_data.T).dot(feats_data)
            N=N+feats_data.shape[0]
        
    mean=mean_acc/float(N)	
    cov=cov_acc/float(N) -(mean.T).dot(mean) 
    return mean.flatten(),  np.sqrt(cov.diagonal())                  





mean,std=compute_mean_var_large_scale_maximum_likelihood(config.fmllr_train_feats_rspecifier)
#print (mean, std)
np.savez("mean_std_fmllr", mean=mean, std=std)