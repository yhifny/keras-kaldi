#!/usr/bin/env python
# see https://github.com/dspavankumar/keras-kaldi/blob/master/steps_kt/compute_priors.py
import numpy as np
from kaldi.util.table import SequentialMatrixReader
import config_kaldi as config
import sys
import numpy
from subprocess import Popen, PIPE



## Compute priors
def compute_priors ():
    dim = config.get_num_classes()    
    counts = numpy.zeros(dim)

    ## Prepare string
    ali_str = 'ark:gunzip -c ' + config.train_align_dir+'/ali.*.gz ' +  '|'
    p = Popen(['ali-to-pdf', config.final_model, ali_str, 'ark,t:-'], stdout=PIPE)
    print (p.stdout)
    ## Compute counts
    for line in p.stdout:
        line = line.split()
        for index in line[1:]:
            counts[int(index)] += 1

    ## Compute priors
    priors = counts / numpy.sum(counts)

    ## Floor zero values
    priors[priors==0] = 1e-5

    ## Write to file
    priors.tofile ('priors.csv', sep=',', format='%e')


compute_priors()