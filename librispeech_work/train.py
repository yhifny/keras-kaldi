#
# Author: yasser hifny
#
import math
import numpy as np
np.random.seed(1337)  # for reproducibility
from numpy import newaxis
import sys
import codecs
from keras.preprocessing import sequence
from keras.utils import np_utils 
from keras.layers import Activation, Embedding , Dense, Input,LSTM,GlobalAveragePooling1D,GlobalMaxPooling1D
from keras.layers import  SpatialDropout1D,Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.optimizers import RMSprop, Adam, Nadam
from keras.layers import Conv1D, MaxPooling1D
from keras.engine.topology import Layer
from keras import initializers
from keras.preprocessing import sequence
from keras.utils import np_utils 
import keras
from keras import optimizers
from keras.models import load_model
#sys.setdefaultencoding('utf-8')
import glob
import os
from keras.utils import multi_gpu_model
#from tensorflow.python.client import device_lib
#import tensorflow as tf
from keras.callbacks import Callback
import warnings
#import tensorflow as tf
import random
random.seed(9001)
import yaml
import model as model_definition
from kaldi.util.table import RandomAccessMatrixReader, SequentialMatrixReader,RandomAccessIntVectorReader, SequentialIntVectorReader
import config_kaldi
from my_layers import ContextExpansion


from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
#from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader
from data_generator import DataGenerator, DataGeneratorSeq2Seq 

def read_data( mona_align_file,
                 tri_align_file,
                 mono_targets_file,
                 feat_ids_file,
                 feat_file):
        
    output_dic= {}    
    
    # read  data
    output_dic['tri_align_ark'] = RandomAccessIntVectorReader(tri_align_file)
    output_dic['mono_align_ark'] = RandomAccessIntVectorReader(mona_align_file)
    output_dic['mono_targets_ark'] = RandomAccessIntVectorReader(mono_targets_file)

    feat_ids  = [i.split()[0] for i in open(feat_ids_file, 'r').readlines()]
    feat_ids_filtered=[]
    for id in feat_ids:
        try:
            temp = output_dic['tri_align_ark'][id]
            feat_ids_filtered.append(id)
        except:
            pass
    print ("%d survived from %d files" % (len(feat_ids_filtered), len(feat_ids) ))        
    output_dic['feat_ids']      =  feat_ids_filtered   
    output_dic['feats_ark']     = RandomAccessMatrixReader(feat_file)
    output_dic['sample_weight'] = [1]*len(feat_ids_filtered)
    return output_dic
    
    
    


def read_wer(file):
	if os.path.exists(file):
		wer=float(open(file, 'r').readline())
	else:
		wer=1000000.0
	return wer
	
def read_epoch(file):
	if os.path.exists(file):
		epoch=int(open(file, 'r').readline())
	else:
		epoch=0
	return epoch
	
def read_lr(file, default_lr):
	if os.path.exists(file):
		lr=float(open(file, 'r').readline())
	else:
		lr=default_lr
	return lr
    
def	write_line(file,line):
	f = open(file, 'w')
	f.write(line)    



# getting the number of GPUs 
# def get_available_gpus():
   # local_device_protos = device_lib.list_local_devices()
   # return [x.name for x in local_device_protos if x.device_type    == 'GPU']
# num_gpu = len(get_available_gpus())
# print "GPU count: ", num_gpu

   

with open(sys.argv[1]) as stream:
    try:
        cfg=yaml.safe_load(stream)
        print(cfg)
    except yaml.YAMLError as exc:
        print(exc)


seq_length = 1000000
num_classes = cfg['model']['outputs']['output_tri']['layers'][0]['units']
num_mono_classes = cfg['model']['outputs']['output_mono']['layers'][0]['units'] 
batch_size=cfg['optimizer']['batch_size']
learning_rate= cfg['optimizer']['lr']
dev_batch_size = cfg['optimizer']['ngpu']
loss_weights = cfg['loss_weights']
priors = np.genfromtxt('priors.csv', delimiter=',')
#dev_ref_list= open("/media/lumi/alpha/kaldi/egs/librispeech/s5/data/dev_clean/text").readlines()
#test_ref_list= open("/media/lumi/alpha/kaldi/egs/librispeech/s5/data/test_clean/text").readlines()    

# create output dir
out_path=cfg['out_path']
if not os.path.exists(out_path):
    os.makedirs(out_path)


train_data_dic = read_data(config_kaldi.mono_train_rspecifier,
                    config_kaldi.pdf_train_rspecifier,
                    config_kaldi.mono_targets_train_rspecifier,
                    config_kaldi.train_feats_scp,
                    config_kaldi.final_fmllr_train_rspecifier)

dev_data_dic = read_data(config_kaldi.mono_dev_rspecifier,
                    config_kaldi.pdf_dev_rspecifier,
                    config_kaldi.mono_targets_dev_rspecifier,
                    config_kaldi.dev_feats_scp,
                    config_kaldi.final_fmllr_dev_rspecifier)



mean_var_file="mean_std_fmllr.npz"



# build data generator
if 'seq2seq-encoder-decoder' in cfg['model']:
    training_generator = DataGeneratorSeq2Seq( train_data_dic,
                                        mean_var_file, 
                                        num_classes,
                                        num_mono_classes,
                                        min_length=0, 
                                        max_length=seq_length,
                                        batch_size=batch_size,
                                        shuffle=False)
    dev_generator = DataGeneratorSeq2Seq( dev_data_dic,
                                    mean_var_file, 
                                    num_classes,
                                    num_mono_classes,
                                    min_length=0, 
                                    max_length=seq_length,
                                    batch_size=dev_batch_size,
                                    shuffle=False)                                    
                                        
else:                                        
    training_generator = DataGenerator( train_data_dic,
                                        mean_var_file, 
                                        num_classes,
                                        num_mono_classes,
                                        min_length=0, 
                                        max_length=seq_length,
                                        batch_size=batch_size,
                                        shuffle=False)
    dev_generator = DataGenerator( dev_data_dic,
                                    mean_var_file, 
                                    num_classes,
                                    num_mono_classes,
                                    min_length=0, 
                                    max_length=seq_length,
                                    batch_size=dev_batch_size,
                                    shuffle=False)
                                    
                            
                                    

model,parallel_model  = model_definition.create_model(cfg)
sgd = model_definition.create_optimizer(cfg)
parallel_model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                                      metrics=['accuracy'],
                                      loss_weights=loss_weights)
print(model.summary())
#print(parallel_model.summary())



###############
# train model #
###############
print("model fitting...")

						
## Scale learning rate after each epoch
lrHalvingFactor=0.5
start_learning_rate=read_lr(out_path+"/learning_rate.txt", learning_rate)
print ("start_learning_rate", start_learning_rate)
lrScaleCount=20-int(math.log(learning_rate/start_learning_rate,2))
learning_rate=start_learning_rate
print ("lrScaleCount ", lrScaleCount)


val_wer=read_wer(out_path+"/wer.txt")
patienceCount=3
patience=0

equal_patienceCount=2
equal_patience=0

epoch=read_epoch(out_path+"/epoch.txt")
if epoch >0:
    print('recover the training process')
    model = load_model(out_path+'/model.hdf5', custom_objects={'GatedConvBlock':GatedConvBlock, 'NovoGrad': NovoGrad, 'ContextExpansion' : ContextExpansion})	

## newbob strategy
for epoch in range(1,25):

    K.set_value(parallel_model.optimizer.lr, learning_rate)
    history = parallel_model.fit_generator(generator=training_generator,
                                    validation_data=dev_generator,    
                                    epochs=1,
                                    workers=1, use_multiprocessing=False)
    new_val_wer= (1.0 - history.history['val_output_tri_acc'][0])*100.0
    print ('Epoch %d learning rate: %f dev_ferr %f' % (epoch,learning_rate,new_val_wer))

    if ((val_wer-new_val_wer)/new_val_wer)<0.001:
        learning_rate *= lrHalvingFactor
    else:     
        val_wer = new_val_wer
        model.save (out_path+'/model.hdf5', overwrite=True)




