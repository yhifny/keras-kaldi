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
from wer import wers_timit39 

from kaldi.asr import MappedLatticeFasterRecognizer
from kaldi.decoder import LatticeFasterDecoderOptions
#from kaldi.itf import DecodableInterface
from kaldi.matrix import Matrix
from kaldi.util.table import SequentialMatrixReader

def normalize( feature,  feats_mean, feats_std, eps=1e-14):
    return (feature - feats_mean) / (feats_std + eps)

def load_decoder():
    # Construct recognizer
    decoder_opts = LatticeFasterDecoderOptions()
    decoder_opts.beam = 13
    decoder_opts.max_active = 7000
    asr = MappedLatticeFasterRecognizer.from_files(
        config_kaldi.final_model, config_kaldi.graph_file, config_kaldi.words_mapping_file,
        acoustic_scale=1.0, decoder_opts=decoder_opts)
    return asr 
    
def to_categorical(y, num_classes, dtype='int32'):
    
    onehot_encoded = []
    for value in y:
        column = [0 for _ in range(num_classes)]
        if value>=0: 
            column[value] = 1 
        onehot_encoded.append(column)

    return  np.array(onehot_encoded , dtype=dtype)


def eval_dev(kalid_model, asr, priors, feat_norm_file, datset, ref_list):
    #kalid_model.summary()
    model = Model(inputs=[kalid_model.get_input_at(0)], outputs=[kalid_model.get_layer('output_tri').output])

    
    feats_mean = np.load(feat_norm_file)['mean']
    feats_std = np.load(feat_norm_file)['std']

    feats_rspecifier = config_kaldi.fmllr_dev_feats_rspecifier
    if datset == 'test': feats_rspecifier = config_kaldi.fmllr_test_feats_rspecifier
    
    hyp_list =[]
    with SequentialMatrixReader(feats_rspecifier) as f:
        for (fkey, feats)   in f:
            print ('processing: ', fkey, flush=True)        	
            feats=normalize(feats.numpy(), feats_mean, feats_std)[newaxis,...]
            loglikes = np.log (model.predict(feats)[0,:,:] / priors)
            loglikes [loglikes == -np.inf] = -100        
            out = asr.decode(Matrix(loglikes))
            hyp_list.append("%s %s" %(fkey, out["text"]))

    
    return wers_timit39(ref_list, hyp_list)[1]* 100.0

def eval_dev_old(model_file,model, out_path):
    os.system('./test_eval_fmllr.sh '+model_file + ' dev' )
    f1=float(open((os.path.dirname(model_file)+ "/temp_results.txt"), 'r').readline())
    return f1


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

class MultiGPUCheckpointCallback(Callback):

    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)


# getting the number of GPUs 
# def get_available_gpus():
   # local_device_protos = device_lib.list_local_devices()
   # return [x.name for x in local_device_protos if x.device_type    == 'GPU']
# num_gpu = len(get_available_gpus())
# print "GPU count: ", num_gpu


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, feat_ids, feats_ark, tri_align_ark,mono_align_ark, sample_weight,feat_norm_file,
                 n_classes,n_mono_classes, batch_size=32, min_length=0, max_length=20000,
                 shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.list_files = feat_ids
        self.feats_ark  = feats_ark
        self.align_ark  = tri_align_ark
        self.mono_align_ark  = mono_align_ark        
        self.sample_weight  = sample_weight
        self.n_classes = n_classes
        self.n_mono_classes = n_mono_classes        
        self.shuffle = shuffle
        self.max_length = max_length
        self.feats_mean = np.load(feat_norm_file)['mean']
        self.feats_std = np.load(feat_norm_file)['std']
        self.on_epoch_end()
        print ("generator is based on ", len(self.list_files),"files")
		

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find data for batch

        batch_data  = [self.normalize(self.feats_ark[self.list_files[k]]) for k in indexes]
        batch_label = [self.align_ark[self.list_files[k]] for k in indexes]
        batch_label_mono = [self.mono_align_ark[self.list_files[k]] for k in indexes]        
        W = [self.sample_weight[k] for k in indexes]
        # Generate data
        X, Y, Y_mono  = self.__data_generation(batch_data,batch_label,batch_label_mono)
		
        if sum(W)==len(W):		
            return X, [Y, Y_mono]
        else:
            return X, [Y, Y_mono],np.asarray(W)
		

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)
		
    def __data_generation(self, batch_data,batch_label,batch_label_mono):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        max_length = min(self.max_length,max([ len(f) for f in batch_data]))
        #max_length = 1553
        X = sequence.pad_sequences( batch_data,
										maxlen=max_length,
										dtype='float32',
                                        padding='post')
                                        
        Y = sequence.pad_sequences( batch_label,
                                        maxlen=max_length,
                                        dtype='int32',
                                        padding='post',
                                        value = -1)

        Y =  np.array([to_categorical(seq, num_classes=self.n_classes)
                            for seq in Y ])
                                        
        Y_mono = sequence.pad_sequences( batch_label_mono,
                                        maxlen=max_length,
                                        dtype='int32',
                                        padding='post',
                                        value = -1)

        Y_mono =  np.array([to_categorical(seq, num_classes=self.n_mono_classes) 
                                for seq in Y_mono])
                                      
        return X, Y, Y_mono
	
class DataGeneratorSeq2Seq(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, feat_ids, feats_ark, tri_align_ark,mono_align_ark,
                    mono_targets_ark,
                    sample_weight,feat_norm_file,
                    n_classes,n_mono_classes, batch_size=32, min_length=0, max_length=20000,
                    shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.list_files = feat_ids
        self.feats_ark  = feats_ark
        self.align_ark  = tri_align_ark
        self.mono_align_ark  = mono_align_ark
        self.mono_targets_ark  = mono_targets_ark        
        self.sample_weight  = sample_weight
        self.n_classes = n_classes
        self.n_mono_classes = n_mono_classes        
        self.shuffle = shuffle
        self.max_length = max_length
        self.feats_mean = np.load(feat_norm_file)['mean']
        self.feats_std = np.load(feat_norm_file)['std']
        self.on_epoch_end()
        print ("generator is based on ", len(self.list_files),"files")
		

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_files) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find data for batch
        batch_data  = [self.normalize(self.feats_ark[self.list_files[k]]) for k in indexes]
        batch_label = [self.align_ark[self.list_files[k]] for k in indexes]
        batch_label_mono = [self.mono_align_ark[self.list_files[k]] for k in indexes]        
        # seq2seq batch data
        start_token_index=self.n_mono_classes
        end_token_index=self.n_mono_classes+1
        batch_dec_inputs_mono = [[start_token_index]+self.mono_targets_ark[self.list_files[k]] for k in indexes]        
        batch_dec_targets_mono = [self.mono_targets_ark[self.list_files[k]] + [end_token_index] for k in indexes]        
        
        W = [self.sample_weight[k] for k in indexes]
        
        # Generate data
        X,X_dec, Y, Y_mono,Y_dec  = self.__data_generation(batch_data,
                                        batch_label,
                                        batch_label_mono,
                                        batch_dec_inputs_mono,
                                        batch_dec_targets_mono)
		
        if sum(W)==len(W):		
            return [X,X_dec], [Y, Y_mono,Y_dec]
        else:
            return [X,X_dec], [Y, Y_mono,Y_dec],np.asarray(W)
		

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)
		
    def __data_generation(self, batch_data,batch_label,batch_label_mono,
                            batch_dec_inputs_mono,
                            batch_dec_targets_mono):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        max_length = min(self.max_length,max([ len(f) for f in batch_data]))
        X = sequence.pad_sequences( batch_data,
										maxlen=max_length,
										dtype='float32',
                                        padding='post')
                                        
        Y = sequence.pad_sequences( batch_label,
                                        maxlen=max_length,
                                        dtype='int32',
                                        padding='post',
                                        value = -1)

        Y =  np.array([to_categorical(seq, num_classes=self.n_classes)
                            for seq in Y ])
                                        
        Y_mono = sequence.pad_sequences( batch_label_mono,
                                        maxlen=max_length,
                                        dtype='int32',
                                        padding='post',
                                        value = -1)

        Y_mono =  np.array([to_categorical(seq, num_classes=self.n_mono_classes) 
                                for seq in Y_mono])
        # seq2seq processing                        
        max_length = max([ len(f) for f in batch_dec_inputs_mono])
        X_dec = sequence.pad_sequences( batch_dec_inputs_mono,
										maxlen=max_length,
										dtype='int32',
                                        padding='post',
                                        value = -1)
                                        
        Y_dec = sequence.pad_sequences( batch_dec_targets_mono,
                                        maxlen=max_length,
                                        dtype='int32',
                                        padding='post',
                                        value = -1)

        X_dec =  np.array([to_categorical(seq, num_classes=self.n_mono_classes+2)
                            for seq in X_dec ])                                        
        Y_dec =  np.array([to_categorical(seq, num_classes=self.n_mono_classes+2)
                            for seq in Y_dec ])
                            
                                      
        return X,X_dec, Y, Y_mono,Y_dec




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
priors = np.genfromtxt('priors.csv', delimiter=',')
dev_ref_list= open("/media/lumi/alpha/kaldi/egs/timit/s5/data/dev/text").readlines()
test_ref_list= open("/media/lumi/alpha/kaldi/egs/timit/s5/data/test/text").readlines()    

# create output dir
out_path=cfg['out_path']
if not os.path.exists(out_path):
    os.makedirs(out_path)



# read training data
feat_ids  = [i.split()[0] for i in open(config_kaldi.train_feats_scp, 'r').readlines()]
train_sample_weight= [1]*len(feat_ids) 
feats_ark = RandomAccessMatrixReader(config_kaldi.final_fmllr_train_rspecifier)
tri_align_ark = RandomAccessIntVectorReader(config_kaldi.pdf_train_rspecifier)
mono_align_ark = RandomAccessIntVectorReader(config_kaldi.mono_train_rspecifier)
mono_targets_ark = RandomAccessIntVectorReader(config_kaldi.mono_targets_train_rspecifier)

mean_var_file="mean_std_fmllr.npz"

# 
asr = load_decoder()


# build data generator
if 'seq2seq-encoder-decoder' in cfg['model']:
    training_generator = DataGeneratorSeq2Seq( feat_ids, feats_ark,
                                        tri_align_ark,
                                        mono_align_ark,
                                        mono_targets_ark,
                                        train_sample_weight,
                                        mean_var_file, 
                                        num_classes,
                                        num_mono_classes,
                                        min_length=0, 
                                        max_length=seq_length,
                                        batch_size=batch_size,
                                        shuffle=False)
else:                                        
    training_generator = DataGenerator( feat_ids, feats_ark,
                                        tri_align_ark,
                                        mono_align_ark,
                                        train_sample_weight,
                                        mean_var_file, 
                                        num_classes,
                                        num_mono_classes,
                                        min_length=0, 
                                        max_length=seq_length,
                                        batch_size=batch_size,
                                        shuffle=False)

model,parallel_model  = model_definition.create_model(cfg)
sgd = model_definition.create_optimizer(cfg)
parallel_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
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
    parallel_model.fit_generator(generator=training_generator, 
                                    epochs=1,
                                    workers=1, use_multiprocessing=False)
    #model.save (out_path+'/model_temp.hdf5', overwrite=True)
                                
    #new_val_wer=eval_dev_old(out_path+'/model_temp.hdf5', model, out_path)\
    new_val_wer=eval_dev(model, asr, priors, mean_var_file, "dev", dev_ref_list)
    print ('Epoch %d learning rate: %f local_dev_wer %f global_dev_wer %f patience %d equal_patience %d' % (epoch,learning_rate,new_val_wer,val_wer,patience,equal_patience))

    if ((val_wer-new_val_wer)/new_val_wer)<0.001:
        learning_rate *= lrHalvingFactor
    else:     
        val_wer = new_val_wer
        model.save (out_path+'/model.hdf5', overwrite=True)




