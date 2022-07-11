#
# Author: yasser hifny
#
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence

# smooth_labels from https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/

def smooth_labels(y, smooth_factor):
    '''Convert a matrix of one-hot row-vector labels into smoothed versions.

    # Arguments
        y: matrix of one-hot row-vector labels to be smoothed
        smooth_factor: label smoothing factor (between 0 and 1)

    # Returns
        A matrix of smoothed labels.
    '''
    assert len(y.shape) == 2
    if 0 <= smooth_factor <= 1:
        # label smoothing ref: https://www.robots.ox.ac.uk/~vgg/rg/papers/reinception.pdf
        y *= 1 - smooth_factor
        y += smooth_factor / y.shape[1]
    else:
        raise Exception(
            'Invalid label smoothing factor: ' + str(smooth_factor))
    return y


def normalize( feature,  feats_mean, feats_std, eps=1e-14):
    return (feature - feats_mean) / (feats_std + eps)

    
def to_categorical(y, num_classes, dtype='float32'):
    
    onehot_encoded = []
    for value in y:
        column = [0.0 for _ in range(num_classes)]
        if value>=0: 
            column[value] = 1.0 
        onehot_encoded.append(column)
    y = np.array(onehot_encoded , dtype=dtype)
    #y = smooth_labels(y, 0.1)
    return  y



class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_dic,feat_norm_file,
                 n_classes,n_mono_classes, batch_size=32, min_length=0, max_length=20000,
                 shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.list_files = data_dic['feat_ids']
        self.feats_ark  = data_dic['feats_ark']
        self.align_ark  = data_dic['tri_align_ark']
        self.mono_align_ark  = data_dic['mono_align_ark']        
        self.sample_weight  = data_dic['sample_weight']
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
        #print (X.shape)
        #for k in indexes:
        #    print (self.list_files[k]) 
        #print ("index", index)        
        
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
	
class DataGeneratorSeq2Seq(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_dic, feat_norm_file,
                    n_classes,n_mono_classes, batch_size=32, min_length=0, max_length=20000,
                    shuffle=False):
        'Initialization'
        self.batch_size = batch_size
        self.list_files = data_dic['feat_ids']
        self.feats_ark  = data_dic['feats_ark']
        self.align_ark  = data_dic['tri_align_ark']
        self.mono_align_ark  = data_dic['mono_align_ark']        
        self.sample_weight  = data_dic['sample_weight']
        self.mono_targets_ark  =  data_dic['mono_targets_ark']        
        self.n_classes = n_classes
        self.n_mono_classes = n_mono_classes        
        self.shuffle = shuffle
        self.max_length = max_length
        self.feats_mean = np.load(feat_norm_file)['mean']
        self.feats_std = np.load(feat_norm_file)['std']
        self.on_epoch_end()
        print ("generator  seq2seq is based on ", len(self.list_files),"files")
		

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
                                        padding='pre')
                                        
        Y = sequence.pad_sequences( batch_label,
                                        maxlen=max_length,
                                        dtype='int32',
                                        padding='pre',
                                        value = -1)

        Y =  np.array([to_categorical(seq, num_classes=self.n_classes)
                            for seq in Y ])
                                        
        Y_mono = sequence.pad_sequences( batch_label_mono,
                                        maxlen=max_length,
                                        dtype='int32',
                                        padding='pre',
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


