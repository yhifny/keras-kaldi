#
# Author: yasser hifny
#
import math
import numpy as np
np.random.seed(1337)  # for reproducibility
import sys
import codecs
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import utils 
from tensorflow.keras.layers import dot, concatenate, Activation, Embedding , Dense, Input,LSTM,GlobalAveragePooling1D,GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import  SpatialDropout1D,Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing import sequence
import tensorflow.keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
#sys.setdefaultencoding('utf-8')
import glob
import os

from tensorflow.keras.callbacks import Callback
import warnings
import tensorflow as tf
import random

import yaml

#from gated_cnn import  GatedConvBlock
from GCNN import GatedConv1D
from novograd import NovoGrad
from my_layers import ContextExpansion
import transformer 
#from parallel_model import MultiGpuModel, multi_gpu_wrapper
#from keras_transformer.keras_transformer import TransformerBlock

def create_optimizer(config):

    name = config['optimizer']['name']
    lr = config['optimizer']['lr']
    
    if name == 'novograd':
        return NovoGrad(lr=lr, 
                    beta_1=config['optimizer']['beta_1'], 
                    beta_2=config['optimizer']['beta_2'], 
                    weight_decay=config['optimizer']['weight_decay'],
                    epsilon=None, 
                    decay=0.0, amsgrad=False, clipnorm=100)

    elif name == 'adam':
        return tensorflow.keras.optimizers.Adam(lr=lr, 
                    beta_1=config['optimizer']['beta_1'], 
                    beta_2=config['optimizer']['beta_2'], 
                    epsilon=None, 
                    decay=0.0, amsgrad=False, clipnorm=100)
    elif name == 'rmsprop':
        return tensorflow.keras.optimizers.RMSprop(lr=lr,
                    rho=config['optimizer']['rho'], 
                    epsilon=None,
                    decay=0.0, clipnorm=100)
    elif name== 'sgd':
        return tensorflow.keras.optimizers.SGD(lr=lr, 
                    momentum=config['optimizer']['momentum'],
                    decay=0.0, 
                    nesterov=True, clipnorm=100)
    else:
        raise("optimizer %s is not supported"%config['optimizer']['name'])
        


def create_layer(config):

    name=None
    if 'name' in config:
        name=config['name']


    #postionalembedding    
    if config['type']=='postional_embedding':
        return transformer.Position_Embedding(name=name)

    if config['type']=='transformer_encoder_layer':
        return False
        # TransformerBlock( name=name,num_heads=config['n_head'],
                                 # n_state=config['n_state'],    
                                 # residual_dropout=config['residual_dropout'],
                                 # attention_dropout=config['attention_dropout'],
                                 # use_masking=False, vanilla_wiring=False)


   #context_expansion
    if config['type']=='context_expansion':
        return ContextExpansion(config['left'], config['right'])

    #masking
    if config['type']=='masking':
        return tensorflow.keras.layers.Masking(mask_value=config['value'], name=name)
    #lstm    
    elif config['type']=='lstm':
        layer=tensorflow.keras.layers.LSTM(config['units'],
                 return_sequences=True,
                 dropout=config['dropout'],
                 recurrent_dropout=config['rec_dropout'],
                 name=name)
        if config['bidirectional']:                
            return tensorflow.keras.layers.Bidirectional(layer, merge_mode=config['merge_mode'])
        else:
            return layer
            
    elif config['type']=='cu_lstm':
        layer=tensorflow.keras.layers.CuDNNLSTM(config['units'],
                 return_sequences=True,
                 name=name)
        if config['bidirectional']:                
            return tensorflow.keras.layers.Bidirectional(layer, merge_mode=config['merge_mode'])
        else:
            return layer            
    #gru    
    elif config['type']=='gru':
        layer=tensorflow.keras.layers.GRU(config['units'],
                 return_sequences=True,
                 dropout=config['dropout'],
                 recurrent_dropout=config['rec_dropout'],
                 name=name)
        if config['bidirectional']:                
            return tensorflow.keras.layers.Bidirectional(layer, merge_mode=config['merge_mode'])
        else:
            return layer
            
    elif config['type']=='cu_gru':
        layer=tensorflow.keras.layers.CuDNNGRU(config['units'],
                 return_sequences=True,
                 name=name)
        if config['bidirectional']:                
            return tensorflow.keras.layers.Bidirectional(layer, merge_mode=config['merge_mode'])
        else:
            return layer            


    #dropout    
    elif config['type']=='dropout':
        return tensorflow.keras.layers.Dropout(config['value'], name=name)

    #dropout    
    elif config['type']=='spatialdropout1d':
        return tensorflow.keras.layers.SpatialDropout1D(config['value'], name=name)

    #batchnormalization    
    elif config['type']=='batchnormalization':
        return tensorflow.keras.layers.BatchNormalization(name=name)

    #layer normalization     
    elif config['type']=='layernormalizaton':
        return tensorflow.keras.layers.batchnormalization(name=name)

    #conv1d
    elif config['type']=='conv1d':
        layer = tensorflow.keras.layers.Conv1D(config['units'], 
                    config['kernel_width'], padding='same',
                    activation=config['activation'],
                    strides=config['strides'],
                    name=name)
        if 'gated_conv' not in config: return layer
        name1 = name2 = None
        if name is not None:
            name1 = name+'_c'
            name2 = name+'_g'
        return GatedConv1D(config['units'], 
                config['kernel_width'], 
                kwargs_conv={'name': name1,'padding':'same', 'activation':config['activation'], 'strides':config['strides']},
                kwargs_gate={'name': name2,'padding':'same', 'strides':config['strides']})
                    
    #dense
    elif config['type']=='dense':
        return TimeDistributed(tensorflow.keras.layers.Dense(config['units'],             
                    activation=config['activation']),
                    name=name)
                    
    #activation
    elif config['type']=='activation':
        return tensorflow.keras.layers.Activation(config['function'], name=name)
       
    else:
        raise("Layer construction is not supported %s"%config['type'])
        

def create_model(config):

    # random seed
    np.random.seed(config['seed'])
    tensorflow.random.set_seed(config['seed'])
    random.seed(config['seed'])



    encoder_input_tesnor=[]
    encoder_hidden_tesnor=[]
    
    # input features
    input_dim = config['model']["input_dim"]
    input_feat = config['model']["input_feat"]
    
    # create input layer
    input_tensor = Input([None, input_dim], name='X_%s'%input_feat)
    encoder_input_tesnor.append(input_tensor)
    
    # encoders
    for encoder_config in config['model']['encoders']:
        x = input_tensor
        for layer_config in config['model']['encoders'][encoder_config]['layers']:
            x = create_layer(layer_config)(x)
        encoder_hidden_tesnor.append(x)
        
    merged_tensor=encoder_hidden_tesnor[0]    
    if len(encoder_hidden_tesnor)>1:
        merged_tensor=tensorflow.keras.layers.concatenate(encoder_hidden_tesnor)    
    
    decoder_output_tesnor=[]
    for output_config in config['model']['outputs']:
        x=merged_tensor
        for layer_config in config['model']['outputs'][output_config]['layers']:
            x = create_layer(layer_config)(x)
        decoder_output_tesnor.append(x)

    if 'seq2seq-encoder-decoderx' in config['model']:     
        for seq2seq in config['model']['seq2seq-encoder-decoder']:
            x=merged_tensor
            seq_2seq_encoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['encoder']
            x,  state_h, state_c =  tensorflow.keras.layers.CuDNNLSTM(seq_2seq_encoder_config[0]['lstm_units'], return_state=True)(x)
            encoder_states = [state_h, state_c]
            seq_2seq_decoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['decoder']
            decoder_inputs = Input([None, seq_2seq_decoder_config[0]['units']], name = 'input_seq2seq' )
            encoder_input_tesnor.append(decoder_inputs)
            x, _, _ = tensorflow.keras.layers.CuDNNLSTM(seq_2seq_decoder_config[0]['lstm_units'], 
                            return_sequences=True, 
                            return_state=True)(decoder_inputs, initial_state=encoder_states)
            x = Dense(seq_2seq_decoder_config[0]['units'],
                            activation='softmax', name='output_seq2seq')(x)
            decoder_output_tesnor.append(x)
        
    if 'seq2seq-encoder-decoderxx' in config['model']:     
        for seq2seq in config['model']['seq2seq-encoder-decoder']:
            x=merged_tensor
            seq_2seq_encoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['encoder']
            enc_units = seq_2seq_encoder_config[0]['lstm_units']
            seq_2seq_decoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['decoder']
            output_units = seq_2seq_decoder_config[0]['units']
            dec_units = seq_2seq_decoder_config[0]['lstm_units']
            context_units = seq_2seq_decoder_config[0]['context_units']
            decoder_inputs = Input([None, output_units ], name = 'input_seq2seq' )
            encoder_input_tesnor.append(decoder_inputs)
            enc_dec = transformer.EncoderDecoderLayer(enc_units, dec_units, 
                                        context_units )([x, decoder_inputs])
            
            output = TimeDistributed(Dense(output_units,
                            activation='softmax'), name='output_seq2seq')(enc_dec)
            decoder_output_tesnor.append(output)        

    if 'seq2seq-encoder-decoderxxx' in config['model']:     
        for seq2seq in config['model']['seq2seq-encoder-decoder']:
            x=merged_tensor
            seq_2seq_encoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['encoder']
            enc_units = seq_2seq_encoder_config[0]['lstm_units']
            seq_2seq_decoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['decoder']
            output_units = seq_2seq_decoder_config[0]['units']
            dec_units = seq_2seq_decoder_config[0]['lstm_units']
            context_units = seq_2seq_decoder_config[0]['context_units']
            n_state = seq_2seq_decoder_config[0]['n_state']
            n_head = seq_2seq_decoder_config[0]['n_head']
            
            decoder_inputs = Input([None, output_units ], name = 'input_seq2seq' )
            encoder_input_tesnor.append(decoder_inputs)
            enc_dec = transformer.EncoderDecoderMultiHeadAttentionLayer(enc_units, 
                                        dec_units,
                                        n_state,
                                        n_head,                                        
                                        context_units )([x, decoder_inputs])
            
            output = TimeDistributed(Dense(output_units,
                            activation='softmax'), name='output_seq2seq')(enc_dec)
            decoder_output_tesnor.append(output)

    if 'seq2seq-encoder-decoder' in config['model']:     
        for seq2seq in config['model']['seq2seq-encoder-decoder']:
            x=merged_tensor
            #seq_2seq_encoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['encoder']
            #enc_units = seq_2seq_encoder_config[0]['lstm_units']
            seq_2seq_decoder_config=config['model']['seq2seq-encoder-decoder'][seq2seq]['decoder']
            output_units = seq_2seq_decoder_config[0]['units']
            dec_units = seq_2seq_decoder_config[0]['lstm_units']
            context_units = seq_2seq_decoder_config[0]['context_units']
            n_state = seq_2seq_decoder_config[0]['n_state']
            n_head = seq_2seq_decoder_config[0]['n_head']
            
            decoder_inputs = Input([None, output_units ], name = 'input_seq2seq' )
            encoder_input_tesnor.append(decoder_inputs)
            enc_dec = transformer.DecoderMultiHeadAttentionLayer( dec_units,
                                        n_state,
                                        n_head,                                        
                                        context_units )([x, decoder_inputs])
            
            output = TimeDistributed(Dense(output_units,
                            activation='softmax'), name='output_seq2seq')(enc_dec)
            decoder_output_tesnor.append(output)

            
    
    model = Model(encoder_input_tesnor, decoder_output_tesnor, name='kaldi-deep-speech')
    return [model, model] 
