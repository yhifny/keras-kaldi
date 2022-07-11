# based on https://www.kaggle.com/fareise/multi-head-self-attention-for-text-classification/comments
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, dot, concatenate,Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D, GRU,  BatchNormalization
from tensorflow.keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten, TimeDistributed, Activation
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Ones, Zeros



    
    
class Position_Embedding(tf.keras.layers.Layer):
    
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)
        
    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)
        
    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)

def gelu(x):
    return 0.5 * x * (1 + K.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * K.pow(x, 3))))
    

'''
output dimention: [batch_size, time_step, nb_head*size_per_head]
every word can be represented as a vector [nb_head*size_per_head]
'''
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ', 
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK', 
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV', 
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(MultiHeadAttention, self).build(input_shape)
        
    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
                
    def call(self, x, **kwargs):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))    
        A = K.softmax(A)
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)
        
    def get_config(self):
        config = {
            'nb_head': self.nb_head,
            'size_per_head':self.size_per_head,
            
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, eps: float = 1e-5, **kwargs) -> None:
        self.eps = eps
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, x, **kwargs):
        u = K.mean(x, axis=-1, keepdims=True)
        s = K.mean(K.square(x - u), axis=-1, keepdims=True)
        z = (x - u) / K.sqrt(s + self.eps)
        return self.gamma * z + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'eps': self.eps,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
# https://github.com/Separius/BERT-tensorflow.keras/blob/master/transformer/model.py

class Gelu(tf.keras.layers.Layer):
    def __init__(self, accurate= False, **kwargs):
        super().__init__(**kwargs)
        self.accurate = accurate

    def call(self, inputs, **kwargs):
        if not self.accurate:
            return gelu(inputs)
        if K.backend() == 'tensorflow':
            erf = K.tf.erf
        else:
            erf = K.T.erf
        return inputs * 0.5 * (1.0 + erf(inputs / math.sqrt(2.0)))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'accurate': self.accurate,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
        
        
class PositionWiseFF:
    def __init__(self, n_state, d_hid, layer_id, accurate_gelu=True):
        self.c_fc = Conv1D(d_hid, 1)
        self.activation = Gelu(accurate=accurate_gelu)
        self.c_ffn_proj = Conv1D(n_state, 1)

    def __call__(self, x):
        output = self.activation(self.c_fc(x))
        return self.c_ffn_proj(output)



class TransformerEncoderLayer:
    def __init__(self, n_state, n_head, d_hid,  attention_dropout, residual_dropout, **kwargs):
        self.attention = MultiHeadAttention(n_head, n_state)
        self.drop1 = Dropout(attention_dropout)
        self.add1 = Add()
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ffn = PositionWiseFF(n_head*n_state, d_hid, True)
        self.drop2 = Dropout(residual_dropout)
        self.add2 = Add()
        self.ln2 = tf.keras.layers.LayerNormalization()

    def __call__(self, x, **kwargs):
        a = self.attention([x,x,x]) #output: [batch_size, time_step, nb_head*size_per_head]
        n = self.ln1(self.add1([x, self.drop1(a)]))
        f = self.ffn(n)
        return self.ln2(self.add2([n, self.drop2(f)]))


#x = Position_Embedding()(x)
#x = class TransformerEncodertf.keras.layers.Layer(256, 2, 1024, 0.2, 0.2)(x)


def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


class EncoderDecoderLayer:
    def __init__(self, enc_units,dec_units, context_units):
        self.enc_lstm=tensorflow.keras.layers.CuDNNGRU(enc_units,
                             return_sequences=True,
                             name = 'lstm_encoder')
        self.dec_lstm = tensorflow.keras.layers.CuDNNGRU(dec_units, 
                            return_sequences=True, 
                            name = 'lstm_decoder')
                            
        self.dense1 = TimeDistributed(Dense(context_units,
                            activation="tanh"))

    def __call__(self, x, **kwargs):
        x_enc, x_dec = x
        encoder = self.enc_lstm (x_enc)
        encoder_last = Lambda(lambda t:  t[:,-1,:])(encoder)
        decoder= self.dec_lstm(x_dec,initial_state=[encoder_last])
        attention = dot([decoder, encoder], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder], axes=[2,1])
        decoder_combined_context = concatenate([context, decoder])
        output = self.dense1(decoder_combined_context)
        return output    

class EncoderDecoderMultiHeadAttentionLayer:
    def __init__(self, enc_units,dec_units, n_state, n_head, context_units):
        
        self.attention = MultiHeadAttention(n_head, n_state)
        
        self.enc_lstm=tensorflow.keras.layers.CuDNNGRU(enc_units,
                             return_sequences=True,
                             name = 'lstm_encoder')
        self.dec_lstm = tensorflow.keras.layers.CuDNNGRU(dec_units, 
                            return_sequences=True, 
                            name = 'lstm_decoder')
                            
        self.dense1 = TimeDistributed(Dense(context_units,
                            activation="tanh"))

    def __call__(self, x, **kwargs):
        x_enc, x_dec = x
        encoder = self.enc_lstm (x_enc)
        encoder_last = Lambda(lambda t:  t[:,-1,:])(encoder)
        decoder = self.dec_lstm(x_dec,initial_state=[encoder_last])
        context = self.attention ([ decoder, encoder, encoder])
        decoder_combined_context = concatenate([context, decoder])
        output = self.dense1(decoder_combined_context)
        return output        
        

class DecoderMultiHeadAttentionLayer:
    def __init__(self, dec_units, n_state, n_head, context_units):
        
        self.attention = MultiHeadAttention(n_head, n_state)
        
        self.dec_lstm = tensorflow.keras.layers.CuDNNGRU(dec_units, 
                            return_sequences=True, 
                            name = 'lstm_decoder')
                            
        self.dense1 = TimeDistributed(Dense(context_units,
                            activation="tanh"))

    def __call__(self, x, **kwargs):
        encoder, x_dec = x
        encoder_last = Lambda(lambda t:  t[:,-1,:])(encoder)
        decoder = self.dec_lstm(x_dec,initial_state=[encoder_last])
        context = self.attention ([ decoder, encoder, encoder])
        decoder_combined_context = concatenate([context, decoder])
        output = self.dense1(decoder_combined_context)
        return output        

        
#https://stackoverflow.com/questions/58372387/scheduled-sampling-in-tensorflow.keras