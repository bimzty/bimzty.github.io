#! -*- coding: utf-8 -*-

import tensorflow as tf

'''
inputs are a tensor of the form (batch_size, seq_len, word_size);
Function returns a position tensor of the form (batch_size, seq_len, position_size).
'''
def Position_Embedding(inputs, position_size):
    batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000., \
                             2 * tf.range(position_size / 2, dtype=tf.float32 \
                            ) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding


'''
inputs are a tensor of more than second order, representing the sequence of inputs, such as (batch_size, seq_len, input_size);

seq_len is a tensor of the form (batch_size,) that represents the actual length of each sequence, with any additional parts ignored;

mode is divided into mul and add. mul indicates that the excess part is set to zero, which is generally used before the full connection layer.

add means to subtract all the excess from a large constant, usually used before softmax.
'''
def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

'''
Normal full connection

inputs are a tensor of second order or more, i.e. (batch_size,... ,input_size).

Matrix multiplication is performed only on the last dimension, that is, output a matrix of the form (batch_size,... ,ouput_size) tensor.
'''
def Dense(inputs, ouput_size, bias=True, seq_len=None):
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, \
                         tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                        )
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs

'''
Implementation of Multi-Head Attention
'''
def Attention(Q, K, V, nb_head, size_per_head, Q_len=None, V_len=None):
    # Linear mapping of Q, K and V respectively
    Q = Dense(Q, nb_head * size_per_head, False)
    Q = tf.reshape(Q, (-1, tf.shape(Q)[1], nb_head, size_per_head))
    Q = tf.transpose(Q, [0, 2, 1, 3])
    K = Dense(K, nb_head * size_per_head, False)
    K = tf.reshape(K, (-1, tf.shape(K)[1], nb_head, size_per_head))
    K = tf.transpose(K, [0, 2, 1, 3])
    V = Dense(V, nb_head * size_per_head, False)
    V = tf.reshape(V, (-1, tf.shape(V)[1], nb_head, size_per_head))
    V = tf.transpose(V, [0, 2, 1, 3])
    # Calculate inner product, then mask, then softmax
    A = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(float(size_per_head))
    A = tf.transpose(A, [0, 3, 2, 1])
    A = Mask(A, V_len, mode='add')
    A = tf.transpose(A, [0, 3, 2, 1])
    A = tf.nn.softmax(A)
    # Output and mask
    O = tf.matmul(A, V)
    O = tf.transpose(O, [0, 2, 1, 3])
    O = tf.reshape(O, (-1, tf.shape(O)[1], nb_head * size_per_head))
    O = Mask(O, Q_len, 'mul')
    return O