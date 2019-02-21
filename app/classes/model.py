#!/usr/bin/env python3

import tensorflow as tf


def get_model(dims):

    seq_input = tf.keras.layers.Input(shape = dims)

    x = tf.keras.layers.LSTM(128,
            return_sequences = True,
            use_bias = True)(seq_input)

    x = tf.keras.layers.Dropout(.5)(x)
    
    x = tf.keras.layers.LSTM(128,
            return_sequences = False,
            use_bias = True)(x)

    x = tf.keras.layers.Dropout(.5)(x)

    x = tf.keras.layers.Dense(128,
            activation = 'relu')(x)

    seq_output = tf.keras.layers.Dense(dims[1],
            activation = 'softmax')(x)

    model = tf.keras.models.Model(
            inputs = seq_input,
            outputs = seq_output)

    model.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['acc']
            )

    return model
