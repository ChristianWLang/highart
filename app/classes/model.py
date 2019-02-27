#!/usr/bin/env python3

import tensorflow as tf
import keras


def get_model(dims):

    seq_input = keras.layers.Input(shape = dims)

    x = keras.layers.CuDNNLSTM(512,
            return_sequences = True)(seq_input)

    x = keras.layers.Dropout(.5)(x)
    
    x = keras.layers.CuDNNLSTM(512,
            return_sequences = True)(seq_input)

    x = keras.layers.Dropout(.5)(x)
    
    x = keras.layers.CuDNNLSTM(512,
            return_sequences = False)(x)

    x = keras.layers.Dropout(.5)(x)

    x = keras.layers.Dense(128,
            activation = 'relu')(x)

    seq_output = keras.layers.Dense(dims[1],
            activation = 'softmax')(x)

    model = keras.models.Model(
            inputs = seq_input,
            outputs = seq_output)

    model.compile(
            optimizer = 'adam',
            loss = 'categorical_crossentropy',
            metrics = ['acc']
            )

    es = keras.callbacks.EarlyStopping(monitor = 'loss',
            min_delta = 0,
            patience = 50,
            verbose = 1,
            mode = 'auto',
            restore_best_weights = True
            )

    return model, es
