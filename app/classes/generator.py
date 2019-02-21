#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


class Generator(tf.keras.utils.Sequence):
    def __init__(self, X, charmap, dims, batch_size = 32, shuffle=True):
        self.X = X
        self.charmap = charmap
        self.dims = dims
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.X) // self.batch_size

    def __getitem__(self, idx):
        idxs = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        return self.get_batch(idxs)

    def get_batch(self, idxs):

        X, y = [], []
        for idx in idxs:

            if idx - self.dims[0] < 0:
                _X_text, _y_text = self.X[0:idx], self.X[idx]
            else:
                _X_text, _y_text = self.X[idx - self.dims[0]:idx], self.X[idx]

            _X, _y = np.zeros(shape = (len(_X_text), self.dims[1])), np.zeros(shape = (self.dims[1],))
            
            for i, x in enumerate(_X_text):
                if x in self.charmap:
                    _X[i, self.charmap[x]] = 1

            if len(_X) < self.dims[0]:
                padding = np.zeros(shape = (self.dims[0] - len(_X), self.dims[1]))
                _X = np.vstack([padding, _X])

            if _y_text in self.charmap:
                _y[self.charmap[_y_text]] = 1

            X.append(_X)
            y.append(_y)

        X, y = np.array(X), np.array(y)

        return X, y

    def get_tensor(self, text):

        text = text[-self.dims[0]:]

        X = np.zeros(shape = (len(text), self.dims[1]))
        for i, v in enumerate(text):
            if v in self.charmap:
                X[i, self.charmap[v]] = 1

        if len(X) < self.dims[0]:
            padding = np.zeros(shape = (self.dims[0] - len(X), self.dims[1]))
            X = np.vstack([padding, X])

        X = X.reshape(1, X.shape[0], X.shape[1])

        return X
