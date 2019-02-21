#!/usr/bin/env python3

import numpy as np


def generate_text(start, generator, model, charmap, length = 100, verbose = False):
    text = '' + start
    while len(text) < length:
        
        tensor = generator.get_tensor(text)
        
        probabilities = model.predict(tensor).flatten()
        
        char = np.random.choice(list(charmap.keys()), p = probabilities)

        text += char

        if verbose:
            print(text)

        yield text
