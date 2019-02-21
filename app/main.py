#!/usr/bin/env python3

import urllib.request as ur
from .classes.generator import Generator
from .classes.model import get_model


def main():

    url = 'http://www.gutenberg.org/cache/epub/16328/pg16328.txt'
    txt = str(ur.urlopen(url).read())

    chars = sorted(set(txt))
    charmap = {v:i for i,v in enumerate(chars)}
    inverse_charmap = {i:v for i,v in enumerate(chars)}

    generator = Generator(X = txt,
            charmap = charmap,
            dims = (128, len(charmap)),
            batch_size = 32,
            shuffle = True)

    model = get_model((128, len(charmap)))

    model.fit_generator(
            generator = generator,
            use_multiprocessing = True
            )

    return
if __name__ == '__main__':
    main()
