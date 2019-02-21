#!/usr/bin/env python3

import urllib.request as ur
import configparser

import os
import logging

from .classes.generator import Generator
from .classes.model import get_model
from .funcs.generate_text import generate_text


def main():

    os.mkdir('tmp')

    logging.basicConfig(
            filename = 'tmp/training.log',
            level = logging.DEBUG,
            format = '%(asctime)s;%(levelname)s;%(message)s',
            datefmt = '%Y-%m-%d %H:%M:%S'
            )

    config = configparser.ConfigParser()
    config.read('config.ini')

    timesteps = int(config['TRAINING']['TIMESTEPS'])
    batch_size = int(config['TRAINING']['BATCH_SIZE'])
    epochs = int(config['TRAINING']['EPOCHS'])

    url = 'http://www.gutenberg.org/cache/epub/16328/pg16328.txt'
    txt = str(ur.urlopen(url).read())

    chars = sorted(set(txt))
    charmap = {v:i for i,v in enumerate(chars)}
    inverse_charmap = {i:v for i,v in enumerate(chars)}

    generator = Generator(X = txt,
            charmap = charmap,
            dims = (timesteps, len(charmap)),
            batch_size = batch_size,
            shuffle = True)

    model = get_model((timesteps, len(charmap)))

    for epoch in range(epochs):
        model.fit_generator(
                generator = generator,
                use_multiprocessing = True
                )

        generated_text = ''
        generated = generate_text(
                start = generated_text,
                generator = generator,
                model = model,
                charmap = charmap
                )

        for i in generated:
            logging.info('Epoch: {} Text: {}'.format(epoch, generated_text))

    return
if __name__ == '__main__':
    main()
