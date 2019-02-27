#!/usr/bin/env python3

import configparser

import os
import shutil
import logging

from .classes.generator import Generator
from .classes.model import get_model
from .funcs.generate_text import generate_text
from .funcs.get_text import get_text


def main():

    if not os.path.exists('tmp/'):
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

    txt = get_text()

    chars = sorted(set(txt))
    charmap = {v:i for i,v in enumerate(chars)}
    inverse_charmap = {i:v for i,v in enumerate(chars)}

    generator = Generator(X = txt,
            charmap = charmap,
            dims = (timesteps, len(charmap)),
            batch_size = batch_size,
            shuffle = True)

    model, es = get_model((timesteps, len(charmap)))

    model.fit_generator(
            generator = generator,
            epochs = epochs,
            callbacks = [es],
            use_multiprocessing = True
            )

    generated_text = ''
    generated = generate_text(
            start = generated_text,
            generator = generator,
            model = model,
            charmap = charmap,
            length = 1000
            )

    for i in generated:
        logging.info('Text: {}'.format(epoch, i))

    return
if __name__ == '__main__':
    main()
