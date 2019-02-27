#!/usr/bin/env python3

import urllib.request as ur


def get_text():

    text = []
    with open('urls.txt', 'r') as lines:
        for url in lines:
            text.append(str(ur.urlopen(url).read()))

    text = '/r/r/r/r/r'.join(text)

    print(text[:100])

    return text
if __name__ == '__main__':
    get_text()
