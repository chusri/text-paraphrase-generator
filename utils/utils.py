from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pandas as pd
import numpy as np
import tensorflow as tf


def download_file():
    # Download the file
    path_to_zip = tf.keras.utils.get_file(
        'opusparcus_en.zip',
        origin='https://korp.csc.fi/download/opusparcus/opusparcus_en.zip',
        extract=True)

    return path_to_zip


def get_train(nrows):
    path_to_zip = download_file()
    path_to_file = os.path.join(
        os.path.dirname(path_to_zip),
        'opusparcus_v1/en/train/en-train.txt.bz2')
    train = pd.read_csv(
        path_to_file,
        header=None,
        names=['firstSentence', 'secondSentence'],
        usecols=[1, 2],
        sep='\t',
        compression='bz2',
        nrows=nrows,
        encoding='utf-8')

    return train


def get_test():
    path_to_zip = download_file()
    path_to_file = os.path.join(
        os.path.dirname(path_to_zip), 'opusparcus_v1/en/test/en-test.txt')
    train = pd.read_csv(
        path_to_file,
        header=None,
        names=['firstSentence', 'secondSentence'],
        usecols=[1, 2],
        sep='\t',
        encoding='utf-8')

    return train


def get_dev():
    path_to_zip = download_file()
    path_to_file = os.path.join(
        os.path.dirname(path_to_zip), 'opusparcus_v1/en/test/en-test.txt')
    train = pd.read_csv(
        path_to_file,
        header=None,
        names=['firstSentence', 'secondSentence'],
        usecols=[1, 2],
        encoding='utf-8')

    return train


def convert(tokenizer, tensor):
    for t in tensor:
        if t != 0:
            print("{}\t{}".format(t, tokenizer.index_word[t]))


if __name__ == "__main__":
    df = get_train()
    print(df.head())
    df = get_test()
    print(df.head())
    df = get_dev()
    print(df.head())
