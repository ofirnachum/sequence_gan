__doc__ = """Char-based Seq-GAN on data from a book."""

import model
import train

import numpy as np
import tensorflow as tf
import random
import subprocess

EMB_DIM = 10
HIDDEN_DIM = 15
SEQ_LENGTH = 7
START_TOKEN = 0

EPOCH_ITER = 1000
CURRICULUM_RATE = 0.03  # how quickly to move from supervised training to unsupervised
TRAIN_ITER = 100000  # generator/discriminator alternating
D_STEPS = 3  # how many times to train the discriminator per generator step
LEARNING_RATE = 0.01 * SEQ_LENGTH
SEED = 88

DATA_FILE = 'book.txt'


def tokenize(s):
    return [c for c in ' '.join(s.split())]


def get_data(download=True):
    """Downloads and parses Moby Dick."""
    if download:
        subprocess.check_output(
            ['wget', 'http://www.gutenberg.org/cache/epub/2701/pg2701.txt',
             '-O', DATA_FILE])

    token_stream = []
    with open(DATA_FILE, 'r') as f:
        for line in f:
            if 'Call me Ishmael.' in line or token_stream:
                token_stream.extend(tokenize(line.strip().lower()))
                token_stream.append(' ')
            if len(token_stream) > 10000 * SEQ_LENGTH:  # enouch data
                break

    return token_stream


def get_trainable_model(num_emb):
    return model.GRU(
        num_emb, EMB_DIM, HIDDEN_DIM,
        SEQ_LENGTH, START_TOKEN,
        learning_rate=LEARNING_RATE)


def get_random_sequence(token_stream, word2idx):
    """Returns random subequence."""
    start_idx = random.randint(0, len(token_stream) - SEQ_LENGTH)
    return [word2idx[tok] for tok in token_stream[start_idx:start_idx + SEQ_LENGTH]]


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    token_stream = get_data()
    assert START_TOKEN == 0
    words = ['_START'] + list(set(token_stream))
    word2idx = dict((word, i) for i, word in enumerate(words))
    num_words = len(words)
    print 'num words', num_words
    print 'stream length', len(token_stream)

    trainable_model = get_trainable_model(num_words)
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    print 'training'
    for epoch in xrange(TRAIN_ITER // EPOCH_ITER):
        print 'epoch', epoch
        proportion_supervised = max(0.0, 1.0 - CURRICULUM_RATE * epoch)
        train.train_epoch(
            sess, trainable_model, EPOCH_ITER,
            proportion_supervised=proportion_supervised,
            g_steps=1, d_steps=D_STEPS,
            next_sequence=lambda: get_random_sequence(token_stream, word2idx),
            words=words)


if __name__ == '__main__':
    main()
