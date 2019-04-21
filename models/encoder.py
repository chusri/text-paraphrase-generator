from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


if __name__ == "__main__":
    vocab_inp_size = 1000 + 1
    BATCH_SIZE = 64
    embedding_dim = 256
    units = 1024
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    # sample input
    sample_hidden = encoder.initialize_hidden_state()
    # sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # print(
    #     'Encoder output shape: (batch size, sequence length, units) {}'.format(
    #         sample_output.shape))
    # print('Encoder Hidden state shape: (batch size, units) {}'.format(
    #     sample_hidden.shape))
