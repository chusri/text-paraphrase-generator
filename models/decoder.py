from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from models.keyvalueattention import KeyValueAttention


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super().__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            self.dec_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights


if __name__ == "__main__":
    vocab_inp_size = 1000 + 1
    BATCH_SIZE = 64
    embedding_dim = 256
    units = 1024
    vocab_tar_size = 1000
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    print(decoder)