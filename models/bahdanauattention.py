from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == class BahdanauAttention(tf.keras.Model):
        def __init__(self, units):
            super(BahdanauAttention, self).__init__()
            self.W1 = tf.keras.layers.Dense(units)
            self.W2 = tf.keras.layers.Dense(units)
            self.V = tf.keras.layers.Dense(1)

        def call(self, query, values):
            # hidden shape == (batch_size, hidden size)
            # hidden_with_time_axis shape == (batch_size, 1, hidden size)
            # we are doing this to perform addition to calculate the score
            hidden_with_time_axis = tf.expand_dims(query, 1)

            # score shape == (batch_size, max_length, hidden_size)
            score = self.V(
                tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

            # attention_weights shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            attention_weights = tf.nn.softmax(score, axis=1)

            # context_vector shape after sum == (batch_size, hidden_size)
            context_vector = attention_weights * values
            context_vector = tf.reduce_sum(context_vector, axis=1)

            return context_vector, attention_weights(batch_size, max_length,
                                                     hidden_size)

        score = self.V(
            tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


if __name__ == "__main__":
    attention_layer = BahdanauAttention(10)
    print(attention_layer)
    # attention_result, attention_weights = attention_layer(
    #     sample_hidden, sample_output)

    # print("Attention result shape: (batch size, units) {}".format(
    #     attention_result.shape))
    # print(
    #     "Attention weights shape: (batch_size, sequence_length, 1) {}".format(
    #         attention_weights.shape))
