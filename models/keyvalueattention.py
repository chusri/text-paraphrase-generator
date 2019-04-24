from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf


class KeyValueAttention(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)
        split_length = lambda x: x.shape[2] // 2
        key = tf.keras.layers.Lambda(lambda x: x[:, :, :split_length(x)])(
            values)
        value = tf.keras.layers.Lambda(
            lambda x: x[:, :, split_length(x):])(values)

        repeator = tf.keras.layers.RepeatVector(key.shape[1])
        s_prev = repeator(query)
        concat = tf.keras.layers.Concatenate(axis=-1)([key, s_prev])
        e = self.W1(concat)
        score = self.V(e)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.keras.layers.Dot(axes=1)(
            [attention_weights, values])
        return context_vector, attention_weights


if __name__ == "__main__":
    attention_layer = KeyValueAttention(10)
    print(attention_layer)
    # attention_result, attention_weights = attention_layer(
    #     sample_hidden, sample_output)

    # print("Attention result shape: (batch size, units) {}".format(
    #     attention_result.shape))
    # print(
    #     "Attention weights shape: (batch_size, sequence_length, 1) {}".format(
    #         attention_weights.shape))
