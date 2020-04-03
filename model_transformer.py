import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Layer, LayerNormalization, Embedding, Lambda


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)

    return tf.maximum(look_ahead_mask, padding_mask)


def scaled_dot_product_attention(query, key, value, mask=None):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    nor_k_size = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(nor_k_size)

    if mask is not None:
        logits += mask * -1e9

    attention_weight = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(attention_weight, value)


class MultiHeadAttention(Layer):

    def __init__(self, emb_dim, head=8, name="multi_head_attention"):
        super().__init__(name=name)
        self.emb_dim = emb_dim
        self.head = head

        assert emb_dim % head == 0

        self.depth = emb_dim // head

        self.q_dense = Dense(units=emb_dim)
        self.k_dense = Dense(units=emb_dim)
        self.v_dense = Dense(units=emb_dim)

        self.last_dense = Dense(emb_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'emb_dim': self.emb_dim,
            'head': self.head,
            'depth': self.depth,
            'q_dense': self.q_dense,
            'k_dense': self.k_dense,
            'v_dense': self.v_dense,
            'last_dense': self.last_dense,
        })
        return config

    def _split_head(self, inputs, batch_size):

        inp = tf.reshape(inputs, shape=(batch_size, -1, self.head, self.depth))
        return tf.transpose(inp, perm=[0, 2, 1, 3])

    def call(self, inputs):
        q, k, v, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(q)[0]

        # split heads
        q = self.q_dense(q)
        q = self._split_head(q, batch_size)

        k = self.k_dense(k)
        k = self._split_head(k, batch_size)

        v = self.v_dense(v)
        v = self._split_head(v, batch_size)

        attention = scaled_dot_product_attention(q, k, v, mask)

        # concat heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        attention = tf.reshape(attention, (batch_size, -1, self.emb_dim))

        outputs = self.last_dense(attention)

        return outputs


class PositionalEncoding(Layer):

    def __init__(self, position, emb_dim):
        super().__init__()
        self.pos_encoding = self._positional_encoding(position, emb_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'pos_encoding': self.pos_encoding,
        })
        return config

    def _get_angles(self, position, i, emb_dim):

        denominator = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(emb_dim, tf.float32))
        return position * denominator

    def _positional_encoding(self, sentence_length, emb_dim):

        # 計算を効率化するためにpositionとiを行列にしてangle計算を行列の積で一度に実行する
        angle = self._get_angles(
            position=tf.expand_dims(tf.range(sentence_length, dtype=tf.float32), -1),
            i=tf.expand_dims(tf.range(emb_dim, dtype=tf.float32), 0),
            emb_dim=emb_dim
        )

        # インデックスが偶数のものはサイン関数に適応
        sine = tf.math.sin(angle[:, 0::2])
        # インデックスが奇数のものはコサイン関数に適応
        cos = tf.math.cos(angle[:, 1::2])

        pos_encoding = tf.concat([sine, cos], axis=-1)
        pos_encoding = tf.expand_dims(pos_encoding, 0)
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        """
        inputs: shape=(batch, sentence_length, emb_dim)
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class Encoder_Layer(Layer):

    def __init__(self, units, emb_dim, head, dropout_rate):

        self.self_attention = MultiHeadAttention(emb_dim=emb_dim, head=head, name="attention")
        self.dropout = Dropout(rate=dropout_rate)
        self.layernormalization = LayerNormalization(epsilon=1e-6)
        self.ff1 = Dense(units=units, activation='relu')
        self.ff2 = Dense(units=emb_dim)

    def __call__(self, inputs, padding_mask, name):
        # self multi head attention and dropout
        self_attention = self.self_attention(
            {
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': padding_mask
            }
        )
        self_attention = self.dropout(self_attention)

        # Add & Norm
        attention = self.layernormalization(inputs + self_attention)

        # feed forward
        ff = self.ff1(attention)
        ff = self.ff2(ff)
        ff = self.dropout(ff)

        # Add & Norm
        outputs = self.layernormalization(attention + ff)

        return outputs


class Encoder(Layer):

    def __init__(self, vocab_size, num_layers, units, emb_dim, head, dropout_rate):
        self.emb = Embedding(vocab_size, emb_dim)
        self.pos = PositionalEncoding(vocab_size, emb_dim)
        self.dropout = Dropout(rate=dropout_rate)

        self.encoder_layer = Encoder_Layer(
                      units=units,
                      emb_dim=emb_dim,
                      head=head,
                      dropout_rate=dropout_rate,
        )

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.units = units
        self.emb_dim = emb_dim
        self.head = head
        self.dropout_rate = dropout_rate

    def __call__(self, inputs, padding_mask, name):

        emb = self.emb(inputs)
        emb *= tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))
        emb = self.pos(emb)

        outputs = self.dropout(emb)

        for i in range(self.num_layers):
            outputs = self.encoder_layer(
                outputs, padding_mask,
                name="decoder_layer_{}".format(i)
                )

        return outputs


class Decoder_Layer(Layer):

    def __init__(self, units, emb_dim, head, dropout_rate):
        self.self_attention = MultiHeadAttention(emb_dim=emb_dim, head=head, name="attention_1")
        self.attention2 = MultiHeadAttention(emb_dim=emb_dim, head=head, name="attention_2")
        self.layernormalization = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(rate=dropout_rate)
        self.ff1 = Dense(units=units, activation='relu')
        self.ff2 = Dense(units=emb_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'self_attention': self.self_attention,
            'attention2': self.attention2,
            'layernormalization': self.layernormalization,
            'dropout': self.dropout,
            'ff1': self.ff1,
            'ff2': self.ff2,
        })
        return config

    def __call__(self, inputs, encoder_outputs, look_ahead_mask, padding_mask, name):

        # self multi head attention and dropout
        self_attention = self.self_attention(
            {
                'query': inputs,
                'key': inputs,
                'value': inputs,
                'mask': look_ahead_mask
            }
        )

        # Add & Norm
        attention1 = self.layernormalization(inputs + self_attention)
        attention2 = self.attention2(
            {
                'query': attention1,
                'key': encoder_outputs,
                'value': encoder_outputs,
                'mask': padding_mask
            }
        )
        attention2 = self.dropout(attention2)

        # Add & Norm
        attention = self.layernormalization(attention1 + attention2)

        # feed forward
        ff = self.ff1(attention)
        ff = self.ff2(ff)
        ff = self.dropout(ff)

        # Add & Norm
        outputs = self.layernormalization(attention + ff)

        return outputs


class Decoder(Layer):

    def __init__(self, vocab_size, num_layers, units, emb_dim, head, dropout_rate):
        self.emb = Embedding(vocab_size, emb_dim)
        self.pos = PositionalEncoding(vocab_size, emb_dim)
        self.dropout = Dropout(rate=dropout_rate)

        self.decoder_layer = Decoder_Layer(
                      units=units,
                      emb_dim=emb_dim,
                      head=head,
                      dropout_rate=dropout_rate,
        )

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.units = units
        self.emb_dim = emb_dim
        self.head = head
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'emb': self.emb,
            'pos': self.pos,
            'dropout': self.dropout,
            'decoder_layer': self.decoder_layer,
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'units': self.units,
            'emb_dim': self.emb_dim,
            'head': self.head,
            'dropout_rate': self.decoder_layer
        })
        return config

    def __call__(self, decoder_inputs, encoder_outputs, look_ahead_mask, decoder_padding_mask, name):
        """
        emb -> positional_encoding -> some decoder_layer
        inputs = Input(shape=(None,), name="inputs")
        encoder_outputs = Input(shape=(None, self.emb_dim), name="encoder_outputs")
        padding_mask = Input(shape=(1, 1, None), name="padding_mask")
        look_ahead_mask = Input(shape=(1, None, None), name='look_ahead_mask')
        """
        padding_mask = decoder_padding_mask

        emb = self.emb(decoder_inputs)
        emb *= tf.math.sqrt(tf.cast(self.emb_dim, tf.float32))
        emb = self.pos(emb)

        outputs = self.dropout(emb)

        for i in range(self.num_layers):
            outputs = self.decoder_layer(
                outputs, encoder_outputs, look_ahead_mask, padding_mask,
                name="decoder_layer_{}".format(i)
                )

        return outputs


class Make_Model():

    def __init__(self, max_length, emb_dim, vocab_size):

        self.max_length = max_length
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # エンコーダのinput
        self.input_ids = Input(shape=(max_length, ), dtype='int32', name='input_ids')
        self.attention_mask = Input(shape=(max_length, ), dtype='int32', name='attention_mask')
        self.token_type_ids = Input(shape=(max_length, ), dtype='int32', name='token_type_ids')
        self.inputs = [self.input_ids, self.attention_mask, self.token_type_ids]
        self.encoder_outputs = Dense(units=emb_dim, name='encoder_outputs')
        self.decoder_inputs = Input(shape=(None,), name="decoder_inputs")
        self.decoder_padding_mask = Lambda(create_padding_mask, output_shape=(1, 1, None), name="decoder_padding_mask")
        self.look_ahead_mask = Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')
        self.last_dense = Dense(units=vocab_size, name="outputs")

    def __call__(self, num_layers, units, head, dropout_rate, bert_frozen=True):

        # 層をfreeze(学習させないように)する. bertモデルはリストになっているので, 取り出す.
        self.bert.layers[0].trainable = not bert_frozen

        # エンコーダ(bert)
        x = self.bert.layers[0](self.inputs)
        encoder_outputs = self.encoder_outputs(x[0])

        # デコーダ(transformer)
        decoder_padding_mask = self.decoder_padding_mask(self.input_ids)
        look_ahead_mask = self.look_ahead_mask(self.decoder_inputs)
        decoder = Decoder(vocab_size=self.vocab_size,
                          num_layers=num_layers,
                          units=units,
                          emb_dim=self.emb_dim,
                          head=head,
                          dropout_rate=dropout_rate,
                          )
        decoder_outputs = decoder(self.decoder_inputs, encoder_outputs, look_ahead_mask, decoder_padding_mask, name="decoder")
        outputs = self.last_dense(decoder_outputs)

        return Model(inputs=[self.input_ids, self.attention_mask, self.token_type_ids, self.decoder_inputs], outputs=outputs)


class Make_Model_Transformer():

    def __init__(self, max_length, emb_dim, vocab_size):

        self.max_length = max_length
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size

        # input
        self.inputs = Input(shape=(None, ), name="inputs")
        self.decoder_inputs = Input(shape=(None,), name="decoder_inputs")

        # mask
        self.encoder_padding_mask = Lambda(create_padding_mask, output_shape=(1, 1, None), name="encoder_padding_mask")
        self.decoder_padding_mask = Lambda(create_padding_mask, output_shape=(1, 1, None), name="decoder_padding_mask")
        self.look_ahead_mask = Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')

        # dense
        self.last_dense = Dense(units=vocab_size, name="outputs")

    def __call__(self, num_layers, units, head, dropout_rate, bert_frozen=True):

        # デコーダ(transformer)
        encoder_padding_mask = self.encoder_padding_mask(self.inputs)
        decoder_padding_mask = self.decoder_padding_mask(self.inputs)
        look_ahead_mask = self.look_ahead_mask(self.decoder_inputs)

        encoder = Encoder(vocab_size=self.vocab_size,
                          num_layers=num_layers,
                          units=units,
                          emb_dim=self.emb_dim,
                          head=head,
                          dropout_rate=dropout_rate,
                          )
        decoder = Decoder(vocab_size=self.vocab_size,
                          num_layers=num_layers,
                          units=units,
                          emb_dim=self.emb_dim,
                          head=head,
                          dropout_rate=dropout_rate,
                          )

        encoder_outputs = encoder(self.inputs, encoder_padding_mask, name="encoder")
        decoder_outputs = decoder(self.decoder_inputs, encoder_outputs, look_ahead_mask, decoder_padding_mask, name="decoder")

        outputs = self.last_dense(decoder_outputs)

        return Model(inputs=[self.inputs, self.decoder_inputs], outputs=outputs)