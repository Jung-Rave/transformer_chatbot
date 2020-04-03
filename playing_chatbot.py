import tensorflow as tf
import numpy as np

import os
import pickle
import config

from model_transformer import Make_Model_Transformer


# トークナイザをロードする.
def load_text_tokenizer(file_name):
    with open(file_name + ".pickle", 'rb') as handle:
      return pickle.load(handle)

# START_TOKEN, END_TOKEN, VOCAB_SIZEをロードする.
def load_SEV(file_name):
    with open(file_name + ".pickle", 'rb') as handle:
      return pickle.load(handle)


# 入力された文字列に対して, 文字列を出力する.
def evaluate_transformer_conversation(sentence, model, max_length):

    inputs = [START_TOKEN] + tokenizer.encode(sentence) + [END_TOKEN]

    def padding_list(list, max_length):
        list = list + [0] * (max_length-len(list))
        return list

    input_ids = tf.expand_dims(padding_list(inputs, max_length), 0)
    output = tf.expand_dims([START_TOKEN], 0)

    # modelがEND_TOKENを予測するまで繰り返しpredictさせる = 文章の終わりまで
    for i in range(max_length-1):

        # 入力データの処理
        inputs_list = [input_ids, output]
        predictions = model(inputs_list, training=False)

        # 出力データの処理
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
        if i == max_length-2:
            output = tf.concat([output, tf.expand_dims([END_TOKEN], 0)], -1)
        else:
            output = tf.concat([output, predicted_id], -1)

        # predicted_idがEND_TOKENと一致したら予測を終了
        if tf.equal(predicted_id, tf.cast(END_TOKEN, tf.int32)):
            break

    return tf.squeeze(output, axis=0).numpy()


def conversation(model, max_length):

    while True:

        # 入力する.
        sentence = input('貴方: ')

        # 入力に対して出力する.
        prediction = evaluate_transformer_conversation(sentence, model, max_length)

        # 最初と末尾を除外する.
        prediction = np.delete(prediction, 0)
        prediction = np.delete(prediction, -1)

        # デコーダで翻訳する.
        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size])

        # 分かち書きのスペースを取り除きつつ, 応答を表示する.
        print("応答: " + str(predicted_sentence.replace(' ', '')))


print("Now Loading...")

# 定数設定
NUM_LAYERS = config.NUM_LAYERS
EMB_DIM = config.EMB_DIM
NUM_HEADS = config.NUM_HEADS
UNITS = config.UNITS
DROPOUT_RATE = config.DROPOUT_RATE
MAX_LENGTH = config.MAX_LENGTH
NUMBER = load_SEV(config.SEV_filename)
START_TOKEN, END_TOKEN, VOCAB_SIZE = NUMBER[0], NUMBER[1], NUMBER[2]

# トークナイザをロードする.
tokenizer = load_text_tokenizer(config.tokenizer_filename)

# モデルを設定する.
make_model = Make_Model_Transformer(max_length=MAX_LENGTH, emb_dim=EMB_DIM, vocab_size=VOCAB_SIZE)
model = make_model(num_layers=NUM_LAYERS, units=UNITS, head=NUM_HEADS, dropout_rate=DROPOUT_RATE)

# 重みをロードする.
model.load_weights(os.path.join(config.path, config.load_filename_playing))


if __name__ == "__main__":

    # 無限に会話する.
    conversation(model, MAX_LENGTH)