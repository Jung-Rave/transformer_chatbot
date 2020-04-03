import tensorflow as tf
import numpy as np

import MeCab
import re
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

from multiprocessing import Pool

import config


# ----------------prepro----------------

# 前処理したデータ等を保存する.
def save_prepro_dataset(prepro_dataset, file_name):
  with open(file_name+".pickle", 'wb') as handle:
      pickle.dump(prepro_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)


# 前処理したデータ等をロードする.
def load_prepro_dataset(file_name):
  with open(file_name+".pickle", 'rb') as handle:
      return pickle.load(handle)


# Mecabで分かち書き.
def preprocess_sentence(sentence):
    tagger = MeCab.Tagger("-Owakati")
    parser = tagger.parse(sentence)
    return re.sub('\n', '', parser)


# データの前処理を行う.
def preprocess_dataset(TARGET_VOCAB_SIZE):

    # 既に前処理したデータを使用する場合.
    if config.LOAD_PREPROCESS_DATASET_FLAG == True:
        print("Now Loading...")
        prepro_dataset = load_prepro_dataset(config.prepro_filename)
        tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE, input_data, output_data \
            = prepro_dataset[0], prepro_dataset[1], prepro_dataset[2], prepro_dataset[3], prepro_dataset[4], prepro_dataset[5]

    # 初めて前処理する場合.
    else:
        # ファイルを読み込む.
        print("data mapping...")
        with open(config.input_filename, 'r', encoding='utf-8') as f:
            input_data = f.read().split('\n')
        with open(config.output_filename, 'r', encoding='utf-8') as f:
            output_data = f.read().split('\n')

        # わかち書きなどの前処理を行う.
        with Pool(16) as pool:
            input_data = list(pool.map(preprocess_sentence, input_data))
            output_data = list(pool.map(preprocess_sentence, output_data))

        # tokenizerを作成する.
        print("tokenizer making...")
        # 最初から作る場合
        tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(input_data + output_data, TARGET_VOCAB_SIZE)
        # 既にあるTokenizerを使う場合
        # tokenizer = tfds.features.text.SubwordTextEncoder(input_data + output_data)
        # tokenizer = tokenizer.load_from_file(filename_prefix="tokenizer_save_to_file_60000_touhou2_mix")

        # 文字列の最初と最後に固有のIDを定義
        START_TOKEN, END_TOKEN = tokenizer.vocab_size, tokenizer.vocab_size + 1

        # ボキャブラリサイズに最初と最後を表すIDの数を足す
        VOCAB_SIZE = tokenizer.vocab_size + 2
        # VOCAB_SIZE = 60001

        # 前処理部分を保存する.
        prepro_dataset = [tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE, input_data, output_data]
        save_prepro_dataset(prepro_dataset, config.prepro_filename)
        # tokenizerの保存
        save_text_tokenizer(tokenizer, config.tokenizer_filename)
        # TOKENとVACAB_SIZEの保存
        save_SEV(START_TOKEN, END_TOKEN, VOCAB_SIZE, config.SEV_filename)

    return tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE, input_data, output_data


# データをトークン化する.
def tokenize_and_filter(max_length, inputs, outputs, tokenizer, START_TOKEN, END_TOKEN):
    tokenized_inputs, tokenized_outputs = [], []
    to_inp_a, to_out_a = tokenized_inputs.append, tokenized_outputs.append

    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenize sentence
        sentence1 = [START_TOKEN] + tokenizer.encode(sentence1) + [END_TOKEN]
        sentence2 = [START_TOKEN] + tokenizer.encode(sentence2) + [END_TOKEN]
        # max sentence lengthをこえていないデータだけ使用
        if len(sentence1) <= max_length and len(sentence2) <= max_length:
            to_inp_a(sentence1)
            to_out_a(sentence2)

    # token化されたsentenceを0埋めする
    tokenized_inputs = pad_sequences(tokenized_inputs, maxlen=max_length, padding='post')
    tokenized_outputs = pad_sequences(tokenized_outputs, maxlen=max_length, padding='post')

    return tokenized_inputs, tokenized_outputs


# データをモデルの入力に合わせて整形する.
def load_dataset_transformer(max_length, train_batch, val_batch, train_size, input_data, output_data, tokenizer, START_TOKEN, END_TOKEN):

    # Prepare dataset for BERT as a tf.data.Dataset instance
    def split_train_test(data, TRAIN_SIZE: int, BUFFER_SIZE: int, SEED=123):
        valid_data = data.skip(TRAIN_SIZE)
        train_data = data.take(TRAIN_SIZE).shuffle(BUFFER_SIZE, seed=SEED)
        return train_data, valid_data

    input, output = tokenize_and_filter(max_length, input_data, output_data, tokenizer, START_TOKEN, END_TOKEN)
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': input,
            'decoder_inputs': output[:, :-1]
        },
        {
            'outputs': output[:, 1:]
        },
    ))

    # SIZE関連は, なんとなくです.
    train_dataset, valid_dataset = split_train_test(dataset, TRAIN_SIZE=train_size, BUFFER_SIZE=train_size)

    # Prepare dataset for BERT as a tf.data.Dataset instance
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(train_size)
    train_dataset = train_dataset.batch(train_batch)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # validation dataにも同じ処理.
    valid_dataset = valid_dataset.cache()
    valid_dataset = valid_dataset.batch(val_batch)
    valid_dataset = valid_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, valid_dataset

# ----------------prepro----------------


# ----------------model----------------

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
        })
        return config

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps**-1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# 損失関数
class Loss():

    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, y_true, y_pred, sample_weight):

        # 多分, [1:]で教師データの最初のトークンを無視しているからMAX_LENGTH-1になっていた.
        y_true = tf.reshape(y_true, shape=(-1, self.max_length-1))
        loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')(y_true, y_pred)
        # y_true!=0 -> 1.0, y_true==0 -> 0.0
        # に変換し、lossに乗算し、y_true=0部分のlossをすべて0.0にする

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = tf.multiply(loss, mask)

        return tf.reduce_mean(loss)


# 正確さ
class Accuracy():

    def __init__(self, name, max_length):
        self.name = name
        self.max_length = max_length
        self.y_true = None
        self.y_pred = None
        # SparseCategoricalAccurancyは, ラベルがone hotエンコーディングされていない場合に使う.
        self.sparsecategoricalaccuracy = tf.metrics.SparseCategoricalAccuracy()

    def __call__(self, y_true, y_pred):
        # ensure labels have shape (batch_size, MAX_LENGTH - 1)
        y_true = tf.reshape(y_true, shape=(-1, self.max_length-1))
        accuracy = self.sparsecategoricalaccuracy(y_true, y_pred)
        return accuracy

# ----------------model----------------


# ----------------prediction----------------

# 評価するために結果を出力する.
def evaluate_transformer(sentence, tokenizer, model, max_length, START_TOKEN, END_TOKEN):

    inputs = [START_TOKEN] + tokenizer.encode(sentence) + [END_TOKEN]

    def padding_list(list, max_length):
        list = list + [0] * (max_length-len(list))
        return list

    input_ids = tf.expand_dims(padding_list(inputs, max_length), 0)
    output = tf.expand_dims([START_TOKEN], 0)

    # modelがEND_TOKENを予測するまで繰り返しpredictさせる = 文章の終わりまで
    for i in range(max_length-1):

        inputs_list = [input_ids, output]
        predictions = model(inputs_list, training=False)

        # 最後の単語を取得
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


# エポックごとに推論結果を表示する.
class Pred_Transformer(tf.keras.callbacks.Callback):
    def __init__(self, max_length, tokenizer, START_TOKEN, END_TOKEN, validation_data=(), interval=10):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.output_log = 'output_log.txt'

    def on_epoch_end(self, epoch, logs={}):
        sample = [
            'こんにちは',
            'ビール飲みたい',
            '大学は退学するべき'
        ]
        for sentence in sample:
            self._pred(sentence)

    def _pred(self, sentence):

        prediction = evaluate_transformer(sentence, self.tokenizer, self.model, self.max_length, self.START_TOKEN, self.END_TOKEN)

        # 最初と末尾を除外
        prediction = np.delete(prediction, 0)
        prediction = np.delete(prediction, -1)

        # デコーダ
        predicted_sentence = self.tokenizer.decode(
            [i for i in prediction if i < self.tokenizer.vocab_size])

        predicted_sentence = predicted_sentence.replace(' ', '')

        print('Input : {}'.format(sentence))
        print('Output: {}'.format(predicted_sentence))

# ----------------prediction----------------


# ----------------plot----------------

def plot(EPOCHS, history):

    plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))

    plt.plot(range(1, EPOCHS + 1), history.history["accuracy"], label="training")
    plt.plot(range(1, EPOCHS + 1), history.history["val_accuracy"], label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(range(1, EPOCHS + 1), history.history["loss"], label="training")
    plt.plot(range(1, EPOCHS + 1), history.history["val_loss"], label="validation")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# ----------------plot----------------


# ----------------save----------------

def save_text_tokenizer(tokenizer, file_name):
    print("tokenizer saving...")
    with open(file_name+".pickle", 'wb') as handle:
      pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def save_SEV(START_TOKEN, END_TOKEN, VOCAB_SIZE, file_name):
    print("SEV saving...")
    number = [START_TOKEN, END_TOKEN, VOCAB_SIZE]
    with open(file_name+".pickle", 'wb') as handle:
      pickle.dump(number, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ----------------save----------------