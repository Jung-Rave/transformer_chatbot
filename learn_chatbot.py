import tensorflow as tf
import os

from model_transformer import Make_Model_Transformer
from function import preprocess_dataset, load_dataset_transformer, \
    CustomSchedule, Accuracy, Loss, Pred_Transformer, plot
import config


# ----------------定数設定----------------
NUM_LAYERS = config.NUM_LAYERS
EMB_DIM = config.EMB_DIM
NUM_HEADS = config.NUM_HEADS
UNITS = config.UNITS
DROPOUT_RATE = config.DROPOUT_RATE
MAX_LENGTH = config.MAX_LENGTH

EPOCHS = config.EPOCHS
TRAIN_BATCH = config.TRAIN_BATCH
VALID_BATCH = config.VALID_BATCH
TARGET_VOCAB_SIZE = config.TARGET_VOCAB_SIZE
TRAIN_SIZE = config.TRAIN_SIZE

path = config.path


if __name__ == "__main__":


# ----------------前処理---------------

    # データの前処理を行う.
    tokenizer, START_TOKEN, END_TOKEN, VOCAB_SIZE, input_data, output_data = preprocess_dataset(TARGET_VOCAB_SIZE)
    # transformerで使用できるように整形する.
    train_dataset, valid_dataset = load_dataset_transformer(MAX_LENGTH, TRAIN_BATCH, VALID_BATCH, TRAIN_SIZE, input_data, output_data, tokenizer, START_TOKEN, END_TOKEN)

    print("prepro finished.")


# ----------------モデル作成----------------

    # 学習率を設定する.
    learning_rate = CustomSchedule(EMB_DIM)
    # 最適化アルゴリズムを設定する.
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # 損失関数を設定する.
    loss = Loss(MAX_LENGTH)
    # 最適化関数を設定する.
    accuracy = Accuracy('accuracy', MAX_LENGTH)

    # モデルを設定する.
    make_model = Make_Model_Transformer(max_length=MAX_LENGTH, emb_dim=EMB_DIM, vocab_size=VOCAB_SIZE)
    model = make_model(num_layers=NUM_LAYERS, units=UNITS, head=NUM_HEADS, dropout_rate=DROPOUT_RATE)

    # モデルをコンパイルする.
    model.compile(optimizer=optimizer, loss=loss, metrics=[accuracy])

    # モデルの形状を見る.
    # model.summary()
    # モデルの図を出力する.
    # tf.keras.utils.plot_model(model)
    # 重みをロードする
    if config.LOAD_WEIGHT == True:
        model.load_weights(os.path.join(path, config.load_filename))

    print("model finished.")


# ----------------学習----------------

    # callbacksの設定.
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path, config.callback_filename),
                                           monitor='val_loss',
                                           verbose=1,
                                           save_best_only=True,
                                           save_weights_only=True,
                                           mode='min',
                                           save_epoc=1
                                           ),
        Pred_Transformer(MAX_LENGTH, tokenizer, START_TOKEN, END_TOKEN)
    ]

    # 学習を行う.
    history = model.fit(train_dataset, epochs=EPOCHS, validation_data=valid_dataset, callbacks=callbacks, steps_per_epoch=None)

    # 結果のプロット(訓練用と評価用)
    plot(EPOCHS, history)

    # 重みを保存する.
    model.save_weights(os.path.join(path, config.save_filename), save_format='tf')

    # 終了メッセージ
    print("learning finished.")