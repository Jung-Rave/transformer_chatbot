# ----------------定数設定----------------
# ほぼ論文通り
NUM_LAYERS = 2
EMB_DIM = 256
NUM_HEADS = 8
UNITS = 512
DROPOUT_RATE = 0.1
TRAIN_BATCH = 64
VALID_BATCH = 64

# データセットによって変更する
EPOCHS = 1500
MAX_LENGTH = 64
TARGET_VOCAB_SIZE = 10000
TRAIN_SIZE = 1936

# ファイルがあるパス
path = "ファイルがあるパス"

# データセット
input_filename = "入力データセット"
output_filename = "出力データセット"

# playing用の保存するファイル名
tokenizer_filename = "トークナイザのpickleファイル"
SEV_filename = "SEVのpickleファイル"
load_filename_playing = "保存した重み"

# 既に前処理したデータファイル
LOAD_PREPROCESS_DATASET_FLAG = False
prepro_filename = "前処理したpickleファイル"

# 保存した重み / 事前学習した重み
LOAD_WEIGHT = False
load_filename = "保存した重み/事前学習した重み"

# 保存する重みのファイル名
callback_filename = "エポックごとに保存する重み"
save_filename = "最終的に保存する重み"
