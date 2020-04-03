# transformer_chatbot
## 概要
博麗霊夢さん＆琴葉茜ちゃんと会話してみた①(Transformerとキャラ対話データを用いた転移学習による対話botの作成)で紹介した内容のソースコードです.

## ソースコード
### config.py
ハイパーパラメータやファイル名を格納する定数が記載されているコードです.
### function.py
前処理, 推論, 保存, 読み込み, プロットなどの関数やクラスが記載されているコードです.
### learn_chatbot.py
学習を行うコードです.
### model_transformer.py
transformerモデルの構築を行うコードです.
### playing_chatbot.py
推論のみを行うコードです.

## 使用上の注意
tensoflow, tensorflow-gpu, Mecabのインストールが必要です. 
また, 開発環境はPython 3.6.2, Tensorflow 2.0.0です.
学習データ等は権利の関係で添付していませんので, 予めご了承下さい.

## 最後に
丸投げで申し訳ないのですが, 記事の中で紹介したリンク先が詳しいので, そちらを参照して頂ければ幸いです.
