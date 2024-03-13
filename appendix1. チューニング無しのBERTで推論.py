# Databricks notebook source
# MAGIC %md
# MAGIC # PART 1. Hugging Face トランスフォーマーベースのテキスト分類モデルを動かす
# MAGIC このノートブックは、"tohoku-nlp/bert-base-japanese-v3"をベースモデルとした文章分類モデルをシングルGPUマシンで動かします。
# MAGIC [Transformers](https://huggingface.co/docs/transformers/index)ライブラリを使います。
# MAGIC
# MAGIC ## クラスタのセットアップ
# MAGIC このノートブックでは、AWS の `g4dn.xlarge` や Azure の `Standard_NC4as_T4_v3` のようなシングル GPU クラスタを推奨します。シングルマシンクラスタの作成](https://docs.databricks.com/clusters/configure.html) は、パーソナルコンピュートポリシーを使用するか、クラスタ作成時に "Single Node" を選択することで可能です。このノートブックはDatabricks Runtime ML GPUバージョン14.3 LTSで動作確認しております。

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. ライブラリのインストール
# MAGIC Databricks Runtime ML には `transformers` ライブラリがデフォルトでインストールされています。このノートブックでは他にもいくつかライブラリが必要で、これらは `%pip` を使ってインストールします。

# COMMAND ----------

# MAGIC %pip install accelerate fugashi unidic-lite
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. BERTを動かしてみる

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-1. ノートブックのパラメータを設定する。
# MAGIC - `base_model`は東北大学から提供されている新しいBERTモデルである[tohoku-nlp/bert-base-japanese-v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3)を指定します。このノートブックではモデルを動かしていきます。
# MAGIC - `max_length`は、BERTに投入するトークン長の最大値です。

# COMMAND ----------

# HuggingFace モデル名。「東北大が公開しているBertモデル」を利用します。
base_model = "tohoku-nlp/bert-base-japanese-v3"

# BERTに投入する文章の長さの最大値
max_length = 128

# COMMAND ----------

# MAGIC %md
# MAGIC ## トークナイザーのダウンロードと実行
# MAGIC モデルに入力するトークン配列を文章データから作成する日本語対応のトークナイザーをダウンロードします。

# COMMAND ----------

from transformers import AutoTokenizer

#日本語をBERTに投入できるようにトークン化する関数をインスタンス化します。
tokenizer = AutoTokenizer.from_pretrained(base_model)

#日本語トークナイザーを用いて日本語文章を変換
exp_string = 'このハンズオンを受けて本当に良かった！'
encoded_tokens = tokenizer(
    text=exp_string,
    max_length=max_length,                  #BERTに入力する最終的なinput_idsの長さ
    padding='max_length', truncation=True,  #input_idsの長さをmax_lengthに調整するための引数
    return_tensors='pt',                    #出力データをpytorchの形式で返す(haggingfaceはtorchで実装されているため)
)

encoded_tokens                              #返り値は辞書型で"input_ids", "token_type_ids", "attention_mask"のkeyを持つ

# COMMAND ----------

# MAGIC %md
# MAGIC トークンIDから対応する文字に変換して出力してみます。文章がどのように区切られたか具体的なイメージを持てると思います。

# COMMAND ----------

print(tokenizer.convert_ids_to_tokens(encoded_tokens['input_ids'][0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## BERTモデルのダウンロード
# MAGIC 続いてBERTモデルをダウンロードします。その後、モデルのコンフィグを表示します。例えば、以下のパラメータに着目してみましょう。
# MAGIC - `max_position_embeddings`: 512 → 入力できる最大の文章の長さ
# MAGIC - `hidden_size`: 768　→ 出力における、各tokenの次元数

# COMMAND ----------

import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
  base_model,                                                 #東北大の事前学習済みの重みを用いてBERTをインスタンス化
  id2label = {0: "Positive", 1: "Negative"}                   #後の出力用にIDとラベルのマッピングを設定
)  
model.config                                                  #インスタンス化したBERTモデルの詳細を確認

# COMMAND ----------

# DBTITLE 1,BERTで推論実行
#トークン化した文章をBERTに投入し、文脈が練り込まれた行列に変換
output = model(**encoded_tokens, output_hidden_states=True)

# Softmaxして、確率に変換
import torch.nn.functional as F
probabilities = F.softmax(output.logits, dim=1)
probabilities

# COMMAND ----------

output

# COMMAND ----------

# DBTITLE 1,最終隠れ層から各トークンのベクトルを得る
#output.hidden_states[-1]　には入力したtokenに対応する行列が格納
print(output.hidden_states[-1].shape)                 #（[文章数, tokenの長さ, 各tokenの次元数]）に対応しています。
print(output.hidden_states[-1])

# "素晴らしい"　というトークンのベクトルはこちら
print(output.hidden_states[-1][0][1])

# COMMAND ----------

# DBTITLE 1,トークナイザー→モデルというパイプラインを作って実行
from transformers import pipeline

bert_pipeline = pipeline(
    "text-classification", 
    tokenizer=tokenizer,
    model=model, 
    batch_size=1, 
    top_k=2,
    device=(0 if torch.cuda.is_available() else -1))

bert_pipeline(exp_string)

# COMMAND ----------

# MAGIC %md
# MAGIC ## お疲れ様でした！
