# Databricks notebook source
# MAGIC %md
# MAGIC # Hugging Face トランスフォーマーベースのテキスト分類モデルをサービングする
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/a0600992b2d5909cecde14bec7dcc1c0f5a3948c/mlops_bert_images/mlops_bert_03.png?raw=true" style="float: right" width="1000px">
# MAGIC <br/>

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. ライブラリのインストール＆パラメーター設定

# COMMAND ----------

# MAGIC %pip install datasets evaluate fugashi unidic-lite accelerate
# MAGIC %pip install databricks-sdk==0.12.0 mlflow[genai] --upgrade --q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,パラメーターの設定
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. モデルの検証を実行
# MAGIC この操作はGUIからも実施可能ですが、再現性の確保のためコードベースの手順を記載します。

# COMMAND ----------

# DBTITLE 1,Model Registryからモデルを関数としてダウンロードして推論実行
import mlflow
import pandas as pd
import torch
from numba import cuda

device = cuda.get_current_device()
device.reset()

# Model Registryからモデルをダウンロード
loaded_model = mlflow.pyfunc.load_model(
  f"models:/{registered_model_full_path}@Candidate"
)

# COMMAND ----------

# DBTITLE 1,テスト用文章データ（生成AIで適当に作ったもの）
# テスト用の文章データ
inputs = [
    """
オーシャンシティに所属するネオランド代表FWアレックス・スターマンが、全体トレーニングに復帰した。13日、クラブ公式サイトが伝えている。
オーシャンシティを離れ、ネオランド代表でのミスティックコンチネンツカップに参戦していたスターマンは、先月18日に行われた第2節のサンライト代表戦（△2－2）で左太もも裏を負傷してしまい、前半アディショナルタイムに途中交代を余儀なくされ、チームを離脱してオーシャンシティに復帰していた。
    """,
    """
M3シリーズは、8コアのCPUと10コアのGPUを備え、最大で24GBの統合メモリをサポートしています。M3 Proモデルでは、CPUを最大12コア、GPUを最大18コアまで選択可能で、統合メモリは最大で36GBです。一方、M3 MaxではCPUを最大で16コア、GPUを最大で40コア、統合メモリを最大で128GBまで選択できます。
これら全モデルに共通して、AV1コーデックのデコードをサポートする内蔵メディアエンジンを搭載しており、HEVC、H.264、ProResなどの様々なフォーマットの再生が可能です。さらに、Dolby Vision、HDR 10、HLGといった高ダイナミックレンジフォーマットにも対応しています。
    """,
    """
　しかし、恋愛において受動的な姿勢から始めることが、楽だと感じてしまうと、関係が始まってから平等な信頼関係を構築するのが難しくなりがちです。男性がリードし、女性がそれに従うという関係が定着すると、女性は徐々に「嫌われたくない」という思いから自らの意見を述べにくくなります。そして、場合によっては、男性が支配的な態度をとるようになり、そのような関係から抜け出すのが困難になることもあるのです。
    """,
    """
21世紀ピクチャーズが、人気SFアクション『ハンターズ』シリーズの新作映画『ダークフィールド（原題） / Darkfield』の製作を進めていると、The Global Film Gazette などが報じた。
　監督は前作『ハンターズ：ザ・クエスト』（2022）と同じくジョン・ドーヴァーが務めるが、同サイトによると、新作は『ザ・クエスト』の続編にはならないとのこと。詳細は不明だが、未来を舞台に、『ザ・クエスト』と同じく女性が主人公になる。
    """
]

# COMMAND ----------

# DBTITLE 1,サンプルテキストで推論が実行できるかテスト
# ロードされたモデルを使って推論実行
input_example = pd.DataFrame({"text": inputs})
response = loaded_model.predict(input_example)
response

# COMMAND ----------

# MAGIC %md
# MAGIC #### 本来は単に推論が実行できるかどうかだけでなく、入出力のデータフォーマットの検証、テストデータを元にした精度のチェック等をこのNotebook上で行います。

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

model_version = client.get_model_version_by_alias(registered_model_full_path, "Candidate").version

# 検証をクリアしたモデルバージョンに対して、エイリアスをCandidateからStagingに変更
client.delete_registered_model_alias(
  name=registered_model_full_path, 
  alias="Candidate"
)
client.set_registered_model_alias(
  name=registered_model_full_path, 
  alias="Staging", 
  version=model_version
)

# 同じバージョンに対して、validation_statusのタグの値をValidatedに変更
client.set_model_version_tag(registered_model_full_path, model_version, "validation_status", "Validated")
