# Databricks notebook source
# MAGIC %md
# MAGIC # データを取り込み、テーブル化する
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/a0600992b2d5909cecde14bec7dcc1c0f5a3948c/mlops_bert_images/mlops_bert_01.png?raw=true" style="float: right" width="1000px">
# MAGIC <br/>

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. ライブラリのインストールとインポート

# COMMAND ----------

# MAGIC %pip install fugashi unidic-lite accelerate
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,パラメーターの設定
# MAGIC %run ./config

# COMMAND ----------

# DBTITLE 1,Unity Catalogにカタログ、スキーマ、ボリュームを無ければ作る
# Catalogを新規に作成し、デフォルトで使用されるよう設定
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")

# Schemaを新規に作成し、デフォルトで使用されるよう設定
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# テーブル以外のファイルを格納するためのVolumeを作成
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog_name}.{schema_name}.{volume_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. ダウンロードしたデータの取り込み
# MAGIC ダウンロードしたデータをSpark DataFrameとしてロードし、Delta TableとしてUnity Catalogへ保存します。
# MAGIC
# MAGIC 今回用いるデータセット[livedoor ニュースコーパス](https://www.rondhuit.com/download/ldcc-20140209.tar.gz)は、[Rondhuitのサイト](https://www.rondhuit.com/download.html)から入手できます。

# COMMAND ----------

# DBTITLE 1,適当なファイルの内容を確認（最初の3行はメタデータで、4行目から本文）
with open(f"{volume_path}/text/it-life-hack/it-life-hack-6342280.txt", "r") as f:
  print(f.read())

# COMMAND ----------

# DBTITLE 1,ファイル名とクラスの一覧をPandas Dataframeとして作成
import glob
import pandas as pd
import json

# 各データの形式を整える
columns = ['label', 'label_name', 'file_path']
dataset_label_text = pd.DataFrame(columns=columns)
id2label = {} 
label2id = {}
for label, category in enumerate(category_list):
  
  #対象メディアの記事が保存されているファイルのlistを取得します。
  file_names_list = sorted(glob.glob(f'{volume_path}/text/{category}/{category}*'))
  print(f"{category}の記事を処理しています。　{category}に対応する番号は{label}で、データ個数は{len(file_names_list)}です。")

  id2label[label] = category
  label2id[category] = label
  
  for file in file_names_list:
      list = [[label, category, file]]
      df_append = pd.DataFrame(data=list, columns=columns)
      dataset_label_text = pd.concat([dataset_label_text, df_append], ignore_index=True, axis=0)

# id2labelとlabel2idは後でファインチューニング時に使用するため、jsonファイルとしてvolumeに記録
with open(f'{volume_path}/id2label.json', 'w') as f:
    json.dump(id2label, f, indent=2)
with open(f'{volume_path}/label2id.json', 'w') as f:
    json.dump(label2id, f, indent=2)

dataset_label_text.head()

# COMMAND ----------

# DBTITLE 1,ファイルの内容をロードする関数（Pandas UDF）
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType

# 以下のようにPandasUDFを使用すると、Pythonで表現したロジックをSpark上で実行できるようになる
@pandas_udf(StringType())
def read_text(paths: pd.Series) -> pd.Series:

  all_text = []
  for index, file in paths.items():             #取得したlistに従って実際のFileからデータを取得します。
      lines = open(file).read().splitlines()
      text = '\n'.join(lines[3:])               # ファイルの4行目からを抜き出す。
      all_text.append(text)

  return pd.Series(all_text)

# COMMAND ----------

# DBTITLE 1,Pandas DataframeからSpark Dataframeに変換し、ファイルの内容を新規列として追加
from pyspark.sql.functions import col

dataset_df = spark.createDataFrame(dataset_label_text)
# 上で作成したPandasUDFを適用
dataset_df = dataset_df.withColumn('text', read_text(col('file_path')))
display(dataset_df.head(5))

# COMMAND ----------

# DBTITLE 1,Spark DataframeをDelta TableとしてUnity Catalogへ保存
dataset_df.write.mode("overwrite").saveAsTable(table_name_full_path)
