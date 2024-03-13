# Databricks notebook source
# MAGIC %md
# MAGIC # サンプルデータのダウンロード・クラウドストレージへの保存
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/a0600992b2d5909cecde14bec7dcc1c0f5a3948c/mlops_bert_images/mlops_bert_00.png?raw=true" style="float: right" width="1000px">
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

# MAGIC %md
# MAGIC ## 1-1. データのダウンロードとロード
# MAGIC まずデータセットをダウンロードします。その後、Spark DataFrameとしてロードし、Delta TableとしてUnity Catalogへ保存します。
# MAGIC
# MAGIC 今回用いるデータセット[livedoor ニュースコーパス](https://www.rondhuit.com/download/ldcc-20140209.tar.gz)は、[Rondhuitのサイト](https://www.rondhuit.com/download.html)から入手できます。

# COMMAND ----------

# DBTITLE 1,ファイルのダウンロード
import requests

url = 'https://www.rondhuit.com/download/ldcc-20140209.tar.gz'
tar_file_path = f'{volume_path}/ldcc-20140209.tar.gz'

urlData = requests.get(url).content

with open(tar_file_path, mode='wb') as f: # wb でバイト型を書き込める
  f.write(urlData)

# COMMAND ----------

# DBTITLE 1,ダウンロードしたファイルを解凍
import tarfile

tar = tarfile.TarFile(tar_file_path)
tar.extractall(path=volume_path)
tar.close()
