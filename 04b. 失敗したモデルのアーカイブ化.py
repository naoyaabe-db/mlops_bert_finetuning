# Databricks notebook source
# MAGIC %md
# MAGIC # (モデル検証失敗) ステージングへのモデルのデプロイ・サービング
# MAGIC
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/a0600992b2d5909cecde14bec7dcc1c0f5a3948c/mlops_bert_images/mlops_bert_04.png?raw=true" style="float: right" width="1000px">
# MAGIC <br/>

# COMMAND ----------

# MAGIC %md
# MAGIC # 0. パラメーター設定

# COMMAND ----------

# DBTITLE 1,パラメーターの設定
# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. 検証に失敗したモデルのタグを変更
# MAGIC この操作はGUIからも実施可能ですが、再現性の確保のためコードベースの手順を記載します。

# COMMAND ----------

from mlflow import MlflowClient
client = MlflowClient()

model_version = client.get_model_version_by_alias(registered_model_full_path, "Candidate").version

# 検証をクリアしたモデルバージョンから、エイリアス（Candidate）を除外
client.delete_registered_model_alias(
  name=registered_model_full_path, 
  alias="Candidate"
)
client.set_registered_model_alias(
  name=registered_model_full_path, 
  alias="LastFaild", 
  version=model_version
)

# 同じバージョンに対して、validation_statusのタグの値をValidatedに変更
client.set_model_version_tag(registered_model_full_path, model_version, "validation_status", "Failed")
