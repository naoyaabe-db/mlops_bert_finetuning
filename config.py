# Databricks notebook source
################################################
# ユーザー毎に変更が必要なパラメーター
################################################

# Unity Catalog カタログ名
catalog_name = "nabe_bert_catalog"

# Unity Catalog スキーマ名
schema_name = "nabe_bert_schema"

# モデルレジストリーに登録するモデル名
registered_model_name = "nabe_bert_model_ja"

# モデルが記録されているMLFlow実験のRun ID
run_id = "5833bd8fde6d4ad3a3a5c0b335561c8c"

# サービングエンドポイントの名前
endpoint_name = 'nabe_bert_endpoint'

################################################
# 変更不要なパラメーター
################################################

# Unity Catalog テーブル名
table_name = "raw_documents"

# Unity Catalog VOLUME名 (ファイルの保存場所)
volume_name = "raw_files_vol"

# HuggingFace モデル名
base_model = "tohoku-nlp/bert-base-japanese-v3"

# BERTに投入する文章の最大長
max_length = 128

# mlflowで実験結果を記録するエクスペリメント名
experiment_name = "nabe_exp"

# MLFlow Trackingに記録されているモデル名（アーティファクト名）
model_artifact_path = "bert_model_ja"

# テーブル以外のファイルを保存するVolumeのパス
volume_path = f'/Volumes/{catalog_name}/{schema_name}/{volume_name}'

# データを格納するためのテーブル名（catalog_name.schema_name.table_name）
table_name_full_path = f"{catalog_name}.{schema_name}.{table_name}"

# モデルをMLFLow Model Registerへ登録する際に名前（catalog_name.schema_name.model_name）
registered_model_full_path = f"{catalog_name}.{schema_name}.{registered_model_name}"

# ステージングのモデルサービングエンドポイント
stg_serving_endpoint_name=f"stg_{endpoint_name}"

# プロダクションのモデルサービングエンドポイント
prod_serving_endpoint_name=f"prod_{endpoint_name}"

# カテゴリーのリスト
category_list = [
    'dokujo-tsushin',
    'it-life-hack',
    'kaden-channel',
    'livedoor-homme',
    'movie-enter',
    'peachy',
    'smax',
    'sports-watch',
    'topic-news'
]

# COMMAND ----------


