# Databricks notebook source
# MAGIC %md
# MAGIC # Hugging Face でテキスト分類モデルをファインチューニング
# MAGIC <br/>
# MAGIC <img src="https://github.com/naoyaabe-db/public_demo_images/blob/a0600992b2d5909cecde14bec7dcc1c0f5a3948c/mlops_bert_images/mlops_bert_02.png?raw=true" style="float: right" width="1000px">
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
# MAGIC # 1. BERTのファインチューニング

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-2. データ加工
# MAGIC
# MAGIC Spark DataframeからHugging Face Datasetsへ変換をし、text列（文章データ）の内容をトークン化して、新たな列として追加します。
# MAGIC
# MAGIC Hugging Face の `datasets` は `datasets.Dataset.from_spark` を使って Spark DataFrames からのロードをサポートしています。[from_spark()](https://huggingface.co/docs/datasets/use_with_spark) メソッドの詳細については Hugging Face のドキュメントを参照してください。また、Databricksの[こちらのドキュメント](https://docs.databricks.com/ja/machine-learning/train-model/huggingface/load-data.html)も参考にしてください。
# MAGIC
# MAGIC Dataset.from_sparkはデータセットをキャッシュします。この例では、モデルはドライバ上で学習され、キャッシュされたデータはSparkを使用して並列化されるため、`cache_dir`はドライバとすべてのワーカーからアクセス可能でなければなりません。Databricks File System (DBFS)のルート([AWS](https://docs.databricks.com/dbfs/index.html#what-is-the-dbfs-root)|[Azure](https://learn.microsoft.com/azure/databricks/dbfs/#what-is-the-dbfs-root)|[GCP](https://docs.gcp.databricks.com/dbfs/index.html#what-is-the-dbfs-root))やマウントポイント([AWS](https://docs.databricks.com/dbfs/mounts.html)|[Azure](https://learn.microsoft.com/azure/databricks/dbfs/mounts)|[GCP](https://docs.gcp.databricks.com/dbfs/mounts.html))を利用することができます。
# MAGIC
# MAGIC DBFSを使用することで、モデル学習に使用する`transformers`互換のデータセットを作成する際に、"ローカル"パスを参照することができます。

# COMMAND ----------

# DBTITLE 1,データを学習用／検証用に分割し、HuggigFace Datasetに変換
dataset_df = spark.read.table(table_name_full_path)
(train_df, test_df) = dataset_df.persist().randomSplit([0.8, 0.2], seed=47)

import datasets
train_dataset = datasets.Dataset.from_spark(train_df, cache_dir="/dbfs/cache/train")
test_dataset = datasets.Dataset.from_spark(test_df, cache_dir="/dbfs/cache/test")

# COMMAND ----------

# MAGIC %md
# MAGIC 学習用のデータセットをトークン化してシャッフルする。また、学習処理では`text`カラムを必要としないので、データセットから削除します。
# MAGIC このステップでは、`datasets`は変換されたデータセットをローカルディスクにキャッシュし、モデルの学習時に高速に読み込めるようにする。

# COMMAND ----------

# DBTITLE 1,テキスト列からトークン列を作成（最終的にテキスト列は削除）
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)

def tokenize_function(examples):
    return tokenizer(
        examples["text"], 
        max_length=max_length,
        padding='max_length', 
        truncation=True)

train_tokenized = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
test_tokenized = test_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
train_dataset = train_tokenized.shuffle(seed=47)
test_dataset = test_tokenized.shuffle(seed=47)

# COMMAND ----------

# DBTITLE 1,Datasetsから1レコード取り出し、中身を見てみる
train_dataset[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1-3. モデルトレーニング

# COMMAND ----------

# MAGIC %md
# MAGIC ログに記録する評価指標を定義します。今回はAccuracyを記録します。損失（Loss）は設定せずとも自動でログに記録されます。

# COMMAND ----------

import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   return {"accuracy": accuracy}

# COMMAND ----------

# MAGIC %md
# MAGIC 学習用のパラメーターはほぼデフォルト値を使用しますが、Epoch数（デフォルトは3）とバッチサイズ（デフォルトは8）のみデフォルトから変更します。他にも必要に応じて学習率などの多くの学習パラメータを設定できます。詳細は
# MAGIC [transformersドキュメント](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) を参照ください。

# COMMAND ----------

from transformers import TrainingArguments

training_output_dir = f"{volume_path}/bert_trainer"
training_args = TrainingArguments(
  output_dir=training_output_dir, 
  logging_dir = f"{volume_path}/logs",    # TensorBoard用にログを記録するディレクトリ
  evaluation_strategy="epoch",
  num_train_epochs=2)

training_args.set_dataloader(train_batch_size=12, eval_batch_size=32)

print(f"学習時のバッチサイズは　{training_args.per_device_train_batch_size}、検証時のバッチサイズは　{training_args.per_device_eval_batch_size}　です。")


# COMMAND ----------

# MAGIC %md
# MAGIC ラベルマッピングとクラス数を指定して、ベースモデルから学習するモデルを作成します。

# COMMAND ----------

from transformers import AutoModelForSequenceClassification
import json

with open(f'{volume_path}/id2label.json') as f:
    id2label = json.load(f)
with open(f'{volume_path}/label2id.json') as f:
    label2id = json.load(f)

model = AutoModelForSequenceClassification.from_pretrained(
  base_model, 
  num_labels=len(category_list), 
  label2id=label2id, 
  id2label=id2label)

# COMMAND ----------

# MAGIC %md
# MAGIC [data collator](https://huggingface.co/docs/transformers/main_classes/data_collator)を使うことで、訓練データと評価データセットの入力をバッチ化することができる。`DataCollatorWithPadding`をデフォルトで使用すると、テキスト分類のベースライン性能が良くなる。

# COMMAND ----------

from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC 上記で作成したモデル、引数、データセット、照合器、メトリクスを用いて、トレーナーオブジェクトを構築します。

# COMMAND ----------

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをサービングエンドポイントとしてデプロイするためのMLFlow登録用ラッパークラスです。
# MAGIC 本来はMLFlowのTransformerフレーバーを使用することでより簡単に登録できるのですが、それだとサービング時にGPUが使用されないケースがあるため、GPU使用をするにはこの実装が必要です。

# COMMAND ----------

import mlflow
import torch

class TextClassificationPipelineModel(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model, tokenizer):
    device = 0 if torch.cuda.is_available() else -1
    self.pipeline = pipeline(
      "text-classification", 
      model=model, 
      tokenizer=tokenizer,
      batch_size=1,
      device=device)
    self.tokenizer = tokenizer
    
  def predict(self, context, model_input): 
    messages = model_input["text"].to_list()
    answers = self.pipeline(messages, max_length=max_length, padding='max_length', truncation=True)

    label_list = []
    score_list = []
    for answer in answers:
      label_list.append(answer['label'])
      score_list.append(str(answer['score']))

    return {"label": label_list, "score": score_list}

# COMMAND ----------

# MAGIC %md
# MAGIC モデルをトレーニングし、メトリクスと結果を MLflow に記録します。
# MAGIC MLFlowのTransformerフレーバーを使用することで、簡単に記録できます。

# COMMAND ----------

# DBTITLE 1,実験管理の記録場所を事前に指定
model_output_dir = f"{volume_path}/trained_model"
pipeline_output_dir = f"{volume_path}/trained_pipeline"

# mlflowの実験結果を記録するエクスペリメントを作成
username = spark.sql('select current_user() as user').collect()[0]['user']
try:
  experiment_id = mlflow.create_experiment(f"/Users/{username}/{experiment_name}")
except:
  print(f"The experiment /Users/{username}/{experiment_name} already exists.")
  experiment_id = mlflow.get_experiment_by_name(f'/Users/{username}/{experiment_name}').experiment_id

# COMMAND ----------

# DBTITLE 1,TensorBoard用にログのディレクトリーを指定
# MAGIC %load_ext tensorboard
# MAGIC experiment_log_dir = f"{volume_path}/logs"

# COMMAND ----------

# DBTITLE 1,モデルの学習を実行
from transformers import pipeline
import pandas as pd

from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

with mlflow.start_run(experiment_id=experiment_id) as run:
  
  # 学習開始。学習のメトリックが自動的にMLFLowにロギングされる
  trainer.train()

  # 学習終了後にモデルを保存
  trainer.save_model(model_output_dir)
  
  # 学習済みモデルを読み込んでパイプライン化して、更に保存。
  bert = AutoModelForSequenceClassification.from_pretrained(model_output_dir)

  # pipe = pipeline(
  #   "text-classification", 
  #   model=bert, 
  #   batch_size=1, 
  #   tokenizer=tokenizer,
  #   device=0)
  # pipe.save_pretrained(pipeline_output_dir)
  
  #######################################
  # CPUのみでのサービングで良ければこちらでも可
  #######################################
  # # MLFlow Trackingにパイプラインを記録する。
  # mlflow.transformers.log_model(
  #   transformers_model=pipe, 
  #   artifact_path=model_artifact_path+"_CPU", 
  #   input_example=["これはサンプル１です。", "これはサンプル２です。"],
  #   pip_requirements=["torch", "transformers", "accelerate", "sentencepiece", "datasets", "fugashi", "unidic-lite"],
  #   # registered_model_name=registered_model_name,
  #   model_config={ 
  #     "max_length": max_length, 
  #     "padding": "max_length", 
  #     "truncation": True 
  #   }
  # )

  # エンドポイントの入力と出力のスキーマを定義
  input_schema = Schema([ColSpec(DataType.string, "text")])
  output_schema = Schema([ColSpec(DataType.string, "label"), ColSpec(DataType.double, "score")])
  signature = ModelSignature(inputs=input_schema, outputs=output_schema)

  # 入力データのサンプルを用意
  input_example = pd.DataFrame({"text": ["これはサンプル１です。", "これはサンプル２です。"]})
  
  # モデルをMLFlow Trackingに記録
  mlflow.pyfunc.log_model(
      artifact_path=model_artifact_path,
      python_model=TextClassificationPipelineModel(bert, tokenizer),
      pip_requirements=["torch", "transformers", "accelerate", "sentencepiece", "datasets", "fugashi", "unidic-lite"],
      input_example=input_example,
      signature=signature
  )

print(f"モデルはMLFlow実験のRun(ID:{run.info.run_id})に記録されました。このIDを記録しておいてください。")

# COMMAND ----------

# DBTITLE 1,TensorBoard起動
# MAGIC %tensorboard --logdir $experiment_log_dir

# COMMAND ----------

# DBTITLE 1,MLFlow Trackingに記録したモデルをModel Registryへ登録
import mlflow

# モデルレジストリから最新バージョンの番号を取得する
username = spark.sql('select current_user() as user').collect()[0]['user']
run_id = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(f'/Users/{username}/{experiment_name}').experiment_id, order_by=["start_time desc"], max_results=1).iloc[0].run_id

# Unity Catalogにモデルを登録する場合
mlflow.set_registry_uri('databricks-uc')
result = mlflow.register_model(
    "runs:/"+run_id+f"/{model_artifact_path}",
    registered_model_full_path,
)

# COMMAND ----------

# DBTITLE 1,Model Registryへ登録したモデルバージョンにAlias / Tagを設定
from mlflow import MlflowClient
client = MlflowClient()

# これから検証するモデルバージョンに対して、Candidateというエイリアスを付与
client.set_registered_model_alias(
  name=registered_model_full_path, 
  alias="Candidate", 
  version=result.version
)

# 同じバージョンに対して、validation_statusのタグも付与し、NotValidatedの値を付与
client.set_model_version_tag(registered_model_full_path, result.version, "validation_status", "NotValidated")

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## おまけ: TorchDistributerを用いた分散学習
# MAGIC TorchDistributor は PySpark のオープンソース モジュールであり、ユーザーが Spark クラスターで PyTorch を使用して分散トレーニングを行うのに役立つため、PyTorch トレーニング ジョブを Spark ジョブとして起動できます。 内部的には、ワーカー間の環境と通信チャネルを初期化し、CLIコマンド torch.distributed.run を利用してワーカーノード間で分散トレーニングを実行します。
# MAGIC
# MAGIC 参考：https://docs.databricks.com/ja/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html

# COMMAND ----------

# import torch
 
# NUM_WORKERS = 4
 
# def get_gpus_per_worker(_):
#   import torch
#   return torch.cuda.device_count()
 
# NUM_GPUS_PER_WORKER = sc.parallelize(range(4), 4).map(get_gpus_per_worker).collect()[0]
# USE_GPU = NUM_GPUS_PER_WORKER > 0

# COMMAND ----------

# from transformers import AutoModelForSequenceClassification

# def train_model():
#     from transformers import TrainingArguments, Trainer
 
#     training_args = TrainingArguments(
#       output_dir=model_output_dir, 
#       evaluation_strategy="epoch",
#       save_strategy="epoch",
#       report_to=[], # REMOVE MLFLOW INTEGRATION FOR NOW
#       push_to_hub=False,  # DO NOT PUSH TO MODEL HUB FOR NOW,
#       load_best_model_at_end=True, # RECOMMENDED
#       metric_for_best_model="eval_loss", # RECOMMENDED
#       num_train_epochs=5)
 
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         compute_metrics=compute_metrics,
#         data_collator=data_collator,
#     )
    
#     trainer.train()
#     return trainer.state.best_model_checkpoint
 
# # It is recommended to create a separate local trainer from pretrained model instead of using the trainer used in distributed training
# def test_model(ckpt_path):
#     model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
#     local_trainer = Trainer(
#         model=model,
#         eval_dataset=test_dataset,
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics
#     )
#     return local_trainer.evaluate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### シングルノードでマルチGPU

# COMMAND ----------

# from pyspark.ml.torch.distributor import TorchDistributor

# NUM_PROCESSES = torch.cuda.device_count()
# print(f"We're using {NUM_PROCESSES} GPUs")
# single_node_multi_gpu_ckpt_path = TorchDistributor(
#   num_processes=NUM_PROCESSES, 
#   local_mode=True, 
#   use_gpu=USE_GPU).run(train_model)

# COMMAND ----------

# test_model(single_node_multi_gpu_ckpt_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### マルチノードでシングル/マルチGPU

# COMMAND ----------

# from pyspark.ml.torch.distributor import TorchDistributor
 
# NUM_PROCESSES = NUM_GPUS_PER_WORKER * NUM_WORKERS
# print(f"We're using {NUM_PROCESSES} GPUs")
# multi_node_ckpt_path = TorchDistributor(
#   num_processes=NUM_PROCESSES, 
#   local_mode=False, 
#   use_gpu=USE_GPU).run(train_model)

# COMMAND ----------

# test_model(multi_node_ckpt_path)
