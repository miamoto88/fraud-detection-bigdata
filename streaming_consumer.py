"""
Couche SPEED (temps réel) du pipeline Big Data.
Lit le flux Kafka en continu avec Spark Structured Streaming,
applique le modèle entraîné en batch, et écrit les transactions
scorées dans Elasticsearch pour visualisation dans Kibana.
"""

import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType
from pyspark.ml import PipelineModel
from elasticsearch import Elasticsearch, helpers
import sys, os
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

KAFKA_BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"
MODEL_PATH = "hdfs://namenode:9000/models/fraud_model"
ES_HOST = "http://localhost:9200"
ES_INDEX = "fraud-scores"

# Schéma du dataset Credit Card Fraud (Time, V1..V28, Amount)
FEATURE_COLS = ["id"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
schema = StructType(
    [StructField(c, DoubleType(), True) for c in FEATURE_COLS]
    + [StructField("transaction_id", IntegerType(), True)]
)


def write_to_elasticsearch(batch_df, batch_id):
    """Envoie chaque micro-batch scoré vers Elasticsearch (foreachBatch)."""
    es = Elasticsearch(ES_HOST)
    rows = batch_df.select(
        "transaction_id", "Amount", "prediction", "probability_fraud"
    ).collect()

    actions = [
        {
            "_index": ES_INDEX,
            "_source": {
                "transaction_id": r["transaction_id"],
                "amount": r["Amount"],
                "is_fraud_predicted": bool(r["prediction"]),
                "fraud_probability": r["probability_fraud"],
            },
        }
        for r in rows
    ]
    if actions:
        helpers.bulk(es, actions)
        print(f">> Batch {batch_id}: {len(actions)} transactions indexées dans Elasticsearch")


def main():
    spark = (
        SparkSession.builder
        .appName("FraudDetection-RealtimeScoring")
        .master("local[*]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    print(">> Chargement du modèle entraîné...")
    model = PipelineModel.load(MODEL_PATH)

    print(">> Connexion au topic Kafka...")
    raw_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed = raw_stream.select(
        from_json(col("value").cast("string"), schema).alias("data")
    ).select("data.*")

    predictions = model.transform(parsed)

    # extraire la probabilité de fraude (classe 1) du vecteur de probabilité
    from pyspark.sql.functions import udf
    from pyspark.ml.linalg import VectorUDT

    @udf(DoubleType())
    def extract_proba(v):
        return float(v[1])

    scored = predictions.withColumn(
        "probability_fraud", extract_proba(col("probability"))
    ).withColumn("scored_at", current_timestamp())

    print(">> Démarrage du scoring en streaming (écriture vers Elasticsearch)...")
    query = (
        scored.writeStream
        .foreachBatch(write_to_elasticsearch)
        .outputMode("update")
        .start()
    )

    query.awaitTermination()


if __name__ == "__main__":
    main()
