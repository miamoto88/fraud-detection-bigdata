"""
Couche BATCH du pipeline Big Data.
Entraîne un modèle de classification (Random Forest) avec Spark MLlib
pour distinguer transactions normales / frauduleuses.

Dataset attendu : data/creditcard.csv
(dataset public "Credit Card Fraud Detection" - Kaggle,
 colonnes : Time, V1..V28 (features anonymisées par PCA), Amount, Class)
Télécharger sur : https://www.kaggle.com/mlg-ulb/creditcardfraud
et placer le fichier dans data/creditcard.csv
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

DATA_PATH = "hdfs://namenode:9000/data/creditcard.csv"
MODEL_PATH = "hdfs://namenode:9000/models/fraud_model"

def main():
    spark = (
        SparkSession.builder
        .appName("FraudDetection-BatchTraining")
        .master("local[*]")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "true")
        .getOrCreate()
    )

    print(">> Lecture du dataset...")
    df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

    feature_cols = [c for c in df.columns if c not in ("Class",)]

    print(">> Construction du pipeline de features...")
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    rf = RandomForestClassifier(
        labelCol="Class",
        featuresCol="features",
        numTrees=100,
        maxDepth=10,
        seed=42,
        # Le dataset est très déséquilibré (fraudes = ~0.17%),
        # on pondère les classes pour ne pas ignorer la fraude.
    )

    pipeline = Pipeline(stages=[assembler, scaler, rf])

    print(">> Split train/test...")
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

    print(">> Entraînement du modèle (Spark MLlib, distribué)...")
    model = pipeline.fit(train_df)

    print(">> Évaluation...")
    predictions = model.transform(test_df)

    auc_eval = BinaryClassificationEvaluator(
        labelCol="Class", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    f1_eval = MulticlassClassificationEvaluator(
        labelCol="Class", predictionCol="prediction", metricName="f1"
    )
    precision_eval = MulticlassClassificationEvaluator(
        labelCol="Class", predictionCol="prediction", metricName="weightedPrecision"
    )
    recall_eval = MulticlassClassificationEvaluator(
        labelCol="Class", predictionCol="prediction", metricName="weightedRecall"
    )

    print(f"AUC       : {auc_eval.evaluate(predictions):.4f}")
    print(f"F1-score  : {f1_eval.evaluate(predictions):.4f}")
    print(f"Precision : {precision_eval.evaluate(predictions):.4f}")
    print(f"Recall    : {recall_eval.evaluate(predictions):.4f}")

    print(f">> Sauvegarde du modèle dans {MODEL_PATH}")
    model.write().overwrite().save(MODEL_PATH)

    spark.stop()


if __name__ == "__main__":
    main()
