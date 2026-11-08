# Détection de fraude bancaire en temps réel — Pipeline Big Data (architecture Lambda)

Système de détection de fraude par carte bancaire reposant sur un écosystème Big Data complet :
ingestion en flux, stockage distribué, entraînement distribué, scoring en temps réel et visualisation.

> Projet réalisé par **[Mohamed Amine CHABLAOUI]**, encadré par **[Chaymae RAMI]**
> — **[Intelligence Artificielle et Technologies Émergentes(MIATE)]**, **[2025/2026]**.

## Architecture

```
                    COUCHE BATCH (hors ligne)
  creditcard.csv ──► HDFS ──► Spark MLlib (train_model.py) ──► modèle sauvegardé dans HDFS
                                                                        │
                    COUCHE VITESSE (temps réel)                         │ chargement du modèle
  kafka_producer.py ──► Kafka ──► Spark Structured Streaming ◄──────────┘
                                  (streaming_consumer.py)
                                           │
                    COUCHE DE SERVICE      ▼
                                   Elasticsearch ──► Kibana (tableau de bord)
```

| Outil | Rôle |
|---|---|
| Apache Kafka (+ Zookeeper) | Ingestion du flux de transactions (topic `transactions`) |
| HDFS | Data lake : stockage de l'historique et du modèle entraîné |
| Apache Spark MLlib | Entraînement distribué d'un Random Forest (couche batch) |
| Spark Structured Streaming | Scoring des transactions par micro-batchs (couche vitesse) |
| Elasticsearch | Stockage indexé des transactions scorées |
| Kibana | Tableau de bord de visualisation |
| Docker Compose | Déploiement des 6 services d'infrastructure |

Les trois scripts Python (Spark 3.5.1 via son API Python) s'exécutent sur la machine hôte ;
les six services d'infrastructure s'exécutent dans des conteneurs Docker.

## Structure du dépôt

```
.
├── docker-compose.yml        # Infrastructure : zookeeper, kafka, elasticsearch, kibana, namenode, datanode
├── requirements.txt          # Dépendances Python
├── train_model.py            # Couche batch : entraînement et sauvegarde du modèle dans HDFS
├── streaming_consumer.py     # Couche vitesse : Kafka -> scoring -> Elasticsearch
├── kafka_producer.py         # Simulateur de flux de transactions
└── data/                     # Dataset (non versionné, voir ci-dessous)
```

## Données

Le dataset n'est pas inclus dans le dépôt (325 Mo). Téléchargez
**« Credit Card Fraud Detection Dataset 2023 »** sur Kaggle
([kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023)),
renommez le fichier en `creditcard.csv` et placez-le dans le dossier `data/`.

- 568 630 transactions, réparties à 50 % / 50 % entre normales et frauduleuses
- Colonnes : `id`, `V1` à `V28` (issues d'une ACP, donc anonymisées), `Amount`, `Class`

## Prérequis (Windows 10/11)

- Docker Desktop (moteur WSL2)
- Python **3.11** (PySpark 3.5 ne prend pas en charge Python 3.13)
- JDK 17 (variable `JAVA_HOME` définie)
- `winutils.exe` et `hadoop.dll` dans `C:\hadoop\bin`, variable `HADOOP_HOME=C:\hadoop`
- Fichier `C:\Windows\System32\drivers\etc\hosts` : ajouter les lignes
  ```
  127.0.0.1 namenode
  127.0.0.1 datanode
  ```
- Le projet doit se trouver dans un chemin **sans espace** (ex. `C:\fraude-bigdata`)

## Installation

```
cd C:\fraude-bigdata
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

**1. Démarrer l'infrastructure**
```
docker compose up -d
docker ps
```

**2. Créer le topic Kafka** (à refaire après chaque redémarrage des conteneurs)
```
docker exec -it fraude-bigdata-kafka-1 kafka-topics --create --topic transactions --bootstrap-server localhost:9092
```

**3. Charger le dataset dans HDFS** (première fois seulement)
```
docker exec -it fraude-bigdata-namenode-1 hdfs dfs -chmod -R 777 /
docker exec -it fraude-bigdata-namenode-1 hdfs dfs -mkdir -p /data
docker cp C:\fraude-bigdata\data\creditcard.csv fraude-bigdata-namenode-1:/tmp/creditcard.csv
docker exec -it fraude-bigdata-namenode-1 hdfs dfs -put -f /tmp/creditcard.csv /data/creditcard.csv
```

**4. Entraîner le modèle** (première fois seulement ; le modèle est sauvegardé dans HDFS)
```
venv\Scripts\activate
python train_model.py
```

**5. Lancer le scoring en temps réel** (fenêtre dédiée, à laisser ouverte)
```
venv\Scripts\activate
python streaming_consumer.py
```

**6. Lancer le simulateur de transactions** (autre fenêtre)
```
venv\Scripts\activate
python kafka_producer.py
```

**7. Visualiser** — ouvrir Kibana sur <http://localhost:5601>, créer une *data view* `fraud-scores*`
puis un tableau de bord (donut `is_fraud_predicted`, compteur, top 10 des `fraud_probability`).

Interfaces utiles : Kibana `localhost:5601` · NameNode HDFS `localhost:9870` · Elasticsearch `localhost:9200`.

## Arrêt

`Ctrl+C` dans chaque fenêtre Python, puis :
```
docker compose down
```
Sans l'option `-v`, les volumes (HDFS, Elasticsearch, Kibana) sont conservés.

## Résultats et limites

Sur le jeu de test (20 %), le premier entraînement donne AUC = 1,0000 et F1 pondéré = 0,9988.

**Ces résultats sont optimistes.** Dans ce jeu de données, les lignes sont triées par classe : la colonne
`id` est fortement corrélée à `Class` (corrélation ≈ 0,86) et constitue une fuite d'information
lorsqu'elle est utilisée comme variable explicative. Il est recommandé de l'exclure
(`feature_cols = [c for c in df.columns if c not in ("Class", "id")]`) puis de réentraîner.
Les métriques affichées sont par ailleurs des moyennes pondérées sur les deux classes.

Autres limites connues :
- HDFS ne compte qu'un seul DataNode : la tolérance aux pannes n'est pas démontrée
- Kafka et Zookeeper n'ont pas de volume persistant (le topic est perdu au redémarrage)
- Les droits HDFS sont ouverts (`chmod 777`) et la sécurité d'Elasticsearch est désactivée : configuration de démonstration uniquement
- Spark s'exécute en mode local (`local[*]`), pas sur un cluster
- Le flux simulé est un échantillon du même fichier que l'entraînement : il démontre le pipeline, pas la généralisation du modèle

## Pistes d'amélioration

Réentraînement périodique orchestré (Apache Airflow), modèles de type Gradient Boosted Trees,
métriques par classe (rappel de la classe fraude), système d'alertes, déploiement sur cluster.
