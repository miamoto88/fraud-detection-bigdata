"""
Simule un flux de transactions bancaires en temps réel en publiant
des lignes du dataset (jeu de test) dans un topic Kafka.
En production, ce producteur serait remplacé par le système
transactionnel réel de la banque (core banking system).
"""

import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC = "transactions"
BOOTSTRAP_SERVERS = "localhost:9092"
DATA_PATH = "data/creditcard.csv"
DELAY_SECONDS = 0.5  # intervalle entre deux transactions simulées


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

    df = pd.read_csv(DATA_PATH)
    # on simule le flux avec les dernières transactions (non vues à l'entraînement)
    stream_df = df.sample(frac=0.05, random_state=7).reset_index(drop=True)

    print(f">> Démarrage du flux de {len(stream_df)} transactions vers le topic '{TOPIC}'")
    for i, row in stream_df.iterrows():
        message = row.to_dict()
        message["transaction_id"] = i
        producer.send(TOPIC, value=message)
        print(f"Transaction envoyée #{i} (montant={message.get('Amount')})")
        time.sleep(DELAY_SECONDS)

    producer.flush()
    print(">> Flux terminé.")


if __name__ == "__main__":
    main()
