import json
import time
import uuid
from datetime import datetime, timezone
from random import choice, randint, random

from faker import Faker
from kafka import KafkaProducer

fake = Faker()

TOPIC = "bank_txn_stream"
BOOTSTRAP = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def generate_txn():
    amount = round(randint(1, 5000) + random(), 2)

    channel = choice(["POS", "ECOM", "ATM"])
    status = choice(["APPROVED", "DECLINED"])

    # Create occasional "suspicious" patterns (~5%)
    is_high_risk = random() < 0.05
    risk_reason = None

    if is_high_risk:
        amount = round(randint(7000, 15000) + random(), 2)
        channel = "ECOM"
        status = "DECLINED"
        risk_reason = "HIGH_AMOUNT_ECOM_DECLINED"

    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "customer_id": f"C{randint(1000, 9999)}",
        "merchant_id": f"M{randint(100, 999)}",
        "amount": amount,
        "currency": "CAD",
        "channel": channel,
        "city": fake.city(),
        "country": "CA",
        "device_id": f"D{randint(10000, 99999)}",
        "status": status,
        "is_high_risk": is_high_risk,
        "risk_reason": risk_reason,
    }


if __name__ == "__main__":
    print("Producing transactions to Kafka... Ctrl+C to stop.")
    try:
        while True:
            txn = generate_txn()
            producer.send(TOPIC, txn)
            print(txn)
            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        producer.close()
