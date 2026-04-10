import json
import time
import uuid
from datetime import datetime, timedelta
from random import choice, randint, random, uniform
from zoneinfo import ZoneInfo

from faker import Faker
from kafka import KafkaProducer

fake = Faker()

TOPIC = "bank_txn_stream_v2"
BOOTSTRAP = "localhost:9092"

producer = KafkaProducer(
    bootstrap_servers=BOOTSTRAP,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

CHANNELS = ["POS", "ECOM", "ATM", "MOBILE", "ONLINE_BANKING"]
CURRENCIES = ["CAD", "USD", "EUR", "GBP"]

MERCHANT_CATEGORIES = {
    "GROCERY": (10, 220),
    "RESTAURANT": (15, 250),
    "FUEL": (20, 180),
    "ECOMMERCE": (25, 1200),
    "PHARMACY": (5, 120),
    "TRAVEL": (80, 5000),
    "ELECTRONICS": (120, 4500),
    "LUXURY": (500, 15000),
    "GAMING": (10, 400),
    "ATM_CASH": (20, 1000),
    "BILLPAY": (40, 2500),
}

CITIES_BY_COUNTRY = {
    "CA": ["Toronto", "Mississauga", "Brampton", "Ottawa", "Hamilton", "Montreal", "Calgary", "Vancouver"],
    "US": ["New York", "Chicago", "Miami", "Dallas", "Los Angeles"],
    "GB": ["London", "Manchester", "Birmingham"],
    "AE": ["Dubai", "Abu Dhabi"],
    "IN": ["Mumbai", "Hyderabad", "Delhi", "Bengaluru"],
    "FR": ["Paris", "Lyon"],
}

CUSTOMER_HOME_COUNTRIES = ["CA", "CA", "CA", "CA", "US", "GB", "IN"]

FRAUD_PATTERNS = [
    "NONE",
    "HIGH_AMOUNT_ECOM_DECLINED",
    "IMPOSSIBLE_TRAVEL",
    "ATM_CASHOUT_BURST",
    "FOREIGN_ECOM",
    "DEVICE_CHANGE",
    "RAPID_FIRE_SMALL_TXNS",
    "LATE_NIGHT_HIGH_RISK",
    "COMPROMISED_ACCOUNT_BILLPAY",
    "LUXURY_GOODS_SPIKE",
]

CUSTOMER_PROFILES: dict[str, dict] = {}

CANADA_TZ = ZoneInfo("America/Toronto")

BACKFILL_START = datetime(2026, 4, 6, 0, 0, 0, tzinfo=CANADA_TZ)
TXNS_PER_DAY = 1000

BACKFILL_SLEEP = 0.002
LIVE_SLEEP = 1.0
LIVE_LOG_EVERY = 10
BACKFILL_LOG_EVERY = 500


def weighted_choice(items):
    total = sum(weight for _, weight in items)
    r = uniform(0, total)
    upto = 0
    for item, weight in items:
        if upto + weight >= r:
            return item
        upto += weight
    return items[-1][0]


def usual_channels_for_segment(segment: str) -> list[str]:
    if segment == "Student":
        return ["ECOM", "MOBILE", "POS"]
    if segment == "Mass Market":
        return ["POS", "ECOM", "ATM", "MOBILE"]
    if segment == "Premium":
        return ["POS", "ECOM", "MOBILE", "ONLINE_BANKING"]
    if segment == "Business":
        return ["ONLINE_BANKING", "POS", "ECOM", "MOBILE"]
    return ["POS", "ECOM", "ONLINE_BANKING", "MOBILE"]


def create_customer_profile(customer_id: str) -> dict:
    home_country = choice(CUSTOMER_HOME_COUNTRIES)
    home_city = choice(CITIES_BY_COUNTRY[home_country])

    segment = weighted_choice([
        ("Student", 12),
        ("Mass Market", 45),
        ("Premium", 25),
        ("Business", 10),
        ("High Net Worth", 8),
    ])

    age_band = weighted_choice([
        ("18-25", 15),
        ("26-35", 28),
        ("36-50", 30),
        ("51-65", 18),
        ("65+", 9),
    ])

    if segment == "Student":
        preferred_categories = ["GROCERY", "RESTAURANT", "ECOMMERCE", "GAMING", "PHARMACY"]
        amount_multiplier = 0.7
    elif segment == "Mass Market":
        preferred_categories = ["GROCERY", "FUEL", "RESTAURANT", "PHARMACY", "BILLPAY", "ECOMMERCE"]
        amount_multiplier = 1.0
    elif segment == "Premium":
        preferred_categories = ["GROCERY", "TRAVEL", "ECOMMERCE", "ELECTRONICS", "RESTAURANT", "LUXURY"]
        amount_multiplier = 1.5
    elif segment == "Business":
        preferred_categories = ["TRAVEL", "FUEL", "RESTAURANT", "BILLPAY", "ECOMMERCE", "ELECTRONICS"]
        amount_multiplier = 1.8
    else:
        preferred_categories = ["LUXURY", "TRAVEL", "ECOMMERCE", "ELECTRONICS", "RESTAURANT"]
        amount_multiplier = 2.4

    return {
        "customer_id": customer_id,
        "home_country": home_country,
        "home_city": home_city,
        "home_device_id": f"D{randint(10000, 99999)}",
        "customer_age_band": age_band,
        "customer_segment": segment,
        "preferred_categories": preferred_categories,
        "amount_multiplier": amount_multiplier,
        "usual_channels": usual_channels_for_segment(segment),
    }


def get_customer_profile() -> dict:
    if random() < 0.8 and CUSTOMER_PROFILES:
        return CUSTOMER_PROFILES[choice(list(CUSTOMER_PROFILES.keys()))]

    customer_id = f"C{randint(1000, 9999)}"
    if customer_id not in CUSTOMER_PROFILES:
        CUSTOMER_PROFILES[customer_id] = create_customer_profile(customer_id)
    return CUSTOMER_PROFILES[customer_id]


def choose_merchant_for_customer(customer: dict) -> dict:
    category_pool = customer["preferred_categories"] + list(MERCHANT_CATEGORIES.keys())
    category = choice(category_pool)

    merchant_country = weighted_choice([
        (customer["home_country"], 70),
        ("CA", 20),
        ("US", 5),
        ("GB", 2),
        ("AE", 1),
        ("IN", 1),
        ("FR", 1),
    ])
    merchant_city = choice(CITIES_BY_COUNTRY[merchant_country])

    merchant_risk_level = weighted_choice([
        ("LOW", 60),
        ("MEDIUM", 30),
        ("HIGH", 10),
    ])

    return {
        "merchant_id": f"M{randint(100, 999)}",
        "merchant_category": category,
        "merchant_country": merchant_country,
        "merchant_city": merchant_city,
        "merchant_risk_level": merchant_risk_level,
    }


def base_amount(category: str, multiplier: float, weekend: bool, payday: bool) -> float:
    low, high = MERCHANT_CATEGORIES[category]
    amount = uniform(low, high) * multiplier

    if weekend and category in ["RESTAURANT", "TRAVEL", "LUXURY", "GAMING"]:
        amount *= 1.2

    if payday and category in ["ECOMMERCE", "LUXURY", "TRAVEL", "ELECTRONICS", "BILLPAY"]:
        amount *= 1.25

    return round(amount, 2)


def weighted_status(is_high_risk: bool) -> str:
    if is_high_risk:
        return weighted_choice([
            ("DECLINED", 55),
            ("APPROVED", 25),
            ("PENDING", 20),
        ])
    return weighted_choice([
        ("APPROVED", 86),
        ("PENDING", 8),
        ("DECLINED", 6),
    ])


def normal_hour_boost(hour: int) -> float:
    if 7 <= hour <= 9:
        return 1.15
    if 12 <= hour <= 14:
        return 1.2
    if 17 <= hour <= 21:
        return 1.35
    if 0 <= hour <= 4:
        return 0.45
    return 1.0


def build_backfill_timestamps():
    now_local = datetime.now(CANADA_TZ)

    if BACKFILL_START >= now_local:
        return []

    total_seconds = (now_local - BACKFILL_START).total_seconds()
    total_days = total_seconds / 86400
    total_txns = max(1, int(total_days * TXNS_PER_DAY))
    step_seconds = total_seconds / total_txns

    timestamps = []
    current_ts = BACKFILL_START

    for _ in range(total_txns):
        jitter_ms = randint(0, 900)
        ts = current_ts + timedelta(milliseconds=jitter_ms)

        if ts > now_local:
            break

        timestamps.append(ts)
        current_ts += timedelta(seconds=step_seconds)

    return timestamps


def generate_transaction(event_time: datetime):
    hour = event_time.hour
    weekday = event_time.weekday()
    day = event_time.day

    weekend = weekday >= 5
    payday = day in [1, 15, 28, 30, 31]

    customer = get_customer_profile()
    merchant = choose_merchant_for_customer(customer)

    transaction_id = str(uuid.uuid4())
    channel = choice(customer["usual_channels"])

    amount = base_amount(
        merchant["merchant_category"],
        customer["amount_multiplier"],
        weekend,
        payday,
    )
    amount = round(amount * normal_hour_boost(hour), 2)

    country = merchant["merchant_country"]
    city = merchant["merchant_city"]
    currency = "CAD" if country == "CA" else choice(CURRENCIES)
    device_id = customer["home_device_id"]
    card_present = int(channel in ["POS", "ATM"])
    is_international = int(country != customer["home_country"])
    is_new_device = 0
    velocity_hint = randint(1, 3)
    geo_distance_risk = 0
    suspicious_hour = int(hour in [0, 1, 2, 3, 4])
    fraud_pattern = "NONE"
    is_high_risk = False
    risk_reason = None

    risk_score = 0.04
    if merchant["merchant_risk_level"] == "HIGH":
        risk_score += 0.03
    if is_international:
        risk_score += 0.02
    if suspicious_hour:
        risk_score += 0.02
    if channel in ["ECOM", "ONLINE_BANKING", "MOBILE"]:
        risk_score += 0.01

    if random() < risk_score:
        is_high_risk = True
        fraud_pattern = choice(FRAUD_PATTERNS[1:])

        if fraud_pattern == "HIGH_AMOUNT_ECOM_DECLINED":
            channel = "ECOM"
            card_present = 0
            amount = round(uniform(2500, 12000), 2)
            risk_reason = "VERY_HIGH_ECOM_DECLINE"

        elif fraud_pattern == "IMPOSSIBLE_TRAVEL":
            country = choice(["AE", "GB", "FR", "IN"])
            city = choice(CITIES_BY_COUNTRY[country])
            currency = choice(CURRENCIES)
            amount = round(uniform(1200, 9000), 2)
            geo_distance_risk = 1
            is_international = 1
            risk_reason = "IMPOSSIBLE_TRAVEL_PATTERN"

        elif fraud_pattern == "ATM_CASHOUT_BURST":
            channel = "ATM"
            card_present = 1
            merchant["merchant_category"] = "ATM_CASH"
            amount = round(uniform(800, 4000), 2)
            velocity_hint = randint(6, 15)
            risk_reason = "MULTIPLE_ATM_WITHDRAWALS"

        elif fraud_pattern == "FOREIGN_ECOM":
            channel = "ECOM"
            card_present = 0
            country = choice(["AE", "IN", "GB", "FR", "US"])
            city = choice(CITIES_BY_COUNTRY[country])
            currency = choice(CURRENCIES)
            amount = round(uniform(900, 7000), 2)
            is_international = 1
            risk_reason = "FOREIGN_CARD_NOT_PRESENT"

        elif fraud_pattern == "DEVICE_CHANGE":
            channel = choice(["MOBILE", "ECOM", "ONLINE_BANKING"])
            card_present = 0
            device_id = f"D{randint(10000, 99999)}"
            is_new_device = 1
            amount = round(uniform(500, 6000), 2)
            risk_reason = "NEW_DEVICE_LOGIN_OR_PURCHASE"

        elif fraud_pattern == "RAPID_FIRE_SMALL_TXNS":
            channel = choice(["ECOM", "POS"])
            amount = round(uniform(1, 35), 2)
            velocity_hint = randint(8, 25)
            risk_reason = "MICRO_TRANSACTION_VELOCITY"

        elif fraud_pattern == "LATE_NIGHT_HIGH_RISK":
            channel = choice(["ECOM", "ONLINE_BANKING", "ATM"])
            card_present = int(channel in ["ATM"])
            amount = round(uniform(1500, 10000), 2)
            suspicious_hour = 1
            risk_reason = "ODD_HOUR_HIGH_VALUE_ACTIVITY"

        elif fraud_pattern == "COMPROMISED_ACCOUNT_BILLPAY":
            channel = "ONLINE_BANKING"
            card_present = 0
            merchant["merchant_category"] = "BILLPAY"
            amount = round(uniform(1200, 8500), 2)
            device_id = f"D{randint(10000, 99999)}"
            is_new_device = 1
            risk_reason = "UNUSUAL_BILLPAY_FROM_NEW_DEVICE"

        elif fraud_pattern == "LUXURY_GOODS_SPIKE":
            channel = "ECOM"
            card_present = 0
            merchant["merchant_category"] = "LUXURY"
            amount = round(uniform(2500, 20000), 2)
            risk_reason = "LUXURY_SPENDING_SPIKE"

    status = weighted_status(is_high_risk)

    txn = {
        "transaction_id": transaction_id,
        "timestamp": event_time.isoformat(),
        "customer_id": customer["customer_id"],
        "customer_segment": customer["customer_segment"],
        "customer_age_band": customer["customer_age_band"],
        "merchant_id": merchant["merchant_id"],
        "merchant_category": merchant["merchant_category"],
        "merchant_risk_level": merchant["merchant_risk_level"],
        "amount": amount,
        "currency": currency,
        "channel": channel,
        "city": city,
        "country": country,
        "home_city": customer["home_city"],
        "home_country": customer["home_country"],
        "device_id": device_id,
        "home_device_id": customer["home_device_id"],
        "status": status,
        "card_present": card_present,
        "is_international": is_international,
        "is_new_device": is_new_device,
        "velocity_hint": velocity_hint,
        "geo_distance_risk": geo_distance_risk,
        "suspicious_hour": suspicious_hour,
        "is_high_risk": is_high_risk,
        "fraud_pattern": fraud_pattern,
        "risk_reason": risk_reason,
        "weekday": weekday,
        "is_weekend": int(weekend),
        "is_payday_window": int(payday),
    }

    return txn


if __name__ == "__main__":
    print("Producing banking transactions...")
    print(f"Backfill start: {BACKFILL_START.isoformat()}")
    print("Backfill end: current Canada local time")
    print("Then switch to live mode")

    sent_count = 0

    try:
        backfill_timestamps = build_backfill_timestamps()
        print(f"Backfill records to produce: {len(backfill_timestamps)}")

        for ts in backfill_timestamps:
            txn = generate_transaction(ts)
            producer.send(TOPIC, txn)
            sent_count += 1

            if sent_count % BACKFILL_LOG_EVERY == 0:
                producer.flush()
                print(f"Sent {sent_count} records... latest timestamp={txn['timestamp']}")

            time.sleep(BACKFILL_SLEEP)

        producer.flush()
        print("Backfill complete. Switching to live mode...")

        while True:
            now_local = datetime.now(CANADA_TZ)
            txn = generate_transaction(now_local)
            producer.send(TOPIC, txn)
            sent_count += 1

            if sent_count % LIVE_LOG_EVERY == 0:
                producer.flush()
                print(f"Live sent {sent_count} records... latest timestamp={txn['timestamp']}")

            time.sleep(LIVE_SLEEP)

    except KeyboardInterrupt:
        print("Producer stopped by user.")
    finally:
        producer.flush()
        producer.close()