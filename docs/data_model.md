# Data Model Definition

This document defines the structure of the core banking transaction event used throughout the data pipeline.

---

## Transaction Event Schema

Each transaction event represents a single banking transaction generated in real time.

### Fields

| Field Name | Data Type | Description |
|---------|----------|------------|
| transaction_id | String | Unique identifier for the transaction |
| timestamp | ISO 8601 String | Event timestamp in UTC |
| customer_id | String | Unique customer identifier |
| merchant_id | String | Unique merchant identifier |
| amount | Decimal | Transaction amount |
| currency | String | Currency code (e.g., CAD) |
| channel | String | Transaction channel (POS, ECOM, ATM) |
| city | String | City where transaction occurred |
| country | String | Country code |
| device_id | String | Device identifier |
| status | String | Transaction result (APPROVED, DECLINED) |

---

## Notes

- This schema acts as a data contract between producers and consumers.
- Future versions may include risk flags and fraud indicators.
