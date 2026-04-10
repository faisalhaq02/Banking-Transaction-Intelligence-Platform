# Week 2 Progress Report  
**Project:** Real-Time Banking Transaction Intelligence Platform  
**Focus:** Kafka → Spark Structured Streaming → Bronze Lakehouse  
**Duration:** Week 2 (Streaming & Ingestion Phase)

---

## 1. Objective of Week 2
The goal of Week 2 was to design and implement a real-time data ingestion pipeline that simulates banking transactions and ingests them into a Bronze data lake layer using industry-standard data engineering tools.

Specifically, this week focused on:
- Real-time event generation
- Streaming ingestion using Apache Kafka
- Processing using Apache Spark Structured Streaming
- Persisting raw events into a Bronze lakehouse layer as Parquet files

---

## 2. Architecture Implemented (Week 2 Scope)

**Data Flow:**

Kafka Producer  
→ Apache Kafka Topic (`bank_txn_stream`)  
→ Spark Structured Streaming  
→ Bronze Layer (Parquet + Snappy Compression)

Kafka is used as the real-time message broker, while Spark Structured Streaming consumes and processes events before writing them to the data lake.

---

## 3. Technologies Used

| Component | Technology |
|---------|------------|
| Event Streaming | Apache Kafka |
| Stream Processing | Apache Spark Structured Streaming |
| Data Format | JSON (input), Parquet (output) |
| Compression | Snappy |
| Storage (Local) | Local filesystem (Docker volume) |
| Orchestration (later) | Apache Airflow |
| Language | Python (PySpark, kafka-python) |
| Containerization | Docker |

---

## 4. Data Model (Transaction Event)

Each Kafka message represents a banking transaction event with the following schema:

| Field | Type | Description |
|------|------|------------|
| transaction_id | String | Unique transaction identifier |
| timestamp | String (ISO) | Event creation timestamp |
| customer_id | String | Customer identifier |
| merchant_id | String | Merchant identifier |
| amount | Double | Transaction amount |
| currency | String | Currency (CAD) |
| channel | String | POS / ATM / ECOM |
| city | String | Transaction city |
| country | String | Transaction country |
| device_id | String | Device identifier |
| status | String | APPROVED / DECLINED |

This schema acts as the **data contract** between producers and consumers.

---

## 5. Kafka Implementation

### Topics
- `bank_txn_stream`
  - Partitions: 3
  - Replication factor: 1 (local development)

### Producer
- Python-based Kafka producer simulates banking transactions
- Uses Faker library for realistic synthetic data
- Configurable throughput to simulate production-scale traffic

### Kafka Observations
- Kafka UI confirms real-time message ingestion
- Kafka log directories show ~1GB of raw JSON data stored
- Offset lag remains at zero, indicating healthy consumption

---

## 6. Spark Structured Streaming (Bronze Layer)

### Streaming Job Responsibilities
- Consume Kafka messages
- Deserialize JSON payloads
- Apply schema enforcement
- Add ingestion metadata (`ingest_ts`, `date`, `hour`)
- Write raw immutable records to Bronze storage

### Output Characteristics
- File format: Parquet
- Compression: Snappy
- Partitioning:
  - `date`
  - `hour`
- Checkpointing enabled for fault tolerance

### Bronze Layer Design
- Stores raw, immutable events
- No transformations applied beyond parsing
- Serves as the source of truth for downstream processing

---

## 7. Data Volume Generated

| Metric | Value |
|------|------|
| Kafka Topic Size | ~1 GB |
| Bronze Parquet Size | ~424 MB |
| Number of Parquet Files | 11 |
| Average File Size | 6–17 KB |

**Explanation:**  
The reduction in size is due to columnar storage and Snappy compression in Parquet.

---

## 8. Key Engineering Concepts Demonstrated

- Event-driven architecture
- Streaming ingestion
- Schema enforcement
- Partitioned data lake design
- Checkpointing for fault tolerance
- Separation of concerns (Kafka vs Storage)
- Small-file problem (to be addressed in Silver layer)

---

## 9. Challenges Encountered & Resolutions

### Issue: Spark Kafka connector dependency errors
- **Resolution:** Configured custom Ivy cache directory via Docker volume mapping

### Issue: Small Parquet files generated
- **Resolution:** Adjusted Spark micro-batch trigger interval
- **Future fix:** File compaction in Silver layer

---

## 10. Outcome of Week 2
By the end of Week 2:
- A fully functional real-time ingestion pipeline is operational
- Data is continuously streamed from Kafka to the Bronze lakehouse
- The system is scalable and cloud-ready
- The foundation is established for data quality, transformation, and analytics layers

---

## 11. Next Steps (Week 3)
Week 3 will focus on building the **Silver layer**, including:
- Data validation and cleansing
- Deduplication using transaction_id
- Event-time processing
- File compaction
- Writing analytics-ready datasets

---

**Status:** Week 2 Completed Successfully