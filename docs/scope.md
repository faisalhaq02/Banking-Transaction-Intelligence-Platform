## 🎓 Why this README is GOOD
- Academic tone ✔
- Industry clarity ✔
- Not over-technical ✔
- Easy to explain in viva/interview ✔

---

# 🔹 PART 2 — Scope Freeze (Day 1 remaining)

## 👉 What is `scope.md`?
Scope freeze:
- protects you from **overworking**
- shows your professor **planning & discipline**
- helps you justify **why some features are not included**

---

## ✅ Create `docs/scope.md`

```md
# Project Scope Definition

This document defines the scope of the Graduate Capstone Project to ensure realistic execution and prevent scope creep.

---

## Must-Have Features (Core Scope)

The following components are mandatory and will be completed as part of the project:

1. Real-time data ingestion using Apache Kafka
2. Spark Structured Streaming job to process Kafka data
3. Lakehouse architecture with Bronze, Silver, and Gold layers using Parquet and Hive
4. Workflow orchestration using Apache Airflow
5. Basic machine learning model for transaction fraud/risk scoring
6. Business intelligence dashboards using Power BI or Tableau
7. Documentation, architecture diagrams, and reproducible codebase

---

## Optional Features (Time-Permitting)

The following features may be implemented only if sufficient time remains:

1. Kafka Schema Registry (Avro/Protobuf)
2. Monitoring and alerting using Grafana / Prometheus
3. Advanced lakehouse formats such as Delta Lake, Apache Iceberg, or Apache Hudi
4. Advanced model tuning and feature engineering

## Out of Scope

The project does not aim to:
- Build a production banking application
- Use real customer or financial data
- Implement enterprise security infrastructure

---

## Scope Rationale

The defined scope focuses on demonstrating data engineering architecture, scalability, and reliability, which are the primary learning objectives of this project.