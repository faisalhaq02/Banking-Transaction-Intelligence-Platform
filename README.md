# Banking Transaction Intelligence Platform

## Overview

This project implements a **Banking Transaction Intelligence Platform** that processes financial transaction data using both **batch and real-time streaming pipelines**.

The system integrates:

* **Kafka** for streaming ingestion
* **Airflow** for orchestration
* **Azure Blob Storage** for scalable storage
* **Azure SQL Database** for analytics-ready data
* **Machine Learning models** for anomaly detection and risk scoring
* **Tableau dashboards** for visualization
* **Agentic AI chatbot** for querying insights

The platform enables **real-time monitoring, risk detection, and business intelligence for banking systems**.

---

##  Tech Stack

* Python
* Apache Kafka
* Apache Airflow
* Azure Blob Storage
* Azure SQL Database
* Scikit-learn (ML models)
* Tableau
* Docker

---

## 🔗 Project Links

### 📂 Code Repository

https://github.com/YOUR_USERNAME/banking-transaction-intelligence-platform

### 📦 Full Project Download (Backup)

https://drive.google.com/YOUR_PROJECT_LINK

### 📊 Dataset Download

https://drive.google.com/YOUR_DATASET_LINK

---

## 📁 Project Structure

producer/ → transaction generator
consumer/ → streaming/batch processing
airflow_docker/ → Airflow setup
dags/ → workflow pipelines
cloud/ → Azure integration
ml/ → ML models
bi_exports/ → BI outputs
tableau/ → dashboards & screenshots
dataset/ → dataset reference
docs/ → extra docs
README.md → instructions

---

## ⚙️ Setup Instructions

### 1. Clone Repo

git clone https://github.com/YOUR_USERNAME/banking-transaction-intelligence-platform.git
cd banking-transaction-intelligence-platform

### 2. Install Dependencies

pip install -r requirements.txt

### 3. Create `.env`

Add:
All the Values are available in the report
AZURE_STORAGE_CONNECTION_STRING=
AZURE_SQL_SERVER=
AZURE_SQL_DATABASE=banking_intelligence_db
AZURE_SQL_USERNAME=username
AZURE_SQL_PASSWORD=password
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

---

##  Run Project

Start services:
docker compose up -d

Run producer:
python producer/transaction_producer_v2.py

Run consumer:
python consumer/consumer_file.py

Trigger Airflow DAG from UI

---

## 🤖 Machine Learning

* KMeans + PCA → segmentation
* Isolation Forest / LOF / SVM → anomaly detection
* Random Forest → risk scoring

---

## 📊 Tableau Dashboards

Located in `/tableau`

Includes:

* Executive Dashboard
* Real Time Dashboard
* Geography Dashboard


---

## 📊 Dataset Usage

Download dataset:
https://drive.google.com/YOUR_DATASET_LINK

Unzip:
unzip banking-dataset.zip

Place in:
data/

---

## 📈 Outputs

* transaction data
* Azure storage files
* SQL tables
* ML outputs
* BI summaries
* dashboards

---

## ⚠️ Notes

* `.env` is excluded from GitHub
* Use your own credentials
* Use Drive links for large files

---

## Author

Faisal Ul Haque Mohammed
Darshil K shah
Willfrid Laurier University
Master of Applied Computing
