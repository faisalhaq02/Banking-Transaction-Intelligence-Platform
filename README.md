# Real Time Banking Transaction Intelligence Platform Usong Streaming and Lakehouse Architecture

---

## Overview

This project implements a Real-Time Banking Transaction Intelligence Platform that processes financial transaction data using both batch and streaming pipelines.

The platform enables:
- Real-time transaction monitoring  
- Fraud and anomaly detection  
- Risk scoring using machine learning  
- Business intelligence dashboards  
- AI-powered chatbot for querying insights  

---

## Tech Stack

- Python  
- Apache Kafka  
- Apache Airflow  
- Azure Blob Storage  
- Azure SQL Database  
- Scikit-learn  
- Tableau  
- Docker  

---

##  Project Links

###  Code Repository
https://github.com/YOUR_USERNAME/banking-transaction-intelligence-platform

### Full Project Download
https://drive.google.com/file/d/1g8MpLdHdomAwMdZoPDPenqAinD2yN1-M/view?usp=share_link

### Dataset Download
https://drive.google.com/file/d/1Xa-Lte88z_kCO4P5ntrdEVGb3kNzulxs/view?usp=sharing

---

## Project Structure
producer/ → Transaction generator
consumer/ → Streaming & batch processing
airflow_docker/ → Airflow setup
dags/ → Workflow pipelines
cloud/ → Azure integration
ml/ → Machine learning models
bi_exports/ → BI outputs
tableau/ → Dashboards & screenshots
dataset/ → Dataset reference
docs/ → Documentation
README.md → Project instructions

---

## ⚙️ Environment Configuration

Create a `.env` file in the root directory:

AIRFLOW_UID=50000
AZURE_STORAGE_ACCOUNT_NAME=your_storage_account
BI_CONTAINER_NAME=bi-exports

AZURE_STORAGE_CONNECTION_STRING=___________

AZURE_SQL_SERVER=your-server.database.windows.net
AZURE_SQL_DATABASE=banking_intelligence_db
AZURE_SQL_USERNAME=username
AZURE_SQL_PASSWORD=password
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
AZURE_SQL_ODBC_DRIVER=ODBC Driver 18 for SQL Serve

Notes:
- `.env` file is not included in the repository    
- Refer to report for details  

---

## Setup Instructions

### 1. Clone Repository
git clone https://github.com/faisalhaq02/Real-Time-Banking-Transaction-Intelligence-Platform-Using-Streaming-and-Lakehouse-Architecture.git
cd Real-Time-Banking-Transaction-Intelligence-Platform-Using-Streaming-and-Lakehouse-Architecture

### 2. Create Virtual Environment
python3 -m venv airflow_venv
source airflow_venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

---

## Run the Project

### Start Services
docker compose up -d

### Run Producer
python producer/transaction_producer_v2.py

### Run Consumer
python consumer/consumer_file.py

### Run Airflow
cd airflow_docker
docker-compose up

Open Airflow UI:
http://localhost:8080

Trigger DAGs manually

---

### Run Machine Learning
python ml/train_models.py

---

### Run Chatbot
python agentic_ai/app.py

---

## 🤖 Machine Learning Models

- KMeans + PCA → Customer Segmentation  
- Isolation Forest / LOF / One-Class SVM → Anomaly Detection  
- Random Forest → Risk Scoring  

---

## 📊 Tableau Dashboards

Located in `/tableau`

- Executive Dashboard  
- Real-Time Dashboard  
- Geography Dashboard  

---

## 📊 Dataset Usage

Download dataset:
https://drive.google.com/file/d/1Xa-Lte88z_kCO4P5ntrdEVGb3kNzulxs/view?usp=sharing

Extract:

unzip banking-dataset.zip

Move:
mv dataset_folder data/

---

## 📈 Outputs

- Transaction data  
- Azure Blob Storage files  
- Azure SQL tables  
- ML predictions  
- BI summaries  
- Dashboards  

---

## ⚠️ Notes

- `.env` is excluded  
- Use your own credentials  
- Ensure Docker and Kafka are running  

---

## Authors

Faisal Ul Haque Mohammed  
Darshil K Shah  

Master of Applied Computing  
Wilfrid Laurier University  

---

## Conclusion

This project demonstrates a **production-level data engineering system** integrating streaming, batch processing, machine learning, and cloud-based analytics for banking intelligence.
