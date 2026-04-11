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


